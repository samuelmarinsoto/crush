package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/signal"
	"strings"

	"github.com/charmbracelet/crush/internal/event"
	"github.com/charmbracelet/crush/internal/proto"
	"github.com/charmbracelet/crush/internal/pubsub"
	"github.com/spf13/cobra"
	"github.com/tidwall/gjson"
)

var tailCmd = &cobra.Command{
	Use:     "tail <session-id>",
	Aliases: []string{"t"},
	Short:   "Stream a session's live output",
	Long: `Stream a session's events to stdout, like tail -f for a crush session.

Connects to the crush server and follows the given session: assistant
text is printed as it streams, and each finished turn is separated by a
newline. Unlike crush run, no prompt is sent and nothing terminates the
stream — it keeps following the session across turns until interrupted.

The session's workspace stays alive while tail is connected. Detached
runs keep running on the server independently of any client, so tail is
also the way to watch a turn that outlived the client that started it.

Options control what is streamed beyond the assistant's text:

  --tools     one-line summaries of tool calls and their results
              ("* bash — go test ./...", "* bash (ok)")
  --thinking  the model's reasoning deltas, each line prefixed with "# "
  --json      machine-readable NDJSON on stdout instead of text; one
              object per event with a discriminating "type" field
              (text, thinking, tool_call, tool_result, run_complete,
              eof). Filters still apply: --tools and --thinking gate
              the tool_call/tool_result and thinking objects.

Requires client/server mode (CRUSH_CLIENT_SERVER=1).`,
	Example: `
# Follow a session by ID (or unique ID prefix)
crush tail 1c0f3eab

# Find sessions to tail
crush session list

# Watch everything the agent does, tools included
crush tail --tools --thinking 1c0f3eab

# Machine-readable feed (e.g. for gateways and bridges)
crush tail --json --tools --thinking 1c0f3eab
  `,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		if !useClientServer() {
			return fmt.Errorf("crush tail requires client/server mode: set CRUSH_CLIENT_SERVER=1")
		}

		// Cancel on SIGINT or SIGTERM: stopping the tail stops the
		// watch, never the run.
		ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, os.Kill)
		defer cancel()

		event.SetNonInteractive(true)

		c, ws, cleanup, err := connectToServer(cmd)
		if err != nil {
			return err
		}
		defer cleanup()

		sess, err := resolveSessionByID(ctx, c, ws.ID, args[0])
		if err != nil {
			return err
		}

		events, err := c.SubscribeEvents(ctx, ws.ID)
		if err != nil {
			return fmt.Errorf("failed to subscribe to events: %w", err)
		}

		fmt.Fprintf(os.Stderr, "Tailing session %s (Ctrl-C to stop)\n", sess.ID)

		stream := &tailStream{
			sessionID: sess.ID,
			out:       os.Stdout,
			read:      make(map[string]int),
			thinkRead: make(map[string]int),
			seenCalls: make(map[string]bool),
		}
		if stream.opts, err = readTailOpts(cmd); err != nil {
			return err
		}

		for {
			select {
			case ev, ok := <-events:
				if !ok {
					// Distinguish a deliberate stop from the
					// stream ending under us (server shutting
					// down, workspace torn down elsewhere).
					if ctx.Err() != nil {
						return nil
					}
					if stream.opts.json {
						stream.emit(tailJSON{Type: "eof"})
					}
					fmt.Fprintln(os.Stderr, "Event stream closed")
					return nil
				}
				if err := stream.handle(ev); err != nil {
					return err
				}
			case <-ctx.Done():
				return nil
			}
		}
	},
}

func init() {
	tailCmd.Flags().Bool("tools", false, "Also print tool calls and results as one-line summaries")
	tailCmd.Flags().Bool("thinking", false, "Also print reasoning/thinking deltas, lines prefixed with \"# \"")
	tailCmd.Flags().Bool("json", false, "Emit machine-readable NDJSON events instead of text")
}

type tailOpts struct {
	tools    bool
	thinking bool
	json     bool
}

func readTailOpts(cmd *cobra.Command) (tailOpts, error) {
	var opts tailOpts
	var err error
	if opts.tools, err = cmd.Flags().GetBool("tools"); err != nil {
		return opts, err
	}
	if opts.thinking, err = cmd.Flags().GetBool("thinking"); err != nil {
		return opts, err
	}
	if opts.json, err = cmd.Flags().GetBool("json"); err != nil {
		return opts, err
	}
	return opts, nil
}

// tailJSON is the NDJSON envelope used with --json. The Type field
// discriminates the payload; omitted fields keep the objects small.
type tailJSON struct {
	Type      string `json:"type"`
	MessageID string `json:"message_id,omitempty"`
	Delta     string `json:"delta,omitempty"`
	Tool      string `json:"tool,omitempty"`
	CallID    string `json:"call_id,omitempty"`
	Summary   string `json:"summary,omitempty"`
	Input     string `json:"input,omitempty"`
	Content   string `json:"content,omitempty"`
	IsError   bool   `json:"is_error,omitempty"`
	Error     string `json:"error,omitempty"`
	Cancelled bool   `json:"cancelled,omitempty"`
}

// tailStream prints a session's live event stream. It is the follow-mode
// counterpart of runStream: same assistant-delta printing, but no RunID
// correlation and no exit on RunComplete — every turn of the session
// streams through until the caller stops.
type tailStream struct {
	sessionID string
	out       io.Writer
	opts      tailOpts
	// read tracks how many bytes of each assistant message's text have
	// already been emitted; thinkRead does the same for reasoning.
	read      map[string]int
	thinkRead map[string]int
	// seenCalls deduplicates tool-call parts, which stream repeatedly
	// as the same message grows.
	seenCalls map[string]bool
	// printed suppresses output until the first non-blank content of a
	// turn; midLine tracks whether a line is open (no trailing newline
	// yet) so tool/thinking lines can start on fresh lines.
	printed  bool
	midLine  bool
	thinkBuf strings.Builder
}

func (s *tailStream) handle(ev any) error {
	switch e := ev.(type) {
	case pubsub.Event[proto.Message]:
		msg := e.Payload
		if msg.SessionID != s.sessionID || len(msg.Parts) == 0 {
			return nil
		}
		switch msg.Role {
		case proto.Assistant:
			// Reasoning precedes the answer it produced, so emit
			// thinking deltas before text to preserve turn order.
			if s.opts.thinking {
				s.printThinking(msg.ID, thinkingText(msg))
			}
			s.printDelta(msg.ID, msg.Content().String())
			if s.opts.tools {
				for _, part := range msg.Parts {
					if call, ok := part.(proto.ToolCall); ok && !s.seenCalls[call.ID] {
						s.seenCalls[call.ID] = true
						s.emitToolCall(call)
					}
				}
			}
		case proto.Tool:
			if !s.opts.tools {
				return nil
			}
			for _, part := range msg.Parts {
				if res, ok := part.(proto.ToolResult); ok {
					s.emitToolResult(res)
				}
			}
		}
		return nil

	case pubsub.Event[proto.RunComplete]:
		if e.Payload.SessionID != s.sessionID {
			return nil
		}
		// Reconcile stdout against the authoritative final assistant
		// text carried in the event; the pubsub fan-in does not
		// serialize publishes across upstream brokers, so the final
		// message event may not have reached this loop yet.
		if e.Payload.MessageID != "" {
			s.printDelta(e.Payload.MessageID, e.Payload.Text)
		}
		// Flush any buffered thinking line before the turn separator.
		if s.opts.thinking {
			s.flushThinking()
		}
		if e.Payload.Error != "" && !e.Payload.Cancelled {
			if s.opts.json {
				s.emit(tailJSON{Type: "run_complete", Error: e.Payload.Error})
			} else {
				fmt.Fprintf(os.Stderr, "run error: %s\n", e.Payload.Error)
			}
		} else if s.opts.json {
			s.emit(tailJSON{Type: "run_complete", Cancelled: e.Payload.Cancelled})
		}
		if s.midLine {
			fmt.Fprintln(s.out)
			s.midLine = false
		}
		s.printed = false
		return nil

	case pubsub.Event[proto.AgentEvent]:
		if e.Payload.Error != nil && e.Payload.SessionID == s.sessionID {
			slog.Warn("agent event", "error", e.Payload.Error)
		}
		return nil
	}
	return nil
}

// printDelta prints the not-yet-read suffix of a streaming assistant
// message, tracking how much of each message ID has already been emitted.
func (s *tailStream) printDelta(messageID, content string) {
	readBytes := s.read[messageID]
	if len(content) < readBytes {
		slog.Error("Tail: message content shorter than read bytes",
			"message_length", len(content), "read_bytes", readBytes)
		return
	}
	part := content[readBytes:]
	if readBytes == 0 {
		part = strings.TrimLeft(part, " \t")
	}
	if s.printed || strings.TrimSpace(part) != "" {
		s.printed = true
		s.writeText(part)
	}
	s.read[messageID] = len(content)
}

// writeText emits an assistant text chunk, in JSON mode as a delta
// object.
func (s *tailStream) writeText(part string) {
	if s.opts.json {
		s.emit(tailJSON{Type: "text", Delta: part})
		return
	}
	fmt.Fprint(s.out, part)
	s.midLine = true
}

// printThinking tracks the reasoning deltas of a message. In JSON mode
// deltas pass through raw; in text mode complete lines are buffered and
// printed prefixed with "# " so mid-line stream splits stay readable.
func (s *tailStream) printThinking(messageID, thinking string) {
	readBytes := s.thinkRead[messageID]
	if len(thinking) < readBytes {
		return
	}
	delta := thinking[readBytes:]
	s.thinkRead[messageID] = len(thinking)
	if delta == "" {
		return
	}
	if s.opts.json {
		s.emit(tailJSON{Type: "thinking", MessageID: messageID, Delta: delta})
		return
	}
	s.thinkBuf.WriteString(delta)
	for {
		line, rest, found := strings.Cut(s.thinkBuf.String(), "\n")
		if !found {
			s.thinkBuf.Reset()
			s.thinkBuf.WriteString(line)
			return
		}
		s.thinkBuf.Reset()
		s.thinkBuf.WriteString(rest)
		s.writeThinkingLine(line)
	}
}

func (s *tailStream) flushThinking() {
	if s.opts.json || s.thinkBuf.Len() == 0 {
		return
	}
	line := s.thinkBuf.String()
	s.thinkBuf.Reset()
	s.writeThinkingLine(strings.TrimRight(line, "\n"))
}

func (s *tailStream) writeThinkingLine(line string) {
	if strings.TrimSpace(line) == "" {
		return
	}
	s.newlineIfNeeded()
	fmt.Fprintf(s.out, "# %s\n", line)
	s.midLine = false
}

func (s *tailStream) emitToolCall(call proto.ToolCall) {
	if s.opts.json {
		s.emit(tailJSON{
			Type:    "tool_call",
			Tool:    call.Name,
			CallID:  call.ID,
			Summary: toolCallSummary(call),
			Input:   call.Input,
		})
		return
	}
	s.newlineIfNeeded()
	fmt.Fprintf(s.out, "* %s — %s\n", call.Name, toolCallSummary(call))
	s.midLine = false
}

func (s *tailStream) emitToolResult(res proto.ToolResult) {
	if s.opts.json {
		s.emit(tailJSON{
			Type:    "tool_result",
			Tool:    res.Name,
			CallID:  res.ToolCallID,
			Content: res.Content,
			IsError: res.IsError,
		})
		return
	}
	s.newlineIfNeeded()
	status := "ok"
	if res.IsError {
		status = "error"
	}
	fmt.Fprintf(s.out, "* %s (%s)\n", res.Name, status)
	s.midLine = false
}

func (s *tailStream) newlineIfNeeded() {
	if s.midLine {
		fmt.Fprintln(s.out)
		s.midLine = false
	}
}

func (s *tailStream) emit(obj tailJSON) {
	// Write errors (e.g. a closed pipe) surface through the next
	// stream read; nothing useful to do here.
	if err := json.NewEncoder(s.out).Encode(obj); err != nil {
		slog.Debug("Tail: failed to encode JSON event", "error", err)
	}
}

// thinkingText concatenates the reasoning parts of a message, mirroring
// Message.Content() for text.
func thinkingText(msg proto.Message) string {
	var b strings.Builder
	for _, part := range msg.Parts {
		if rc, ok := part.(proto.ReasoningContent); ok {
			b.WriteString(rc.Thinking)
		}
	}
	return b.String()
}

// toolArgKeys maps a tool name to the parameter that best identifies
// the call for a one-line summary.
var toolArgKeys = map[string]string{
	"bash":      "command",
	"download":  "url",
	"fetch":     "url",
	"edit":      "file_path",
	"multiedit": "file_path",
	"write":     "file_path",
	"view":      "file_path",
	"grep":      "pattern",
	"glob":      "pattern",
	"ls":        "path",
}

// toolCallSummary renders a tool call's identifying argument, truncated,
// for human-readable output.
func toolCallSummary(call proto.ToolCall) string {
	if key, ok := toolArgKeys[call.Name]; ok {
		if v := gjson.Get(call.Input, key); v.Exists() && v.String() != "" {
			return truncateRunes(v.String(), 100)
		}
	}
	if s := strings.TrimSpace(call.Input); s != "" {
		return truncateRunes(s, 100)
	}
	return "(no input)"
}

func truncateRunes(s string, max int) string {
	runes := []rune(s)
	if len(runes) <= max {
		return s
	}
	return string(runes[:max]) + "…"
}
