package cmd

import (
	"context"
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
)

var tailCmd = &cobra.Command{Use: "tail <session-id>",
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

Requires client/server mode (CRUSH_CLIENT_SERVER=1).`,
	Example: `
# Follow a session by ID (or unique ID prefix)
crush tail 1c0f3eab

# Find sessions to tail
crush session list
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

// tailStream prints a session's live event stream. It is the follow-mode
// counterpart of runStream: same assistant-delta printing, but no RunID
// correlation and no exit on RunComplete — every turn of the session
// streams through until the caller stops.
type tailStream struct {
	sessionID string
	out       io.Writer
	read      map[string]int
	printed   bool
}

func (s *tailStream) handle(ev any) error {
	switch e := ev.(type) {
	case pubsub.Event[proto.Message]:
		msg := e.Payload
		if msg.SessionID != s.sessionID || msg.Role != proto.Assistant || len(msg.Parts) == 0 {
			return nil
		}
		s.printDelta(msg.ID, msg.Content().String())
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
		if e.Payload.Error != "" && !e.Payload.Cancelled {
			fmt.Fprintf(os.Stderr, "run error: %s\n", e.Payload.Error)
		}
		if s.printed {
			fmt.Fprintln(s.out)
			s.printed = false
		}
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
		fmt.Fprint(s.out, part)
	}
	s.read[messageID] = len(content)
}
