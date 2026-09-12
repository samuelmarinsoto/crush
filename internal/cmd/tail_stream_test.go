package cmd

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/charmbracelet/crush/internal/proto"
	"github.com/charmbracelet/crush/internal/pubsub"
	"github.com/stretchr/testify/require"
)

func newTailStream(opts tailOpts) (*tailStream, *bytes.Buffer) {
	buf := &bytes.Buffer{}
	return &tailStream{
		sessionID: "S",
		out:       buf,
		opts:      opts,
		read:      make(map[string]int),
		thinkRead: make(map[string]int),
		seenCalls: make(map[string]bool),
	}, buf
}

func assistantMsg(id string, parts ...proto.ContentPart) pubsub.Event[proto.Message] {
	return pubsub.Event[proto.Message]{Payload: proto.Message{
		ID: id, SessionID: "S", Role: proto.Assistant, Parts: parts,
	}}
}

// TestTailStream_TurnSeparator verifies plain text mode: streamed text
// deltas print incrementally and each finished turn ends with a newline
// so the next turn starts on a fresh line.
func TestTailStream_TurnSeparator(t *testing.T) {
	t.Parallel()

	s, buf := newTailStream(tailOpts{})

	require.NoError(t, s.handle(assistantMsg("m1", proto.TextContent{Text: "hello wor"})))
	require.NoError(t, s.handle(assistantMsg("m1", proto.TextContent{Text: "hello world"})))
	require.NoError(t, s.handle(pubsub.Event[proto.RunComplete]{Payload: proto.RunComplete{SessionID: "S"}}))
	require.NoError(t, s.handle(assistantMsg("m2", proto.TextContent{Text: "second turn"})))
	require.NoError(t, s.handle(pubsub.Event[proto.RunComplete]{Payload: proto.RunComplete{SessionID: "S"}}))

	require.Equal(t, "hello world\nsecond turn\n", buf.String())
}

// TestTailStream_ToolLines verifies --tools: a tool call part emits a
// one-line summary (deduplicated across message growth) and a tool-role
// result message emits a status line. Without the flag neither appears.
func TestTailStream_ToolLines(t *testing.T) {
	t.Parallel()

	call := proto.ToolCall{
		ID:    "call-1",
		Name:  "bash",
		Input: `{"command":"go test ./internal/server/...","description":"run tests"}`,
	}
	msg := assistantMsg("m1", call)
	msg.Payload.Parts = append([]proto.ContentPart{proto.TextContent{Text: ""}}, msg.Payload.Parts...)

	s, buf := newTailStream(tailOpts{tools: true})
	require.NoError(t, s.handle(msg))
	// The same message grows (deltas stream): the call must print once.
	msg.Payload.Parts = append(msg.Payload.Parts, proto.TextContent{Text: "running"})
	require.NoError(t, s.handle(msg))

	result := pubsub.Event[proto.Message]{Payload: proto.Message{
		ID: "t1", SessionID: "S", Role: proto.Tool,
		Parts: []proto.ContentPart{proto.ToolResult{ToolCallID: "call-1", Name: "bash", Content: "ok"}},
	}}
	require.NoError(t, s.handle(result))

	out := buf.String()
	require.Contains(t, out, "* bash — go test ./internal/server/...\n")
	require.Contains(t, out, "* bash (ok)\n")
	require.Equal(t, 1, strings.Count(out, "* bash —"), "tool call summary must be deduplicated")

	// Default opts: no tool output at all.
	sPlain, plainBuf := newTailStream(tailOpts{})
	require.NoError(t, sPlain.handle(msg))
	require.NoError(t, sPlain.handle(result))
	require.Empty(t, plainBuf.String())
}

// TestTailStream_ToolResultError verifies the error status rendering.
func TestTailStream_ToolResultError(t *testing.T) {
	t.Parallel()

	s, buf := newTailStream(tailOpts{tools: true})
	require.NoError(t, s.handle(pubsub.Event[proto.Message]{Payload: proto.Message{
		ID: "t1", SessionID: "S", Role: proto.Tool,
		Parts: []proto.ContentPart{proto.ToolResult{ToolCallID: "c1", Name: "edit", IsError: true}},
	}}))
	require.Equal(t, "* edit (error)\n", buf.String())
}

// TestTailStream_Thinking verifies --thinking: reasoning deltas print
// as "# "-prefixed lines, partial trailing lines are buffered until a
// newline arrives (or the run completes), and thinking never merges
// into an open text line.
func TestTailStream_Thinking(t *testing.T) {
	t.Parallel()

	s, buf := newTailStream(tailOpts{thinking: true})

	s.handle(assistantMsg("m1", proto.ReasoningContent{Thinking: "first line of thou"}))
	require.Empty(t, buf.String(), "partial line must be buffered")

	s.handle(assistantMsg("m1",
		proto.ReasoningContent{Thinking: "first line of thought\nsecond line\n"},
		proto.TextContent{Text: "the answer"},
	))
	require.NoError(t, s.handle(pubsub.Event[proto.RunComplete]{Payload: proto.RunComplete{SessionID: "S"}}))

	require.Equal(t, "# first line of thought\n# second line\nthe answer\n", buf.String())
}

// TestTailStream_JSON verifies --json: every stream item becomes one
// NDJSON object with a discriminating type, raw tool input is preserved,
// and tool/thinking objects respect the same flags.
func TestTailStream_JSON(t *testing.T) {
	t.Parallel()

	s, buf := newTailStream(tailOpts{json: true, tools: true, thinking: true})

	call := proto.ToolCall{ID: "c1", Name: "edit", Input: `{"file_path":"a.go"}`}
	require.NoError(t, s.handle(assistantMsg("m1",
		proto.ReasoningContent{Thinking: "thinking aloud"},
		proto.TextContent{Text: "editing"},
		call,
	)))
	require.NoError(t, s.handle(pubsub.Event[proto.Message]{Payload: proto.Message{
		ID: "t1", SessionID: "S", Role: proto.Tool,
		Parts: []proto.ContentPart{proto.ToolResult{ToolCallID: "c1", Name: "edit", Content: "done"}},
	}}))
	require.NoError(t, s.handle(pubsub.Event[proto.RunComplete]{Payload: proto.RunComplete{SessionID: "S"}}))

	var objs []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(buf.String()), "\n") {
		var obj map[string]any
		require.NoError(t, json.Unmarshal([]byte(line), &obj), "each line must be valid JSON: %s", line)
		objs = append(objs, obj)
	}
	require.Len(t, objs, 5)
	require.Equal(t, "thinking", objs[0]["type"])
	require.Equal(t, "thinking aloud", objs[0]["delta"])
	require.Equal(t, "text", objs[1]["type"])
	require.Equal(t, "editing", objs[1]["delta"])
	require.Equal(t, "tool_call", objs[2]["type"])
	require.Equal(t, "edit", objs[2]["tool"])
	require.Equal(t, `{"file_path":"a.go"}`, objs[2]["input"])
	require.Equal(t, "a.go", objs[2]["summary"])
	require.Equal(t, "tool_result", objs[3]["type"])
	require.Equal(t, "done", objs[3]["content"])
	require.Equal(t, "run_complete", objs[4]["type"])
}

// TestTailStream_JSONFilters verifies that in JSON mode --tools and
// --thinking still gate their object types.
func TestTailStream_JSONFilters(t *testing.T) {
	t.Parallel()

	s, buf := newTailStream(tailOpts{json: true})

	call := proto.ToolCall{ID: "c1", Name: "bash", Input: `{"command":"ls"}`}
	require.NoError(t, s.handle(assistantMsg("m1",
		proto.ReasoningContent{Thinking: "hm"},
		proto.TextContent{Text: "hi"},
		call,
	)))
	require.NoError(t, s.handle(pubsub.Event[proto.RunComplete]{Payload: proto.RunComplete{SessionID: "S"}}))

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	require.Len(t, lines, 2)
	require.Contains(t, lines[0], `"type":"text"`)
	require.Contains(t, lines[1], `"type":"run_complete"`)
}

// TestTailStream_SessionFilter verifies events from other sessions are
// dropped entirely.
func TestTailStream_SessionFilter(t *testing.T) {
	t.Parallel()

	s, buf := newTailStream(tailOpts{tools: true})
	other := pubsub.Event[proto.Message]{Payload: proto.Message{
		ID: "x", SessionID: "other", Role: proto.Assistant,
		Parts: []proto.ContentPart{proto.TextContent{Text: "not mine"}},
	}}
	require.NoError(t, s.handle(other))
	require.NoError(t, s.handle(pubsub.Event[proto.RunComplete]{Payload: proto.RunComplete{SessionID: "other"}}))
	require.Empty(t, buf.String())
}
