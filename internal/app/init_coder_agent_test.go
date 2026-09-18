package app

import (
	"context"
	"testing"

	"charm.land/catwalk/pkg/catwalk"
	"github.com/charmbracelet/crush/internal/agent"
	"github.com/charmbracelet/crush/internal/config"
	"github.com/charmbracelet/crush/internal/csync"
	"github.com/charmbracelet/crush/internal/db"
	"github.com/charmbracelet/crush/internal/skills"
	"github.com/stretchr/testify/require"
)

// stubCoordinator satisfies agent.Coordinator without behavior: the
// idempotence test only observes which instance initCoderAgent leaves
// behind, never calls into it.
type stubCoordinator struct {
	agent.Coordinator
}

// TestInitCoderAgent_NeverReplacesLiveCoordinator is the loose-agent
// regression test: every client connect runs agent init, and with
// keep-alive a reattaching client finds a workspace whose in-flight run
// is bound to the existing coordinator. Replacing that instance would
// orphan the run — cancels and queued prompts would land on the new
// coordinator while the old one kept executing.
func TestInitCoderAgent_NeverReplacesLiveCoordinator(t *testing.T) {
	t.Parallel()

	app := &App{}
	live := stubCoordinator{}
	app.AgentCoordinator = live

	require.NoError(t, app.initCoderAgent(context.Background(), true))
	require.Equal(t, live, app.AgentCoordinator,
		"init must keep the live coordinator")

	require.NoError(t, app.initCoderAgent(context.Background(), false))
	require.Equal(t, live, app.AgentCoordinator,
		"a second init (reattach with a different interactive flag) must keep the live coordinator")
}

// TestInitCoderAgent_InitializesWhenAbsent verifies the guard does not
// skip genuine first-time initialization: a freshly constructed app
// must come out of init with a working coordinator installed.
func TestInitCoderAgent_InitializesWhenAbsent(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	// Use the provider catalog bundled with this build: no network.
	t.Setenv("CRUSH_DISABLE_PROVIDER_AUTO_UPDATE", "1")

	ctx := context.Background()

	cfgStore, err := config.Init(t.TempDir(), t.TempDir(), false)
	require.NoError(t, err)

	// A bare config has no coder agent or selected models until the
	// user configures a provider; inject the minimum NewCoordinator
	// needs to construct.
	cfg := cfgStore.Config()
	if cfg.Agents == nil {
		cfg.Agents = make(map[string]config.Agent)
	}
	cfg.Agents[config.AgentCoder] = config.Agent{ID: config.AgentCoder}
	cfg.Models = map[config.SelectedModelType]config.SelectedModel{
		config.SelectedModelTypeLarge: {Provider: "anthropic", Model: "claude-sonnet-4-6"},
		config.SelectedModelTypeSmall: {Provider: "anthropic", Model: "claude-haiku-4-5"},
	}
	if cfg.Providers == nil {
		cfg.Providers = csync.NewMap[string, config.ProviderConfig]()
	}
	cfg.Providers.Set("anthropic", config.ProviderConfig{
		ID:     "anthropic",
		Name:   "Anthropic",
		Type:   catwalk.TypeAnthropic,
		APIKey: "test-key",
		Models: []catwalk.Model{
			{ID: "claude-sonnet-4-6", Name: "Claude Sonnet 4.6"},
			{ID: "claude-haiku-4-5", Name: "Claude Haiku 4.5"},
		},
	})

	conn, err := db.Connect(ctx, t.TempDir())
	require.NoError(t, err)
	t.Cleanup(func() { conn.Close() })

	app, err := New(ctx, conn, cfgStore, skills.NewManager(nil, nil, nil))
	require.NoError(t, err)
	t.Cleanup(func() { conn.Close() })

	// app.New installs the coordinator during workspace creation; the
	// server's per-connect init endpoint must not rebuild that live
	// instance.
	installed := app.AgentCoordinator
	require.NotNil(t, installed, "New must install the coordinator")

	require.NoError(t, app.initCoderAgent(ctx, true))
	require.Equal(t, installed, app.AgentCoordinator,
		"re-init must keep the coordinator New installed")

	require.NoError(t, app.initCoderAgent(ctx, false))
	require.Equal(t, installed, app.AgentCoordinator)
}
