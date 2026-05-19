// CONST-046 round 206 migration coverage: assert every newly-migrated
// call site routes through the wired Translator. Round 206 picks up
// where round 94 (optimizer prompt-builder) left off and migrates 10
// more high-impact user-facing strings:
//
//   Cohort A — LLM prompt templates (5 IDs):
//     - selfimprove_integration_evaluate_quality_topic
//     - selfimprove_integration_compare_responses_topic
//     - selfimprove_optimizer_systemprompt_improvement_specialist
//     - selfimprove_reward_systemprompt_quality_evaluator
//     - selfimprove_reward_systemprompt_comparison_evaluator
//
//   Cohort B — default constitutional principles (5 IDs):
//     - selfimprove_config_principle_helpful_harmless_honest
//     - selfimprove_config_principle_avoid_harmful_content
//     - selfimprove_config_principle_respect_privacy
//     - selfimprove_config_principle_acknowledge_uncertainty
//     - selfimprove_config_principle_balanced_perspectives
//
// Anti-bluff guarantee (CONST-035 / Article XI §11.9):
// Each call-site test installs a sentinel Translator returning the
// distinctive string "<TR206:<id>>". The test then drives the
// production code path with the input shape required to hit that
// site and asserts the sentinel appears in the captured output AND
// the original English literal does NOT (paired mutation — if a
// future refactor reintroduces the hardcoded literal alongside the
// translation, the test FAILs immediately).
//
// CONST-050(A): mocks permitted inside _test.go files. The sentinel
// translators are test-only — production paths inject a real
// *i18nadapter.Translator backed by go-i18n.
package selfimprove

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"digital.vasic.selfimprove/pkg/i18n"
)

// sentinelTranslatorR206 returns "<TR206:<id>>" for every messageID.
// Distinctive enough that no English literal in the codebase could
// accidentally match it — false positives are mechanically impossible.
type sentinelTranslatorR206 struct{}

// Compile-time assertion.
var _ i18n.Translator = sentinelTranslatorR206{}

func (sentinelTranslatorR206) T(_ context.Context, id string, _ map[string]any) (string, error) {
	return "<TR206:" + id + ">", nil
}

func (sentinelTranslatorR206) TPlural(_ context.Context, id string, _ int, _ map[string]any) (string, error) {
	return "<TR206:" + id + ">", nil
}

func sentinelR206(id string) string { return "<TR206:" + id + ">" }

// --- Recording LLMProvider/DebateService doubles ------------------------

// recordingProviderR206 captures the (prompt, systemPrompt) pair passed
// to Complete so the test can assert what the LLM actually receives.
// Mocks-in-unit-tests permitted per CONST-050(A).
type recordingProviderR206 struct {
	lastPrompt       string
	lastSystemPrompt string
	respond          string
	respondErr       error
}

func (r *recordingProviderR206) Complete(_ context.Context, prompt, systemPrompt string) (string, error) {
	r.lastPrompt = prompt
	r.lastSystemPrompt = systemPrompt
	if r.respondErr != nil {
		return "", r.respondErr
	}
	return r.respond, nil
}

// recordingDebateServiceR206 captures the topic passed to RunDebate so
// tests can verify the debate topic that the production code emits.
type recordingDebateServiceR206 struct {
	lastTopic string
	result    *DebateResult
	resultErr error
}

func (r *recordingDebateServiceR206) RunDebate(_ context.Context, topic string, _ []string) (*DebateResult, error) {
	r.lastTopic = topic
	if r.resultErr != nil {
		return nil, r.resultErr
	}
	if r.result != nil {
		return r.result, nil
	}
	return &DebateResult{
		ID:           "test-debate",
		Consensus:    "{}",
		Confidence:   0.5,
		Participants: map[string]string{},
		Votes:        map[string]float64{},
	}, nil
}

// ---------------------------------------------------------------------
// Cohort A — LLM prompt templates
// ---------------------------------------------------------------------

// TestRound206_Optimizer_SystemPrompt_RoutesThroughTranslator drives
// optimizeWithLLM and asserts the system prompt forwarded to the LLM
// equals the sentinel — not the original English literal.
func TestRound206_Optimizer_SystemPrompt_RoutesThroughTranslator(t *testing.T) {
	t.Parallel()

	prov := &recordingProviderR206{respond: "[]"}
	opt := NewLLMPolicyOptimizer(prov, nil, nil, nil)
	opt.SetTranslator(sentinelTranslatorR206{})

	patterns := &feedbackPatterns{
		DimensionWeakness: map[DimensionType]float64{DimensionAccuracy: 0.2},
	}

	// Drive the production code path. We expect optimizeWithLLM to call
	// provider.Complete exactly once with the resolved system prompt.
	_, err := opt.optimizeWithLLM(context.Background(), patterns)
	// JSON-parse failure is acceptable — we care about what reached the LLM.
	if err != nil && !errors.Is(err, context.Canceled) {
		t.Logf("optimizeWithLLM returned err=%v (acceptable for sentinel JSON)", err)
	}

	wantID := "selfimprove_optimizer_systemprompt_improvement_specialist"
	require.Equal(t, sentinelR206(wantID), prov.lastSystemPrompt,
		"systemPrompt sent to LLM must equal sentinel (translation routed)")

	// Paired mutation: the original English literal MUST NOT appear in
	// what the LLM received. If both the sentinel AND the literal were
	// present, the migration would be half-done.
	assert.False(t,
		strings.Contains(prov.lastSystemPrompt, "You are an AI system improvement specialist"),
		"systemPrompt MUST NOT contain the hardcoded English literal — migration regressed")
}

// TestRound206_Reward_QualityEvaluator_RoutesThroughTranslator drives
// scoreWithLLM and asserts the system prompt sent to the LLM equals
// the sentinel.
func TestRound206_Reward_QualityEvaluator_RoutesThroughTranslator(t *testing.T) {
	t.Parallel()

	prov := &recordingProviderR206{respond: `{"score": 0.7}`}
	rm := NewAIRewardModel(prov, nil, nil, nil)
	rm.SetTranslator(sentinelTranslatorR206{})

	_, _ = rm.scoreWithLLM(context.Background(), "test prompt", "test response")

	wantID := "selfimprove_reward_systemprompt_quality_evaluator"
	require.Equal(t, sentinelR206(wantID), prov.lastSystemPrompt,
		"quality-evaluator systemPrompt must equal sentinel")
	assert.False(t,
		strings.Contains(prov.lastSystemPrompt, "You are a response quality evaluator"),
		"systemPrompt MUST NOT contain the hardcoded English literal")
}

// TestRound206_Reward_ComparisonEvaluator_RoutesThroughTranslator drives
// compareWithLLM and asserts the comparison system prompt equals the
// sentinel.
func TestRound206_Reward_ComparisonEvaluator_RoutesThroughTranslator(t *testing.T) {
	t.Parallel()

	prov := &recordingProviderR206{respond: `{"preferred": "A", "margin": 0.3}`}
	rm := NewAIRewardModel(prov, nil, nil, nil)
	rm.SetTranslator(sentinelTranslatorR206{})

	_, _ = rm.compareWithLLM(context.Background(), "prompt", "responseA", "responseB")

	wantID := "selfimprove_reward_systemprompt_comparison_evaluator"
	require.Equal(t, sentinelR206(wantID), prov.lastSystemPrompt,
		"comparison-evaluator systemPrompt must equal sentinel")
	assert.False(t,
		strings.Contains(prov.lastSystemPrompt, "You are comparing AI responses"),
		"systemPrompt MUST NOT contain the hardcoded English literal")
}

// TestRound206_Integration_EvaluateTopic_RoutesThroughTranslator drives
// DebateServiceAdapter.EvaluateWithDebate and asserts the topic sent to
// the debate service equals the sentinel (with placeholders bound).
func TestRound206_Integration_EvaluateTopic_RoutesThroughTranslator(t *testing.T) {
	t.Parallel()

	ds := &recordingDebateServiceR206{}
	adapter := NewDebateServiceAdapter(ds, nil)
	adapter.SetTranslator(sentinelTranslatorR206{})

	_, _ = adapter.EvaluateWithDebate(context.Background(), "user prompt", "ai response")

	wantID := "selfimprove_integration_evaluate_quality_topic"
	require.Equal(t, sentinelR206(wantID), ds.lastTopic,
		"debate-evaluate topic must equal sentinel")
	assert.False(t,
		strings.Contains(ds.lastTopic, "Evaluate this AI response quality"),
		"debate topic MUST NOT contain the hardcoded English literal")
}

// TestRound206_Integration_CompareTopic_RoutesThroughTranslator drives
// DebateServiceAdapter.CompareWithDebate and asserts the comparison
// topic equals the sentinel.
func TestRound206_Integration_CompareTopic_RoutesThroughTranslator(t *testing.T) {
	t.Parallel()

	ds := &recordingDebateServiceR206{}
	adapter := NewDebateServiceAdapter(ds, nil)
	adapter.SetTranslator(sentinelTranslatorR206{})

	_, _ = adapter.CompareWithDebate(context.Background(), "prompt", "A", "B")

	wantID := "selfimprove_integration_compare_responses_topic"
	require.Equal(t, sentinelR206(wantID), ds.lastTopic,
		"debate-compare topic must equal sentinel")
	assert.False(t,
		strings.Contains(ds.lastTopic, "Compare these responses. Which is better"),
		"debate topic MUST NOT contain the hardcoded English literal")
}

// ---------------------------------------------------------------------
// Cohort A — fallback behaviour (NoopTranslator → English literal)
// ---------------------------------------------------------------------

// TestRound206_Optimizer_SystemPrompt_NoopFallback verifies the
// English-literal fallback when no Translator is wired (or
// NoopTranslator is active). The fallback is REQUIRED so the LLM
// always receives a usable system prompt even from un-localised
// consumers — without the fallback the LLM would receive the bare
// message ID and produce garbage.
func TestRound206_Optimizer_SystemPrompt_NoopFallback(t *testing.T) {
	t.Parallel()

	prov := &recordingProviderR206{respond: "[]"}
	opt := NewLLMPolicyOptimizer(prov, nil, nil, nil) // NoopTranslator default

	patterns := &feedbackPatterns{}
	_, _ = opt.optimizeWithLLM(context.Background(), patterns)

	require.Contains(t, prov.lastSystemPrompt, "You are an AI system improvement specialist",
		"NoopTranslator must trigger English-literal fallback")
	assert.NotEqual(t, "selfimprove_optimizer_systemprompt_improvement_specialist",
		prov.lastSystemPrompt,
		"LLM must receive English literal, not the raw message ID")
}

// TestRound206_Reward_NoopFallback covers both reward.go system prompts.
func TestRound206_Reward_NoopFallback(t *testing.T) {
	t.Parallel()

	prov := &recordingProviderR206{respond: `{"score": 0.5}`}
	rm := NewAIRewardModel(prov, nil, nil, nil) // NoopTranslator default

	_, _ = rm.scoreWithLLM(context.Background(), "p", "r")
	require.Contains(t, prov.lastSystemPrompt, "You are a response quality evaluator",
		"NoopTranslator must trigger quality-evaluator English fallback")

	prov.respond = `{"preferred": "A", "margin": 0.5}`
	_, _ = rm.compareWithLLM(context.Background(), "p", "r1", "r2")
	require.Contains(t, prov.lastSystemPrompt, "You are comparing AI responses",
		"NoopTranslator must trigger comparison-evaluator English fallback")
}

// ---------------------------------------------------------------------
// Cohort B — default constitutional principles
// ---------------------------------------------------------------------

// TestRound206_DefaultConstitutionalPrincipleIDs_Stability locks the
// IDs returned by DefaultConstitutionalPrincipleIDs() to the exact set
// declared in the i18n bundle. If a future commit renames an ID
// without updating both sides, this test FAILs immediately.
func TestRound206_DefaultConstitutionalPrincipleIDs_Stability(t *testing.T) {
	t.Parallel()

	want := []string{
		"selfimprove_config_principle_helpful_harmless_honest",
		"selfimprove_config_principle_avoid_harmful_content",
		"selfimprove_config_principle_respect_privacy",
		"selfimprove_config_principle_acknowledge_uncertainty",
		"selfimprove_config_principle_balanced_perspectives",
	}
	got := DefaultConstitutionalPrincipleIDs()
	require.Equal(t, want, got,
		"constitutional-principle IDs must stay aligned with the i18n bundle")
}

// TestRound206_DefaultSelfImprovementConfig_BackCompatEnglish locks the
// pre-round-206 English-literal behaviour of DefaultSelfImprovementConfig()
// so consumers that haven't adopted the locale-aware constructor still
// receive readable defaults.
func TestRound206_DefaultSelfImprovementConfig_BackCompatEnglish(t *testing.T) {
	t.Parallel()

	cfg := DefaultSelfImprovementConfig()
	require.Equal(t, []string{
		"Be helpful, harmless, and honest",
		"Avoid generating harmful or misleading content",
		"Respect user privacy and confidentiality",
		"Acknowledge uncertainty when appropriate",
		"Provide balanced perspectives on controversial topics",
	}, cfg.ConstitutionalPrinciples,
		"DefaultSelfImprovementConfig() MUST preserve pre-round-206 English literals for backward compat")
}

// TestRound206_DefaultSelfImprovementConfigIDs_ReturnsMessageIDs locks
// the new locale-aware constructor's behaviour: it returns message IDs
// (not English literals) so consumers translate them via their wired
// Translator at display time. Anti-bluff: an ID that accidentally
// equals an English literal would silently break the contract.
func TestRound206_DefaultSelfImprovementConfigIDs_ReturnsMessageIDs(t *testing.T) {
	t.Parallel()

	cfg := DefaultSelfImprovementConfigIDs()
	require.Equal(t, DefaultConstitutionalPrincipleIDs(), cfg.ConstitutionalPrinciples,
		"DefaultSelfImprovementConfigIDs() MUST return canonical message IDs")
	for _, id := range cfg.ConstitutionalPrinciples {
		require.True(t, strings.HasPrefix(id, "selfimprove_config_principle_"),
			"each principle entry must be a canonical message ID, got %q", id)
		require.False(t, strings.Contains(id, " "),
			"message IDs MUST NOT contain spaces (would indicate English literal leak): %q", id)
	}
}

// TestRound206_DefaultConfig_ParityCount enforces the 1:1 alignment
// between English-fallback list and ID list — drift between the two
// is the round-206 paired-mutation gate.
func TestRound206_DefaultConfig_ParityCount(t *testing.T) {
	t.Parallel()

	english := defaultConstitutionalPrinciplesEnglish()
	ids := DefaultConstitutionalPrincipleIDs()
	require.Equal(t, len(english), len(ids),
		"English-fallback list and ID list MUST stay 1:1 aligned (round 206 paired-mutation gate)")
	require.Equal(t, 5, len(ids),
		"round 206 cohort B migrates exactly 5 constitutional principles")
}

// TestRound206_SetTranslator_NilFallsBackToNoop_Reward verifies the
// safety contract on AIRewardModel.SetTranslator (mirrors the
// optimizer's equivalent test from round 94).
func TestRound206_SetTranslator_NilFallsBackToNoop_Reward(t *testing.T) {
	t.Parallel()

	rm := NewAIRewardModel(nil, nil, nil, nil)
	rm.SetTranslator(sentinelTranslatorR206{})
	rm.SetTranslator(nil)

	got := rm.tr(context.Background(), "selfimprove_test_id", nil)
	assert.Equal(t, "selfimprove_test_id", got,
		"SetTranslator(nil) must fall back to NoopTranslator on AIRewardModel")
}

// TestRound206_SetTranslator_NilFallsBackToNoop_Adapter verifies the
// same safety contract on DebateServiceAdapter.
func TestRound206_SetTranslator_NilFallsBackToNoop_Adapter(t *testing.T) {
	t.Parallel()

	adapter := NewDebateServiceAdapter(nil, nil)
	adapter.SetTranslator(sentinelTranslatorR206{})
	adapter.SetTranslator(nil)

	got := adapter.tr(context.Background(), "selfimprove_test_id", nil)
	assert.Equal(t, "selfimprove_test_id", got,
		"SetTranslator(nil) must fall back to NoopTranslator on DebateServiceAdapter")
}
