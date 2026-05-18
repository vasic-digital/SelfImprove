// CONST-046 round 94 migration coverage: assert every migrated call
// site in buildOptimizationTopicCtx routes through the wired Translator.
//
// Anti-bluff guarantee (CONST-035 / Article XI §11.9):
// Each sub-test installs a sentinel Translator whose T returns the
// distinctive string "<TRANSLATED:<id>>". Tests then drive
// buildOptimizationTopicCtx with the input shape required to hit that
// particular call site and assert the output contains the sentinel.
// If the production code regresses to a hardcoded literal, the test
// FAILs immediately (paired mutation verified in /tmp/round94_mutation.txt).
//
// CONST-050(A): mocks permitted inside _test.go files. The sentinel
// translator is test-only — production paths inject a real
// *i18nadapter.Translator backed by go-i18n.
package selfimprove

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"digital.vasic.selfimprove/pkg/i18n"
)

// Compile-time assertion that sentinelTranslator satisfies i18n.Translator.
var _ i18n.Translator = sentinelTranslator{}

// sentinelTranslator returns "<TRANSLATED:<id>>" for every messageID.
// Distinctive enough that no English literal in the codebase could
// accidentally match it — false positives are mechanically impossible.
type sentinelTranslator struct{}

func (sentinelTranslator) T(_ context.Context, id string, _ map[string]any) (string, error) {
	return "<TRANSLATED:" + id + ">", nil
}

func (sentinelTranslator) TPlural(_ context.Context, id string, _ int, _ map[string]any) (string, error) {
	return "<TRANSLATED:" + id + ">", nil
}

// TestBuildOptimizationTopic_Const046_MigrationRoutesThroughTranslator
// is the table-driven proof that all 8 migrated call sites resolve via
// the wired Translator. Each row asserts on a sentinel substring that
// MUST appear in the rendered prompt.
func TestBuildOptimizationTopic_Const046_MigrationRoutesThroughTranslator(t *testing.T) {
	t.Parallel()

	// Build patterns that trigger every conditional branch in
	// buildOptimizationTopicCtx so all 8 migration sites are exercised.
	patterns := &feedbackPatterns{
		DimensionWeakness: map[DimensionType]float64{
			DimensionAccuracy: 0.3,
		},
		CommonIssues: map[string]int{
			"too vague": 3,
		},
		NegativeExamples: []*TrainingExample{
			{Prompt: "test prompt", RewardScore: 0.2},
		},
	}

	opt := NewLLMPolicyOptimizer(nil, nil, nil, nil)
	opt.SetCurrentPolicy("dummy policy") // force current-policy branch
	opt.SetTranslator(sentinelTranslator{})

	topic := opt.buildOptimizationTopicCtx(context.Background(), patterns)

	tests := []struct {
		name          string
		expectMsgID   string
		expectLiteral string
	}{
		{
			name:          "analyze_header",
			expectMsgID:   "selfimprove_optimizer_prompt_analyze_header",
			expectLiteral: "Analyze these feedback patterns and suggest system prompt improvements",
		},
		{
			name:          "current_policy_label",
			expectMsgID:   "selfimprove_optimizer_prompt_current_policy_label",
			expectLiteral: "Current System Prompt:",
		},
		{
			name:          "weak_dimensions_label",
			expectMsgID:   "selfimprove_optimizer_prompt_weak_dimensions_label",
			expectLiteral: "Weak Dimensions:",
		},
		{
			name:          "dimension_bullet",
			expectMsgID:   "selfimprove_optimizer_prompt_dimension_bullet",
			expectLiteral: "", // bullet rendered as sentinel, not the literal form
		},
		{
			name:          "common_issues_label",
			expectMsgID:   "selfimprove_optimizer_prompt_common_issues_label",
			expectLiteral: "Common Issues:",
		},
		{
			name:          "issue_bullet",
			expectMsgID:   "selfimprove_optimizer_prompt_issue_bullet",
			expectLiteral: "",
		},
		{
			name:          "sample_responses_label",
			expectMsgID:   "selfimprove_optimizer_prompt_sample_responses_label",
			expectLiteral: "Sample Low-Score Responses:",
		},
		{
			name:          "suggest_improvements_footer",
			expectMsgID:   "selfimprove_optimizer_prompt_suggest_improvements_footer",
			expectLiteral: "Suggest specific improvements to the system prompt that would address these issues.",
		},
	}

	sentinel := func(id string) string { return "<TRANSLATED:" + id + ">" }

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			require.Contains(t, topic, sentinel(tc.expectMsgID),
				"migrated call site %q must route through Translator (sentinel %q not found in output)",
				tc.name, sentinel(tc.expectMsgID))
			// Mutation cross-check: the original English literal MUST
			// NOT appear (it was replaced by the sentinel). If both
			// appear the migration is half-done.
			if tc.expectLiteral != "" {
				assert.False(t, strings.Contains(topic, tc.expectLiteral),
					"migrated call site %q still emits the original hardcoded literal %q — migration regressed",
					tc.name, tc.expectLiteral)
			}
		})
	}
}

// TestSetTranslator_NilFallsBackToNoop verifies the safety contract: a
// nil Translator does NOT panic and does NOT silently disable lookup —
// it resets to NoopTranslator (loud message-ID echo).
func TestSetTranslator_NilFallsBackToNoop(t *testing.T) {
	t.Parallel()
	opt := NewLLMPolicyOptimizer(nil, nil, nil, nil)
	opt.SetTranslator(sentinelTranslator{}) // wire real
	opt.SetTranslator(nil)                  // reset to noop

	got := opt.tr(context.Background(), "selfimprove_test_id", nil)
	assert.Equal(t, "selfimprove_test_id", got,
		"SetTranslator(nil) must fall back to NoopTranslator (loud id echo)")
}
