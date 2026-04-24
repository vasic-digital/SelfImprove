package security

import (
	"context"
	"runtime"
	"testing"

	"digital.vasic.selfimprove/selfimprove"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func init() {
	runtime.GOMAXPROCS(2)
}

// ---------------------------------------------------------------------------
// Feedback Collector security tests
// ---------------------------------------------------------------------------

func TestCollectNilFeedback_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	collector := selfimprove.NewInMemoryFeedbackCollector(nil, 1000)
	err := collector.Collect(context.Background(), nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "nil")
}

func TestCollectEmptyFeedback_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	collector := selfimprove.NewInMemoryFeedbackCollector(nil, 1000)
	fb := &selfimprove.Feedback{}
	// Empty feedback should still be collected (ID auto-generated)
	err := collector.Collect(context.Background(), fb)
	require.NoError(t, err)
	assert.NotEmpty(t, fb.ID)
}

func TestGetBySessionNonexistent_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	collector := selfimprove.NewInMemoryFeedbackCollector(nil, 1000)
	result, err := collector.GetBySession(context.Background(), "nonexistent")
	require.NoError(t, err)
	assert.Empty(t, result)
}

func TestGetByPromptNonexistent_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	collector := selfimprove.NewInMemoryFeedbackCollector(nil, 1000)
	result, err := collector.GetByPrompt(context.Background(), "nonexistent")
	require.NoError(t, err)
	assert.Empty(t, result)
}

func TestGetAggregatedEmpty_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	collector := selfimprove.NewInMemoryFeedbackCollector(nil, 1000)
	agg, err := collector.GetAggregated(context.Background(), nil)
	require.NoError(t, err)
	assert.Equal(t, 0, agg.TotalCount)
}

func TestGetAggregatedWithNilFilter_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	collector := selfimprove.NewInMemoryFeedbackCollector(nil, 1000)
	err := collector.Collect(context.Background(), &selfimprove.Feedback{
		Score: 0.5,
	})
	require.NoError(t, err)

	agg, err := collector.GetAggregated(context.Background(), nil)
	require.NoError(t, err)
	assert.Equal(t, 1, agg.TotalCount)
}

func TestExportEmptyCollector_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	collector := selfimprove.NewInMemoryFeedbackCollector(nil, 1000)
	examples, err := collector.Export(context.Background(), nil)
	require.NoError(t, err)
	assert.Empty(t, examples)
}

// ---------------------------------------------------------------------------
// Reward Model security tests
// ---------------------------------------------------------------------------

func TestScoreWithNilProvider_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	rm := selfimprove.NewAIRewardModel(nil, nil, nil, nil)
	score, err := rm.Score(context.Background(), "prompt", "response")
	assert.Equal(t, 0.5, score)
	assert.Error(t, err)
}

func TestScoreWithEmptyInputs_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	rm := selfimprove.NewAIRewardModel(nil, nil, nil, nil)
	score, err := rm.Score(context.Background(), "", "")
	assert.Equal(t, 0.5, score)
	assert.Error(t, err)
}

func TestCompareWithNilProvider_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	rm := selfimprove.NewAIRewardModel(nil, nil, nil, nil)
	_, err := rm.Compare(context.Background(), "prompt", "resp1", "resp2")
	require.Error(t, err)
}

func TestTrainWithNilExamples_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	rm := selfimprove.NewAIRewardModel(nil, nil, nil, nil)
	// nil slice should be treated as empty
	err := rm.Train(context.Background(), nil)
	require.NoError(t, err)
}

// ---------------------------------------------------------------------------
// Policy Optimizer security tests
// ---------------------------------------------------------------------------

func TestRollbackNonexistentUpdate_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	optimizer := selfimprove.NewLLMPolicyOptimizer(nil, nil, nil, nil)
	err := optimizer.Rollback(context.Background(), "nonexistent-id")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

func TestOptimizeWithInsufficientExamples_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	config := selfimprove.DefaultSelfImprovementConfig()
	config.MinExamplesForUpdate = 50

	optimizer := selfimprove.NewLLMPolicyOptimizer(nil, nil, config, nil)

	// Only 5 examples, need 50
	examples := make([]*selfimprove.TrainingExample, 5)
	for i := range examples {
		examples[i] = &selfimprove.TrainingExample{
			Prompt:      "test",
			RewardScore: 0.5,
		}
	}

	_, err := optimizer.Optimize(context.Background(), examples)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "insufficient")
}

func TestGetHistoryEmptyOptimizer_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	optimizer := selfimprove.NewLLMPolicyOptimizer(nil, nil, nil, nil)
	history, err := optimizer.GetHistory(context.Background(), 10)
	require.NoError(t, err)
	assert.Equal(t, 0, len(history))
}

func TestGetHistoryWithZeroLimit_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	optimizer := selfimprove.NewLLMPolicyOptimizer(nil, nil, nil, nil)

	// Apply a policy update first
	update := &selfimprove.PolicyUpdate{
		ID:        "test-update",
		NewPolicy: "test policy",
	}
	err := optimizer.Apply(context.Background(), update)
	require.NoError(t, err)

	// Zero limit returns all
	history, err := optimizer.GetHistory(context.Background(), 0)
	require.NoError(t, err)
	assert.Equal(t, 1, len(history))
}

func TestPolicyDailyLimit_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	config := selfimprove.DefaultSelfImprovementConfig()
	config.MaxPolicyUpdatesPerDay = 2

	optimizer := selfimprove.NewLLMPolicyOptimizer(nil, nil, config, nil)

	// Apply up to the limit
	for i := 0; i < 2; i++ {
		update := &selfimprove.PolicyUpdate{
			ID:        "update-" + string(rune('a'+i)),
			NewPolicy: "policy-" + string(rune('a'+i)),
		}
		err := optimizer.Apply(context.Background(), update)
		require.NoError(t, err)
	}

	// Third should fail
	err := optimizer.Apply(context.Background(), &selfimprove.PolicyUpdate{
		ID:        "update-c",
		NewPolicy: "policy-c",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "limit")
}

// ---------------------------------------------------------------------------
// AutoFeedbackCollector security tests
// ---------------------------------------------------------------------------

func TestAutoFeedbackWithNilRewardModel_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	afc := selfimprove.NewAutoFeedbackCollector(nil, nil, nil)
	_, err := afc.CollectAuto(context.Background(),
		"session", "prompt", "What?", "Response", "provider", "model")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "reward model")
}

// ---------------------------------------------------------------------------
// System-level security tests
// ---------------------------------------------------------------------------

func TestSystemCollectFeedbackNil_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	config := selfimprove.DefaultSelfImprovementConfig()
	config.AutoCollectFeedback = false

	system := selfimprove.NewSelfImprovementSystem(config, nil)
	err := system.Initialize(nil, nil)
	require.NoError(t, err)

	err = system.CollectFeedback(context.Background(), nil)
	require.Error(t, err)
}
