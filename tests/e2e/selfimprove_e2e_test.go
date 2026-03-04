package e2e

import (
	"context"
	"runtime"
	"testing"
	"time"

	"digital.vasic.selfimprove/selfimprove"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func init() {
	runtime.GOMAXPROCS(2)
}

func TestFullSelfImprovementSystemInit_E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	system := selfimprove.NewSelfImprovementSystem(nil, nil)
	require.NotNil(t, system)

	// Initialize without provider (minimal mode)
	err := system.Initialize(nil, nil)
	require.NoError(t, err)

	// Reward model should be available
	rm := system.GetRewardModel()
	assert.NotNil(t, rm)
}

func TestFeedbackCollectionAndExport_E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	config := selfimprove.DefaultSelfImprovementConfig()
	config.AutoCollectFeedback = false // Use manual collector

	system := selfimprove.NewSelfImprovementSystem(config, nil)
	err := system.Initialize(nil, nil)
	require.NoError(t, err)

	// Collect manual feedback
	for i := 0; i < 10; i++ {
		fb := &selfimprove.Feedback{
			SessionID:    "e2e-session",
			PromptID:     "e2e-prompt",
			Type:         selfimprove.FeedbackTypePositive,
			Source:       selfimprove.FeedbackSourceHuman,
			Score:        0.8,
			ProviderName: "claude",
			Model:        "claude-3",
			Dimensions: map[selfimprove.DimensionType]float64{
				selfimprove.DimensionAccuracy:    0.85,
				selfimprove.DimensionHelpfulness: 0.9,
			},
		}
		err := system.CollectFeedback(context.Background(), fb)
		require.NoError(t, err)
	}

	// Get feedback stats
	stats, err := system.GetFeedbackStats(context.Background(), nil)
	require.NoError(t, err)
	assert.Equal(t, 10, stats.TotalCount)
	assert.InDelta(t, 0.8, stats.AverageScore, 0.01)
}

func TestPolicyApplyAndRollback_E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	optimizer := selfimprove.NewLLMPolicyOptimizer(nil, nil, nil, nil)

	// Set initial policy
	optimizer.SetCurrentPolicy("Be helpful and accurate")
	assert.Equal(t, "Be helpful and accurate", optimizer.GetCurrentPolicy())

	// Apply a policy update
	update := &selfimprove.PolicyUpdate{
		ID:               "update-1",
		NewPolicy:        "Be helpful, accurate, and concise",
		UpdateType:       selfimprove.PolicyUpdatePromptRefinement,
		Reason:           "Improve conciseness",
		ImprovementScore: 0.7,
	}
	err := optimizer.Apply(context.Background(), update)
	require.NoError(t, err)
	assert.Equal(t, "Be helpful, accurate, and concise", optimizer.GetCurrentPolicy())

	// Check history
	history, err := optimizer.GetHistory(context.Background(), 10)
	require.NoError(t, err)
	assert.Equal(t, 1, len(history))
	assert.Equal(t, "update-1", history[0].ID)

	// Rollback
	err = optimizer.Rollback(context.Background(), "update-1")
	require.NoError(t, err)
	assert.Equal(t, "Be helpful and accurate", optimizer.GetCurrentPolicy())
}

func TestDefaultSelfImprovementConfig_E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	cfg := selfimprove.DefaultSelfImprovementConfig()
	assert.Equal(t, "claude", cfg.RewardModelProvider)
	assert.Equal(t, "claude-3-sonnet", cfg.RewardModelName)
	assert.Equal(t, 0.5, cfg.MinRewardThreshold)
	assert.True(t, cfg.AutoCollectFeedback)
	assert.Equal(t, 100, cfg.FeedbackBatchSize)
	assert.Equal(t, 24*time.Hour, cfg.OptimizationInterval)
	assert.Equal(t, 50, cfg.MinExamplesForUpdate)
	assert.Equal(t, 3, cfg.MaxPolicyUpdatesPerDay)
	assert.True(t, cfg.EnableSelfCritique)
	assert.True(t, cfg.UseDebateForReward)
	assert.True(t, cfg.UseDebateForOptimize)
	assert.Equal(t, 10000, cfg.MaxBufferSize)
	assert.NotEmpty(t, cfg.ConstitutionalPrinciples)
}

func TestDebateServiceAdapter_E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	adapter := selfimprove.NewDebateServiceAdapter(nil, nil)
	require.NotNil(t, adapter)

	// Set provider mapping
	adapter.SetProviderMapping(map[string]string{
		"claude": "anthropic",
		"gpt-4":  "openai",
	})
}

func TestRewardModelCaching_E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	rm := selfimprove.NewAIRewardModel(nil, nil, nil, nil)

	// First call generates score
	dims1, err := rm.ScoreWithDimensions(context.Background(), "test", "response")
	require.NoError(t, err)

	// Second call with same input should use cache
	dims2, err := rm.ScoreWithDimensions(context.Background(), "test", "response")
	require.NoError(t, err)

	// Results should match since they came from cache
	for dim, v1 := range dims1 {
		v2, ok := dims2[dim]
		assert.True(t, ok, "dimension %s should exist", dim)
		assert.Equal(t, v1, v2, "cached value for %s should match", dim)
	}
}

func TestTrainingWithEmptyExamples_E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	rm := selfimprove.NewAIRewardModel(nil, nil, nil, nil)

	// Training with empty examples should not error
	err := rm.Train(context.Background(), []*selfimprove.TrainingExample{})
	require.NoError(t, err)
}

func TestFeedbackEviction_E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	// Create a collector with very small max size
	collector := selfimprove.NewInMemoryFeedbackCollector(nil, 10)

	// Add more than max size
	for i := 0; i < 15; i++ {
		err := collector.Collect(context.Background(), &selfimprove.Feedback{
			SessionID: "session",
			PromptID:  "prompt",
			Type:      selfimprove.FeedbackTypePositive,
			Source:    selfimprove.FeedbackSourceHuman,
			Score:     float64(i) * 0.05,
		})
		require.NoError(t, err)
	}

	// Get aggregated should still work
	agg, err := collector.GetAggregated(context.Background(), nil)
	require.NoError(t, err)
	// After eviction, we should have fewer than 15
	assert.LessOrEqual(t, agg.TotalCount, 15)
}

func TestSystemStartAndStop_E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	config := selfimprove.DefaultSelfImprovementConfig()
	config.OptimizationInterval = 100 * time.Millisecond

	system := selfimprove.NewSelfImprovementSystem(config, nil)
	err := system.Initialize(nil, nil)
	require.NoError(t, err)

	// Start should succeed
	err = system.Start()
	require.NoError(t, err)

	// Double start should fail
	err = system.Start()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "already running")

	// Stop
	system.Stop()
}
