package integration

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

func TestRewardModelAndFeedbackIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	// Create reward model without provider (uses heuristic scoring)
	rm := selfimprove.NewAIRewardModel(nil, nil, nil, nil)

	// Score should return a default score when no provider is available
	score, err := rm.Score(context.Background(), "What is Go?", "Go is a programming language")
	// Without a provider, it returns 0.5 with an error
	assert.Equal(t, 0.5, score)
	assert.Error(t, err)

	// ScoreWithDimensions should return default dimension scores
	dims, err := rm.ScoreWithDimensions(context.Background(),
		"What is Go?", "Go is a programming language")
	require.NoError(t, err)
	assert.NotEmpty(t, dims)

	// Every dimension should have a value
	for _, v := range dims {
		assert.GreaterOrEqual(t, v, 0.0)
		assert.LessOrEqual(t, v, 1.0)
	}
}

func TestFeedbackCollectorIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	collector := selfimprove.NewInMemoryFeedbackCollector(nil, 1000)

	// Collect feedback entries
	fb1 := &selfimprove.Feedback{
		SessionID:    "session-1",
		PromptID:     "prompt-1",
		Type:         selfimprove.FeedbackTypePositive,
		Source:       selfimprove.FeedbackSourceHuman,
		Score:        0.9,
		ProviderName: "openai",
		Model:        "gpt-4",
		Dimensions: map[selfimprove.DimensionType]float64{
			selfimprove.DimensionAccuracy:    0.95,
			selfimprove.DimensionHelpfulness: 0.85,
		},
	}
	err := collector.Collect(context.Background(), fb1)
	require.NoError(t, err)
	assert.NotEmpty(t, fb1.ID)

	fb2 := &selfimprove.Feedback{
		SessionID:    "session-1",
		PromptID:     "prompt-2",
		Type:         selfimprove.FeedbackTypeNegative,
		Source:       selfimprove.FeedbackSourceAI,
		Score:        0.3,
		ProviderName: "openai",
		Model:        "gpt-4",
	}
	err = collector.Collect(context.Background(), fb2)
	require.NoError(t, err)

	// Retrieve by session
	sessionFeedback, err := collector.GetBySession(context.Background(), "session-1")
	require.NoError(t, err)
	assert.Equal(t, 2, len(sessionFeedback))

	// Retrieve by prompt
	promptFeedback, err := collector.GetByPrompt(context.Background(), "prompt-1")
	require.NoError(t, err)
	assert.Equal(t, 1, len(promptFeedback))

	// Get aggregated stats
	agg, err := collector.GetAggregated(context.Background(), nil)
	require.NoError(t, err)
	assert.Equal(t, 2, agg.TotalCount)
	assert.InDelta(t, 0.6, agg.AverageScore, 0.01)
}

func TestFeedbackCollectorExportIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	collector := selfimprove.NewInMemoryFeedbackCollector(nil, 1000)

	for i := 0; i < 5; i++ {
		err := collector.Collect(context.Background(), &selfimprove.Feedback{
			SessionID:    "session-1",
			PromptID:     "prompt-A",
			Type:         selfimprove.FeedbackTypePositive,
			Source:       selfimprove.FeedbackSourceHuman,
			Score:        0.8,
			ProviderName: "claude",
			Model:        "claude-3",
		})
		require.NoError(t, err)
	}

	examples, err := collector.Export(context.Background(), nil)
	require.NoError(t, err)
	assert.Greater(t, len(examples), 0)

	// Each example should have a reward score
	for _, ex := range examples {
		assert.NotEmpty(t, ex.ID)
		assert.Greater(t, ex.RewardScore, 0.0)
	}
}

func TestPolicyOptimizerIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	optimizer := selfimprove.NewLLMPolicyOptimizer(nil, nil, nil, nil)

	// Set and get policy
	optimizer.SetCurrentPolicy("Be helpful and accurate")
	policy := optimizer.GetCurrentPolicy()
	assert.Equal(t, "Be helpful and accurate", policy)

	// History should be empty
	history, err := optimizer.GetHistory(context.Background(), 10)
	require.NoError(t, err)
	assert.Equal(t, 0, len(history))
}

func TestRewardModelTrainIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	rm := selfimprove.NewAIRewardModel(nil, nil, nil, nil)

	examples := []*selfimprove.TrainingExample{
		{
			Prompt:      "What is Go?",
			Response:    "Go is a programming language",
			RewardScore: 0.9,
			Dimensions: map[selfimprove.DimensionType]float64{
				selfimprove.DimensionAccuracy:  0.95,
				selfimprove.DimensionRelevance: 0.9,
			},
		},
		{
			Prompt:      "What is Python?",
			Response:    "I don't know",
			RewardScore: 0.2,
			Dimensions: map[selfimprove.DimensionType]float64{
				selfimprove.DimensionAccuracy:  0.1,
				selfimprove.DimensionRelevance: 0.3,
			},
		},
	}

	// Train should not error even without a provider
	err := rm.Train(context.Background(), examples)
	require.NoError(t, err)
}

func TestFeedbackFilterIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	collector := selfimprove.NewInMemoryFeedbackCollector(nil, 1000)

	// Add different types of feedback
	for i := 0; i < 10; i++ {
		fbType := selfimprove.FeedbackTypePositive
		if i%2 == 0 {
			fbType = selfimprove.FeedbackTypeNegative
		}
		err := collector.Collect(context.Background(), &selfimprove.Feedback{
			SessionID:    "session-1",
			PromptID:     "prompt-1",
			Type:         fbType,
			Source:       selfimprove.FeedbackSourceHuman,
			Score:        float64(i) * 0.1,
			ProviderName: "openai",
		})
		require.NoError(t, err)
	}

	// Filter by type
	agg, err := collector.GetAggregated(context.Background(), &selfimprove.FeedbackFilter{
		Types: []selfimprove.FeedbackType{selfimprove.FeedbackTypePositive},
	})
	require.NoError(t, err)
	assert.Equal(t, 5, agg.TotalCount)

	// Filter by score
	minScore := 0.5
	agg, err = collector.GetAggregated(context.Background(), &selfimprove.FeedbackFilter{
		MinScore: &minScore,
	})
	require.NoError(t, err)
	assert.Greater(t, agg.TotalCount, 0)
}
