package benchmark

import (
	"context"
	"fmt"
	"runtime"
	"testing"

	"digital.vasic.selfimprove/selfimprove"
	"github.com/stretchr/testify/require"
)

func init() {
	runtime.GOMAXPROCS(2)
}

func BenchmarkFeedbackCollect(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	collector := selfimprove.NewInMemoryFeedbackCollector(nil, b.N+100)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = collector.Collect(context.Background(), &selfimprove.Feedback{
			SessionID:    fmt.Sprintf("session-%d", i),
			PromptID:     fmt.Sprintf("prompt-%d", i),
			Type:         selfimprove.FeedbackTypePositive,
			Source:       selfimprove.FeedbackSourceHuman,
			Score:        0.8,
			ProviderName: "test",
		})
	}
}

func BenchmarkFeedbackGetBySession(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	collector := selfimprove.NewInMemoryFeedbackCollector(nil, 10000)
	for i := 0; i < 100; i++ {
		_ = collector.Collect(context.Background(), &selfimprove.Feedback{
			SessionID: "target-session",
			PromptID:  fmt.Sprintf("prompt-%d", i),
			Score:     0.8,
		})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = collector.GetBySession(context.Background(), "target-session")
	}
}

func BenchmarkFeedbackGetByPrompt(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	collector := selfimprove.NewInMemoryFeedbackCollector(nil, 10000)
	for i := 0; i < 100; i++ {
		_ = collector.Collect(context.Background(), &selfimprove.Feedback{
			SessionID: fmt.Sprintf("session-%d", i),
			PromptID:  "target-prompt",
			Score:     0.8,
		})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = collector.GetByPrompt(context.Background(), "target-prompt")
	}
}

func BenchmarkFeedbackGetAggregated(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	collector := selfimprove.NewInMemoryFeedbackCollector(nil, 10000)
	for i := 0; i < 200; i++ {
		_ = collector.Collect(context.Background(), &selfimprove.Feedback{
			SessionID:    fmt.Sprintf("session-%d", i%10),
			PromptID:     fmt.Sprintf("prompt-%d", i%20),
			Type:         selfimprove.FeedbackTypePositive,
			Source:       selfimprove.FeedbackSourceHuman,
			Score:        float64(i%10) * 0.1,
			ProviderName: "provider",
			Dimensions: map[selfimprove.DimensionType]float64{
				selfimprove.DimensionAccuracy:    0.8,
				selfimprove.DimensionHelpfulness: 0.9,
			},
		})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = collector.GetAggregated(context.Background(), nil)
	}
}

func BenchmarkFeedbackExport(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	collector := selfimprove.NewInMemoryFeedbackCollector(nil, 10000)
	for i := 0; i < 100; i++ {
		_ = collector.Collect(context.Background(), &selfimprove.Feedback{
			SessionID:    "session",
			PromptID:     fmt.Sprintf("prompt-%d", i%10),
			Type:         selfimprove.FeedbackTypePositive,
			Source:       selfimprove.FeedbackSourceHuman,
			Score:        0.8,
			ProviderName: "provider",
		})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = collector.Export(context.Background(), nil)
	}
}

func BenchmarkRewardModelScore(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	rm := selfimprove.NewAIRewardModel(nil, nil, nil, nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = rm.Score(context.Background(), "What is Go?", "Go is a programming language")
	}
}

func BenchmarkRewardModelScoreWithDimensions(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	rm := selfimprove.NewAIRewardModel(nil, nil, nil, nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = rm.ScoreWithDimensions(context.Background(),
			fmt.Sprintf("prompt-%d", i),
			fmt.Sprintf("response-%d", i))
	}
}

func BenchmarkRewardModelTrain(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	rm := selfimprove.NewAIRewardModel(nil, nil, nil, nil)

	examples := make([]*selfimprove.TrainingExample, 50)
	for i := range examples {
		examples[i] = &selfimprove.TrainingExample{
			Prompt:      fmt.Sprintf("prompt-%d", i),
			Response:    fmt.Sprintf("response-%d", i),
			RewardScore: float64(i%10) * 0.1,
			Dimensions: map[selfimprove.DimensionType]float64{
				selfimprove.DimensionAccuracy:    float64(i%10) * 0.1,
				selfimprove.DimensionRelevance:   0.8,
				selfimprove.DimensionHelpfulness: 0.7,
			},
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = rm.Train(context.Background(), examples)
	}
}

func BenchmarkPolicyOptimizerSetGet(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	optimizer := selfimprove.NewLLMPolicyOptimizer(nil, nil, nil, nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optimizer.SetCurrentPolicy(fmt.Sprintf("policy-%d", i))
		_ = optimizer.GetCurrentPolicy()
	}
}

func BenchmarkPolicyOptimizerApply(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	config := selfimprove.DefaultSelfImprovementConfig()
	config.MaxPolicyUpdatesPerDay = b.N + 100

	optimizer := selfimprove.NewLLMPolicyOptimizer(nil, nil, config, nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = optimizer.Apply(context.Background(), &selfimprove.PolicyUpdate{
			ID:               fmt.Sprintf("update-%d", i),
			NewPolicy:        fmt.Sprintf("policy-%d", i),
			UpdateType:       selfimprove.PolicyUpdatePromptRefinement,
			ImprovementScore: 0.5,
		})
	}
}

func BenchmarkPolicyOptimizerGetHistory(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	config := selfimprove.DefaultSelfImprovementConfig()
	config.MaxPolicyUpdatesPerDay = 200

	optimizer := selfimprove.NewLLMPolicyOptimizer(nil, nil, config, nil)

	for i := 0; i < 100; i++ {
		_ = optimizer.Apply(context.Background(), &selfimprove.PolicyUpdate{
			ID:        fmt.Sprintf("update-%d", i),
			NewPolicy: fmt.Sprintf("policy-%d", i),
		})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = optimizer.GetHistory(context.Background(), 10)
	}
}

func BenchmarkDefaultSelfImprovementConfig(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cfg := selfimprove.DefaultSelfImprovementConfig()
		require.NotNil(b, cfg)
	}
}

func BenchmarkNewInMemoryFeedbackCollector(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = selfimprove.NewInMemoryFeedbackCollector(nil, 1000)
	}
}
