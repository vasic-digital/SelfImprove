package stress

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"

	"digital.vasic.selfimprove/selfimprove"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func init() {
	runtime.GOMAXPROCS(2)
}

func TestConcurrentFeedbackCollection_Stress(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")  // SKIP-OK: #short-mode
	}

	const goroutines = 100

	collector := selfimprove.NewInMemoryFeedbackCollector(nil, 10000)

	var wg sync.WaitGroup
	var successCount atomic.Int64

	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func(idx int) {
			defer wg.Done()

			fb := &selfimprove.Feedback{
				SessionID:    fmt.Sprintf("session-%d", idx),
				PromptID:     fmt.Sprintf("prompt-%d", idx%10),
				Type:         selfimprove.FeedbackTypePositive,
				Source:       selfimprove.FeedbackSourceHuman,
				Score:        float64(idx%10) * 0.1,
				ProviderName: "test-provider",
				Model:        "test-model",
			}
			if err := collector.Collect(context.Background(), fb); err == nil {
				successCount.Add(1)
			}
		}(i)
	}

	wg.Wait()
	assert.Equal(t, int64(goroutines), successCount.Load())
}

func TestConcurrentFeedbackRetrieval_Stress(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")  // SKIP-OK: #short-mode
	}

	const goroutines = 75

	collector := selfimprove.NewInMemoryFeedbackCollector(nil, 10000)

	// Pre-populate with feedback
	for i := 0; i < 50; i++ {
		err := collector.Collect(context.Background(), &selfimprove.Feedback{
			SessionID: fmt.Sprintf("session-%d", i%5),
			PromptID:  fmt.Sprintf("prompt-%d", i%10),
			Type:      selfimprove.FeedbackTypePositive,
			Source:    selfimprove.FeedbackSourceHuman,
			Score:     0.8,
		})
		require.NoError(t, err)
	}

	var wg sync.WaitGroup
	var successCount atomic.Int64

	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func(idx int) {
			defer wg.Done()

			switch idx % 3 {
			case 0:
				_, err := collector.GetBySession(
					context.Background(), fmt.Sprintf("session-%d", idx%5))
				if err == nil {
					successCount.Add(1)
				}
			case 1:
				_, err := collector.GetByPrompt(
					context.Background(), fmt.Sprintf("prompt-%d", idx%10))
				if err == nil {
					successCount.Add(1)
				}
			case 2:
				_, err := collector.GetAggregated(context.Background(), nil)
				if err == nil {
					successCount.Add(1)
				}
			}
		}(i)
	}

	wg.Wait()
	assert.Equal(t, int64(goroutines), successCount.Load())
}

func TestConcurrentRewardScoring_Stress(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")  // SKIP-OK: #short-mode
	}

	const goroutines = 80

	rm := selfimprove.NewAIRewardModel(nil, nil, nil, nil)

	var wg sync.WaitGroup
	var completedCount atomic.Int64

	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func(idx int) {
			defer wg.Done()

			_, _ = rm.Score(context.Background(),
				fmt.Sprintf("prompt-%d", idx),
				fmt.Sprintf("response-%d", idx))
			completedCount.Add(1)
		}(i)
	}

	wg.Wait()
	assert.Equal(t, int64(goroutines), completedCount.Load())
}

func TestConcurrentScoreWithDimensions_Stress(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")  // SKIP-OK: #short-mode
	}

	const goroutines = 60

	rm := selfimprove.NewAIRewardModel(nil, nil, nil, nil)

	var wg sync.WaitGroup
	var completedCount atomic.Int64

	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func(idx int) {
			defer wg.Done()

			dims, err := rm.ScoreWithDimensions(context.Background(),
				fmt.Sprintf("prompt-%d", idx),
				fmt.Sprintf("response-%d", idx))
			if err == nil && len(dims) > 0 {
				completedCount.Add(1)
			}
		}(i)
	}

	wg.Wait()
	assert.Equal(t, int64(goroutines), completedCount.Load())
}

func TestConcurrentPolicyOperations_Stress(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")  // SKIP-OK: #short-mode
	}

	const goroutines = 50

	optimizer := selfimprove.NewLLMPolicyOptimizer(nil, nil, nil, nil)
	optimizer.SetCurrentPolicy("initial policy")

	var wg sync.WaitGroup
	var readCount atomic.Int64
	var writeCount atomic.Int64

	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func(idx int) {
			defer wg.Done()

			if idx%3 == 0 {
				// Write operation
				optimizer.SetCurrentPolicy(fmt.Sprintf("policy-%d", idx))
				writeCount.Add(1)
			} else {
				// Read operation
				policy := optimizer.GetCurrentPolicy()
				if policy != "" {
					readCount.Add(1)
				}
			}
		}(i)
	}

	wg.Wait()
	total := readCount.Load() + writeCount.Load()
	assert.Equal(t, int64(goroutines), total)
}

func TestConcurrentTraining_Stress(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")  // SKIP-OK: #short-mode
	}

	const goroutines = 50

	rm := selfimprove.NewAIRewardModel(nil, nil, nil, nil)

	var wg sync.WaitGroup
	var successCount atomic.Int64

	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func(idx int) {
			defer wg.Done()

			examples := []*selfimprove.TrainingExample{
				{
					Prompt:      fmt.Sprintf("prompt-%d", idx),
					Response:    fmt.Sprintf("response-%d", idx),
					RewardScore: float64(idx%10) * 0.1,
					Dimensions: map[selfimprove.DimensionType]float64{
						selfimprove.DimensionAccuracy: float64(idx%10) * 0.1,
					},
				},
			}
			if err := rm.Train(context.Background(), examples); err == nil {
				successCount.Add(1)
			}
		}(i)
	}

	wg.Wait()
	assert.Equal(t, int64(goroutines), successCount.Load())
}

func TestConcurrentFeedbackCollectionAndExport_Stress(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")  // SKIP-OK: #short-mode
	}

	const writers = 50
	const readers = 25

	collector := selfimprove.NewInMemoryFeedbackCollector(nil, 10000)

	var wg sync.WaitGroup
	var writeSuccess atomic.Int64
	var readSuccess atomic.Int64

	// Writers
	wg.Add(writers)
	for i := 0; i < writers; i++ {
		go func(idx int) {
			defer wg.Done()

			fb := &selfimprove.Feedback{
				SessionID: "session-stress",
				PromptID:  fmt.Sprintf("prompt-%d", idx),
				Type:      selfimprove.FeedbackTypePositive,
				Source:    selfimprove.FeedbackSourceHuman,
				Score:     0.8,
			}
			if err := collector.Collect(context.Background(), fb); err == nil {
				writeSuccess.Add(1)
			}
		}(i)
	}

	// Readers
	wg.Add(readers)
	for i := 0; i < readers; i++ {
		go func(idx int) {
			defer wg.Done()

			_, err := collector.Export(context.Background(), nil)
			if err == nil {
				readSuccess.Add(1)
			}
		}(i)
	}

	wg.Wait()
	assert.Equal(t, int64(writers), writeSuccess.Load())
	assert.Equal(t, int64(readers), readSuccess.Load())
}
