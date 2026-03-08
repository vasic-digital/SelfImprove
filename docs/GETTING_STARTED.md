# SelfImprove - Getting Started

**Module:** `digital.vasic.selfimprove`

## Installation

```go
import "digital.vasic.selfimprove/selfimprove"
```

## Quick Start: Setting Up a Self-Improvement Loop

### 1. Create Core Components

```go
package main

import (
    "context"
    "fmt"

    "digital.vasic.selfimprove/selfimprove"
    "github.com/sirupsen/logrus"
)

func main() {
    logger := logrus.New()

    rewardModel := selfimprove.NewInMemoryRewardModel(logger)
    feedbackStore := selfimprove.NewInMemoryFeedbackStore(logger)
    optimizer := selfimprove.NewInMemoryOptimizer(rewardModel, feedbackStore, logger)
```

### 2. Collect Feedback

Record human or AI feedback on model responses:

```go
    feedback := &selfimprove.Feedback{
        SessionID:    "session-001",
        PromptID:     "prompt-001",
        ResponseID:   "response-001",
        Type:         selfimprove.FeedbackTypePositive,
        Source:       selfimprove.FeedbackSourceHuman,
        Score:        0.85,
        ProviderName: "openai",
        Model:        "gpt-4",
        Dimensions: map[selfimprove.DimensionType]float64{
            selfimprove.DimensionAccuracy:    0.90,
            selfimprove.DimensionRelevance:   0.85,
            selfimprove.DimensionHelpfulness: 0.80,
            selfimprove.DimensionCoherence:   0.88,
        },
        Comment: "Clear and accurate response",
    }

    err := feedbackStore.Store(context.Background(), feedback)
    if err != nil {
        panic(err)
    }
```

### 3. Compute Reward Scores

The reward model evaluates responses across weighted dimensions:

```go
    signal, err := rewardModel.Score(context.Background(), selfimprove.RewardInput{
        Prompt:   "What is the capital of France?",
        Response: "The capital of France is Paris.",
        Dimensions: map[selfimprove.DimensionType]float64{
            selfimprove.DimensionAccuracy:  1.0,
            selfimprove.DimensionRelevance: 0.95,
        },
    })
    if err != nil {
        panic(err)
    }

    fmt.Printf("Reward score: %.2f\n", signal.Score)
```

### 4. Run the Optimizer

The optimizer uses accumulated feedback and reward signals to
produce optimization recommendations:

```go
    result, err := optimizer.Optimize(context.Background(), selfimprove.OptimizeConfig{
        TargetProvider: "openai",
        TargetModel:    "gpt-4",
        MinFeedback:    10,
        Dimensions:     []selfimprove.DimensionType{
            selfimprove.DimensionAccuracy,
            selfimprove.DimensionHelpfulness,
        },
    })
    if err != nil {
        panic(err)
    }

    fmt.Printf("Optimization: %s\n", result.Summary)
    for _, rec := range result.Recommendations {
        fmt.Printf("  - %s: %s\n", rec.Dimension, rec.Action)
    }
}
```

## Feedback Types

| Type | Constant | Description |
|------|----------|-------------|
| Positive | `FeedbackTypePositive` | Response was good |
| Negative | `FeedbackTypeNegative` | Response was bad |
| Neutral | `FeedbackTypeNeutral` | No strong preference |
| Suggestion | `FeedbackTypeSuggestion` | Improvement suggested |
| Correction | `FeedbackTypeCorrection` | Corrected response provided |

## Feedback Sources

| Source | Constant | Description |
|--------|----------|-------------|
| Human | `FeedbackSourceHuman` | Direct user feedback |
| AI | `FeedbackSourceAI` | Automated AI evaluation |
| Debate | `FeedbackSourceDebate` | From debate system |
| Verifier | `FeedbackSourceVerifier` | From verification pipeline |
| Metric | `FeedbackSourceMetric` | From metric collection |

## Evaluation Dimensions

| Dimension | Constant | Description |
|-----------|----------|-------------|
| Accuracy | `DimensionAccuracy` | Factual correctness |
| Relevance | `DimensionRelevance` | Pertinence to the query |
| Helpfulness | `DimensionHelpfulness` | Practical usefulness |
| Harmlessness | `DimensionHarmless` | Safety and harmlessness |
| Honesty | `DimensionHonest` | Truthfulness about uncertainty |
| Coherence | `DimensionCoherence` | Logical consistency |
| Creativity | `DimensionCreativity` | Novel insights |
| Formatting | `DimensionFormatting` | Presentation quality |

## Preference Pairs (DPO/RLAIF)

For direct preference optimization, record pairwise comparisons:

```go
pair := &selfimprove.PreferencePair{
    Prompt:            "Explain Go interfaces",
    PreferredResponse: "Interfaces in Go define method sets...",
    RejectedResponse:  "Go has interfaces. They are types.",
    PreferenceScore:   0.92,
    Source:            selfimprove.FeedbackSourceHuman,
}
```

## The Improvement Loop

```
LLM Response --> Reward Scoring --> Feedback Collection
                                          |
                                    Optimization Engine
                                          |
                                    Updated Weights/Prompts
                                          |
                                    Better Responses (next iteration)
```

## Integration with HelixAgent

The SelfImprove module integrates through:

- **Adapter** at `internal/adapters/selfimprove/adapter.go`
- Connects to the debate system for debate-sourced feedback
- Feeds optimization results into provider selection weights

## Next Steps

- See [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- See [API_REFERENCE.md](API_REFERENCE.md) for the full type catalog
