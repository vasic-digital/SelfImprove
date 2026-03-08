# SelfImprove - API Reference

**Module:** `digital.vasic.selfimprove`
**Package:** `selfimprove`

## Constructor Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `NewInMemoryRewardModel` | `NewInMemoryRewardModel(logger *logrus.Logger) *InMemoryRewardModel` | Creates an in-memory reward scoring model. |
| `NewInMemoryFeedbackStore` | `NewInMemoryFeedbackStore(logger *logrus.Logger) *InMemoryFeedbackStore` | Creates an in-memory feedback storage. |
| `NewInMemoryOptimizer` | `NewInMemoryOptimizer(reward RewardModel, feedback FeedbackStore, logger *logrus.Logger) *InMemoryOptimizer` | Creates an optimizer using reward signals and feedback. |

## Interfaces

### RewardModel

Multi-dimensional reward scoring for LLM outputs.

| Method | Signature | Description |
|--------|-----------|-------------|
| `Score` | `Score(ctx context.Context, input RewardInput) (*RewardSignal, error)` | Computes a weighted reward score. |
| `SetWeights` | `SetWeights(weights map[DimensionType]float64) error` | Updates dimension weights. |
| `GetWeights` | `GetWeights() map[DimensionType]float64` | Returns current dimension weights. |

### FeedbackStore

Stores and retrieves feedback entries.

| Method | Signature | Description |
|--------|-----------|-------------|
| `Store` | `Store(ctx context.Context, feedback *Feedback) error` | Persists a feedback entry. |
| `Get` | `Get(ctx context.Context, feedbackID string) (*Feedback, error)` | Retrieves a specific entry. |
| `List` | `List(ctx context.Context, filter FeedbackFilter) ([]*Feedback, error)` | Lists feedback matching a filter. |
| `GetBySession` | `GetBySession(ctx context.Context, sessionID string) ([]*Feedback, error)` | Retrieves all feedback for a session. |

### Optimizer

Uses reward signals and feedback to produce improvement recommendations.

| Method | Signature | Description |
|--------|-----------|-------------|
| `Optimize` | `Optimize(ctx context.Context, config OptimizeConfig) (*OptimizationResult, error)` | Runs the optimization pipeline. |
| `GetHistory` | `GetHistory(ctx context.Context) ([]*OptimizationResult, error)` | Returns past optimization results. |

## Core Types

### Feedback

```go
type Feedback struct {
    ID           string                    `json:"id"`
    SessionID    string                    `json:"session_id"`
    PromptID     string                    `json:"prompt_id"`
    ResponseID   string                    `json:"response_id"`
    Type         FeedbackType              `json:"type"`
    Source       FeedbackSource            `json:"source"`
    Score        float64                   `json:"score"`        // -1.0 to 1.0
    Dimensions   map[DimensionType]float64 `json:"dimensions,omitempty"`
    Comment      string                    `json:"comment,omitempty"`
    Correction   string                    `json:"correction,omitempty"`
    ProviderName string                    `json:"provider_name,omitempty"`
    Model        string                    `json:"model,omitempty"`
    Metadata     map[string]interface{}    `json:"metadata,omitempty"`
    CreatedAt    time.Time                 `json:"created_at"`
}
```

### TrainingExample

Represents a training example for model improvement.

```go
type TrainingExample struct {
    ID                string                    `json:"id"`
    Prompt            string                    `json:"prompt"`
    Response          string                    `json:"response"`
    PreferredResponse string                    `json:"preferred_response,omitempty"`
    RejectedResponse  string                    `json:"rejected_response,omitempty"`
    Feedback          []*Feedback               `json:"feedback"`
    RewardScore       float64                   `json:"reward_score"`
    Dimensions        map[DimensionType]float64 `json:"dimensions"`
    SystemPrompt      string                    `json:"system_prompt,omitempty"`
    ProviderName      string                    `json:"provider_name"`
    Model             string                    `json:"model"`
    Metadata          map[string]interface{}    `json:"metadata,omitempty"`
    CreatedAt         time.Time                 `json:"created_at"`
}
```

### PreferencePair

Pairwise preference comparison for DPO/RLAIF training.

```go
type PreferencePair struct {
    Prompt            string
    PreferredResponse string
    RejectedResponse  string
    PreferenceScore   float64
    Source            FeedbackSource
}
```

### RewardInput / RewardSignal

```go
type RewardInput struct {
    Prompt     string
    Response   string
    Dimensions map[DimensionType]float64
}

type RewardSignal struct {
    Score      float64
    Dimensions map[DimensionType]float64
    Confidence float64
}
```

### OptimizeConfig / OptimizationResult

```go
type OptimizeConfig struct {
    TargetProvider string
    TargetModel    string
    MinFeedback    int
    Dimensions     []DimensionType
}

type OptimizationResult struct {
    Summary         string
    Recommendations []Recommendation
    ImprovedScore   float64
    PreviousScore   float64
    Timestamp       time.Time
}

type Recommendation struct {
    Dimension DimensionType
    Action    string
    Impact    float64
}
```

## Enums

### FeedbackType

| Constant | Value |
|----------|-------|
| `FeedbackTypePositive` | `"positive"` |
| `FeedbackTypeNegative` | `"negative"` |
| `FeedbackTypeNeutral` | `"neutral"` |
| `FeedbackTypeSuggestion` | `"suggestion"` |
| `FeedbackTypeCorrection` | `"correction"` |

### FeedbackSource

| Constant | Value |
|----------|-------|
| `FeedbackSourceHuman` | `"human"` |
| `FeedbackSourceAI` | `"ai"` |
| `FeedbackSourceDebate` | `"debate"` |
| `FeedbackSourceVerifier` | `"verifier"` |
| `FeedbackSourceMetric` | `"metric"` |

### DimensionType

| Constant | Value |
|----------|-------|
| `DimensionAccuracy` | `"accuracy"` |
| `DimensionRelevance` | `"relevance"` |
| `DimensionHelpfulness` | `"helpfulness"` |
| `DimensionHarmless` | `"harmlessness"` |
| `DimensionHonest` | `"honesty"` |
| `DimensionCoherence` | `"coherence"` |
| `DimensionCreativity` | `"creativity"` |
| `DimensionFormatting` | `"formatting"` |
