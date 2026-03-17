# SelfImprove

`digital.vasic.selfimprove` -- AI self-improvement through RLHF, reward modelling, feedback integration, policy optimization, and dimension-weighted scoring.

## Overview

SelfImprove is a Go module that implements a complete Reinforcement Learning from AI Feedback (RLAIF) pipeline for continuous AI quality improvement. It provides reward models that score LLM responses across multiple quality dimensions (accuracy, relevance, helpfulness, harmlessness, honesty, coherence, creativity, formatting), feedback collection with aggregation and trend analysis, and policy optimization that refines system prompts based on accumulated evidence.

The module supports two evaluation modes: single-LLM evaluation where one provider scores responses, and multi-LLM debate evaluation where an ensemble of models collectively assess response quality through structured debate. Constitutional AI principles are enforced through self-critique filtering, ensuring that policy updates never violate safety constraints.

SelfImprove integrates with HelixAgent's debate service and LLMsVerifier to leverage provider trust scores when making optimization decisions. The background optimization loop periodically exports feedback as training examples, generates policy improvements, and applies them with configurable daily rate limits and minimum improvement thresholds.

## Architecture

```
                    +---------------------------+
                    |  SelfImprovementSystem    |
                    |  (Main Orchestrator)      |
                    +------+--------+-----------+
                           |        |
              +------------+        +-------------+
              |                                   |
   +----------v----------+           +-----------v-----------+
   |   AIRewardModel     |           |  LLMPolicyOptimizer   |
   | (Score, Compare,    |           | (Optimize, Apply,     |
   |  ScoreWithDimensions)|          |  Rollback, Critique)  |
   +----------+----------+           +-----------+-----------+
              |                                   |
   +----------v----------+           +-----------v-----------+
   | Debate / LLM Eval   |           | Debate / LLM Analysis |
   | (Fallback chain)    |           | (Fallback chain)      |
   +---------------------+           +-----------------------+
              |
   +----------v-----------+
   | FeedbackCollector     |
   | (InMemory / Auto)     |
   | Collect, Aggregate,   |
   | Export, Trend          |
   +------------------------+
```

## Package Structure

| Package | Purpose |
|---------|---------|
| `selfimprove` | Core module: reward models, feedback collection, policy optimization, integration orchestrator |

### Source Files

| File | Description |
|------|-------------|
| `types.go` | All type definitions, interfaces, enums, and configuration defaults |
| `reward.go` | `AIRewardModel` -- LLM/debate-based response scoring with caching and dimension weights |
| `feedback.go` | `InMemoryFeedbackCollector` and `AutoFeedbackCollector` -- feedback storage, filtering, aggregation, export |
| `optimizer.go` | `LLMPolicyOptimizer` -- feedback pattern analysis, policy generation, constitutional self-critique |
| `integration.go` | `SelfImprovementSystem` orchestrator, `DebateServiceAdapter`, background optimization loop |

## API Reference

### Types

**Feedback types**: `FeedbackTypePositive`, `FeedbackTypeNegative`, `FeedbackTypeNeutral`, `FeedbackTypeSuggestion`, `FeedbackTypeCorrection`

**Feedback sources**: `FeedbackSourceHuman`, `FeedbackSourceAI`, `FeedbackSourceDebate`, `FeedbackSourceVerifier`, `FeedbackSourceMetric`

**Evaluation dimensions**: `DimensionAccuracy`, `DimensionRelevance`, `DimensionHelpfulness`, `DimensionHarmless`, `DimensionHonest`, `DimensionCoherence`, `DimensionCreativity`, `DimensionFormatting`

**Policy update types**: `PolicyUpdatePromptRefinement`, `PolicyUpdateGuidelineAddition`, `PolicyUpdateExampleAddition`, `PolicyUpdateConstraintUpdate`, `PolicyUpdateToneAdjustment`

### Core Interfaces

```go
// RewardModel evaluates response quality
type RewardModel interface {
    Score(ctx context.Context, prompt, response string) (float64, error)
    ScoreWithDimensions(ctx context.Context, prompt, response string) (map[DimensionType]float64, error)
    Compare(ctx context.Context, prompt, response1, response2 string) (*PreferencePair, error)
    Train(ctx context.Context, examples []*TrainingExample) error
}

// FeedbackCollector collects and processes feedback
type FeedbackCollector interface {
    Collect(ctx context.Context, feedback *Feedback) error
    GetBySession(ctx context.Context, sessionID string) ([]*Feedback, error)
    GetByPrompt(ctx context.Context, promptID string) ([]*Feedback, error)
    GetAggregated(ctx context.Context, filter *FeedbackFilter) (*AggregatedFeedback, error)
    Export(ctx context.Context, filter *FeedbackFilter) ([]*TrainingExample, error)
}

// PolicyOptimizer optimizes policies based on feedback
type PolicyOptimizer interface {
    Optimize(ctx context.Context, examples []*TrainingExample) ([]*PolicyUpdate, error)
    Apply(ctx context.Context, update *PolicyUpdate) error
    Rollback(ctx context.Context, updateID string) error
    GetHistory(ctx context.Context, limit int) ([]*PolicyUpdate, error)
    GetCurrentPolicy() string
    SetCurrentPolicy(policy string)
}
```

### SelfImprovementSystem Methods

```go
func NewSelfImprovementSystem(config *SelfImprovementConfig, logger *logrus.Logger) *SelfImprovementSystem
func (sis *SelfImprovementSystem) Initialize(provider LLMProvider, debateService DebateService) error
func (sis *SelfImprovementSystem) Start() error
func (sis *SelfImprovementSystem) Stop()
func (sis *SelfImprovementSystem) CollectFeedback(ctx context.Context, feedback *Feedback) error
func (sis *SelfImprovementSystem) CollectAutoFeedback(ctx, sessionID, promptID, prompt, response, provider, model string) (*Feedback, error)
func (sis *SelfImprovementSystem) ScoreResponse(ctx context.Context, prompt, response string) (float64, error)
func (sis *SelfImprovementSystem) CompareResponses(ctx context.Context, prompt, r1, r2 string) (*PreferencePair, error)
func (sis *SelfImprovementSystem) GetFeedbackStats(ctx context.Context, filter *FeedbackFilter) (*AggregatedFeedback, error)
func (sis *SelfImprovementSystem) GetPolicyHistory(ctx context.Context, limit int) ([]*PolicyUpdate, error)
func (sis *SelfImprovementSystem) RollbackPolicy(ctx context.Context, updateID string) error
```

## Usage Examples

### Basic setup and auto-feedback collection

```go
config := selfimprove.DefaultSelfImprovementConfig()
config.AutoCollectFeedback = true
config.UseDebateForReward = true

system := selfimprove.NewSelfImprovementSystem(config, logger)
err := system.Initialize(llmProvider, debateService)
system.Start() // starts background optimization loop

// Auto-collect feedback on every response
feedback, err := system.CollectAutoFeedback(ctx,
    sessionID, promptID, prompt, response, "claude", "claude-3-sonnet")

// Score a response manually
score, err := system.ScoreResponse(ctx, prompt, response)

// Compare two responses (DPO-style)
pair, err := system.CompareResponses(ctx, prompt, responseA, responseB)
// pair.Chosen, pair.Rejected, pair.Margin

system.Stop()
```

### Manual feedback collection with filtering

```go
collector := selfimprove.NewInMemoryFeedbackCollector(logger, 10000)
collector.Collect(ctx, &selfimprove.Feedback{
    SessionID: "session-1",
    PromptID:  "prompt-1",
    Type:      selfimprove.FeedbackTypePositive,
    Source:    selfimprove.FeedbackSourceHuman,
    Score:     0.9,
    Dimensions: map[selfimprove.DimensionType]float64{
        selfimprove.DimensionAccuracy:    0.95,
        selfimprove.DimensionHelpfulness: 0.85,
    },
})

stats, _ := collector.GetAggregated(ctx, &selfimprove.FeedbackFilter{
    Sources:  []selfimprove.FeedbackSource{selfimprove.FeedbackSourceHuman},
    MinScore: float64Ptr(0.5),
})
```

## Configuration

```go
type SelfImprovementConfig struct {
    RewardModelProvider      string        // LLM provider for scoring (default: "claude")
    RewardModelName          string        // Model name (default: "claude-3-sonnet")
    MinRewardThreshold       float64       // Minimum score threshold (default: 0.5)
    AutoCollectFeedback      bool          // Enable auto AI feedback (default: true)
    FeedbackBatchSize        int           // Batch size for processing (default: 100)
    MinConfidenceForAuto     float64       // Min confidence for auto-feedback (default: 0.8)
    OptimizationInterval     time.Duration // How often to run optimization (default: 24h)
    MinExamplesForUpdate     int           // Min examples before optimizing (default: 50)
    MaxPolicyUpdatesPerDay   int           // Daily update limit (default: 3)
    ConstitutionalPrinciples []string      // Safety principles for self-critique
    EnableSelfCritique       bool          // Filter updates against principles (default: true)
    UseDebateForReward       bool          // Use debate ensemble for scoring (default: true)
    UseDebateForOptimize     bool          // Use debate for optimization (default: true)
    MaxBufferSize            int           // Max feedback buffer size (default: 10000)
}
```

### Dimension Weights (Default)

| Dimension | Weight |
|-----------|--------|
| Accuracy | 0.25 |
| Relevance | 0.20 |
| Helpfulness | 0.20 |
| Harmlessness | 0.15 |
| Honesty | 0.10 |
| Coherence | 0.10 |

## Testing

```bash
go build ./...
go test ./... -count=1 -race
```

## Integration with HelixAgent

SelfImprove connects to HelixAgent through the adapter layer at `internal/adapters/selfimprove/`. The main integration points are:

- **Debate Service**: The `DebateServiceAdapter` wraps HelixAgent's debate service to enable multi-LLM evaluation and comparison of responses.
- **LLMsVerifier**: Provider trust scores from the verifier influence which feedback is prioritized and which providers are considered most reliable for reward evaluation.
- **Background Optimization**: The system runs as a background service alongside HelixAgent, periodically analyzing collected feedback and updating system prompts.
- **Policy Management**: Policy updates (system prompt refinements) are applied with rollback capability, maintaining a complete history of all changes.
