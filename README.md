# SelfImprove

**Module**: `digital.vasic.selfimprove` · **Status**: i18n-migrated (round 94) + deep-doc + Challenge enriched (round 201) · **Production LOC**: ~7,200 across 5 source files in `selfimprove/` + `pkg/i18n/`

`digital.vasic.selfimprove` -- AI self-improvement through RLHF, reward modelling, feedback integration, policy optimization, and dimension-weighted scoring. Provides an injected `Translator` seam (CONST-046) so consumers receive locale-aware user-facing strings without SelfImprove ever importing a project-specific i18n stack.

---

## Status Banner

- **2026-05-19 (round 201)** -- deep-doc + test-matrix enrichment per operator's broader directive; this README expanded, `docs/test-coverage.md` added, `challenges/selfimprove_optimizer_challenge.sh` added, public `BuildOptimizationTopicForChallenge` seam exposed for production-path-identical Challenge runner.
- **2026-05-17 (round 94)** -- CONST-046 i18n migration landed: 8 hardcoded English literals in `optimizer.go::buildOptimizationTopicCtx` routed through `pkg/i18n.Translator`; `LLMPolicyOptimizer.SetTranslator` + NoopTranslator safety default; bilingual fixture seed (EN + sr-Latn) added at round 201.
- **2026-05-15** -- CONST-047..061 governance cascade.
- **2026-04-30** -- Bedrock + Azure provider integrations, debate-based reward adapter, constitutional self-critique gate.

---

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

## Integration Seams

SelfImprove is **project-not-aware** (CONST-051(B)). Consumers integrate via four seams:

1. **`LLMProvider` injection** -- the consumer's HTTP-backed LLM client. SelfImprove never dials a provider directly; the consumer wires its real client (Bedrock, Azure, Anthropic, OpenAI, Ollama, ...) into `SelfImprovementSystem.Initialize`.
2. **`DebateService` injection (optional)** -- the consumer's multi-LLM debate orchestrator. When wired, SelfImprove uses it for ensemble reward scoring and ensemble optimization. When nil, SelfImprove falls back to single-LLM mode transparently.
3. **`Translator` injection (CONST-046)** -- `LLMPolicyOptimizer.SetTranslator(tr)` lets the consumer route the 8 migrated optimizer-prompt messageIDs through its own i18n stack (HelixCode wires this to `helix_code/internal/i18nadapter`). The default `i18n.NoopTranslator` returns the messageID verbatim -- safe for logs, **untranslated for end users** (production consumers MUST inject a real Translator).
4. **`ProviderVerifier` injection (optional, via `SetVerifier`)** -- trust scores from the consumer's verifier (HelixCode's `LLMsVerifier`) influence which feedback is prioritized.

No reverse coupling: SelfImprove never reaches into a consumer's tree, never imports a project-specific package, never assumes a HelixCode-specific layout. Consumers depend on `digital.vasic.selfimprove`; the converse is forbidden by CONST-051(B).

### CONST-046 round-94 migrated message IDs

The following 8 messageIDs are emitted by `selfimprove/optimizer.go::buildOptimizationTopicCtx` via the wired Translator. Consumers MUST seed their i18n bundles with these keys per locale:

| Message ID                                                  | Triggered when                          | Interpolation placeholders     |
|-------------------------------------------------------------|-----------------------------------------|--------------------------------|
| `selfimprove_optimizer_prompt_analyze_header`               | always emitted (prompt header)          | none                           |
| `selfimprove_optimizer_prompt_current_policy_label`         | when `CurrentPolicy != ""`              | none                           |
| `selfimprove_optimizer_prompt_weak_dimensions_label`        | when `DimensionWeakness` non-empty      | none                           |
| `selfimprove_optimizer_prompt_dimension_bullet`             | per weak dimension                      | `{{.Dimension}}`, `{{.Score}}` |
| `selfimprove_optimizer_prompt_common_issues_label`          | when `CommonIssues` non-empty           | none                           |
| `selfimprove_optimizer_prompt_issue_bullet`                 | per issue with count >= 2               | `{{.Issue}}`, `{{.Count}}`     |
| `selfimprove_optimizer_prompt_sample_responses_label`       | when `NegativeExamples` non-empty       | none                           |
| `selfimprove_optimizer_prompt_suggest_improvements_footer`  | always emitted (prompt footer)          | none                           |

Reference bilingual bundles (EN + sr-Latn) ship under `challenges/fixtures/` and are consumed by the round-201 Challenge for anti-bluff round-trip verification.

---

## Testing

```bash
# Unit + integration + e2e + security + stress + benchmark suites (race detector on)
go test ./... -count=1 -race

# Coverage report
go test ./... -cover -count=1

# Full Challenge (build + unit + bilingual round-trip + anti-bluff mutation)
bash challenges/selfimprove_optimizer_challenge.sh
```

The Challenge exits non-zero if any of:

- `go vet ./...` or `go build ./...` fails
- unit suite fails or skips
- EN+SR bundles fail to load or are missing any of the 8 migrated message IDs
- `BuildOptimizationTopicForChallenge` returns output missing any expected locale-specific marker
- cross-locale sanity detects EN text leaking into the SR-rendered prompt (or vice versa)
- the anti-bluff mutation (corrupt one EN YAML entry) is NOT caught by the round-trip assertion

See [`docs/test-coverage.md`](docs/test-coverage.md) for the full test-type matrix (CONST-050(B) compliance ledger).

---

## Governance

- Anti-bluff prime directive: [`CLAUDE.md`](CLAUDE.md) preamble + Article XI §11.9 anchor.
- CONST-035 (anti-bluff), CONST-046 (no hardcoded content), CONST-050 (test-type coverage), CONST-051 (decoupling), CONST-054 (dependency manifest), CONST-055 (post-pull validation), CONST-061 (pre-force-push merge-first) -- see [`CONSTITUTION.md`](CONSTITUTION.md).
- Canonical-root inheritance (CONST-059): governance text in this submodule is consumer-side; universal rules live in the HelixConstitution submodule.

---

## Integration with HelixAgent

SelfImprove connects to HelixAgent through the adapter layer at `internal/adapters/selfimprove/`. The main integration points are:

- **Debate Service**: The `DebateServiceAdapter` wraps HelixAgent's debate service to enable multi-LLM evaluation and comparison of responses.
- **LLMsVerifier**: Provider trust scores from the verifier influence which feedback is prioritized and which providers are considered most reliable for reward evaluation.
- **Background Optimization**: The system runs as a background service alongside HelixAgent, periodically analyzing collected feedback and updating system prompts.
- **Policy Management**: Policy updates (system prompt refinements) are applied with rollback capability, maintaining a complete history of all changes.

---

## Module Identity

| Field            | Value                                                       |
|------------------|-------------------------------------------------------------|
| Go module        | `digital.vasic.selfimprove`                                 |
| Go version       | 1.24                                                        |
| Direct deps      | `github.com/google/uuid`, `github.com/sirupsen/logrus`, `github.com/stretchr/testify` (test-only), `gopkg.in/yaml.v3` (challenge-only) |
| Upstream remotes | `vasic-digital/SelfImprove` (GitHub + GitLab)               |
| License          | See repository                                              |

