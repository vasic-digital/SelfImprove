# CLAUDE.md - SelfImprove Module


## Definition of Done

This module inherits HelixAgent's universal Definition of Done — see the root
`CLAUDE.md` and `docs/development/definition-of-done.md`. In one line: **no
task is done without pasted output from a real run of the real system in the
same session as the change.** Coverage and green suites are not evidence.

### Acceptance demo for this module

```bash
# Feedback collection → reward score → policy update → rollback
cd SelfImprove && GOMAXPROCS=2 nice -n 19 go test -count=1 -race -v \
  -run 'TestFeedbackCollectionAndExport_E2E|TestPolicyApplyAndRollback_E2E|TestFullSelfImprovementSystemInit_E2E' \
  ./tests/e2e/...
```
Expect: three E2E PASS; aggregate stats, apply+rollback preserves history, reward model accessible.


## Overview

`digital.vasic.selfimprove` is the RLHF and policy-optimization layer: collects multi-dimensional human / AI / debate feedback, runs a reward model (LLM-based with optional debate scoring), generates preference pairs for DPO-style training, and periodically refines system prompts via an LLM-driven policy optimizer with rollback support.

**Module:** `digital.vasic.selfimprove` (Go 1.24+, ~7,200 LOC across 6 files).

## Architecture

```
SelfImprovementSystem
    │
    ├── Background loop (every OptimizationInterval, default 24h)
    │     auto-apply threshold: improvement_score ≥ 0.3
    │
    ├── AIRewardModel
    │     • Score(prompt, response) → float64
    │     • ScoreWithDimensions → 8 dimensions:
    │       accuracy, relevance, helpfulness, harmlessness,
    │       honesty, coherence, + 2 others
    │     • Compare(r1, r2) → PreferencePair (for DPO)
    │     • Scores cached 15 minutes (no explicit invalidation)
    │
    ├── FeedbackCollector
    │     • Sources: human | ai | debate | verifier | metric
    │     • AutoFeedbackCollector spawns collector goroutines
    │     • Aggregates stats, exports TrainingExample slices
    │
    ├── LLMPolicyOptimizer
    │     • Optimize(examples) → []*PolicyUpdate
    │     • Apply / Rollback / GetHistory (unbounded history!)
    │     • GetCurrentPolicy / SetCurrentPolicy
    │
    └── DebateServiceAdapter (optional)
          • Wraps debate service; consensus → dimension scores
          • CompareWithDebate: naive — prefers "B" if consensus string contains "B"
```

## Key types and interfaces

```go
type RewardModel interface {
    Score(ctx, prompt, response string) (float64, error)
    ScoreWithDimensions(ctx, prompt, response string) (map[DimensionType]float64, error)
    Compare(ctx, prompt, r1, r2 string) (*PreferencePair, error)
    Train(ctx context.Context, examples []*TrainingExample) error
}

type FeedbackCollector interface {
    Collect, GetBySession, GetByPrompt, GetAggregated, Export
}

type PolicyOptimizer interface {
    Optimize(ctx, examples) ([]*PolicyUpdate, error)
    Apply, Rollback, GetHistory
    GetCurrentPolicy() string
    SetCurrentPolicy(policy string)
}

type Feedback struct {
    Score      float64
    Dimensions map[DimensionType]float64  // 8 dimensions
    Source     FeedbackSource             // human | ai | debate | verifier | metric
}

func (s *SelfImprovementSystem) Initialize(provider LLMProvider, debateService DebateService)
func (s *SelfImprovementSystem) SetVerifier(verifier ProviderVerifier)
```

## Integration Seams

- **Upstream (imports):** none.
- **Downstream (sibling consumer):** root HelixAgent via `internal/handlers/selfimprove_handler_test.go`.
- **Sibling complements:** `LLMProvider` (for the reward-model LLM), `DebateOrchestrator` (for the debate-based reward adapter), `LLMsVerifier` (trust-based filtering), `Benchmark` and `Agentic` (consumers of the reward model for self-evaluation).

## Gotchas

1. **Auto-apply threshold is a magic constant (0.3)** — tune with care; there is no config knob. A lower threshold means more aggressive prompt churn.
2. **Reward-model cache is unbounded TTL** — 15 min entries, no explicit invalidation. Old scores can linger if the eval loop is slower than the cache refresh.
3. **Debate → dimension score mapping is naive** — votes are averaged without domain weighting. Treat debate-sourced scores as a signal, not ground truth.
4. **`AutoFeedbackCollector` blocks on a full channel** — buffer size is configurable; under bursty collection, you must size it for your workload or feedback is dropped (silently unless you log it).
5. **Policy-update history is unbounded** — no retention policy. Long-lived systems will bloat memory.
6. **`CompareWithDebate` is a string match** — prefers "B" if consensus string contains "B" (case-insensitive). Fragile.

## Acceptance demo

```bash
GOMAXPROCS=2 nice -n 19 go test -race -v \
  -run TestFeedbackCollectionAndExport_E2E ./tests/e2e/selfimprove_e2e_test.go -count=1

GOMAXPROCS=2 nice -n 19 go test -race -v \
  -run TestPolicyApplyAndRollback_E2E ./tests/e2e/selfimprove_e2e_test.go -count=1

GOMAXPROCS=2 nice -n 19 go test -race -v \
  -run TestFullSelfImprovementSystemInit_E2E ./tests/e2e/selfimprove_e2e_test.go -count=1

# Expected:
#   PASS: Collect 10 feedback items, aggregate stats (avg score ~0.8)
#   PASS: Create policy update, apply, rollback; history preserved
#   PASS: System initializes, reward model available, optimizer callable
```

A production-path demo (real LLM, real debate service, real prompt rotation with rollback) belongs in the HelixAgent consumer; add it when the handler integration matures.
