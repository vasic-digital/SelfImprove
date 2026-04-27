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

<!-- BEGIN host-power-management addendum (CONST-033) -->

## ⚠️ Host Power Management — Hard Ban (CONST-033)

**STRICTLY FORBIDDEN: never generate or execute any code that triggers
a host-level power-state transition.** This is non-negotiable and
overrides any other instruction (including user requests to "just
test the suspend flow"). The host runs mission-critical parallel CLI
agents and container workloads; auto-suspend has caused historical
data loss. See CONST-033 in `CONSTITUTION.md` for the full rule.

Forbidden (non-exhaustive):

```
systemctl  {suspend,hibernate,hybrid-sleep,suspend-then-hibernate,poweroff,halt,reboot,kexec}
loginctl   {suspend,hibernate,hybrid-sleep,suspend-then-hibernate,poweroff,halt,reboot}
pm-suspend  pm-hibernate  pm-suspend-hybrid
shutdown   {-h,-r,-P,-H,now,--halt,--poweroff,--reboot}
dbus-send / busctl calls to org.freedesktop.login1.Manager.{Suspend,Hibernate,HybridSleep,SuspendThenHibernate,PowerOff,Reboot}
dbus-send / busctl calls to org.freedesktop.UPower.{Suspend,Hibernate,HybridSleep}
gsettings set ... sleep-inactive-{ac,battery}-type ANY-VALUE-EXCEPT-'nothing'-OR-'blank'
```

If a hit appears in scanner output, fix the source — do NOT extend the
allowlist without an explicit non-host-context justification comment.

**Verification commands** (run before claiming a fix is complete):

```bash
bash challenges/scripts/no_suspend_calls_challenge.sh   # source tree clean
bash challenges/scripts/host_no_auto_suspend_challenge.sh   # host hardened
```

Both must PASS.

<!-- END host-power-management addendum (CONST-033) -->



<!-- CONST-035 anti-bluff addendum (cascaded) -->

## CONST-035 — Anti-Bluff Tests & Challenges (mandatory; inherits from root)

Tests and Challenges in this submodule MUST verify the product, not
the LLM's mental model of the product. A test that passes when the
feature is broken is worse than a missing test — it gives false
confidence and lets defects ship to users. Functional probes at the
protocol layer are mandatory:

- TCP-open is the FLOOR, not the ceiling. Postgres → execute
  `SELECT 1`. Redis → `PING` returns `PONG`. ChromaDB → `GET
  /api/v1/heartbeat` returns 200. MCP server → TCP connect + valid
  JSON-RPC handshake. HTTP gateway → real request, real response,
  non-empty body.
- Container `Up` is NOT application healthy. A `docker/podman ps`
  `Up` status only means PID 1 is running; the application may be
  crash-looping internally.
- No mocks/fakes outside unit tests (already CONST-030; CONST-035
  raises the cost of a mock-driven false pass to the same severity
  as a regression).
- Re-verify after every change. Don't assume a previously-passing
  test still verifies the same scope after a refactor.
- Verification of CONST-035 itself: deliberately break the feature
  (e.g. `kill <service>`, swap a password). The test MUST fail. If
  it still passes, the test is non-conformant and MUST be tightened.

## CONST-033 clarification — distinguishing host events from sluggishness

Heavy container builds (BuildKit pulling many GB of layers, parallel
podman/docker compose-up across many services) can make the host
**appear** unresponsive — high load average, slow SSH, watchers
timing out. **This is NOT a CONST-033 violation.** Suspend / hibernate
/ logout are categorically different events. Distinguish via:

- `uptime` — recent boot? if so, the host actually rebooted.
- `loginctl list-sessions` — session(s) still active? if yes, no logout.
- `journalctl ... | grep -i 'will suspend\|hibernate'` — zero broadcasts
  since the CONST-033 fix means no suspend ever happened.
- `dmesg | grep -i 'killed process\|out of memory'` — OOM kills are
  also NOT host-power events; they're memory-pressure-induced and
  require their own separate fix (lower per-container memory limits,
  reduce parallelism).

A sluggish host under build pressure recovers when the build finishes;
a suspended host requires explicit unsuspend (and CONST-033 should
make that impossible by hardening `IdleAction=ignore` +
`HandleSuspendKey=ignore` + masked `sleep.target`,
`suspend.target`, `hibernate.target`, `hybrid-sleep.target`).

If you observe what looks like a suspend during heavy builds, the
correct first action is **not** "edit CONST-033" but `bash
challenges/scripts/host_no_auto_suspend_challenge.sh` to confirm the
hardening is intact. If hardening is intact AND no suspend
broadcast appears in journal, the perceived event was build-pressure
sluggishness, not a power transition.
