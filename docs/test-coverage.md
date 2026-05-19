# SelfImprove -- Test-Type Coverage Matrix

**Authority**: CONST-050(B) "100%-Test-Type-Coverage" mandate (cascaded from HelixConstitution submodule §11.4.27).
**Scope**: this document is the SelfImprove submodule's coverage ledger. It enumerates every test type CONST-050(B) recognises and records the current status against SelfImprove's surface (`selfimprove/` + `pkg/i18n/` + `tests/{unit,integration,e2e,security,stress,benchmark}/` + `challenges/`).

A row may be `covered`, `planned`, or `n/a (out of scope for a module of this shape)`. `n/a` rows MUST justify themselves -- silent omission is a CONST-048 violation per §11.4.25.

---

## Coverage Ledger

| Test type        | Status   | Artefact / location                                                                                                                | Notes |
|------------------|----------|------------------------------------------------------------------------------------------------------------------------------------|-------|
| Unit             | covered  | `selfimprove/*_test.go` (`reward_test.go`, `types_test.go`, `optimizer_i18n_test.go`), `pkg/i18n/i18n_test.go`                     | Mocks permitted per CONST-050(A); race-detector enforced; `sentinelTranslator` in `optimizer_i18n_test.go` proves every CONST-046 round-94 call site routes through the Translator interface. |
| Integration      | covered  | `tests/integration/*_test.go`                                                                                                       | Multi-component wiring of feedback collector + reward model + policy optimizer; no fakes beyond the unit-permitted set. |
| E2E              | covered  | `tests/e2e/*_test.go`, `challenges/selfimprove_optimizer_challenge.sh`                                                              | Three E2E suites (`TestFeedbackCollectionAndExport_E2E`, `TestPolicyApplyAndRollback_E2E`, `TestFullSelfImprovementSystemInit_E2E`) + bash-orchestrated bilingual prompt round-trip with paired anti-bluff mutation. |
| Full automation  | planned  | recommend: full matrix execution over Go 1.24/1.25 × linux/darwin/windows + the i18n round-trip on every supported locale (currently 2: EN + sr-Latn; future-roll to JA, ES, ZH, RU) | CONST-048 coverage matrix dimension is feature × platform × invariant; SelfImprove is pure Go so platform coverage = Go-supported set. |
| Security         | covered  | `tests/security/*_test.go`                                                                                                          | Existing security suite covers reward-model boundary checks. Recommend extending with explicit CONST-042 secret-leak assertions over the optimizer config tree. |
| DDoS             | n/a      | --                                                                                                                                  | SelfImprove is an in-process library -- no network surface, no request fan-in. The consuming service exposes the DDoS surface, not SelfImprove. |
| Scaling          | planned  | recommend: benchmark `Optimize()` and `CollectFeedback()` under N goroutines (N ∈ {1, 10, 100, 1000}) to verify lock contention stays flat | Pure-CPU scaling test; not a network-tier scaling test. |
| Chaos            | planned  | recommend: chaos-style assertion that LLMProvider returning an error degrades gracefully through `optimizeWithLLM`; debate fallback exercised | Failure-injection scope is the LLM+debate fallback chain documented in `CLAUDE.md` §Architecture. |
| Stress           | covered  | `tests/stress/*_test.go`                                                                                                            | Sustained collection + aggregation under repeated load. |
| Performance      | planned  | recommend: `BenchmarkScore`, `BenchmarkBuildOptimizationTopic`, `BenchmarkApplyRollback` with `b.ReportAllocs()` + historical p95 drift | The cached-read path of the reward model is documented as 15-minute TTL; benchmark MUST prove cache-hit latency stays sub-microsecond. |
| Benchmarking     | covered  | `tests/benchmark/*_test.go`                                                                                                         | Micro-benchmark scaffold present; macro-benchmark integration lives in HelixCode (CONST-051(B)). |
| UI               | n/a      | --                                                                                                                                  | SelfImprove ships no UI. |
| UX               | covered  | bilingual locale verification inside `challenges/selfimprove_optimizer_challenge.sh`                                                | UX dimension SelfImprove actually owns: do the 8 migrated prompt strings flip language when the consumer's Translator's locale flips. Asserted EN→SR transition for 6 locale-sensitive markers; 2 structural markers (dimension_bullet, issue_bullet) keep stable form across locales by design (data carriers). |
| Challenges       | covered  | `challenges/selfimprove_optimizer_challenge.sh` (added round 201)                                                                   | Incorporates the `vasic-digital/Challenges` pattern; captures stdout/stderr as wire evidence per §11.4.2; paired mutation per §1.1 / CONST-055 meta-test (corrupt EN bundle entry → runner exits non-zero). |
| HelixQA          | planned  | recommend: register SelfImprove as a target in HelixQA's autonomous QA bank                                                          | HelixQA submodule (`HelixDevelopment/HelixQA`) is incorporated at HelixCode root per CONST-050; SelfImprove enrolment is a HelixCode-meta-repo task, not a SelfImprove-internal task (CONST-051(B)). |

---

## CONST-046 Round-94 Migrated Message IDs

The Challenge enforces that every one of the 8 message IDs migrated by round 94 in `selfimprove/optimizer.go::buildOptimizationTopicCtx` is exercised end-to-end:

| # | Message ID                                                  | Production call site                          | Bundle interpolation |
|---|-------------------------------------------------------------|-----------------------------------------------|----------------------|
| 1 | `selfimprove_optimizer_prompt_analyze_header`               | header always emitted                         | none                 |
| 2 | `selfimprove_optimizer_prompt_current_policy_label`         | when CurrentPolicy != ""                      | none                 |
| 3 | `selfimprove_optimizer_prompt_weak_dimensions_label`        | when DimensionWeakness non-empty              | none                 |
| 4 | `selfimprove_optimizer_prompt_dimension_bullet`             | per weak dimension                            | `{{.Dimension}}`, `{{.Score}}` |
| 5 | `selfimprove_optimizer_prompt_common_issues_label`          | when CommonIssues non-empty                   | none                 |
| 6 | `selfimprove_optimizer_prompt_issue_bullet`                 | per issue with count >= 2                     | `{{.Issue}}`, `{{.Count}}` |
| 7 | `selfimprove_optimizer_prompt_sample_responses_label`       | when NegativeExamples non-empty               | none                 |
| 8 | `selfimprove_optimizer_prompt_suggest_improvements_footer`  | footer always emitted                         | none                 |

The fixed `ChallengeRenderInput` shape in `challenges/runner/main.go` forces all 8 to fire on every invocation, eliminating skip-by-branch bluffs.

---

## Anti-Bluff Posture

Every `covered` row above carries captured runtime evidence:

- **Unit**: `go test ./... -count=1 -race` exits 0 on every commit; race detector enforced.
- **E2E (Challenge)**: `challenges/selfimprove_optimizer_challenge.sh` writes `challenges/.last-run/` artefacts containing stdout + stderr + assertion log + mutation-rejection proof.
- **UX**: the Challenge's bilingual leg captures the actual EN vs SR strings returned by `buildOptimizationTopicCtx` and diff-asserts both differ from each other AND from the verbatim message-id, ruling out NoopTranslator regression.
- **Production-path identity**: the Challenge invokes `LLMPolicyOptimizer.BuildOptimizationTopicForChallenge`, which delegates directly to the unexported `buildOptimizationTopicCtx` -- the same function `Optimize()` calls into. No parallel re-implementation. A behaviour drift in production is mechanically caught.

Rows marked `planned` are **deliverables for future rounds**, NOT bluffs -- CONST-048 (Six Invariants) tolerates documented gaps in the ledger only when the gap is explicit, dated, and owner-assigned. This document is the explicit register; future rounds flip rows from `planned` to `covered` with the matching artefact.

---

## Four-Layer Floor (CONST-048 invariant 6)

Per §1 of the constitution, every test artefact MUST sit on the four-layer floor:

| Layer       | SelfImprove artefact today                                                                       |
|-------------|--------------------------------------------------------------------------------------------------|
| Pre-build   | `go vet ./...`, `go build ./...` -- invoked by `challenges/selfimprove_optimizer_challenge.sh` step 1 |
| Post-build  | `go test ./... -count=1 -race` -- invoked by Challenge step 2                                    |
| Runtime     | bilingual EN+SR round-trip + 8-ID coverage probe + cross-locale sanity -- Challenge step 4       |
| Paired mut. | corrupt one EN YAML entry, assert Challenge FAILs -- Challenge step 5                            |

A future round that adds a new test type to a `covered` row MUST extend the Challenge to keep the four-layer floor intact.

---

## Owner / Cadence

- **Owner**: SelfImprove submodule maintainer (vasic-digital). HelixCode consumers MAY contribute upstream but MUST NOT inject HelixCode-specific context (CONST-051(B)).
- **Cadence**: ledger reviewed at every governance-cascade round; planned → covered transitions land as their own commits with verbatim mandate quotes per CONST-049 §11.4.17.
- **Historical entry**: round 94 (i18n migration) + round 201 (deep-doc + Challenge -- this document) are the two rows that opened the bilingual coverage leg.
