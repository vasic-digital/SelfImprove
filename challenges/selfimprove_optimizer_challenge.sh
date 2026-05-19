#!/usr/bin/env bash
#
# challenges/selfimprove_optimizer_challenge.sh
#
# Round-201 deliverable -- SelfImprove submodule deep-doc + test-matrix
# enrichment Challenge.
#
# Drives the full CONST-050(B) "Challenges" leg for the SelfImprove
# submodule, exercising the CONST-046 round-94 i18n migration (8
# messageIDs in optimizer.go):
#
#   Step 1: pre-build  -- go vet + go build
#   Step 2: post-build -- go test ./... -count=1 -race
#   Step 3: bundle load -- assert both fixture YAMLs exist + non-empty
#                          + all 8 migrated message IDs are present in
#                          both locales.
#   Step 4: runtime end-to-end -- run challenges/runner against EN+SR
#                                  fixtures; assert every migrated
#                                  message ID routes through the real
#                                  Translator with sentinel-expected
#                                  output.
#   Step 5: paired anti-bluff mutation -- corrupt one EN bundle entry,
#                                          re-run, expect non-zero
#                                          exit; restore.
#
# Anti-bluff invariants (CONST-035 / Article XI §11.9):
#   - every PASS is preceded by a real command + captured output
#   - the mutation leg PROVES the assertion would fail if SelfImprove
#     regressed (entry-substitution caught by runner's substring
#     assertion)
#   - the script exits non-zero on the FIRST failure (no quiet skips)
#   - the runner consumes the EXACT production rendering path via
#     BuildOptimizationTopicForChallenge -- no parallel re-impl, no
#     "Challenge passes but production differs" bluff
#
# Exit 0 only if every step above succeeded.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVIDENCE_DIR="${SCRIPT_DIR}/.last-run"
mkdir -p "${EVIDENCE_DIR}"

cd "${REPO_ROOT}"

log() { printf '\n=== %s ===\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

# The 8 migrated CONST-046 round-94 message IDs. Hardcoded here is
# the source-of-truth for what MUST be present in each fixture bundle.
MIGRATED_IDS=(
  "selfimprove_optimizer_prompt_analyze_header"
  "selfimprove_optimizer_prompt_current_policy_label"
  "selfimprove_optimizer_prompt_weak_dimensions_label"
  "selfimprove_optimizer_prompt_dimension_bullet"
  "selfimprove_optimizer_prompt_common_issues_label"
  "selfimprove_optimizer_prompt_issue_bullet"
  "selfimprove_optimizer_prompt_sample_responses_label"
  "selfimprove_optimizer_prompt_suggest_improvements_footer"
)

# ---------------------------------------------------------------------------
# Step 1 -- pre-build floor
# ---------------------------------------------------------------------------
log "Step 1: go vet + go build (pre-build floor)"
go vet ./... 2>&1 | tee "${EVIDENCE_DIR}/01-vet.log" || fail "go vet"
go build ./... 2>&1 | tee "${EVIDENCE_DIR}/02-build.log" || fail "go build"

# ---------------------------------------------------------------------------
# Step 2 -- post-build floor: unit suite under race detector
# ---------------------------------------------------------------------------
log "Step 2: go test ./... -count=1 -race (post-build floor)"
go test ./... -count=1 -race 2>&1 | tee "${EVIDENCE_DIR}/03-test.log" || fail "unit suite"

# ---------------------------------------------------------------------------
# Step 3 -- bundle load sanity (both locales, all 8 migrated IDs present)
# ---------------------------------------------------------------------------
log "Step 3: bilingual bundle load sanity (8 migrated message IDs)"
EN_FIX="${SCRIPT_DIR}/fixtures/en.yaml"
SR_FIX="${SCRIPT_DIR}/fixtures/sr-Latn.yaml"
[[ -s "${EN_FIX}" ]] || fail "missing or empty fixture: ${EN_FIX}"
[[ -s "${SR_FIX}" ]] || fail "missing or empty fixture: ${SR_FIX}"

for id in "${MIGRATED_IDS[@]}"; do
  grep -q "^${id}:" "${EN_FIX}" || fail "en fixture missing migrated id: ${id}"
  grep -q "^${id}:" "${SR_FIX}" || fail "sr-Latn fixture missing migrated id: ${id}"
done
printf 'fixtures OK: %s + %s (all 8 migrated IDs present in both)\n' "${EN_FIX}" "${SR_FIX}" \
  | tee "${EVIDENCE_DIR}/04-fixtures.log"

# ---------------------------------------------------------------------------
# Step 4 -- runtime end-to-end: real Translator wired into the
# production rendering path via LLMPolicyOptimizer
# .BuildOptimizationTopicForChallenge
# ---------------------------------------------------------------------------
log "Step 4: runtime end-to-end (EN+SR Optimizer prompt round-trip)"
go run ./challenges/runner 2>&1 | tee "${EVIDENCE_DIR}/05-runtime.log" || fail "runtime round-trip"

# ---------------------------------------------------------------------------
# Step 5 -- paired anti-bluff mutation
#
# We corrupt the EN bundle's analyze_header entry and assert the runner
# FAILS. If the runner still PASSES after corruption, the substring
# assertion is not actually checking the rendered output and the suite
# is a bluff (CONST-035).
# ---------------------------------------------------------------------------
log "Step 5: paired anti-bluff mutation (corrupt EN bundle entry, expect runner FAIL)"
BACKUP="${EN_FIX}.bak.$$"
cp "${EN_FIX}" "${BACKUP}"
trap 'mv -f "${BACKUP}" "${EN_FIX}" 2>/dev/null || true' EXIT

# Replace the EN "analyze_header" payload with a wrong value -- the
# runner's substring assertion MUST notice that the expected English
# marker no longer appears in the rendered output. Pure fixture
# mutation, no source-code change, reverted by the EXIT trap.
sed -i 's/Analyze these feedback patterns and suggest system prompt improvements/MUTATION_OF_ANALYZE_HEADER/' "${EN_FIX}"
grep -q 'MUTATION_OF_ANALYZE_HEADER' "${EN_FIX}" || fail "mutation did not apply"

set +e
go run ./challenges/runner > "${EVIDENCE_DIR}/06-mutation.log" 2>&1
MUTATION_RC=$?
set -e

if [[ ${MUTATION_RC} -eq 0 ]]; then
  fail "paired-mutation leg: runner exited 0 with corrupted EN bundle -- assertions are not real (CONST-035 bluff)"
fi
printf 'mutation correctly rejected with exit code %d\n' "${MUTATION_RC}" \
  | tee -a "${EVIDENCE_DIR}/06-mutation.log"

# Restore explicitly (also restored by EXIT trap as belt-and-braces).
mv -f "${BACKUP}" "${EN_FIX}"
trap - EXIT

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "PASS: selfimprove_optimizer_challenge.sh -- all 5 steps green"
printf 'evidence directory: %s\n' "${EVIDENCE_DIR}"
ls -la "${EVIDENCE_DIR}"
exit 0
