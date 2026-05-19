// SelfImprove optimizer-Challenge runner.
//
// This is a CHALLENGE program (not production code) -- it exercises
// the real LLMPolicyOptimizer.BuildOptimizationTopicForChallenge API
// (which delegates to the same buildOptimizationTopicCtx that
// production Optimize() invokes) against a real Translator reading
// real YAML bundles off disk in two locales (en + sr-Latn) and
// asserts the 8 migrated message-IDs (CONST-046 round 94) each route
// through the wired Translator and emit the locale-correct localized
// content.
//
// Anti-bluff posture (CONST-035 / Article XI §11.9 / CONST-050):
//   - no mocks beyond the Translator implementation itself, which IS
//     the "real consumer integration" the Challenge is verifying;
//   - every assertion captures the actual returned string verbatim so
//     a regression cannot disguise itself as "test still passing";
//   - the paired-mutation leg (driven from
//     challenges/selfimprove_optimizer_challenge.sh) corrupts a YAML
//     entry and re-runs this program -- non-zero exit expected.
//   - the runner consumes the EXACT same production rendering path
//     via BuildOptimizationTopicForChallenge -- no parallel
//     re-implementation, eliminating the "Challenge passes but
//     production differs" bluff vector.
//
// Exit codes:
//
//	0 -- all assertions held; bilingual round-trip green.
//	1 -- bundle load failure, prompt regression, or locale-mismatch.
package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"text/template"

	"gopkg.in/yaml.v3"

	"digital.vasic.selfimprove/pkg/i18n"
	"digital.vasic.selfimprove/selfimprove"
)

// bundleTranslator is the Challenge's real Translator implementation.
// It reads a single-locale YAML bundle into memory once and resolves
// message IDs from the flat key->string map. Values may carry Go
// text/template placeholders ({{.Dimension}}, {{.Score}}, etc.) which
// are interpolated against the data map at lookup time.
// Lookup-miss returns the message ID verbatim so a missing-key
// regression is loud, not silent.
type bundleTranslator struct {
	locale  string
	entries map[string]string
}

// Compile-time check the bundleTranslator implements i18n.Translator.
var _ i18n.Translator = (*bundleTranslator)(nil)

func loadBundle(path, locale string) (*bundleTranslator, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read bundle %s: %w", path, err)
	}
	entries := map[string]string{}
	if err := yaml.Unmarshal(raw, &entries); err != nil {
		return nil, fmt.Errorf("parse bundle %s: %w", path, err)
	}
	if len(entries) == 0 {
		return nil, fmt.Errorf("bundle %s parsed but contains no entries", path)
	}
	return &bundleTranslator{locale: locale, entries: entries}, nil
}

func (b *bundleTranslator) render(raw string, data map[string]any) (string, error) {
	if data == nil || !strings.Contains(raw, "{{") {
		return raw, nil
	}
	tmpl, err := template.New("entry").Parse(raw)
	if err != nil {
		return raw, fmt.Errorf("template parse: %w", err)
	}
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return raw, fmt.Errorf("template execute: %w", err)
	}
	return buf.String(), nil
}

func (b *bundleTranslator) T(_ context.Context, id string, data map[string]any) (string, error) {
	v, ok := b.entries[id]
	if !ok {
		return id, fmt.Errorf("translator(%s): missing key %s", b.locale, id)
	}
	out, err := b.render(v, data)
	if err != nil {
		return id, err
	}
	return out, nil
}

func (b *bundleTranslator) TPlural(ctx context.Context, id string, _ int, data map[string]any) (string, error) {
	return b.T(ctx, id, data)
}

// expectedMarkers is the source-of-truth table the Challenge compares
// rendered prompt output against. Hardcoded here ON PURPOSE -- this
// is challenge/test code, not production output (CONST-046 carve-out:
// fixtures + test assertions are not user-facing text).
//
// Each entry is the literal substring the rendered prompt MUST
// contain when the named message ID was routed through the wired
// Translator for the named locale.
func expectedMarkers(locale string) map[string]string {
	table := map[string]map[string]string{
		"en": {
			"selfimprove_optimizer_prompt_analyze_header":             "Analyze these feedback patterns and suggest system prompt improvements",
			"selfimprove_optimizer_prompt_current_policy_label":       "Current System Prompt:",
			"selfimprove_optimizer_prompt_weak_dimensions_label":      "Weak Dimensions:",
			"selfimprove_optimizer_prompt_dimension_bullet":           "- accuracy: 0.30",
			"selfimprove_optimizer_prompt_common_issues_label":        "Common Issues:",
			"selfimprove_optimizer_prompt_issue_bullet":               "- too vague (count: 3)",
			"selfimprove_optimizer_prompt_sample_responses_label":     "Sample Low-Score Responses:",
			"selfimprove_optimizer_prompt_suggest_improvements_footer": "Suggest specific improvements to the system prompt that would address these issues.",
		},
		"sr-Latn": {
			"selfimprove_optimizer_prompt_analyze_header":             "Analiziraj ove obrasce povratnih informacija i predloži poboljšanja sistemskog prompta",
			"selfimprove_optimizer_prompt_current_policy_label":       "Trenutni sistemski prompt:",
			"selfimprove_optimizer_prompt_weak_dimensions_label":      "Slabe dimenzije:",
			"selfimprove_optimizer_prompt_dimension_bullet":           "- accuracy: 0.30",
			"selfimprove_optimizer_prompt_common_issues_label":        "Česti problemi:",
			"selfimprove_optimizer_prompt_issue_bullet":               "- too vague (broj: 3)",
			"selfimprove_optimizer_prompt_sample_responses_label":     "Uzorak odgovora sa niskom ocenom:",
			"selfimprove_optimizer_prompt_suggest_improvements_footer": "Predloži konkretna poboljšanja sistemskog prompta koja bi rešila ove probleme.",
		},
	}
	return table[locale]
}

// fixedInput renders the same shape for both locales so the only
// variable between runs is the wired Translator. Forces all 8
// migrated call sites to fire:
//   - analyze_header (always)
//   - current_policy_label (set via CurrentPolicy)
//   - weak_dimensions_label + dimension_bullet (one weak dim)
//   - common_issues_label + issue_bullet (one issue with count >= 2)
//   - sample_responses_label (one negative sample)
//   - suggest_improvements_footer (always)
func fixedInput() selfimprove.ChallengeRenderInput {
	return selfimprove.ChallengeRenderInput{
		CurrentPolicy:     "You are a helpful assistant.",
		DimensionWeakness: map[string]float64{"accuracy": 0.30},
		CommonIssues:      map[string]int{"too vague": 3},
		NegativeSamples: []selfimprove.ChallengeSample{
			{Prompt: "What is 2+2?", RewardScore: 0.2},
		},
	}
}

func runLocale(ctx context.Context, fixturePath, locale string) error {
	tr, err := loadBundle(fixturePath, locale)
	if err != nil {
		return err
	}
	if !strings.HasSuffix(fixturePath, locale+".yaml") {
		return fmt.Errorf("fixture path %q does not match locale %q", fixturePath, locale)
	}

	fmt.Printf("--- locale: %s (%s)\n", locale, fixturePath)

	opt := selfimprove.NewLLMPolicyOptimizer(nil, nil, nil, nil)
	opt.SetTranslator(tr)

	rendered := opt.BuildOptimizationTopicForChallenge(ctx, fixedInput())

	markers := expectedMarkers(locale)
	if len(markers) == 0 {
		return fmt.Errorf("no expected markers registered for locale %q", locale)
	}

	for id, marker := range markers {
		if !strings.Contains(rendered, marker) {
			return fmt.Errorf("[%s] id=%s: rendered prompt missing expected marker %q\n--- rendered ---\n%s\n--- end ---",
				locale, id, marker, rendered)
		}
		fmt.Printf("  OK  [%s] %s -> contains %q\n", locale, id, marker)

		// Loud-fallback guard: the verbatim message ID MUST NOT leak
		// into the rendered output. If it does, the Translator was
		// not wired or the entry is missing -- a CONST-046 regression
		// equivalent to NoopTranslator passthrough.
		if strings.Contains(rendered, id) {
			return fmt.Errorf("[%s] id=%s: verbatim message id leaked into rendered output (NoopTranslator regression?)\n--- rendered ---\n%s\n--- end ---",
				locale, id, rendered)
		}
	}

	return nil
}

// crossLocaleSanity asserts that the EN and SR rendered prompts
// differ for the locale-sensitive markers -- a sanity check that we
// are not accidentally serving the same language for both locales
// (CONST-046 regression guard). The two structural-marker IDs
// (dimension_bullet, NegativeSamples line) intentionally share form
// across locales and are excluded.
func crossLocaleSanity(ctx context.Context) error {
	root, err := os.Getwd()
	if err != nil {
		return err
	}
	fixturesDir := filepath.Join(root, "challenges", "fixtures")

	enTr, err := loadBundle(filepath.Join(fixturesDir, "en.yaml"), "en")
	if err != nil {
		return err
	}
	srTr, err := loadBundle(filepath.Join(fixturesDir, "sr-Latn.yaml"), "sr-Latn")
	if err != nil {
		return err
	}

	optEN := selfimprove.NewLLMPolicyOptimizer(nil, nil, nil, nil)
	optEN.SetTranslator(enTr)
	enOut := optEN.BuildOptimizationTopicForChallenge(ctx, fixedInput())

	optSR := selfimprove.NewLLMPolicyOptimizer(nil, nil, nil, nil)
	optSR.SetTranslator(srTr)
	srOut := optSR.BuildOptimizationTopicForChallenge(ctx, fixedInput())

	if enOut == srOut {
		return fmt.Errorf("cross-locale sanity: EN and SR rendered outputs are byte-identical -- locale wiring is a no-op")
	}

	// Sample a few locale-sensitive markers and confirm they differ.
	for _, ids := range [][2]string{
		{"selfimprove_optimizer_prompt_analyze_header", "Analyze these feedback patterns and suggest system prompt improvements"},
		{"selfimprove_optimizer_prompt_current_policy_label", "Current System Prompt:"},
		{"selfimprove_optimizer_prompt_weak_dimensions_label", "Weak Dimensions:"},
		{"selfimprove_optimizer_prompt_common_issues_label", "Common Issues:"},
		{"selfimprove_optimizer_prompt_sample_responses_label", "Sample Low-Score Responses:"},
		{"selfimprove_optimizer_prompt_suggest_improvements_footer", "Suggest specific improvements to the system prompt that would address these issues."},
	} {
		id, enLit := ids[0], ids[1]
		if strings.Contains(srOut, enLit) {
			return fmt.Errorf("cross-locale sanity: SR output contains the English literal for %s (%q) -- locale leak", id, enLit)
		}
		if !strings.Contains(enOut, enLit) {
			return fmt.Errorf("cross-locale sanity: EN output missing English literal for %s (%q)", id, enLit)
		}
		fmt.Printf("  OK  cross-locale differs for %s\n", id)
	}

	return nil
}

func main() {
	root, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "FAIL: cwd: %v\n", err)
		os.Exit(1)
	}
	fixturesDir := filepath.Join(root, "challenges", "fixtures")
	enFixture := filepath.Join(fixturesDir, "en.yaml")
	srFixture := filepath.Join(fixturesDir, "sr-Latn.yaml")

	ctx := context.Background()

	if err := runLocale(ctx, enFixture, "en"); err != nil {
		fmt.Fprintf(os.Stderr, "FAIL en: %v\n", err)
		os.Exit(1)
	}
	if err := runLocale(ctx, srFixture, "sr-Latn"); err != nil {
		fmt.Fprintf(os.Stderr, "FAIL sr-Latn: %v\n", err)
		os.Exit(1)
	}
	if err := crossLocaleSanity(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "FAIL cross-locale: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("PASS: selfimprove optimizer-Challenge -- 8 migrated IDs route through real Translator (EN+SR) + cross-locale sanity green")
}
