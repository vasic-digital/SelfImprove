package i18n

import (
	"context"
	"testing"
)

// TestNoopTranslator_T_ReturnsID confirms the safety-default Translator
// echoes the messageID unchanged — anti-bluff: silent missing-translation
// is worse than a loud "selfimprove_optimizer_prompt_..." appearing in
// user-facing output, which is impossible to miss in review.
func TestNoopTranslator_T_ReturnsID(t *testing.T) {
	t.Parallel()
	tr := NoopTranslator{}
	got, err := tr.T(context.Background(), "selfimprove_test_id_singular", nil)
	if err != nil {
		t.Fatalf("NoopTranslator.T returned unexpected err: %v", err)
	}
	if got != "selfimprove_test_id_singular" {
		t.Fatalf("NoopTranslator.T = %q, want verbatim id %q", got, "selfimprove_test_id_singular")
	}
}

// TestNoopTranslator_TPlural_ReturnsID mirrors the singular guarantee
// for plural-form calls — count + templateData ignored, id echoed.
func TestNoopTranslator_TPlural_ReturnsID(t *testing.T) {
	t.Parallel()
	tr := NoopTranslator{}
	got, err := tr.TPlural(context.Background(), "selfimprove_test_id_plural", 7, map[string]any{"Count": 7})
	if err != nil {
		t.Fatalf("NoopTranslator.TPlural returned unexpected err: %v", err)
	}
	if got != "selfimprove_test_id_plural" {
		t.Fatalf("NoopTranslator.TPlural = %q, want verbatim id %q", got, "selfimprove_test_id_plural")
	}
}
