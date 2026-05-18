// Package i18n declares SelfImprove's hardcoded-content abstraction.
//
// Consumers (helix_code or any other project) inject a concrete
// implementation that resolves message IDs to localised strings.
// SelfImprove itself NEVER imports any project-specific i18n package —
// the contract here keeps SelfImprove fully decoupled per CONST-051(B)
// (decoupling / reusability mandate) and unblocks CONST-046 compliance
// for every consumer.
package i18n

import "context"

// Translator is the contract SelfImprove uses for every user-facing
// string. Consumers MUST wire a concrete Translator at construction
// time (via NewLLMPolicyOptimizer / NewSelfImprovementSystem). Calls
// that pass nil receive NoopTranslator (loud message-ID echo, never a
// silent missing-translation).
type Translator interface {
	// T resolves messageID against the active locale. templateData
	// supplies named placeholders for go-i18n style interpolation; pass
	// nil when the message has no placeholders.
	T(ctx context.Context, messageID string, templateData map[string]any) (string, error)

	// TPlural resolves messageID with plural-form selection driven by
	// count. templateData carries any non-count placeholders.
	TPlural(ctx context.Context, messageID string, count int, templateData map[string]any) (string, error)
}

// NoopTranslator returns the messageID verbatim. SAFETY default for
// unit tests within this submodule + backward-compat for consumers who
// have not wired a Translator yet. Production paths MUST inject a real
// Translator (helix_code wires *i18nadapter.Translator at boot).
type NoopTranslator struct{}

// T returns id unchanged (loud echo). Never returns an error.
func (NoopTranslator) T(_ context.Context, id string, _ map[string]any) (string, error) {
	return id, nil
}

// TPlural returns id unchanged (loud echo). Never returns an error.
func (NoopTranslator) TPlural(_ context.Context, id string, _ int, _ map[string]any) (string, error) {
	return id, nil
}
