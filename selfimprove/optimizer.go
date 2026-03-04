package selfimprove

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
)

// LLMPolicyOptimizer implements PolicyOptimizer using LLM-based analysis
type LLMPolicyOptimizer struct {
	provider      LLMProvider
	debateService DebateService
	config        *SelfImprovementConfig
	logger        *logrus.Logger
	history       []*PolicyUpdate
	historyMu     sync.RWMutex
	currentPolicy string
	policyMu      sync.RWMutex
	appliedToday  int
	lastApplyDate time.Time
}

// NewLLMPolicyOptimizer creates a new LLM-based policy optimizer
func NewLLMPolicyOptimizer(provider LLMProvider, debateService DebateService, config *SelfImprovementConfig, logger *logrus.Logger) *LLMPolicyOptimizer {
	if config == nil {
		config = DefaultSelfImprovementConfig()
	}
	if logger == nil {
		logger = logrus.New()
	}
	return &LLMPolicyOptimizer{
		provider:      provider,
		debateService: debateService,
		config:        config,
		logger:        logger,
		history:       make([]*PolicyUpdate, 0),
	}
}

// SetCurrentPolicy sets the current system prompt/policy
func (opt *LLMPolicyOptimizer) SetCurrentPolicy(policy string) {
	opt.policyMu.Lock()
	defer opt.policyMu.Unlock()
	opt.currentPolicy = policy
}

// GetCurrentPolicy returns the current policy
func (opt *LLMPolicyOptimizer) GetCurrentPolicy() string {
	opt.policyMu.RLock()
	defer opt.policyMu.RUnlock()
	return opt.currentPolicy
}

// Optimize generates policy updates from feedback
func (opt *LLMPolicyOptimizer) Optimize(ctx context.Context, examples []*TrainingExample) ([]*PolicyUpdate, error) {
	if len(examples) < opt.config.MinExamplesForUpdate {
		return nil, fmt.Errorf("insufficient examples: need %d, have %d",
			opt.config.MinExamplesForUpdate, len(examples))
	}

	// Analyze feedback patterns
	patterns := opt.analyzeFeedbackPatterns(examples)

	// Generate improvement suggestions
	var updates []*PolicyUpdate

	// Use debate for optimization if enabled
	if opt.config.UseDebateForOptimize && opt.debateService != nil {
		suggestions, err := opt.optimizeWithDebate(ctx, patterns)
		if err != nil {
			opt.logger.WithError(err).Warn("Debate optimization failed, falling back to LLM")
			suggestions, err = opt.optimizeWithLLM(ctx, patterns)
			if err != nil {
				return nil, err
			}
		}
		updates = suggestions
	} else {
		suggestions, err := opt.optimizeWithLLM(ctx, patterns)
		if err != nil {
			return nil, err
		}
		updates = suggestions
	}

	// Apply constitutional principles
	if opt.config.EnableSelfCritique {
		updates = opt.applySelfCritique(ctx, updates)
	}

	return updates, nil
}

// Apply applies a policy update
func (opt *LLMPolicyOptimizer) Apply(ctx context.Context, update *PolicyUpdate) error {
	// Check daily limit
	now := time.Now()
	if now.Format("2006-01-02") != opt.lastApplyDate.Format("2006-01-02") {
		opt.appliedToday = 0
		opt.lastApplyDate = now
	}
	if opt.appliedToday >= opt.config.MaxPolicyUpdatesPerDay {
		return fmt.Errorf("daily policy update limit reached (%d)", opt.config.MaxPolicyUpdatesPerDay)
	}

	// Store old policy for rollback
	opt.policyMu.Lock()
	update.OldPolicy = opt.currentPolicy
	opt.currentPolicy = update.NewPolicy
	opt.policyMu.Unlock()

	// Record application
	appliedAt := time.Now()
	update.AppliedAt = &appliedAt
	opt.appliedToday++

	// Add to history
	opt.historyMu.Lock()
	opt.history = append(opt.history, update)
	opt.historyMu.Unlock()

	opt.logger.WithFields(logrus.Fields{
		"update_id":         update.ID,
		"type":              update.UpdateType,
		"improvement_score": update.ImprovementScore,
	}).Info("Policy update applied")

	return nil
}

// Rollback reverts a policy update
func (opt *LLMPolicyOptimizer) Rollback(ctx context.Context, updateID string) error {
	opt.historyMu.RLock()
	var update *PolicyUpdate
	for _, u := range opt.history {
		if u.ID == updateID {
			update = u
			break
		}
	}
	opt.historyMu.RUnlock()

	if update == nil {
		return fmt.Errorf("update not found: %s", updateID)
	}

	if update.OldPolicy == "" {
		return fmt.Errorf("no old policy to rollback to")
	}

	opt.policyMu.Lock()
	opt.currentPolicy = update.OldPolicy
	opt.policyMu.Unlock()

	opt.logger.WithField("update_id", updateID).Info("Policy update rolled back")

	return nil
}

// GetHistory returns policy update history
func (opt *LLMPolicyOptimizer) GetHistory(ctx context.Context, limit int) ([]*PolicyUpdate, error) {
	opt.historyMu.RLock()
	defer opt.historyMu.RUnlock()

	if limit <= 0 || limit > len(opt.history) {
		limit = len(opt.history)
	}

	// Return most recent first
	result := make([]*PolicyUpdate, limit)
	for i := 0; i < limit; i++ {
		result[i] = opt.history[len(opt.history)-1-i]
	}

	return result, nil
}

type feedbackPatterns struct {
	CommonIssues      map[string]int            `json:"common_issues"`
	DimensionWeakness map[DimensionType]float64 `json:"dimension_weakness"`
	ProviderIssues    map[string]float64        `json:"provider_issues"`
	NegativeExamples  []*TrainingExample        `json:"negative_examples"`
	PositiveExamples  []*TrainingExample        `json:"positive_examples"`
	SuggestedFixes    []string                  `json:"suggested_fixes"`
}

func (opt *LLMPolicyOptimizer) analyzeFeedbackPatterns(examples []*TrainingExample) *feedbackPatterns {
	patterns := &feedbackPatterns{
		CommonIssues:      make(map[string]int),
		DimensionWeakness: make(map[DimensionType]float64),
		ProviderIssues:    make(map[string]float64),
		NegativeExamples:  make([]*TrainingExample, 0),
		PositiveExamples:  make([]*TrainingExample, 0),
	}

	dimensionSums := make(map[DimensionType]float64)
	dimensionCounts := make(map[DimensionType]int)
	providerScores := make(map[string][]float64)

	for _, ex := range examples {
		// Categorize by score
		if ex.RewardScore < 0.4 {
			patterns.NegativeExamples = append(patterns.NegativeExamples, ex)
		} else if ex.RewardScore > 0.7 {
			patterns.PositiveExamples = append(patterns.PositiveExamples, ex)
		}

		// Track dimension scores
		for dim, score := range ex.Dimensions {
			dimensionSums[dim] += score
			dimensionCounts[dim]++
		}

		// Track provider scores
		if ex.ProviderName != "" {
			providerScores[ex.ProviderName] = append(providerScores[ex.ProviderName], ex.RewardScore)
		}

		// Extract issues from feedback comments
		for _, f := range ex.Feedback {
			if f.Type == FeedbackTypeNegative && f.Comment != "" {
				patterns.CommonIssues[f.Comment]++
			}
		}
	}

	// Calculate dimension weaknesses (below 0.5 average)
	for dim, sum := range dimensionSums {
		if count := dimensionCounts[dim]; count > 0 {
			avg := sum / float64(count)
			if avg < 0.5 {
				patterns.DimensionWeakness[dim] = avg
			}
		}
	}

	// Calculate provider issues (below average)
	var totalAvg float64
	var totalCount int
	for _, scores := range providerScores {
		for _, s := range scores {
			totalAvg += s
			totalCount++
		}
	}
	if totalCount > 0 {
		overallAvg := totalAvg / float64(totalCount)
		for provider, scores := range providerScores {
			var sum float64
			for _, s := range scores {
				sum += s
			}
			avg := sum / float64(len(scores))
			if avg < overallAvg-0.1 {
				patterns.ProviderIssues[provider] = avg
			}
		}
	}

	return patterns
}

func (opt *LLMPolicyOptimizer) optimizeWithDebate(ctx context.Context, patterns *feedbackPatterns) ([]*PolicyUpdate, error) {
	topic := opt.buildOptimizationTopic(patterns)

	result, err := opt.debateService.RunDebate(ctx, topic, nil)
	if err != nil {
		return nil, err
	}

	return opt.parseDebateOptimizations(result, patterns)
}

func (opt *LLMPolicyOptimizer) optimizeWithLLM(ctx context.Context, patterns *feedbackPatterns) ([]*PolicyUpdate, error) {
	if opt.provider == nil {
		return nil, fmt.Errorf("no LLM provider available")
	}

	systemPrompt := `You are an AI system improvement specialist. Analyze feedback patterns and suggest policy improvements.
Output JSON array of improvements:
[{"type": "prompt_refinement|guideline_addition|example_addition|constraint_update|tone_adjustment",
  "change": "specific change to make",
  "reason": "why this helps",
  "improvement_score": 0.X}]`

	prompt := opt.buildOptimizationTopic(patterns)

	response, err := opt.provider.Complete(ctx, prompt, systemPrompt)
	if err != nil {
		return nil, err
	}

	return opt.parseLLMOptimizations(response, patterns)
}

func (opt *LLMPolicyOptimizer) buildOptimizationTopic(patterns *feedbackPatterns) string {
	var sb strings.Builder
	sb.WriteString("Analyze these feedback patterns and suggest system prompt improvements:\n\n")

	opt.policyMu.RLock()
	currentPolicy := opt.currentPolicy
	opt.policyMu.RUnlock()

	if currentPolicy != "" {
		sb.WriteString("Current System Prompt:\n")
		sb.WriteString(currentPolicy)
		sb.WriteString("\n\n")
	}

	if len(patterns.DimensionWeakness) > 0 {
		sb.WriteString("Weak Dimensions:\n")
		for dim, score := range patterns.DimensionWeakness {
			sb.WriteString(fmt.Sprintf("- %s: %.2f\n", dim, score))
		}
		sb.WriteString("\n")
	}

	if len(patterns.CommonIssues) > 0 {
		sb.WriteString("Common Issues:\n")
		for issue, count := range patterns.CommonIssues {
			if count >= 2 {
				sb.WriteString(fmt.Sprintf("- %s (count: %d)\n", issue, count))
			}
		}
		sb.WriteString("\n")
	}

	// Include sample negative examples
	if len(patterns.NegativeExamples) > 0 {
		sb.WriteString("Sample Low-Score Responses:\n")
		for i, ex := range patterns.NegativeExamples {
			if i >= 3 {
				break
			}
			sb.WriteString(fmt.Sprintf("- Score: %.2f, Prompt: %s\n",
				ex.RewardScore, truncateStr(ex.Prompt, 100)))
		}
		sb.WriteString("\n")
	}

	sb.WriteString("Suggest specific improvements to the system prompt that would address these issues.")

	return sb.String()
}

func (opt *LLMPolicyOptimizer) parseDebateOptimizations(result *DebateResult, patterns *feedbackPatterns) ([]*PolicyUpdate, error) {
	// Try to extract JSON from consensus
	return opt.extractOptimizations(result.Consensus, patterns)
}

func (opt *LLMPolicyOptimizer) parseLLMOptimizations(response string, patterns *feedbackPatterns) ([]*PolicyUpdate, error) {
	return opt.extractOptimizations(response, patterns)
}

func (opt *LLMPolicyOptimizer) extractOptimizations(text string, patterns *feedbackPatterns) ([]*PolicyUpdate, error) {
	// Find JSON array in text
	start := strings.Index(text, "[")
	end := strings.LastIndex(text, "]")
	if start < 0 || end <= start {
		// Try single object
		start = strings.Index(text, "{")
		end = strings.LastIndex(text, "}")
		if start < 0 || end <= start {
			return nil, fmt.Errorf("no JSON found in optimization response")
		}
		text = "[" + text[start:end+1] + "]"
		start = 0
		end = len(text) - 1
	}

	jsonStr := text[start : end+1]

	var parsed []struct {
		Type             string  `json:"type"`
		Change           string  `json:"change"`
		Reason           string  `json:"reason"`
		ImprovementScore float64 `json:"improvement_score"`
	}

	if err := json.Unmarshal([]byte(jsonStr), &parsed); err != nil {
		return nil, fmt.Errorf("failed to parse optimizations: %w", err)
	}

	updates := make([]*PolicyUpdate, 0, len(parsed))
	opt.policyMu.RLock()
	currentPolicy := opt.currentPolicy
	opt.policyMu.RUnlock()

	for _, p := range parsed {
		updateType := PolicyUpdatePromptRefinement
		switch p.Type {
		case "guideline_addition":
			updateType = PolicyUpdateGuidelineAddition
		case "example_addition":
			updateType = PolicyUpdateExampleAddition
		case "constraint_update":
			updateType = PolicyUpdateConstraintUpdate
		case "tone_adjustment":
			updateType = PolicyUpdateToneAdjustment
		}

		// Generate new policy by applying change
		newPolicy := opt.applyChange(currentPolicy, p.Change, updateType)

		update := &PolicyUpdate{
			ID:               uuid.New().String(),
			OldPolicy:        currentPolicy,
			NewPolicy:        newPolicy,
			UpdateType:       updateType,
			Reason:           p.Reason,
			ImprovementScore: p.ImprovementScore,
			Examples:         patterns.NegativeExamples[:min(3, len(patterns.NegativeExamples))],
			CreatedAt:        time.Now(),
		}

		updates = append(updates, update)
	}

	// Sort by improvement score
	sort.Slice(updates, func(i, j int) bool {
		return updates[i].ImprovementScore > updates[j].ImprovementScore
	})

	return updates, nil
}

func (opt *LLMPolicyOptimizer) applyChange(policy, change string, updateType PolicyUpdateType) string {
	if policy == "" {
		return change
	}

	switch updateType {
	case PolicyUpdateGuidelineAddition:
		return policy + "\n\nAdditional Guideline:\n" + change
	case PolicyUpdateExampleAddition:
		return policy + "\n\nExample:\n" + change
	case PolicyUpdateConstraintUpdate:
		return policy + "\n\nConstraint:\n" + change
	case PolicyUpdateToneAdjustment:
		return policy + "\n\nTone Note:\n" + change
	default:
		// For refinement, append as modification note
		return policy + "\n\nRefinement:\n" + change
	}
}

func (opt *LLMPolicyOptimizer) applySelfCritique(ctx context.Context, updates []*PolicyUpdate) []*PolicyUpdate {
	if len(opt.config.ConstitutionalPrinciples) == 0 {
		return updates
	}

	// Filter updates that violate constitutional principles
	filtered := make([]*PolicyUpdate, 0, len(updates))
	for _, update := range updates {
		violates := false
		for _, principle := range opt.config.ConstitutionalPrinciples {
			if opt.violatesPrinciple(update.NewPolicy, principle) {
				opt.logger.WithFields(logrus.Fields{
					"update_id": update.ID,
					"principle": principle,
				}).Warn("Update violates constitutional principle, skipping")
				violates = true
				break
			}
		}
		if !violates {
			filtered = append(filtered, update)
		}
	}

	return filtered
}

func (opt *LLMPolicyOptimizer) violatesPrinciple(policy, principle string) bool {
	// Simple heuristic checks
	policyLower := strings.ToLower(policy)
	principleLower := strings.ToLower(principle)

	// Check for contradictions
	if strings.Contains(principleLower, "harmless") {
		harmfulTerms := []string{"ignore safety", "bypass", "override restrictions", "harmful"}
		for _, term := range harmfulTerms {
			if strings.Contains(policyLower, term) {
				return true
			}
		}
	}

	if strings.Contains(principleLower, "honest") {
		dishonestTerms := []string{"pretend", "lie", "deceive", "hide the truth"}
		for _, term := range dishonestTerms {
			if strings.Contains(policyLower, term) {
				return true
			}
		}
	}

	return false
}

func truncateStr(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// SetDebateService sets the debate service for optimization
func (opt *LLMPolicyOptimizer) SetDebateService(service DebateService) {
	opt.debateService = service
}

// SetProvider sets the LLM provider
func (opt *LLMPolicyOptimizer) SetProvider(provider LLMProvider) {
	opt.provider = provider
}
