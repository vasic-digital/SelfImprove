package selfimprove

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
)

// SelfImprovementSystem is the main orchestrator for RLAIF
type SelfImprovementSystem struct {
	rewardModel       RewardModel
	feedbackCollector FeedbackCollector
	policyOptimizer   PolicyOptimizer
	config            *SelfImprovementConfig
	logger            *logrus.Logger

	// Integration points
	debateAdapter *DebateServiceAdapter
	verifier      ProviderVerifier

	// Background optimization
	optimizationTicker *time.Ticker
	stopCh             chan struct{}
	running            bool
	mu                 sync.Mutex
}

// DebateServiceAdapter adapts the debate service for self-improvement
type DebateServiceAdapter struct {
	service     DebateService
	providerMap map[string]string // Maps debate participant names to provider names
	logger      *logrus.Logger
}

// NewDebateServiceAdapter creates a new adapter
func NewDebateServiceAdapter(service DebateService, logger *logrus.Logger) *DebateServiceAdapter {
	if logger == nil {
		logger = logrus.New()
	}
	return &DebateServiceAdapter{
		service:     service,
		providerMap: make(map[string]string),
		logger:      logger,
	}
}

// SetProviderMapping maps debate participant names to provider names
func (a *DebateServiceAdapter) SetProviderMapping(mapping map[string]string) {
	a.providerMap = mapping
}

// EvaluateWithDebate implements DebateRewardEvaluator
func (a *DebateServiceAdapter) EvaluateWithDebate(ctx context.Context, prompt, response string) (*DebateEvaluation, error) {
	topic := fmt.Sprintf(`Evaluate this AI response quality (0.0-1.0):
Prompt: %s
Response: %s
Rate: accuracy, helpfulness, safety, clarity. Return JSON: {"score": X, "dimensions": {...}}`, prompt, response)

	result, err := a.service.RunDebate(ctx, topic, nil)
	if err != nil {
		return nil, err
	}

	// Parse dimensions from consensus
	eval := &DebateEvaluation{
		Score:            result.Confidence,
		DebateID:         result.ID,
		Consensus:        result.Consensus,
		ParticipantVotes: result.Votes,
		Confidence:       result.Confidence,
		Dimensions:       make(map[DimensionType]float64),
	}

	// Use votes as dimension proxies
	for participant, vote := range result.Votes {
		if _, ok := a.providerMap[participant]; ok {
			// Could map to specific dimensions based on provider strengths
			eval.Dimensions[DimensionAccuracy] = (eval.Dimensions[DimensionAccuracy] + vote) / 2
		} else {
			eval.Dimensions[DimensionHelpfulness] = (eval.Dimensions[DimensionHelpfulness] + vote) / 2
		}
	}

	return eval, nil
}

// CompareWithDebate implements DebateRewardEvaluator
func (a *DebateServiceAdapter) CompareWithDebate(ctx context.Context, prompt, response1, response2 string) (*DebateComparison, error) {
	topic := fmt.Sprintf(`Compare these responses. Which is better?
Prompt: %s
A: %s
B: %s
Return JSON: {"preferred": "A" or "B", "margin": 0-1, "reasoning": "..."}`, prompt, response1, response2)

	result, err := a.service.RunDebate(ctx, topic, nil)
	if err != nil {
		return nil, err
	}

	comparison := &DebateComparison{
		PreferredIndex:   0, // Default to A
		DebateID:         result.ID,
		Reasoning:        result.Consensus,
		ParticipantPrefs: make(map[string]int),
		Confidence:       result.Confidence,
	}

	// Parse preference from consensus
	if containsIgnoreCase(result.Consensus, "\"B\"") || containsIgnoreCase(result.Consensus, "Response B") {
		comparison.PreferredIndex = 1
	}

	// Count participant preferences
	for participant, response := range result.Participants {
		if containsIgnoreCase(response, "A") {
			comparison.ParticipantPrefs[participant] = 0
		} else {
			comparison.ParticipantPrefs[participant] = 1
		}
	}

	// Calculate margin from vote spread
	aCount, bCount := 0, 0
	for _, pref := range comparison.ParticipantPrefs {
		if pref == 0 {
			aCount++
		} else {
			bCount++
		}
	}
	total := aCount + bCount
	if total > 0 {
		comparison.Margin = float64(abs(aCount-bCount)) / float64(total)
	}

	return comparison, nil
}

// NewSelfImprovementSystem creates the main self-improvement system
func NewSelfImprovementSystem(config *SelfImprovementConfig, logger *logrus.Logger) *SelfImprovementSystem {
	if config == nil {
		config = DefaultSelfImprovementConfig()
	}
	if logger == nil {
		logger = logrus.New()
	}

	return &SelfImprovementSystem{
		config: config,
		logger: logger,
		stopCh: make(chan struct{}),
	}
}

// Initialize sets up all components
func (sis *SelfImprovementSystem) Initialize(provider LLMProvider, debateService DebateService) error {
	sis.mu.Lock()
	defer sis.mu.Unlock()

	// Create debate adapter
	if debateService != nil {
		sis.debateAdapter = NewDebateServiceAdapter(debateService, sis.logger)
	}

	// Create reward model
	sis.rewardModel = NewAIRewardModel(provider, debateService, sis.config, sis.logger)

	// Create feedback collector
	if sis.config.AutoCollectFeedback {
		sis.feedbackCollector = NewAutoFeedbackCollector(sis.rewardModel, sis.config, sis.logger)
	} else {
		sis.feedbackCollector = NewInMemoryFeedbackCollector(sis.logger, sis.config.FeedbackBatchSize*10)
	}

	// Create policy optimizer
	sis.policyOptimizer = NewLLMPolicyOptimizer(provider, debateService, sis.config, sis.logger)

	sis.logger.Info("Self-improvement system initialized")
	return nil
}

// SetVerifier sets the provider verifier for trust-based decisions
func (sis *SelfImprovementSystem) SetVerifier(verifier ProviderVerifier) {
	sis.verifier = verifier
}

// Start begins background optimization
func (sis *SelfImprovementSystem) Start() error {
	sis.mu.Lock()
	defer sis.mu.Unlock()

	if sis.running {
		return fmt.Errorf("already running")
	}

	sis.optimizationTicker = time.NewTicker(sis.config.OptimizationInterval)
	sis.running = true

	go sis.optimizationLoop()

	sis.logger.WithField("interval", sis.config.OptimizationInterval).Info("Self-improvement system started")
	return nil
}

// Stop stops background optimization
func (sis *SelfImprovementSystem) Stop() {
	sis.mu.Lock()
	defer sis.mu.Unlock()

	if !sis.running {
		return
	}

	sis.running = false
	if sis.optimizationTicker != nil {
		sis.optimizationTicker.Stop()
	}
	close(sis.stopCh)

	sis.logger.Info("Self-improvement system stopped")
}

func (sis *SelfImprovementSystem) optimizationLoop() {
	for {
		select {
		case <-sis.stopCh:
			return
		case <-sis.optimizationTicker.C:
			if err := sis.runOptimizationCycle(context.Background()); err != nil {
				sis.logger.WithError(err).Warn("Optimization cycle failed")
			}
		}
	}
}

func (sis *SelfImprovementSystem) runOptimizationCycle(ctx context.Context) error {
	// Export training examples from feedback
	examples, err := sis.feedbackCollector.Export(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to export feedback: %w", err)
	}

	if len(examples) < sis.config.MinExamplesForUpdate {
		sis.logger.WithFields(logrus.Fields{
			"examples": len(examples),
			"required": sis.config.MinExamplesForUpdate,
		}).Debug("Not enough examples for optimization")
		return nil
	}

	// Generate optimizations
	updates, err := sis.policyOptimizer.Optimize(ctx, examples)
	if err != nil {
		return fmt.Errorf("optimization failed: %w", err)
	}

	if len(updates) == 0 {
		sis.logger.Debug("No optimizations suggested")
		return nil
	}

	// Apply best update (with highest improvement score)
	best := updates[0]
	if best.ImprovementScore >= 0.3 { // Only apply if meaningful improvement expected
		if err := sis.policyOptimizer.Apply(ctx, best); err != nil {
			return fmt.Errorf("failed to apply optimization: %w", err)
		}
		sis.logger.WithFields(logrus.Fields{
			"update_id":         best.ID,
			"improvement_score": best.ImprovementScore,
			"reason":            best.Reason,
		}).Info("Policy update applied")
	}

	return nil
}

// CollectFeedback adds feedback to the system
func (sis *SelfImprovementSystem) CollectFeedback(ctx context.Context, feedback *Feedback) error {
	return sis.feedbackCollector.Collect(ctx, feedback)
}

// CollectAutoFeedback automatically evaluates and collects feedback
func (sis *SelfImprovementSystem) CollectAutoFeedback(ctx context.Context, sessionID, promptID, prompt, response, provider, model string) (*Feedback, error) {
	if autoCollector, ok := sis.feedbackCollector.(*AutoFeedbackCollector); ok {
		return autoCollector.CollectAuto(ctx, sessionID, promptID, prompt, response, provider, model)
	}

	// Manual evaluation
	score, err := sis.rewardModel.Score(ctx, prompt, response)
	if err != nil {
		return nil, err
	}

	feedbackType := FeedbackTypeNeutral
	if score >= 0.7 {
		feedbackType = FeedbackTypePositive
	} else if score < 0.4 {
		feedbackType = FeedbackTypeNegative
	}

	feedback := &Feedback{
		ID:           uuid.New().String(),
		SessionID:    sessionID,
		PromptID:     promptID,
		Type:         feedbackType,
		Source:       FeedbackSourceAI,
		Score:        score,
		ProviderName: provider,
		Model:        model,
		CreatedAt:    time.Now(),
	}

	if err := sis.feedbackCollector.Collect(ctx, feedback); err != nil {
		return nil, err
	}

	return feedback, nil
}

// ScoreResponse scores a response using the reward model
func (sis *SelfImprovementSystem) ScoreResponse(ctx context.Context, prompt, response string) (float64, error) {
	return sis.rewardModel.Score(ctx, prompt, response)
}

// CompareResponses compares two responses
func (sis *SelfImprovementSystem) CompareResponses(ctx context.Context, prompt, response1, response2 string) (*PreferencePair, error) {
	return sis.rewardModel.Compare(ctx, prompt, response1, response2)
}

// GetFeedbackStats returns aggregated feedback statistics
func (sis *SelfImprovementSystem) GetFeedbackStats(ctx context.Context, filter *FeedbackFilter) (*AggregatedFeedback, error) {
	return sis.feedbackCollector.GetAggregated(ctx, filter)
}

// GetPolicyHistory returns policy update history
func (sis *SelfImprovementSystem) GetPolicyHistory(ctx context.Context, limit int) ([]*PolicyUpdate, error) {
	return sis.policyOptimizer.GetHistory(ctx, limit)
}

// RollbackPolicy rolls back a policy update
func (sis *SelfImprovementSystem) RollbackPolicy(ctx context.Context, updateID string) error {
	return sis.policyOptimizer.Rollback(ctx, updateID)
}

// GetRewardModel returns the reward model
func (sis *SelfImprovementSystem) GetRewardModel() RewardModel {
	return sis.rewardModel
}

// GetDebateAdapter returns the debate adapter
func (sis *SelfImprovementSystem) GetDebateAdapter() *DebateServiceAdapter {
	return sis.debateAdapter
}

func containsIgnoreCase(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr ||
		len(s) > 0 && len(substr) > 0 &&
			(s[0] == substr[0] || s[0]+32 == substr[0] || s[0] == substr[0]+32))
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
