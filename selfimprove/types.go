package selfimprove

import (
	"context"
	"time"
)

// FeedbackType represents types of feedback
type FeedbackType string

const (
	FeedbackTypePositive   FeedbackType = "positive"
	FeedbackTypeNegative   FeedbackType = "negative"
	FeedbackTypeNeutral    FeedbackType = "neutral"
	FeedbackTypeSuggestion FeedbackType = "suggestion"
	FeedbackTypeCorrection FeedbackType = "correction"
)

// FeedbackSource represents where feedback came from
type FeedbackSource string

const (
	FeedbackSourceHuman    FeedbackSource = "human"
	FeedbackSourceAI       FeedbackSource = "ai"
	FeedbackSourceDebate   FeedbackSource = "debate"
	FeedbackSourceVerifier FeedbackSource = "verifier"
	FeedbackSourceMetric   FeedbackSource = "metric"
)

// DimensionType represents evaluation dimensions
type DimensionType string

const (
	DimensionAccuracy    DimensionType = "accuracy"
	DimensionRelevance   DimensionType = "relevance"
	DimensionHelpfulness DimensionType = "helpfulness"
	DimensionHarmless    DimensionType = "harmlessness"
	DimensionHonest      DimensionType = "honesty"
	DimensionCoherence   DimensionType = "coherence"
	DimensionCreativity  DimensionType = "creativity"
	DimensionFormatting  DimensionType = "formatting"
)

// Feedback represents feedback on a model response
type Feedback struct {
	ID           string                    `json:"id"`
	SessionID    string                    `json:"session_id"`
	PromptID     string                    `json:"prompt_id"`
	ResponseID   string                    `json:"response_id"`
	Type         FeedbackType              `json:"type"`
	Source       FeedbackSource            `json:"source"`
	Score        float64                   `json:"score"` // -1.0 to 1.0
	Dimensions   map[DimensionType]float64 `json:"dimensions,omitempty"`
	Comment      string                    `json:"comment,omitempty"`
	Correction   string                    `json:"correction,omitempty"` // Corrected response if applicable
	ProviderName string                    `json:"provider_name,omitempty"`
	Model        string                    `json:"model,omitempty"`
	Metadata     map[string]interface{}    `json:"metadata,omitempty"`
	CreatedAt    time.Time                 `json:"created_at"`
}

// TrainingExample represents a training example for improvement
type TrainingExample struct {
	ID                string                    `json:"id"`
	Prompt            string                    `json:"prompt"`
	Response          string                    `json:"response"`
	PreferredResponse string                    `json:"preferred_response,omitempty"`
	RejectedResponse  string                    `json:"rejected_response,omitempty"`
	Feedback          []*Feedback               `json:"feedback"`
	RewardScore       float64                   `json:"reward_score"`
	Dimensions        map[DimensionType]float64 `json:"dimensions"`
	SystemPrompt      string                    `json:"system_prompt,omitempty"`
	ProviderName      string                    `json:"provider_name"`
	Model             string                    `json:"model"`
	Metadata          map[string]interface{}    `json:"metadata,omitempty"`
	CreatedAt         time.Time                 `json:"created_at"`
}

// PreferencePair represents a preference comparison (for DPO/RLAIF)
type PreferencePair struct {
	ID            string                 `json:"id"`
	Prompt        string                 `json:"prompt"`
	Chosen        string                 `json:"chosen"`   // Preferred response
	Rejected      string                 `json:"rejected"` // Less preferred response
	ChosenScore   float64                `json:"chosen_score"`
	RejectedScore float64                `json:"rejected_score"`
	Margin        float64                `json:"margin"` // Confidence margin
	Source        FeedbackSource         `json:"source"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt     time.Time              `json:"created_at"`
}

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

// FeedbackFilter for querying feedback
type FeedbackFilter struct {
	SessionIDs    []string         `json:"session_ids,omitempty"`
	PromptIDs     []string         `json:"prompt_ids,omitempty"`
	Types         []FeedbackType   `json:"types,omitempty"`
	Sources       []FeedbackSource `json:"sources,omitempty"`
	MinScore      *float64         `json:"min_score,omitempty"`
	MaxScore      *float64         `json:"max_score,omitempty"`
	ProviderNames []string         `json:"provider_names,omitempty"`
	Models        []string         `json:"models,omitempty"`
	StartTime     *time.Time       `json:"start_time,omitempty"`
	EndTime       *time.Time       `json:"end_time,omitempty"`
	Limit         int              `json:"limit,omitempty"`
	Offset        int              `json:"offset,omitempty"`
}

// AggregatedFeedback represents aggregated feedback stats
type AggregatedFeedback struct {
	TotalCount         int                               `json:"total_count"`
	AverageScore       float64                           `json:"average_score"`
	ScoreDistribution  map[string]int                    `json:"score_distribution"`
	TypeDistribution   map[FeedbackType]int              `json:"type_distribution"`
	SourceDistribution map[FeedbackSource]int            `json:"source_distribution"`
	DimensionAverages  map[DimensionType]float64         `json:"dimension_averages"`
	ProviderStats      map[string]*ProviderFeedbackStats `json:"provider_stats"`
	TrendData          []*TrendPoint                     `json:"trend_data,omitempty"`
}

// ProviderFeedbackStats represents feedback stats for a provider
type ProviderFeedbackStats struct {
	ProviderName string                    `json:"provider_name"`
	TotalCount   int                       `json:"total_count"`
	AverageScore float64                   `json:"average_score"`
	Dimensions   map[DimensionType]float64 `json:"dimensions"`
}

// TrendPoint represents a point in time for trend analysis
type TrendPoint struct {
	Timestamp    time.Time `json:"timestamp"`
	AverageScore float64   `json:"average_score"`
	Count        int       `json:"count"`
}

// PolicyUpdate represents a policy/prompt update based on feedback
type PolicyUpdate struct {
	ID               string                 `json:"id"`
	OldPolicy        string                 `json:"old_policy"`
	NewPolicy        string                 `json:"new_policy"`
	UpdateType       PolicyUpdateType       `json:"update_type"`
	Change           string                 `json:"change"`
	Reason           string                 `json:"reason"`
	ImprovementScore float64                `json:"improvement_score"`
	Examples         []*TrainingExample     `json:"examples"`
	AppliedAt        *time.Time             `json:"applied_at,omitempty"`
	Metadata         map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt        time.Time              `json:"created_at"`
}

// PolicyUpdateType represents types of policy updates
type PolicyUpdateType string

const (
	PolicyUpdatePromptRefinement  PolicyUpdateType = "prompt_refinement"
	PolicyUpdateGuidelineAddition PolicyUpdateType = "guideline_addition"
	PolicyUpdateExampleAddition   PolicyUpdateType = "example_addition"
	PolicyUpdateConstraintUpdate  PolicyUpdateType = "constraint_update"
	PolicyUpdateToneAdjustment    PolicyUpdateType = "tone_adjustment"
)

// PolicyOptimizer optimizes policies based on feedback
type PolicyOptimizer interface {
	Optimize(ctx context.Context, examples []*TrainingExample) ([]*PolicyUpdate, error)
	Apply(ctx context.Context, update *PolicyUpdate) error
	Rollback(ctx context.Context, updateID string) error
	GetHistory(ctx context.Context, limit int) ([]*PolicyUpdate, error)
	GetCurrentPolicy() string
	SetCurrentPolicy(policy string)
}

// SelfImprovementConfig configuration for self-improvement
type SelfImprovementConfig struct {
	RewardModelProvider      string        `json:"reward_model_provider"`
	RewardModelName          string        `json:"reward_model_name"`
	MinRewardThreshold       float64       `json:"min_reward_threshold"`
	AutoCollectFeedback      bool          `json:"auto_collect_feedback"`
	FeedbackBatchSize        int           `json:"feedback_batch_size"`
	MinConfidenceForAuto     float64       `json:"min_confidence_for_auto"`
	OptimizationInterval     time.Duration `json:"optimization_interval"`
	MinExamplesForUpdate     int           `json:"min_examples_for_update"`
	MaxPolicyUpdatesPerDay   int           `json:"max_policy_updates_per_day"`
	ConstitutionalPrinciples []string      `json:"constitutional_principles,omitempty"`
	EnableSelfCritique       bool          `json:"enable_self_critique"`
	UseDebateForReward       bool          `json:"use_debate_for_reward"`
	UseDebateForOptimize     bool          `json:"use_debate_for_optimize"`
	MaxBufferSize            int           `json:"max_buffer_size"`
}

// DefaultConstitutionalPrincipleIDs returns the canonical i18n message
// IDs for the 5 default constitutional principles (CONST-046 round 206
// migration). Consumers wanting localised principle text resolve each
// ID through their wired *i18n.Translator at display time. Returning
// IDs (not literals) keeps this submodule decoupled per CONST-051(B) —
// SelfImprove never reaches into any consumer's locale machinery.
func DefaultConstitutionalPrincipleIDs() []string {
	return []string{
		"selfimprove_config_principle_helpful_harmless_honest",
		"selfimprove_config_principle_avoid_harmful_content",
		"selfimprove_config_principle_respect_privacy",
		"selfimprove_config_principle_acknowledge_uncertainty",
		"selfimprove_config_principle_balanced_perspectives",
	}
}

// defaultConstitutionalPrinciplesEnglish returns the English-literal
// fallback used by DefaultSelfImprovementConfig() for backward compat
// with callers that have not adopted the locale-aware constructor.
// Each entry MUST stay 1:1 aligned (same order, same intent) with
// DefaultConstitutionalPrincipleIDs() — drift between the two breaks
// the round-206 paired-mutation gate.
func defaultConstitutionalPrinciplesEnglish() []string {
	return []string{
		"Be helpful, harmless, and honest",
		"Avoid generating harmful or misleading content",
		"Respect user privacy and confidentiality",
		"Acknowledge uncertainty when appropriate",
		"Provide balanced perspectives on controversial topics",
	}
}

// DefaultSelfImprovementConfig returns default configuration with
// English-literal constitutional principles (backward-compat — pre
// round 206 behaviour preserved). Consumers wanting locale-aware
// principles SHOULD prefer DefaultSelfImprovementConfigIDs(), then
// translate each ID at display time via their wired Translator.
func DefaultSelfImprovementConfig() *SelfImprovementConfig {
	cfg := DefaultSelfImprovementConfigIDs()
	cfg.ConstitutionalPrinciples = defaultConstitutionalPrinciplesEnglish()
	return cfg
}

// DefaultSelfImprovementConfigIDs returns default configuration where
// ConstitutionalPrinciples holds the canonical i18n message IDs (see
// DefaultConstitutionalPrincipleIDs). Consumers resolve each ID at
// render time via their wired *i18n.Translator. This is the
// CONST-046-compliant constructor — production paths SHOULD prefer
// this over DefaultSelfImprovementConfig() so principle text reaches
// non-English users in their own locale.
func DefaultSelfImprovementConfigIDs() *SelfImprovementConfig {
	return &SelfImprovementConfig{
		RewardModelProvider:      "claude",
		RewardModelName:          "claude-3-sonnet",
		MinRewardThreshold:       0.5,
		AutoCollectFeedback:      true,
		FeedbackBatchSize:        100,
		MinConfidenceForAuto:     0.8,
		OptimizationInterval:     24 * time.Hour,
		MinExamplesForUpdate:     50,
		MaxPolicyUpdatesPerDay:   3,
		EnableSelfCritique:       true,
		UseDebateForReward:       true,
		UseDebateForOptimize:     true,
		MaxBufferSize:            10000,
		ConstitutionalPrinciples: DefaultConstitutionalPrincipleIDs(),
	}
}

// DebateResult represents the result of an AI debate
type DebateResult struct {
	ID           string             `json:"id"`
	Consensus    string             `json:"consensus"`
	Confidence   float64            `json:"confidence"`
	Participants map[string]string  `json:"participants"`
	Votes        map[string]float64 `json:"votes"`
}

// DebateService interface for AI debate service
type DebateService interface {
	RunDebate(ctx context.Context, topic string, participants []string) (*DebateResult, error)
}

// DebateEvaluation represents debate-based evaluation result
type DebateEvaluation struct {
	Score            float64                   `json:"score"`
	Dimensions       map[DimensionType]float64 `json:"dimensions"`
	Consensus        string                    `json:"consensus"`
	DebateID         string                    `json:"debate_id"`
	ParticipantVotes map[string]float64        `json:"participant_votes"`
	Confidence       float64                   `json:"confidence"`
}

// DebateRewardEvaluator interface for debate-based reward evaluation
type DebateRewardEvaluator interface {
	EvaluateWithDebate(ctx context.Context, prompt, response string) (*DebateEvaluation, error)
	CompareWithDebate(ctx context.Context, prompt, response1, response2 string) (*DebateComparison, error)
}

// DebateComparison represents debate-based comparison result
type DebateComparison struct {
	PreferredIndex   int            `json:"preferred_index"` // 0 or 1
	Margin           float64        `json:"margin"`
	Reasoning        string         `json:"reasoning"`
	DebateID         string         `json:"debate_id"`
	ParticipantPrefs map[string]int `json:"participant_prefs"` // Who preferred what
	Confidence       float64        `json:"confidence"`
}

// LLMProvider interface for LLM providers
type LLMProvider interface {
	Complete(ctx context.Context, prompt, systemPrompt string) (string, error)
}

// ProviderVerifier interface for provider verification
type ProviderVerifier interface {
	GetProviderScore(name string) float64
	IsProviderHealthy(name string) bool
}
