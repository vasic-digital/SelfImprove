package selfimprove

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
)

// InMemoryFeedbackCollector implements FeedbackCollector with in-memory storage
type InMemoryFeedbackCollector struct {
	feedback    []*Feedback
	bySession   map[string][]*Feedback
	byPrompt    map[string][]*Feedback
	mu          sync.RWMutex
	logger      *logrus.Logger
	maxSize     int
	rewardModel RewardModel
}

// NewInMemoryFeedbackCollector creates a new in-memory feedback collector
func NewInMemoryFeedbackCollector(logger *logrus.Logger, maxSize int) *InMemoryFeedbackCollector {
	if logger == nil {
		logger = logrus.New()
	}
	if maxSize <= 0 {
		maxSize = 10000
	}
	return &InMemoryFeedbackCollector{
		feedback:  make([]*Feedback, 0),
		bySession: make(map[string][]*Feedback),
		byPrompt:  make(map[string][]*Feedback),
		logger:    logger,
		maxSize:   maxSize,
	}
}

// SetRewardModel sets the reward model for auto-scoring
func (fc *InMemoryFeedbackCollector) SetRewardModel(rm RewardModel) {
	fc.rewardModel = rm
}

// Collect adds new feedback
func (fc *InMemoryFeedbackCollector) Collect(ctx context.Context, feedback *Feedback) error {
	if feedback == nil {
		return fmt.Errorf("feedback cannot be nil")
	}

	fc.mu.Lock()
	defer fc.mu.Unlock()

	// Generate ID if not present
	if feedback.ID == "" {
		feedback.ID = uuid.New().String()
	}
	if feedback.CreatedAt.IsZero() {
		feedback.CreatedAt = time.Now()
	}

	// Evict old entries if at capacity
	if len(fc.feedback) >= fc.maxSize {
		fc.evictOldest(fc.maxSize / 10)
	}

	// Add to main list
	fc.feedback = append(fc.feedback, feedback)

	// Index by session
	if feedback.SessionID != "" {
		fc.bySession[feedback.SessionID] = append(fc.bySession[feedback.SessionID], feedback)
	}

	// Index by prompt
	if feedback.PromptID != "" {
		fc.byPrompt[feedback.PromptID] = append(fc.byPrompt[feedback.PromptID], feedback)
	}

	fc.logger.WithFields(logrus.Fields{
		"feedback_id": feedback.ID,
		"type":        feedback.Type,
		"source":      feedback.Source,
		"score":       feedback.Score,
	}).Debug("Feedback collected")

	return nil
}

// GetBySession retrieves feedback for a session
func (fc *InMemoryFeedbackCollector) GetBySession(ctx context.Context, sessionID string) ([]*Feedback, error) {
	fc.mu.RLock()
	defer fc.mu.RUnlock()

	result := fc.bySession[sessionID]
	if result == nil {
		return []*Feedback{}, nil
	}

	// Return copy
	copy := make([]*Feedback, len(result))
	for i, f := range result {
		copy[i] = f
	}
	return copy, nil
}

// GetByPrompt retrieves feedback for a prompt
func (fc *InMemoryFeedbackCollector) GetByPrompt(ctx context.Context, promptID string) ([]*Feedback, error) {
	fc.mu.RLock()
	defer fc.mu.RUnlock()

	result := fc.byPrompt[promptID]
	if result == nil {
		return []*Feedback{}, nil
	}

	copy := make([]*Feedback, len(result))
	for i, f := range result {
		copy[i] = f
	}
	return copy, nil
}

// GetAggregated returns aggregated feedback statistics
func (fc *InMemoryFeedbackCollector) GetAggregated(ctx context.Context, filter *FeedbackFilter) (*AggregatedFeedback, error) {
	fc.mu.RLock()
	defer fc.mu.RUnlock()

	filtered := fc.applyFilter(filter)

	if len(filtered) == 0 {
		return &AggregatedFeedback{
			TotalCount:         0,
			ScoreDistribution:  make(map[string]int),
			TypeDistribution:   make(map[FeedbackType]int),
			SourceDistribution: make(map[FeedbackSource]int),
			DimensionAverages:  make(map[DimensionType]float64),
			ProviderStats:      make(map[string]*ProviderFeedbackStats),
		}, nil
	}

	agg := &AggregatedFeedback{
		TotalCount:         len(filtered),
		ScoreDistribution:  make(map[string]int),
		TypeDistribution:   make(map[FeedbackType]int),
		SourceDistribution: make(map[FeedbackSource]int),
		DimensionAverages:  make(map[DimensionType]float64),
		ProviderStats:      make(map[string]*ProviderFeedbackStats),
	}

	// Calculate aggregates
	var totalScore float64
	dimensionSums := make(map[DimensionType]float64)
	dimensionCounts := make(map[DimensionType]int)
	providerScoreSums := make(map[string]float64)
	providerCounts := make(map[string]int)

	for _, f := range filtered {
		totalScore += f.Score

		// Score distribution buckets
		bucket := fc.scoreBucket(f.Score)
		agg.ScoreDistribution[bucket]++

		// Type distribution
		agg.TypeDistribution[f.Type]++

		// Source distribution
		agg.SourceDistribution[f.Source]++

		// Dimension averages
		for dim, score := range f.Dimensions {
			dimensionSums[dim] += score
			dimensionCounts[dim]++
		}

		// Provider stats
		if f.ProviderName != "" {
			providerScoreSums[f.ProviderName] += f.Score
			providerCounts[f.ProviderName]++
		}
	}

	agg.AverageScore = totalScore / float64(len(filtered))

	// Calculate dimension averages
	for dim, sum := range dimensionSums {
		if count := dimensionCounts[dim]; count > 0 {
			agg.DimensionAverages[dim] = sum / float64(count)
		}
	}

	// Calculate provider stats
	for provider, sum := range providerScoreSums {
		count := providerCounts[provider]
		agg.ProviderStats[provider] = &ProviderFeedbackStats{
			ProviderName: provider,
			TotalCount:   count,
			AverageScore: sum / float64(count),
			Dimensions:   make(map[DimensionType]float64),
		}
	}

	// Add trend data
	agg.TrendData = fc.calculateTrend(filtered)

	return agg, nil
}

// Export exports feedback as training examples
func (fc *InMemoryFeedbackCollector) Export(ctx context.Context, filter *FeedbackFilter) ([]*TrainingExample, error) {
	fc.mu.RLock()
	defer fc.mu.RUnlock()

	filtered := fc.applyFilter(filter)

	// Group by prompt for creating examples
	byPrompt := make(map[string][]*Feedback)
	for _, f := range filtered {
		if f.PromptID != "" {
			byPrompt[f.PromptID] = append(byPrompt[f.PromptID], f)
		}
	}

	examples := make([]*TrainingExample, 0)
	for promptID, feedbacks := range byPrompt {
		if len(feedbacks) == 0 {
			continue
		}

		// Calculate average dimensions
		dimensions := make(map[DimensionType]float64)
		dimCounts := make(map[DimensionType]int)
		var totalScore float64
		for _, f := range feedbacks {
			totalScore += f.Score
			for dim, score := range f.Dimensions {
				dimensions[dim] += score
				dimCounts[dim]++
			}
		}

		for dim, sum := range dimensions {
			if count := dimCounts[dim]; count > 0 {
				dimensions[dim] = sum / float64(count)
			}
		}

		// Find best correction if available
		var preferredResponse string
		for _, f := range feedbacks {
			if f.Correction != "" && f.Score > 0.7 {
				preferredResponse = f.Correction
				break
			}
		}

		example := &TrainingExample{
			ID:                uuid.New().String(),
			Prompt:            promptID, // Will need actual prompt text
			Feedback:          feedbacks,
			RewardScore:       totalScore / float64(len(feedbacks)),
			Dimensions:        dimensions,
			PreferredResponse: preferredResponse,
			CreatedAt:         time.Now(),
		}

		// Get provider/model from first feedback
		if len(feedbacks) > 0 {
			example.ProviderName = feedbacks[0].ProviderName
			example.Model = feedbacks[0].Model
		}

		examples = append(examples, example)
	}

	return examples, nil
}

func (fc *InMemoryFeedbackCollector) applyFilter(filter *FeedbackFilter) []*Feedback {
	if filter == nil {
		return fc.feedback
	}

	var result []*Feedback
	for _, f := range fc.feedback {
		if fc.matchesFilter(f, filter) {
			result = append(result, f)
		}
	}

	// Apply limit/offset
	if filter.Offset > 0 {
		if filter.Offset >= len(result) {
			return []*Feedback{}
		}
		result = result[filter.Offset:]
	}
	if filter.Limit > 0 && filter.Limit < len(result) {
		result = result[:filter.Limit]
	}

	return result
}

func (fc *InMemoryFeedbackCollector) matchesFilter(f *Feedback, filter *FeedbackFilter) bool {
	// Session filter
	if len(filter.SessionIDs) > 0 && !contains(filter.SessionIDs, f.SessionID) {
		return false
	}

	// Prompt filter
	if len(filter.PromptIDs) > 0 && !contains(filter.PromptIDs, f.PromptID) {
		return false
	}

	// Type filter
	if len(filter.Types) > 0 {
		found := false
		for _, t := range filter.Types {
			if f.Type == t {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Source filter
	if len(filter.Sources) > 0 {
		found := false
		for _, s := range filter.Sources {
			if f.Source == s {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Score filters
	if filter.MinScore != nil && f.Score < *filter.MinScore {
		return false
	}
	if filter.MaxScore != nil && f.Score > *filter.MaxScore {
		return false
	}

	// Provider filter
	if len(filter.ProviderNames) > 0 && !contains(filter.ProviderNames, f.ProviderName) {
		return false
	}

	// Model filter
	if len(filter.Models) > 0 && !contains(filter.Models, f.Model) {
		return false
	}

	// Time filters
	if filter.StartTime != nil && f.CreatedAt.Before(*filter.StartTime) {
		return false
	}
	if filter.EndTime != nil && f.CreatedAt.After(*filter.EndTime) {
		return false
	}

	return true
}

func (fc *InMemoryFeedbackCollector) evictOldest(count int) {
	if count <= 0 || len(fc.feedback) == 0 {
		return
	}

	// Sort by time
	sort.Slice(fc.feedback, func(i, j int) bool {
		return fc.feedback[i].CreatedAt.Before(fc.feedback[j].CreatedAt)
	})

	// Remove oldest
	toRemove := fc.feedback[:count]
	fc.feedback = fc.feedback[count:]

	// Clean up indexes
	for _, f := range toRemove {
		fc.removeFromIndex(fc.bySession, f.SessionID, f.ID)
		fc.removeFromIndex(fc.byPrompt, f.PromptID, f.ID)
	}
}

func (fc *InMemoryFeedbackCollector) removeFromIndex(index map[string][]*Feedback, key, id string) {
	if key == "" {
		return
	}
	if list, ok := index[key]; ok {
		for i, f := range list {
			if f.ID == id {
				index[key] = append(list[:i], list[i+1:]...)
				break
			}
		}
		if len(index[key]) == 0 {
			delete(index, key)
		}
	}
}

func (fc *InMemoryFeedbackCollector) scoreBucket(score float64) string {
	switch {
	case score < 0.2:
		return "0.0-0.2"
	case score < 0.4:
		return "0.2-0.4"
	case score < 0.6:
		return "0.4-0.6"
	case score < 0.8:
		return "0.6-0.8"
	default:
		return "0.8-1.0"
	}
}

func (fc *InMemoryFeedbackCollector) calculateTrend(feedback []*Feedback) []*TrendPoint {
	if len(feedback) == 0 {
		return nil
	}

	// Sort by time
	sorted := make([]*Feedback, len(feedback))
	copy(sorted, feedback)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].CreatedAt.Before(sorted[j].CreatedAt)
	})

	// Group by day
	dayGroups := make(map[string][]*Feedback)
	for _, f := range sorted {
		day := f.CreatedAt.Format("2006-01-02")
		dayGroups[day] = append(dayGroups[day], f)
	}

	// Create trend points
	var points []*TrendPoint
	for day, group := range dayGroups {
		var sum float64
		for _, f := range group {
			sum += f.Score
		}
		t, _ := time.Parse("2006-01-02", day)
		points = append(points, &TrendPoint{
			Timestamp:    t,
			AverageScore: sum / float64(len(group)),
			Count:        len(group),
		})
	}

	// Sort by time
	sort.Slice(points, func(i, j int) bool {
		return points[i].Timestamp.Before(points[j].Timestamp)
	})

	return points
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// AutoFeedbackCollector automatically collects feedback using AI evaluation
type AutoFeedbackCollector struct {
	*InMemoryFeedbackCollector
	rewardModel RewardModel
	config      *SelfImprovementConfig
}

// NewAutoFeedbackCollector creates a collector that auto-generates feedback
func NewAutoFeedbackCollector(rewardModel RewardModel, config *SelfImprovementConfig, logger *logrus.Logger) *AutoFeedbackCollector {
	if config == nil {
		config = DefaultSelfImprovementConfig()
	}
	return &AutoFeedbackCollector{
		InMemoryFeedbackCollector: NewInMemoryFeedbackCollector(logger, config.FeedbackBatchSize*10),
		rewardModel:               rewardModel,
		config:                    config,
	}
}

// CollectAuto automatically generates and collects feedback for a response
func (afc *AutoFeedbackCollector) CollectAuto(ctx context.Context, sessionID, promptID, prompt, response, providerName, model string) (*Feedback, error) {
	if afc.rewardModel == nil {
		return nil, fmt.Errorf("no reward model available")
	}

	// Get dimensions from reward model
	dimensions, err := afc.rewardModel.ScoreWithDimensions(ctx, prompt, response)
	if err != nil {
		afc.logger.WithError(err).Warn("Failed to get dimensions, using simple score")
		score, err := afc.rewardModel.Score(ctx, prompt, response)
		if err != nil {
			return nil, fmt.Errorf("reward model scoring failed: %w", err)
		}
		dimensions = map[DimensionType]float64{
			DimensionHelpfulness: score,
		}
	}

	// Calculate overall score
	var sum float64
	for _, s := range dimensions {
		sum += s
	}
	overallScore := sum / float64(len(dimensions))

	// Determine feedback type
	feedbackType := FeedbackTypeNeutral
	if overallScore >= 0.7 {
		feedbackType = FeedbackTypePositive
	} else if overallScore < 0.4 {
		feedbackType = FeedbackTypeNegative
	}

	feedback := &Feedback{
		ID:           uuid.New().String(),
		SessionID:    sessionID,
		PromptID:     promptID,
		Type:         feedbackType,
		Source:       FeedbackSourceAI,
		Score:        overallScore,
		Dimensions:   dimensions,
		ProviderName: providerName,
		Model:        model,
		CreatedAt:    time.Now(),
	}

	// Only collect if confidence meets threshold
	if overallScore >= afc.config.MinConfidenceForAuto || overallScore <= (1-afc.config.MinConfidenceForAuto) {
		if err := afc.Collect(ctx, feedback); err != nil {
			return nil, err
		}
	}

	return feedback, nil
}
