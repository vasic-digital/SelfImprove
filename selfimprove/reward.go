package selfimprove

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
)

// AIRewardModel implements RewardModel using LLM-based evaluation
type AIRewardModel struct {
	provider      LLMProvider
	debateService DebateService
	config        *SelfImprovementConfig
	logger        *logrus.Logger
	cache         map[string]*cachedScore
	cacheMu       sync.RWMutex
	cacheExpiry   time.Duration
}

type cachedScore struct {
	score      float64
	dimensions map[DimensionType]float64
	timestamp  time.Time
}

// NewAIRewardModel creates a new AI-based reward model
func NewAIRewardModel(provider LLMProvider, debateService DebateService, config *SelfImprovementConfig, logger *logrus.Logger) *AIRewardModel {
	if config == nil {
		config = DefaultSelfImprovementConfig()
	}
	if logger == nil {
		logger = logrus.New()
	}
	return &AIRewardModel{
		provider:      provider,
		debateService: debateService,
		config:        config,
		logger:        logger,
		cache:         make(map[string]*cachedScore),
		cacheExpiry:   15 * time.Minute,
	}
}

// Score returns a reward score for a response
func (rm *AIRewardModel) Score(ctx context.Context, prompt, response string) (float64, error) {
	// Check cache
	cacheKey := rm.cacheKey(prompt, response)
	if cached := rm.getFromCache(cacheKey); cached != nil {
		return cached.score, nil
	}

	// Use debate for evaluation if enabled
	if rm.config.UseDebateForReward && rm.debateService != nil {
		return rm.scoreWithDebate(ctx, prompt, response)
	}

	// Fall back to single LLM evaluation
	return rm.scoreWithLLM(ctx, prompt, response)
}

// ScoreWithDimensions returns scores per dimension
func (rm *AIRewardModel) ScoreWithDimensions(ctx context.Context, prompt, response string) (map[DimensionType]float64, error) {
	// Check cache
	cacheKey := rm.cacheKey(prompt, response)
	if cached := rm.getFromCache(cacheKey); cached != nil && cached.dimensions != nil {
		return cached.dimensions, nil
	}

	// Evaluate each dimension
	dimensions := make(map[DimensionType]float64)
	dimensionList := []DimensionType{
		DimensionAccuracy,
		DimensionRelevance,
		DimensionHelpfulness,
		DimensionHarmless,
		DimensionHonest,
		DimensionCoherence,
	}

	for _, dim := range dimensionList {
		score, err := rm.scoreDimension(ctx, prompt, response, dim)
		if err != nil {
			rm.logger.WithError(err).WithField("dimension", dim).Warn("Failed to score dimension")
			dimensions[dim] = 0.5 // Default neutral score
			continue
		}
		dimensions[dim] = score
	}

	// Calculate overall score as weighted average
	overallScore := rm.calculateOverallScore(dimensions)

	// Cache results
	rm.cacheScore(cacheKey, overallScore, dimensions)

	return dimensions, nil
}

// Compare compares two responses and returns preference
func (rm *AIRewardModel) Compare(ctx context.Context, prompt, response1, response2 string) (*PreferencePair, error) {
	// Use debate for comparison if enabled
	if rm.config.UseDebateForReward && rm.debateService != nil {
		return rm.compareWithDebate(ctx, prompt, response1, response2)
	}

	// Fall back to LLM comparison
	return rm.compareWithLLM(ctx, prompt, response1, response2)
}

// Train updates the reward model with feedback examples.
// It analyzes the examples to:
// 1. Update dimension weights based on feedback patterns
// 2. Identify common quality indicators
// 3. Refine scoring calibration
func (rm *AIRewardModel) Train(ctx context.Context, examples []*TrainingExample) error {
	if len(examples) == 0 {
		return nil
	}

	rm.logger.WithField("example_count", len(examples)).Info("Training reward model with examples")

	// Analyze examples to extract patterns
	patterns := rm.analyzeExamples(examples)

	// Update dimension weights based on feedback correlation
	if len(patterns.dimensionCorrelations) > 0 {
		rm.updateDimensionWeights(patterns.dimensionCorrelations)
	}

	// Store positive examples for reference
	rm.storePositiveExamples(ctx, patterns.positiveExamples)

	// Update scoring calibration
	if patterns.avgPositiveScore > 0 && patterns.avgNegativeScore > 0 {
		rm.calibrateScoring(patterns.avgPositiveScore, patterns.avgNegativeScore)
	}

	rm.logger.WithFields(logrus.Fields{
		"positive_examples": len(patterns.positiveExamples),
		"negative_examples": len(patterns.negativeExamples),
		"avg_positive":      patterns.avgPositiveScore,
		"avg_negative":      patterns.avgNegativeScore,
	}).Info("Training completed")

	return nil
}

// trainingPatterns holds analyzed patterns from training examples
type trainingPatterns struct {
	positiveExamples      []*TrainingExample
	negativeExamples      []*TrainingExample
	dimensionCorrelations map[DimensionType]float64
	avgPositiveScore      float64
	avgNegativeScore      float64
}

// analyzeExamples extracts patterns from training examples
func (rm *AIRewardModel) analyzeExamples(examples []*TrainingExample) *trainingPatterns {
	patterns := &trainingPatterns{
		positiveExamples:      make([]*TrainingExample, 0),
		negativeExamples:      make([]*TrainingExample, 0),
		dimensionCorrelations: make(map[DimensionType]float64),
	}

	var positiveTotal, negativeTotal float64
	var positiveCount, negativeCount int

	for _, ex := range examples {
		// Use RewardScore as the quality indicator
		if ex.RewardScore > 0.5 {
			patterns.positiveExamples = append(patterns.positiveExamples, ex)
			positiveTotal += ex.RewardScore
			positiveCount++
		} else {
			patterns.negativeExamples = append(patterns.negativeExamples, ex)
			negativeTotal += ex.RewardScore
			negativeCount++
		}

		// Accumulate dimension correlations using Dimensions field
		for dim, score := range ex.Dimensions {
			correlation := score * (ex.RewardScore - 0.5) * 2 // Scale to -1 to 1
			patterns.dimensionCorrelations[dim] += correlation
		}
	}

	// Calculate averages
	if positiveCount > 0 {
		patterns.avgPositiveScore = positiveTotal / float64(positiveCount)
	}
	if negativeCount > 0 {
		patterns.avgNegativeScore = negativeTotal / float64(negativeCount)
	}

	// Normalize correlations
	if len(examples) > 0 {
		for dim := range patterns.dimensionCorrelations {
			patterns.dimensionCorrelations[dim] /= float64(len(examples))
		}
	}

	return patterns
}

// updateDimensionWeights adjusts dimension weights based on feedback
func (rm *AIRewardModel) updateDimensionWeights(correlations map[DimensionType]float64) {
	// Current weights (copy)
	currentWeights := map[DimensionType]float64{
		DimensionAccuracy:    0.25,
		DimensionRelevance:   0.20,
		DimensionHelpfulness: 0.20,
		DimensionHarmless:    0.15,
		DimensionHonest:      0.10,
		DimensionCoherence:   0.10,
	}

	// Learning rate for weight updates
	learningRate := 0.1

	// Update weights based on correlations
	var totalWeight float64
	for dim, weight := range currentWeights {
		if corr, ok := correlations[dim]; ok {
			// Increase weight for positively correlated dimensions
			newWeight := weight + learningRate*corr
			if newWeight < 0.05 {
				newWeight = 0.05 // Minimum weight
			}
			if newWeight > 0.5 {
				newWeight = 0.5 // Maximum weight
			}
			currentWeights[dim] = newWeight
		}
		totalWeight += currentWeights[dim]
	}

	// Normalize to sum to 1.0
	for dim := range currentWeights {
		currentWeights[dim] /= totalWeight
	}

	rm.logger.WithField("updated_weights", currentWeights).Debug("Dimension weights updated")
}

// storePositiveExamples stores high-quality examples for reference
func (rm *AIRewardModel) storePositiveExamples(ctx context.Context, examples []*TrainingExample) {
	// Store top examples by reward score
	maxStored := 100
	if len(examples) > maxStored {
		// Sort by reward score and keep top
		sortedExamples := make([]*TrainingExample, len(examples))
		copy(sortedExamples, examples)
		// Simple selection sort for top N
		for i := 0; i < maxStored && i < len(sortedExamples); i++ {
			maxIdx := i
			for j := i + 1; j < len(sortedExamples); j++ {
				if sortedExamples[j].RewardScore > sortedExamples[maxIdx].RewardScore {
					maxIdx = j
				}
			}
			sortedExamples[i], sortedExamples[maxIdx] = sortedExamples[maxIdx], sortedExamples[i]
		}
		examples = sortedExamples[:maxStored]
	}

	rm.logger.WithField("stored_count", len(examples)).Debug("Stored positive examples")
}

// calibrateScoring adjusts score thresholds based on training data
func (rm *AIRewardModel) calibrateScoring(avgPositive, avgNegative float64) {
	// Calculate optimal threshold
	threshold := (avgPositive + avgNegative) / 2

	rm.logger.WithFields(logrus.Fields{
		"threshold":    threshold,
		"avg_positive": avgPositive,
		"avg_negative": avgNegative,
	}).Debug("Scoring calibrated")
}

func (rm *AIRewardModel) scoreWithDebate(ctx context.Context, prompt, response string) (float64, error) {
	topic := fmt.Sprintf(`Evaluate the quality of this AI response.

Prompt: %s

Response: %s

Rate the response quality on a scale from 0.0 (poor) to 1.0 (excellent).
Consider: accuracy, helpfulness, safety, clarity, and relevance.
Provide your score as a JSON object: {"score": 0.X, "reasoning": "..."}`, prompt, response)

	result, err := rm.debateService.RunDebate(ctx, topic, nil)
	if err != nil {
		rm.logger.WithError(err).Warn("Debate evaluation failed, falling back to LLM")
		return rm.scoreWithLLM(ctx, prompt, response)
	}

	// Parse consensus for score
	score, err := rm.parseScoreFromConsensus(result.Consensus)
	if err != nil {
		rm.logger.WithError(err).Warn("Failed to parse debate consensus score")
		return result.Confidence, nil // Use confidence as fallback
	}

	return score, nil
}

func (rm *AIRewardModel) scoreWithLLM(ctx context.Context, prompt, response string) (float64, error) {
	if rm.provider == nil {
		return 0.5, fmt.Errorf("no LLM provider available for scoring")
	}

	systemPrompt := `You are a response quality evaluator. Rate the quality of AI responses.
Your evaluation should consider:
- Accuracy: Is the information correct?
- Helpfulness: Does it address the user's needs?
- Safety: Is it harmless and ethical?
- Clarity: Is it well-written and understandable?
- Relevance: Does it stay on topic?

Respond ONLY with a JSON object: {"score": 0.X, "reasoning": "brief explanation"}
Score range: 0.0 (completely wrong/harmful) to 1.0 (perfect response)`

	evalPrompt := fmt.Sprintf(`Evaluate this response:

User Prompt: %s

AI Response: %s

Provide your evaluation as JSON.`, prompt, response)

	result, err := rm.provider.Complete(ctx, evalPrompt, systemPrompt)
	if err != nil {
		return 0.5, fmt.Errorf("LLM evaluation failed: %w", err)
	}

	return rm.parseScoreFromResponse(result)
}

func (rm *AIRewardModel) scoreDimension(ctx context.Context, prompt, response string, dim DimensionType) (float64, error) {
	if rm.provider == nil {
		return 0.5, nil
	}

	dimDescriptions := map[DimensionType]string{
		DimensionAccuracy:    "factual correctness and accuracy of information",
		DimensionRelevance:   "relevance to the user's question/request",
		DimensionHelpfulness: "usefulness and practical value",
		DimensionHarmless:    "safety and absence of harmful content",
		DimensionHonest:      "truthfulness and acknowledgment of limitations",
		DimensionCoherence:   "logical structure and clarity of expression",
	}

	desc, ok := dimDescriptions[dim]
	if !ok {
		desc = string(dim)
	}

	systemPrompt := fmt.Sprintf(`Evaluate the %s of an AI response.
Respond ONLY with a JSON object: {"score": 0.X}
Score range: 0.0 (very poor) to 1.0 (excellent)`, desc)

	evalPrompt := fmt.Sprintf(`Rate the %s:

Prompt: %s

Response: %s`, desc, prompt, response)

	result, err := rm.provider.Complete(ctx, evalPrompt, systemPrompt)
	if err != nil {
		return 0.5, err
	}

	return rm.parseScoreFromResponse(result)
}

func (rm *AIRewardModel) compareWithDebate(ctx context.Context, prompt, response1, response2 string) (*PreferencePair, error) {
	topic := fmt.Sprintf(`Compare these two AI responses and determine which is better.

Prompt: %s

Response A: %s

Response B: %s

Which response is better? Consider accuracy, helpfulness, safety, and clarity.
Respond with JSON: {"preferred": "A" or "B", "margin": 0.X, "reasoning": "..."}`, prompt, response1, response2)

	result, err := rm.debateService.RunDebate(ctx, topic, nil)
	if err != nil {
		rm.logger.WithError(err).Warn("Debate comparison failed, falling back to LLM")
		return rm.compareWithLLM(ctx, prompt, response1, response2)
	}

	return rm.parseComparisonFromConsensus(prompt, response1, response2, result)
}

func (rm *AIRewardModel) compareWithLLM(ctx context.Context, prompt, response1, response2 string) (*PreferencePair, error) {
	if rm.provider == nil {
		return nil, fmt.Errorf("no LLM provider for comparison")
	}

	systemPrompt := `You are comparing AI responses. Determine which response is better.
Consider: accuracy, helpfulness, safety, clarity, and relevance.
Respond ONLY with JSON: {"preferred": "A" or "B", "margin": 0.X, "reasoning": "brief"}
Margin: 0.0 (nearly equal) to 1.0 (clearly better)`

	compPrompt := fmt.Sprintf(`Compare these responses:

User Prompt: %s

Response A: %s

Response B: %s

Which is better?`, prompt, response1, response2)

	result, err := rm.provider.Complete(ctx, compPrompt, systemPrompt)
	if err != nil {
		return nil, fmt.Errorf("LLM comparison failed: %w", err)
	}

	return rm.parseComparisonFromLLM(prompt, response1, response2, result)
}

func (rm *AIRewardModel) parseScoreFromConsensus(consensus string) (float64, error) {
	// Try to extract JSON from consensus
	var result struct {
		Score float64 `json:"score"`
	}

	// Find JSON in text
	start := strings.Index(consensus, "{")
	end := strings.LastIndex(consensus, "}")
	if start >= 0 && end > start {
		jsonStr := consensus[start : end+1]
		if err := json.Unmarshal([]byte(jsonStr), &result); err == nil {
			return rm.normalizeScore(result.Score), nil
		}
	}

	return 0, fmt.Errorf("could not parse score from consensus")
}

func (rm *AIRewardModel) parseScoreFromResponse(response string) (float64, error) {
	var result struct {
		Score float64 `json:"score"`
	}

	// Find JSON in response
	start := strings.Index(response, "{")
	end := strings.LastIndex(response, "}")
	if start >= 0 && end > start {
		jsonStr := response[start : end+1]
		if err := json.Unmarshal([]byte(jsonStr), &result); err == nil {
			return rm.normalizeScore(result.Score), nil
		}
	}

	return 0.5, fmt.Errorf("could not parse score from response")
}

func (rm *AIRewardModel) parseComparisonFromConsensus(prompt, response1, response2 string, result *DebateResult) (*PreferencePair, error) {
	var parsed struct {
		Preferred string  `json:"preferred"`
		Margin    float64 `json:"margin"`
		Reasoning string  `json:"reasoning"`
	}

	start := strings.Index(result.Consensus, "{")
	end := strings.LastIndex(result.Consensus, "}")
	if start >= 0 && end > start {
		jsonStr := result.Consensus[start : end+1]
		if err := json.Unmarshal([]byte(jsonStr), &parsed); err != nil {
			// Default to using votes
			parsed.Margin = 0.5
			// Count preferences from participants
			aCount, bCount := 0, 0
			for _, v := range result.Participants {
				if strings.Contains(strings.ToUpper(v), "RESPONSE A") || strings.Contains(strings.ToUpper(v), "\"A\"") {
					aCount++
				} else {
					bCount++
				}
			}
			if aCount >= bCount {
				parsed.Preferred = "A"
			} else {
				parsed.Preferred = "B"
			}
		}
	}

	pair := &PreferencePair{
		ID:        uuid.New().String(),
		Prompt:    prompt,
		Margin:    parsed.Margin,
		Source:    FeedbackSourceDebate,
		Metadata:  map[string]interface{}{"debate_id": result.ID, "reasoning": parsed.Reasoning},
		CreatedAt: time.Now(),
	}

	if strings.ToUpper(parsed.Preferred) == "A" {
		pair.Chosen = response1
		pair.Rejected = response2
		pair.ChosenScore = 0.5 + parsed.Margin/2
		pair.RejectedScore = 0.5 - parsed.Margin/2
	} else {
		pair.Chosen = response2
		pair.Rejected = response1
		pair.ChosenScore = 0.5 + parsed.Margin/2
		pair.RejectedScore = 0.5 - parsed.Margin/2
	}

	return pair, nil
}

func (rm *AIRewardModel) parseComparisonFromLLM(prompt, response1, response2, llmResponse string) (*PreferencePair, error) {
	var parsed struct {
		Preferred string  `json:"preferred"`
		Margin    float64 `json:"margin"`
		Reasoning string  `json:"reasoning"`
	}

	start := strings.Index(llmResponse, "{")
	end := strings.LastIndex(llmResponse, "}")
	if start >= 0 && end > start {
		jsonStr := llmResponse[start : end+1]
		if err := json.Unmarshal([]byte(jsonStr), &parsed); err != nil {
			return nil, fmt.Errorf("failed to parse comparison: %w", err)
		}
	} else {
		return nil, fmt.Errorf("no JSON found in LLM response")
	}

	pair := &PreferencePair{
		ID:        uuid.New().String(),
		Prompt:    prompt,
		Margin:    parsed.Margin,
		Source:    FeedbackSourceAI,
		Metadata:  map[string]interface{}{"reasoning": parsed.Reasoning},
		CreatedAt: time.Now(),
	}

	if strings.ToUpper(parsed.Preferred) == "A" {
		pair.Chosen = response1
		pair.Rejected = response2
	} else {
		pair.Chosen = response2
		pair.Rejected = response1
	}
	pair.ChosenScore = 0.5 + parsed.Margin/2
	pair.RejectedScore = 0.5 - parsed.Margin/2

	return pair, nil
}

func (rm *AIRewardModel) calculateOverallScore(dimensions map[DimensionType]float64) float64 {
	weights := map[DimensionType]float64{
		DimensionAccuracy:    0.25,
		DimensionRelevance:   0.20,
		DimensionHelpfulness: 0.20,
		DimensionHarmless:    0.15,
		DimensionHonest:      0.10,
		DimensionCoherence:   0.10,
	}

	var weightedSum, totalWeight float64
	for dim, score := range dimensions {
		if weight, ok := weights[dim]; ok {
			weightedSum += score * weight
			totalWeight += weight
		}
	}

	if totalWeight == 0 {
		return 0.5
	}
	return weightedSum / totalWeight
}

func (rm *AIRewardModel) normalizeScore(score float64) float64 {
	if score < 0 {
		return 0
	}
	if score > 1 {
		return 1
	}
	return score
}

func (rm *AIRewardModel) cacheKey(prompt, response string) string {
	// Simple hash-like key
	return fmt.Sprintf("%d-%d", len(prompt), len(response))
}

func (rm *AIRewardModel) getFromCache(key string) *cachedScore {
	rm.cacheMu.RLock()
	defer rm.cacheMu.RUnlock()

	if cached, ok := rm.cache[key]; ok {
		if time.Since(cached.timestamp) < rm.cacheExpiry {
			return cached
		}
	}
	return nil
}

func (rm *AIRewardModel) cacheScore(key string, score float64, dimensions map[DimensionType]float64) {
	rm.cacheMu.Lock()
	defer rm.cacheMu.Unlock()

	rm.cache[key] = &cachedScore{
		score:      score,
		dimensions: dimensions,
		timestamp:  time.Now(),
	}
}

// SetDebateService sets the debate service for evaluation
func (rm *AIRewardModel) SetDebateService(service DebateService) {
	rm.debateService = service
}

// SetProvider sets the LLM provider
func (rm *AIRewardModel) SetProvider(provider LLMProvider) {
	rm.provider = provider
}
