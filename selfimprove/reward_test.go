package selfimprove

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func init() {
	runtime.GOMAXPROCS(2)
}

// ---------------------------------------------------------------------------
// Mock implementations
// ---------------------------------------------------------------------------

// mockLLMProvider is a configurable mock for the LLMProvider interface
type mockLLMProvider struct {
	mu          sync.Mutex
	response    string
	err         error
	callCount   int
	lastPrompt  string
	lastSystem  string
	responses   []string // sequential responses to return
	responseIdx int
}

func newMockLLMProvider(response string) *mockLLMProvider {
	return &mockLLMProvider{response: response}
}

func newMockLLMProviderWithError(err error) *mockLLMProvider {
	return &mockLLMProvider{err: err}
}

func newMockLLMProviderSequential(responses []string) *mockLLMProvider {
	return &mockLLMProvider{responses: responses}
}

func (m *mockLLMProvider) Complete(_ context.Context, prompt, systemPrompt string) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.callCount++
	m.lastPrompt = prompt
	m.lastSystem = systemPrompt
	if m.err != nil {
		return "", m.err
	}
	if len(m.responses) > 0 {
		idx := m.responseIdx
		if idx >= len(m.responses) {
			idx = len(m.responses) - 1
		}
		m.responseIdx++
		return m.responses[idx], nil
	}
	return m.response, nil
}

func (m *mockLLMProvider) getCallCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.callCount
}

// mockDebateService is a configurable mock for the DebateService interface
type mockDebateService struct {
	mu        sync.Mutex
	result    *DebateResult
	err       error
	callCount int
	lastTopic string
}

func newMockDebateService(result *DebateResult) *mockDebateService {
	return &mockDebateService{result: result}
}

func newMockDebateServiceWithError(err error) *mockDebateService {
	return &mockDebateService{err: err}
}

func (m *mockDebateService) RunDebate(_ context.Context, topic string, _ []string) (*DebateResult, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.callCount++
	m.lastTopic = topic
	if m.err != nil {
		return nil, m.err
	}
	return m.result, nil
}

func (m *mockDebateService) getCallCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.callCount
}

// ---------------------------------------------------------------------------
// NewAIRewardModel
// ---------------------------------------------------------------------------

func TestNewAIRewardModel_WithAllParams(t *testing.T) {
	provider := newMockLLMProvider(`{"score": 0.8}`)
	debate := newMockDebateService(nil)
	config := DefaultSelfImprovementConfig()

	rm := NewAIRewardModel(provider, debate, config, nil)

	require.NotNil(t, rm)
	assert.NotNil(t, rm.provider)
	assert.NotNil(t, rm.debateService)
	assert.NotNil(t, rm.config)
	assert.NotNil(t, rm.logger)
	assert.NotNil(t, rm.cache)
	assert.Equal(t, 15*time.Minute, rm.cacheExpiry)
}

func TestNewAIRewardModel_NilConfig(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	require.NotNil(t, rm)
	require.NotNil(t, rm.config)
	assert.Equal(t, "claude", rm.config.RewardModelProvider)
	assert.Equal(t, "claude-3-sonnet", rm.config.RewardModelName)
}

func TestNewAIRewardModel_NilLogger(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, DefaultSelfImprovementConfig(), nil)

	require.NotNil(t, rm)
	assert.NotNil(t, rm.logger)
}

func TestNewAIRewardModel_NilProvider(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)
	require.NotNil(t, rm)
	assert.Nil(t, rm.provider)
}

func TestNewAIRewardModel_NilDebateService(t *testing.T) {
	provider := newMockLLMProvider(`{"score": 0.5}`)
	rm := NewAIRewardModel(provider, nil, nil, nil)
	require.NotNil(t, rm)
	assert.Nil(t, rm.debateService)
}

func TestNewAIRewardModel_CacheInitialized(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)
	require.NotNil(t, rm.cache)
	assert.Len(t, rm.cache, 0)
}

// ---------------------------------------------------------------------------
// Score — LLM path (debate disabled)
// ---------------------------------------------------------------------------

func TestAIRewardModel_Score_LLMPath_ValidJSON(t *testing.T) {
	provider := newMockLLMProvider(`{"score": 0.85, "reasoning": "good response"}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	score, err := rm.Score(context.Background(), "test prompt", "test response")
	require.NoError(t, err)
	assert.Equal(t, 0.85, score)
}

func TestAIRewardModel_Score_LLMPath_WrappedJSON(t *testing.T) {
	provider := newMockLLMProvider(`Here is my evaluation: {"score": 0.72, "reasoning": "decent"} end`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	score, err := rm.Score(context.Background(), "prompt", "response")
	require.NoError(t, err)
	assert.Equal(t, 0.72, score)
}

func TestAIRewardModel_Score_LLMPath_NoJSON(t *testing.T) {
	provider := newMockLLMProvider("This response has no JSON at all")
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	score, err := rm.Score(context.Background(), "prompt", "response")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "could not parse score")
	assert.Equal(t, 0.5, score) // default fallback
}

func TestAIRewardModel_Score_LLMPath_ProviderError(t *testing.T) {
	provider := newMockLLMProviderWithError(fmt.Errorf("network timeout"))
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	score, err := rm.Score(context.Background(), "prompt", "response")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "LLM evaluation failed")
	assert.Equal(t, 0.5, score)
}

func TestAIRewardModel_Score_LLMPath_NilProvider(t *testing.T) {
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(nil, nil, config, nil)

	score, err := rm.Score(context.Background(), "prompt", "response")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no LLM provider available")
	assert.Equal(t, 0.5, score)
}

func TestAIRewardModel_Score_LLMPath_ScoreNormalization_Negative(t *testing.T) {
	provider := newMockLLMProvider(`{"score": -0.5}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	score, err := rm.Score(context.Background(), "prompt", "response")
	require.NoError(t, err)
	assert.Equal(t, float64(0), score) // normalized to 0
}

func TestAIRewardModel_Score_LLMPath_ScoreNormalization_AboveOne(t *testing.T) {
	provider := newMockLLMProvider(`{"score": 1.5}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	score, err := rm.Score(context.Background(), "prompt", "response")
	require.NoError(t, err)
	assert.Equal(t, float64(1), score) // normalized to 1
}

func TestAIRewardModel_Score_LLMPath_ScoreNormalization_Zero(t *testing.T) {
	provider := newMockLLMProvider(`{"score": 0.0}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	score, err := rm.Score(context.Background(), "prompt", "response")
	require.NoError(t, err)
	assert.Equal(t, float64(0), score)
}

func TestAIRewardModel_Score_LLMPath_ScoreNormalization_One(t *testing.T) {
	provider := newMockLLMProvider(`{"score": 1.0}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	score, err := rm.Score(context.Background(), "prompt", "response")
	require.NoError(t, err)
	assert.Equal(t, float64(1), score)
}

func TestAIRewardModel_Score_LLMPath_InvalidJSON(t *testing.T) {
	provider := newMockLLMProvider(`{score: not valid json}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	score, err := rm.Score(context.Background(), "prompt", "response")
	assert.Error(t, err)
	assert.Equal(t, 0.5, score)
}

// ---------------------------------------------------------------------------
// Score — Debate path
// ---------------------------------------------------------------------------

func TestAIRewardModel_Score_DebatePath_ValidJSON(t *testing.T) {
	debateResult := &DebateResult{
		ID:         "d-1",
		Consensus:  `The score is {"score": 0.9, "reasoning": "excellent"}`,
		Confidence: 0.85,
	}
	debate := newMockDebateService(debateResult)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = true

	rm := NewAIRewardModel(nil, debate, config, nil)

	score, err := rm.Score(context.Background(), "prompt", "response")
	require.NoError(t, err)
	assert.Equal(t, 0.9, score)
}

func TestAIRewardModel_Score_DebatePath_NoJSON_FallbackToConfidence(t *testing.T) {
	debateResult := &DebateResult{
		ID:         "d-2",
		Consensus:  "The response is good but no JSON here",
		Confidence: 0.75,
	}
	debate := newMockDebateService(debateResult)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = true

	rm := NewAIRewardModel(nil, debate, config, nil)

	score, err := rm.Score(context.Background(), "prompt", "response")
	require.NoError(t, err)
	assert.Equal(t, 0.75, score) // falls back to Confidence
}

func TestAIRewardModel_Score_DebatePath_Error_FallbackToLLM(t *testing.T) {
	debate := newMockDebateServiceWithError(fmt.Errorf("debate failed"))
	provider := newMockLLMProvider(`{"score": 0.65}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = true

	rm := NewAIRewardModel(provider, debate, config, nil)

	score, err := rm.Score(context.Background(), "prompt", "response")
	require.NoError(t, err)
	assert.Equal(t, 0.65, score)
	assert.Equal(t, 1, debate.getCallCount())
	assert.Equal(t, 1, provider.getCallCount())
}

func TestAIRewardModel_Score_DebatePath_Error_FallbackToLLM_AlsoFails(t *testing.T) {
	debate := newMockDebateServiceWithError(fmt.Errorf("debate failed"))
	provider := newMockLLMProviderWithError(fmt.Errorf("llm also failed"))
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = true

	rm := NewAIRewardModel(provider, debate, config, nil)

	score, err := rm.Score(context.Background(), "prompt", "response")
	assert.Error(t, err)
	assert.Equal(t, 0.5, score)
}

func TestAIRewardModel_Score_DebateEnabled_ButNilDebateService(t *testing.T) {
	provider := newMockLLMProvider(`{"score": 0.7}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = true

	rm := NewAIRewardModel(provider, nil, config, nil)

	score, err := rm.Score(context.Background(), "prompt", "response")
	require.NoError(t, err)
	assert.Equal(t, 0.7, score) // falls through to LLM
}

func TestAIRewardModel_Score_DebateDisabled(t *testing.T) {
	debate := newMockDebateService(&DebateResult{Consensus: `{"score": 0.9}`})
	provider := newMockLLMProvider(`{"score": 0.6}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, debate, config, nil)

	score, err := rm.Score(context.Background(), "prompt", "response")
	require.NoError(t, err)
	assert.Equal(t, 0.6, score) // uses LLM, not debate
	assert.Equal(t, 0, debate.getCallCount())
}

// ---------------------------------------------------------------------------
// Score — Caching
// ---------------------------------------------------------------------------

func TestAIRewardModel_Score_Caching_HitOnSecondCall(t *testing.T) {
	// Score method checks cache but scoreWithLLM does NOT write to cache.
	// Only ScoreWithDimensions writes to cache. So Score calls the provider
	// every time unless the cache was populated by ScoreWithDimensions.
	provider := newMockLLMProvider(`{"score": 0.8}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	// Pre-populate cache manually to verify cache read path
	key := rm.cacheKey("prompt", "response")
	rm.cacheScore(key, 0.85, nil)

	score, err := rm.Score(context.Background(), "prompt", "response")
	require.NoError(t, err)
	assert.Equal(t, 0.85, score) // from cache
	assert.Equal(t, 0, provider.getCallCount()) // provider NOT called
}

func TestAIRewardModel_Score_NoCacheWrite_FromLLMPath(t *testing.T) {
	// Verify that Score via LLM path does NOT cache and calls provider each time
	provider := newMockLLMProvider(`{"score": 0.8}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	// First call
	score1, err := rm.Score(context.Background(), "prompt", "response")
	require.NoError(t, err)
	assert.Equal(t, 0.8, score1)
	assert.Equal(t, 1, provider.getCallCount())

	// Second call: same inputs, but Score+LLM does not write cache
	score2, err := rm.Score(context.Background(), "prompt", "response")
	require.NoError(t, err)
	assert.Equal(t, 0.8, score2)
	assert.Equal(t, 2, provider.getCallCount()) // called again, no caching
}

func TestAIRewardModel_Score_Caching_DifferentInputs(t *testing.T) {
	// Note: cacheKey uses len(prompt)-len(response), so inputs with
	// different lengths produce different cache keys.
	provider := newMockLLMProviderSequential([]string{
		`{"score": 0.7}`,
		`{"score": 0.9}`,
	})
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	s1, err := rm.Score(context.Background(), "short", "resp")
	require.NoError(t, err)

	s2, err := rm.Score(context.Background(), "a much longer prompt text", "a much longer response text here")
	require.NoError(t, err)

	// If the lengths differ, different cache keys are used
	if len("short") != len("a much longer prompt text") || len("resp") != len("a much longer response text here") {
		assert.NotEqual(t, s1, s2)
	}
}

func TestAIRewardModel_Score_Caching_Expiry(t *testing.T) {
	provider := newMockLLMProviderSequential([]string{
		`{"score": 0.7}`,
		`{"score": 0.9}`,
	})
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)
	rm.cacheExpiry = 1 * time.Millisecond // very short expiry

	s1, err := rm.Score(context.Background(), "prompt", "response")
	require.NoError(t, err)
	assert.Equal(t, 0.7, s1)

	// Wait for cache to expire
	time.Sleep(5 * time.Millisecond)

	s2, err := rm.Score(context.Background(), "prompt", "response")
	require.NoError(t, err)
	assert.Equal(t, 0.9, s2)
	assert.Equal(t, 2, provider.getCallCount())
}

// ---------------------------------------------------------------------------
// ScoreWithDimensions
// ---------------------------------------------------------------------------

func TestAIRewardModel_ScoreWithDimensions_AllDimensions(t *testing.T) {
	// Provider returns the same score for each dimension call
	provider := newMockLLMProvider(`{"score": 0.8}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	dims, err := rm.ScoreWithDimensions(context.Background(), "test prompt", "test response")
	require.NoError(t, err)
	require.NotNil(t, dims)

	expectedDimensions := []DimensionType{
		DimensionAccuracy,
		DimensionRelevance,
		DimensionHelpfulness,
		DimensionHarmless,
		DimensionHonest,
		DimensionCoherence,
	}

	for _, dim := range expectedDimensions {
		score, ok := dims[dim]
		assert.True(t, ok, "missing dimension: %s", dim)
		assert.Equal(t, 0.8, score, "unexpected score for %s", dim)
	}
}

func TestAIRewardModel_ScoreWithDimensions_ProviderError_DefaultsTo05(t *testing.T) {
	provider := newMockLLMProviderWithError(fmt.Errorf("provider error"))
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	dims, err := rm.ScoreWithDimensions(context.Background(), "test", "response")
	require.NoError(t, err) // individual failures are logged, not returned
	require.NotNil(t, dims)

	// All dimensions should default to 0.5
	for _, dim := range []DimensionType{DimensionAccuracy, DimensionRelevance, DimensionHelpfulness, DimensionHarmless, DimensionHonest, DimensionCoherence} {
		assert.Equal(t, 0.5, dims[dim], "dimension %s should default to 0.5", dim)
	}
}

func TestAIRewardModel_ScoreWithDimensions_NilProvider_DefaultsTo05(t *testing.T) {
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(nil, nil, config, nil)

	dims, err := rm.ScoreWithDimensions(context.Background(), "test", "response")
	require.NoError(t, err)
	require.NotNil(t, dims)

	for _, dim := range []DimensionType{DimensionAccuracy, DimensionRelevance, DimensionHelpfulness, DimensionHarmless, DimensionHonest, DimensionCoherence} {
		assert.Equal(t, 0.5, dims[dim])
	}
}

func TestAIRewardModel_ScoreWithDimensions_VaryingScores(t *testing.T) {
	// Return different scores per dimension call
	responses := []string{
		`{"score": 0.95}`, // accuracy
		`{"score": 0.80}`, // relevance
		`{"score": 0.70}`, // helpfulness
		`{"score": 0.90}`, // harmless
		`{"score": 0.60}`, // honest
		`{"score": 0.85}`, // coherence
	}
	provider := newMockLLMProviderSequential(responses)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	dims, err := rm.ScoreWithDimensions(context.Background(), "test prompt", "test response")
	require.NoError(t, err)

	// Verify that 6 calls were made (one per dimension)
	assert.Equal(t, 6, provider.getCallCount())

	// Verify dimension count
	assert.Len(t, dims, 6)
}

func TestAIRewardModel_ScoreWithDimensions_CachesResults(t *testing.T) {
	provider := newMockLLMProvider(`{"score": 0.75}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	// First call
	dims1, err := rm.ScoreWithDimensions(context.Background(), "test prompt", "test response")
	require.NoError(t, err)
	callCount1 := provider.getCallCount()

	// Second call should use cache
	dims2, err := rm.ScoreWithDimensions(context.Background(), "test prompt", "test response")
	require.NoError(t, err)
	callCount2 := provider.getCallCount()

	assert.Equal(t, callCount1, callCount2) // no additional calls
	assert.Equal(t, dims1, dims2)
}

// ---------------------------------------------------------------------------
// Compare — LLM path
// ---------------------------------------------------------------------------

func TestAIRewardModel_Compare_LLMPath_PreferA(t *testing.T) {
	provider := newMockLLMProvider(`{"preferred": "A", "margin": 0.7, "reasoning": "A is better"}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	pair, err := rm.Compare(context.Background(), "prompt", "responseA", "responseB")
	require.NoError(t, err)
	require.NotNil(t, pair)

	assert.Equal(t, "responseA", pair.Chosen)
	assert.Equal(t, "responseB", pair.Rejected)
	assert.Equal(t, 0.7, pair.Margin)
	assert.Equal(t, FeedbackSourceAI, pair.Source)
	assert.InDelta(t, 0.5+0.7/2, pair.ChosenScore, 0.0001)
	assert.InDelta(t, 0.5-0.7/2, pair.RejectedScore, 0.0001)
	assert.NotEmpty(t, pair.ID)
	assert.False(t, pair.CreatedAt.IsZero())
}

func TestAIRewardModel_Compare_LLMPath_PreferB(t *testing.T) {
	provider := newMockLLMProvider(`{"preferred": "B", "margin": 0.3, "reasoning": "B is slightly better"}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	pair, err := rm.Compare(context.Background(), "prompt", "responseA", "responseB")
	require.NoError(t, err)
	require.NotNil(t, pair)

	assert.Equal(t, "responseB", pair.Chosen)
	assert.Equal(t, "responseA", pair.Rejected)
	assert.Equal(t, 0.3, pair.Margin)
}

func TestAIRewardModel_Compare_LLMPath_LowercasePreference(t *testing.T) {
	provider := newMockLLMProvider(`{"preferred": "a", "margin": 0.5, "reasoning": "A wins"}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	pair, err := rm.Compare(context.Background(), "prompt", "r1", "r2")
	require.NoError(t, err)
	require.NotNil(t, pair)
	assert.Equal(t, "r1", pair.Chosen)
	assert.Equal(t, "r2", pair.Rejected)
}

func TestAIRewardModel_Compare_LLMPath_NoJSON(t *testing.T) {
	provider := newMockLLMProvider("No JSON response")
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	pair, err := rm.Compare(context.Background(), "prompt", "r1", "r2")
	assert.Error(t, err)
	assert.Nil(t, pair)
	assert.Contains(t, err.Error(), "no JSON found")
}

func TestAIRewardModel_Compare_LLMPath_InvalidJSON(t *testing.T) {
	provider := newMockLLMProvider(`{invalid json here}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	pair, err := rm.Compare(context.Background(), "prompt", "r1", "r2")
	assert.Error(t, err)
	assert.Nil(t, pair)
	assert.Contains(t, err.Error(), "failed to parse comparison")
}

func TestAIRewardModel_Compare_LLMPath_ProviderError(t *testing.T) {
	provider := newMockLLMProviderWithError(fmt.Errorf("connection refused"))
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	pair, err := rm.Compare(context.Background(), "prompt", "r1", "r2")
	assert.Error(t, err)
	assert.Nil(t, pair)
	assert.Contains(t, err.Error(), "LLM comparison failed")
}

func TestAIRewardModel_Compare_LLMPath_NilProvider(t *testing.T) {
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(nil, nil, config, nil)

	pair, err := rm.Compare(context.Background(), "prompt", "r1", "r2")
	assert.Error(t, err)
	assert.Nil(t, pair)
	assert.Contains(t, err.Error(), "no LLM provider")
}

func TestAIRewardModel_Compare_LLMPath_ZeroMargin(t *testing.T) {
	provider := newMockLLMProvider(`{"preferred": "A", "margin": 0.0, "reasoning": "tie"}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	pair, err := rm.Compare(context.Background(), "prompt", "r1", "r2")
	require.NoError(t, err)
	require.NotNil(t, pair)

	assert.Equal(t, 0.0, pair.Margin)
	assert.Equal(t, 0.5, pair.ChosenScore)
	assert.Equal(t, 0.5, pair.RejectedScore)
}

func TestAIRewardModel_Compare_LLMPath_MetadataContainsReasoning(t *testing.T) {
	provider := newMockLLMProvider(`{"preferred": "A", "margin": 0.4, "reasoning": "response A more accurate"}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	pair, err := rm.Compare(context.Background(), "p", "r1", "r2")
	require.NoError(t, err)
	require.NotNil(t, pair.Metadata)
	assert.Equal(t, "response A more accurate", pair.Metadata["reasoning"])
}

// ---------------------------------------------------------------------------
// Compare — Debate path
// ---------------------------------------------------------------------------

func TestAIRewardModel_Compare_DebatePath_PreferA(t *testing.T) {
	debateResult := &DebateResult{
		ID:        "cd-1",
		Consensus: `{"preferred": "A", "margin": 0.8, "reasoning": "A wins"}`,
		Participants: map[string]string{
			"judge1": "Response A is better",
		},
	}
	debate := newMockDebateService(debateResult)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = true

	rm := NewAIRewardModel(nil, debate, config, nil)

	pair, err := rm.Compare(context.Background(), "prompt", "rA", "rB")
	require.NoError(t, err)
	require.NotNil(t, pair)

	assert.Equal(t, "rA", pair.Chosen)
	assert.Equal(t, "rB", pair.Rejected)
	assert.Equal(t, 0.8, pair.Margin)
	assert.Equal(t, FeedbackSourceDebate, pair.Source)
	assert.Equal(t, "cd-1", pair.Metadata["debate_id"])
}

func TestAIRewardModel_Compare_DebatePath_PreferB(t *testing.T) {
	debateResult := &DebateResult{
		ID:        "cd-2",
		Consensus: `{"preferred": "B", "margin": 0.6, "reasoning": "B wins"}`,
	}
	debate := newMockDebateService(debateResult)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = true

	rm := NewAIRewardModel(nil, debate, config, nil)

	pair, err := rm.Compare(context.Background(), "prompt", "rA", "rB")
	require.NoError(t, err)
	require.NotNil(t, pair)

	assert.Equal(t, "rB", pair.Chosen)
	assert.Equal(t, "rA", pair.Rejected)
}

func TestAIRewardModel_Compare_DebatePath_InvalidJSON_FallbackToVoteCounting(t *testing.T) {
	debateResult := &DebateResult{
		ID:        "cd-3",
		Consensus: `{invalid json}`,
		Participants: map[string]string{
			"judge1": `Preferred RESPONSE A`,
			"judge2": `Preferred RESPONSE A`,
			"judge3": `Response B is okay`,
		},
	}
	debate := newMockDebateService(debateResult)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = true

	rm := NewAIRewardModel(nil, debate, config, nil)

	pair, err := rm.Compare(context.Background(), "prompt", "rA", "rB")
	require.NoError(t, err)
	require.NotNil(t, pair)

	// A has 2 votes, B has 1, so A should be chosen
	assert.Equal(t, "rA", pair.Chosen)
	assert.Equal(t, "rB", pair.Rejected)
}

func TestAIRewardModel_Compare_DebatePath_InvalidJSON_FallbackBPreferred(t *testing.T) {
	// Consensus has braces (so JSON extraction is attempted) but is invalid JSON.
	// Vote counting should prefer B since 2 participants vote B, 1 votes A.
	debateResult := &DebateResult{
		ID:        "cd-4",
		Consensus: `{invalid json content here}`,
		Participants: map[string]string{
			"j1": "Response B is better",
			"j2": "Response B wins",
			"j3": `Preferred RESPONSE A clearly`,
		},
	}
	debate := newMockDebateService(debateResult)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = true

	rm := NewAIRewardModel(nil, debate, config, nil)

	pair, err := rm.Compare(context.Background(), "prompt", "rA", "rB")
	require.NoError(t, err)
	require.NotNil(t, pair)

	// j3 matches RESPONSE A pattern (1 vote for A), j1 and j2 go to B (2 votes)
	// bCount(2) > aCount(1) => B wins
	assert.Equal(t, "rB", pair.Chosen)
}

func TestAIRewardModel_Compare_DebatePath_Error_FallbackToLLM(t *testing.T) {
	debate := newMockDebateServiceWithError(fmt.Errorf("debate error"))
	provider := newMockLLMProvider(`{"preferred": "A", "margin": 0.5, "reasoning": "LLM fallback"}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = true

	rm := NewAIRewardModel(provider, debate, config, nil)

	pair, err := rm.Compare(context.Background(), "prompt", "rA", "rB")
	require.NoError(t, err)
	require.NotNil(t, pair)

	assert.Equal(t, "rA", pair.Chosen)
	assert.Equal(t, FeedbackSourceAI, pair.Source) // LLM fallback uses AI source
}

func TestAIRewardModel_Compare_DebateEnabled_NilDebate_FallbackToLLM(t *testing.T) {
	provider := newMockLLMProvider(`{"preferred": "B", "margin": 0.3}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = true

	rm := NewAIRewardModel(provider, nil, config, nil)

	pair, err := rm.Compare(context.Background(), "p", "r1", "r2")
	require.NoError(t, err)
	require.NotNil(t, pair)
	assert.Equal(t, "r2", pair.Chosen)
}

func TestAIRewardModel_Compare_DebatePath_NoJSON_EmptyParticipants(t *testing.T) {
	debateResult := &DebateResult{
		ID:           "cd-5",
		Consensus:    `no json`,
		Participants: map[string]string{},
	}
	debate := newMockDebateService(debateResult)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = true

	rm := NewAIRewardModel(nil, debate, config, nil)

	pair, err := rm.Compare(context.Background(), "p", "r1", "r2")
	require.NoError(t, err)
	require.NotNil(t, pair)
	// Consensus has no braces, so the JSON extraction block is skipped entirely.
	// parsed.Preferred stays empty string; strings.ToUpper("") != "A" => else branch => response2 is Chosen.
	assert.Equal(t, "r2", pair.Chosen)
}

// ---------------------------------------------------------------------------
// Train
// ---------------------------------------------------------------------------

func TestAIRewardModel_Train_EmptyExamples(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	err := rm.Train(context.Background(), []*TrainingExample{})
	assert.NoError(t, err)
}

func TestAIRewardModel_Train_NilExamples(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	err := rm.Train(context.Background(), nil)
	assert.NoError(t, err)
}

func TestAIRewardModel_Train_SinglePositiveExample(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	examples := []*TrainingExample{
		{
			ID:          "ex-1",
			Prompt:      "What is Go?",
			Response:    "Go is a programming language.",
			RewardScore: 0.9,
			Dimensions: map[DimensionType]float64{
				DimensionAccuracy:    0.95,
				DimensionHelpfulness: 0.85,
			},
		},
	}

	err := rm.Train(context.Background(), examples)
	assert.NoError(t, err)
}

func TestAIRewardModel_Train_SingleNegativeExample(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	examples := []*TrainingExample{
		{
			ID:          "ex-2",
			Prompt:      "What is Go?",
			Response:    "Go is a board game from China.",
			RewardScore: 0.2,
			Dimensions: map[DimensionType]float64{
				DimensionAccuracy:    0.1,
				DimensionRelevance:   0.3,
				DimensionHelpfulness: 0.2,
			},
		},
	}

	err := rm.Train(context.Background(), examples)
	assert.NoError(t, err)
}

func TestAIRewardModel_Train_MixedExamples(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	examples := []*TrainingExample{
		{
			ID: "ex-pos-1", Prompt: "What is Go?", Response: "Go is a language.",
			RewardScore: 0.85,
			Dimensions: map[DimensionType]float64{
				DimensionAccuracy: 0.9, DimensionRelevance: 0.8,
			},
		},
		{
			ID: "ex-pos-2", Prompt: "Explain AI", Response: "AI is artificial intelligence.",
			RewardScore: 0.75,
			Dimensions: map[DimensionType]float64{
				DimensionAccuracy: 0.8, DimensionHelpfulness: 0.7,
			},
		},
		{
			ID: "ex-neg-1", Prompt: "What is 2+2?", Response: "2+2 is 5.",
			RewardScore: 0.1,
			Dimensions: map[DimensionType]float64{
				DimensionAccuracy: 0.0, DimensionHelpfulness: 0.2,
			},
		},
		{
			ID: "ex-neg-2", Prompt: "Explain quantum", Response: "It is magic.",
			RewardScore: 0.3,
			Dimensions: map[DimensionType]float64{
				DimensionAccuracy: 0.1, DimensionCoherence: 0.3,
			},
		},
	}

	err := rm.Train(context.Background(), examples)
	assert.NoError(t, err)
}

func TestAIRewardModel_Train_ExamplesWithNilDimensions(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	examples := []*TrainingExample{
		{
			ID:          "ex-nil-dim",
			Prompt:      "Test",
			Response:    "Test response",
			RewardScore: 0.7,
			Dimensions:  nil,
		},
	}

	err := rm.Train(context.Background(), examples)
	assert.NoError(t, err)
}

func TestAIRewardModel_Train_ExamplesWithEmptyDimensions(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	examples := []*TrainingExample{
		{
			ID:          "ex-empty-dim",
			Prompt:      "Test",
			Response:    "Response",
			RewardScore: 0.6,
			Dimensions:  map[DimensionType]float64{},
		},
	}

	err := rm.Train(context.Background(), examples)
	assert.NoError(t, err)
}

func TestAIRewardModel_Train_BoundaryRewardScore(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	// 0.5 boundary: <= 0.5 is negative, > 0.5 is positive
	examples := []*TrainingExample{
		{ID: "ex-boundary", RewardScore: 0.5, Dimensions: map[DimensionType]float64{DimensionAccuracy: 0.5}},
	}

	err := rm.Train(context.Background(), examples)
	assert.NoError(t, err)
}

func TestAIRewardModel_Train_LargeExampleSet(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	examples := make([]*TrainingExample, 200)
	for i := 0; i < 200; i++ {
		score := float64(i) / 200.0
		examples[i] = &TrainingExample{
			ID:          fmt.Sprintf("ex-%d", i),
			Prompt:      fmt.Sprintf("prompt %d", i),
			Response:    fmt.Sprintf("response %d", i),
			RewardScore: score,
			Dimensions: map[DimensionType]float64{
				DimensionAccuracy:    score,
				DimensionRelevance:   score * 0.9,
				DimensionHelpfulness: score * 0.8,
			},
		}
	}

	err := rm.Train(context.Background(), examples)
	assert.NoError(t, err)
}

// ---------------------------------------------------------------------------
// analyzeExamples
// ---------------------------------------------------------------------------

func TestAIRewardModel_analyzeExamples_PartitionsCorrectly(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	examples := []*TrainingExample{
		{ID: "pos1", RewardScore: 0.8, Dimensions: map[DimensionType]float64{DimensionAccuracy: 0.9}},
		{ID: "pos2", RewardScore: 0.6, Dimensions: map[DimensionType]float64{DimensionAccuracy: 0.7}},
		{ID: "neg1", RewardScore: 0.3, Dimensions: map[DimensionType]float64{DimensionAccuracy: 0.2}},
		{ID: "neg2", RewardScore: 0.1, Dimensions: map[DimensionType]float64{DimensionAccuracy: 0.1}},
		{ID: "boundary", RewardScore: 0.5, Dimensions: map[DimensionType]float64{DimensionAccuracy: 0.5}},
	}

	patterns := rm.analyzeExamples(examples)

	assert.Len(t, patterns.positiveExamples, 2)   // 0.8 and 0.6
	assert.Len(t, patterns.negativeExamples, 3)    // 0.3, 0.1, and 0.5 (boundary)
	assert.InDelta(t, 0.7, patterns.avgPositiveScore, 0.001) // (0.8+0.6)/2
	assert.InDelta(t, 0.3, patterns.avgNegativeScore, 0.001) // (0.3+0.1+0.5)/3
}

func TestAIRewardModel_analyzeExamples_AllPositive(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	examples := []*TrainingExample{
		{ID: "p1", RewardScore: 0.9, Dimensions: map[DimensionType]float64{}},
		{ID: "p2", RewardScore: 0.7, Dimensions: map[DimensionType]float64{}},
	}

	patterns := rm.analyzeExamples(examples)

	assert.Len(t, patterns.positiveExamples, 2)
	assert.Len(t, patterns.negativeExamples, 0)
	assert.InDelta(t, 0.8, patterns.avgPositiveScore, 0.001)
	assert.Equal(t, float64(0), patterns.avgNegativeScore)
}

func TestAIRewardModel_analyzeExamples_AllNegative(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	examples := []*TrainingExample{
		{ID: "n1", RewardScore: 0.1, Dimensions: map[DimensionType]float64{}},
		{ID: "n2", RewardScore: 0.3, Dimensions: map[DimensionType]float64{}},
	}

	patterns := rm.analyzeExamples(examples)

	assert.Len(t, patterns.positiveExamples, 0)
	assert.Len(t, patterns.negativeExamples, 2)
	assert.Equal(t, float64(0), patterns.avgPositiveScore)
	assert.InDelta(t, 0.2, patterns.avgNegativeScore, 0.001)
}

func TestAIRewardModel_analyzeExamples_DimensionCorrelations(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	examples := []*TrainingExample{
		{
			ID: "e1", RewardScore: 0.9,
			Dimensions: map[DimensionType]float64{
				DimensionAccuracy:  0.95,
				DimensionRelevance: 0.8,
			},
		},
		{
			ID: "e2", RewardScore: 0.1,
			Dimensions: map[DimensionType]float64{
				DimensionAccuracy:  0.1,
				DimensionRelevance: 0.2,
			},
		},
	}

	patterns := rm.analyzeExamples(examples)

	// Correlations should be normalized by number of examples
	require.Contains(t, patterns.dimensionCorrelations, DimensionAccuracy)
	require.Contains(t, patterns.dimensionCorrelations, DimensionRelevance)

	// For e1: acc=0.95 * (0.9-0.5)*2 = 0.95 * 0.8 = 0.76
	// For e2: acc=0.1 * (0.1-0.5)*2 = 0.1 * -0.8 = -0.08
	// Sum = 0.68, normalized = 0.68/2 = 0.34
	assert.InDelta(t, 0.34, patterns.dimensionCorrelations[DimensionAccuracy], 0.01)
}

func TestAIRewardModel_analyzeExamples_EmptyList(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	patterns := rm.analyzeExamples([]*TrainingExample{})

	assert.Len(t, patterns.positiveExamples, 0)
	assert.Len(t, patterns.negativeExamples, 0)
	assert.Equal(t, float64(0), patterns.avgPositiveScore)
	assert.Equal(t, float64(0), patterns.avgNegativeScore)
	assert.Len(t, patterns.dimensionCorrelations, 0)
}

// ---------------------------------------------------------------------------
// updateDimensionWeights
// ---------------------------------------------------------------------------

func TestAIRewardModel_updateDimensionWeights_PositiveCorrelations(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	correlations := map[DimensionType]float64{
		DimensionAccuracy:    0.5,
		DimensionRelevance:   0.3,
		DimensionHelpfulness: 0.2,
	}

	// Should not panic and should update weights
	rm.updateDimensionWeights(correlations)
}

func TestAIRewardModel_updateDimensionWeights_NegativeCorrelations(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	correlations := map[DimensionType]float64{
		DimensionAccuracy:  -0.5,
		DimensionRelevance: -0.3,
	}

	// Should not panic; weights have min 0.05 floor
	rm.updateDimensionWeights(correlations)
}

func TestAIRewardModel_updateDimensionWeights_LargePositiveCorrelation(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	correlations := map[DimensionType]float64{
		DimensionAccuracy: 10.0, // very large; should cap at 0.5
	}

	rm.updateDimensionWeights(correlations)
}

func TestAIRewardModel_updateDimensionWeights_EmptyCorrelations(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)
	rm.updateDimensionWeights(map[DimensionType]float64{})
}

func TestAIRewardModel_updateDimensionWeights_LargeNegativeCorrelation(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	correlations := map[DimensionType]float64{
		DimensionAccuracy: -10.0, // very negative; should floor at 0.05
	}

	rm.updateDimensionWeights(correlations)
}

func TestAIRewardModel_updateDimensionWeights_AllDimensions(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	correlations := map[DimensionType]float64{
		DimensionAccuracy:    0.1,
		DimensionRelevance:   0.2,
		DimensionHelpfulness: -0.1,
		DimensionHarmless:    0.0,
		DimensionHonest:      0.3,
		DimensionCoherence:   -0.2,
	}

	rm.updateDimensionWeights(correlations)
}

// ---------------------------------------------------------------------------
// calculateOverallScore
// ---------------------------------------------------------------------------

func TestAIRewardModel_calculateOverallScore_AllDimensionsEqual(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	dims := map[DimensionType]float64{
		DimensionAccuracy:    0.8,
		DimensionRelevance:   0.8,
		DimensionHelpfulness: 0.8,
		DimensionHarmless:    0.8,
		DimensionHonest:      0.8,
		DimensionCoherence:   0.8,
	}

	score := rm.calculateOverallScore(dims)
	assert.InDelta(t, 0.8, score, 0.001)
}

func TestAIRewardModel_calculateOverallScore_VaryingScores(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	dims := map[DimensionType]float64{
		DimensionAccuracy:    1.0,
		DimensionRelevance:   0.8,
		DimensionHelpfulness: 0.6,
		DimensionHarmless:    0.4,
		DimensionHonest:      0.2,
		DimensionCoherence:   0.0,
	}

	// weighted: 1.0*0.25 + 0.8*0.20 + 0.6*0.20 + 0.4*0.15 + 0.2*0.10 + 0.0*0.10
	// = 0.25 + 0.16 + 0.12 + 0.06 + 0.02 + 0.00 = 0.61
	// totalWeight = 1.0
	score := rm.calculateOverallScore(dims)
	assert.InDelta(t, 0.61, score, 0.001)
}

func TestAIRewardModel_calculateOverallScore_EmptyDimensions(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	score := rm.calculateOverallScore(map[DimensionType]float64{})
	assert.Equal(t, 0.5, score) // default when totalWeight == 0
}

func TestAIRewardModel_calculateOverallScore_PartialDimensions(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	dims := map[DimensionType]float64{
		DimensionAccuracy:  1.0,
		DimensionRelevance: 1.0,
	}

	// weighted: 1.0*0.25 + 1.0*0.20 = 0.45
	// totalWeight: 0.25 + 0.20 = 0.45
	// result: 0.45/0.45 = 1.0
	score := rm.calculateOverallScore(dims)
	assert.InDelta(t, 1.0, score, 0.001)
}

func TestAIRewardModel_calculateOverallScore_UnknownDimensions(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	dims := map[DimensionType]float64{
		DimensionCreativity: 0.9, // Not in weight map
		DimensionFormatting: 0.8, // Not in weight map
	}

	score := rm.calculateOverallScore(dims)
	assert.Equal(t, 0.5, score) // no known dimensions match
}

func TestAIRewardModel_calculateOverallScore_MixedKnownAndUnknown(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	dims := map[DimensionType]float64{
		DimensionAccuracy:   0.9,
		DimensionCreativity: 0.5, // ignored
	}

	// Only accuracy contributes: 0.9*0.25 / 0.25 = 0.9
	score := rm.calculateOverallScore(dims)
	assert.InDelta(t, 0.9, score, 0.001)
}

// ---------------------------------------------------------------------------
// normalizeScore
// ---------------------------------------------------------------------------

func TestAIRewardModel_normalizeScore(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	tests := []struct {
		name     string
		input    float64
		expected float64
	}{
		{"Negative", -0.5, 0.0},
		{"LargeNegative", -100.0, 0.0},
		{"Zero", 0.0, 0.0},
		{"Normal_05", 0.5, 0.5},
		{"Normal_075", 0.75, 0.75},
		{"One", 1.0, 1.0},
		{"AboveOne", 1.5, 1.0},
		{"LargeAboveOne", 100.0, 1.0},
		{"SmallPositive", 0.001, 0.001},
		{"AlmostOne", 0.999, 0.999},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := rm.normalizeScore(tt.input)
			assert.InDelta(t, tt.expected, result, 0.0001)
		})
	}
}

// ---------------------------------------------------------------------------
// cacheKey
// ---------------------------------------------------------------------------

func TestAIRewardModel_cacheKey_SameInputs(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	key1 := rm.cacheKey("prompt", "response")
	key2 := rm.cacheKey("prompt", "response")

	assert.Equal(t, key1, key2)
}

func TestAIRewardModel_cacheKey_DifferentLengths(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	key1 := rm.cacheKey("short", "resp")
	key2 := rm.cacheKey("a longer prompt", "a longer response")

	assert.NotEqual(t, key1, key2)
}

func TestAIRewardModel_cacheKey_EmptyInputs(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	key := rm.cacheKey("", "")
	assert.Equal(t, "0-0", key)
}

func TestAIRewardModel_cacheKey_CollisionSameLength(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	// Different content but same lengths produce same key
	key1 := rm.cacheKey("aaaa", "bbbb")
	key2 := rm.cacheKey("cccc", "dddd")

	assert.Equal(t, key1, key2, "cache keys based on length will collide for same-length inputs")
}

// ---------------------------------------------------------------------------
// getFromCache / cacheScore
// ---------------------------------------------------------------------------

func TestAIRewardModel_getFromCache_Miss(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	result := rm.getFromCache("nonexistent-key")
	assert.Nil(t, result)
}

func TestAIRewardModel_cacheScore_AndRetrieve(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	dims := map[DimensionType]float64{DimensionAccuracy: 0.9}
	rm.cacheScore("test-key", 0.85, dims)

	cached := rm.getFromCache("test-key")
	require.NotNil(t, cached)
	assert.Equal(t, 0.85, cached.score)
	assert.Equal(t, 0.9, cached.dimensions[DimensionAccuracy])
}

func TestAIRewardModel_getFromCache_Expired(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)
	rm.cacheExpiry = 1 * time.Millisecond

	rm.cacheScore("expiring-key", 0.5, nil)
	time.Sleep(5 * time.Millisecond)

	cached := rm.getFromCache("expiring-key")
	assert.Nil(t, cached)
}

func TestAIRewardModel_cacheScore_Overwrite(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	rm.cacheScore("key", 0.5, nil)
	rm.cacheScore("key", 0.9, map[DimensionType]float64{DimensionAccuracy: 1.0})

	cached := rm.getFromCache("key")
	require.NotNil(t, cached)
	assert.Equal(t, 0.9, cached.score)
	assert.Equal(t, 1.0, cached.dimensions[DimensionAccuracy])
}

func TestAIRewardModel_cacheScore_NilDimensions(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	rm.cacheScore("nil-dims", 0.7, nil)

	cached := rm.getFromCache("nil-dims")
	require.NotNil(t, cached)
	assert.Equal(t, 0.7, cached.score)
	assert.Nil(t, cached.dimensions)
}

// ---------------------------------------------------------------------------
// parseScoreFromConsensus
// ---------------------------------------------------------------------------

func TestAIRewardModel_parseScoreFromConsensus_ValidJSON(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	tests := []struct {
		name      string
		consensus string
		expected  float64
	}{
		{"PureJSON", `{"score": 0.85}`, 0.85},
		{"JSONInText", `The result is {"score": 0.72} as expected`, 0.72},
		{"JSONWithExtra", `{"score": 0.6, "reasoning": "okay"}`, 0.6},
		{"ZeroScore", `{"score": 0.0}`, 0.0},
		{"PerfectScore", `{"score": 1.0}`, 1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score, err := rm.parseScoreFromConsensus(tt.consensus)
			require.NoError(t, err)
			assert.InDelta(t, tt.expected, score, 0.001)
		})
	}
}

func TestAIRewardModel_parseScoreFromConsensus_InvalidJSON(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	score, err := rm.parseScoreFromConsensus(`{not valid json}`)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "could not parse score")
	assert.Equal(t, float64(0), score)
}

func TestAIRewardModel_parseScoreFromConsensus_NoJSON(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	score, err := rm.parseScoreFromConsensus("just text, no braces")
	assert.Error(t, err)
	assert.Equal(t, float64(0), score)
}

func TestAIRewardModel_parseScoreFromConsensus_EmptyString(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	score, err := rm.parseScoreFromConsensus("")
	assert.Error(t, err)
	assert.Equal(t, float64(0), score)
}

func TestAIRewardModel_parseScoreFromConsensus_NormalizesHighScore(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	score, err := rm.parseScoreFromConsensus(`{"score": 1.5}`)
	require.NoError(t, err)
	assert.Equal(t, float64(1), score)
}

func TestAIRewardModel_parseScoreFromConsensus_NormalizesNegativeScore(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	score, err := rm.parseScoreFromConsensus(`{"score": -0.3}`)
	require.NoError(t, err)
	assert.Equal(t, float64(0), score)
}

// ---------------------------------------------------------------------------
// parseScoreFromResponse
// ---------------------------------------------------------------------------

func TestAIRewardModel_parseScoreFromResponse_ValidJSON(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	tests := []struct {
		name     string
		response string
		expected float64
	}{
		{"PureJSON", `{"score": 0.8}`, 0.8},
		{"TextWrapped", `Evaluation: {"score": 0.65} done.`, 0.65},
		{"WithReasoning", `{"score": 0.9, "reasoning": "excellent"}`, 0.9},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score, err := rm.parseScoreFromResponse(tt.response)
			require.NoError(t, err)
			assert.InDelta(t, tt.expected, score, 0.001)
		})
	}
}

func TestAIRewardModel_parseScoreFromResponse_InvalidJSON(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	score, err := rm.parseScoreFromResponse(`{broken}`)
	assert.Error(t, err)
	assert.Equal(t, 0.5, score) // default
}

func TestAIRewardModel_parseScoreFromResponse_NoJSON(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	score, err := rm.parseScoreFromResponse("plain text")
	assert.Error(t, err)
	assert.Equal(t, 0.5, score)
}

func TestAIRewardModel_parseScoreFromResponse_EmptyString(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	score, err := rm.parseScoreFromResponse("")
	assert.Error(t, err)
	assert.Equal(t, 0.5, score)
}

// ---------------------------------------------------------------------------
// parseComparisonFromLLM
// ---------------------------------------------------------------------------

func TestAIRewardModel_parseComparisonFromLLM_PreferA(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	pair, err := rm.parseComparisonFromLLM(
		"prompt", "r1", "r2",
		`{"preferred": "A", "margin": 0.7, "reasoning": "A is better"}`,
	)
	require.NoError(t, err)
	require.NotNil(t, pair)

	assert.Equal(t, "r1", pair.Chosen)
	assert.Equal(t, "r2", pair.Rejected)
	assert.Equal(t, 0.7, pair.Margin)
	assert.Equal(t, "prompt", pair.Prompt)
	assert.Equal(t, FeedbackSourceAI, pair.Source)
	assert.InDelta(t, 0.85, pair.ChosenScore, 0.001)
	assert.InDelta(t, 0.15, pair.RejectedScore, 0.001)
}

func TestAIRewardModel_parseComparisonFromLLM_PreferB(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	pair, err := rm.parseComparisonFromLLM(
		"prompt", "r1", "r2",
		`{"preferred": "B", "margin": 0.3, "reasoning": "B wins"}`,
	)
	require.NoError(t, err)
	require.NotNil(t, pair)

	assert.Equal(t, "r2", pair.Chosen)
	assert.Equal(t, "r1", pair.Rejected)
}

func TestAIRewardModel_parseComparisonFromLLM_NoJSON(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	pair, err := rm.parseComparisonFromLLM("p", "r1", "r2", "no json here")
	assert.Error(t, err)
	assert.Nil(t, pair)
	assert.Contains(t, err.Error(), "no JSON found")
}

func TestAIRewardModel_parseComparisonFromLLM_InvalidJSON(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	pair, err := rm.parseComparisonFromLLM("p", "r1", "r2", `{invalid}`)
	assert.Error(t, err)
	assert.Nil(t, pair)
}

func TestAIRewardModel_parseComparisonFromLLM_WrappedJSON(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	pair, err := rm.parseComparisonFromLLM(
		"p", "r1", "r2",
		`Here is the result: {"preferred": "A", "margin": 0.5, "reasoning": "A wins"} end`,
	)
	require.NoError(t, err)
	require.NotNil(t, pair)
	assert.Equal(t, "r1", pair.Chosen)
}

// ---------------------------------------------------------------------------
// parseComparisonFromConsensus
// ---------------------------------------------------------------------------

func TestAIRewardModel_parseComparisonFromConsensus_ValidJSON_PreferA(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	result := &DebateResult{
		ID:        "d-1",
		Consensus: `{"preferred": "A", "margin": 0.6, "reasoning": "better"}`,
	}

	pair, err := rm.parseComparisonFromConsensus("prompt", "rA", "rB", result)
	require.NoError(t, err)
	require.NotNil(t, pair)

	assert.Equal(t, "rA", pair.Chosen)
	assert.Equal(t, "rB", pair.Rejected)
	assert.Equal(t, 0.6, pair.Margin)
	assert.Equal(t, FeedbackSourceDebate, pair.Source)
}

func TestAIRewardModel_parseComparisonFromConsensus_ValidJSON_PreferB(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	result := &DebateResult{
		ID:        "d-2",
		Consensus: `{"preferred": "B", "margin": 0.4}`,
	}

	pair, err := rm.parseComparisonFromConsensus("prompt", "rA", "rB", result)
	require.NoError(t, err)
	require.NotNil(t, pair)

	assert.Equal(t, "rB", pair.Chosen)
	assert.Equal(t, "rA", pair.Rejected)
}

func TestAIRewardModel_parseComparisonFromConsensus_InvalidJSON_VoteCounting(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	result := &DebateResult{
		ID:        "d-3",
		Consensus: `{broken json}`,
		Participants: map[string]string{
			"j1": `RESPONSE A is clearly better`,
			"j2": `I think "A" is the right choice`,
			"j3": "Response B looks good",
		},
	}

	pair, err := rm.parseComparisonFromConsensus("prompt", "rA", "rB", result)
	require.NoError(t, err)
	require.NotNil(t, pair)

	// j1 matches RESPONSE A, j2 matches "A", j3 goes to B: 2 vs 1 => A wins
	assert.Equal(t, "rA", pair.Chosen)
}

func TestAIRewardModel_parseComparisonFromConsensus_NoJSON_NoBraces(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	result := &DebateResult{
		ID:           "d-4",
		Consensus:    "no json here at all",
		Participants: map[string]string{},
	}

	pair, err := rm.parseComparisonFromConsensus("p", "r1", "r2", result)
	require.NoError(t, err)
	require.NotNil(t, pair)

	// No braces in consensus => parsed.Preferred stays empty => else branch => B
	// But wait, no, if no braces found, the if block is skipped entirely,
	// and parsed.Preferred stays empty. strings.ToUpper("") != "A" => else => B
	assert.Equal(t, "r2", pair.Chosen)
}

func TestAIRewardModel_parseComparisonFromConsensus_ScoresCalculation(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	result := &DebateResult{
		ID:        "d-5",
		Consensus: `{"preferred": "A", "margin": 0.8}`,
	}

	pair, err := rm.parseComparisonFromConsensus("p", "r1", "r2", result)
	require.NoError(t, err)
	require.NotNil(t, pair)

	assert.InDelta(t, 0.5+0.8/2, pair.ChosenScore, 0.001)    // 0.9
	assert.InDelta(t, 0.5-0.8/2, pair.RejectedScore, 0.001)  // 0.1
}

// ---------------------------------------------------------------------------
// scoreDimension
// ---------------------------------------------------------------------------

func TestAIRewardModel_scoreDimension_ValidDimension(t *testing.T) {
	provider := newMockLLMProvider(`{"score": 0.88}`)
	rm := NewAIRewardModel(provider, nil, nil, nil)

	score, err := rm.scoreDimension(context.Background(), "prompt", "response", DimensionAccuracy)
	require.NoError(t, err)
	assert.Equal(t, 0.88, score)
}

func TestAIRewardModel_scoreDimension_UnknownDimension(t *testing.T) {
	provider := newMockLLMProvider(`{"score": 0.7}`)
	rm := NewAIRewardModel(provider, nil, nil, nil)

	// DimensionCreativity is not in dimDescriptions map but should still work
	score, err := rm.scoreDimension(context.Background(), "prompt", "response", DimensionCreativity)
	require.NoError(t, err)
	assert.Equal(t, 0.7, score)
}

func TestAIRewardModel_scoreDimension_NilProvider(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	score, err := rm.scoreDimension(context.Background(), "prompt", "response", DimensionAccuracy)
	require.NoError(t, err)
	assert.Equal(t, 0.5, score) // default when no provider
}

func TestAIRewardModel_scoreDimension_ProviderError(t *testing.T) {
	provider := newMockLLMProviderWithError(fmt.Errorf("timeout"))
	rm := NewAIRewardModel(provider, nil, nil, nil)

	score, err := rm.scoreDimension(context.Background(), "prompt", "response", DimensionAccuracy)
	assert.Error(t, err)
	assert.Equal(t, 0.5, score) // error returned from provider
}

func TestAIRewardModel_scoreDimension_AllKnownDimensions(t *testing.T) {
	provider := newMockLLMProvider(`{"score": 0.75}`)
	rm := NewAIRewardModel(provider, nil, nil, nil)

	knownDims := []DimensionType{
		DimensionAccuracy,
		DimensionRelevance,
		DimensionHelpfulness,
		DimensionHarmless,
		DimensionHonest,
		DimensionCoherence,
	}

	for _, dim := range knownDims {
		t.Run(string(dim), func(t *testing.T) {
			score, err := rm.scoreDimension(context.Background(), "p", "r", dim)
			require.NoError(t, err)
			assert.Equal(t, 0.75, score)
		})
	}
}

// ---------------------------------------------------------------------------
// SetDebateService / SetProvider
// ---------------------------------------------------------------------------

func TestAIRewardModel_SetDebateService(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)
	assert.Nil(t, rm.debateService)

	debate := newMockDebateService(nil)
	rm.SetDebateService(debate)
	assert.NotNil(t, rm.debateService)
}

func TestAIRewardModel_SetDebateService_Nil(t *testing.T) {
	debate := newMockDebateService(nil)
	rm := NewAIRewardModel(nil, debate, nil, nil)
	assert.NotNil(t, rm.debateService)

	rm.SetDebateService(nil)
	assert.Nil(t, rm.debateService)
}

func TestAIRewardModel_SetProvider(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)
	assert.Nil(t, rm.provider)

	provider := newMockLLMProvider(`{"score": 0.5}`)
	rm.SetProvider(provider)
	assert.NotNil(t, rm.provider)
}

func TestAIRewardModel_SetProvider_Nil(t *testing.T) {
	provider := newMockLLMProvider(`{"score": 0.5}`)
	rm := NewAIRewardModel(provider, nil, nil, nil)
	assert.NotNil(t, rm.provider)

	rm.SetProvider(nil)
	assert.Nil(t, rm.provider)
}

func TestAIRewardModel_SetProvider_ReplacesExisting(t *testing.T) {
	p1 := newMockLLMProvider(`{"score": 0.5}`)
	p2 := newMockLLMProvider(`{"score": 0.9}`)

	rm := NewAIRewardModel(p1, nil, nil, nil)
	rm.SetProvider(p2)

	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false
	rm.config = config

	// Clear cache to force new provider call
	rm.cacheMu.Lock()
	rm.cache = make(map[string]*cachedScore)
	rm.cacheMu.Unlock()

	score, err := rm.Score(context.Background(), "prompt", "response")
	require.NoError(t, err)
	assert.Equal(t, 0.9, score)
	assert.Equal(t, 0, p1.getCallCount())
	assert.Equal(t, 1, p2.getCallCount())
}

// ---------------------------------------------------------------------------
// storePositiveExamples
// ---------------------------------------------------------------------------

func TestAIRewardModel_storePositiveExamples_Empty(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)
	// Should not panic
	rm.storePositiveExamples(context.Background(), []*TrainingExample{})
}

func TestAIRewardModel_storePositiveExamples_UnderLimit(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	examples := make([]*TrainingExample, 50)
	for i := 0; i < 50; i++ {
		examples[i] = &TrainingExample{ID: fmt.Sprintf("e-%d", i), RewardScore: float64(i) / 50.0}
	}

	// Should not panic, stores all
	rm.storePositiveExamples(context.Background(), examples)
}

func TestAIRewardModel_storePositiveExamples_OverLimit(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	examples := make([]*TrainingExample, 200)
	for i := 0; i < 200; i++ {
		examples[i] = &TrainingExample{
			ID:          fmt.Sprintf("e-%d", i),
			RewardScore: float64(i) / 200.0,
		}
	}

	// Should trim to top 100 by reward score, not panic
	rm.storePositiveExamples(context.Background(), examples)
}

// ---------------------------------------------------------------------------
// calibrateScoring
// ---------------------------------------------------------------------------

func TestAIRewardModel_calibrateScoring_NormalValues(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)
	// Should not panic
	rm.calibrateScoring(0.8, 0.3)
}

func TestAIRewardModel_calibrateScoring_EqualValues(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)
	rm.calibrateScoring(0.5, 0.5)
}

func TestAIRewardModel_calibrateScoring_ZeroValues(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)
	rm.calibrateScoring(0.0, 0.0)
}

// ---------------------------------------------------------------------------
// Concurrent safety
// ---------------------------------------------------------------------------

func TestAIRewardModel_ConcurrentScore(t *testing.T) {
	provider := newMockLLMProvider(`{"score": 0.75}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	var wg sync.WaitGroup
	errs := make([]error, 20)
	scores := make([]float64, 20)

	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			s, err := rm.Score(context.Background(),
				fmt.Sprintf("prompt-%d", idx),
				fmt.Sprintf("response-%d", idx))
			scores[idx] = s
			errs[idx] = err
		}(i)
	}

	wg.Wait()

	for i := 0; i < 20; i++ {
		if errs[i] != nil {
			t.Logf("goroutine %d error: %v", i, errs[i])
		}
	}
}

func TestAIRewardModel_ConcurrentCacheReadWrite(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	var wg sync.WaitGroup

	// Writers
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			key := fmt.Sprintf("key-%d", idx)
			rm.cacheScore(key, float64(idx)/10.0, map[DimensionType]float64{
				DimensionAccuracy: float64(idx) / 10.0,
			})
		}(i)
	}

	// Readers
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			key := fmt.Sprintf("key-%d", idx)
			_ = rm.getFromCache(key) // may or may not find it
		}(i)
	}

	wg.Wait()
}

func TestAIRewardModel_ConcurrentScoreWithDimensions(t *testing.T) {
	provider := newMockLLMProvider(`{"score": 0.8}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			dims, err := rm.ScoreWithDimensions(context.Background(),
				fmt.Sprintf("prompt-%d", idx),
				fmt.Sprintf("response-%d", idx))
			if err != nil {
				t.Logf("goroutine %d error: %v", idx, err)
			}
			if dims != nil {
				assert.Len(t, dims, 6)
			}
		}(i)
	}

	wg.Wait()
}

func TestAIRewardModel_ConcurrentTrain(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			examples := []*TrainingExample{
				{
					ID:          fmt.Sprintf("ex-%d", idx),
					RewardScore: float64(idx) / 5.0,
					Dimensions: map[DimensionType]float64{
						DimensionAccuracy: float64(idx) / 5.0,
					},
				},
			}
			err := rm.Train(context.Background(), examples)
			assert.NoError(t, err)
		}(i)
	}

	wg.Wait()
}

// ---------------------------------------------------------------------------
// Context cancellation
// ---------------------------------------------------------------------------

func TestAIRewardModel_Score_CancelledContext(t *testing.T) {
	provider := &mockLLMProvider{
		err: fmt.Errorf("context canceled"),
	}
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := rm.Score(ctx, "prompt", "response")
	assert.Error(t, err)
}

// ---------------------------------------------------------------------------
// Integration-style tests (Score -> cache -> ScoreWithDimensions flow)
// ---------------------------------------------------------------------------

func TestAIRewardModel_ScoreAndScoreWithDimensions_CacheInteraction(t *testing.T) {
	// Score caches with the LLM path (score only, no dimensions in cache).
	// ScoreWithDimensions should still work because it checks
	// cached.dimensions != nil before returning from cache.
	responses := []string{
		// Score call:
		`{"score": 0.85}`,
		// ScoreWithDimensions (6 dimension calls):
		`{"score": 0.9}`,
		`{"score": 0.8}`,
		`{"score": 0.7}`,
		`{"score": 0.6}`,
		`{"score": 0.5}`,
		`{"score": 0.4}`,
	}
	provider := newMockLLMProviderSequential(responses)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	// Score first — caches score but with nil dimensions
	score, err := rm.Score(context.Background(), "prompt", "response")
	require.NoError(t, err)
	assert.Equal(t, 0.85, score)

	// ScoreWithDimensions — cache has entry but dimensions are nil,
	// so it should NOT use cache and call provider for each dimension
	dims, err := rm.ScoreWithDimensions(context.Background(), "prompt", "response")
	require.NoError(t, err)
	require.NotNil(t, dims)
	assert.Len(t, dims, 6)
}

func TestAIRewardModel_CompareAndScore_IndependentOperations(t *testing.T) {
	responses := []string{
		// Compare call:
		`{"preferred": "A", "margin": 0.6, "reasoning": "A better"}`,
		// Score call:
		`{"score": 0.77}`,
	}
	provider := newMockLLMProviderSequential(responses)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	pair, err := rm.Compare(context.Background(), "p", "rA", "rB")
	require.NoError(t, err)
	require.NotNil(t, pair)
	assert.Equal(t, "rA", pair.Chosen)

	score, err := rm.Score(context.Background(), "p2", "r2")
	require.NoError(t, err)
	assert.Equal(t, 0.77, score)
}

// ---------------------------------------------------------------------------
// AIRewardModel implements RewardModel interface
// ---------------------------------------------------------------------------

func TestAIRewardModel_ImplementsRewardModel(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)
	var iface RewardModel = rm
	assert.NotNil(t, iface)
}

// ---------------------------------------------------------------------------
// Edge cases and boundary conditions
// ---------------------------------------------------------------------------

func TestAIRewardModel_Score_EmptyPromptAndResponse(t *testing.T) {
	provider := newMockLLMProvider(`{"score": 0.5}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	score, err := rm.Score(context.Background(), "", "")
	require.NoError(t, err)
	assert.Equal(t, 0.5, score)
}

func TestAIRewardModel_Score_VeryLongInput(t *testing.T) {
	longStr := ""
	for i := 0; i < 10000; i++ {
		longStr += "a"
	}
	provider := newMockLLMProvider(`{"score": 0.6}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	score, err := rm.Score(context.Background(), longStr, longStr)
	require.NoError(t, err)
	assert.Equal(t, 0.6, score)
}

func TestAIRewardModel_Compare_SameResponses(t *testing.T) {
	provider := newMockLLMProvider(`{"preferred": "A", "margin": 0.0, "reasoning": "identical"}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	pair, err := rm.Compare(context.Background(), "p", "same response", "same response")
	require.NoError(t, err)
	require.NotNil(t, pair)
	assert.Equal(t, 0.0, pair.Margin)
}

func TestAIRewardModel_Train_AllBoundaryScores(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	examples := []*TrainingExample{
		{ID: "zero", RewardScore: 0.0, Dimensions: map[DimensionType]float64{DimensionAccuracy: 0}},
		{ID: "half", RewardScore: 0.5, Dimensions: map[DimensionType]float64{DimensionAccuracy: 0.5}},
		{ID: "one", RewardScore: 1.0, Dimensions: map[DimensionType]float64{DimensionAccuracy: 1.0}},
	}

	err := rm.Train(context.Background(), examples)
	assert.NoError(t, err)
}

func TestAIRewardModel_Score_DebatePath_JSONWithMultipleBraces(t *testing.T) {
	debateResult := &DebateResult{
		ID:         "d-multi",
		Consensus:  `Some text {"inner": "data"} more text {"score": 0.82} end`,
		Confidence: 0.5,
	}
	debate := newMockDebateService(debateResult)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = true

	rm := NewAIRewardModel(nil, debate, config, nil)

	// parseScoreFromConsensus uses strings.Index (first {) and
	// strings.LastIndex (last }), so it gets everything from first { to last }
	score, err := rm.Score(context.Background(), "p", "r")
	// The JSON extracted will be: {"inner": "data"} more text {"score": 0.82}
	// This is invalid JSON, so it should fall back
	if err != nil {
		// Falls back to confidence
		assert.Equal(t, 0.5, score)
	} else {
		// If it somehow parsed, that's fine too
		assert.GreaterOrEqual(t, score, 0.0)
		assert.LessOrEqual(t, score, 1.0)
	}
}

func TestAIRewardModel_parseScoreFromResponse_NestedJSON(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	// First { to last } captures entire blob
	response := `{"score": 0.75, "details": {"sub": "data"}}`
	score, err := rm.parseScoreFromResponse(response)
	require.NoError(t, err)
	assert.Equal(t, 0.75, score)
}

func TestAIRewardModel_parseComparisonFromLLM_LowercaseB(t *testing.T) {
	rm := NewAIRewardModel(nil, nil, nil, nil)

	pair, err := rm.parseComparisonFromLLM(
		"p", "r1", "r2",
		`{"preferred": "b", "margin": 0.4}`,
	)
	require.NoError(t, err)
	require.NotNil(t, pair)
	assert.Equal(t, "r2", pair.Chosen)
	assert.Equal(t, "r1", pair.Rejected)
}

func TestAIRewardModel_ScoreWithDimensions_CacheHitWithDimensions(t *testing.T) {
	provider := newMockLLMProvider(`{"score": 0.8}`)
	config := DefaultSelfImprovementConfig()
	config.UseDebateForReward = false

	rm := NewAIRewardModel(provider, nil, config, nil)

	// Pre-populate cache with dimensions
	key := rm.cacheKey("prompt", "response")
	rm.cacheScore(key, 0.85, map[DimensionType]float64{
		DimensionAccuracy:    0.9,
		DimensionRelevance:   0.8,
		DimensionHelpfulness: 0.7,
		DimensionHarmless:    0.6,
		DimensionHonest:      0.5,
		DimensionCoherence:   0.4,
	})

	dims, err := rm.ScoreWithDimensions(context.Background(), "prompt", "response")
	require.NoError(t, err)
	require.NotNil(t, dims)

	// Should return cached dimensions without calling provider
	assert.Equal(t, 0.9, dims[DimensionAccuracy])
	assert.Equal(t, 0, provider.getCallCount())
}
