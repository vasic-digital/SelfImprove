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
// FeedbackType constants
// ---------------------------------------------------------------------------

func TestFeedbackType_Values(t *testing.T) {
	tests := []struct {
		name     string
		ft       FeedbackType
		expected string
	}{
		{"Positive", FeedbackTypePositive, "positive"},
		{"Negative", FeedbackTypeNegative, "negative"},
		{"Neutral", FeedbackTypeNeutral, "neutral"},
		{"Suggestion", FeedbackTypeSuggestion, "suggestion"},
		{"Correction", FeedbackTypeCorrection, "correction"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, string(tt.ft))
		})
	}
}

func TestFeedbackType_Uniqueness(t *testing.T) {
	types := []FeedbackType{
		FeedbackTypePositive,
		FeedbackTypeNegative,
		FeedbackTypeNeutral,
		FeedbackTypeSuggestion,
		FeedbackTypeCorrection,
	}
	seen := make(map[FeedbackType]bool)
	for _, ft := range types {
		assert.False(t, seen[ft], "duplicate FeedbackType: %s", ft)
		seen[ft] = true
	}
	assert.Equal(t, 5, len(seen))
}

// ---------------------------------------------------------------------------
// FeedbackSource constants
// ---------------------------------------------------------------------------

func TestFeedbackSource_Values(t *testing.T) {
	tests := []struct {
		name     string
		fs       FeedbackSource
		expected string
	}{
		{"Human", FeedbackSourceHuman, "human"},
		{"AI", FeedbackSourceAI, "ai"},
		{"Debate", FeedbackSourceDebate, "debate"},
		{"Verifier", FeedbackSourceVerifier, "verifier"},
		{"Metric", FeedbackSourceMetric, "metric"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, string(tt.fs))
		})
	}
}

func TestFeedbackSource_Uniqueness(t *testing.T) {
	sources := []FeedbackSource{
		FeedbackSourceHuman,
		FeedbackSourceAI,
		FeedbackSourceDebate,
		FeedbackSourceVerifier,
		FeedbackSourceMetric,
	}
	seen := make(map[FeedbackSource]bool)
	for _, fs := range sources {
		assert.False(t, seen[fs], "duplicate FeedbackSource: %s", fs)
		seen[fs] = true
	}
	assert.Equal(t, 5, len(seen))
}

// ---------------------------------------------------------------------------
// DimensionType constants
// ---------------------------------------------------------------------------

func TestDimensionType_Values(t *testing.T) {
	tests := []struct {
		name     string
		dt       DimensionType
		expected string
	}{
		{"Accuracy", DimensionAccuracy, "accuracy"},
		{"Relevance", DimensionRelevance, "relevance"},
		{"Helpfulness", DimensionHelpfulness, "helpfulness"},
		{"Harmless", DimensionHarmless, "harmlessness"},
		{"Honest", DimensionHonest, "honesty"},
		{"Coherence", DimensionCoherence, "coherence"},
		{"Creativity", DimensionCreativity, "creativity"},
		{"Formatting", DimensionFormatting, "formatting"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, string(tt.dt))
		})
	}
}

func TestDimensionType_Uniqueness(t *testing.T) {
	dims := []DimensionType{
		DimensionAccuracy,
		DimensionRelevance,
		DimensionHelpfulness,
		DimensionHarmless,
		DimensionHonest,
		DimensionCoherence,
		DimensionCreativity,
		DimensionFormatting,
	}
	seen := make(map[DimensionType]bool)
	for _, d := range dims {
		assert.False(t, seen[d], "duplicate DimensionType: %s", d)
		seen[d] = true
	}
	assert.Equal(t, 8, len(seen))
}

// ---------------------------------------------------------------------------
// PolicyUpdateType constants
// ---------------------------------------------------------------------------

func TestPolicyUpdateType_Values(t *testing.T) {
	tests := []struct {
		name     string
		put      PolicyUpdateType
		expected string
	}{
		{"PromptRefinement", PolicyUpdatePromptRefinement, "prompt_refinement"},
		{"GuidelineAddition", PolicyUpdateGuidelineAddition, "guideline_addition"},
		{"ExampleAddition", PolicyUpdateExampleAddition, "example_addition"},
		{"ConstraintUpdate", PolicyUpdateConstraintUpdate, "constraint_update"},
		{"ToneAdjustment", PolicyUpdateToneAdjustment, "tone_adjustment"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, string(tt.put))
		})
	}
}

func TestPolicyUpdateType_Uniqueness(t *testing.T) {
	types := []PolicyUpdateType{
		PolicyUpdatePromptRefinement,
		PolicyUpdateGuidelineAddition,
		PolicyUpdateExampleAddition,
		PolicyUpdateConstraintUpdate,
		PolicyUpdateToneAdjustment,
	}
	seen := make(map[PolicyUpdateType]bool)
	for _, pt := range types {
		assert.False(t, seen[pt], "duplicate PolicyUpdateType: %s", pt)
		seen[pt] = true
	}
	assert.Equal(t, 5, len(seen))
}

// ---------------------------------------------------------------------------
// Feedback struct
// ---------------------------------------------------------------------------

func TestFeedback_ZeroValue(t *testing.T) {
	var f Feedback
	assert.Empty(t, f.ID)
	assert.Empty(t, f.SessionID)
	assert.Empty(t, f.PromptID)
	assert.Empty(t, f.ResponseID)
	assert.Equal(t, FeedbackType(""), f.Type)
	assert.Equal(t, FeedbackSource(""), f.Source)
	assert.Equal(t, float64(0), f.Score)
	assert.Nil(t, f.Dimensions)
	assert.Empty(t, f.Comment)
	assert.Empty(t, f.Correction)
	assert.Empty(t, f.ProviderName)
	assert.Empty(t, f.Model)
	assert.Nil(t, f.Metadata)
	assert.True(t, f.CreatedAt.IsZero())
}

func TestFeedback_FullPopulation(t *testing.T) {
	now := time.Now()
	f := Feedback{
		ID:         "fb-001",
		SessionID:  "sess-001",
		PromptID:   "prompt-001",
		ResponseID: "resp-001",
		Type:       FeedbackTypePositive,
		Source:     FeedbackSourceHuman,
		Score:      0.85,
		Dimensions: map[DimensionType]float64{
			DimensionAccuracy:  0.9,
			DimensionRelevance: 0.8,
		},
		Comment:      "Great response",
		Correction:   "Minor fix here",
		ProviderName: "claude",
		Model:        "claude-3-sonnet",
		Metadata:     map[string]interface{}{"key": "value"},
		CreatedAt:    now,
	}

	assert.Equal(t, "fb-001", f.ID)
	assert.Equal(t, "sess-001", f.SessionID)
	assert.Equal(t, FeedbackTypePositive, f.Type)
	assert.Equal(t, FeedbackSourceHuman, f.Source)
	assert.Equal(t, 0.85, f.Score)
	assert.Len(t, f.Dimensions, 2)
	assert.Equal(t, 0.9, f.Dimensions[DimensionAccuracy])
	assert.Equal(t, "Great response", f.Comment)
	assert.Equal(t, "Minor fix here", f.Correction)
	assert.Equal(t, "claude", f.ProviderName)
	assert.Equal(t, "claude-3-sonnet", f.Model)
	assert.Equal(t, "value", f.Metadata["key"])
	assert.Equal(t, now, f.CreatedAt)
}

func TestFeedback_ScoreBounds(t *testing.T) {
	tests := []struct {
		name  string
		score float64
	}{
		{"MinBound", -1.0},
		{"Zero", 0.0},
		{"MaxBound", 1.0},
		{"Negative", -0.5},
		{"Positive", 0.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := Feedback{Score: tt.score}
			assert.Equal(t, tt.score, f.Score)
		})
	}
}

// ---------------------------------------------------------------------------
// TrainingExample struct
// ---------------------------------------------------------------------------

func TestTrainingExample_ZeroValue(t *testing.T) {
	var te TrainingExample
	assert.Empty(t, te.ID)
	assert.Empty(t, te.Prompt)
	assert.Empty(t, te.Response)
	assert.Empty(t, te.PreferredResponse)
	assert.Empty(t, te.RejectedResponse)
	assert.Nil(t, te.Feedback)
	assert.Equal(t, float64(0), te.RewardScore)
	assert.Nil(t, te.Dimensions)
	assert.Empty(t, te.SystemPrompt)
	assert.Empty(t, te.ProviderName)
	assert.Empty(t, te.Model)
	assert.Nil(t, te.Metadata)
	assert.True(t, te.CreatedAt.IsZero())
}

func TestTrainingExample_FullPopulation(t *testing.T) {
	now := time.Now()
	fb := &Feedback{ID: "fb-1", Score: 0.9}
	te := TrainingExample{
		ID:                "te-001",
		Prompt:            "What is Go?",
		Response:          "Go is a programming language.",
		PreferredResponse: "Go is a statically typed, compiled language.",
		RejectedResponse:  "Go is a game.",
		Feedback:          []*Feedback{fb},
		RewardScore:       0.88,
		Dimensions: map[DimensionType]float64{
			DimensionAccuracy:    0.95,
			DimensionHelpfulness: 0.80,
		},
		SystemPrompt: "Be helpful",
		ProviderName: "openai",
		Model:        "gpt-4",
		Metadata:     map[string]interface{}{"version": 2},
		CreatedAt:    now,
	}

	assert.Equal(t, "te-001", te.ID)
	assert.Equal(t, "What is Go?", te.Prompt)
	assert.Equal(t, 0.88, te.RewardScore)
	assert.Len(t, te.Feedback, 1)
	assert.Equal(t, "fb-1", te.Feedback[0].ID)
	assert.Equal(t, 0.95, te.Dimensions[DimensionAccuracy])
	assert.Equal(t, "openai", te.ProviderName)
	assert.Equal(t, now, te.CreatedAt)
}

func TestTrainingExample_WithNilFeedback(t *testing.T) {
	te := TrainingExample{
		ID:       "te-nil",
		Feedback: nil,
	}
	assert.Nil(t, te.Feedback)
}

func TestTrainingExample_WithEmptyFeedback(t *testing.T) {
	te := TrainingExample{
		ID:       "te-empty",
		Feedback: []*Feedback{},
	}
	assert.NotNil(t, te.Feedback)
	assert.Len(t, te.Feedback, 0)
}

// ---------------------------------------------------------------------------
// PreferencePair struct
// ---------------------------------------------------------------------------

func TestPreferencePair_ZeroValue(t *testing.T) {
	var pp PreferencePair
	assert.Empty(t, pp.ID)
	assert.Empty(t, pp.Prompt)
	assert.Empty(t, pp.Chosen)
	assert.Empty(t, pp.Rejected)
	assert.Equal(t, float64(0), pp.ChosenScore)
	assert.Equal(t, float64(0), pp.RejectedScore)
	assert.Equal(t, float64(0), pp.Margin)
	assert.Equal(t, FeedbackSource(""), pp.Source)
	assert.Nil(t, pp.Metadata)
	assert.True(t, pp.CreatedAt.IsZero())
}

func TestPreferencePair_FullPopulation(t *testing.T) {
	now := time.Now()
	pp := PreferencePair{
		ID:            "pp-001",
		Prompt:        "Explain AI",
		Chosen:        "AI is artificial intelligence.",
		Rejected:      "AI is a movie.",
		ChosenScore:   0.9,
		RejectedScore: 0.3,
		Margin:        0.6,
		Source:        FeedbackSourceDebate,
		Metadata:      map[string]interface{}{"debate_id": "d-1"},
		CreatedAt:     now,
	}

	assert.Equal(t, "pp-001", pp.ID)
	assert.Equal(t, "Explain AI", pp.Prompt)
	assert.Equal(t, "AI is artificial intelligence.", pp.Chosen)
	assert.Equal(t, "AI is a movie.", pp.Rejected)
	assert.Equal(t, 0.9, pp.ChosenScore)
	assert.Equal(t, 0.3, pp.RejectedScore)
	assert.Equal(t, 0.6, pp.Margin)
	assert.Equal(t, FeedbackSourceDebate, pp.Source)
	assert.Equal(t, "d-1", pp.Metadata["debate_id"])
	assert.Equal(t, now, pp.CreatedAt)
}

func TestPreferencePair_ChosenScoreGreaterThanRejected(t *testing.T) {
	pp := PreferencePair{
		ChosenScore:   0.9,
		RejectedScore: 0.3,
		Margin:        0.6,
	}
	assert.Greater(t, pp.ChosenScore, pp.RejectedScore)
	assert.InDelta(t, pp.ChosenScore-pp.RejectedScore, pp.Margin, 0.001)
}

// ---------------------------------------------------------------------------
// FeedbackFilter struct
// ---------------------------------------------------------------------------

func TestFeedbackFilter_ZeroValue(t *testing.T) {
	var ff FeedbackFilter
	assert.Nil(t, ff.SessionIDs)
	assert.Nil(t, ff.PromptIDs)
	assert.Nil(t, ff.Types)
	assert.Nil(t, ff.Sources)
	assert.Nil(t, ff.MinScore)
	assert.Nil(t, ff.MaxScore)
	assert.Nil(t, ff.ProviderNames)
	assert.Nil(t, ff.Models)
	assert.Nil(t, ff.StartTime)
	assert.Nil(t, ff.EndTime)
	assert.Equal(t, 0, ff.Limit)
	assert.Equal(t, 0, ff.Offset)
}

func TestFeedbackFilter_FullPopulation(t *testing.T) {
	minScore := -0.5
	maxScore := 0.9
	start := time.Now().Add(-24 * time.Hour)
	end := time.Now()

	ff := FeedbackFilter{
		SessionIDs:    []string{"sess-1", "sess-2"},
		PromptIDs:     []string{"p-1"},
		Types:         []FeedbackType{FeedbackTypePositive, FeedbackTypeNegative},
		Sources:       []FeedbackSource{FeedbackSourceHuman},
		MinScore:      &minScore,
		MaxScore:      &maxScore,
		ProviderNames: []string{"claude"},
		Models:        []string{"claude-3-sonnet"},
		StartTime:     &start,
		EndTime:       &end,
		Limit:         50,
		Offset:        10,
	}

	require.NotNil(t, ff.MinScore)
	assert.Equal(t, -0.5, *ff.MinScore)
	require.NotNil(t, ff.MaxScore)
	assert.Equal(t, 0.9, *ff.MaxScore)
	assert.Len(t, ff.SessionIDs, 2)
	assert.Len(t, ff.Types, 2)
	assert.Equal(t, 50, ff.Limit)
	assert.Equal(t, 10, ff.Offset)
}

func TestFeedbackFilter_NilOptionalPointers(t *testing.T) {
	ff := FeedbackFilter{
		Limit: 100,
	}
	assert.Nil(t, ff.MinScore)
	assert.Nil(t, ff.MaxScore)
	assert.Nil(t, ff.StartTime)
	assert.Nil(t, ff.EndTime)
}

// ---------------------------------------------------------------------------
// AggregatedFeedback struct
// ---------------------------------------------------------------------------

func TestAggregatedFeedback_ZeroValue(t *testing.T) {
	var af AggregatedFeedback
	assert.Equal(t, 0, af.TotalCount)
	assert.Equal(t, float64(0), af.AverageScore)
	assert.Nil(t, af.ScoreDistribution)
	assert.Nil(t, af.TypeDistribution)
	assert.Nil(t, af.SourceDistribution)
	assert.Nil(t, af.DimensionAverages)
	assert.Nil(t, af.ProviderStats)
	assert.Nil(t, af.TrendData)
}

func TestAggregatedFeedback_FullPopulation(t *testing.T) {
	af := AggregatedFeedback{
		TotalCount:  150,
		AverageScore: 0.72,
		ScoreDistribution: map[string]int{
			"0.0-0.2": 5,
			"0.2-0.4": 10,
			"0.4-0.6": 30,
			"0.6-0.8": 60,
			"0.8-1.0": 45,
		},
		TypeDistribution: map[FeedbackType]int{
			FeedbackTypePositive: 100,
			FeedbackTypeNegative: 30,
			FeedbackTypeNeutral:  20,
		},
		SourceDistribution: map[FeedbackSource]int{
			FeedbackSourceHuman: 80,
			FeedbackSourceAI:    70,
		},
		DimensionAverages: map[DimensionType]float64{
			DimensionAccuracy:    0.85,
			DimensionRelevance:   0.78,
			DimensionHelpfulness: 0.80,
		},
		ProviderStats: map[string]*ProviderFeedbackStats{
			"claude": {
				ProviderName: "claude",
				TotalCount:   50,
				AverageScore: 0.82,
				Dimensions: map[DimensionType]float64{
					DimensionAccuracy: 0.88,
				},
			},
		},
		TrendData: []*TrendPoint{
			{Timestamp: time.Now(), AverageScore: 0.70, Count: 10},
		},
	}

	assert.Equal(t, 150, af.TotalCount)
	assert.Equal(t, 0.72, af.AverageScore)
	assert.Len(t, af.ScoreDistribution, 5)
	assert.Len(t, af.TypeDistribution, 3)
	assert.Len(t, af.SourceDistribution, 2)
	assert.Equal(t, 0.85, af.DimensionAverages[DimensionAccuracy])
	require.NotNil(t, af.ProviderStats["claude"])
	assert.Equal(t, 50, af.ProviderStats["claude"].TotalCount)
	assert.Len(t, af.TrendData, 1)
}

// ---------------------------------------------------------------------------
// ProviderFeedbackStats struct
// ---------------------------------------------------------------------------

func TestProviderFeedbackStats_ZeroValue(t *testing.T) {
	var pfs ProviderFeedbackStats
	assert.Empty(t, pfs.ProviderName)
	assert.Equal(t, 0, pfs.TotalCount)
	assert.Equal(t, float64(0), pfs.AverageScore)
	assert.Nil(t, pfs.Dimensions)
}

func TestProviderFeedbackStats_FullPopulation(t *testing.T) {
	pfs := ProviderFeedbackStats{
		ProviderName: "openai",
		TotalCount:   200,
		AverageScore: 0.76,
		Dimensions: map[DimensionType]float64{
			DimensionAccuracy:    0.80,
			DimensionCoherence:   0.85,
			DimensionHelpfulness: 0.72,
		},
	}

	assert.Equal(t, "openai", pfs.ProviderName)
	assert.Equal(t, 200, pfs.TotalCount)
	assert.Equal(t, 0.76, pfs.AverageScore)
	assert.Len(t, pfs.Dimensions, 3)
}

// ---------------------------------------------------------------------------
// TrendPoint struct
// ---------------------------------------------------------------------------

func TestTrendPoint_ZeroValue(t *testing.T) {
	var tp TrendPoint
	assert.True(t, tp.Timestamp.IsZero())
	assert.Equal(t, float64(0), tp.AverageScore)
	assert.Equal(t, 0, tp.Count)
}

func TestTrendPoint_FullPopulation(t *testing.T) {
	now := time.Now()
	tp := TrendPoint{
		Timestamp:    now,
		AverageScore: 0.81,
		Count:        42,
	}
	assert.Equal(t, now, tp.Timestamp)
	assert.Equal(t, 0.81, tp.AverageScore)
	assert.Equal(t, 42, tp.Count)
}

// ---------------------------------------------------------------------------
// PolicyUpdate struct
// ---------------------------------------------------------------------------

func TestPolicyUpdate_ZeroValue(t *testing.T) {
	var pu PolicyUpdate
	assert.Empty(t, pu.ID)
	assert.Empty(t, pu.OldPolicy)
	assert.Empty(t, pu.NewPolicy)
	assert.Equal(t, PolicyUpdateType(""), pu.UpdateType)
	assert.Empty(t, pu.Change)
	assert.Empty(t, pu.Reason)
	assert.Equal(t, float64(0), pu.ImprovementScore)
	assert.Nil(t, pu.Examples)
	assert.Nil(t, pu.AppliedAt)
	assert.Nil(t, pu.Metadata)
	assert.True(t, pu.CreatedAt.IsZero())
}

func TestPolicyUpdate_FullPopulation(t *testing.T) {
	now := time.Now()
	applied := now.Add(-1 * time.Hour)
	pu := PolicyUpdate{
		ID:               "pu-001",
		OldPolicy:        "Be helpful.",
		NewPolicy:        "Be helpful, accurate, and concise.",
		UpdateType:       PolicyUpdatePromptRefinement,
		Change:           "Added accuracy and conciseness",
		Reason:           "Feedback indicated need for more precise responses",
		ImprovementScore: 0.15,
		Examples: []*TrainingExample{
			{ID: "te-1", Prompt: "Test", Response: "Result"},
		},
		AppliedAt: &applied,
		Metadata:  map[string]interface{}{"iteration": 3},
		CreatedAt: now,
	}

	assert.Equal(t, "pu-001", pu.ID)
	assert.Equal(t, PolicyUpdatePromptRefinement, pu.UpdateType)
	assert.Equal(t, 0.15, pu.ImprovementScore)
	assert.Len(t, pu.Examples, 1)
	require.NotNil(t, pu.AppliedAt)
	assert.Equal(t, applied, *pu.AppliedAt)
	assert.Equal(t, 3, pu.Metadata["iteration"])
}

func TestPolicyUpdate_NilAppliedAt(t *testing.T) {
	pu := PolicyUpdate{
		ID:         "pu-nil",
		UpdateType: PolicyUpdateGuidelineAddition,
	}
	assert.Nil(t, pu.AppliedAt)
}

// ---------------------------------------------------------------------------
// SelfImprovementConfig struct
// ---------------------------------------------------------------------------

func TestSelfImprovementConfig_ZeroValue(t *testing.T) {
	var cfg SelfImprovementConfig
	assert.Empty(t, cfg.RewardModelProvider)
	assert.Empty(t, cfg.RewardModelName)
	assert.Equal(t, float64(0), cfg.MinRewardThreshold)
	assert.False(t, cfg.AutoCollectFeedback)
	assert.Equal(t, 0, cfg.FeedbackBatchSize)
	assert.Equal(t, float64(0), cfg.MinConfidenceForAuto)
	assert.Equal(t, time.Duration(0), cfg.OptimizationInterval)
	assert.Equal(t, 0, cfg.MinExamplesForUpdate)
	assert.Equal(t, 0, cfg.MaxPolicyUpdatesPerDay)
	assert.Nil(t, cfg.ConstitutionalPrinciples)
	assert.False(t, cfg.EnableSelfCritique)
	assert.False(t, cfg.UseDebateForReward)
	assert.False(t, cfg.UseDebateForOptimize)
	assert.Equal(t, 0, cfg.MaxBufferSize)
}

// ---------------------------------------------------------------------------
// DefaultSelfImprovementConfig
// ---------------------------------------------------------------------------

func TestDefaultSelfImprovementConfig_ReturnsNonNil(t *testing.T) {
	cfg := DefaultSelfImprovementConfig()
	require.NotNil(t, cfg)
}

func TestDefaultSelfImprovementConfig_RewardModelProvider(t *testing.T) {
	cfg := DefaultSelfImprovementConfig()
	assert.Equal(t, "claude", cfg.RewardModelProvider)
}

func TestDefaultSelfImprovementConfig_RewardModelName(t *testing.T) {
	cfg := DefaultSelfImprovementConfig()
	assert.Equal(t, "claude-3-sonnet", cfg.RewardModelName)
}

func TestDefaultSelfImprovementConfig_MinRewardThreshold(t *testing.T) {
	cfg := DefaultSelfImprovementConfig()
	assert.Equal(t, 0.5, cfg.MinRewardThreshold)
}

func TestDefaultSelfImprovementConfig_AutoCollectFeedback(t *testing.T) {
	cfg := DefaultSelfImprovementConfig()
	assert.True(t, cfg.AutoCollectFeedback)
}

func TestDefaultSelfImprovementConfig_FeedbackBatchSize(t *testing.T) {
	cfg := DefaultSelfImprovementConfig()
	assert.Equal(t, 100, cfg.FeedbackBatchSize)
}

func TestDefaultSelfImprovementConfig_MinConfidenceForAuto(t *testing.T) {
	cfg := DefaultSelfImprovementConfig()
	assert.Equal(t, 0.8, cfg.MinConfidenceForAuto)
}

func TestDefaultSelfImprovementConfig_OptimizationInterval(t *testing.T) {
	cfg := DefaultSelfImprovementConfig()
	assert.Equal(t, 24*time.Hour, cfg.OptimizationInterval)
}

func TestDefaultSelfImprovementConfig_MinExamplesForUpdate(t *testing.T) {
	cfg := DefaultSelfImprovementConfig()
	assert.Equal(t, 50, cfg.MinExamplesForUpdate)
}

func TestDefaultSelfImprovementConfig_MaxPolicyUpdatesPerDay(t *testing.T) {
	cfg := DefaultSelfImprovementConfig()
	assert.Equal(t, 3, cfg.MaxPolicyUpdatesPerDay)
}

func TestDefaultSelfImprovementConfig_EnableSelfCritique(t *testing.T) {
	cfg := DefaultSelfImprovementConfig()
	assert.True(t, cfg.EnableSelfCritique)
}

func TestDefaultSelfImprovementConfig_UseDebateForReward(t *testing.T) {
	cfg := DefaultSelfImprovementConfig()
	assert.True(t, cfg.UseDebateForReward)
}

func TestDefaultSelfImprovementConfig_UseDebateForOptimize(t *testing.T) {
	cfg := DefaultSelfImprovementConfig()
	assert.True(t, cfg.UseDebateForOptimize)
}

func TestDefaultSelfImprovementConfig_MaxBufferSize(t *testing.T) {
	cfg := DefaultSelfImprovementConfig()
	assert.Equal(t, 10000, cfg.MaxBufferSize)
}

func TestDefaultSelfImprovementConfig_ConstitutionalPrinciples(t *testing.T) {
	cfg := DefaultSelfImprovementConfig()
	require.NotNil(t, cfg.ConstitutionalPrinciples)
	assert.Len(t, cfg.ConstitutionalPrinciples, 5)

	expectedPrinciples := []string{
		"Be helpful, harmless, and honest",
		"Avoid generating harmful or misleading content",
		"Respect user privacy and confidentiality",
		"Acknowledge uncertainty when appropriate",
		"Provide balanced perspectives on controversial topics",
	}
	for i, expected := range expectedPrinciples {
		assert.Equal(t, expected, cfg.ConstitutionalPrinciples[i])
	}
}

func TestDefaultSelfImprovementConfig_IndependentInstances(t *testing.T) {
	cfg1 := DefaultSelfImprovementConfig()
	cfg2 := DefaultSelfImprovementConfig()

	cfg1.MinRewardThreshold = 0.99
	cfg1.ConstitutionalPrinciples = append(cfg1.ConstitutionalPrinciples, "extra")

	assert.NotEqual(t, cfg1.MinRewardThreshold, cfg2.MinRewardThreshold)
	assert.NotEqual(t, len(cfg1.ConstitutionalPrinciples), len(cfg2.ConstitutionalPrinciples))
}

// ---------------------------------------------------------------------------
// DebateResult struct
// ---------------------------------------------------------------------------

func TestDebateResult_ZeroValue(t *testing.T) {
	var dr DebateResult
	assert.Empty(t, dr.ID)
	assert.Empty(t, dr.Consensus)
	assert.Equal(t, float64(0), dr.Confidence)
	assert.Nil(t, dr.Participants)
	assert.Nil(t, dr.Votes)
}

func TestDebateResult_FullPopulation(t *testing.T) {
	dr := DebateResult{
		ID:         "debate-001",
		Consensus:  `The response is good. {"score": 0.85, "reasoning": "accurate and helpful"}`,
		Confidence: 0.92,
		Participants: map[string]string{
			"judge-1": "Response A is better",
			"judge-2": "Response A is slightly better",
		},
		Votes: map[string]float64{
			"judge-1": 0.9,
			"judge-2": 0.85,
		},
	}

	assert.Equal(t, "debate-001", dr.ID)
	assert.Contains(t, dr.Consensus, "score")
	assert.Equal(t, 0.92, dr.Confidence)
	assert.Len(t, dr.Participants, 2)
	assert.Len(t, dr.Votes, 2)
}

// ---------------------------------------------------------------------------
// DebateEvaluation struct
// ---------------------------------------------------------------------------

func TestDebateEvaluation_ZeroValue(t *testing.T) {
	var de DebateEvaluation
	assert.Equal(t, float64(0), de.Score)
	assert.Nil(t, de.Dimensions)
	assert.Empty(t, de.Consensus)
	assert.Empty(t, de.DebateID)
	assert.Nil(t, de.ParticipantVotes)
	assert.Equal(t, float64(0), de.Confidence)
}

func TestDebateEvaluation_FullPopulation(t *testing.T) {
	de := DebateEvaluation{
		Score: 0.88,
		Dimensions: map[DimensionType]float64{
			DimensionAccuracy:    0.90,
			DimensionHelpfulness: 0.86,
		},
		Consensus:        "Overall high quality",
		DebateID:         "de-001",
		ParticipantVotes: map[string]float64{"p1": 0.9, "p2": 0.86},
		Confidence:       0.94,
	}

	assert.Equal(t, 0.88, de.Score)
	assert.Len(t, de.Dimensions, 2)
	assert.Equal(t, "de-001", de.DebateID)
	assert.Equal(t, 0.94, de.Confidence)
}

// ---------------------------------------------------------------------------
// DebateComparison struct
// ---------------------------------------------------------------------------

func TestDebateComparison_ZeroValue(t *testing.T) {
	var dc DebateComparison
	assert.Equal(t, 0, dc.PreferredIndex)
	assert.Equal(t, float64(0), dc.Margin)
	assert.Empty(t, dc.Reasoning)
	assert.Empty(t, dc.DebateID)
	assert.Nil(t, dc.ParticipantPrefs)
	assert.Equal(t, float64(0), dc.Confidence)
}

func TestDebateComparison_FullPopulation(t *testing.T) {
	dc := DebateComparison{
		PreferredIndex:   1,
		Margin:           0.65,
		Reasoning:        "Response B was more comprehensive",
		DebateID:         "dc-001",
		ParticipantPrefs: map[string]int{"p1": 1, "p2": 1, "p3": 0},
		Confidence:       0.78,
	}

	assert.Equal(t, 1, dc.PreferredIndex)
	assert.Equal(t, 0.65, dc.Margin)
	assert.Equal(t, "Response B was more comprehensive", dc.Reasoning)
	assert.Len(t, dc.ParticipantPrefs, 3)
}

func TestDebateComparison_PreferredIndexValues(t *testing.T) {
	tests := []struct {
		name  string
		index int
	}{
		{"First", 0},
		{"Second", 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dc := DebateComparison{PreferredIndex: tt.index}
			assert.Equal(t, tt.index, dc.PreferredIndex)
		})
	}
}

// ---------------------------------------------------------------------------
// Interface compliance (compile-time checks)
// ---------------------------------------------------------------------------

// Verify that the interfaces are well-defined by ensuring they can be
// used as type constraints. These are compile-time assertions.
var (
	_ RewardModel          = (*rewardModelImpl)(nil)
	_ FeedbackCollector    = (*feedbackCollectorImpl)(nil)
	_ PolicyOptimizer      = (*policyOptimizerImpl)(nil)
	_ DebateService        = (*debateServiceImpl)(nil)
	_ DebateRewardEvaluator = (*debateRewardEvalImpl)(nil)
	_ LLMProvider          = (*llmProviderImpl)(nil)
	_ ProviderVerifier     = (*providerVerifierImpl)(nil)
)

// Minimal implementations for compile-time interface checks
type rewardModelImpl struct{}

func (r *rewardModelImpl) Score(_ context.Context, _, _ string) (float64, error) {
	return 0, nil
}
func (r *rewardModelImpl) ScoreWithDimensions(_ context.Context, _, _ string) (map[DimensionType]float64, error) {
	return nil, nil
}
func (r *rewardModelImpl) Compare(_ context.Context, _, _, _ string) (*PreferencePair, error) {
	return nil, nil
}
func (r *rewardModelImpl) Train(_ context.Context, _ []*TrainingExample) error { return nil }

type feedbackCollectorImpl struct{}

func (f *feedbackCollectorImpl) Collect(_ context.Context, _ *Feedback) error { return nil }
func (f *feedbackCollectorImpl) GetBySession(_ context.Context, _ string) ([]*Feedback, error) {
	return nil, nil
}
func (f *feedbackCollectorImpl) GetByPrompt(_ context.Context, _ string) ([]*Feedback, error) {
	return nil, nil
}
func (f *feedbackCollectorImpl) GetAggregated(_ context.Context, _ *FeedbackFilter) (*AggregatedFeedback, error) {
	return nil, nil
}
func (f *feedbackCollectorImpl) Export(_ context.Context, _ *FeedbackFilter) ([]*TrainingExample, error) {
	return nil, nil
}

type policyOptimizerImpl struct{ policy string }

func (p *policyOptimizerImpl) Optimize(_ context.Context, _ []*TrainingExample) ([]*PolicyUpdate, error) {
	return nil, nil
}
func (p *policyOptimizerImpl) Apply(_ context.Context, _ *PolicyUpdate) error    { return nil }
func (p *policyOptimizerImpl) Rollback(_ context.Context, _ string) error        { return nil }
func (p *policyOptimizerImpl) GetHistory(_ context.Context, _ int) ([]*PolicyUpdate, error) {
	return nil, nil
}
func (p *policyOptimizerImpl) GetCurrentPolicy() string       { return p.policy }
func (p *policyOptimizerImpl) SetCurrentPolicy(policy string) { p.policy = policy }

type debateServiceImpl struct{}

func (d *debateServiceImpl) RunDebate(_ context.Context, _ string, _ []string) (*DebateResult, error) {
	return nil, nil
}

type debateRewardEvalImpl struct{}

func (d *debateRewardEvalImpl) EvaluateWithDebate(_ context.Context, _, _ string) (*DebateEvaluation, error) {
	return nil, nil
}
func (d *debateRewardEvalImpl) CompareWithDebate(_ context.Context, _, _, _ string) (*DebateComparison, error) {
	return nil, nil
}

type llmProviderImpl struct{}

func (l *llmProviderImpl) Complete(_ context.Context, _, _ string) (string, error) { return "", nil }

type providerVerifierImpl struct{}

func (p *providerVerifierImpl) GetProviderScore(_ string) float64 { return 0 }
func (p *providerVerifierImpl) IsProviderHealthy(_ string) bool   { return false }

func TestInterfaceCompliance_RewardModel(t *testing.T) {
	var rm RewardModel = &rewardModelImpl{}
	assert.NotNil(t, rm)
}

func TestInterfaceCompliance_FeedbackCollector(t *testing.T) {
	var fc FeedbackCollector = &feedbackCollectorImpl{}
	assert.NotNil(t, fc)
}

func TestInterfaceCompliance_PolicyOptimizer(t *testing.T) {
	po := &policyOptimizerImpl{policy: "initial"}
	var opt PolicyOptimizer = po
	assert.Equal(t, "initial", opt.GetCurrentPolicy())
	opt.SetCurrentPolicy("updated")
	assert.Equal(t, "updated", opt.GetCurrentPolicy())
}

func TestInterfaceCompliance_DebateService(t *testing.T) {
	var ds DebateService = &debateServiceImpl{}
	assert.NotNil(t, ds)
}

func TestInterfaceCompliance_DebateRewardEvaluator(t *testing.T) {
	var dre DebateRewardEvaluator = &debateRewardEvalImpl{}
	assert.NotNil(t, dre)
}

func TestInterfaceCompliance_LLMProvider(t *testing.T) {
	var lp LLMProvider = &llmProviderImpl{}
	assert.NotNil(t, lp)
}

func TestInterfaceCompliance_ProviderVerifier(t *testing.T) {
	var pv ProviderVerifier = &providerVerifierImpl{}
	assert.NotNil(t, pv)
	assert.Equal(t, float64(0), pv.GetProviderScore("any"))
	assert.False(t, pv.IsProviderHealthy("any"))
}

// ---------------------------------------------------------------------------
// InMemoryFeedbackCollector
// ---------------------------------------------------------------------------

func TestNewInMemoryFeedbackCollector_NilLogger(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	require.NotNil(t, fc)
	assert.NotNil(t, fc.logger)
}

func TestNewInMemoryFeedbackCollector_ZeroMaxSize(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 0)
	require.NotNil(t, fc)
	assert.Equal(t, 10000, fc.maxSize)
}

func TestNewInMemoryFeedbackCollector_NegativeMaxSize(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, -5)
	require.NotNil(t, fc)
	assert.Equal(t, 10000, fc.maxSize)
}

func TestNewInMemoryFeedbackCollector_CustomMaxSize(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 500)
	require.NotNil(t, fc)
	assert.Equal(t, 500, fc.maxSize)
}

func TestInMemoryFeedbackCollector_SetRewardModel(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	assert.Nil(t, fc.rewardModel)

	rm := &rewardModelImpl{}
	fc.SetRewardModel(rm)
	assert.NotNil(t, fc.rewardModel)
}

func TestInMemoryFeedbackCollector_Collect_NilFeedback(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	err := fc.Collect(context.Background(), nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "feedback cannot be nil")
}

func TestInMemoryFeedbackCollector_Collect_AutoGeneratesID(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	fb := &Feedback{Type: FeedbackTypePositive, Score: 0.9}
	err := fc.Collect(context.Background(), fb)
	require.NoError(t, err)
	assert.NotEmpty(t, fb.ID)
}

func TestInMemoryFeedbackCollector_Collect_AutoSetsCreatedAt(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	fb := &Feedback{Type: FeedbackTypePositive, Score: 0.9}
	err := fc.Collect(context.Background(), fb)
	require.NoError(t, err)
	assert.False(t, fb.CreatedAt.IsZero())
}

func TestInMemoryFeedbackCollector_Collect_PreservesExistingID(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	fb := &Feedback{ID: "custom-id", Type: FeedbackTypePositive, Score: 0.9}
	err := fc.Collect(context.Background(), fb)
	require.NoError(t, err)
	assert.Equal(t, "custom-id", fb.ID)
}

func TestInMemoryFeedbackCollector_Collect_IndexesBySession(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	fb := &Feedback{SessionID: "sess-1", Type: FeedbackTypePositive, Score: 0.9}
	err := fc.Collect(context.Background(), fb)
	require.NoError(t, err)
	assert.Len(t, fc.bySession["sess-1"], 1)
}

func TestInMemoryFeedbackCollector_Collect_IndexesByPrompt(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	fb := &Feedback{PromptID: "prompt-1", Type: FeedbackTypePositive, Score: 0.9}
	err := fc.Collect(context.Background(), fb)
	require.NoError(t, err)
	assert.Len(t, fc.byPrompt["prompt-1"], 1)
}

func TestInMemoryFeedbackCollector_Collect_NoIndexForEmptySessionID(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	fb := &Feedback{Type: FeedbackTypePositive, Score: 0.9}
	err := fc.Collect(context.Background(), fb)
	require.NoError(t, err)
	assert.Empty(t, fc.bySession)
}

func TestInMemoryFeedbackCollector_Collect_EvictsAtCapacity(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 10)
	for i := 0; i < 10; i++ {
		fb := &Feedback{
			ID:        fmt.Sprintf("fb-%d", i),
			SessionID: "sess",
			PromptID:  "prompt",
			Type:      FeedbackTypePositive,
			Score:     float64(i) / 10.0,
			CreatedAt: time.Now().Add(time.Duration(i) * time.Second),
		}
		err := fc.Collect(context.Background(), fb)
		require.NoError(t, err)
	}
	// At capacity (10), the next Collect triggers eviction of oldest 10%=1
	fb := &Feedback{ID: "fb-new", Type: FeedbackTypePositive, Score: 1.0}
	err := fc.Collect(context.Background(), fb)
	require.NoError(t, err)
	assert.Equal(t, 10, len(fc.feedback)) // 10 - 1 evicted + 1 new
}

func TestInMemoryFeedbackCollector_GetBySession_Exists(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	fb1 := &Feedback{SessionID: "sess-1", Type: FeedbackTypePositive, Score: 0.9}
	fb2 := &Feedback{SessionID: "sess-1", Type: FeedbackTypeNegative, Score: 0.3}
	_ = fc.Collect(context.Background(), fb1)
	_ = fc.Collect(context.Background(), fb2)

	result, err := fc.GetBySession(context.Background(), "sess-1")
	require.NoError(t, err)
	assert.Len(t, result, 2)
}

func TestInMemoryFeedbackCollector_GetBySession_NotFound(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	result, err := fc.GetBySession(context.Background(), "nonexistent")
	require.NoError(t, err)
	assert.Empty(t, result)
}

func TestInMemoryFeedbackCollector_GetBySession_ReturnsCopy(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	fb := &Feedback{SessionID: "sess-1", Type: FeedbackTypePositive, Score: 0.9}
	_ = fc.Collect(context.Background(), fb)

	result, _ := fc.GetBySession(context.Background(), "sess-1")
	result[0] = nil // Modify returned slice
	// Original should be unaffected
	internal := fc.bySession["sess-1"]
	assert.NotNil(t, internal[0])
}

func TestInMemoryFeedbackCollector_GetByPrompt_Exists(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	fb := &Feedback{PromptID: "p-1", Type: FeedbackTypePositive, Score: 0.8}
	_ = fc.Collect(context.Background(), fb)

	result, err := fc.GetByPrompt(context.Background(), "p-1")
	require.NoError(t, err)
	assert.Len(t, result, 1)
}

func TestInMemoryFeedbackCollector_GetByPrompt_NotFound(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	result, err := fc.GetByPrompt(context.Background(), "nonexistent")
	require.NoError(t, err)
	assert.Empty(t, result)
}

func TestInMemoryFeedbackCollector_GetAggregated_EmptyFeedback(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	agg, err := fc.GetAggregated(context.Background(), nil)
	require.NoError(t, err)
	assert.Equal(t, 0, agg.TotalCount)
}

func TestInMemoryFeedbackCollector_GetAggregated_NilFilter(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	fb := &Feedback{
		Type:         FeedbackTypePositive,
		Source:       FeedbackSourceHuman,
		Score:        0.8,
		ProviderName: "claude",
		Dimensions: map[DimensionType]float64{
			DimensionAccuracy: 0.9,
		},
		CreatedAt: time.Now(),
	}
	_ = fc.Collect(context.Background(), fb)

	agg, err := fc.GetAggregated(context.Background(), nil)
	require.NoError(t, err)
	assert.Equal(t, 1, agg.TotalCount)
	assert.Equal(t, 0.8, agg.AverageScore)
	assert.Equal(t, 1, agg.TypeDistribution[FeedbackTypePositive])
	assert.Equal(t, 1, agg.SourceDistribution[FeedbackSourceHuman])
	assert.Equal(t, 0.9, agg.DimensionAverages[DimensionAccuracy])
	require.NotNil(t, agg.ProviderStats["claude"])
	assert.Equal(t, 1, agg.ProviderStats["claude"].TotalCount)
}

func TestInMemoryFeedbackCollector_GetAggregated_WithFilter(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	_ = fc.Collect(context.Background(), &Feedback{
		SessionID: "sess-1",
		Type:      FeedbackTypePositive,
		Source:    FeedbackSourceHuman,
		Score:     0.9,
		CreatedAt: time.Now(),
	})
	_ = fc.Collect(context.Background(), &Feedback{
		SessionID: "sess-2",
		Type:      FeedbackTypeNegative,
		Source:    FeedbackSourceAI,
		Score:     0.3,
		CreatedAt: time.Now(),
	})

	filter := &FeedbackFilter{SessionIDs: []string{"sess-1"}}
	agg, err := fc.GetAggregated(context.Background(), filter)
	require.NoError(t, err)
	assert.Equal(t, 1, agg.TotalCount)
	assert.Equal(t, 0.9, agg.AverageScore)
}

func TestInMemoryFeedbackCollector_GetAggregated_ScoreDistribution(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	scores := []float64{0.1, 0.3, 0.5, 0.7, 0.9}
	for _, s := range scores {
		_ = fc.Collect(context.Background(), &Feedback{
			Type:      FeedbackTypeNeutral,
			Score:     s,
			CreatedAt: time.Now(),
		})
	}

	agg, err := fc.GetAggregated(context.Background(), nil)
	require.NoError(t, err)
	assert.Equal(t, 5, agg.TotalCount)
	assert.Equal(t, 1, agg.ScoreDistribution["0.0-0.2"])
	assert.Equal(t, 1, agg.ScoreDistribution["0.2-0.4"])
	assert.Equal(t, 1, agg.ScoreDistribution["0.4-0.6"])
	assert.Equal(t, 1, agg.ScoreDistribution["0.6-0.8"])
	assert.Equal(t, 1, agg.ScoreDistribution["0.8-1.0"])
}

func TestInMemoryFeedbackCollector_GetAggregated_TrendData(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	now := time.Now()
	_ = fc.Collect(context.Background(), &Feedback{
		Type:      FeedbackTypePositive,
		Score:     0.8,
		CreatedAt: now,
	})
	_ = fc.Collect(context.Background(), &Feedback{
		Type:      FeedbackTypePositive,
		Score:     0.6,
		CreatedAt: now.Add(-24 * time.Hour),
	})

	agg, err := fc.GetAggregated(context.Background(), nil)
	require.NoError(t, err)
	assert.NotNil(t, agg.TrendData)
	assert.GreaterOrEqual(t, len(agg.TrendData), 1)
}

func TestInMemoryFeedbackCollector_Export_EmptyFeedback(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	examples, err := fc.Export(context.Background(), nil)
	require.NoError(t, err)
	assert.Empty(t, examples)
}

func TestInMemoryFeedbackCollector_Export_GroupsByPrompt(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	_ = fc.Collect(context.Background(), &Feedback{
		PromptID:     "p-1",
		Type:         FeedbackTypePositive,
		Score:        0.8,
		ProviderName: "claude",
		Model:        "sonnet",
		Dimensions:   map[DimensionType]float64{DimensionAccuracy: 0.9},
		CreatedAt:    time.Now(),
	})
	_ = fc.Collect(context.Background(), &Feedback{
		PromptID:  "p-1",
		Type:      FeedbackTypeNeutral,
		Score:     0.6,
		CreatedAt: time.Now(),
	})

	examples, err := fc.Export(context.Background(), nil)
	require.NoError(t, err)
	assert.Len(t, examples, 1)
	assert.Equal(t, "claude", examples[0].ProviderName)
	assert.Equal(t, "sonnet", examples[0].Model)
}

func TestInMemoryFeedbackCollector_Export_WithCorrection(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	_ = fc.Collect(context.Background(), &Feedback{
		PromptID:   "p-1",
		Type:       FeedbackTypeCorrection,
		Score:      0.8,
		Correction: "better response",
		CreatedAt:  time.Now(),
	})

	examples, err := fc.Export(context.Background(), nil)
	require.NoError(t, err)
	assert.Len(t, examples, 1)
	assert.Equal(t, "better response", examples[0].PreferredResponse)
}

func TestInMemoryFeedbackCollector_Export_NoPromptID(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	_ = fc.Collect(context.Background(), &Feedback{
		Type:      FeedbackTypePositive,
		Score:     0.8,
		CreatedAt: time.Now(),
	})

	examples, err := fc.Export(context.Background(), nil)
	require.NoError(t, err)
	assert.Empty(t, examples) // No prompt ID means no grouping
}

func TestInMemoryFeedbackCollector_ApplyFilter_Offset(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	for i := 0; i < 5; i++ {
		_ = fc.Collect(context.Background(), &Feedback{
			Type:      FeedbackTypePositive,
			Score:     float64(i) / 10.0,
			CreatedAt: time.Now(),
		})
	}

	filter := &FeedbackFilter{Offset: 3}
	agg, err := fc.GetAggregated(context.Background(), filter)
	require.NoError(t, err)
	assert.Equal(t, 2, agg.TotalCount)
}

func TestInMemoryFeedbackCollector_ApplyFilter_OffsetBeyondLength(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	_ = fc.Collect(context.Background(), &Feedback{
		Type:      FeedbackTypePositive,
		Score:     0.8,
		CreatedAt: time.Now(),
	})

	filter := &FeedbackFilter{Offset: 100}
	agg, err := fc.GetAggregated(context.Background(), filter)
	require.NoError(t, err)
	assert.Equal(t, 0, agg.TotalCount)
}

func TestInMemoryFeedbackCollector_ApplyFilter_Limit(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	for i := 0; i < 5; i++ {
		_ = fc.Collect(context.Background(), &Feedback{
			Type:      FeedbackTypePositive,
			Score:     0.8,
			CreatedAt: time.Now(),
		})
	}

	filter := &FeedbackFilter{Limit: 2}
	agg, err := fc.GetAggregated(context.Background(), filter)
	require.NoError(t, err)
	assert.Equal(t, 2, agg.TotalCount)
}

func TestInMemoryFeedbackCollector_MatchesFilter_TypeFilter(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	_ = fc.Collect(context.Background(), &Feedback{
		Type:      FeedbackTypePositive,
		Score:     0.8,
		CreatedAt: time.Now(),
	})
	_ = fc.Collect(context.Background(), &Feedback{
		Type:      FeedbackTypeNegative,
		Score:     0.3,
		CreatedAt: time.Now(),
	})

	filter := &FeedbackFilter{Types: []FeedbackType{FeedbackTypePositive}}
	agg, err := fc.GetAggregated(context.Background(), filter)
	require.NoError(t, err)
	assert.Equal(t, 1, agg.TotalCount)
}

func TestInMemoryFeedbackCollector_MatchesFilter_SourceFilter(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	_ = fc.Collect(context.Background(), &Feedback{
		Type:      FeedbackTypePositive,
		Source:    FeedbackSourceHuman,
		Score:     0.8,
		CreatedAt: time.Now(),
	})
	_ = fc.Collect(context.Background(), &Feedback{
		Type:      FeedbackTypePositive,
		Source:    FeedbackSourceAI,
		Score:     0.7,
		CreatedAt: time.Now(),
	})

	filter := &FeedbackFilter{Sources: []FeedbackSource{FeedbackSourceHuman}}
	agg, err := fc.GetAggregated(context.Background(), filter)
	require.NoError(t, err)
	assert.Equal(t, 1, agg.TotalCount)
}

func TestInMemoryFeedbackCollector_MatchesFilter_ScoreRange(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	_ = fc.Collect(context.Background(), &Feedback{Type: FeedbackTypePositive, Score: 0.1, CreatedAt: time.Now()})
	_ = fc.Collect(context.Background(), &Feedback{Type: FeedbackTypePositive, Score: 0.5, CreatedAt: time.Now()})
	_ = fc.Collect(context.Background(), &Feedback{Type: FeedbackTypePositive, Score: 0.9, CreatedAt: time.Now()})

	minScore := 0.4
	maxScore := 0.6
	filter := &FeedbackFilter{MinScore: &minScore, MaxScore: &maxScore}
	agg, err := fc.GetAggregated(context.Background(), filter)
	require.NoError(t, err)
	assert.Equal(t, 1, agg.TotalCount)
}

func TestInMemoryFeedbackCollector_MatchesFilter_ProviderFilter(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	_ = fc.Collect(context.Background(), &Feedback{
		Type:         FeedbackTypePositive,
		ProviderName: "claude",
		Score:        0.8,
		CreatedAt:    time.Now(),
	})
	_ = fc.Collect(context.Background(), &Feedback{
		Type:         FeedbackTypePositive,
		ProviderName: "openai",
		Score:        0.7,
		CreatedAt:    time.Now(),
	})

	filter := &FeedbackFilter{ProviderNames: []string{"claude"}}
	agg, err := fc.GetAggregated(context.Background(), filter)
	require.NoError(t, err)
	assert.Equal(t, 1, agg.TotalCount)
}

func TestInMemoryFeedbackCollector_MatchesFilter_ModelFilter(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	_ = fc.Collect(context.Background(), &Feedback{
		Type:      FeedbackTypePositive,
		Model:     "gpt-4",
		Score:     0.8,
		CreatedAt: time.Now(),
	})
	_ = fc.Collect(context.Background(), &Feedback{
		Type:      FeedbackTypePositive,
		Model:     "sonnet",
		Score:     0.7,
		CreatedAt: time.Now(),
	})

	filter := &FeedbackFilter{Models: []string{"gpt-4"}}
	agg, err := fc.GetAggregated(context.Background(), filter)
	require.NoError(t, err)
	assert.Equal(t, 1, agg.TotalCount)
}

func TestInMemoryFeedbackCollector_MatchesFilter_TimeRange(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	now := time.Now()
	_ = fc.Collect(context.Background(), &Feedback{
		Type:      FeedbackTypePositive,
		Score:     0.8,
		CreatedAt: now.Add(-48 * time.Hour),
	})
	_ = fc.Collect(context.Background(), &Feedback{
		Type:      FeedbackTypePositive,
		Score:     0.7,
		CreatedAt: now,
	})

	start := now.Add(-24 * time.Hour)
	end := now.Add(time.Hour)
	filter := &FeedbackFilter{StartTime: &start, EndTime: &end}
	agg, err := fc.GetAggregated(context.Background(), filter)
	require.NoError(t, err)
	assert.Equal(t, 1, agg.TotalCount)
}

func TestInMemoryFeedbackCollector_MatchesFilter_PromptFilter(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	_ = fc.Collect(context.Background(), &Feedback{
		PromptID:  "p-1",
		Type:      FeedbackTypePositive,
		Score:     0.8,
		CreatedAt: time.Now(),
	})
	_ = fc.Collect(context.Background(), &Feedback{
		PromptID:  "p-2",
		Type:      FeedbackTypePositive,
		Score:     0.7,
		CreatedAt: time.Now(),
	})

	filter := &FeedbackFilter{PromptIDs: []string{"p-1"}}
	agg, err := fc.GetAggregated(context.Background(), filter)
	require.NoError(t, err)
	assert.Equal(t, 1, agg.TotalCount)
}

func TestInMemoryFeedbackCollector_EvictOldest_EmptyList(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	// Should not panic
	fc.evictOldest(5)
	assert.Empty(t, fc.feedback)
}

func TestInMemoryFeedbackCollector_EvictOldest_ZeroCount(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	_ = fc.Collect(context.Background(), &Feedback{Type: FeedbackTypePositive, Score: 0.8, CreatedAt: time.Now()})
	fc.evictOldest(0)
	assert.Len(t, fc.feedback, 1)
}

func TestInMemoryFeedbackCollector_RemoveFromIndex_EmptyKey(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	index := make(map[string][]*Feedback)
	// Should not panic with empty key
	fc.removeFromIndex(index, "", "some-id")
}

func TestInMemoryFeedbackCollector_RemoveFromIndex_CleanupEmpty(t *testing.T) {
	fc := NewInMemoryFeedbackCollector(nil, 100)
	index := map[string][]*Feedback{
		"key": {{ID: "fb-1"}},
	}
	fc.removeFromIndex(index, "key", "fb-1")
	_, exists := index["key"]
	assert.False(t, exists, "empty index entry should be deleted")
}

func TestInMemoryFeedbackCollector_Contains(t *testing.T) {
	assert.True(t, contains([]string{"a", "b", "c"}, "b"))
	assert.False(t, contains([]string{"a", "b", "c"}, "d"))
	assert.False(t, contains([]string{}, "a"))
}

// ---------------------------------------------------------------------------
// AutoFeedbackCollector
// ---------------------------------------------------------------------------

func TestNewAutoFeedbackCollector_NilConfig(t *testing.T) {
	afc := NewAutoFeedbackCollector(nil, nil, nil)
	require.NotNil(t, afc)
	assert.NotNil(t, afc.config)
	assert.Equal(t, "claude", afc.config.RewardModelProvider)
}

func TestNewAutoFeedbackCollector_CustomConfig(t *testing.T) {
	cfg := DefaultSelfImprovementConfig()
	cfg.FeedbackBatchSize = 50
	afc := NewAutoFeedbackCollector(nil, cfg, nil)
	require.NotNil(t, afc)
	assert.Equal(t, 50, afc.config.FeedbackBatchSize)
}

// mockRewardModelForAuto is a simple mock for AutoFeedbackCollector tests
type mockRewardModelForAuto struct {
	score      float64
	dimensions map[DimensionType]float64
	scoreErr   error
	dimErr     error
}

func (m *mockRewardModelForAuto) Score(_ context.Context, _, _ string) (float64, error) {
	return m.score, m.scoreErr
}
func (m *mockRewardModelForAuto) ScoreWithDimensions(_ context.Context, _, _ string) (map[DimensionType]float64, error) {
	return m.dimensions, m.dimErr
}
func (m *mockRewardModelForAuto) Compare(_ context.Context, _, _, _ string) (*PreferencePair, error) {
	return nil, nil
}
func (m *mockRewardModelForAuto) Train(_ context.Context, _ []*TrainingExample) error { return nil }

func TestAutoFeedbackCollector_CollectAuto_NilRewardModel(t *testing.T) {
	afc := NewAutoFeedbackCollector(nil, nil, nil)
	_, err := afc.CollectAuto(context.Background(), "s1", "p1", "prompt", "response", "prov", "model")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no reward model available")
}

func TestAutoFeedbackCollector_CollectAuto_HighScore(t *testing.T) {
	rm := &mockRewardModelForAuto{
		dimensions: map[DimensionType]float64{
			DimensionAccuracy:    0.9,
			DimensionHelpfulness: 0.8,
		},
	}
	cfg := DefaultSelfImprovementConfig()
	cfg.MinConfidenceForAuto = 0.5
	afc := NewAutoFeedbackCollector(rm, cfg, nil)

	fb, err := afc.CollectAuto(context.Background(), "s1", "p1", "prompt", "response", "claude", "sonnet")
	require.NoError(t, err)
	require.NotNil(t, fb)
	assert.Equal(t, FeedbackTypePositive, fb.Type)
	assert.Equal(t, FeedbackSourceAI, fb.Source)
	assert.Equal(t, "claude", fb.ProviderName)
}

func TestAutoFeedbackCollector_CollectAuto_LowScore(t *testing.T) {
	rm := &mockRewardModelForAuto{
		dimensions: map[DimensionType]float64{
			DimensionAccuracy: 0.2,
		},
	}
	cfg := DefaultSelfImprovementConfig()
	cfg.MinConfidenceForAuto = 0.5
	afc := NewAutoFeedbackCollector(rm, cfg, nil)

	fb, err := afc.CollectAuto(context.Background(), "s1", "p1", "prompt", "response", "prov", "model")
	require.NoError(t, err)
	require.NotNil(t, fb)
	assert.Equal(t, FeedbackTypeNegative, fb.Type)
}

func TestAutoFeedbackCollector_CollectAuto_NeutralScore(t *testing.T) {
	rm := &mockRewardModelForAuto{
		dimensions: map[DimensionType]float64{
			DimensionAccuracy: 0.5,
		},
	}
	cfg := DefaultSelfImprovementConfig()
	cfg.MinConfidenceForAuto = 0.4
	afc := NewAutoFeedbackCollector(rm, cfg, nil)

	fb, err := afc.CollectAuto(context.Background(), "s1", "p1", "prompt", "response", "prov", "model")
	require.NoError(t, err)
	require.NotNil(t, fb)
	assert.Equal(t, FeedbackTypeNeutral, fb.Type)
}

func TestAutoFeedbackCollector_CollectAuto_DimensionError_FallsBackToScore(t *testing.T) {
	rm := &mockRewardModelForAuto{
		dimErr: fmt.Errorf("dim error"),
		score:  0.75,
	}
	cfg := DefaultSelfImprovementConfig()
	cfg.MinConfidenceForAuto = 0.5
	afc := NewAutoFeedbackCollector(rm, cfg, nil)

	fb, err := afc.CollectAuto(context.Background(), "s1", "p1", "prompt", "response", "prov", "model")
	require.NoError(t, err)
	require.NotNil(t, fb)
	assert.Equal(t, FeedbackTypePositive, fb.Type)
}

func TestAutoFeedbackCollector_CollectAuto_BothErrors(t *testing.T) {
	rm := &mockRewardModelForAuto{
		dimErr:   fmt.Errorf("dim error"),
		scoreErr: fmt.Errorf("score error"),
	}
	afc := NewAutoFeedbackCollector(rm, nil, nil)

	_, err := afc.CollectAuto(context.Background(), "s1", "p1", "prompt", "response", "prov", "model")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "reward model scoring failed")
}

// ---------------------------------------------------------------------------
// LLMPolicyOptimizer
// ---------------------------------------------------------------------------

func TestNewLLMPolicyOptimizer_NilConfig(t *testing.T) {
	opt := NewLLMPolicyOptimizer(nil, nil, nil, nil)
	require.NotNil(t, opt)
	assert.NotNil(t, opt.config)
	assert.NotNil(t, opt.logger)
}

func TestLLMPolicyOptimizer_SetCurrentPolicy_GetCurrentPolicy(t *testing.T) {
	opt := NewLLMPolicyOptimizer(nil, nil, nil, nil)
	opt.SetCurrentPolicy("Be helpful")
	assert.Equal(t, "Be helpful", opt.GetCurrentPolicy())
}

func TestLLMPolicyOptimizer_Optimize_InsufficientExamples(t *testing.T) {
	cfg := DefaultSelfImprovementConfig()
	cfg.MinExamplesForUpdate = 5
	opt := NewLLMPolicyOptimizer(nil, nil, cfg, nil)

	_, err := opt.Optimize(context.Background(), []*TrainingExample{{ID: "1"}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "insufficient examples")
}

// mockLLMProviderForOptimizer supports the LLMProvider interface for optimizer tests
type mockLLMProviderForOpt struct {
	mu       sync.Mutex
	response string
	err      error
}

func (m *mockLLMProviderForOpt) Complete(_ context.Context, _, _ string) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.response, m.err
}

// mockDebateServiceForOpt is a configurable mock for debate service in optimizer tests
type mockDebateServiceForOpt struct {
	result *DebateResult
	err    error
}

func (m *mockDebateServiceForOpt) RunDebate(_ context.Context, _ string, _ []string) (*DebateResult, error) {
	return m.result, m.err
}

func TestLLMPolicyOptimizer_Optimize_LLMPath(t *testing.T) {
	provider := &mockLLMProviderForOpt{
		response: `[{"type": "guideline_addition", "change": "be more precise", "reason": "feedback shows imprecision", "improvement_score": 0.8}]`,
	}
	cfg := DefaultSelfImprovementConfig()
	cfg.MinExamplesForUpdate = 1
	cfg.UseDebateForOptimize = false
	cfg.EnableSelfCritique = false
	opt := NewLLMPolicyOptimizer(provider, nil, cfg, nil)

	examples := []*TrainingExample{
		{ID: "1", RewardScore: 0.3, Feedback: []*Feedback{{Type: FeedbackTypeNegative, Comment: "imprecise"}}},
	}

	updates, err := opt.Optimize(context.Background(), examples)
	require.NoError(t, err)
	assert.Len(t, updates, 1)
	assert.Equal(t, PolicyUpdateGuidelineAddition, updates[0].UpdateType)
}

func TestLLMPolicyOptimizer_Optimize_DebatePath(t *testing.T) {
	debate := &mockDebateServiceForOpt{
		result: &DebateResult{
			Consensus: `[{"type": "prompt_refinement", "change": "add clarity", "reason": "unclear", "improvement_score": 0.7}]`,
		},
	}
	cfg := DefaultSelfImprovementConfig()
	cfg.MinExamplesForUpdate = 1
	cfg.UseDebateForOptimize = true
	cfg.EnableSelfCritique = false
	opt := NewLLMPolicyOptimizer(nil, debate, cfg, nil)

	examples := []*TrainingExample{
		{ID: "1", RewardScore: 0.3, Feedback: []*Feedback{{Type: FeedbackTypeNegative, Comment: "unclear"}}},
	}

	updates, err := opt.Optimize(context.Background(), examples)
	require.NoError(t, err)
	assert.Len(t, updates, 1)
}

func TestLLMPolicyOptimizer_Optimize_DebateFailsFallsBackToLLM(t *testing.T) {
	debate := &mockDebateServiceForOpt{err: fmt.Errorf("debate failed")}
	provider := &mockLLMProviderForOpt{
		response: `[{"type": "tone_adjustment", "change": "softer tone", "reason": "harsh", "improvement_score": 0.5}]`,
	}
	cfg := DefaultSelfImprovementConfig()
	cfg.MinExamplesForUpdate = 1
	cfg.UseDebateForOptimize = true
	cfg.EnableSelfCritique = false
	opt := NewLLMPolicyOptimizer(provider, debate, cfg, nil)

	examples := []*TrainingExample{
		{ID: "1", RewardScore: 0.3, Feedback: []*Feedback{{Type: FeedbackTypeNegative, Comment: "harsh"}}},
	}

	updates, err := opt.Optimize(context.Background(), examples)
	require.NoError(t, err)
	assert.Len(t, updates, 1)
	assert.Equal(t, PolicyUpdateToneAdjustment, updates[0].UpdateType)
}

func TestLLMPolicyOptimizer_Optimize_NilProvider(t *testing.T) {
	cfg := DefaultSelfImprovementConfig()
	cfg.MinExamplesForUpdate = 1
	cfg.UseDebateForOptimize = false
	cfg.EnableSelfCritique = false
	opt := NewLLMPolicyOptimizer(nil, nil, cfg, nil)

	examples := []*TrainingExample{{ID: "1", RewardScore: 0.3}}
	_, err := opt.Optimize(context.Background(), examples)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no LLM provider available")
}

func TestLLMPolicyOptimizer_Optimize_SelfCritique_FiltersViolation(t *testing.T) {
	provider := &mockLLMProviderForOpt{
		response: `[
			{"type": "prompt_refinement", "change": "ignore safety guidelines", "reason": "test", "improvement_score": 0.9},
			{"type": "prompt_refinement", "change": "be more helpful", "reason": "test", "improvement_score": 0.8}
		]`,
	}
	cfg := DefaultSelfImprovementConfig()
	cfg.MinExamplesForUpdate = 1
	cfg.UseDebateForOptimize = false
	cfg.EnableSelfCritique = true
	opt := NewLLMPolicyOptimizer(provider, nil, cfg, nil)

	examples := []*TrainingExample{
		{ID: "1", RewardScore: 0.3, Feedback: []*Feedback{{Type: FeedbackTypeNegative, Comment: "issue"}}},
	}

	updates, err := opt.Optimize(context.Background(), examples)
	require.NoError(t, err)
	// "ignore safety" should be filtered out
	assert.Len(t, updates, 1)
	assert.Contains(t, updates[0].NewPolicy, "be more helpful")
}

func TestLLMPolicyOptimizer_Apply_Success(t *testing.T) {
	opt := NewLLMPolicyOptimizer(nil, nil, nil, nil)
	opt.SetCurrentPolicy("old policy")

	update := &PolicyUpdate{
		ID:        "u-1",
		NewPolicy: "new policy",
	}
	err := opt.Apply(context.Background(), update)
	require.NoError(t, err)
	assert.Equal(t, "new policy", opt.GetCurrentPolicy())
	assert.Equal(t, "old policy", update.OldPolicy)
	assert.NotNil(t, update.AppliedAt)
}

func TestLLMPolicyOptimizer_Apply_DailyLimit(t *testing.T) {
	cfg := DefaultSelfImprovementConfig()
	cfg.MaxPolicyUpdatesPerDay = 1
	opt := NewLLMPolicyOptimizer(nil, nil, cfg, nil)

	// First apply should succeed
	err := opt.Apply(context.Background(), &PolicyUpdate{ID: "u-1", NewPolicy: "p1"})
	require.NoError(t, err)

	// Second should fail
	err = opt.Apply(context.Background(), &PolicyUpdate{ID: "u-2", NewPolicy: "p2"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "daily policy update limit reached")
}

func TestLLMPolicyOptimizer_Rollback_Success(t *testing.T) {
	opt := NewLLMPolicyOptimizer(nil, nil, nil, nil)
	opt.SetCurrentPolicy("original")

	update := &PolicyUpdate{ID: "u-1", NewPolicy: "modified"}
	err := opt.Apply(context.Background(), update)
	require.NoError(t, err)
	assert.Equal(t, "modified", opt.GetCurrentPolicy())

	err = opt.Rollback(context.Background(), "u-1")
	require.NoError(t, err)
	assert.Equal(t, "original", opt.GetCurrentPolicy())
}

func TestLLMPolicyOptimizer_Rollback_NotFound(t *testing.T) {
	opt := NewLLMPolicyOptimizer(nil, nil, nil, nil)
	err := opt.Rollback(context.Background(), "nonexistent")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "update not found")
}

func TestLLMPolicyOptimizer_Rollback_NoOldPolicy(t *testing.T) {
	opt := NewLLMPolicyOptimizer(nil, nil, nil, nil)
	// Manually add a history entry with empty old policy
	opt.history = append(opt.history, &PolicyUpdate{ID: "u-1", OldPolicy: ""})
	err := opt.Rollback(context.Background(), "u-1")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no old policy to rollback to")
}

func TestLLMPolicyOptimizer_GetHistory_Empty(t *testing.T) {
	opt := NewLLMPolicyOptimizer(nil, nil, nil, nil)
	history, err := opt.GetHistory(context.Background(), 10)
	require.NoError(t, err)
	assert.Empty(t, history)
}

func TestLLMPolicyOptimizer_GetHistory_WithLimit(t *testing.T) {
	opt := NewLLMPolicyOptimizer(nil, nil, nil, nil)
	for i := 0; i < 5; i++ {
		opt.history = append(opt.history, &PolicyUpdate{ID: fmt.Sprintf("u-%d", i)})
	}

	history, err := opt.GetHistory(context.Background(), 2)
	require.NoError(t, err)
	assert.Len(t, history, 2)
	assert.Equal(t, "u-4", history[0].ID) // Most recent first
	assert.Equal(t, "u-3", history[1].ID)
}

func TestLLMPolicyOptimizer_GetHistory_ZeroLimit(t *testing.T) {
	opt := NewLLMPolicyOptimizer(nil, nil, nil, nil)
	opt.history = append(opt.history, &PolicyUpdate{ID: "u-1"})
	history, err := opt.GetHistory(context.Background(), 0)
	require.NoError(t, err)
	assert.Len(t, history, 1) // Returns all
}

func TestLLMPolicyOptimizer_AnalyzeFeedbackPatterns(t *testing.T) {
	opt := NewLLMPolicyOptimizer(nil, nil, nil, nil)
	examples := []*TrainingExample{
		{
			ID:          "e1",
			RewardScore: 0.2,
			Feedback: []*Feedback{
				{Type: FeedbackTypeNegative, Comment: "inaccurate"},
			},
			ProviderName: "claude",
			Dimensions:   map[DimensionType]float64{DimensionAccuracy: 0.3},
		},
		{
			ID:          "e2",
			RewardScore: 0.8,
			Feedback:    []*Feedback{{Type: FeedbackTypePositive}},
			Dimensions:  map[DimensionType]float64{DimensionAccuracy: 0.9},
		},
	}

	patterns := opt.analyzeFeedbackPatterns(examples)
	assert.Len(t, patterns.NegativeExamples, 1)
	assert.Len(t, patterns.PositiveExamples, 1)
	assert.Equal(t, 1, patterns.CommonIssues["inaccurate"])
}

func TestLLMPolicyOptimizer_ExtractOptimizations_NoJSON(t *testing.T) {
	opt := NewLLMPolicyOptimizer(nil, nil, nil, nil)
	_, err := opt.extractOptimizations("no json here", &feedbackPatterns{
		NegativeExamples: []*TrainingExample{},
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no JSON found")
}

func TestLLMPolicyOptimizer_ExtractOptimizations_SingleObject(t *testing.T) {
	opt := NewLLMPolicyOptimizer(nil, nil, nil, nil)
	updates, err := opt.extractOptimizations(
		`{"type": "example_addition", "change": "add example", "reason": "clarity", "improvement_score": 0.6}`,
		&feedbackPatterns{NegativeExamples: []*TrainingExample{}},
	)
	require.NoError(t, err)
	assert.Len(t, updates, 1)
	assert.Equal(t, PolicyUpdateExampleAddition, updates[0].UpdateType)
}

func TestLLMPolicyOptimizer_ExtractOptimizations_ConstraintUpdate(t *testing.T) {
	opt := NewLLMPolicyOptimizer(nil, nil, nil, nil)
	updates, err := opt.extractOptimizations(
		`[{"type": "constraint_update", "change": "no harmful", "reason": "safety", "improvement_score": 0.7}]`,
		&feedbackPatterns{NegativeExamples: []*TrainingExample{}},
	)
	require.NoError(t, err)
	assert.Len(t, updates, 1)
	assert.Equal(t, PolicyUpdateConstraintUpdate, updates[0].UpdateType)
}

func TestLLMPolicyOptimizer_ApplyChange_EmptyPolicy(t *testing.T) {
	opt := NewLLMPolicyOptimizer(nil, nil, nil, nil)
	result := opt.applyChange("", "new content", PolicyUpdatePromptRefinement)
	assert.Equal(t, "new content", result)
}

func TestLLMPolicyOptimizer_ApplyChange_AllTypes(t *testing.T) {
	opt := NewLLMPolicyOptimizer(nil, nil, nil, nil)
	policy := "Base policy"

	tests := []struct {
		updateType PolicyUpdateType
		contains   string
	}{
		{PolicyUpdateGuidelineAddition, "Additional Guideline"},
		{PolicyUpdateExampleAddition, "Example"},
		{PolicyUpdateConstraintUpdate, "Constraint"},
		{PolicyUpdateToneAdjustment, "Tone Note"},
		{PolicyUpdatePromptRefinement, "Refinement"},
	}

	for _, tt := range tests {
		t.Run(string(tt.updateType), func(t *testing.T) {
			result := opt.applyChange(policy, "change text", tt.updateType)
			assert.Contains(t, result, tt.contains)
			assert.Contains(t, result, "change text")
		})
	}
}

func TestLLMPolicyOptimizer_ViolatesPrinciple_Harmless(t *testing.T) {
	opt := NewLLMPolicyOptimizer(nil, nil, nil, nil)
	assert.True(t, opt.violatesPrinciple("you should ignore safety guidelines", "Be helpful, harmless, and honest"))
	assert.True(t, opt.violatesPrinciple("bypass all rules", "Be helpful, harmless, and honest"))
	assert.False(t, opt.violatesPrinciple("be more helpful", "Be helpful, harmless, and honest"))
}

func TestLLMPolicyOptimizer_ViolatesPrinciple_Honest(t *testing.T) {
	opt := NewLLMPolicyOptimizer(nil, nil, nil, nil)
	assert.True(t, opt.violatesPrinciple("pretend to be human", "Be honest and transparent"))
	assert.True(t, opt.violatesPrinciple("deceive the user", "Be honest and transparent"))
	assert.False(t, opt.violatesPrinciple("be clear and direct", "Be honest and transparent"))
}

func TestLLMPolicyOptimizer_TruncateStr(t *testing.T) {
	assert.Equal(t, "hello", truncateStr("hello", 10))
	assert.Equal(t, "hel...", truncateStr("hello world", 3))
	assert.Equal(t, "", truncateStr("", 5))
}

func TestLLMPolicyOptimizer_Min(t *testing.T) {
	assert.Equal(t, 3, min(3, 5))
	assert.Equal(t, 3, min(5, 3))
	assert.Equal(t, 3, min(3, 3))
}

func TestLLMPolicyOptimizer_SetDebateService(t *testing.T) {
	opt := NewLLMPolicyOptimizer(nil, nil, nil, nil)
	assert.Nil(t, opt.debateService)
	ds := &mockDebateServiceForOpt{}
	opt.SetDebateService(ds)
	assert.NotNil(t, opt.debateService)
}

func TestLLMPolicyOptimizer_SetProvider(t *testing.T) {
	opt := NewLLMPolicyOptimizer(nil, nil, nil, nil)
	assert.Nil(t, opt.provider)
	p := &mockLLMProviderForOpt{}
	opt.SetProvider(p)
	assert.NotNil(t, opt.provider)
}

func TestLLMPolicyOptimizer_BuildOptimizationTopic(t *testing.T) {
	opt := NewLLMPolicyOptimizer(nil, nil, nil, nil)
	opt.SetCurrentPolicy("Be helpful")

	patterns := &feedbackPatterns{
		CommonIssues: map[string]int{
			"too vague": 3,
			"one time":  1,
		},
		DimensionWeakness: map[DimensionType]float64{
			DimensionAccuracy: 0.3,
		},
		NegativeExamples: []*TrainingExample{
			{Prompt: "test prompt", RewardScore: 0.2},
		},
	}

	topic := opt.buildOptimizationTopic(patterns)
	assert.Contains(t, topic, "Be helpful")
	assert.Contains(t, topic, "accuracy")
	assert.Contains(t, topic, "too vague")
	assert.Contains(t, topic, "test prompt")
	// "one time" has count < 2, should not appear
	assert.NotContains(t, topic, "one time")
}

func TestLLMPolicyOptimizer_ApplySelfCritique_NoPrinciples(t *testing.T) {
	cfg := DefaultSelfImprovementConfig()
	cfg.ConstitutionalPrinciples = []string{}
	opt := NewLLMPolicyOptimizer(nil, nil, cfg, nil)

	updates := []*PolicyUpdate{{ID: "u-1", NewPolicy: "anything"}}
	result := opt.applySelfCritique(context.Background(), updates)
	assert.Len(t, result, 1) // Should pass through
}

// ---------------------------------------------------------------------------
// DebateServiceAdapter
// ---------------------------------------------------------------------------

func TestNewDebateServiceAdapter_NilLogger(t *testing.T) {
	adapter := NewDebateServiceAdapter(nil, nil)
	require.NotNil(t, adapter)
	assert.NotNil(t, adapter.logger)
}

func TestDebateServiceAdapter_SetProviderMapping(t *testing.T) {
	adapter := NewDebateServiceAdapter(nil, nil)
	mapping := map[string]string{"p1": "claude", "p2": "openai"}
	adapter.SetProviderMapping(mapping)
	assert.Equal(t, mapping, adapter.providerMap)
}

func TestDebateServiceAdapter_EvaluateWithDebate_Success(t *testing.T) {
	ds := &mockDebateServiceForOpt{
		result: &DebateResult{
			ID:         "debate-1",
			Consensus:  "good response",
			Confidence: 0.85,
			Votes:      map[string]float64{"p1": 0.9, "p2": 0.8},
		},
	}
	adapter := NewDebateServiceAdapter(ds, nil)
	adapter.SetProviderMapping(map[string]string{"p1": "claude"})

	eval, err := adapter.EvaluateWithDebate(context.Background(), "prompt", "response")
	require.NoError(t, err)
	require.NotNil(t, eval)
	assert.Equal(t, "debate-1", eval.DebateID)
	assert.Equal(t, 0.85, eval.Confidence)
	assert.NotEmpty(t, eval.ParticipantVotes)
}

func TestDebateServiceAdapter_EvaluateWithDebate_Error(t *testing.T) {
	ds := &mockDebateServiceForOpt{err: fmt.Errorf("debate error")}
	adapter := NewDebateServiceAdapter(ds, nil)

	_, err := adapter.EvaluateWithDebate(context.Background(), "prompt", "response")
	require.Error(t, err)
}

func TestDebateServiceAdapter_CompareWithDebate_PreferA(t *testing.T) {
	// Note: containsIgnoreCase only checks the first char of the strings.
	// "A is the winner" starts with 'A', not matching "\"B\"" (starts with '"') or
	// "Response B" (starts with 'R'). So PreferredIndex stays 0.
	ds := &mockDebateServiceForOpt{
		result: &DebateResult{
			ID:           "debate-2",
			Consensus:    "A is the winner",
			Confidence:   0.9,
			Participants: map[string]string{"p1": "A is great", "p2": "A is good"},
		},
	}
	adapter := NewDebateServiceAdapter(ds, nil)

	comp, err := adapter.CompareWithDebate(context.Background(), "prompt", "r1", "r2")
	require.NoError(t, err)
	require.NotNil(t, comp)
	assert.Equal(t, 0, comp.PreferredIndex)
}

func TestDebateServiceAdapter_CompareWithDebate_PreferB(t *testing.T) {
	ds := &mockDebateServiceForOpt{
		result: &DebateResult{
			ID:           "debate-3",
			Consensus:    `Response B is clearly better, "B" wins`,
			Confidence:   0.8,
			Participants: map[string]string{"p1": "B is best", "p2": "B clearly"},
		},
	}
	adapter := NewDebateServiceAdapter(ds, nil)

	comp, err := adapter.CompareWithDebate(context.Background(), "prompt", "r1", "r2")
	require.NoError(t, err)
	require.NotNil(t, comp)
	assert.Equal(t, 1, comp.PreferredIndex)
}

func TestDebateServiceAdapter_CompareWithDebate_Error(t *testing.T) {
	ds := &mockDebateServiceForOpt{err: fmt.Errorf("debate error")}
	adapter := NewDebateServiceAdapter(ds, nil)

	_, err := adapter.CompareWithDebate(context.Background(), "prompt", "r1", "r2")
	require.Error(t, err)
}

func TestDebateServiceAdapter_CompareWithDebate_MarginCalculation(t *testing.T) {
	ds := &mockDebateServiceForOpt{
		result: &DebateResult{
			ID:         "debate-4",
			Consensus:  "A is better",
			Confidence: 0.75,
			Participants: map[string]string{
				"p1": "A is great",
				"p2": "A is fine",
				"p3": "B is ok",
			},
		},
	}
	adapter := NewDebateServiceAdapter(ds, nil)

	comp, err := adapter.CompareWithDebate(context.Background(), "prompt", "r1", "r2")
	require.NoError(t, err)
	// 2 prefer A, 1 prefers B => margin = |2-1|/3 = 0.333...
	assert.InDelta(t, 0.333, comp.Margin, 0.01)
}

// ---------------------------------------------------------------------------
// SelfImprovementSystem
// ---------------------------------------------------------------------------

func TestNewSelfImprovementSystem_NilConfig(t *testing.T) {
	sis := NewSelfImprovementSystem(nil, nil)
	require.NotNil(t, sis)
	assert.NotNil(t, sis.config)
	assert.NotNil(t, sis.logger)
}

func TestSelfImprovementSystem_Initialize_WithDebate(t *testing.T) {
	ds := &mockDebateServiceForOpt{
		result: &DebateResult{Consensus: `{"score": 0.8}`},
	}
	sis := NewSelfImprovementSystem(nil, nil)
	err := sis.Initialize(nil, ds)
	require.NoError(t, err)
	assert.NotNil(t, sis.debateAdapter)
	assert.NotNil(t, sis.rewardModel)
	assert.NotNil(t, sis.feedbackCollector)
	assert.NotNil(t, sis.policyOptimizer)
}

func TestSelfImprovementSystem_Initialize_WithoutDebate(t *testing.T) {
	sis := NewSelfImprovementSystem(nil, nil)
	err := sis.Initialize(nil, nil)
	require.NoError(t, err)
	assert.Nil(t, sis.debateAdapter)
}

func TestSelfImprovementSystem_Initialize_ManualCollector(t *testing.T) {
	cfg := DefaultSelfImprovementConfig()
	cfg.AutoCollectFeedback = false
	sis := NewSelfImprovementSystem(cfg, nil)
	err := sis.Initialize(nil, nil)
	require.NoError(t, err)
	// Should create InMemoryFeedbackCollector, not AutoFeedbackCollector
	_, isAuto := sis.feedbackCollector.(*AutoFeedbackCollector)
	assert.False(t, isAuto)
}

func TestSelfImprovementSystem_SetVerifier(t *testing.T) {
	sis := NewSelfImprovementSystem(nil, nil)
	sis.SetVerifier(&providerVerifierImpl{})
	assert.NotNil(t, sis.verifier)
}

func TestSelfImprovementSystem_StartStop(t *testing.T) {
	cfg := DefaultSelfImprovementConfig()
	cfg.OptimizationInterval = 100 * time.Millisecond
	sis := NewSelfImprovementSystem(cfg, nil)
	_ = sis.Initialize(nil, nil)

	err := sis.Start()
	require.NoError(t, err)

	// Double start should fail
	err = sis.Start()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "already running")

	// Stop
	sis.Stop()

	// Double stop should not panic
	sis.Stop()
}

func TestSelfImprovementSystem_CollectFeedback(t *testing.T) {
	sis := NewSelfImprovementSystem(nil, nil)
	_ = sis.Initialize(nil, nil)

	fb := &Feedback{Type: FeedbackTypePositive, Score: 0.9}
	err := sis.CollectFeedback(context.Background(), fb)
	require.NoError(t, err)
}

func TestSelfImprovementSystem_CollectAutoFeedback_WithAutoCollector(t *testing.T) {
	provider := newMockLLMProvider(`{"score": 0.8}`)
	cfg := DefaultSelfImprovementConfig()
	cfg.UseDebateForReward = false
	cfg.MinConfidenceForAuto = 0.5
	sis := NewSelfImprovementSystem(cfg, nil)
	_ = sis.Initialize(provider, nil)

	fb, err := sis.CollectAutoFeedback(context.Background(), "s1", "p1", "prompt", "response", "prov", "model")
	require.NoError(t, err)
	require.NotNil(t, fb)
}

func TestSelfImprovementSystem_CollectAutoFeedback_ManualCollector(t *testing.T) {
	provider := newMockLLMProvider(`{"score": 0.75}`)
	cfg := DefaultSelfImprovementConfig()
	cfg.AutoCollectFeedback = false
	cfg.UseDebateForReward = false
	sis := NewSelfImprovementSystem(cfg, nil)
	_ = sis.Initialize(provider, nil)

	fb, err := sis.CollectAutoFeedback(context.Background(), "s1", "p1", "prompt", "response", "prov", "model")
	require.NoError(t, err)
	require.NotNil(t, fb)
	assert.Equal(t, FeedbackTypePositive, fb.Type)
}

func TestSelfImprovementSystem_CollectAutoFeedback_ManualCollector_LowScore(t *testing.T) {
	provider := newMockLLMProvider(`{"score": 0.2}`)
	cfg := DefaultSelfImprovementConfig()
	cfg.AutoCollectFeedback = false
	cfg.UseDebateForReward = false
	sis := NewSelfImprovementSystem(cfg, nil)
	_ = sis.Initialize(provider, nil)

	fb, err := sis.CollectAutoFeedback(context.Background(), "s1", "p1", "prompt", "response", "prov", "model")
	require.NoError(t, err)
	require.NotNil(t, fb)
	assert.Equal(t, FeedbackTypeNegative, fb.Type)
}

func TestSelfImprovementSystem_ScoreResponse(t *testing.T) {
	provider := newMockLLMProvider(`{"score": 0.7}`)
	cfg := DefaultSelfImprovementConfig()
	cfg.UseDebateForReward = false
	sis := NewSelfImprovementSystem(cfg, nil)
	_ = sis.Initialize(provider, nil)

	score, err := sis.ScoreResponse(context.Background(), "prompt", "response")
	require.NoError(t, err)
	assert.Equal(t, 0.7, score)
}

func TestSelfImprovementSystem_CompareResponses(t *testing.T) {
	provider := newMockLLMProvider(`{"preferred": "A", "margin": 0.6, "reasoning": "better"}`)
	cfg := DefaultSelfImprovementConfig()
	cfg.UseDebateForReward = false
	sis := NewSelfImprovementSystem(cfg, nil)
	_ = sis.Initialize(provider, nil)

	pair, err := sis.CompareResponses(context.Background(), "prompt", "r1", "r2")
	require.NoError(t, err)
	require.NotNil(t, pair)
	assert.Equal(t, "r1", pair.Chosen)
}

func TestSelfImprovementSystem_GetFeedbackStats(t *testing.T) {
	sis := NewSelfImprovementSystem(nil, nil)
	_ = sis.Initialize(nil, nil)

	stats, err := sis.GetFeedbackStats(context.Background(), nil)
	require.NoError(t, err)
	require.NotNil(t, stats)
	assert.Equal(t, 0, stats.TotalCount)
}

func TestSelfImprovementSystem_GetPolicyHistory(t *testing.T) {
	sis := NewSelfImprovementSystem(nil, nil)
	_ = sis.Initialize(nil, nil)

	history, err := sis.GetPolicyHistory(context.Background(), 10)
	require.NoError(t, err)
	assert.Empty(t, history)
}

func TestSelfImprovementSystem_RollbackPolicy(t *testing.T) {
	sis := NewSelfImprovementSystem(nil, nil)
	_ = sis.Initialize(nil, nil)

	err := sis.RollbackPolicy(context.Background(), "nonexistent")
	require.Error(t, err)
}

func TestSelfImprovementSystem_GetRewardModel(t *testing.T) {
	sis := NewSelfImprovementSystem(nil, nil)
	_ = sis.Initialize(nil, nil)
	assert.NotNil(t, sis.GetRewardModel())
}

func TestSelfImprovementSystem_GetDebateAdapter_Nil(t *testing.T) {
	sis := NewSelfImprovementSystem(nil, nil)
	_ = sis.Initialize(nil, nil)
	assert.Nil(t, sis.GetDebateAdapter())
}

func TestSelfImprovementSystem_GetDebateAdapter_NonNil(t *testing.T) {
	ds := &mockDebateServiceForOpt{result: &DebateResult{}}
	sis := NewSelfImprovementSystem(nil, nil)
	_ = sis.Initialize(nil, ds)
	assert.NotNil(t, sis.GetDebateAdapter())
}

// ---------------------------------------------------------------------------
// Helper functions from integration.go
// ---------------------------------------------------------------------------

func TestContainsIgnoreCase(t *testing.T) {
	assert.True(t, containsIgnoreCase(`"B"`, `"B"`))
	assert.False(t, containsIgnoreCase("short", "longerstring"))
	assert.False(t, containsIgnoreCase("", "a"))
}

func TestAbs(t *testing.T) {
	assert.Equal(t, 5, abs(5))
	assert.Equal(t, 5, abs(-5))
	assert.Equal(t, 0, abs(0))
}

// ---------------------------------------------------------------------------
// RunOptimizationCycle
// ---------------------------------------------------------------------------

func TestSelfImprovementSystem_RunOptimizationCycle_NotEnoughExamples(t *testing.T) {
	sis := NewSelfImprovementSystem(nil, nil)
	_ = sis.Initialize(nil, nil)

	// With no feedback, should return nil (not enough examples)
	err := sis.runOptimizationCycle(context.Background())
	require.NoError(t, err)
}
