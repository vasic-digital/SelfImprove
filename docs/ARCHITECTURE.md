# SelfImprove Module Architecture

**Module:** `digital.vasic.selfimprove`

## Overview

AI self-improvement framework implementing reward modelling, RLHF feedback integration, and dimension-weighted scoring optimization.

## Core Components

### Reward Model (`reward.go`)
Multi-dimensional reward scoring with configurable dimension weights. Evaluates LLM outputs across quality axes: accuracy, coherence, helpfulness, safety, creativity.

### Feedback Integration (`feedback.go`)
RLHF (Reinforcement Learning from Human Feedback) integration. Collects, stores, and processes human preference signals to improve model selection and prompt engineering.

### Optimizer (`optimizer.go`)
Optimization engine that uses reward signals and feedback to adjust provider selection weights, prompt templates, and ensemble strategies.

### Integration (`integration.go`)
Connects self-improvement loop with HelixAgent's debate system and provider registry.

### Types (`types.go`)
Shared types: RewardSignal, FeedbackEntry, OptimizationResult, DimensionWeight.

## Improvement Loop

```
LLM Output → Reward Scoring → Feedback Collection
                                     ↓
                              Optimization Engine
                                     ↓
                              Updated Weights/Prompts
                                     ↓
                              Better LLM Output (next iteration)
```

## Package Structure

```
selfimprove/
├── reward.go        # Multi-dimensional reward model
├── feedback.go      # RLHF feedback integration
├── optimizer.go     # Optimization engine
├── integration.go   # HelixAgent integration
└── types.go         # Shared types
```

## Integration

Adapter: `internal/adapters/selfimprove/adapter.go`
