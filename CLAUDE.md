# CLAUDE.md - SelfImprove Module

## Overview

`digital.vasic.selfimprove` is a Go module for AI self-improvement: reinforcement learning from human feedback (RLHF), reward modeling, preference optimization, and continuous self-refinement.

**Module**: `digital.vasic.selfimprove` (Go 1.24+)

## Build & Test

```bash
go build ./...
go test ./... -count=1 -race
```

## Package Structure

| Package | Purpose |
|---------|---------|
| `selfimprove` | Core: reward models, feedback collection, optimization |
