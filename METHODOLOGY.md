# METHODOLOGY.md

## Experimental Design: 153 Controlled Experiments

Research methodology for systematic evaluation of ternary quantization on CNNs.

## Architecture Choice: CIFAR-Adapted Stems

Standard ImageNet stems (7×7 stride-2 + maxpool) destroy spatial information on 32×32 images.

**Solution:** CIFAR-adapted stem (3×3 stride-1, no maxpool) preserves 32×32 → 32×32 resolution.

**Validation:** Recovers +6-17 percentage points on CIFAR-10/100, matching published baselines.

## Phase Structure

### Phase 1: FP32 Baselines (18 experiments)
Establish proper FP32 baselines with CIFAR-adapted stems.
- 2 models × 3 datasets × 3 seeds
- Recipe: 300 epochs, SGD, cosine schedule, warmup 5 epochs
- Augmentation: mixup/smoothing for CIFAR-10/Tiny-ImageNet only

### Phase 2: FP32+KD Control (9 experiments)
Isolate KD benefit from quantization penalty (critical baseline for reviewers).

### Phase 3: BitNet Baselines (18 experiments)
Establish ternary quantization gaps with strong training recipe.

### Phase 4: BitNet + Recipe (18 experiments)
Full recipe: FP32 conv1 + ternary elsewhere (no KD after discovering failure mode).

### Phase 5: Statistical Power (14 experiments)
Increase n=3 to n=10 for near-parity claims on CIFAR-100 and Tiny-ImageNet.

### Phase 6: TTQ Comparison (18 experiments)
Compare against Trained Ternary Quantization under matched conditions.

## Key Findings

1. **Conv1 dominates:** 30-74% of recoverable accuracy despite 0.08% of parameters
2. **KD failure:** Degrades ternary networks (-0.9% to -3.1%), benefits FP32 (+0.9% to +1.6%)
3. **Recipe effectiveness:** FP32 conv1 achieves 1.0% gap on CIFAR-10 without KD

## Result Aggregation Pipeline

```bash
# Aggregate 153 experiments → CSV
uv run python -m analysis.aggregate_results

# Generate paper tables (LaTeX)
uv run python -m analysis.generate_tables

# Generate paper figures (PDF)
uv run python -m analysis.generate_figures

# Compile paper
cd paper && make
```

All tables and figures are programmatically generated from `results/processed/aggregated.csv`.
