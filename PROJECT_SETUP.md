# PROJECT_SETUP.md

## Project: BitNet CNN Ternary Quantization Research

Research project studying BitNet b1.58 (1.58-bit ternary quantization) applied to standard CNN architectures.

## Quick Start

```bash
# Setup environment
uv sync

# Run single experiment
uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset cifar10 --bit-version

# Generate paper artifacts
uv run python -m analysis.aggregate_results
uv run python -m analysis.generate_tables
uv run python -m analysis.generate_figures
```

## Project Structure

- `experiments/` - Training scripts (train.py, train_kd.py, sweep.py)
- `bitnet/` - BitLinear layer implementation
- `analysis/` - Result aggregation and paper artifact generation
- `results/` - Experiment results (results.json + config.json per experiment)
- `paper/` - TMLR paper source (LaTeX)

## Expected Baselines (CIFAR-adapted stem)

- CIFAR-10: ~93% (ResNet-18), ~93.5% (ResNet-50)
- CIFAR-100: ~76% (ResNet-18), ~78% (ResNet-50)
- Tiny-ImageNet: ~62% (ResNet-18), ~65% (ResNet-50)

## Reproducibility

All experiments use CIFAR-adapted stems (3×3 stride-1, no maxpool) for 32×32 and 64×64 images.

See `EXPERIMENTS_REFERENCE.sh` for full experiment commands or `reproduce.sh` for validation workflow.
