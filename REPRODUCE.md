# Reproducing Experiments

This document provides high-level guidance for reproducing the experiments in the paper.

## Prerequisites

```bash
# Clone and setup
git clone <repo-url>
cd bitnet
uv sync
```

## Full Experiment List

**See [PROPER_BASELINE_COMMANDS.sh](PROPER_BASELINE_COMMANDS.sh)** for complete, organized experiment commands.

The paper results are from **Wave 1 + Wave 2** experiments (135 baseline + 14 statistical power):
- **Wave 1** (36 experiments): FP32 baselines + BitNet baselines with CIFAR-adapted stem
- **Wave 2** (99 experiments): FP32+KD controls, layer ablations (with/without KD), full recipe

All commands in PROPER_BASELINE_COMMANDS.sh include proper training recipe:
- 300 epochs with 5-epoch warmup
- Mixup (α=0.2) + label smoothing (0.1) for CIFAR-10 and Tiny-ImageNet
- No mixup/smoothing for CIFAR-100 (hurts fine-grained classification)

## Critical Architecture Note

**All CIFAR-10, CIFAR-100, and Tiny-ImageNet experiments use `--use-cifar-stem` flag.**

This modifies the ResNet stem architecture:
- ❌ ImageNet stem: 7×7 stride-2 conv + maxpool (destroys 32×32 spatial info)
- ✅ CIFAR-adapted stem: 3×3 stride-1 conv, no maxpool (preserves spatial resolution)

This is standard practice (kuangliu/pytorch-cifar) and essential for fair comparison. Without it:
- CIFAR-10: ~89% (wrong) vs ~96% (correct)
- CIFAR-100: ~62% (wrong) vs ~79% (correct)

See `paper/research/10_resnet_stem_architecture.md` for detailed analysis.

## Quick Examples

### Single Training Run

```bash
# FP32 baseline (ResNet-18, CIFAR-10)
uv run python -m experiments.train \
  --use-cifar-stem --model resnet18 --dataset cifar10 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
  --mixup-alpha 0.2 --label-smoothing 0.1 --seed 42

# BitNet baseline
uv run python -m experiments.train \
  --use-cifar-stem --model resnet18 --dataset cifar10 --bit-version \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
  --mixup-alpha 0.2 --label-smoothing 0.1 --seed 42

# BitNet + Recipe (keep_conv1 + KD)
uv run python -m experiments.train_kd \
  --use-cifar-stem --model resnet18 --dataset cifar10 \
  --teacher-path results/raw/cifar10/resnet18/std_s42/best_model.pth \
  --ablation keep_conv1 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
  --mixup-alpha 0.2 --label-smoothing 0.1 --seed 42
```

### Sweep Multiple Configurations

```bash
# Use sweep.py for rapid parallel execution
uv run python -m experiments.sweep \
  --models resnet18 resnet50 \
  --datasets cifar10 cifar100 tiny_imagenet \
  --seeds 42 123 456
```

Note: sweep.py automatically adds `--use-cifar-stem` for CIFAR/Tiny-ImageNet datasets.

## Safety Features

Training scripts include safety checks:
- **Auto-generated paths**: Based on configuration (no manual naming)
- **Overwrite protection**: Refuses to run if results exist
- **Force flag**: Use `--force` to intentionally overwrite

Path naming convention:
- Standard: `results/raw/{dataset}/{model}/{version}[_augment][_ablation]_s{seed}`
- KD: `results/raw_kd/{dataset}/{model}/bit_kd[_ablation][_t{temp}][_a{alpha}]_s{seed}`

## Analysis and Figure Generation

```bash
# Aggregate all results from raw/ and raw_kd/
uv run python -m analysis.aggregate_results

# Generate LaTeX tables
uv run python -m analysis.generate_tables

# Generate figures
uv run python -m analysis.generate_figures
```

Outputs:
- `results/processed/aggregated.csv` - All experiment results
- `paper/tmlr/tables/*.tex` - LaTeX tables
- `paper/tmlr/figures/*.{png,pdf}` - Figures for paper

## Expected Results (Wave 1 - CIFAR-adapted stem)

| Experiment | CIFAR-10 | CIFAR-100 | Tiny-ImageNet |
|------------|----------|-----------|---------------|
| FP32 (ResNet-18) | 96.07 ± 0.15% | 79.14 ± 0.11% | 67.04 ± 0.23% |
| BitNet | 94.64 ± 0.20% | 74.93 ± 0.23% | 62.10 ± 0.22% |
| **Gap** | **1.43%** | **4.21%** | **4.95%** |

Wave 2 results (Phase 4 Recipe: keep_conv1 + KD) pending completion.

## Hardware Requirements

- GPU: NVIDIA RTX A6000 (48GB) or equivalent
- Training times:
  - CIFAR-10/100: ~2.5 hours per experiment (300 epochs)
  - Tiny-ImageNet: ~5 hours per experiment (300 epochs)
- Total compute: ~135 experiments × 2.5-5 hrs ≈ 340-675 GPU hours

Note: ResNet-50 + Tiny-ImageNet requires `--batch-size 64` (OOM with 128).

## Troubleshooting

**Low CIFAR-100 accuracy (~62% instead of ~79%)**
- Missing `--use-cifar-stem` flag
- Check that stem uses 3×3 stride-1 conv, not 7×7 stride-2

**Out of Memory (OOM)**
- ResNet-50 + Tiny-ImageNet: Use `--batch-size 64`
- BitNet uses MORE memory during training than FP32 (FP32 gradients + STE overhead)

**Results don't match paper**
- Ensure using proper training recipe (300 epochs, warmup, mixup, label smoothing)
- Verify teacher model exists at specified path for KD experiments
- Check augmentation flags: CIFAR-10/Tiny-IN get mixup, CIFAR-100 doesn't

## Directory Structure

```
results/
├── raw/           # Standard training (FP32, BitNet, ablations without KD)
│   ├── cifar10/
│   │   └── resnet18/
│   │       ├── std_s42/           # FP32 baseline seed 42
│   │       ├── bit_s42/           # BitNet baseline seed 42
│   │       └── bit_keep_conv1_s42/ # BitNet + conv1 FP32 (no KD)
│   ├── cifar100/
│   └── tiny_imagenet/
│
└── raw_kd/        # Knowledge distillation experiments
    ├── cifar10/
    │   └── resnet18/
    │       ├── bit_kd_s42/           # BitNet + KD
    │       └── bit_kd_keep_conv1_s42/ # Recipe: conv1 + KD
    ├── cifar100/
    └── tiny_imagenet/
```

Each experiment directory contains:
- `config.json` - Full training configuration
- `results.json` - Final metrics (test_acc, train_loss, etc.)
- `best_model.pth` - Best checkpoint
- `tensorboard/` - TensorBoard logs
