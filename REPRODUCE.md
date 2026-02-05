# Reproducing All Experiments

This document contains all commands needed to reproduce the experiments in the paper.

## Prerequisites

```bash
# Clone and setup
git clone <repo-url>
cd bitnet
uv sync
```

## Safety Features

Both training scripts include safety checks to prevent accidental overwrites:

- **Auto-generated paths**: Experiment directories are automatically named based on configuration
- **Overwrite protection**: Scripts refuse to run if results already exist
- **Force flag**: Use `--force` to intentionally overwrite existing results

Path naming convention (non-default hyperparameters are automatically included):

- Standard: `results/raw/{dataset}/{model}/{version}[_augment][_ablation]_s{seed}`
- KD: `results/raw_kd/{dataset}/{model}/bit_kd[_ablation][_t{temp}][_a{alpha}]_s{seed}`

## 1. Main Experiments (FP32 and BitNet baselines)

```bash
# All model/dataset combinations with 3 seeds each
uv run python -m experiments.sweep \
  --models resnet18 resnet50 \
  --datasets cifar10 cifar100 \
  --seeds 42 123 456
```

## 2. Augmentation Study

```bash
# Test all augmentation strategies
uv run python -m experiments.sweep \
  --models resnet18 resnet50 \
  --datasets cifar10 cifar100 \
  --augments basic cutout randaug full \
  --seeds 42 123 456
```

## 3. Layer-wise Ablation Study

```bash
# Test keeping individual layers in FP32
uv run python -m experiments.sweep \
  --models resnet18 resnet50 \
  --datasets cifar10 cifar100 \
  --ablations none keep_conv1 keep_layer1 keep_layer4 keep_fc \
  --seeds 42 123 456
```

## 4. Knowledge Distillation Experiments

### 4a. KD without ablation (BitNet + KD)

```bash
# CIFAR-10
uv run python -m experiments.sweep_kd \
  --models resnet18 \
  --datasets cifar10 \
  --seeds 42 123 456

# CIFAR-100
uv run python -m experiments.sweep_kd \
  --models resnet18 \
  --datasets cifar100 \
  --seeds 42 123 456
```

### 4b. conv1 + KD (Practical Recipe, T=4)

```bash
# CIFAR-10 with conv1 in FP32 + KD
for seed in 42 123 456; do
  uv run python -m experiments.train_kd --model resnet18 --dataset cifar10 \
    --teacher-path results/raw/cifar10/resnet18/std_s42/best_model.pth \
    --ablation keep_conv1 --temperature 4.0 --seed $seed
done

# CIFAR-100 with conv1 in FP32 + KD
for seed in 42 123 456; do
  uv run python -m experiments.train_kd --model resnet18 --dataset cifar100 \
    --teacher-path results/raw/cifar100/resnet18/std_s42/best_model.pth \
    --ablation keep_conv1 --temperature 4.0 --seed $seed
done
```

### 4c. Temperature Ablation (T=6, T=8)

```bash
# T=6
uv run python -m experiments.train_kd --model resnet18 --dataset cifar10 \
  --teacher-path results/raw/cifar10/resnet18/std_s42/best_model.pth \
  --ablation keep_conv1 --temperature 6.0 --seed 42 \
  --output-dir results/raw_kd/cifar10/resnet18/bit_kd_keep_conv1_t6_s42

# T=8
uv run python -m experiments.train_kd --model resnet18 --dataset cifar10 \
  --teacher-path results/raw/cifar10/resnet18/std_s42/best_model.pth \
  --ablation keep_conv1 --temperature 8.0 --seed 42 \
  --output-dir results/raw_kd/cifar10/resnet18/bit_kd_keep_conv1_t8_s42
```

### 4d. ResNet50 + KD (Generalization)

```bash
# ResNet50 CIFAR-10
uv run python -m experiments.train_kd --model resnet50 --dataset cifar10 \
  --teacher-path results/raw/cifar10/resnet50/std_s42/best_model.pth \
  --ablation keep_conv1 --seed 42

# ResNet50 CIFAR-100
uv run python -m experiments.train_kd --model resnet50 --dataset cifar100 \
  --teacher-path results/raw/cifar100/resnet50/std_s42/best_model.pth \
  --ablation keep_conv1 --seed 42
```

## 5. Analysis and Figure Generation

```bash
# Aggregate all results
uv run python -m analysis.aggregate_results

# Generate LaTeX tables
uv run python -m analysis.generate_tables

# Generate figures
uv run python -m analysis.generate_figures
```

## Expected Results

| Experiment | CIFAR-10 | CIFAR-100 |
|------------|----------|-----------|
| FP32 (ResNet18) | 88.89% | 62.40% |
| BitNet | 85.40% | 58.06% |
| BitNet + KD (T=4) | 86.66% | 60.55% |
| BitNet + keep_conv1 | 87.40% | 61.27% |
| **BitNet + keep_conv1 + KD** | **88.48%** | **63.40%** |

Note: CIFAR-100 with conv1+KD exceeds FP32 baseline (63.40% vs 62.40%).

## Hardware

- 2x NVIDIA RTX A6000 (48GB VRAM each)
- Training times: ResNet18/CIFAR ~2h, ResNet50/CIFAR ~4h
