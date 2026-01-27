# 1.58-bit Neural Networks: A Systematic Comparison Study

[![CI](https://github.com/LabStrangeLoop/bitnet/actions/workflows/ci.yml/badge.svg)](https://github.com/LabStrangeLoop/bitnet/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Implementation of BitNet b1.58 (ternary weights {-1, 0, +1}) for convolutional neural networks, with systematic comparison against standard FP32 models.

Based on:
- [BitNet: Scaling 1-bit Transformers](https://arxiv.org/abs/2310.11453)
- [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764)

## Requirements

- Python 3.11+
- CUDA-capable GPU (tested on RTX A6000)
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
git clone https://github.com/LabStrangeLoop/bitnet.git
cd bitnet
uv sync
```

## Datasets

### CIFAR-10/100
Downloaded automatically on first run.

### ImageNet-1k
Requires HuggingFace authentication:

1. Create account at [huggingface.co](https://huggingface.co)
2. Get access token at [Settings > Tokens](https://huggingface.co/settings/tokens) (click "New token", select "Read")
3. Accept ImageNet terms at [imagenet-1k dataset page](https://huggingface.co/datasets/imagenet-1k)
4. Login and download:

```bash
# Login (paste your token when prompted)
uv run python -c "from huggingface_hub import login; login()"

# Download (~150GB, takes several hours)
uv run python scripts/download_imagenet.py --data-dir ./data
```

## Usage

### Single Experiment

```bash
# Standard ResNet18 on CIFAR-10
uv run python -m experiments.train --model resnet18 --dataset cifar10 --epochs 200

# BitNet version
uv run python -m experiments.train --model resnet18 --dataset cifar10 --epochs 200 --bit-version

# Quiet mode (suppress progress bars and per-epoch logs, useful for batch runs)
uv run python -m experiments.train --model resnet18 --dataset cifar10 --epochs 200 --quiet
```

### Full Experiment Sweep

```bash
# Dry run (shows commands without executing)
uv run python -m experiments.sweep --dry-run

# Run all experiments (90 total: 5 models × 3 datasets × 2 versions × 3 seeds)
# Uses --quiet mode automatically, shows compact progress: [1/90] Running... 92.34%
uv run python -m experiments.sweep

# Run subset
uv run python -m experiments.sweep --models resnet18 resnet50 --datasets cifar10 cifar100

# With different augmentation levels
uv run python -m experiments.sweep --augments basic randaug cutout full

# With different optimizer (AdamW instead of default SGD)
uv run python -m experiments.sweep --optimizer adamw --output-dir results/raw_adamw

# Longer training (400 epochs instead of 200)
uv run python -m experiments.sweep --epochs 400 --output-dir results/raw_400ep

# Ctrl+C shows summary of completed/skipped/failed experiments
```

### Layer-wise Ablation

Investigate which layers contribute most to the accuracy gap by keeping specific layers in FP32:

```bash
# Single ablation experiment (keep first conv in FP32, rest quantized)
uv run python -m experiments.train --model resnet18 --dataset cifar10 \
    --bit-version --ablation keep_conv1

# Available ablation modes:
#   none        - Full BitNet (all layers quantized)
#   keep_conv1  - Keep first conv layer in FP32
#   keep_layer1 - Keep first residual block in FP32
#   keep_layer4 - Keep last residual block in FP32
#   keep_fc     - Keep classifier in FP32

# Sweep all ablation modes
uv run python -m experiments.sweep --models resnet18 --datasets cifar10 \
    --ablations none keep_conv1 keep_layer1 keep_layer4 keep_fc
```

### Monitoring with TensorBoard

```bash
# Local machine
uv run tensorboard --logdir results/raw

# Remote server (accessible from browser)
uv run tensorboard --logdir results/raw --bind_all --port 6006
```

### Analysis

After experiments complete:

```bash
# Aggregate results
uv run python -m analysis.aggregate_results

# Generate tables and figures
uv run python -m analysis.generate_tables
uv run python -m analysis.generate_figures
```

## Supported Models

| Model | timm name |
|-------|-----------|
| ResNet-18 | `resnet18` |
| ResNet-50 | `resnet50` |
| VGG-16 | `vgg16` |
| MobileNetV2 | `mobilenetv2_100` |
| EfficientNet-B0 | `efficientnet_b0` |

## Project Structure

```
bitnet/
├── bitnet/
│   ├── nn/                      # 1.58-bit layer implementations
│   │   ├── quantization.py      # Shared quantization functions
│   │   ├── bitlinear.py         # BitLinear layer
│   │   └── bitconv2d.py         # BitConv2d layer
│   ├── layer_swap.py            # Full model quantization
│   └── layer_swap_selective.py  # Selective quantization (ablation)
├── experiments/
│   ├── train.py                 # Training script
│   ├── sweep.py                 # Experiment runner
│   ├── config.py                # Configuration and enums
│   └── datasets/                # Dataset loaders
├── analysis/                    # Result analysis and paper generation
├── results/raw/                 # Experiment outputs
└── paper/                       # LaTeX paper and generated tables/figures
```

## Development

```bash
uv sync --group dev
uv run python -m pre_commit install
uv run ruff check .
uv run mypy .
```

## Reproducibility

- Fixed random seeds (42, 123, 456)
- Deterministic CUDA operations
- Complete environment in `uv.lock`
- Hardware: 2x NVIDIA RTX A6000

## Experiment Plan

### Main Experiments

Compare standard FP32 vs BitNet 1.58-bit across:

- **Models**: ResNet-18, ResNet-50, VGG-16, MobileNetV2, EfficientNet-B0
- **Datasets**: CIFAR-10, CIFAR-100, ImageNet-1k
- **Seeds**: 3 seeds per configuration for statistical significance

### Augmentation Ablation Study

Investigate how data augmentation affects the accuracy gap between FP32 and BitNet:

| Level      | Augmentation                  | Description                            |
| ---------- | ----------------------------- | -------------------------------------- |
| `basic`    | Crop + Flip                   | Baseline (RandomCrop, HorizontalFlip)  |
| `randaug`  | + RandAugment                 | Learned augmentation policy            |
| `cutout`   | + RandomErasing               | Patch-based regularization             |
| `full`     | RandAugment + RandomErasing   | SOTA augmentation                      |

**Research questions:**

1. Does the FP32-BitNet accuracy gap narrow with stronger augmentation?
2. Which augmentation benefits BitNet the most?
3. What is the "true" accuracy cost of 1.58-bit quantization under SOTA training?

## Citation

```bibtex
@article{cazzani2025bitconv,
  title={1.58-bit Convolutional Neural Networks: A Systematic Comparison Study},
  author={Cazzani, Dario},
  year={2025}
}
```
