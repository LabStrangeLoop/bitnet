# 1.58-bit Neural Networks: A Systematic Comparison Study

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

```bash
# 1. Create account at huggingface.co
# 2. Accept ImageNet terms at: https://huggingface.co/datasets/imagenet-1k
# 3. Login
uv run huggingface-cli login

# 4. Download (~150GB, takes several hours)
uv run python scripts/download_imagenet.py --data-dir ./data
```

## Usage

### Single Experiment

```bash
# Standard ResNet18 on CIFAR-10
uv run python -m experiments.train --model resnet18 --dataset cifar10 --epochs 200

# BitNet version
uv run python -m experiments.train --model resnet18 --dataset cifar10 --epochs 200 --bit-version
```

### Full Experiment Sweep

```bash
# Dry run (shows commands without executing)
uv run python -m experiments.sweep --dry-run

# Run all experiments (90 total: 5 models × 3 datasets × 2 versions × 3 seeds)
uv run python -m experiments.sweep

# Run subset
uv run python -m experiments.sweep --models resnet18 resnet50 --datasets cifar10 cifar100
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
├── bitnet/nn/           # 1.58-bit layer implementations
│   ├── quantization.py  # Shared quantization functions
│   ├── bitlinear.py     # BitLinear layer
│   └── bitconv2d.py     # BitConv2d layer
├── experiments/
│   ├── train.py         # Training script
│   ├── sweep.py         # Experiment runner
│   └── datasets/        # Dataset loaders
├── analysis/            # Result analysis and paper generation
├── results/raw/         # Experiment outputs
└── paper/               # Generated tables and figures
```

## Development

```bash
uv sync --all-extras
uv run pre-commit install
uv run ruff check .
uv run mypy .
```

## Reproducibility

- Fixed random seeds (42, 123, 456)
- Deterministic CUDA operations
- Complete environment in `uv.lock`
- Hardware: 2x NVIDIA RTX A6000

## Citation

```bibtex
@article{cazzani2025bitconv,
  title={1.58-bit Convolutional Neural Networks: A Systematic Comparison Study},
  author={Cazzani, Dario},
  year={2025}
}
```
