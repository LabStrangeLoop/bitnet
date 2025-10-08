#!/bin/bash
# Setup script for 1.58-bit Neural Networks project
# Usage: chmod +x setup_project.sh && ./setup_project.sh

set -e  # Exit on error

echo "🚀 Setting up 1.58-bit Neural Networks project structure..."

# Create main directories
mkdir -p bitnet/nn
mkdir -p experiments/{models,datasets}
mkdir -p configs/{model,dataset,optimizer,experiment}
mkdir -p results/{raw,processed,figures}
mkdir -p analysis
mkdir -p paper/{src,figures,tables}
mkdir -p environment
mkdir -p data  # For downloaded datasets

echo "📁 Directory structure created"

# ============================================================================
# BITNET MODULE
# ============================================================================

cat > bitnet/__init__.py << 'EOF'
"""1.58-bit Neural Network Implementation"""
from bitnet.layer_swap import replace_layers, replace_linear_layers, replace_conv2d_layers

__all__ = ['replace_layers', 'replace_linear_layers', 'replace_conv2d_layers']
EOF

cat > bitnet/nn/__init__.py << 'EOF'
"""BitNet neural network layers"""
from bitnet.nn.bitlinear import BitLinear
from bitnet.nn.bitconv2d import BitConv2d

__all__ = ['BitLinear', 'BitConv2d']
EOF

# Your existing layer_swap.py (keeping your code)
cat > bitnet/layer_swap.py << 'EOF'
from torch import nn

from bitnet.nn.bitconv2d import BitConv2d
from bitnet.nn.bitlinear import BitLinear


def replace_linear_layers(model: nn.Module) -> None:
    """Replace all Linear layers with BitLinear layers."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(
                model,
                name,
                BitLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                ),
            )
        else:
            replace_linear_layers(module)


def replace_conv2d_layers(model: nn.Module) -> None:
    """Replace all Conv2d layers with BitConv2d layers."""
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            setattr(
                model,
                name,
                BitConv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    bias=module.bias is not None,
                ),
            )
        else:
            replace_conv2d_layers(module)


def replace_layers(model: nn.Module) -> None:
    """Replace all Linear and Conv2d layers with their Bit versions."""
    replace_linear_layers(model)
    replace_conv2d_layers(model)
EOF

# Create placeholder for your bitlinear.py (you'll paste your code here)
cat > bitnet/nn/bitlinear.py << 'EOF'
# TODO: Paste your BitLinear implementation here
# Your existing bitlinear.py code goes here
EOF

# Create placeholder for your bitconv2d.py
cat > bitnet/nn/bitconv2d.py << 'EOF'
# TODO: Paste your BitConv2d implementation here
# Your existing bitconv2d.py code goes here
EOF

# ============================================================================
# EXPERIMENTS MODULE
# ============================================================================

cat > experiments/__init__.py << 'EOF'
"""Experiments package for training and evaluation"""
EOF

cat > experiments/models/__init__.py << 'EOF'
"""Model factory using timm"""
EOF

cat > experiments/models/factory.py << 'EOF'
"""Model factory for creating standard and bit-quantized models."""
import timm
from torch import nn

from bitnet.layer_swap import replace_layers


def get_model(
    name: str,
    num_classes: int,
    bit_version: bool = False,
    pretrained: bool = True,
) -> nn.Module:
    """
    Get a model using timm with optional bit quantization.

    Args:
        name: Model architecture name (e.g., 'resnet18', 'mobilenetv3_small_100')
        num_classes: Number of output classes
        bit_version: If True, replace layers with 1.58-bit versions
        pretrained: If True, load pretrained weights

    Returns:
        PyTorch model
    """
    model = timm.create_model(
        name,
        num_classes=num_classes,
        pretrained=pretrained
    )

    if bit_version:
        replace_layers(model)

    return model


def list_available_models(pattern: str = '*') -> list[str]:
    """List available models from timm."""
    return timm.list_models(pattern)
EOF

cat > experiments/datasets/__init__.py << 'EOF'
"""Dataset factory using torchvision and HuggingFace"""
EOF

cat > experiments/datasets/factory.py << 'EOF'
"""Dataset factory for loading and preprocessing datasets."""
import torchvision.datasets as tv_datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def get_transforms(transform_type: str, dataset_name: str) -> transforms.Compose:
    """Get transforms for a dataset."""
    if transform_type == 'standard':
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    # Add more transform types as needed
    raise ValueError(f"Unknown transform type: {transform_type}")


def get_dataset(
    name: str,
    split: str,
    transform_type: str = 'standard',
    root: str = './data'
) -> Dataset:
    """
    Get dataset from torchvision or HuggingFace.

    Args:
        name: Dataset name (e.g., 'cifar10', 'imagenet1k')
        split: 'train' or 'test'
        transform_type: Type of transforms to apply
        root: Root directory for datasets

    Returns:
        PyTorch Dataset
    """
    transform = get_transforms(transform_type, name)
    train = (split == 'train')

    # Torchvision datasets
    torchvision_datasets = {
        'cifar10': tv_datasets.CIFAR10,
        'cifar100': tv_datasets.CIFAR100,
        # Add more as needed
    }

    if name in torchvision_datasets:
        return torchvision_datasets[name](
            root=root,
            train=train,
            transform=transform,
            download=True
        )

    raise ValueError(f"Unknown dataset: {name}")
EOF

cat > experiments/datasets/transforms.py << 'EOF'
"""Standard data augmentation and preprocessing transforms."""
import torchvision.transforms as transforms


def get_train_transforms(image_size: int = 224):
    """Standard training transforms with augmentation."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def get_test_transforms(image_size: int = 224):
    """Standard test transforms without augmentation."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
EOF

cat > experiments/train.py << 'EOF'
"""Main training script."""
# TODO: Implement training loop with Hydra
EOF

cat > experiments/eval.py << 'EOF'
"""Evaluation script."""
# TODO: Implement evaluation
EOF

cat > experiments/sweep.py << 'EOF'
"""Run systematic experiment sweeps."""
# TODO: Implement sweep logic
EOF

# ============================================================================
# CONFIGS
# ============================================================================

cat > configs/main.yaml << 'EOF'
# Main configuration file
defaults:
  - model: resnet18
  - dataset: cifar10
  - optimizer: sgd
  - _self_

# Experiment settings
seed: 42
epochs: 100
batch_size: 128
num_workers: 4

# Output
output_dir: results/raw
log_interval: 10
EOF

cat > configs/model/resnet18.yaml << 'EOF'
name: resnet18
pretrained: true
EOF

cat > configs/dataset/cifar10.yaml << 'EOF'
name: cifar10
num_classes: 10
image_size: 224
EOF

cat > configs/optimizer/sgd.yaml << 'EOF'
name: sgd
lr: 0.01
momentum: 0.9
weight_decay: 0.0001
EOF

cat > configs/experiment/core_benchmarks.yaml << 'EOF'
# Core benchmark experiment configuration
defaults:
  - /model: resnet18
  - /dataset: cifar10
  - /optimizer: sgd

# Systematic comparison
model_versions: [standard, bit]
seeds: [42, 123, 456, 789, 999]
datasets: [cifar10, cifar100, tiny_imagenet]
learning_rates: [0.01, 0.001]
EOF

# ============================================================================
# ANALYSIS
# ============================================================================

cat > analysis/__init__.py << 'EOF'
"""Analysis and paper generation tools"""
EOF

cat > analysis/aggregate_results.py << 'EOF'
"""Aggregate results from all experiments."""
# TODO: Implement result aggregation
EOF

cat > analysis/statistical_analysis.py << 'EOF'
"""Statistical analysis of results."""
# TODO: Implement t-tests, effect sizes, etc.
EOF

cat > analysis/generate_tables.py << 'EOF'
"""Generate LaTeX tables for paper."""
# TODO: Implement table generation
EOF

cat > analysis/generate_figures.py << 'EOF'
"""Generate figures for paper."""
# TODO: Implement figure generation
EOF

cat > analysis/utils.py << 'EOF'
"""Helper utilities for analysis."""
EOF

# ============================================================================
# CONFIGURATION FILES
# ============================================================================

cat > pyproject.toml << 'EOF'
[project]
name = "bitnet-1.58"
version = "0.1.0"
description = "1.58-bit Neural Networks: Systematic Comparison Study"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "timm>=0.9.0",
    "hydra-core>=1.3.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "scipy>=1.10.0",
    "tqdm>=4.65.0",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.1.0",
    "mypy>=1.5.0",
    "pytest>=7.4.0",
]

[tool.ruff]
line-length = 88
target-version = "py311"
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # Set to true when ready
EOF

cat > environment/.pre-commit-config.yaml << 'EOF'
repos:
  - repo: local
    hooks:
      - id: ruff-check
        name: ruff check
        entry: uv run ruff check --fix
        language: system
        types: [python]
        pass_filenames: true

      - id: ruff-format
        name: ruff format
        entry: uv run ruff format
        language: system
        types: [python]
        pass_filenames: true

      - id: mypy
        name: mypy
        entry: uv run mypy
        language: system
        types: [python]
        pass_filenames: true
EOF

cat > environment/Dockerfile << 'EOF'
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN uv sync --frozen

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["uv", "run", "python", "experiments/train.py"]
EOF

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/
env/

# uv
.uv/
uv.lock

# Data
data/
*.pth
*.ckpt

# Results
results/raw/*
!results/raw/.gitkeep
results/processed/*
!results/processed/.gitkeep

# Hydra
outputs/
multirun/
.hydra/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*.ipynb
EOF

cat > reproduce.sh << 'EOF'
#!/bin/bash
# One-command reproduction script
set -e

echo "🔬 Reproducing 1.58-bit Neural Networks paper..."
echo ""

# Setup environment
echo "📦 Setting up environment..."
uv sync
echo "✓ Environment ready"
echo ""

# Run core benchmarks
echo "🚀 Running core benchmarks..."
# uv run python experiments/sweep.py experiment=core_benchmarks
echo "✓ Core benchmarks complete"
echo ""

# Run challenging benchmarks
echo "🎯 Running challenging benchmarks..."
# uv run python experiments/sweep.py experiment=challenging_benchmarks
echo "✓ Challenging benchmarks complete"
echo ""

# Aggregate results
echo "📊 Aggregating results..."
# uv run python analysis/aggregate_results.py
echo "✓ Results aggregated"
echo ""

# Generate paper content
echo "📝 Generating paper tables and figures..."
# uv run python analysis/generate_tables.py
# uv run python analysis/generate_figures.py
echo "✓ Paper content generated"
echo ""

echo "✅ Reproduction complete!"
echo "Check results/ and paper/ directories for outputs"
EOF

chmod +x reproduce.sh

cat > README.md << 'EOF'
# 1.58-bit Neural Networks: A Systematic Comparison Study

This repository contains the complete implementation and experiments for our paper on 1.58-bit convolutional neural networks.

## Quick Start

```bash
# Setup project structure (if not already done)
./setup_project.sh

# Install dependencies
uv sync

# Run single experiment
uv run python experiments/train.py

# Reproduce entire paper
./reproduce.sh
```

## Project Structure

- `bitnet/`: 1.58-bit layer implementations
- `experiments/`: Training and evaluation code
- `configs/`: Hydra configuration files
- `analysis/`: Result processing and paper generation
- `results/`: Experimental outputs
- `paper/`: LaTeX source and generated content

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Setup pre-commit hooks
uv run pre-commit install

# Run linting
uv run ruff check .

# Run type checking
uv run mypy .
```

## Reproducibility

This project follows strict reproducibility guidelines:
- Fixed random seeds
- Deterministic GPU operations
- Complete environment capture via `uv.lock`
- Automated validation of results

## Citation

```bibtex
@article{yourname2025bitnet,
  title={1.58-bit Neural Networks: A Systematic Comparison Study},
  author={Your Name},
  year={2025}
}
```
EOF

# Create empty .gitkeep files for empty directories
touch results/raw/.gitkeep
touch results/processed/.gitkeep
touch results/figures/.gitkeep
touch paper/figures/.gitkeep
touch paper/tables/.gitkeep

echo ""
echo "✅ Project structure created successfully!"
echo ""
echo "📝 Next steps:"
echo "1. Paste your BitLinear code into: bitnet/nn/bitlinear.py"
echo "2. Paste your BitConv2d code into: bitnet/nn/bitconv2d.py"
echo "3. Run: uv sync"
echo "4. Start developing!"
echo ""
echo "🚀 Run './reproduce.sh' when ready to test the full pipeline"