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
@article{dariocazzani2025bitnet,
  title={1.58-bit Neural Networks: A Systematic Comparison Study},
  author={Dario Cazzani},
  year={2025}
}
```
