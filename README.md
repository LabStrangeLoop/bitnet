# Understanding and Closing the 1.58-bit Quantization Gap in CNNs: An Empirical Study

> Research reproducibility repository for TMLR submission

Implementation of BitNet b1.58 (ternary quantization) applied to ResNet architectures, with systematic analysis across 153 controlled experiments.

Based on:

- [BitNet: Scaling 1-bit Transformers](https://arxiv.org/abs/2310.11453)
- [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764)

---

## Research Scope

**Models:** ResNet-18 (11.17M params), ResNet-50 (23.52M params)

**Datasets:** CIFAR-10, CIFAR-100, Tiny-ImageNet

**Experiments:** 153 total across 6 phases

**Compute:** 920 GPU-hours (reproducible with fixed seeds)

---

## Key Findings

This work provides three main contributions:

1. **Layer sensitivity is asymmetric:** First convolutional layer (conv1) accounts for 30-74% of the accuracy gap despite representing only 0.08% of parameters.

2. **Knowledge distillation fails for ternary networks:** BitNet+KD performs 0.9-3.1% worse than BitNet alone, contradicting standard practice for compressed models.

3. **Mixed-precision recipe:** FP32 conv1 + ternary remainder recovers 30-74% of accuracy gap with 20.2× compression maintained.

**Representative results:**

- CIFAR-10 (ResNet-18): 93.01% FP32 → 92.01% mixed-precision (1.00% gap)
- CIFAR-100 (ResNet-18): 76.12% FP32 → 72.72% mixed-precision (3.40% gap)
- Tiny-ImageNet (ResNet-18): 62.18% FP32 → 57.43% mixed-precision (4.75% gap)

---

## Quick Reproducibility Check (10 minutes)

Regenerate all paper artifacts from pre-computed results:

```bash
./reproduce.sh
```

This will:

1. Aggregate 153 experiment results
2. Generate 12 LaTeX tables
3. Generate 6 PDF figures
4. Compile paper PDF (28 pages, ~550 KB)

**Verification:** All tables/figures should match paper exactly.

---

## Experimental Design

### Phase Structure (153 experiments)

| Phase | Experiments | Description |
|-------|------------|-------------|
| 1. FP32 Baselines | 18 | Standard ResNet (3 seeds) |
| 2. FP32+KD Control | 9 | Isolate KD benefit from quantization |
| 3. BitNet Baselines | 18 | Full ternary quantization |
| 4. BitNet + Recipe | 18 | Mixed-precision (FP32 conv1) |
| 5. Statistical Power | 72 | Extended to n=10 seeds |
| 6. TTQ Comparison | 18 | Validate against prior art |

**Total:** 153 experiments, deterministic training (fixed seeds), programmatic analysis.

### CIFAR-Adapted Stems

**Critical methodological detail:** Standard ImageNet ResNet uses 7×7 stride-2 conv + maxpool, which destroys spatial information on small images (32×32 CIFAR, 64×64 Tiny-IN).

**Our approach:** 3×3 stride-1 conv, no maxpool, preserving spatial resolution.

**Impact:** CIFAR-adapted stem recovers +6.94% (CIFAR-10) and +17.18% (CIFAR-100) compared to ImageNet stem. This is standard practice in literature but critical for reproducibility.

**⚠️ All CIFAR/Tiny-ImageNet experiments require `--use-cifar-stem` flag.**

---

## Full Experimental Reproduction (920 GPU-hours)

To re-run all 153 experiments from scratch:

**Requirements:**

- 2× RTX 4090 or A100 GPUs
- 50 GB disk space
- 2-3 weeks wall-clock time

**Commands:** See `EXPERIMENTS_REFERENCE.sh` for exact commands (all 153).

**Deterministic training:**

- Seeds: 42, 123, 456 (main experiments)
- Extended: 789, 1011, 1213, 1415, 1617, 1819, 2021 (statistical power)
- Bit-exact checkpoint MD5 hashes with fixed seeds

---

## Results Directory Structure

```text
results/
├── raw/                  # 90 standard training experiments
│   └── {dataset}/{model}/{version}_s{seed}/
│       ├── config.json        # Training hyperparameters
│       └── results.json       # Final metrics
├── raw_kd/               # 63 knowledge distillation experiments
└── processed/
    └── aggregated.csv    # All 153 experiments aggregated
```

**Note on Phase 6 (TTQ):** TTQ comparison experiments are in `raw/` directory with `_ttq` suffix (18 additional experiments).

See [`results/README.md`](results/README.md) for detailed naming conventions.

---

## Usage

### Prerequisites

- Python 3.11+
- CUDA-capable GPU
- [uv](https://github.com/astral-sh/uv) package manager

```bash
git clone https://anonymous.4open.science/r/bitnet-0ABF
cd bitnet
uv sync
```

### Single Experiment

**⚠️ Important:** Always use `--use-cifar-stem` for CIFAR-10, CIFAR-100, and Tiny-ImageNet.

```bash
# Standard FP32 baseline
uv run python -m experiments.train \
    --model resnet18 --dataset cifar10 --epochs 200 --use-cifar-stem

# BitNet (full ternary quantization)
uv run python -m experiments.train \
    --model resnet18 --dataset cifar10 --epochs 200 --use-cifar-stem --bit-version

# Mixed-precision recipe (FP32 conv1 + ternary remainder)
uv run python -m experiments.train \
    --model resnet18 --dataset cifar10 --epochs 300 --use-cifar-stem \
    --bit-version --ablation keep_conv1
```

### Layer-wise Ablation

Investigate which layers contribute most to accuracy gap:

```bash
# Available ablation modes:
#   none        - Full BitNet (all layers quantized)
#   keep_conv1  - FP32 conv1 (recovers 30-74% of gap)
#   keep_layer1 - FP32 first residual block
#   keep_layer4 - FP32 last residual block
#   keep_fc     - FP32 classifier only

# Run ablation sweep
uv run python -m experiments.sweep \
    --models resnet18 --datasets cifar10 \
    --ablations none keep_conv1 keep_layer1 keep_layer4 keep_fc
```

**Key insight from paper:** conv1 contributes disproportionately (30-74% of accuracy recovery) despite being only 0.08% of parameters.

### FP32+KD Control Experiments

To isolate KD benefit from quantization penalty:

```bash
# FP32 teacher
uv run python -m experiments.train \
    --model resnet18 --dataset cifar10 --epochs 200 --use-cifar-stem

# FP32 student (no quantization, only KD)
uv run python -m experiments.train_kd \
    --model resnet18 --dataset cifar10 --epochs 200 --use-cifar-stem \
    --teacher-path results/raw/cifar10/resnet18/std_s42/best_model.pth

# BitNet student (quantization + KD)
uv run python -m experiments.train_kd \
    --model resnet18 --dataset cifar10 --epochs 200 --use-cifar-stem \
    --bit-version \
    --teacher-path results/raw/cifar10/resnet18/std_s42/best_model.pth
```

**Finding:** BitNet+KD underperforms BitNet-only by 0.9-3.1%, suggesting gradient mismatch issues.

### Analysis Pipeline

After experiments complete:

```bash
# Aggregate all 153 experiments
uv run python -m analysis.aggregate_results

# Generate paper tables (12 LaTeX files)
uv run python -m analysis.generate_tables

# Generate paper figures (6 PDFs)
uv run python -m analysis.generate_figures
```

---

## Datasets

### CIFAR-10/100

Downloaded automatically on first run.

### Tiny-ImageNet

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

---

## Supported Models

| Model     | Parameters | timm name  |
|-----------|------------|------------|
| ResNet-18 | 11.17M     | `resnet18` |
| ResNet-50 | 23.52M     | `resnet50` |

**Note:** Additional architectures (VGG-16, MobileNetV2, EfficientNet-B0, ConvNeXt) are implemented via timm but not evaluated in the paper.

---

## Documentation

For detailed information, see:

- **[METHODOLOGY.md](METHODOLOGY.md)** - Experimental design and rationale
- **[EXPERIMENTS_REFERENCE.sh](EXPERIMENTS_REFERENCE.sh)** - All 153 commands
- **[PROJECT_SETUP.md](PROJECT_SETUP.md)** - Quick start guide
- **[REPRODUCE.md](REPRODUCE.md)** - Step-by-step reproduction
- **[TTQ_VERIFICATION.md](TTQ_VERIFICATION.md)** - Technical TTQ comparison details
- **[results/README.md](results/README.md)** - Results directory structure

---

## Project Structure

```text
bitnet/
├── bitnet/
│   ├── nn/                      # 1.58-bit layer implementations
│   │   ├── bitlinear.py         # BitLinear layer
│   │   ├── bitconv2d.py         # BitConv2d layer
│   │   ├── ttq_linear.py        # TTQ baseline (Zhu et al. 2017)
│   │   └── ttq_conv2d.py        # TTQ conv2d baseline
│   ├── layer_swap.py            # Full model quantization
│   └── layer_swap_selective.py  # Selective quantization (ablation)
├── experiments/
│   ├── train.py                 # Standard training
│   ├── train_kd.py              # Knowledge distillation
│   ├── sweep.py                 # Experiment runner
│   └── datasets/                # CIFAR-10/100, Tiny-ImageNet loaders
├── analysis/                    # Result aggregation and paper generation
└── paper/                       # LaTeX source (28 pages)
```

---

## Citation

```bibtex
@article{anonymous2026bitconv,
  title={Understanding and Closing the 1.58-bit Quantization Gap in CNNs: An Empirical Study},
  author={Anonymous},
  year={2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
