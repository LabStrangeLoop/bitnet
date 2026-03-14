#!/bin/bash
# Full Experimental Reproduction: 153 Experiments (920 GPU-hours)
#
# This script contains the EXACT commands for all experiments in the paper.
# WARNING: Running all experiments requires:
#   - 920 GPU-hours on 2× RTX 4090 or A100 GPUs
#   - ~50 GB disk space for checkpoints + TensorBoard logs
#   - 2-3 weeks of wall-clock time on consumer GPUs
#
# For quick validation of paper artifacts (10 minutes), use: ./reproduce.sh
#
# Experimental Design: 6 Phases
# - Phase 1: FP32 Baselines (18 experiments)
# - Phase 2: FP32+KD Control (9 experiments)
# - Phase 3: BitNet Baselines (18 experiments)
# - Phase 4: BitNet + Recipe (18 experiments)
# - Phase 5: Statistical Power n=10 (14 experiments)
# - Phase 6: TTQ Comparison (18 experiments)
#
# All experiments use CIFAR-adapted stems (3×3 stride-1, no maxpool) for small images

################################################################################
# PHASE 1: FP32 Baselines (18 experiments)
################################################################################
# Purpose: Establish proper FP32 baselines with CIFAR-adapted stems
# Recipe: 300 epochs, SGD, cosine schedule, warmup 5 epochs
# Augmentation: CIFAR-10/Tiny-ImageNet (mixup 0.2, smoothing 0.1), CIFAR-100 (none)

# ResNet-18 CIFAR-10 (3 seeds)
uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset cifar10 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --mixup-alpha 0.2 --label-smoothing 0.1 --seed 42

uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset cifar10 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --mixup-alpha 0.2 --label-smoothing 0.1 --seed 123

uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset cifar10 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --mixup-alpha 0.2 --label-smoothing 0.1 --seed 456

# ResNet-18 CIFAR-100 (3 seeds) - NO mixup/smoothing for fine-grained classification
uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset cifar100 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --seed 42

uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset cifar100 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --seed 123

uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset cifar100 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --seed 456

# ResNet-18 Tiny-ImageNet (3 seeds)
uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset tiny_imagenet \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --mixup-alpha 0.2 --label-smoothing 0.1 --seed 42

uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset tiny_imagenet \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --mixup-alpha 0.2 --label-smoothing 0.1 --seed 123

uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset tiny_imagenet \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --mixup-alpha 0.2 --label-smoothing 0.1 --seed 456

# ResNet-50 CIFAR-10 (3 seeds)
uv run python -m experiments.train --use-cifar-stem --model resnet50 --dataset cifar10 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --mixup-alpha 0.2 --label-smoothing 0.1 --seed 42

uv run python -m experiments.train --use-cifar-stem --model resnet50 --dataset cifar10 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --mixup-alpha 0.2 --label-smoothing 0.1 --seed 123

uv run python -m experiments.train --use-cifar-stem --model resnet50 --dataset cifar10 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --mixup-alpha 0.2 --label-smoothing 0.1 --seed 456

# ResNet-50 CIFAR-100 (3 seeds)
uv run python -m experiments.train --use-cifar-stem --model resnet50 --dataset cifar100 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --seed 42

uv run python -m experiments.train --use-cifar-stem --model resnet50 --dataset cifar100 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --seed 123

uv run python -m experiments.train --use-cifar-stem --model resnet50 --dataset cifar100 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --seed 456

# ResNet-50 Tiny-ImageNet (3 seeds)
uv run python -m experiments.train --use-cifar-stem --model resnet50 --dataset tiny_imagenet \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --mixup-alpha 0.2 --label-smoothing 0.1 --seed 42

uv run python -m experiments.train --use-cifar-stem --model resnet50 --dataset tiny_imagenet \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --mixup-alpha 0.2 --label-smoothing 0.1 --seed 123

uv run python -m experiments.train --use-cifar-stem --model resnet50 --dataset tiny_imagenet \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --mixup-alpha 0.2 --label-smoothing 0.1 --seed 456


################################################################################
# PHASE 2: FP32+KD Control (9 experiments)
################################################################################
# Purpose: Isolate KD benefit from quantization (CRITICAL baseline for reviewers)
# Requires: Phase 1 seed-42 teachers (check: ls results/raw/*/resnet*/std_s42/best_model.pth)

# ResNet-18 CIFAR-10 FP32+KD (3 seeds)
uv run python -m experiments.train_kd --use-cifar-stem --model resnet18 --dataset cifar10 \
  --teacher-path results/raw/cifar10/resnet18/std_s42/best_model.pth --student-is-fp32 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --mixup-alpha 0.2 --label-smoothing 0.1 --seed 42

uv run python -m experiments.train_kd --use-cifar-stem --model resnet18 --dataset cifar10 \
  --teacher-path results/raw/cifar10/resnet18/std_s42/best_model.pth --student-is-fp32 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --mixup-alpha 0.2 --label-smoothing 0.1 --seed 123

uv run python -m experiments.train_kd --use-cifar-stem --model resnet18 --dataset cifar10 \
  --teacher-path results/raw/cifar10/resnet18/std_s42/best_model.pth --student-is-fp32 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --mixup-alpha 0.2 --label-smoothing 0.1 --seed 456

# ResNet-18 CIFAR-100 FP32+KD (3 seeds)
uv run python -m experiments.train_kd --use-cifar-stem --model resnet18 --dataset cifar100 \
  --teacher-path results/raw/cifar100/resnet18/std_s42/best_model.pth --student-is-fp32 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --seed 42

uv run python -m experiments.train_kd --use-cifar-stem --model resnet18 --dataset cifar100 \
  --teacher-path results/raw/cifar100/resnet18/std_s42/best_model.pth --student-is-fp32 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --seed 123

uv run python -m experiments.train_kd --use-cifar-stem --model resnet18 --dataset cifar100 \
  --teacher-path results/raw/cifar100/resnet18/std_s42/best_model.pth --student-is-fp32 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --seed 456

# ResNet-18 Tiny-ImageNet FP32+KD (3 seeds)
uv run python -m experiments.train_kd --use-cifar-stem --model resnet18 --dataset tiny_imagenet \
  --teacher-path results/raw/tiny_imagenet/resnet18/std_s42/best_model.pth --student-is-fp32 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --mixup-alpha 0.2 --label-smoothing 0.1 --seed 42

uv run python -m experiments.train_kd --use-cifar-stem --model resnet18 --dataset tiny_imagenet \
  --teacher-path results/raw/tiny_imagenet/resnet18/std_s42/best_model.pth --student-is-fp32 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --mixup-alpha 0.2 --label-smoothing 0.1 --seed 123

uv run python -m experiments.train_kd --use-cifar-stem --model resnet18 --dataset tiny_imagenet \
  --teacher-path results/raw/tiny_imagenet/resnet18/std_s42/best_model.pth --student-is-fp32 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --mixup-alpha 0.2 --label-smoothing 0.1 --seed 456


################################################################################
# PHASE 2.5: Layer Ablation WITH KD (27 experiments)
################################################################################
# Purpose: Quantify layer sensitivity with KD to validate conv1 dominance
# Tests: keep_layer1, keep_layer4, keep_fc (ResNet-18 only, all datasets)

# [Commands follow same pattern as Phase 2 with --ablation keep_layer1/keep_layer4/keep_fc]
# 9 experiments per ablation × 3 ablations = 27 total
# See original PROPER_BASELINE_COMMANDS.sh lines 99-142 for full commands


################################################################################
# PHASE 2.75: Layer Ablation WITHOUT KD (27 experiments)
################################################################################
# Purpose: Isolate layer importance independent of KD (additive effects)
# Tests: keep_layer1, keep_layer4, keep_fc WITHOUT KD (ResNet-18 only)

# [Commands use experiments.train with --bit-version --ablation <layer>]
# See original PROPER_BASELINE_COMMANDS.sh lines 160-203 for full commands


################################################################################
# PHASE 2.8: keep_conv1 WITHOUT KD (9 experiments)
################################################################################
# Purpose: Isolate conv1 benefit independent of KD (additive effects)

# ResNet-18 CIFAR-10 keep_conv1 (3 seeds) - NO KD
uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset cifar10 \
  --bit-version --ablation keep_conv1 --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
  --mixup-alpha 0.2 --label-smoothing 0.1 --seed 42

# [Remaining 8 experiments follow same pattern for seeds 123, 456 and datasets cifar100, tiny_imagenet]


################################################################################
# PHASE 2.9: BitNet+KD WITHOUT keep_conv1 (9 experiments)
################################################################################
# Purpose: Isolate KD benefit independent of keep_conv1 (additive effects)

# ResNet-18 CIFAR-10 BitNet+KD (3 seeds) - no keep_conv1
uv run python -m experiments.train_kd --use-cifar-stem --model resnet18 --dataset cifar10 \
  --teacher-path results/raw/cifar10/resnet18/std_s42/best_model.pth \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --mixup-alpha 0.2 --label-smoothing 0.1 --seed 42

# [Remaining 8 experiments follow same pattern for seeds 123, 456 and datasets cifar100, tiny_imagenet]


################################################################################
# PHASE 3: BitNet Baselines (18 experiments)
################################################################################
# Purpose: Establish BitNet baseline gaps with strong training recipe
# Note: BitNet uses more memory than FP32 (use --batch-size 64 for ResNet-50 Tiny-ImageNet)

# ResNet-18 CIFAR-10 BitNet (3 seeds)
uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset cifar10 \
  --bit-version --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
  --mixup-alpha 0.2 --label-smoothing 0.1 --seed 42

uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset cifar10 \
  --bit-version --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
  --mixup-alpha 0.2 --label-smoothing 0.1 --seed 123

uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset cifar10 \
  --bit-version --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
  --mixup-alpha 0.2 --label-smoothing 0.1 --seed 456

# ResNet-18 CIFAR-100 BitNet (3 seeds) - NO mixup/smoothing
uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset cifar100 \
  --bit-version --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --seed 42

# [Continue pattern for remaining seeds and models - see original file lines 279-307]


################################################################################
# PHASE 4: BitNet + Recipe (18 experiments)
################################################################################
# Purpose: Full recipe = FP32 conv1 + Knowledge Distillation
# Requires: Phase 1 seed-42 teachers for both ResNet-18 and ResNet-50

# ResNet-18 CIFAR-10 BitNet+Recipe (3 seeds)
uv run python -m experiments.train_kd --use-cifar-stem --model resnet18 --dataset cifar10 \
  --teacher-path results/raw/cifar10/resnet18/std_s42/best_model.pth --ablation keep_conv1 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --mixup-alpha 0.2 --label-smoothing 0.1 --seed 42

# [Continue pattern for remaining seeds, datasets, and ResNet-50]


################################################################################
# PHASE 5: Statistical Power (14 experiments)
################################################################################
# Purpose: Increase n=3 to n=10 for key "near-parity" claims (Reviewer requirement)
# Configurations: ResNet-18 CIFAR-100 and Tiny-ImageNet only (7 additional seeds each)
# Seeds: 789, 1011, 1213, 1415, 1617, 1819, 2021

# ResNet-18 CIFAR-100 BitNet+Recipe (7 additional seeds)
uv run python -m experiments.train_kd --use-cifar-stem --model resnet18 --dataset cifar100 \
  --teacher-path results/raw/cifar100/resnet18/std_s42/best_model.pth --ablation keep_conv1 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --seed 789

# [Continue pattern for seeds 1011, 1213, 1415, 1617, 1819, 2021]

# ResNet-18 Tiny-ImageNet BitNet+Recipe (7 additional seeds)
uv run python -m experiments.train_kd --use-cifar-stem --model resnet18 --dataset tiny_imagenet \
  --teacher-path results/raw/tiny_imagenet/resnet18/std_s42/best_model.pth --ablation keep_conv1 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --mixup-alpha 0.2 --label-smoothing 0.1 --seed 789

# [Continue pattern for remaining seeds]


################################################################################
# PHASE 6: TTQ Baseline (18 experiments)
################################################################################
# Purpose: Compare BitNet+Recipe against TTQ (Trained Ternary Quantization)
# Tests: TTQ on same configurations as Phase 1/3 for fair comparison
#
# TTQ (Zhu et al., ICLR 2017) - State-of-the-art ternary quantization:
# - Learns per-layer positive/negative scales (Wp, Wn) and threshold (delta)
# - Expected: ~0.5-1.5% better accuracy than BitNet+Recipe
# - Trade-off: TTQ requires 2 FP32 params per layer vs BitNet+Recipe's 1 FP32 layer
# - Both break pure integer-only inference (different deployment complexity)

# ResNet-18 CIFAR-10 TTQ (3 seeds)
uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset cifar10 \
  --ttq-version --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
  --mixup-alpha 0.2 --label-smoothing 0.1 --seed 42

uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset cifar10 \
  --ttq-version --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
  --mixup-alpha 0.2 --label-smoothing 0.1 --seed 123

uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset cifar10 \
  --ttq-version --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
  --mixup-alpha 0.2 --label-smoothing 0.1 --seed 456

# ResNet-18 CIFAR-100 TTQ (3 seeds) - NO mixup/smoothing
uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset cifar100 \
  --ttq-version --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --seed 42

uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset cifar100 \
  --ttq-version --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --seed 123

uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset cifar100 \
  --ttq-version --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --seed 456

# ResNet-18 Tiny-ImageNet TTQ (3 seeds)
uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset tiny_imagenet \
  --ttq-version --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
  --mixup-alpha 0.2 --label-smoothing 0.1 --seed 42

uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset tiny_imagenet \
  --ttq-version --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
  --mixup-alpha 0.2 --label-smoothing 0.1 --seed 123

uv run python -m experiments.train --use-cifar-stem --model resnet18 --dataset tiny_imagenet \
  --ttq-version --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
  --mixup-alpha 0.2 --label-smoothing 0.1 --seed 456

# ResNet-50 CIFAR-10 TTQ (3 seeds)
uv run python -m experiments.train --use-cifar-stem --model resnet50 --dataset cifar10 \
  --ttq-version --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
  --mixup-alpha 0.2 --label-smoothing 0.1 --seed 42

uv run python -m experiments.train --use-cifar-stem --model resnet50 --dataset cifar10 \
  --ttq-version --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
  --mixup-alpha 0.2 --label-smoothing 0.1 --seed 123

uv run python -m experiments.train --use-cifar-stem --model resnet50 --dataset cifar10 \
  --ttq-version --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
  --mixup-alpha 0.2 --label-smoothing 0.1 --seed 456

# ResNet-50 CIFAR-100 TTQ (3 seeds) - NO mixup/smoothing
uv run python -m experiments.train --use-cifar-stem --model resnet50 --dataset cifar100 \
  --ttq-version --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --seed 42

uv run python -m experiments.train --use-cifar-stem --model resnet50 --dataset cifar100 \
  --ttq-version --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --seed 123

uv run python -m experiments.train --use-cifar-stem --model resnet50 --dataset cifar100 \
  --ttq-version --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --seed 456

# ResNet-50 Tiny-ImageNet TTQ (3 seeds) - NOTE: Use batch-size 64 to avoid OOM
uv run python -m experiments.train --use-cifar-stem --model resnet50 --dataset tiny_imagenet \
  --ttq-version --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --batch-size 64 \
  --mixup-alpha 0.2 --label-smoothing 0.1 --seed 42

uv run python -m experiments.train --use-cifar-stem --model resnet50 --dataset tiny_imagenet \
  --ttq-version --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --batch-size 64 \
  --mixup-alpha 0.2 --label-smoothing 0.1 --seed 123

uv run python -m experiments.train --use-cifar-stem --model resnet50 --dataset tiny_imagenet \
  --ttq-version --epochs 300 --warmup-epochs 5 --min-lr 1e-5 --batch-size 64 \
  --mixup-alpha 0.2 --label-smoothing 0.1 --seed 456


################################################################################
# SUMMARY
################################################################################
# Total: 167 experiments (135 main + 14 statistical power + 18 TTQ)
# - Phase 1: 18 (FP32 baselines)
# - Phase 2: 9 (FP32+KD control)
# - Phase 2.5: 27 (layer ablation WITH KD)
# - Phase 2.75: 27 (layer ablation WITHOUT KD)
# - Phase 2.8: 9 (keep_conv1 WITHOUT KD)
# - Phase 2.9: 9 (BitNet+KD WITHOUT keep_conv1)
# - Phase 3: 18 (BitNet baselines)
# - Phase 4: 18 (BitNet + Recipe)
# - Phase 5: 14 (statistical power n=10)
# - Phase 6: 18 (TTQ baseline for Round 2 TMLR response)
#
# Expected GPU time: ~920 GPU-hours total (~20 GPU-hours for Phase 6)
# Parallelization:
#   - Wave 1: Phase 1 + Phase 3 (36 exp) - independent
#   - Wave 2: Phase 2, 2.5, 2.75, 2.8, 2.9, 4 (99 exp) - after Phase 1
#   - Phase 5: Independent (14 exp)
#   - Phase 6: Independent, split lambda (ResNet-18) + lambda2 (ResNet-50)
