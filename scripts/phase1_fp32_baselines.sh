#!/bin/bash
# Phase 1: FP32 Baselines with Proper Training Recipe
# Target: ~94% CIFAR-10, ~77% CIFAR-100
# Time: ~45 hours with 2 GPUs (3 seeds in parallel)

set -e  # Exit on error

DATASETS="cifar10 cifar100 tiny-imagenet"
MODELS="resnet18 resnet50"
SEEDS="42 123 456"

echo "========================================="
echo "Phase 1: FP32 Baselines (Strong Recipe)"
echo "========================================="
echo "Models: resnet18, resnet50"
echo "Datasets: CIFAR-10, CIFAR-100, Tiny ImageNet"
echo "Seeds: 42, 123, 456"
echo "Recipe: 300 epochs, warmup 5, mixup 0.2, label smoothing 0.1"
echo "========================================="
echo ""

total=0
for model in $MODELS; do
  for dataset in $DATASETS; do
    echo "Starting $model on $dataset (3 seeds in parallel)..."

    for seed in $SEEDS; do
      uv run python -m experiments.train \
        --model $model --dataset $dataset \
        --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
        --mixup-alpha 0.2 --label-smoothing 0.1 \
        --seed $seed &
      total=$((total + 1))
    done

    # Wait for all 3 seeds to finish before next dataset
    wait
    echo "  ✓ Completed $model/$dataset (3 seeds)"
    echo ""
  done
done

echo "========================================="
echo "Phase 1 Complete!"
echo "Total experiments: $total"
echo "========================================="
echo ""
echo "Next step: Verify results and run Phase 2 (FP32+KD control)"
echo "  Check: cat results/raw/cifar10/resnet18/std_s42/results.json"
echo "  Expected: ~94% CIFAR-10, ~77% CIFAR-100"
