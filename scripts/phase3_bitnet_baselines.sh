#!/bin/bash
# Phase 3: BitNet Baselines with Strong Recipe
# Time: ~45 hours with 2 GPUs

set -e

DATASETS="cifar10 cifar100 tiny-imagenet"
MODELS="resnet18 resnet50"
SEEDS="42 123 456"

echo "========================================="
echo "Phase 3: BitNet Baselines (Strong Recipe)"
echo "========================================="
echo "Same training recipe as FP32 for fair comparison"
echo "========================================="
echo ""

total=0
for model in $MODELS; do
  for dataset in $DATASETS; do
    echo "Starting $model/$dataset BitNet (3 seeds in parallel)..."

    for seed in $SEEDS; do
      uv run python -m experiments.train \
        --model $model --dataset $dataset --bit-version \
        --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
        --mixup-alpha 0.2 --label-smoothing 0.1 \
        --seed $seed &
      total=$((total + 1))
    done

    wait
    echo "  ✓ Completed $model/$dataset BitNet (3 seeds)"
    echo ""
  done
done

echo "========================================="
echo "Phase 3 Complete!"
echo "Total experiments: $total"
echo "========================================="
echo ""
echo "Next step: Run Phase 4 (BitNet + Recipe)"
echo "  This combines conv1 FP32 + KD with strong training"
