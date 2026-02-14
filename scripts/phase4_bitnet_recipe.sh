#!/bin/bash
# Phase 4: BitNet + Recipe (conv1 FP32 + KD)
# Final recipe test with strong training
# Time: ~45 hours with 2 GPUs

set -e

DATASETS="cifar10 cifar100 tiny-imagenet"
MODELS="resnet18 resnet50"
SEEDS="42 123 456"

echo "========================================="
echo "Phase 4: BitNet + Recipe (Full Pipeline)"
echo "========================================="
echo "Recipe: FP32 conv1 + KD + strong training"
echo "========================================="
echo ""

# Check if teachers exist
for model in $MODELS; do
  for dataset in $DATASETS; do
    TEACHER_PATH="results/raw/${dataset}/${model}/std_s42/best_model.pth"
    if [ ! -f "$TEACHER_PATH" ]; then
      echo "ERROR: Teacher not found at $TEACHER_PATH"
      echo "Run Phase 1 first!"
      exit 1
    fi
  done
done

echo "All teachers found. Starting recipe experiments..."
echo ""

total=0
for model in $MODELS; do
  for dataset in $DATASETS; do
    TEACHER_PATH="results/raw/${dataset}/${model}/std_s42/best_model.pth"
    echo "Starting $model/$dataset recipe (3 seeds in parallel)..."

    for seed in $SEEDS; do
      uv run python -m experiments.train_kd \
        --model $model --dataset $dataset \
        --teacher-path $TEACHER_PATH \
        --ablation keep_conv1 \
        --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
        --seed $seed &
      total=$((total + 1))
    done

    wait
    echo "  ✓ Completed $model/$dataset recipe (3 seeds)"
    echo ""
  done
done

echo "========================================="
echo "Phase 4 Complete!"
echo "Total experiments: $total"
echo "========================================="
echo ""
echo "All phases complete! Time to analyze results."
echo ""
echo "Run analysis:"
echo "  uv run python -m analysis.aggregate_results"
echo "  uv run python -m analysis.generate_tables"
echo "  uv run python -m analysis.generate_figures"
