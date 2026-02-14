#!/bin/bash
# Phase 2: FP32+KD Control Experiment
# Critical missing baseline identified by all 3 reviewers
# Time: ~23 hours with 2 GPUs

set -e

DATASETS="cifar10 cifar100 tiny-imagenet"
SEEDS="42 123 456"

echo "========================================="
echo "Phase 2: FP32+KD Control (Critical!)"
echo "========================================="
echo "Purpose: Isolate KD benefit from quantization benefit"
echo "Trains FP32 student with KD from FP32 teacher"
echo "========================================="
echo ""

# Check if Phase 1 teachers exist
for dataset in $DATASETS; do
  TEACHER_PATH="results/raw/${dataset}/resnet18/std_s42/best_model.pth"
  if [ ! -f "$TEACHER_PATH" ]; then
    echo "ERROR: Teacher not found at $TEACHER_PATH"
    echo "Run Phase 1 first!"
    exit 1
  fi
done

echo "All teachers found. Starting FP32+KD experiments..."
echo ""

total=0
for dataset in $DATASETS; do
  TEACHER_PATH="results/raw/${dataset}/resnet18/std_s42/best_model.pth"
  echo "Starting resnet18/$dataset FP32+KD (3 seeds in parallel)..."

  for seed in $SEEDS; do
    uv run python -m experiments.train_kd \
      --model resnet18 --dataset $dataset \
      --teacher-path $TEACHER_PATH \
      --student-is-fp32 \
      --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
      --seed $seed &
    total=$((total + 1))
  done

  wait
  echo "  ✓ Completed resnet18/$dataset FP32+KD (3 seeds)"
  echo ""
done

echo "========================================="
echo "Phase 2 Complete!"
echo "Total experiments: $total"
echo "========================================="
echo ""
echo "Next step: Compare FP32+KD with BitNet+recipe"
echo "  If FP32+KD ≥ BitNet+recipe → must reframe contribution"
echo "  If FP32+KD < BitNet+recipe → recipe is effective!"
