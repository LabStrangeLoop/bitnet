# Proper FP32 Baselines - Implementation Guide

**Date**: 2026-02-14
**Goal**: Match published results (~94% CIFAR-10, ~77% CIFAR-100) with proper training recipe

## Changes Implemented

### Code Updates (Done)
- ✅ Added `mixup_alpha`, `label_smoothing`, `min_lr` to TrainConfig
- ✅ Implemented mixup augmentation in `train_epoch()`
- ✅ Added label smoothing to CrossEntropyLoss
- ✅ Added `min_lr` to cosine scheduler
- ✅ Added `--student-is-fp32` flag to train_kd.py for FP32+KD control
- ✅ Updated KDLoss to accept label_smoothing

---

## Proper Training Recipe

**Strong recipe (to match published ~94%/~77%)**:
```
--epochs 300
--warmup-epochs 5
--min-lr 1e-5
--mixup-alpha 0.2
--label-smoothing 0.1
```

**vs. Old weak recipe (current 88.88%/62.40%)**:
```
--epochs 200
(no warmup, mixup, or label smoothing)
```

---

## Step 1: Stop All Running Experiments

On lambda server:
```bash
ssh lambda
cd ~/code/lab-strange-loop/bitnet

# Check what's running
ps aux | grep python | grep train

# Kill all training
pkill -f "python -m experiments"

# Verify nothing running
ps aux | grep python | grep train
```

---

## Step 2: Backup and Clean Results

```bash
# Archive current results (just in case)
tar -czf ~/backups/results_backup_feb14_$(date +%H%M).tar.gz results/

# Delete all results
rm -rf results/
mkdir -p results/raw results/processed

# Verify clean slate
ls -la results/
```

---

## Step 3: Define Proper Baseline Experiments

### Architectures & Datasets to Cover

**Priority 1** (Core paper results):
- ResNet-18: CIFAR-10, CIFAR-100, Tiny ImageNet
- ResNet-50: CIFAR-10, CIFAR-100, Tiny ImageNet

**Priority 2** (Architecture extension):
- MobileNetV2: CIFAR-10, CIFAR-100, Tiny ImageNet
- EfficientNet-B0: CIFAR-10, CIFAR-100, Tiny ImageNet
- ConvNeXt-Tiny: CIFAR-10, CIFAR-100, Tiny ImageNet

### Seeds
All experiments: seeds 42, 123, 456 (3 seeds for statistical testing)

---

## Step 4: Generate Experiment Commands

### 4a. Priority 1: ResNet Baselines

**FP32 baselines** (strong recipe, 300 epochs):
```bash
#!/bin/bash
# save as: scripts/run_fp32_baselines_resnet.sh

DATASETS="cifar10 cifar100 tiny-imagenet"
MODELS="resnet18 resnet50"
SEEDS="42 123 456"

for model in $MODELS; do
  for dataset in $DATASETS; do
    for seed in $SEEDS; do
      uv run python -m experiments.train \
        --model $model --dataset $dataset \
        --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
        --mixup-alpha 0.2 --label-smoothing 0.1 \
        --seed $seed &
    done
    wait  # Wait for 3 seeds to finish before next dataset
  done
done
```

**Cost**: 2 models × 3 datasets × 3 seeds = 18 experiments × ~5 hours = **90 GPU hours**
**Parallelization**: Run all 3 seeds in parallel on 2 GPUs → ~45 hours wall-clock

---

### 4b. FP32+KD Control (after FP32 baselines complete)

**Critical control experiment** (uses FP32 teacher, trains FP32 student with KD):
```bash
#!/bin/bash
# save as: scripts/run_fp32_kd_control.sh

DATASETS="cifar10 cifar100 tiny-imagenet"
SEEDS="42 123 456"

for dataset in $DATASETS; do
  # Use seed 42 teacher (best from phase 4a)
  TEACHER_PATH="results/raw/${dataset}/resnet18/std_s42/best_model.pth"

  for seed in $SEEDS; do
    uv run python -m experiments.train_kd \
      --model resnet18 --dataset $dataset \
      --teacher-path $TEACHER_PATH \
      --student-is-fp32 \
      --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
      --seed $seed &
  done
  wait
done
```

**Cost**: 3 datasets × 3 seeds = 9 experiments × ~5 hours = **45 GPU hours**

---

### 4c. BitNet Baselines (with strong recipe)

**BitNet with matched strong recipe**:
```bash
#!/bin/bash
# save as: scripts/run_bitnet_baselines.sh

DATASETS="cifar10 cifar100 tiny-imagenet"
MODELS="resnet18 resnet50"
SEEDS="42 123 456"

for model in $MODELS; do
  for dataset in $DATASETS; do
    for seed in $SEEDS; do
      uv run python -m experiments.train \
        --model $model --dataset $dataset --bit-version \
        --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
        --mixup-alpha 0.2 --label-smoothing 0.1 \
        --seed $seed &
    done
    wait
  done
done
```

**Cost**: 2 models × 3 datasets × 3 seeds = 18 experiments × ~5 hours = **90 GPU hours**

---

### 4d. BitNet + Recipe (KD + conv1)

**Full recipe with strong training**:
```bash
#!/bin/bash
# save as: scripts/run_bitnet_recipe.sh

DATASETS="cifar10 cifar100 tiny-imagenet"
MODELS="resnet18 resnet50"
SEEDS="42 123 456"

for model in $MODELS; do
  for dataset in $DATASETS; do
    # Use seed 42 teacher
    TEACHER_PATH="results/raw/${dataset}/${model}/std_s42/best_model.pth"

    for seed in $SEEDS; do
      uv run python -m experiments.train_kd \
        --model $model --dataset $dataset \
        --teacher-path $TEACHER_PATH \
        --ablation keep_conv1 \
        --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
        --seed $seed &
    done
    wait
  done
done
```

**Cost**: 2 models × 3 datasets × 3 seeds = 18 experiments × ~5 hours = **90 GPU hours**

---

## Step 5: Execution Plan (2 GPUs in Parallel)

### Phase 1: FP32 Baselines (Day 1-2)
**Run**: scripts/run_fp32_baselines_resnet.sh
**Time**: ~45 hours wall-clock (2 GPUs, 3 seeds parallel)
**Result**: 18 FP32 baselines with strong recipe

### Phase 2: FP32+KD Control (Day 3)
**Run**: scripts/run_fp32_kd_control.sh
**Time**: ~23 hours wall-clock
**Result**: Critical control experiment (9 runs)

### Phase 3: BitNet Baselines (Day 4-5)
**Run**: scripts/run_bitnet_baselines.sh
**Time**: ~45 hours wall-clock
**Result**: 18 BitNet baselines with strong recipe

### Phase 4: BitNet + Recipe (Day 6-7)
**Run**: scripts/run_bitnet_recipe.sh
**Time**: ~45 hours wall-clock
**Result**: 18 recipe experiments

**Total wall-clock**: ~7 days with 2 GPUs running continuously

---

## Step 6: Parallel Execution Strategy (Faster)

If you want to finish faster, split across 2 GPUs:

**GPU 0**: ResNet-18 experiments
**GPU 1**: ResNet-50 experiments

Modified scripts:
```bash
# GPU 0 - ResNet-18
CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.train \
  --model resnet18 --dataset cifar10 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
  --mixup-alpha 0.2 --label-smoothing 0.1 \
  --seed 42

# GPU 1 - ResNet-50
CUDA_VISIBLE_DEVICES=1 uv run python -m experiments.train \
  --model resnet50 --dataset cifar10 \
  --epochs 300 --warmup-epochs 5 --min-lr 1e-5 \
  --mixup-alpha 0.2 --label-smoothing 0.1 \
  --seed 42
```

With this parallelization:
- Phase 1-4 can run in **~3.5 days** instead of 7

---

## Step 7: Priority 2 Architectures (Optional)

After Phase 1-4 complete and you verify results, optionally run:

**MobileNetV2, EfficientNet, ConvNeXt** (same 4-phase structure)
**Note**: These need architecture-specific learning rates:
- MobileNetV2: `--lr 0.01`
- EfficientNet: `--lr 0.01`
- ConvNeXt: `--optimizer adamw --lr 0.004`

**Cost**: 3 models × 3 datasets × 3 seeds × 4 phases = **~270 GPU hours**

---

## Step 8: Verification

After Phase 1 (FP32 baselines) complete:

```bash
# Check ResNet-18/CIFAR-10 result
cat results/raw/cifar10/resnet18/std_s42/results.json | grep final_test_acc

# Expected: ~94% (currently 88.88%)
```

After Phase 2 (FP32+KD):
```bash
# Check if FP32+KD exceeds FP32
cat results/raw/cifar10/resnet18/fp32_kd_s42/results.json | grep final_test_acc

# Expected: +1-2% over FP32 baseline
```

---

## Step 9: Analysis & Decision

After all phases complete, analyze:

```python
# aggregate_results.py will load all new experiments
uv run python -m analysis.aggregate_results

# Check key comparisons
import pandas as pd
df = pd.read_csv("results/processed/aggregated.csv")

# Compare old vs new FP32 baselines
old_fp32 = df[(df["model"] == "resnet18") & (df["dataset"] == "cifar100") & (df["version"] == "std")]
print(f"Old FP32 CIFAR-100: {old_fp32['final_test_acc'].mean():.2f}%")  # Should be 62.40%

new_fp32 = df[(df["model"] == "resnet18") & (df["dataset"] == "cifar100") & (df["version"] == "std") & (df["epochs"] == 300)]
print(f"New FP32 CIFAR-100: {new_fp32['final_test_acc'].mean():.2f}%")  # Target ~77%

# Check if recipe still exceeds FP32
```

---

## Expected Outcomes

### Scenario A: Recipe still exceeds strong FP32 ✅
- Paper becomes MUCH stronger
- "Recipe achieves X% with proper training, matching/exceeding well-trained FP32"
- **Acceptance probability: 85-90%**

### Scenario B: Recipe closes gap but doesn't exceed ⚠️
- Paper is honest and defensible
- "Recipe recovers X% of gap; augmentation asymmetry explains training dynamics"
- **Acceptance probability: 75-80%**

### Scenario C: FP32+KD exceeds BitNet+recipe ⚠️
- Must reframe contribution as "training dynamics understanding" not "deployment"
- **Acceptance probability: 70-75%** (weaker but honest)

---

## Quick Start

1. **Stop experiments**: `pkill -f "python -m experiments"`
2. **Clean results**: `rm -rf results/; mkdir -p results/raw results/processed`
3. **Create scripts**: Copy Phase 1-4 scripts above to `scripts/` directory
4. **Run Phase 1**: `bash scripts/run_fp32_baselines_resnet.sh`
5. **Monitor**: `watch -n 10 'ls results/raw/*/resnet18/ | wc -l'`
6. **Verify**: Check first results after ~5 hours
7. **Continue**: Launch Phase 2-4 sequentially

---

## Notes

- All experiments use same random seeds (42, 123, 456) for fair comparison
- Warmup is critical for CIFAR-100 (prevents early divergence)
- Mixup + label smoothing together improve robustness
- Min LR prevents complete collapse at end of cosine schedule
- FP32+KD control is THE most important experiment (all 3 reviewers flagged this)

**Expected total time**: ~3.5-7 days depending on parallelization strategy
**Expected total GPU hours**: ~315 hours for Priority 1 (ResNet only)
