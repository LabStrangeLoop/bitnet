# Augmentation Experiments Summary

## ✅ Completed Fixes

### 1. Table 2 (paper/tmlr/main.tex) - Fixed "---" entry
- **Issue**: ResNet-50/CIFAR-10 showed "---" because BitNet had negative gain (-0.46%)
- **Fix**: Changed to "†" with footnote: "Augmentation degrades BitNet; ratio undefined."

### 2. Figure 1 Generation (analysis/generate_figures.py)
- **Issue**: ImageNet subplot shown but no data
- **Fix**: Modified `augmentation_gap_plot()` to exclude imagenet dataset

### 3. Command Generation
- **Issue**: Wrong flag `--augmentation` → should be `--augment`
- **Fix**: Regenerated all commands with correct flag

## 📊 Missing Experiments

**Total**: 198 experiments across all architectures
- **CIFAR-10**: 54 experiments (EfficientNet, MobileNetV2, ConvNeXt)
- **CIFAR-100**: 54 experiments (EfficientNet, MobileNetV2, ConvNeXt)  
- **Tiny-ImageNet**: 90 experiments (ALL 5 architectures including ResNet)

**Split**: 99 experiments per GPU

## ⏱️ Time Estimate

- **CIFAR-10/100**: ~45 min/exp × 108 = ~81 GPU-hours (41 hrs on 2 GPUs)
- **Tiny-ImageNet**: ~2 hrs/exp × 90 = ~180 GPU-hours (90 hrs on 2 GPUs)
- **Total**: ~131 wall-clock hours on 2 GPUs (~5.5 days)

## 🚀 How to Run

### Option 1: Loop Version (easiest - copy-paste entire blocks)

**File**: `ALL_augmentation_commands.sh`

Look for the section starting with:
```bash
# GPU 0 (paste into tmux pane 1):
for cmd in \
  ...
```

**Instructions**:
1. Open tmux with 2 panes
2. In pane 1: Copy entire GPU 0 loop block (starts line ~205)
3. In pane 2: Copy entire GPU 1 loop block (starts line ~310)
4. Paste and press Enter

The loops will:
- Print progress: `=== Running: <cmd> ===`
- Handle failures: `FAILED: <cmd>`
- Run all 99 experiments sequentially per GPU

### Option 2: Individual Commands

First 99 lines = GPU 0 commands (after header)
Next 99 lines = GPU 1 commands

Copy-paste one at a time or create your own script.

### Option 3: Run by Dataset (staged approach - recommended!)

**Start with CIFAR-10 (fastest validation)**:
```bash
# Extract just CIFAR-10 commands
grep "cifar10.*CUDA_VISIBLE_DEVICES=0" ALL_augmentation_commands.sh  # GPU 0 (27 exps)
grep "cifar10.*CUDA_VISIBLE_DEVICES=1" ALL_augmentation_commands.sh  # GPU 1 (27 exps)
```

**Then CIFAR-100**:
```bash
grep "cifar100.*CUDA_VISIBLE_DEVICES=0" ALL_augmentation_commands.sh  # GPU 0 (27 exps)
grep "cifar100.*CUDA_VISIBLE_DEVICES=1" ALL_augmentation_commands.sh  # GPU 1 (27 exps)
```

**Finally Tiny-ImageNet** (slowest):
```bash
grep "tiny_imagenet.*CUDA_VISIBLE_DEVICES=0" ALL_augmentation_commands.sh  # GPU 0 (45 exps)
grep "tiny_imagenet.*CUDA_VISIBLE_DEVICES=1" ALL_augmentation_commands.sh  # GPU 1 (45 exps)
```

## 🔍 After Experiments Complete

1. **SCP results from server**:
   ```bash
   # On server
   find results/raw -name "*.json" | tar -czvf results_augmentation.tar.gz -T -
   
   # Locally
   scp lambda:/path/to/results_augmentation.tar.gz .
   tar -xzvf results_augmentation.tar.gz
   ```

2. **Regenerate figures** (this will now exclude ImageNet and show all architectures):
   ```bash
   uv run python -m analysis.aggregate_results
   uv run python -m analysis.generate_figures
   ```

3. **Rebuild papers**:
   ```bash
   cd paper/tmlr && make
   cd paper/neurips-workshop && make
   ```

## 📝 Expected Figure 1 After Experiments

**Before**: Only ResNet-18/50 show all 4 augmentation points; other architectures missing
**After**: All 5 architectures (ResNet-18, ResNet-50, EfficientNet-B0, MobileNetV2, ConvNeXt-Tiny) show complete augmentation curves

**Subplots**:
- CIFAR-10
- CIFAR-100  
- Tiny-ImageNet
- ~~ImageNet~~ (removed)

Each subplot will show gap (FP32 - BitNet) across 4 augmentation levels (basic, cutout, randaug, full) for all 5 architectures.

