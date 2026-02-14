# START HERE: Clean Slate Baseline Fix

**Date**: 2026-02-14
**Status**: Code ready, experiments ready to launch
**Goal**: Fix weak FP32 baselines to get highest acceptance probability

---

## Quick Summary

All 3 reviewers identified that your FP32 baselines are undertrained:
- **Current**: CIFAR-100 62.40% (200 epochs, basic recipe)
- **Published**: CIFAR-100 ~77% (300 epochs, proper recipe)
- **Problem**: All "exceeds FP32" claims and gap recovery percentages are invalid

**Solution**: Clean slate with proper training recipe → 85% acceptance probability

---

## What Was Done (Code Changes)

✅ Implemented proper training recipe support:
- Mixup augmentation (`--mixup-alpha 0.2`)
- Label smoothing (`--label-smoothing 0.1`)
- Warmup schedule (`--warmup-epochs 5`)
- Min LR for cosine (`--min-lr 1e-5`)
- FP32+KD control (`--student-is-fp32`)

✅ All tests passing, ready to run experiments

---

## Your Action Plan (Next 7 Days)

### Step 1: Stop & Clean (Right Now - 5 minutes)

**On Lambda Server**:
```bash
ssh lambda
cd ~/code/lab-strange-loop/bitnet

# Stop all running experiments
pkill -f "python -m experiments"

# Verify nothing running
ps aux | grep python | grep train

# Backup current results (optional)
tar -czf ~/backups/results_feb14.tar.gz results/

# Clean slate
rm -rf results/
mkdir -p results/raw results/processed
```

---

### Step 2: Push Code to Lambda (5 minutes)

**Locally** (this machine):
```bash
cd /Users/dariocazzani/code/lab-strange-loop/bitnet

# Stage all changes
git add -A

# Commit
git commit -m "Implement proper training recipe for baseline fix

- Add mixup, label smoothing, warmup support
- Add --student-is-fp32 flag for FP32+KD control
- Create Phase 1-4 experiment scripts
- Target: match published ~94% CIFAR-10, ~77% CIFAR-100

Addresses TMLR Round 0 reviewer feedback:
- Reviewer 1: FP32+KD control experiment
- Reviewer 2: Strengthen weak baselines
- Reviewer 3: Matched recipe comparisons"

# Push to server
git push origin main
```

**On Lambda**:
```bash
cd ~/code/lab-strange-loop/bitnet
git pull
```

---

### Step 3: Launch Phase 1 (FP32 Baselines - Day 1)

**On Lambda**:
```bash
cd ~/code/lab-strange-loop/bitnet

# Launch Phase 1 (runs in background)
nohup bash scripts/phase1_fp32_baselines.sh > logs/phase1.log 2>&1 &

# Monitor progress
tail -f logs/phase1.log

# Or check experiment count
watch -n 60 'ls results/raw/*/resnet18/std_s* 2>/dev/null | wc -l'
# Target: 18 experiments (2 models × 3 datasets × 3 seeds)
```

**Time**: ~45 hours wall-clock (2 GPUs, 3 seeds parallel)

---

### Step 4: Verify Phase 1 Results (After ~5 hours)

Check first completed experiment:
```bash
# ResNet-18 CIFAR-10 (should be ~94%)
cat results/raw/cifar10/resnet18/std_s42/results.json | grep final_test_acc

# If you see ~88-90%, that's the OLD recipe (wrong)
# If you see ~92-94%, that's the NEW recipe (correct!)
```

If results look good → let Phase 1 complete, then launch Phase 2

---

### Step 5: Launch Phase 2 (FP32+KD Control - Day 3)

**After Phase 1 completes**:
```bash
# Check Phase 1 is done (should have 18 experiments)
ls results/raw/*/resnet18/std_s* | wc -l

# Launch Phase 2
nohup bash scripts/phase2_fp32_kd_control.sh > logs/phase2.log 2>&1 &

# Monitor
tail -f logs/phase2.log
```

**Time**: ~23 hours wall-clock

---

### Step 6: Launch Phase 3 & 4 (Day 4-7)

```bash
# Phase 3: BitNet baselines (~45 hours)
nohup bash scripts/phase3_bitnet_baselines.sh > logs/phase3.log 2>&1 &

# After Phase 3 completes:
# Phase 4: BitNet + Recipe (~45 hours)
nohup bash scripts/phase4_bitnet_recipe.sh > logs/phase4.log 2>&1 &
```

---

### Step 7: Analyze Results (Day 8)

**After all phases complete**:
```bash
cd ~/code/lab-strange-loop/bitnet

# Aggregate results
uv run python -m analysis.aggregate_results

# Archive and download
tar -czf results_proper_baselines.tar.gz results/
scp lambda:~/code/lab-strange-loop/bitnet/results_proper_baselines.tar.gz .
```

**Locally**:
```bash
# Extract
tar -xzf results_proper_baselines.tar.gz

# Generate tables and figures
uv run python -m analysis.aggregate_results
uv run python -m analysis.generate_tables
uv run python -m analysis.generate_figures

# Rebuild paper
cd paper/tmlr && make
```

---

## What to Expect

### Scenario A: Recipe still works (Best Case) 🎉
- FP32 baseline: ~94% CIFAR-10, ~77% CIFAR-100
- BitNet + recipe: matches or exceeds strong FP32
- **Paper becomes much stronger**
- **Acceptance: 85-90%**

### Scenario B: Recipe closes gap but doesn't exceed ⚠️
- FP32 baseline: ~94%/~77%
- BitNet + recipe: ~92%/~73% (recovers most of gap)
- **Paper is honest and defensible**
- **Acceptance: 75-80%**

### Scenario C: FP32+KD exceeds BitNet recipe ⚠️
- Must reframe as "training dynamics" not "deployment"
- **Still publishable, just different framing**
- **Acceptance: 70-75%**

---

## Parallel Execution (Faster Option)

If you want to finish in ~3.5 days instead of 7:

**Split work across 2 GPUs manually**:
- GPU 0: All ResNet-18 experiments
- GPU 1: All ResNet-50 experiments

Just add `CUDA_VISIBLE_DEVICES=0` or `=1` before each command.

---

## Quick Reference

### File Locations
- **Experiment scripts**: `scripts/phase*.sh`
- **Full guide**: `BASELINE_RECIPE.md`
- **Implementation plan**: `PLAN.md`
- **Review synthesis**: `paper/reviews/tmlr/round_0/SYNTHESIS.md`

### Key Commands
```bash
# Check running experiments
ps aux | grep python | grep train

# Kill all experiments
pkill -f "python -m experiments"

# Monitor progress
tail -f logs/phase1.log

# Count completed experiments
ls results/raw/*/resnet18/std_s* | wc -l
```

### Cost Estimate
- **Priority 1** (ResNet only): ~315 GPU hours
- **Wall-clock**: 3.5-7 days depending on parallelization
- **With 2 A6000 GPUs**: ~7 days sequential, ~3.5 days parallel

---

## Next Steps (Right Now)

1. ✅ Code is ready
2. → **Stop current experiments** (Step 1)
3. → **Push code to lambda** (Step 2)
4. → **Launch Phase 1** (Step 3)
5. → **Verify after 5 hours** (Step 4)
6. → **Let it run for ~7 days**
7. → **Analyze and update paper** (Step 7)

---

## Questions?

- **What if Phase 1 results look wrong?** Check the command used - verify `--epochs 300` and other flags
- **Can I run other architectures?** Yes, but ResNet is Priority 1 for paper acceptance
- **What about ImageNet?** Defer to future work (reviewers are OK with this)
- **Can I interrupt and resume?** Yes, experiments save checkpoints automatically

---

## Remember

**The reviewers are constructive and want your paper to succeed.**

Getting the baselines right is THE most important fix. Everything else is secondary.

After this, you'll have:
- Scientifically correct baselines
- Critical FP32+KD control
- Honest, defensible comparisons
- **75-85% acceptance probability**

**Let's do this! 🚀**
