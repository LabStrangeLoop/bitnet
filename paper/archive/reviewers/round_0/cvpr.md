# CVPR Review Prompt

You are a senior reviewer for CVPR (Computer Vision and Pattern Recognition). You have 15+ years of experience in computer vision and efficient deep learning, with deep expertise in model compression, quantization, knowledge distillation, and deployment on edge devices. You have reviewed 50+ papers per year for CVPR, ICCV, and ECCV, and have served as Area Chair. You maintain a mental database of current SOTA results on ImageNet and other standard benchmarks.

## About CVPR

- Acceptance rate: ~25%
- Review style: Double-blind, area chairs, strong emphasis on visual results and standard benchmarks
- CVPR values results on standard vision benchmarks -- **ImageNet is essentially required** for papers claiming practical deployment relevance. Novel architectures or training methods must demonstrate clear practical impact. State-of-the-art results or significant efficiency improvements with real measurements are expected.
- What gets rejected: Papers without ImageNet results when claiming practical deployment relevance. Missing comparisons to current SOTA quantization methods. No real latency/throughput measurements when claiming efficiency gains. CIFAR-only papers for deployment-focused work.
- Typical reviewer profile: CV researcher who evaluates papers against current SOTA numbers on ImageNet. Will check if baselines are competitive and if the proposed method advances the field compared to existing quantization methods.

## Your Task

Review the paper pasted below. You have unlimited time. Be thorough, precise, and honest. You are not here to be encouraging -- you are here to determine whether this paper meets the acceptance bar for CVPR.

## Your Review Process

Before writing your review, you must:

1. **Verify every numerical claim** in the paper against the tables and figures. Flag any inconsistency.
2. **Identify the closest 5-10 related papers** in the quantization/efficient ML literature. For each, state whether this paper adequately cites and positions itself against it. Flag missing comparisons -- particularly quantization methods that report ImageNet results.
3. **Assess the experimental methodology**: Are baselines fair? Are hyperparameters tuned with equal effort for all methods? Is the evaluation protocol standard for the vision community?
4. **Evaluate against CVPR's benchmark expectations**: Does this paper provide the scale of results that CVPR reviewers expect? Compare against papers like ReActNet (ECCV 2020), Real-to-Binary (ICLR 2020), IR-Net (CVPR 2020), BNext.
5. **Check deployment claims**: The paper claims 20x compression and 64x speedup. Are these validated with actual measurements? Are there inference benchmarks on any hardware?

## Specific Questions for This Paper at CVPR

- **No ImageNet results.** The paper mentions a preliminary experiment with a 26% accuracy gap but does not include it. For CVPR, this is the single most important issue. How fatal is this?
- **No comparison with modern quantization methods** that report ImageNet results: LSQ, EWGS, PACT, APoT, IR-Net, ReActNet, Real-to-Binary, BNext. These methods achieve 1-5% accuracy gaps on ImageNet. The paper's 3.5% gap on CIFAR is not directly comparable.
- **No real inference latency or throughput measurements.** The paper claims 20x compression and 64x theoretical speedup but provides no hardware validation. For CVPR, does this undermine the practical deployment narrative?
- **CIFAR-10/100 are considered toy benchmarks** by CVPR standards for deployment-focused papers. Tiny ImageNet (64x64, 200 classes) is a step up but still far from ImageNet scale (224x224, 1000 classes).
- The architecture extension (4 families, Section 6) is appreciated but does it compensate for scale limitations?
- The "augmentation paradox" -- is this finding impactful enough for CVPR without larger-scale validation?
- BitNet b1.58's fixed {-1,0,+1} quantization is simpler than learned-scale methods (TTQ, LSQ). Does the paper adequately justify why this constraint is worth studying when more flexible methods exist?

## Required Output Format

### Summary (2-3 sentences)
What the paper does and its main claim.

### Strengths (bulleted, specific)
- Each strength must reference a specific section, table, or figure
- Distinguish between "novel contribution" and "well-executed confirmation of known results"

### Weaknesses (bulleted, specific, ordered by severity)
- Each weakness must be classified as:
  - **[FATAL]**: Would block acceptance regardless of other merits
  - **[MAJOR]**: Significantly weakens the paper, requires substantial revision
  - **[MINOR]**: Should be addressed but wouldn't block acceptance alone
- For each weakness, state what the authors would need to do to address it

### Missing References and Comparisons
List specific papers that should be cited or compared against, with brief justification.

### Questions for Authors
List 3-5 questions you would ask in a rebuttal period.

### Scores

Rate each criterion on 1-10 scale with one-sentence justification:

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Novelty / Originality | /10 | |
| Significance / Impact | /10 | |
| Technical Soundness | /10 | |
| Clarity / Writing | /10 | |
| Experimental Rigor | /10 | |
| Reproducibility | /10 | |
| Overall | /10 | |

### Decision
One of: **Strong Accept**, **Accept**, **Weak Accept**, **Borderline**, **Weak Reject**, **Reject**, **Strong Reject**

### What Would Change Your Decision?
List the minimum set of changes that would move your score up by one tier. Be specific and actionable.

---

## Paper to Review

[PASTE FULL PAPER TEXT HERE]

---

## Reviewer 1

 CVPR Review: "When Augmentation Fails: Knowledge Distillation for Ternary CNNs"

  Summary (2-3 sentences)

  This paper investigates ternary (1.58-bit) quantization of CNNs using the BitNet b1.58 formulation. The authors demonstrate that data augmentation cannot close the ~3.5% accuracy gap between ternary and FP32 models, identify that keeping the first convolutional layer in FP32 recovers 54-74% of the gap, and propose a recipe combining FP32 conv1 with knowledge distillation that recovers 88% of the gap on CIFAR-10 and
  exceeds FP32 accuracy on CIFAR-100 and Tiny ImageNet.

  ---
  Strengths (bulleted, specific)

  - Novel contribution: "Augmentation paradox" finding (Section 4, Table 1, Figure 1): The observation that the FP32-ternary gap remains constant at ~3.5% regardless of augmentation intensity is genuinely surprising and well-demonstrated across four augmentation strategies. This rules out a class of solutions and provides useful negative results.
  - Precise quantification of layer sensitivity (Section 5.1, Table 2, Figure 2): While keeping early layers in FP32 is established practice, the paper provides the first exact quantification for BitNet b1.58: conv1 accounts for 58% gap recovery with only 0.08% of parameters. The asymmetry with layer4 (45% of params, -3% recovery) is informative.
  - Exceeding FP32 on harder tasks (Tables 5, 6, Figure 4): The finding that the conv1+KD recipe achieves +1.0% over FP32 on CIFAR-100 and +1.3% on Tiny ImageNet is notable. The information-theoretic explanation (Appendix B) for why KD helps more on harder tasks is well-reasoned.
  - Cross-architecture analysis (Section 6, Table 8): Testing across four architecture families (ResNet, EfficientNet, MobileNetV2, ConvNeXt) and documenting that depthwise separable architectures show significantly weaker recovery (17-75% vs 80-122% for ResNet) provides practical guidance. The analysis of why depthwise convolutions are particularly vulnerable to ternary quantization is insightful.
  - Reproducibility (Appendix A): The fully reproducible pipeline with determinism guarantees, code release, and single-command regeneration is commendable. The ~300 GPU-hours of experiments across 195+ configurations represents thorough empirical work.
  - Well-executed confirmation: KD variance reduction (Table 3): The observation that KD reduces variance (0.51→0.18 std on CIFAR-10) in addition to improving accuracy is a useful practical finding.

  ---
  Weaknesses (bulleted, specific, ordered by severity)

  - [FATAL] No ImageNet results. The paper claims practical deployment relevance with "20× compression and 64× theoretical speedup" but provides zero ImageNet evaluation. The authors acknowledge (Section 7, Limitations) a preliminary experiment showing a 26% accuracy gap—far larger than TWN (4%) or TTQ (2.7%)—but do not include it. For CVPR, this is disqualifying. The entire deployment narrative rests on CIFAR (32×32,
  10-100 classes) and Tiny ImageNet (64×64, 200 classes), which are considered toy benchmarks for efficiency claims. To address: Include ImageNet results, even if gap is large; compare against published baselines; discuss what recipe adaptations are needed.
  - [FATAL] No comparison with modern quantization methods that report ImageNet results. The paper positions itself against TTQ (2017) but ignores 5+ years of progress: LSQ (ICLR 2020), PACT (ICML 2018), IR-Net (CVPR 2020), ReActNet (ECCV 2020), Real-to-Binary (ICLR 2020), BNext (2022). These methods achieve 1-5% accuracy gaps on ImageNet. The paper's 3.5% gap on CIFAR-10 is simply not comparable. To address: Add a table
  comparing against these methods on ImageNet; explain why BitNet b1.58's simpler formulation is preferable despite larger gaps.
  - [FATAL] No real inference latency or throughput measurements. The 64× theoretical speedup claim (Section 1) is never validated. No measurements on any hardware—CPU, GPU, or edge device. This fundamentally undermines the deployment narrative. To address: Measure actual inference latency on at least one hardware platform; if no optimized ternary kernels exist, state this explicitly and remove deployment claims.
  - [MAJOR] BitNet b1.58's fixed {-1,0,+1} constraint is not justified against learned-scale methods. TTQ learns asymmetric scaling factors and achieves near-zero gaps. The paper argues (Section 7) that learned scales complicate deployment, but this trade-off analysis is superficial. How much deployment complexity does a learned scale add? Is 0.41% vs ~0% accuracy worth it? To address: Quantify the deployment cost of
  learned scales; provide concrete numbers on inference overhead.
  - [MAJOR] CIFAR/Tiny ImageNet scale insufficient for claims. The abstract states the recipe "enables practical ternary CNN deployment." This claim is not supported by 32×32 or 64×64 images. Real deployment scenarios (autonomous driving, mobile vision) operate at ImageNet scale or larger. To address: Either provide ImageNet results or significantly temper the deployment claims.
  - [MINOR] Incomplete ablation on KD hyperparameters. The paper uses standard KD settings (T=4, α=0.9) and briefly mentions (Section 7) that individual changes (T=5-6, α=0.5-0.7) helped but combining them didn't. This is interesting but underexplored. For a paper focused on what works, a proper grid search would strengthen the contribution. To address: Include ablation table for T and α.
  - [MINOR] The "augmentation paradox" is less surprising in context. The finding that regularization doesn't help capacity-constrained models is consistent with prior work on binary networks. The paper could cite Anderson & Berg (2018) more prominently. To address: Discuss relationship to prior theoretical work on binary network capacity limits.
  - [MINOR] Missing error bars on some results. Table 2 (ablation) and Table 6 (Tiny ImageNet recipe) show single runs without error bars. Given the variance discussion elsewhere, this is inconsistent. To address: Run all configurations with 3 seeds.

  ---
  Missing References and Comparisons

  ┌──────────────────────────────────┬────────────┬─────────────────────────────────────────────────────────────────┐
  │              Paper               │   Venue    │                          Why Required                           │
  ├──────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────┤
  │ LSQ (Esser et al.)               │ ICLR 2020  │ Learned step size quantization; SOTA 4-bit and reports ImageNet │
  ├──────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────┤
  │ PACT (Choi et al.)               │ ICML 2018  │ Parameterized clipping; widely used baseline                    │
  ├──────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────┤
  │ IR-Net (Qin et al.)              │ CVPR 2020  │ Information retention for binary networks; reports ImageNet     │
  ├──────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────┤
  │ ReActNet (Liu et al.)            │ ECCV 2020  │ 69.4% ImageNet top-1 with binary; key baseline for 1-bit        │
  ├──────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────┤
  │ Real-to-Binary (Martinez et al.) │ ICLR 2020  │ Strong ImageNet binary baseline                                 │
  ├──────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────┤
  │ BNext (Guo et al.)               │ arXiv 2022 │ 80.6% ImageNet top-1 with binary; current SOTA                  │
  ├──────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────┤
  │ APoT (Li et al.)                 │ ICLR 2020  │ Additive powers-of-two; reports ImageNet                        │
  ├──────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────┤
  │ AdaRound (Nagel et al.)          │ ICML 2020  │ Post-training quantization baseline                             │
  ├──────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────┤
  │ EWGS (Lee et al.)                │ CVPR 2021  │ Per-channel learned scales; reports ImageNet                    │
  └──────────────────────────────────┴────────────┴─────────────────────────────────────────────────────────────────┘

  The paper cites TTQ (2017) and discusses it reasonably, but the quantization field has advanced significantly since then.

  ---
  Questions for Authors

  1. ImageNet: You mention a preliminary experiment with 26% accuracy gap. What was the training recipe? How does this compare to TWN (4%), TTQ (2.7%), or ReActNet (1%)? What prevents you from achieving competitive results?
  2. Why BitNet b1.58 over learned-scale methods? You argue for "deployment simplicity" but don't quantify it. What is the actual inference overhead of TTQ's learned scales? Is it >0.41% accuracy worth of latency?
  3. Real hardware: Have you measured inference on any platform? If ternary kernels don't exist, shouldn't the deployment claims be removed or qualified?
  4. Scaling the recipe: You show KD helps more on harder tasks. Does this trend continue to 1000 classes? Your own citation (Kim et al., QKD) suggests naive KD can hurt when capacity gaps are large.
  5. Augmentation paradox mechanism: You hypothesize the gap reflects "fundamental representation limitation." Can you provide any analysis of what patterns FP32 learns that ternary cannot? Feature visualization or gradient analysis would strengthen this claim.

  ---
  Scores

  ┌───────────────────────┬───────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │       Criterion       │ Score │                                                                      Justification                                                                      │
  ├───────────────────────┼───────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Novelty / Originality │ 5/10  │ "Augmentation paradox" is novel; conv1 sensitivity is confirmation of established practice with precise quantification; overall approach is incremental │
  ├───────────────────────┼───────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Significance / Impact │ 4/10  │ Impact severely limited by absence of ImageNet results; findings may not generalize to practical scales                                                 │
  ├───────────────────────┼───────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Technical Soundness   │ 6/10  │ Methodology is sound within its scope; experiments are well-controlled; but incomplete without ImageNet and real latency                                │
  ├───────────────────────┼───────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Clarity / Writing     │ 8/10  │ Well-written, clear structure, good figures; information-theoretic framing in appendix is elegant                                                       │
  ├───────────────────────┼───────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Experimental Rigor    │ 4/10  │ Thorough on CIFAR/Tiny ImageNet but fatally incomplete for deployment claims; missing SOTA comparisons                                                  │
  ├───────────────────────┼───────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Reproducibility       │ 9/10  │ Excellent: code release, deterministic seeds, single-command pipeline; one of the strongest aspects                                                     │
  ├───────────────────────┼───────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Overall               │ 4/10  │ Solid empirical work but wrong scale for CVPR; would be a good workshop paper                                                                           │
  └───────────────────────┴───────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Decision

  Reject

  The paper presents interesting empirical findings—particularly the augmentation paradox and precise conv1 quantification—but falls critically short of CVPR's expectations for papers claiming practical deployment relevance. The absence of ImageNet results, no comparison with post-2017 quantization methods, and no real inference measurements collectively disqualify the paper. CIFAR-10/100 and Tiny ImageNet are
  insufficient benchmarks for a venue that expects ImageNet as the minimum for efficiency claims.

  The work would be appropriate for a workshop (e.g., ECV Workshop at CVPR) or as an arXiv technical report, but cannot compete with CVPR papers like IR-Net, ReActNet, or BNext that demonstrate their methods at scale with comprehensive baselines.

  ---
  What Would Change Your Decision?

  From Reject → Weak Accept (minimum requirements):
  1. Add ImageNet results with ResNet-18/50, even if the gap is large (e.g., 10-15%). Document why the gap is larger than prior work and what recipe adaptations are needed.
  2. Include comparison table against at least 3 modern methods (LSQ, ReActNet, IR-Net) on ImageNet.
  3. Add inference latency measurements on at least one platform (CPU or GPU), or explicitly remove deployment claims.

  From Weak Accept → Accept (ideal):
  4. Achieve competitive ImageNet results (<5% gap) through recipe adaptation.
  5. Add feature-level distillation experiments (you mention this as future work).
  6. Demonstrate real edge deployment (e.g., on Raspberry Pi, Jetson Nano).


## Reviewer 2


# Binary and ternary quantization for vision: SOTA results and the CVPR bar

**A CVPR-submitted quantization paper reporting only CIFAR-10/100 and Tiny ImageNet results almost certainly falls below the acceptance threshold.** Every accepted quantization paper at CVPR, ECCV, and ICCV since 2019 includes ImageNet-1K evaluation — it is a de facto requirement, not a preference. The field is mature enough that binary networks now exceed **80% top-1 on ImageNet** (BNext-Large) and the newest binary Vision Transformer (BHViT, CVPR 2025) pushes hybrid architectures even further. Ternary methods with full-precision activations reach within **0.9%** of full-precision ResNet-50 accuracy. Below is a comprehensive compilation of current SOTA results, distillation practices, hardware benchmarks, and venue expectations to support your review.

---

## ImageNet-1K accuracy for binary methods on standard backbones

The table below captures the progression of binary (W1/A1) methods on the standard ResNet-18 backbone, where the full-precision baseline is approximately **69.8% top-1**. The gap has narrowed from 18 percentage points in 2016 to roughly 2 points today.

| Method | Year / Venue | W/A | ResNet-18 Top-1 (%) | Notes |
|--------|-------------|-----|---------------------|-------|
| XNOR-Net | 2016 / ECCV | 1/1 | 51.2 | First practical BNN |
| Bi-Real Net | 2018 / ECCV | 1/1 | 56.4 | Identity shortcuts |
| IR-Net | 2020 / CVPR | 1/1 | ~58.1 | Information retention |
| Real-to-Binary | 2020 / ICLR | 1/1 | 65.4 | Two-stage + KD |
| ReCU | 2021 / ICCV | 1/1 | ~61.0 | Reviving dead weights |
| AdaBin | 2022 / — | 1/1 | ~66.4 | Adaptive binary sets |
| ReBNN | 2023 / AAAI | 1/1 | 66.9 | Resilient training |
| BiPer | 2024 / CVPR | 1/1 | ~62.2 | Binary periodic functions |
| A&B BNN | 2024 / arXiv | 1/1 | 66.89 | Hardware-friendly, no FP multiply |
| AB-STE | 2025 / ECML-PKDD | 1/1 | **67.96** | Adaptive blended STE (current R18 SOTA) |

For custom architectures — where the biggest leaps occur — the picture is dramatically better. **ReActNet-A** (ECCV 2020) hit **69.4%** on a MobileNetV1-based binary backbone with only 0.87×10⁸ OPs, becoming the first BNN to exceed full-precision ResNet-18. **BNext-Large** (2022) then shattered the 80% barrier at **80.57% top-1**, using a novel architecture with channel-wise attention and consecutive knowledge distillation. **NAS-BNN** (2024) achieved **70.80%** through neural architecture search at only 100M OPs. The newest entry, **BHViT** (CVPR 2025), is a hybrid CNN-ViT binary architecture that reportedly exceeds ReActNet by **20.6 percentage points** at comparable compute for its larger variant, potentially placing it above 82% — though exact absolute numbers require confirmation from the camera-ready paper.

## Ternary methods and BitNet b1.58 for vision

Ternary quantization (weights ∈ {−1, 0, +1}, ~1.58 bits) with **full-precision activations** delivers substantially higher accuracy than binary methods, though the comparison is not apples-to-apples due to the 32-bit activation overhead.

| Method | Year / Venue | W/A | Backbone | ImageNet Top-1 (%) |
|--------|-------------|-----|----------|---------------------|
| TWN | 2016 / arXiv | 2/32 | ResNet-18 | ~61.8 |
| TTQ | 2017 / ICLR | 2/32 | ResNet-18 | 66.6 |
| TTQ | 2017 / ICLR | 2/32 | ResNet-50 | 74.4 |
| Apprentice | 2018 / ICLR | 2/32 | ResNet-50 | **75.3** (with KD) |
| INQ (2-bit) | 2017 / ICLR | 2/32 | ResNet-18 | ~69.5 |
| He et al. | 2018 / arXiv | 2/32 | ResNet-18 | ~69.3 (residual expansion) |

For reference, **LSQ** at W2/A2 achieves **67.6%** on ResNet-18 and **73.7%** on ResNet-50; **PACT** at W2/A2 reaches **64.4%** and **72.2%** respectively; **APoT** at W3/A3 hits **~69.9%** on ResNet-18. These low-bit methods quantize both weights and activations, making them more hardware-efficient than ternary-weight methods with 32-bit activations.

The application of **BitNet b1.58 to vision** remains nascent. **Nielsen & Schneider-Kamp (2024)** demonstrated that 1.58-bit QAT can match or exceed 16-bit performance on small CNNs (100K–2.2M parameters) tested on CIFAR-10 and CIFAR-100, but reported **no ImageNet results**. **ViT-1.58b** (arXiv 2024) applied BitNet b1.58 to ViT-Large, achieving only **72.27% on CIFAR-10** (well below well-tuned full-precision ViTs at 95%+), with memory reduced to 57 MB from 1.14 GB. **BD-Net** (AAAI 2026) is the most significant entry: it proposes 1.58-bit depthwise convolutions within a BNN framework, achieving up to **9.3 percentage points improvement** over prior BNN methods across CIFAR-10/100, STL-10, Tiny ImageNet, and Oxford Flowers 102, with ImageNet results on MobileNetV1 at 33M OPs. Notably, BD-Net is the **first successful binarization of depthwise separable convolutions**, which prior BNN work entirely excluded. However, the BitNet-for-vision literature conspicuously lacks the large-scale ResNet-18/50 ImageNet benchmarks that the traditional ternary quantization literature established years ago.

## Knowledge distillation is essential, not optional

**Every modern high-accuracy BNN relies on knowledge distillation.** This is not an auxiliary technique — it is a core component of the training pipeline. The accuracy progression from ~51% (XNOR-Net) to 80.57% (BNext) was driven as much by better distillation strategies as by architectural innovation.

ReActNet uses a **distributional loss** that enforces the binary network to match the output distribution of a real-valued teacher. BNext introduced **Consecutive Knowledge Distillation (CKD)** with a "Knowledge Complexity" metric for teacher selection, discovering that simply using the most accurate teacher can paradoxically cause overfitting in high-accuracy BNNs. Their gap-sensitive ensemble of EfficientNet-B0 and EfficientNet-B2 teachers, combined with progressive training rounds, contributed approximately **1.67% top-1** improvement — the difference between a strong and a landmark result.

Recent advances include **QFD** (AAAI 2023), which trains a quantized teacher representation rather than using full-precision features, outperforming QKD and SPEQ across 2/3/4-bit settings on ImageNet. **SQAKD** (AISTATS 2024) introduced self-supervised distillation requiring no labels, improving PACT by up to **15.86%** on Tiny ImageNet in worst-case scenarios and LSQ by up to 12.03%. **SKD-BNN** (2024) exploits latent full-precision weights as implicit teachers, gaining **+1.6%** on ImageNet with an IR-Net backbone. The typical KD improvement for binary networks on ImageNet ranges from **1–5% top-1**, with diminishing returns as base accuracy increases.

Best practices as of early 2026: use feature-level distillation (not just logit matching), select teachers based on knowledge complexity rather than raw accuracy, employ progressive or consecutive distillation rounds, and consider self-supervised approaches (SQAKD) for scenarios where labeled data is limited.

## Hardware deployment benchmarks remain sparse but improving

Real-device latency measurements for binary networks are far less common than accuracy benchmarks, but several reference points exist:

- **daBNN** (ACM MM 2019): The first optimized BNN inference framework for ARM achieved **61.3 ms per image** (43.2 ms with stem optimization) for Bi-Real Net 18 on ImageNet running single-threaded on a Google Pixel 1, delivering **6–23× speedup** over the BMXNet baseline
- **BNN-Clip** (Sensors 2024): Optimized ARM NEON inference on Raspberry Pi 3B/4B achieved **1.3–2.4× speedup** over daBNN and Larq Compute Engine for BiRealNet and ReActNet architectures
- **FINN** (FPGA 2017): On a Xilinx ZC706 embedded FPGA (<25W), BNNs achieved **0.31 μs latency** on MNIST (12.3M classifications/second) and **283 μs on CIFAR-10** at 66 TOPS peak binary throughput
- **LDF-BNN** (2024): An FPGA-accelerated BNext variant achieved **72.6 FPS** at 1826 GOPs with 72.23% ImageNet accuracy
- **CBin-NN** (Electronics 2024): On an STM32F746 microcontroller (ARM Cortex-M7, 216 MHz), BNN inference ran **3.6× faster** than TF-Lite Micro with 7.5× less weight memory

**PokeBNN** (IBM, CVPR Workshops 2022) proposed the **ACE (Arithmetic Computation Effort)** metric as a hardware-agnostic energy proxy, estimating that one binary MAC costs approximately **1/64** the energy of an 8-bit MAC. Their modified ResNet-50 achieved 75.6% at 7.8 ACE. However, the field still lacks standardized latency benchmarks across methods on common hardware platforms — a gap that the 3rd ICCV 2025 Workshop on Binary and Extreme Quantization is working to address.

## The CVPR acceptance bar demands ImageNet-1K and broad comparisons

An exhaustive review of accepted quantization papers at CVPR, ECCV, and ICCV from 2019–2025 reveals a consistent and non-negotiable pattern: **ImageNet-1K is required for any classification-oriented quantization paper**. Not a single accepted main-conference paper in this period was found that reports only CIFAR or small-dataset results.

The specific expectations for a competitive submission include:

- **Datasets**: ImageNet-1K classification as the primary benchmark. CIFAR-10/100 may appear as supplementary ablations. Downstream tasks (COCO detection, ADE20K segmentation) are a strong differentiator but not always mandatory for binary/ternary-specific work
- **Architectures**: At minimum, ResNet-18 plus one additional backbone (ResNet-50 or MobileNetV2). From 2023 onward, Vision Transformer results (ViT, DeiT, Swin) are increasingly expected
- **Baselines**: Typically **8–15+ comparison methods**. BiPer (CVPR 2024) compared against XNOR-Net, Bi-Real Net, IR-Net, PCNN, ReActNet, AdaBin, ReCU, and others. Omitting well-known baselines invites immediate reviewer criticism
- **Ablations**: Component-level ablation studies, convergence analysis, and visualization of learned representations are standard

A paper reporting only CIFAR-10/100 and Tiny ImageNet raises three fatal concerns for reviewers. First, **scalability is undemonstrated**: CIFAR images are 32×32 pixels with 10–100 classes, while ImageNet demands 224×224 resolution across 1000 classes — the quantization challenges are qualitatively different. Second, **comparison with prior work is impossible**: nearly all SOTA binary and ternary methods report their headline numbers on ImageNet, so a paper without these results cannot be positioned against the field. Third, **it signals limited engineering effort**: running ImageNet experiments is a baseline expectation, and its absence suggests the method may not scale or the authors lack compute access — neither of which justifies acceptance at a top venue.

## Conclusion

The binary quantization landscape in early 2026 is remarkably mature. The accuracy frontier stands at **80.57%** (BNext-Large) for custom architectures and **~68%** for standard ResNet-18 backbones, with binary Vision Transformers (BHViT, CVPR 2025) potentially pushing beyond 82%. Ternary methods with full-precision activations reach **75.3%** on ResNet-50, while BitNet b1.58 for vision remains largely unexplored at ImageNet scale — a genuine research opportunity, but one that demands ImageNet evaluation to be taken seriously. Knowledge distillation is not optional but integral to every competitive BNN, contributing 1–5% accuracy. Hardware deployment results exist but remain fragmented across platforms.

For your review: a CVPR submission on binary or ternary quantization that omits ImageNet-1K experiments does not meet the established bar. The community expects ImageNet results with standard backbones, comparison against at least 8–10 baselines (including ReActNet, BNext, and recent 2024–2025 methods), and ideally results on both CNNs and ViTs. A paper focusing solely on CIFAR and Tiny ImageNet would need an extraordinarily compelling theoretical contribution or a fundamentally different application domain to overcome this deficit — and even then, reviewers will almost certainly request ImageNet results in revision.