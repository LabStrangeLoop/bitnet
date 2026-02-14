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

  CVPR Review: "When Augmentation Fails: Knowledge Distillation for Ternary CNNs"

  Summary (2-3 sentences)

  This paper studies BitNet b1.58 (ternary quantization with fixed {-1,0,+1} weights) applied to CNNs, discovering that data augmentation paradoxically widens the accuracy gap between FP32 and ternary models. The authors propose a simple recipe—keeping the first convolutional layer in FP32 combined with knowledge distillation—that recovers 88% of the accuracy gap on CIFAR-10 and reportedly exceeds FP32 on CIFAR-100 and
  Tiny ImageNet.

  ---
  Strengths (bulleted, specific)

  - Well-designed ablation study (Section 5.1, Table 5): The layer-wise ablation cleanly isolates conv1's contribution (54–74% recovery with only 0.08% parameters). This quantifies a known practice but with rigor.
  - The augmentation paradox is a genuine finding (Section 4, Tables 1–2): The observation that FP32 benefits 4–11× more from augmentation than ternary models is non-obvious and well-documented across 16 configurations.
  - Architecture extension adds breadth (Section 6, Table 8): Testing across 4 CNN families (ResNet, EfficientNet, MobileNetV2, ConvNeXt) × 3 datasets provides useful practitioner guidance. The finding that depthwise separable architectures suffer more (17–73% recovery vs 80–122% for ResNet) is actionable.
  - Reproducibility (Appendix A): The code-to-paper pipeline with deterministic seeds and ~300 GPU-hours of experiments is commendable.
  - Honest limitations section (Section 7): The authors explicitly acknowledge the lack of ImageNet results, single-seed Tiny ImageNet numbers, and no real latency measurements.

  ---
  Weaknesses (bulleted, specific, ordered by severity)

  [FATAL] No ImageNet results. The paper claims practical deployment relevance with "20× compression" and "64× theoretical speedup" but provides no results on ImageNet—the standard benchmark for quantization papers claiming practical impact. The authors mention a 26% accuracy gap on ImageNet (Section 7), which is dramatically worse than prior work (TWN: 4%, TTQ: 2.7%, ReActNet: ~1.5%). CVPR reviewers will immediately flag
   this as disqualifying for a deployment-focused paper. The authors cannot claim their recipe "enables practical ternary CNN deployment" without demonstrating it works at ImageNet scale.

  To address: Include full ImageNet results with the proposed recipe. If the 26% gap persists, the paper's deployment narrative collapses.

  [FATAL] No comparison with modern quantization methods on the same benchmarks. The paper cites LSQ, PACT, EWGS, IR-Net, ReActNet, but never compares against them experimentally. Table 10 shows TTQ achieves -0.36% gap on CIFAR-10 vs this paper's 0.40% gap with the full recipe—but TTQ uses ResNet-56 while this paper uses ResNet-18, making comparison meaningless. These methods achieve 1–5% ImageNet gaps while this paper
  reports 26%.

  To address: Run LSQ, PACT, or TWN on the same architectures/datasets and include direct comparisons. Without this, readers cannot assess whether BitNet b1.58 is a competitive choice.

  [MAJOR] No real inference latency/throughput measurements. The 64× speedup is "theoretical" (Section 3, Table from efficiency.tex assumes "64 ternary ops = 1 FP64 op"). CVPR deployment papers are expected to show actual speedups on target hardware (ARM, edge GPUs, specialized accelerators). Without this, the compression claims are unvalidated.

  To address: Provide inference benchmarks using optimized ternary kernels (BitBLAS, TVM) on at least one hardware target.

  [MAJOR] CIFAR-10/100 and Tiny ImageNet are insufficient benchmarks for deployment claims. CVPR quantization papers (ReActNet, IR-Net, BNext) demonstrate on ImageNet. CIFAR-10's 3.5% gap is not directly comparable to ImageNet gaps reported by other methods. Tiny ImageNet (64×64, 200 classes) is a step toward scale but still far from ImageNet (224×224, 1000 classes).

  [MAJOR] Statistical claims on limited seeds (n=3). The "exceeds FP32" claim on CIFAR-100 (Table 7) reports p=0.028 one-sided, but with n=3 per condition, statistical power is low. The Tiny ImageNet result (Table 9) is from a single seed. The claim that the recipe "exceeds FP32" could be noise.

  To address: Run at least 5 seeds for the key claims, or tone down the statistical claims.

  [MINOR] The contribution framing overstates novelty. "The augmentation paradox" is framed as a core contribution, but it's essentially showing that capacity-limited models can't exploit augmentation—a predictable consequence of the information bottleneck. Similarly, the conv1 finding (Contribution 2) is described as "established practice" that the authors "validate with empirical rigor."

  [MINOR] Why study BitNet b1.58's fixed {-1,0,+1} when TTQ/LSQ exist? The paper acknowledges TTQ achieves smaller gaps (Table 10), and justifies BitNet b1.58 by "deployment simplicity" (integer-only arithmetic). But without demonstrating actual deployment speedups, this justification is hollow. If TTQ achieves 0.36% gap with FP multiply while BitNet b1.58 achieves 26% on ImageNet, why would a practitioner choose BitNet?

  ---
  Missing References and Comparisons

  ┌────────────────────────────────────────┬───────┬──────────────────────────────────────────────────────────────────────────────────────────────┐
  │                 Paper                  │ Venue │                                  Why it should be compared                                   │
  ├────────────────────────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │ ReActNet (Liu et al., 2020)            │ ECCV  │ SOTA binary network, 69.4% ImageNet. Should be baseline for extreme quantization.            │
  ├────────────────────────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Real-to-Binary (Martinez et al., 2020) │ ICLR  │ Strong binary network with progressive training—directly relevant methodology.               │
  ├────────────────────────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │ BNext (Guo et al., 2022)               │ arXiv │ 80.6% ImageNet with binary. Shows SOTA is far beyond this paper's results.                   │
  ├────────────────────────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │ APoT (Li et al., 2020)                 │ ICLR  │ Additive powers-of-two quantization—alternative to ternary with similar deployment benefits. │
  ├────────────────────────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │ LSQ+ (Bhalgat et al., 2020)            │ CVPR  │ Improved LSQ with per-channel learning. Should be compared experimentally.                   │
  └────────────────────────────────────────┴───────┴──────────────────────────────────────────────────────────────────────────────────────────────┘

  The paper cites these but never runs experiments against them. For CVPR, experimental comparison is expected.

  ---
  Questions for Authors

  1. Can you provide ImageNet results with the full recipe (conv1 + KD)? The 26% baseline gap is mentioned—what does the recipe achieve? If the gap remains >10%, how do you justify the "practical deployment" framing?
  2. Why not compare experimentally with TWN or LSQ on ResNet-18/CIFAR? This would directly answer whether BitNet b1.58's simpler quantization scheme is worth the accuracy trade-off.
  3. The augmentation paradox assumes capacity limitation—but wouldn't progressive training (as in ReActNet/Real-to-Binary) help? Did you try multi-stage training where augmentation intensity increases as the network adapts?
  4. For the "exceeds FP32" claims: what are the 95% confidence intervals? With n=3, the error bars matter significantly.
  5. Can you provide any inference timing, even on CPU with naive implementation? This would validate whether the theoretical 64× has any grounding.

  ---
  Scores

  ┌───────────────────────┬───────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │       Criterion       │ Score │                                                            Justification                                                             │
  ├───────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Novelty / Originality │ 4/10  │ The augmentation paradox is mildly interesting but predictable; conv1 importance is known; KD for quantization is established.       │
  ├───────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Significance / Impact │ 3/10  │ Without ImageNet results or real speedups, practical impact is undemonstrated. CIFAR results don't transfer to deployment scenarios. │
  ├───────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Technical Soundness   │ 6/10  │ Methodology is sound for what it tests, but the scope is too narrow for the claims made.                                             │
  ├───────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Clarity / Writing     │ 8/10  │ Well-written, clear structure, honest limitations section. Best aspect of the paper.                                                 │
  ├───────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Experimental Rigor    │ 4/10  │ Thorough on small benchmarks, but missing the scale and comparisons CVPR expects. No hardware validation.                            │
  ├───────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Reproducibility       │ 9/10  │ Excellent—code pipeline, deterministic seeds, all configs documented.                                                                │
  ├───────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Overall               │ 4/10  │ A well-executed small-scale study that does not meet CVPR's bar for deployment-focused quantization papers.                          │
  └───────────────────────┴───────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Decision

  Reject

  The paper presents a well-written empirical study with interesting observations (augmentation paradox, conv1 quantification) but fundamentally fails to meet CVPR's expectations for a deployment-focused quantization paper:

  1. No ImageNet results is disqualifying when claiming deployment relevance
  2. No experimental comparison with modern quantization methods (LSQ, TTQ, ReActNet)
  3. No real inference benchmarks to validate the claimed 20×/64× benefits
  4. CIFAR/Tiny ImageNet are insufficient scale for deployment claims

  The core finding—that BitNet b1.58 achieves 26% ImageNet gap versus <5% for prior methods—actually argues against using this approach, contradicting the paper's deployment narrative.

  ---
  What Would Change Your Decision?

  To move from Reject to Weak Accept, the authors would need to address:

  1. ImageNet results with the full recipe achieving ≤5% accuracy gap (comparable to TWN/TTQ). If the recipe reduces 26% → 5%, that's a real contribution.
  2. Direct experimental comparison with at least TWN and one modern method (LSQ or ReActNet) on the same architecture/dataset.
  3. Any hardware inference benchmark showing actual speedup (even 2–5× on CPU would validate the approach).

  To move to Accept, additionally:

  4. ImageNet gap ≤3% to be competitive with CVPR-level quantization work.
  5. 5+ seeds for the "exceeds FP32" claims to establish statistical reliability.
