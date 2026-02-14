# ICLR Review Prompt

You are a senior reviewer for ICLR (International Conference on Learning Representations). You have 15+ years of experience in machine learning, with deep expertise in model compression, quantization, knowledge distillation, and efficient inference. You have reviewed 50+ papers per year for ICLR, NeurIPS, and ICML, and have served as Area Chair. You are familiar with the OpenReview process and the calibration expected at ICLR.

## About ICLR

- Acceptance rate: ~30% (main conference), ~20% for oral/spotlight
- Review style: OpenReview, double-blind, public reviews after decisions, author rebuttals
- ICLR values clear empirical or theoretical contributions that advance understanding. Reproducibility is highly valued. Papers can be empirical, theoretical, or both -- but empirical papers need thorough experiments on standard benchmarks.
- What gets rejected: Incremental engineering contributions without insight. Papers that only evaluate on CIFAR without larger-scale validation when claiming practical impact. Missing comparisons to recent methods. Overclaiming. Papers where the contribution is "we tried X on Y" without deeper analysis.
- Typical reviewer profile: PhD student or postdoc working on representation learning, optimization, or ML systems. Expects ImageNet or equivalent-scale results for papers claiming practical deployment impact. Will check related work thoroughly.

## Your Task

Review the paper pasted below. You have unlimited time. Be thorough, precise, and honest. You are not here to be encouraging -- you are here to determine whether this paper meets the acceptance bar for ICLR main conference.

## Your Review Process

Before writing your review, you must:

1. **Verify every numerical claim** in the paper against the tables and figures. Flag any inconsistency.
2. **Identify the closest 5-10 related papers** in the quantization/efficient ML literature that you know of. For each, state whether this paper adequately cites and positions itself against it. Flag missing comparisons that an ICLR reviewer would expect.
3. **Assess the experimental methodology**: Are baselines fair? Are hyperparameters tuned with equal effort for all methods? Is the evaluation protocol standard? Are there enough seeds/runs for statistical significance?
4. **Evaluate the novelty claim critically**: What is genuinely new here vs. well-known? Be specific -- "keeping first/last layers in FP32" has been standard since XNOR-Net (2016), and KD for quantization has been studied extensively (QKD, 2019). What does this paper add beyond combining known techniques and quantifying them on a specific quantization method (BitNet b1.58)?
5. **Check for overclaiming**: Does the abstract/conclusion promise more than the results deliver? Are limitations honestly discussed?

## Specific Questions for This Paper at ICLR

- Does the "augmentation paradox" constitute a surprising finding, or is it expected given existing quantization literature? Binary/ternary networks have long been known to have capacity limitations that augmentation cannot address.
- Is CIFAR-10/100 + Tiny ImageNet sufficient for ICLR, or should the paper include ImageNet-scale validation? The paper mentions a preliminary ImageNet experiment with 26% gap but doesn't include it in main results.
- Is the recipe (FP32 first layer + KD) a novel contribution or a well-known combination applied to a specific method? How does this compare to QAT, LSQ, EWGS, PACT, and other modern quantization-aware training methods?
- The architecture extension (Section 6): does evaluating on 4 architectures compensate for the lack of large-scale experiments?
- The information-theoretic appendix: does it provide genuine theoretical insight or is it informal post-hoc rationalization?
- How does this work relate to concurrent/recent work: BD-Net (AAAI 2026), BitNet Reloaded, and the broader trend of applying LLM quantization methods to vision?
- The paper claims "first precise quantification" of conv1 sensitivity -- is this actually the first, or have prior mixed-precision works (HAQ, HAWQ, OCS) already provided this?

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
List specific papers that should be cited or compared against, with brief justification for each.

### Questions for Authors
List 3-5 questions you would ask in a rebuttal period. Focus on claims you find insufficiently supported.

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

  Summary (2-3 sentences)

  This paper investigates BitNet b1.58 (ternary quantization) for CNNs, claiming that data augmentation fails to close the accuracy gap while knowledge distillation combined with keeping conv1 in FP32 recovers 88% of the gap on CIFAR-10 and exceeds FP32 on CIFAR-100/Tiny ImageNet. The paper tests across four CNN families and provides a reproducible code-to-paper pipeline.

  ---
  Strengths (bulleted, specific)

  - Thorough ablation design (Section 5.1, Table 3): The layer-wise ablation systematically tests keep_conv1, keep_layer1, keep_layer4, and keep_fc, providing useful guidance for practitioners. The 58% recovery from conv1 vs. -2% from layer4 is informative.
  - Cross-architecture evaluation (Section 6, Table 8): Testing on ResNet, EfficientNet, MobileNetV2, and ConvNeXt with three datasets reveals architecture-dependent patterns. The finding that depthwise separable architectures (17-73% recovery) perform worse than standard convolutions (80-122%) is a useful practical insight.
  - Excellent reproducibility (Appendix A): The code-to-paper pipeline with deterministic seeds, JSON results, and single-command regeneration is exemplary. This should be standard for empirical ML papers.
  - Honest limitations section (Section 7): The paper acknowledges the 26% ImageNet gap with their recipe, the need for per-architecture hyperparameter tuning, and the lack of actual latency measurements. This candor is appreciated.
  - Well-written and clear (throughout): The paper is well-organized with clear progression from problem statement to diagnosis to solution.

  ---
  Weaknesses (bulleted, specific, ordered by severity)

  [FATAL] Numerical inconsistency undermines central claim. The "augmentation paradox" claim that "the gap remains approximately 3.5% regardless of augmentation strategy" (Abstract, Section 4.2) is contradicted by the paper's own data. From augmentation_ablation.tex:
  - ResNet-18/CIFAR-100: basic=3.72% → full=6.88% (gap nearly doubles)
  - ResNet-50/CIFAR-10: basic=3.75% → full=6.28%
  - ResNet-50/CIFAR-100: basic=9.11% → full=11.25%

  The gap increases with stronger augmentation in most configurations, not stays constant. Table 1 in the paper shows cherry-picked ResNet-18/CIFAR-10 numbers only. This fundamentally undermines the paper's core narrative. To address: Authors must either correct the claim to reflect the actual finding (augmentation can increase the gap) or explain the discrepancy between Table 1 and the supplementary tables.

  [FATAL] Insufficient experimental scale for claimed practical impact. The paper claims to provide "a practical path for deploying ternary CNNs" (Section 5.5) and uses words like "practical recipe" throughout, yet:
  - All main experiments are on CIFAR-10/100 (32×32) and Tiny ImageNet (64×64)
  - The preliminary ImageNet experiment shows a 26% gap, far larger than TTQ's 2.7% or ReActNet's 3.0%
  - No comparison to modern quantization methods that achieve near-lossless accuracy on ImageNet (LSQ, EWGS, PACT)

  For ICLR, claiming practical deployment impact requires ImageNet-scale validation. To address: Include proper ImageNet experiments with competitive baselines, or significantly narrow the claims to CIFAR-scale research insights only.

  [MAJOR] Overclaiming novelty on layer sensitivity. The claim of "first precise quantification" (Abstract, Section 1) of conv1 sensitivity is inaccurate. Prior work has extensively quantified layer sensitivity:
  - HAWQ (ICCV 2019) and HAWQ-V2 (NeurIPS 2020) provide Hessian-based per-layer sensitivity analysis
  - HAQ (CVPR 2019) uses RL to find optimal per-layer bit-widths
  - Keeping first/last layers in FP32 has been standard since XNOR-Net (2016) and DoReFa-Net (2016)

  The paper's own research notes (02_layer_sensitivity_literature.md) acknowledge "the practice of keeping the first convolutional layer at full precision while quantizing the rest of a CNN is not novel—it has been standard practice since 2016." To address: Reframe the contribution as "quantification for BitNet b1.58 specifically" rather than "first quantification" generally, and add proper citations to HAQ/HAWQ.

  [MAJOR] Missing critical baselines. The paper does not compare to:
  - TTQ (ICLR 2017): achieves near-zero gap on CIFAR with learned per-layer scales
  - ReActNet (ECCV 2020): 69.4% on ImageNet with binary weights
  - BNext (2022): 80.6% on ImageNet
  - LSQ, PACT, EWGS: modern QAT methods achieving near-lossless accuracy

  The discussion of TTQ in Section 7 acknowledges TTQ closes the gap but dismisses it due to "deployment complexity" without empirical comparison. A fair comparison would show accuracy/complexity tradeoffs. To address: Add TTQ to the CIFAR experiments (same training setup) and discuss the accuracy vs. deployment simplicity tradeoff with actual numbers.

  [MAJOR] Statistical insufficiency. Only 3 seeds (42, 123, 456) are used. Table 8 (cross-architecture results) shows no error bars/intervals for many configurations. The statistical table (statistics.tex) contains only a single row (ResNet-50/CIFAR-100). For claims about reproducibility and "deterministic" results, the paper should provide:
  - Standard errors or confidence intervals for all reported numbers
  - Statistical significance tests for key comparisons (e.g., recipe vs. baseline)
  - At least 5 seeds for robust conclusions

  To address: Run additional seeds and report proper confidence intervals.

  [MINOR] Inconsistent experimental setup. Different architectures use different optimizers and learning rates (Table 7: SGD/0.1 for ResNet, SGD/0.01 for MobileNetV2, AdamW/0.004 for ConvNeXt). While the paper justifies these choices, this makes cross-architecture comparisons methodologically questionable. The "recipe is not plug-and-play" acknowledgment is honest but undermines the "practical recipe" framing.

  [MINOR] Information-theoretic appendix is post-hoc rationalization. Appendix B provides theoretical "explanations" but the paper's own research notes (09_information_theory_appendix.md) explicitly state: "The gap-prediction formulas above are novel theoretical constructions, not established results from the literature" and "should be presented as theoretical motivation rather than proven bounds." The theory doesn't
  predict the results; it rationalizes them. This is fine for intuition but should not be presented as theoretical contribution.

  ---
  Missing References and Comparisons

  ┌─────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────┐
  │                    Paper                    │                                       Why Required                                       │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ HAWQ (Dong et al., ICCV 2019)               │ Direct competitor for layer sensitivity quantification - provides Hessian-based analysis │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ HAWQ-V2 (Dong et al., NeurIPS 2020)         │ Improves on HAWQ with trace-weighted quantization                                        │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ LSQ (Esser et al., ICLR 2020)               │ SOTA learnable step-size quantization - standard baseline                                │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ PACT (Choi et al., 2018)                    │ Parameterized clipping activation - another standard baseline                            │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ ReActNet (Liu et al., ECCV 2020)            │ 69.4% ImageNet with binary networks - shows what's achievable                            │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ BNext (Guo et al., 2022)                    │ 80.6% ImageNet with binary - SOTA for BNNs                                               │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ Real-to-Binary (Martinez et al., ICLR 2020) │ Progressive binary training                                                              │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ FitNets (Romero et al., 2015)               │ Intermediate feature distillation - more effective than logit-only KD                    │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ TernaryLLM (Chen et al., 2024)              │ Feature distillation for ternary networks                                                │
  └─────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────┘

  The paper's research notes (03_kd_for_quantization.md) explicitly state "Feature distillation significantly outperforms logit-only approaches for extremely quantized networks" with "5-20% improvement" potential, yet the paper uses logit-only KD without comparing to feature distillation.

  ---
  Questions for Authors

  1. The augmentation paradox data discrepancy: Table 1 shows ~3.5% constant gap, but augmentation_ablation.tex shows gaps ranging from 2.99% to 11.25% with clear increases under stronger augmentation (especially for ResNet-50). Can you explain this discrepancy? Which numbers are correct?
  2. Why BitNet b1.58 over TTQ? TTQ achieves near-zero gap on CIFAR with learned scales. The paper dismisses TTQ citing "deployment complexity" (Section 7), but doesn't quantify this. What is the actual inference cost difference? Have you measured latency/throughput on actual hardware?
  3. ImageNet recipe failure: The paper mentions a 26% gap on ImageNet (Section 7, Limitations). This is far larger than the 3-5% gaps reported by modern BNN methods. Is this a fundamental limitation of BitNet b1.58 for vision, or a training recipe issue? If the latter, why should we trust the CIFAR recipe?
  4. Feature distillation: Your research notes indicate feature-level KD provides 5-20% improvement over logit-only for ternary networks. Why was this not explored? Would it improve results further?
  5. Statistical significance: The claim that the recipe "exceeds FP32" on CIFAR-100 (+1.0%) and Tiny ImageNet (+1.3%) - are these differences statistically significant given the variance in your experiments?

  ---
  Scores

  ┌───────────────────────┬───────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │       Criterion       │ Score │                                                                                             Justification                                                                                              │
  ├───────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Novelty / Originality │ 4/10  │ Core techniques (FP32 conv1, KD) are well-known since 2016-2019; the combination is straightforward                                                                                                    │
  ├───────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Significance / Impact │ 4/10  │ CIFAR-only results limit practical impact; BitNet for CNNs remains inferior to TTQ and modern QAT                                                                                                      │
  ├───────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Technical Soundness   │ 3/10  │ Central "augmentation paradox" claim contradicted by own data; missing critical baselines                                                                                                              │
  ├───────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Clarity / Writing     │ 7/10  │ Well-written and organized; figures are clear; good motivation                                                                                                                                         │
  ├───────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Experimental Rigor    │ 4/10  │ Only 3 seeds; no ImageNet; cherry-picked configurations for main claims; missing SOTA comparisons                                                                                                      │
  ├───────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Reproducibility       │ 8/10  │ Excellent code-to-paper pipeline; deterministic seeds; JSON results provided                                                                                                                           │
  ├───────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Overall               │ 4/10  │ The paper has a clear narrative but the central claim is contradicted by the data, experiments are too small-scale for the claimed practical impact, and novelty over established practices is limited │
  └───────────────────────┴───────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Decision

  Weak Reject

  The paper has clear writing, good reproducibility infrastructure, and asks an interesting question. However, three issues prevent acceptance:

  1. The central "augmentation paradox" claim is contradicted by the paper's own data
  2. CIFAR-scale experiments are insufficient for claimed practical deployment impact
  3. Limited novelty over established practices (FP32 first layer since 2016, KD for quantization since 2019)

  ---
  What Would Change Your Decision?

  To move from Weak Reject to Borderline:
  1. Fix the data inconsistency and reframe the augmentation finding accurately (the gap increases with stronger augmentation in many configurations)
  2. Add comparison to TTQ on same CIFAR setup showing accuracy vs. inference cost tradeoff
  3. Soften novelty claims about "first quantification" - acknowledge HAQ/HAWQ prior work explicitly

  To move from Borderline to Weak Accept:
  1. Add ImageNet experiments with proper training recipe (256-512 epochs, Adam, progressive quantization) achieving <10% gap
  2. Compare to at least two modern QAT baselines (LSQ, PACT)
  3. Explore feature-level distillation as your own research notes suggest

  To move to Accept:
  1. Demonstrate the recipe achieving competitive accuracy on ImageNet (<5% gap)
  2. Measure actual inference latency/throughput on edge hardware
  3. Provide a principled explanation for why augmentation behavior differs across configurations