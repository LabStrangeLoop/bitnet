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

 ICLR Review: "When Augmentation Fails: Knowledge Distillation for Ternary CNNs"

  Summary

  This paper investigates BitNet b1.58 (ternary quantization) applied to CNNs, demonstrating that data augmentation widens rather than closes the FP32-ternary accuracy gap. The authors propose combining an FP32 first convolutional layer with knowledge distillation, recovering 88% of the gap on CIFAR-10 and claiming to exceed FP32 accuracy on CIFAR-100 and Tiny ImageNet. The work includes evaluation across four CNN
  architectures.

  ---
  Strengths

  - Systematic augmentation study (Section 4, Tables 1-2): The 4×4 grid of augmentation strategies across model-dataset combinations is thorough. The quantification that FP32 benefits 4-11× more than ternary (Table 2) is a useful practical finding.
  - Clean layer-wise ablation (Section 5.1, Table 3): The quantification that conv1 accounts for 58% of accuracy loss with 0.08% of parameters provides empirical rigor to a long-standing convention. Figure 3 presents this clearly.
  - Multi-architecture evaluation (Section 6, Table 8): Testing on ResNet, EfficientNet, MobileNetV2, and ConvNeXt with architecture-specific adaptations (Table 7) goes beyond typical single-architecture studies.
  - Exceptional reproducibility (Appendix A): Full code-to-paper pipeline, deterministic experiments with explicit seeds, and clear directory structure. This is well above average for reproducibility.
  - Honest limitations (Section 7): The authors explicitly acknowledge the 26% ImageNet gap, limited statistical power (n=3), and that the recipe is not plug-and-play. This transparency is appreciated.
  - Clear writing: The paper is well-organized, figures are informative, and the narrative flow (what doesn't work → diagnosis → solution) is logical.

  ---
  Weaknesses

  [MAJOR] Numerical inconsistency between Table 6 and Table 8:
  - Table 6 reports ResNet-18/CIFAR-100 "BitNet + conv1 + KD" = 63.40 ± 0.09 (exceeding FP32 by 1.00%)
  - Table 8 reports ResNet-18/CIFAR-100 "Recipe" = 62.91 with 112% recovery
  - These should be identical but differ by 0.49%. This discrepancy undermines confidence in the reported results.
  - To address: Verify all results, explain discrepancy or correct tables, re-run if necessary.

  [MAJOR] No comparison with modern quantization methods:
  - The paper cites LSQ, PACT, EWGS, ReActNet, and HAWQ but provides no experimental comparison.
  - Table 10 compares only against TTQ and TWN (both 2016-2017), using results from different architectures (ResNet-56 vs ResNet-18).
  - A reviewer would expect direct comparison with at least 2-3 modern QAT methods on the same datasets.
  - To address: Run LSQ, PACT, or learned step-size methods on CIFAR-10/100 with the same training protocol and report accuracy gaps.

  [MAJOR] No ImageNet-scale validation despite practical deployment claims:
  - The abstract promises "practical ternary CNN deployment" and mentions 20× compression.
  - The paper acknowledges a 26.2% ImageNet gap (Table 10) with the CIFAR recipe—far worse than TTQ's 2.7%.
  - For ICLR, claiming "practical deployment" without ImageNet validation is insufficient, especially when prior work achieves much better ImageNet results.
  - To address: Either include ImageNet experiments with adapted training recipes or significantly temper practical deployment claims.

  [MAJOR] Limited novelty beyond combining known techniques:
  - FP32 first/last layers: Standard since XNOR-Net (2016), DoReFa-Net (2016)
  - KD for quantization: QKD (2019), SQAKD (2024), and many others
  - The paper explicitly acknowledges these are "established practice" (line 164) and only claims "first precise quantification"
  - The quantification is useful but may not meet the ICLR bar for novel contribution
  - To address: Strengthen the "augmentation paradox" framing with theoretical analysis (the Appendix B is currently post-hoc) or provide insights that generalize beyond BitNet b1.58.

  [MINOR] The "augmentation paradox" may not be surprising:
  - Binary/ternary networks have long been known to have capacity limitations (cited: Anderson & Berg 2018, Blumenfeld et al. 2019)
  - That augmentation cannot help a capacity-limited model is expected from an information-theoretic perspective
  - The finding that it actively widens the gap is more interesting but follows from FP32 having capacity to exploit augmentation while ternary doesn't
  - To address: Either provide theoretical justification for why this is surprising or reframe as "confirming theoretical predictions with systematic measurement."

  [MINOR] Statistical power limitations:
  - Only 3 seeds per configuration
  - The "exceeds FP32" claim on Tiny ImageNet is from a single seed (line 441)
  - CIFAR-100 significance (p=0.028) would not survive correction for multiple comparisons
  - To address: Run additional seeds, especially for the headline "exceeds FP32" claims.

  ---
  Missing References and Comparisons

  ┌───────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────┐
  │                 Paper                 │                          Why it should be cited/compared                          │
  ├───────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ OCS (Wu et al., 2020)                 │ Mixed-precision via optimal clipping, directly relevant to layer-wise sensitivity │
  ├───────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ ZeroQ (Cai et al., 2020)              │ Data-free quantization baseline for comparing distillation approaches             │
  ├───────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ BRECQ (Li et al., 2021)               │ Block reconstruction for quantization, state-of-art post-training quantization    │
  ├───────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ QDrop (Wei et al., 2022)              │ Dropout for quantization-aware training, relevant to capacity discussion          │
  ├───────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ PokeBNN (Zhang et al., 2022)          │ Recent binary network work with strong results                                    │
  ├───────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ Any 2023-2024 ternary/binary CNN work │ The related work stops at 2022 for CNN quantization methods                       │
  └───────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────┘

  The paper cites BD-Net (AAAI 2026) but provides no comparison despite it being directly relevant (1.58-bit on MobileNet depthwise convolutions).

  ---
  Questions for Authors

  1. Can you explain the discrepancy between Table 6 (63.40%) and Table 8 (62.91%) for ResNet-18/CIFAR-100? These should report the same configuration but differ by 0.49%.
  2. Why not compare directly with LSQ or learned step-size methods? These are the most relevant modern baselines for low-bit quantization and achieve strong results. A comparison would clarify where BitNet b1.58 stands.
  3. The 26% ImageNet gap is dramatic compared to TWN (4%) and TTQ (2.7%). What specific adaptations would you need to close this gap? Is the problem fundamental to BitNet b1.58's fixed scaling, or is it purely a training recipe issue?
  4. Is the "augmentation paradox" actually surprising? From an information-theoretic view, a capacity-limited model cannot exploit additional training signal—so the gap widening seems expected. Can you provide theoretical justification for why this should be considered a paradox rather than a predictable outcome?
  5. For the single-seed Tiny ImageNet result (56.15%), have you verified this is robust? The paper claims the result exceeds FP32 mean + 3σ, but this is a weak statistical test for a headline claim.

  ---
  Scores

  ┌───────────────────────┬───────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │       Criterion       │ Score │                                                                       Justification                                                                       │
  ├───────────────────────┼───────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Novelty / Originality │ 4/10  │ Combines two well-known techniques (FP32 first layer + KD) applied to a specific quantization method; the "paradox" is expected from capacity limitations │
  ├───────────────────────┼───────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Significance / Impact │ 4/10  │ CIFAR-only results limit practical impact; the 26% ImageNet gap undermines deployment claims                                                              │
  ├───────────────────────┼───────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Technical Soundness   │ 6/10  │ Generally solid methodology but Table 6/8 inconsistency and single-seed Tiny ImageNet result raise concerns                                               │
  ├───────────────────────┼───────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Clarity / Writing     │ 8/10  │ Well-organized, clear figures, honest about limitations; the narrative is easy to follow                                                                  │
  ├───────────────────────┼───────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Experimental Rigor    │ 5/10  │ Good within-paper comparisons but missing baselines (no modern QAT methods), limited seeds, no large-scale validation                                     │
  ├───────────────────────┼───────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Reproducibility       │ 9/10  │ Excellent: deterministic seeds, full pipeline, code release promised, clear structure                                                                     │
  ├───────────────────────┼───────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Overall               │ 5/10  │ Well-executed empirical study with limited novelty and insufficient scale for ICLR                                                                        │
  └───────────────────────┴───────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Decision

  Weak Reject

  This is a competently executed empirical paper that provides useful practical findings (augmentation doesn't help, conv1 matters most, recipe for ternary CNNs). However, it does not meet the ICLR acceptance bar because:

  1. The technical contribution is combining two known techniques (FP32 first layer, KD) without novel insight
  2. No comparison with modern quantization methods (LSQ, PACT, etc.)
  3. CIFAR-only experiments are insufficient for practical deployment claims
  4. The numerical inconsistency between tables raises reliability concerns
  5. The "augmentation paradox" framing overclaims what is arguably an expected result

  The paper would be a reasonable fit for a workshop or a more applied venue, but requires substantial strengthening for a main conference paper.

  ---
  What Would Change Your Decision?

  To move from Weak Reject to Borderline/Weak Accept:

  1. Fix Table 6/8 inconsistency and verify all numerical claims
  2. Add comparison with at least 2 modern QAT methods (LSQ, PACT, or EWGS) on the same CIFAR benchmarks
  3. Either include ImageNet results with a competitive gap (<5%) or remove practical deployment claims from abstract/conclusion
  4. Add multi-seed results for Tiny ImageNet "exceeds FP32" claim
  5. Reframe novelty more conservatively: "systematic quantification of known practices for BitNet b1.58" rather than implying the recipe or paradox are novel contributions

  To move to Accept, would additionally need:
  - Theoretical analysis (not post-hoc rationalization) of why augmentation widens the gap
  - Demonstration that insights transfer beyond BitNet b1.58 to other ternary methods
  - Competitive ImageNet results showing the recipe scales
