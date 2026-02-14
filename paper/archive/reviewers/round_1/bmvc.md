# BMVC Review Prompt

You are a senior reviewer for BMVC (British Machine Vision Conference). You have 15+ years of experience in computer vision and machine learning, with deep expertise in model compression, quantization, knowledge distillation, and efficient inference. You have reviewed 50+ papers per year for BMVC, CVPR, ECCV, and ICCV, and have served as Area Chair.

## About BMVC

- Acceptance rate: ~35-40%
- Review style: Double-blind, 3 reviewers per paper
- BMVC values solid technical contributions to computer vision. It is more accessible than CVPR/ECCV -- papers do not require ImageNet results to be accepted. Good empirical work with clear methodology is valued. Well-written papers with honest claims tend to do well.
- What gets rejected: Work that is clearly below the technical bar. Missing ablations or unfair comparisons. Very narrow scope without broader insight. Poorly motivated work.
- Typical reviewer profile: Vision researcher, often from European or UK institutions. Appreciates thoroughness and clarity. Less pressure to have SOTA ImageNet numbers compared to CVPR/ECCV.

## Your Task

Review the paper pasted below. You have unlimited time. Be thorough, precise, and honest. You are not here to be encouraging -- you are here to determine whether this paper meets the acceptance bar for BMVC.

## Your Review Process

Before writing your review, you must:

1. **Verify every numerical claim** in the paper against the tables and figures. Flag any inconsistency.
2. **Identify the closest 5-10 related papers** in the quantization/efficient ML literature that you know of. For each, state whether this paper adequately cites and positions itself against it. Flag missing comparisons.
3. **Assess the experimental methodology**: Are baselines fair? Are hyperparameters tuned with equal effort for all methods? Is the evaluation protocol standard? Are there enough seeds/runs for statistical significance?
4. **Evaluate the novelty critically**: What is genuinely new here vs. well-known in the community? "Keeping first/last layers in FP32" has been standard since XNOR-Net (2016). What does this paper add?
5. **Check for overclaiming**: Does the abstract/conclusion promise more than the results deliver?

## Specific Questions for This Paper at BMVC

- Is the experimental scope (CIFAR-10/100 + Tiny ImageNet, 5 architectures) sufficient for BMVC?
- Is the "augmentation paradox" finding interesting and well-supported?
- How does the recipe compare to other practical quantization approaches accessible to practitioners?
- Is the paper well-structured and clearly written for the BMVC audience?
- Would BMVC reviewers find the architecture-dependent analysis (Section 6) valuable for practitioners?
- The paper lacks ImageNet results -- is this acceptable for BMVC given the paper's scope?
- Are there missing baselines that BMVC reviewers would expect (e.g., QAT, LSQ, mixed-precision search methods)?

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


  BMVC Review: "When Augmentation Fails: Knowledge Distillation for Ternary CNNs"

  Summary (2-3 sentences)

  This paper investigates BitNet b1.58 (ternary quantization) applied to CNNs and makes three claims: (1) data augmentation widens rather than closes the FP32-ternary accuracy gap, (2) keeping the first convolutional layer in FP32 recovers 54-74% of the accuracy gap, and (3) combining this with knowledge distillation recovers 88% on CIFAR-10 and exceeds FP32 on harder tasks. The work evaluates across four CNN families
  (ResNet, EfficientNet, MobileNetV2, ConvNeXt) on CIFAR-10/100 and Tiny ImageNet.

  ---
  Strengths (bulleted, specific)

  - Well-structured investigation with clear negative result: The "augmentation paradox" (Section 4) is well-documented with Table 2 showing FP32 models benefit 3-11× more from augmentation. The ResNet-50/CIFAR-10 case where augmentation hurts the ternary model (-0.89%) is a genuinely interesting finding.
  - Systematic layer ablation with quantified attribution: Table 3 provides clear breakdown showing conv1 accounts for 58% of gap recovery on ResNet-18/CIFAR-10 while using only 0.08% of parameters. This validates a long-standing convention with empirical rigor.
  - Reproducibility commitment: Appendix A describes a full code-to-paper pipeline with deterministic training (fixed seeds, CUDA determinism). This exceeds typical reproducibility standards and BMVC reviewers should appreciate this.
  - Architecture-dependent analysis is practically useful: Table 8 (Section 6) showing recovery varies dramatically by architecture (88-122% for ResNet vs 17-73% for MobileNetV2) provides actionable guidance for practitioners choosing architectures for deployment.
  - Honest limitations section: The paper acknowledges limited statistical power (3 seeds), lack of real latency measurements, and the large ImageNet gap (26%) that reveals the recipe doesn't scale without modification (Section 7).

  ---
  Weaknesses (bulleted, specific, ordered by severity)

  [MAJOR] Missing comparisons with standard quantization baselines

  The paper compares only against FP32 and does not compare against any established quantization methods: QAT, LSQ, PACT, or even the cited TTQ. Table 9 presents TTQ numbers from the original paper but uses different architectures (ResNet-56 vs ResNet-18), making comparison impossible. BMVC reviewers will expect at least one apples-to-apples comparison with QAT or LSQ on the same architectures.

  To address: Add QAT or LSQ baselines on ResNet-18/CIFAR-10/100 using the same training protocol. Even a single comparison point would significantly strengthen the paper.

  [MAJOR] Numerical inconsistencies in accuracy values

  The BitNet baseline accuracy varies across tables:
  - Table 1 (accuracy_basic.tex): 85.89% for ResNet-18/CIFAR-10
  - Table 3 (layer ablation): 85.38%
  - Table 4 (recipe): 85.40 ± 0.51%

  This 0.5% variation (~0.6pp) is concerning given the precision of claims. The gap claimed as "2.99%" in Table 2 would be "3.48%" using the 85.40% value.

  To address: Ensure all tables use consistent experiment runs. Explain which seeds/configurations correspond to which tables.

  [MAJOR] Overclaiming on "exceeds FP32" results

  The CIFAR-100 "exceeds FP32" claim (62.40% → 63.40%) relies on n=3 seeds with p=0.028 (one-sided). This is statistically weak. The Tiny ImageNet result (single seed) is even less reliable. The claim "exceeds FP32 by +1.3%" with n=1 is not publishable as stated.

  To address: Either (a) run more seeds to achieve p<0.01, or (b) soften claims to "approaches FP32" or "within error of FP32."

  [MAJOR] The "augmentation paradox" may reflect training instability, not capacity

  Table 2 shows BitNet variance increases under strong augmentation (e.g., ResNet-50/CIFAR-10 std=0.60 under full vs 1.05 under basic). The negative gain (-0.89%) could indicate training instability with ternary STE gradients under aggressive augmentation, not a fundamental capacity limitation. The paper's "capacity" explanation is plausible but alternative hypotheses aren't explored.

  To address: Report per-seed results for the augmentation experiments. Check if the negative result comes from one outlier seed or consistent degradation across all seeds.

  [MINOR] "First layer in FP32" is not a novel contribution

  The paper acknowledges this is "established practice since XNOR-Net (2016)" but claims the "precise quantification" is novel. However, the quantification depends heavily on architecture (58% for ResNet-18, 54% for ResNet-50, per Table 3), so the "54-74%" range doesn't provide strong actionable guidance beyond "just do what everyone already does."

  To address: Frame this as "empirical validation" rather than implying novel insight. The value is confirming the heuristic works for BitNet specifically.

  [MINOR] Architecture-specific hyperparameter tuning raises fairness concerns

  Section 6 reveals that MobileNetV2/EfficientNet required lr=0.01 (vs 0.1 for ResNet) and ConvNeXt required AdamW. Were FP32 baselines also trained with these adapted hyperparameters? If ResNet used standard recipes while other architectures used less-tuned configurations, the "ResNet is better for quantization" conclusion may reflect hyperparameter effort, not architectural suitability.

  To address: Clarify that FP32 baselines used the same architecture-specific adaptations.

  [MINOR] Limited theoretical grounding for "augmentation paradox"

  The explanation ("ternary networks lack capacity to exploit additional training signal") is intuitive but unsupported. The information-theoretic appendix discusses capacity limits but doesn't directly connect to augmentation. Why would more diverse training data hurt a capacity-limited model? An alternative hypothesis: augmentation increases gradient noise, which combines with STE noise to cause training instability.

  To address: Either provide evidence (gradient norm analysis, loss landscape visualization) or soften claims to "we hypothesize."

  ---
  Missing References and Comparisons

  ┌─────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                    Paper                    │                                                   Why It Should Be Cited/Compared                                                   │
  ├─────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ LSQ (Esser et al., ICLR 2020)               │ Cited but not compared. LSQ is a standard baseline for learned quantization; direct comparison would contextualize BitNet's gap.    │
  ├─────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ PACT (Choi et al., 2018)                    │ Cited but not compared. Another learned quantization baseline.                                                                      │
  ├─────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ IR-Net (Qin et al., CVPR 2020)              │ Cited in related work but no comparison. IR-Net achieves strong binary results; how does it compare to ternary?                     │
  ├─────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Real-to-Binary (Martinez et al., ICLR 2020) │ Cited but not compared. Achieves strong binary ImageNet results with sophisticated training.                                        │
  ├─────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ HAWQ-V2 (Dong et al., NeurIPS 2020)         │ Cited but the connection to mixed-precision allocation could be made stronger. HAWQ-V2 also keeps early layers at higher precision. │
  ├─────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ QKD (Kim et al., 2019)                      │ Cited and discussed, but no empirical comparison of QKD vs standard KD.                                                             │
  └─────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Questions for Authors

  1. Numerical consistency: Why does BitNet accuracy vary by ~0.5pp across tables (85.38%, 85.40%, 85.89%)? Which experiments correspond to which tables?
  2. QAT/LSQ comparison: Can you provide a single QAT or LSQ baseline on ResNet-18/CIFAR-10 using your training protocol? This would help readers understand whether the gap is BitNet-specific or inherent to ternary quantization.
  3. Augmentation instability: For the ResNet-50/CIFAR-10 case where augmentation hurts (-0.89%), do all three seeds show degradation, or is this driven by one outlier?
  4. Architecture tuning fairness: Were FP32 baselines for MobileNetV2, EfficientNet, and ConvNeXt also trained with the adapted hyperparameters (lr=0.01, AdamW), or do they use different configurations?
  5. The "exceeds FP32" claim: The CIFAR-100 result (p=0.028) would not survive Bonferroni correction across your 14+ comparisons. Would you consider softening this claim or running additional seeds?

  ---
  Scores

  ┌───────────────────────┬───────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │       Criterion       │ Score │                                                      Justification                                                       │
  ├───────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Novelty / Originality │ 5/10  │ The augmentation paradox observation is interesting; first-layer FP32 is well-known; KD for quantization is established. │
  ├───────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Significance / Impact │ 6/10  │ Practical recipe for ternary CNNs has value, but limited ImageNet validation reduces impact.                             │
  ├───────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Technical Soundness   │ 5/10  │ Numerical inconsistencies, weak statistical claims, missing baselines undermine confidence.                              │
  ├───────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Clarity / Writing     │ 8/10  │ Well-structured paper with clear narrative. Tables and figures are informative.                                          │
  ├───────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Experimental Rigor    │ 5/10  │ 3 seeds is minimal; no QAT/LSQ baselines; inconsistent accuracy values across tables.                                    │
  ├───────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Reproducibility       │ 8/10  │ Strong reproducibility section with code-to-paper pipeline and deterministic training.                                   │
  ├───────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Overall               │ 5/10  │ Interesting observations weakened by execution issues and missing baselines.                                             │
  └───────────────────────┴───────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Decision

  Weak Reject

  The paper presents interesting empirical observations (augmentation paradox, architecture-dependent recovery), but suffers from: (1) no direct comparison with standard quantization baselines (QAT, LSQ), (2) numerical inconsistencies that undermine confidence in the claims, (3) statistically weak "exceeds FP32" claims, and (4) framing the first-layer observation as more novel than warranted. The experimental scope (CIFAR
   + Tiny ImageNet, no ImageNet) is acceptable for BMVC, but the execution issues are not.

  ---
  What Would Change My Decision?

  To move from Weak Reject to Borderline/Weak Accept:

  1. Add one QAT or LSQ baseline on ResNet-18/CIFAR-10/100 using the same training protocol. This would contextualize the BitNet gap and strengthen the "practical recipe" contribution.
  2. Resolve numerical inconsistencies: Provide a consistent set of baseline numbers or explain which experimental runs correspond to which tables.
  3. Soften the "exceeds FP32" claims: Either run 5+ seeds to achieve p<0.01 on CIFAR-100, or reframe as "approaches FP32 accuracy."
  4. Report per-seed augmentation results: Show that the -0.89% result is consistent across seeds, not driven by training instability in one run.

  If all four issues were addressed in a revision, I would move to Borderline or Weak Accept, as the augmentation paradox finding and architecture-dependent analysis provide practical value to the BMVC community.
