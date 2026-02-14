# TMLR Review Prompt

You are a senior reviewer and action editor for TMLR (Transactions on Machine Learning Research). You have 15+ years of experience in machine learning, with deep expertise in model compression, quantization, knowledge distillation, and efficient inference. You have reviewed 100+ papers for top venues and served as Area Chair at NeurIPS and ICML.

## About TMLR

- Acceptance rate: ~30-40% (rolling submissions, no deadline pressure)
- Review style: OpenReview, single-blind (author names visible), action editors
- TMLR explicitly values **correctness, clarity, and completeness over novelty**. Well-executed empirical studies that provide useful knowledge to the community are valued. Negative results and careful ablations are welcome. Reproducibility is a core value.
- TMLR explicitly states: "We will not reject papers solely based on lack of novelty, as long as the claims are well-supported and the paper provides useful information."
- What gets rejected: Papers with incorrect claims or flawed methodology. Poorly written papers. Papers that add nothing to existing understanding (pure replication without insight).
- Typical reviewer profile: Researcher who reads carefully and checks methodology. Less focused on SOTA numbers, more focused on "is this correct and useful?"

## Your Task

Review the paper pasted below. You have unlimited time. Be thorough, precise, and honest. You are not here to be encouraging -- you are here to determine whether this paper meets the acceptance bar for TMLR.

## Your Review Process

Before writing your review, you must:

1. **Verify every numerical claim** in the paper against the tables and figures. Flag any inconsistency (e.g., abstract says X% but table shows Y%).
2. **Identify the closest 5-10 related papers** in the quantization/efficient ML literature that you know of. For each, state whether this paper adequately cites and positions itself against it. Flag missing comparisons that a TMLR reviewer would expect.
3. **Assess the experimental methodology**: Are baselines fair? Are hyperparameters tuned with equal effort for all methods? Is the evaluation protocol standard? Are there enough seeds/runs for statistical significance?
4. **Evaluate what is genuinely new** vs. well-known in the community. Be specific -- "keeping first/last layers in FP32" has been standard since XNOR-Net (2016). What does this paper add beyond confirming existing practice?
5. **Check for overclaiming**: Does the abstract/conclusion promise more than the results deliver? Are limitations honestly discussed?

## Specific Questions for This Paper at TMLR

- Are all claims precisely supported by the data? Check every number in the abstract against the tables.
- Is the experimental methodology sound? Enough seeds (3)? Fair baselines? Proper ablations?
- Does the paper contribute useful knowledge even if individual components (FP32 first layer, KD) are known? Is the systematic combination and quantification across architectures useful?
- The honest framing of architecture-dependent effectiveness (17-130% recovery range): is this appropriately nuanced, or does it undermine the paper's message?
- Reproducibility: the paper claims a full code-to-paper pipeline. How credible is this claim based on the description?
- Does the paper adequately survey related work, especially recent ternary/binary quantization methods (LSQ, EWGS, PACT, APoT)?
- The information-theoretic appendix: does it add rigor, or is it hand-wavy post-hoc rationalization?

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
| Technical Soundness | /10 | |
| Clarity / Writing | /10 | |
| Experimental Rigor | /10 | |
| Reproducibility | /10 | |
| Significance / Usefulness | /10 | |
| Overall | /10 | |

Note: TMLR does not weight novelty as heavily as NeurIPS/ICML. Focus on correctness, completeness, and usefulness.

### Decision
One of: **Accept**, **Accept with minor revisions**, **Revise and resubmit**, **Reject**

### What Would Change Your Decision?
List the minimum set of changes that would move your recommendation up by one tier. Be specific and actionable.

---

## Paper to Review

 TMLR Review: "When Augmentation Fails: Knowledge Distillation for Ternary CNNs"

  Summary

  This paper systematically studies BitNet b1.58 (ternary, 1.58-bit quantization) applied to standard CNN architectures on CIFAR and Tiny ImageNet. The main claims are: (1) data augmentation disproportionately benefits FP32 models over ternary models, widening rather than closing the accuracy gap; (2) keeping the first convolutional layer in FP32 recovers 54-74% of the accuracy gap with only 0.08% parameter overhead; (3)
  combining this fix with knowledge distillation recovers 88% of the gap on CIFAR-10 and exceeds FP32 accuracy on harder tasks. The paper is positioned as a systematic empirical study rather than a novel method contribution.

  ---
  Strengths

  - Well-executed systematic study: The paper tests 4 augmentation strategies × 2 models × 2 datasets × 3 seeds for the augmentation experiments, providing comprehensive coverage (Section 4, Tables 1-2).
  - Honest quantification of established practice: While keeping first/last layers in FP32 is known since XNOR-Net (2016), this paper provides the first precise quantification for BitNet b1.58: conv1 accounts for 54-74% of accuracy loss with 0.08% of parameters (Section 5.1, Table 3). This is useful empirical data.
  - Cross-architecture validation: Testing across ResNet, EfficientNet, MobileNetV2, and ConvNeXt with honest reporting of architecture-dependent effectiveness (17-130% recovery range in Table 8) demonstrates scientific rigor. The paper doesn't claim universal applicability.
  - Practical recipe with strong results on harder tasks: The finding that the recipe exceeds FP32 on CIFAR-100 (+1.0%) and Tiny ImageNet (+1.3%) is notable (Tables 6-7), with statistical testing reported for CIFAR-100 (p=0.028).
  - Reproducibility commitment: The described code-to-paper pipeline (Appendix A) with determinism guarantees, specific seeds, and commands is a positive example for the field.
  - Appropriate related work coverage: The paper cites and positions against the main ternary/binary quantization methods (TWN, TTQ, XNOR-Net) and recent quantization-aware training techniques (LSQ, PACT, EWGS). Table 9 provides a fair comparison acknowledging TTQ achieves smaller gaps via learned scaling factors.
  - Limitations honestly discussed: Section 7 explicitly acknowledges the 26% ImageNet gap, limited statistical power with 3 seeds, and single-seed Tiny ImageNet result.

  ---
  Weaknesses

  [MAJOR] Numerical inconsistencies in abstract:
  - Abstract claims "4--11×" benefit ratio, but Table 2 shows 3.6×, 3.9×, 10.9× (actual range: 3.6-10.9×)
  - Abstract claims "80--130% for ResNet" but Table 8 shows ResNet recovery is 80-122%; the 130% is ConvNeXt on Tiny ImageNet
  - These are minor but undermine trust in precision. Fix: Correct abstract to match tables exactly.

  [MAJOR] Incomplete layer ablation data: Table 3 shows ablation only for ResNet-18/CIFAR-10. The claim "54-74% across all configurations" requires ablation data for all 8 model-dataset configurations (2 models × 4 datasets minimum). Currently, readers cannot verify this claim. Fix: Include complete ablation table or explicitly state which configurations were tested.

  [MAJOR] Single-seed Tiny ImageNet result for recipe: The headline-grabbing "+1.3% over FP32" on Tiny ImageNet is from a single seed (acknowledged in Section 5.4). This significantly weakens the claim. Fix: Run 3 seeds for Tiny ImageNet recipe experiments; report confidence intervals.

  [MINOR] Title may be misleading: "When Augmentation Fails" is sensational—augmentation doesn't fail; it helps ternary models less than FP32 models. In Table 2, BitNet still gains 0.31-0.81% from augmentation in 3/4 cases. Fix: Consider "When Augmentation Helps Less" or similar.

  [MINOR] No actual inference latency measurements: The paper claims "20× compression and 64× theoretical speedup" but reports no wall-clock inference times. This limits practical relevance. Fix: Add inference latency on at least one edge device, or explicitly state this is theoretical only.

  [MINOR] Information-theoretic appendix is post-hoc: Appendix B provides plausible explanations but no testable predictions. The DPI argument for why conv1 matters is intuitive but doesn't distinguish between alternative explanations. This is acceptable but shouldn't be overclaimed.

  [MINOR] Limited KD hyperparameter exploration: The paper uses fixed T=4, α=0.9 throughout main results, then mentions in Discussion that lower α works better in isolation but combining optimized hyperparameters yields no improvement. This interaction is interesting but underexplored. Fix: Include the KD ablation table in the main paper or appendix.

  ---
  Missing References and Comparisons

  1. AdaRound (Nagel et al., 2020): Post-training quantization method that could contextualize why training-time quantization is needed for ternary.
  2. BRECQ (Li et al., 2021): Block-wise reconstruction for quantization—relevant for understanding layer-wise sensitivity.
  3. QAT vs PTQ comparison: The paper only considers quantization-aware training. Brief discussion of why PTQ approaches fail for ternary would strengthen positioning.
  4. Recent binary network advances: While ReActNet and BNext are mentioned in Future Work, direct comparison of training recipes (multi-stage training, learned per-channel scaling) would help explain the 26% ImageNet gap.
  5. Concurrent work BD-Net (Kim et al., 2025): Cited but not compared experimentally. Since BD-Net also applies 1.58-bit to MobileNet's depthwise convolutions, results comparison would be valuable.

  ---
  Questions for Authors

  1. Layer ablation completeness: Can you provide the conv1 recovery percentages for all tested configurations (ResNet-18/50 × CIFAR-10/100 × augmentation levels)? The 54-74% range is cited repeatedly but only one data point is shown.
  2. Teacher model quality: Your FP32 ResNet-18 achieves 88.88% on CIFAR-10 (Table 5). Modern training recipes achieve 95%+. Did you try distillation from a stronger teacher? Would this change the "exceeds FP32" narrative?
  3. Augmentation paradox mechanism: You hypothesize that ternary networks "lack capacity to exploit additional training signal." Did you measure effective capacity (e.g., via gradient analysis or layer-wise rank)? The hypothesis is plausible but untested.
  4. ImageNet gap: The 26% gap is dramatically larger than TTQ's 2.7%. Beyond training recipe differences, is there something fundamental about BitNet b1.58's fixed {-1,0,+1} vs. TTQ's learned scales that explains this?
  5. Statistical significance for exceeding FP32: For CIFAR-100, you report p=0.028 (one-sided). What is the two-sided p-value? With 3 seeds and multiple comparisons across configurations, how should readers interpret this significance level?

  ---
  Scores

  ┌───────────────────────────┬───────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │         Criterion         │ Score │                                                   Justification                                                    │
  ├───────────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Technical Soundness       │ 7/10  │ Methodology is sound but incomplete layer ablation data and single-seed Tiny ImageNet weaken key claims            │
  ├───────────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Clarity / Writing         │ 8/10  │ Well-organized, clear narrative, good figures; minor abstract inaccuracies                                         │
  ├───────────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Experimental Rigor        │ 7/10  │ Good coverage of architectures/datasets, but 3 seeds is minimum; some key results undersampled                     │
  ├───────────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Reproducibility           │ 9/10  │ Excellent description of pipeline, determinism guarantees, specific commands; code promised                        │
  ├───────────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Significance / Usefulness │ 7/10  │ Useful systematic study validating and quantifying known practices; architecture-dependent findings are actionable │
  ├───────────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Overall                   │ 7/10  │ Solid empirical contribution to TMLR's scope; not novel but provides useful quantification                         │
  └───────────────────────────┴───────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Decision

  Revise and Resubmit

  The paper provides a useful systematic study that fits TMLR's values of correctness and completeness over novelty. However, several issues need addressing before acceptance:

  1. The incomplete layer ablation data undermines the core "54-74%" claim
  2. Numerical inconsistencies in the abstract erode trust
  3. Single-seed Tiny ImageNet result for the headline "+1.3%" claim is insufficient

  These are fixable with additional experiments and careful editing.

  ---
  What Would Change Your Decision?

  To move to "Accept with minor revisions":

  1. Provide complete layer ablation table showing conv1 recovery for at least ResNet-18/50 × CIFAR-10/100 (the claimed 54-74% range)
  2. Correct all numerical claims in abstract to exactly match tables
  3. Add 2 more seeds for Tiny ImageNet recipe result, or explicitly downgrade this claim throughout the paper
  4. Either add inference latency measurements or remove "practical deployment" language from conclusions

  To move to "Accept":
  - All of the above, plus response to Questions 1-4 that addresses concerns about teacher model quality and augmentation mechanism

