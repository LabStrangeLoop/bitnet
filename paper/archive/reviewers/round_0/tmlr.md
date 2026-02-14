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

[PASTE FULL PAPER TEXT HERE]

---

## Reviewer 1

 TMLR Review: "When Augmentation Fails: Knowledge Distillation for Ternary CNNs"

  Summary (2-3 sentences)

  This paper investigates BitNet b1.58 ternary quantization for CNNs, claiming that data augmentation fails to close the accuracy gap between ternary and FP32 models. The authors propose a recipe combining FP32 first-layer retention with knowledge distillation, reporting 88% gap recovery on CIFAR-10 and accuracy exceeding FP32 on harder tasks. The paper tests across four CNN families (ResNet, EfficientNet, MobileNetV2,
  ConvNeXt) and finds architecture-dependent effectiveness.

  ---
  Strengths (bulleted, specific)

  - Well-executed layer-wise ablation (Table 2, Section 5.1): The quantification that conv1 recovers 54-58% of the accuracy gap while comprising only 0.08% of parameters is useful practical knowledge. While keeping first/last layers in FP32 is known practice since XNOR-Net, the precise quantification for BitNet b1.58 is a contribution.
  - Honest reporting of architecture-dependent results (Table 8, Section 6): The paper transparently reports that recovery ranges from 17% (MobileNetV2/CIFAR-100) to 130% (ConvNeXt/Tiny-IN), avoiding cherry-picking. This nuanced finding is useful for practitioners.
  - Reproducibility infrastructure (Appendix A): The described code-to-paper pipeline with deterministic seeds, JSON result storage, and automated figure/table generation represents good research practice. The 300 GPU-hour budget is reasonable.
  - Cross-architecture evaluation (Section 6): Testing beyond ResNet to include EfficientNet, MobileNetV2, and ConvNeXt on three datasets (14 configurations total) provides breadth beyond typical quantization papers.
  - Clear limitation discussion (Section 7): The paper honestly acknowledges failure on ImageNet (26% gap vs. TTQ's 2.7%), lack of real latency measurements, and architecture-specific tuning requirements.

  ---
  Weaknesses (bulleted, specific, ordered by severity)

  [FATAL] Central claim contradicted by own data:
  The abstract and Section 4 claim the accuracy gap "remains approximately 3.5% regardless of augmentation strategy." However, the generated augmentation_ablation.tex table shows:
  - ResNet-18/CIFAR-10: gap ranges from 2.99% (basic) to 3.90% (full)
  - ResNet-18/CIFAR-100: gap ranges from 3.72% (basic) to 6.88% (full) — nearly doubling
  - ResNet-50/CIFAR-100: gap ranges from 9.11% (basic) to 11.25% (full)

  The gap increases with stronger augmentation, the opposite of "constant." This undermines the entire "augmentation paradox" narrative that is the paper's first claimed contribution.

  To address: Re-analyze the data, revise claims to match reality, or explain the discrepancy between the main text tables and generated tables.

  [MAJOR] Inconsistencies between inline tables and generated data files:
  Table 1 in the main text claims FP32=88.89%, BitNet=85.40% for ResNet-18/CIFAR-10/basic. However:
  - The raw results file shows best_acc=88.92% (FP32) and 85.10% (BitNet) for seed 42
  - The generated accuracy_full.tex shows different numbers again (90.10% FP32, 86.20% BitNet for full augmentation)

  This suggests tables were manually constructed rather than auto-generated from experiments, contradicting the reproducibility claim.

  To address: Ensure all tables are programmatically generated from raw experiment outputs. Provide the exact script that produces each table.

  [MAJOR] Missing comparison with comparable methods:
  The paper does not compare BitNet b1.58 + recipe against:
  - TTQ (Zhu et al., 2017): Claims near-FP32 accuracy with learned scaling factors
  - PACT/LSQ: Learned scale quantization methods achieving strong results
  - DoReFa-Net: Directly comparable ternary method

  The paper acknowledges TTQ in discussion but provides no quantitative comparison. A TMLR reviewer expects direct accuracy comparisons with closest prior work on the same benchmarks.

  To address: Add a comparison table showing BitNet b1.58 + recipe vs. TTQ, TWN, and at least one learned-scale method on CIFAR-10/100.

  [MAJOR] Claims about "exceeding FP32" need scrutiny:
  Table 6 claims 63.40% vs 62.40% FP32 (+1.0%), but:
  - Only 3 seeds with no significance test reported for this comparison
  - Standard deviations (0.09 for recipe, not reported for FP32) may overlap
  - The FP32 baseline (62.40%) is weak — published ResNet-18/CIFAR-100 results typically achieve 77%+ with proper training

  To address: Provide p-values for the "exceeds FP32" claims. Acknowledge that the FP32 baselines may be undertrained.

  [MINOR] Abstract numbers don't precisely match tables:
  - Abstract claims "54-74% of accuracy loss" for conv1; Table 2 shows 58% (ResNet-18) and 54% (ResNet-50) — where does 74% come from?
  - Abstract claims "17-130%" recovery range, but Table 8 shows 17-130% is correct only if including ConvNeXt/Tiny-IN (130%) which is an outlier

  To address: Verify all abstract numbers against tables; cite specific configurations when claiming ranges.

  [MINOR] KD hyperparameter inconsistency:
  Section 5.2 uses α=0.9, T=4 and claims this "works out-of-the-box." Section 7 states "ternary networks prefer α=0.5-0.7... contradicts conventional KD wisdom." The conclusion then recommends defaults (T=4, α=0.9). This is confusing — which hyperparameters were actually used for the main results?

  To address: Clarify which hyperparameters produced each table's results.

  [MINOR] Information-theoretic appendix is hand-wavy:
  Appendix B invokes DPI and information bottleneck but provides no quantitative predictions. The claim that gaps scale with log₂(C) is presented as "consistent with" theory but no statistical fit is shown. This is post-hoc rationalization rather than rigorous theory.

  To address: Either remove theoretical claims or provide quantitative predictions with confidence intervals.

  ---
  Missing References and Comparisons

  ┌─────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────┐
  │            Paper            │                            Why it should be cited/compared                             │
  ├─────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ LSQ (Esser et al., 2020)    │ Learned step-size quantization is state-of-art for low-bit training; direct competitor │
  ├─────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ EWGS (Lee et al., 2021)     │ Element-wise gradient scaling for quantization; addresses STE gradient issues          │
  ├─────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ PACT (Choi et al., 2018)    │ Parameterized clipping activation; foundational for activation quantization            │
  ├─────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ APoT (Li et al., 2020)      │ Additive powers-of-two quantization; alternative to uniform ternary                    │
  ├─────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ ReActNet (Liu et al., 2020) │ Cited but not compared; achieves 69.4% on ImageNet with binary weights                 │
  ├─────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ IR-Net (Qin et al., 2020)   │ Information retention for binary networks; directly relevant to conv1 findings         │
  └─────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Questions for Authors

  1. Data discrepancy: Why do Tables 1-4 in the main text show different numbers than the auto-generated tables in paper/tables/? Which represents the actual experimental results?
  2. Augmentation paradox: The augmentation_ablation.tex table shows the gap increases from 3.72% to 6.88% on CIFAR-100 with stronger augmentation. How do you reconcile this with the "constant ~3.5% gap" claim in Section 4?
  3. FP32 baselines: Your FP32 ResNet-18/CIFAR-100 achieves 62.40%. Published results typically achieve 77%+. What explains this gap? Were you using the same training recipe for FP32 and BitNet?
  4. Statistical significance: For the "exceeds FP32" claims (Table 6, 7), can you provide paired t-test p-values comparing recipe vs. FP32 across seeds?
  5. ImageNet failure: You report a 26% gap on ImageNet vs. TTQ's 2.7%. Did you try the training adaptations used by TTQ (learned scales, longer training, different optimizer)? What specifically causes the gap?

  ---
  Scores

  ┌───────────────────────────┬───────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │         Criterion         │ Score │                                                           Justification                                                            │
  ├───────────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Technical Soundness       │ 4/10  │ Central "augmentation paradox" claim contradicted by own data; potential data integrity issues between inline and generated tables │
  ├───────────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Clarity / Writing         │ 7/10  │ Well-structured paper with clear narrative, but confusing inconsistencies in numbers                                               │
  ├───────────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Experimental Rigor        │ 5/10  │ Good seed count (3) and architecture breadth, but missing baselines against prior work and statistical tests for key claims        │
  ├───────────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Reproducibility           │ 6/10  │ Strong infrastructure described, but discrepancies between claimed auto-generation and actual tables undermine confidence          │
  ├───────────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Significance / Usefulness │ 5/10  │ Layer sensitivity quantification is useful; but weak FP32 baselines and missing comparisons limit practical value                  │
  ├───────────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Overall                   │ 4/10  │ Cannot recommend acceptance with fatal data inconsistency; could be a solid paper after major revision                             │
  └───────────────────────────┴───────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Decision

  Reject

  The paper's central contribution — the "augmentation paradox" claiming a constant ~3.5% gap regardless of augmentation — is contradicted by the paper's own generated data tables showing the gap nearly doubles with stronger augmentation on CIFAR-100. Combined with discrepancies between inline tables and generated files, and missing comparisons with prior ternary methods, the paper does not meet TMLR's bar for
  correctness.

  ---
  What Would Change Your Decision?

  To move from Reject to Revise and resubmit:

  1. Resolve the data inconsistency: Explain why augmentation_ablation.tex shows increasing gaps (3.72%→6.88%) while the text claims constant gaps. Either correct the analysis or revise the "augmentation paradox" narrative entirely.
  2. Ensure end-to-end reproducibility: Demonstrate that running make paper from the repository produces exactly the tables shown in the submitted PDF. Fix any discrepancies.
  3. Add baseline comparisons: Include a table comparing BitNet b1.58 + recipe vs. TTQ and at least one learned-scale method (LSQ or PACT) on CIFAR-10/100 with identical training budgets.
  4. Provide statistical tests: Report p-values for all "exceeds FP32" claims.

  To move from Revise and resubmit to Accept with minor revisions:

  5. Strengthen FP32 baselines: Either use standard training recipes achieving published accuracies (~77% ResNet-18/CIFAR-100), or clearly explain why your baselines differ.
  6. Clarify KD hyperparameters: State which exact settings produced each result table.
