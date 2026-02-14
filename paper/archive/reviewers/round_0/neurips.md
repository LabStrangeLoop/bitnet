# NeurIPS Review Prompt

You are a senior reviewer for NeurIPS (Neural Information Processing Systems). You have 15+ years of experience in machine learning, with deep expertise in model compression, quantization, knowledge distillation, and efficient inference. You have reviewed 50+ papers per year for NeurIPS, ICML, and ICLR, and have served as Area Chair. You are reviewing for the main conference track, not a workshop.

## About NeurIPS

- Acceptance rate: ~25% (approximately 3,000 accepted out of ~12,000 submissions)
- Review style: Double-blind, area chairs, author rebuttals, emergency reviewers
- NeurIPS values novelty and broad significance above all. Strong preference for papers that introduce new concepts, methods, or understanding that the community will build upon. Empirical papers need to be exceptionally thorough or reveal genuinely surprising findings.
- What gets rejected: Straightforward applications of known techniques to known problems. Papers where the main contribution is "we tried X on Y and here's how it went." Limited experimental scope. Incremental improvements without deeper insight. Papers better suited for a workshop.
- Typical reviewer profile: Faculty or senior PhD with broad ML knowledge. High bar for novelty. Compares every paper against the ~12,000 competing submissions. Asks: "will people cite this in 3 years?"

## Your Task

Review the paper pasted below. You have unlimited time. Be thorough, precise, and honest. You are not here to be encouraging -- you are here to determine whether this paper meets the acceptance bar for NeurIPS main conference.

## Your Review Process

Before writing your review, you must:

1. **Verify every numerical claim** in the paper against the tables and figures. Flag any inconsistency.
2. **Identify the closest 5-10 related papers** in the quantization/efficient ML literature that you know of. For each, state whether this paper adequately cites and positions itself against it. Flag missing comparisons.
3. **Assess the experimental methodology**: Are baselines fair? Are hyperparameters tuned with equal effort for all methods? Is the evaluation protocol standard? Are there enough seeds/runs for statistical significance?
4. **Evaluate the novelty claim critically**: What is genuinely new here vs. well-known in the community? Be very specific. "Keeping first/last layers in FP32" has been standard since XNOR-Net (2016). KD for quantized models was studied in QKD (2019). The "augmentation doesn't help quantized models" observation exists implicitly in the binary network literature. What is the delta over existing knowledge?
5. **Check for overclaiming**: Does the abstract/conclusion promise more than the results deliver?

## Specific Questions for This Paper at NeurIPS

- The "augmentation paradox" -- is this truly surprising? Prior work on binary networks (XNOR-Net, ReActNet, IR-Net) already showed augmentation has limited effect on quantized networks. What is the delta over existing understanding? Is formalizing this as a "paradox" justified, or is it marketing?
- The recipe components (FP32 first layer, KD) are individually well-known. Is combining them and quantifying their interaction on BitNet b1.58 a NeurIPS-level contribution? What new understanding does this provide beyond "these known techniques work here too"?
- No theoretical contribution beyond informal information-theoretic arguments in an appendix. NeurIPS papers in this space typically provide either formal analysis or very large-scale empirical results. This paper has neither.
- Single-author paper: does the scope of experiments (CIFAR + Tiny ImageNet, no ImageNet) match what NeurIPS expects for an empirical contribution?
- Would this be better suited for the Efficient ML Workshop at NeurIPS rather than the main conference?
- The paper claims "first precise quantification" of conv1 sensitivity for BitNet. But HAQ (CVPR 2019) and HAWQ (ICCV 2019) already provide per-layer sensitivity analysis for quantized networks. What is genuinely first here?
- How does this compare to QAT methods (LSQ, EWGS, PACT) that achieve much smaller accuracy gaps on ImageNet without needing FP32 layers?

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

  NeurIPS Review: "When Augmentation Fails: Knowledge Distillation for Ternary CNNs"

  Summary (2-3 sentences)

  This paper investigates BitNet b1.58 (ternary weights: {-1, 0, +1}) applied to standard CNN architectures (ResNet, EfficientNet, MobileNetV2, ConvNeXt). The main claims are: (1) data augmentation fails to close the ~3.5% accuracy gap between ternary and FP32 models ("augmentation paradox"), (2) keeping the first convolutional layer in FP32 recovers 54-74% of the accuracy gap, and (3) combining this with knowledge
  distillation achieves 88% gap recovery on CIFAR-10 and exceeds FP32 accuracy on harder tasks.

  Strengths (bulleted, specific)

  - Systematic experimental methodology: The paper tests 4 augmentation strategies × 2 architectures × 2 datasets with 3 seeds each, providing reasonable statistical coverage (Table 1, Section 4.2). The ~3.5% constant gap across augmentation strategies is a clear empirical finding.
  - Layer-wise ablation is well-executed: Table 3 and Table 6 provide clean ablations showing conv1's disproportionate contribution (58% recovery with 0.08% parameters). This is a well-executed confirmation of established practice, not a novel discovery.
  - Cross-architecture evaluation: Testing across ResNet, EfficientNet, MobileNetV2, and ConvNeXt (Table 8) reveals architecture-dependent effectiveness, with standard convolutions (80-122% recovery) outperforming depthwise separable designs (17-73%). This is useful practical guidance.
  - Honest limitations section: Section 7 acknowledges ImageNet failures (26% gap), lack of latency measurements, and need for architecture-specific tuning. This candor is appreciated.
  - Reproducibility commitment: Appendix A describes a complete code-to-paper pipeline with deterministic seeds. This is commendable practice.

  Weaknesses (bulleted, specific, ordered by severity)

  [FATAL]: Limited novelty—individual components are well-established

  - Keeping first/last layers in FP32 has been standard practice since XNOR-Net (2016), DoReFa-Net (2016), TWN (2016), and TTQ (2016). The paper's own research notes acknowledge: "The practice of keeping conv1 at full precision is not novel—it has been standard practice since 2016." The claim of "first precise quantification" is misleading given HAWQ (ICCV 2019) and HAWQ-V2 (NeurIPS 2020) provide Hessian-based per-layer
  sensitivity measurements.
  - Knowledge distillation for quantized models was extensively studied in QKD (2019). The KD setup (T=4, α=0.9) is completely standard.
  - What would fix this: Demonstrate that the specific quantification (54-74%) leads to actionable insights beyond "keep conv1 in FP32"—or show this paper's measurements reveal something HAQ/HAWQ did not. Currently, the paper validates a 10-year-old convention with numbers.

  [FATAL]: Insufficient experimental scope for NeurIPS main conference

  - No ImageNet results despite ImageNet being the standard benchmark for quantization papers (XNOR-Net, TTQ, ReActNet, HAWQ all report ImageNet). The paper admits a 26% gap on ImageNet (Section 7), which invalidates the claimed recipe.
  - CIFAR-10/100 and Tiny ImageNet are insufficient for claims about "practical deployment" of ternary CNNs. Every cited quantization paper (HAQ, HAWQ, ReActNet, TTQ) includes ImageNet evaluation.
  - What would fix this: Include ImageNet results showing the recipe achieves competitive accuracy (within 2-3% of FP32), or significantly reduce claims about practicality.

  [MAJOR]: The "augmentation paradox" is not novel or surprising

  - The observation that augmentation provides similar relative benefits to both quantized and full-precision models is implicit in the binary network literature. ReActNet, IR-Net, and Real-to-Binary all use standard augmentation without claiming differential effects. The framing as a "paradox" appears to be marketing.
  - The paper claims this "rules out regularization as a solution" but quantization itself acts as regularization (acknowledged in Section 7), which explains why external regularization shows diminishing returns. This is expected, not paradoxical.
  - What would fix this: Provide formal analysis or cite specific prior claims that augmentation should differentially help quantized networks. Without a clear prior expectation, there is no paradox.

  [MAJOR]: Missing direct comparisons with relevant methods

  - No comparison with QAT methods (LSQ, EWGS, PACT) that achieve smaller accuracy gaps without FP32 layers on ImageNet.
  - No comparison with TTQ on the same datasets—TTQ claims near-zero gap on CIFAR with learned scaling factors.
  - No comparison with BD-Net (AAAI 2026), which applies 1.58-bit quantization to CNNs and is cited in the paper.
  - What would fix this: Include Table comparing BitNet recipe vs. TTQ, LSQ, and BD-Net on CIFAR-10/100 under matched training budgets.

  [MAJOR]: Claims of exceeding FP32 are misleading

  - The paper claims the recipe "exceeds FP32" on CIFAR-100 and Tiny ImageNet (Table 5, Table 6). However, the FP32 baselines are weak: 62.40% on CIFAR-100 (SOTA is >82%), 54.85% on Tiny ImageNet (SOTA is >80%). A quantized model beating an undertrained FP32 baseline is not surprising—it suggests the FP32 model left optimization gains on the table.
  - What would fix this: Report results with properly-tuned FP32 baselines matching published SOTA, or acknowledge that the comparison is against matched-compute baselines rather than SOTA.

  [MINOR]: Inconsistent numerical claims

  - Abstract claims "3-4% accuracy gap" but Table 8 shows MobileNetV2 has 8.2% gap on CIFAR-10 and 14% on CIFAR-100. The 3-4% characterization applies only to ResNet.
  - Table 1 shows ResNet-18/CIFAR-10 FP32 at 88.89%, but Table 4 (accuracy_full.tex) shows 90.10% with full augmentation. The paper should clarify which baseline applies where.

  [MINOR]: No actual deployment validation

  - Claims of "20× compression" and "64× theoretical speedup" without measured inference latency on any hardware platform. Theoretical compression without runtime validation is incomplete for a paper about "practical deployment."

  Missing References and Comparisons

  ┌────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────┐
  │         Paper          │                                         Why it matters                                         │
  ├────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ LSQ (ICLR 2020)        │ Learned step size quantization achieves small gaps without FP32 layers                         │
  ├────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ EWGS (CVPR 2020)       │ Differentiable quantization with smaller ImageNet gap than reported here                       │
  ├────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ PACT (arXiv 2018)      │ Parameterized clipping for activations—standard QAT baseline                                   │
  ├────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ HAQ (CVPR 2019)        │ Provides per-layer sensitivity analysis via RL—the paper cites but doesn't compare             │
  ├────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ HAWQ-V2 (NeurIPS 2020) │ Hessian-based layer sensitivity measurements—directly relevant to "first quantification" claim │
  ├────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ IR-Net (CVPR 2020)     │ Binary networks with information retention—relevant to augmentation discussion                 │
  ├────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ BNext (arXiv 2022)     │ Binary networks achieving 80.6% ImageNet—shows what's achievable                               │
  ├────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ BD-Net (AAAI 2026)     │ 1.58-bit quantization for MobileNet—cited but not compared                                     │
  └────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────┘

  Questions for Authors

  1. On the "augmentation paradox": What specific prior work claimed that augmentation should differentially benefit quantized networks? Without a clear hypothesis to reject, how is this observation a "paradox" rather than an expected null result?
  2. On layer sensitivity: HAWQ/HAWQ-V2 measure per-layer sensitivity via Hessian eigenvalues and find architecture-dependent patterns (not always first-layer dominant). How does your "precise quantification" differ from or extend HAWQ's analysis?
  3. On ImageNet scaling: You report a 26% gap on ImageNet (Section 7). What changes to the recipe would be needed to achieve competitive ImageNet results? Is the recipe fundamentally limited or just undertrained?
  4. On FP32 baselines: Your CIFAR-100 FP32 baseline (62.40%) is well below published results with ResNet-18 (~77% with SOTA training). Would the recipe still "exceed FP32" against properly-tuned baselines?
  5. On comparison with TTQ: TTQ achieves near-zero accuracy gap on CIFAR with learned scaling factors. Why should practitioners use your fixed {-1,0,+1} approach with larger gaps instead of TTQ?

  Scores

  ┌───────────────────────┬───────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │       Criterion       │ Score │                                                    Justification                                                     │
  ├───────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Novelty / Originality │ 3/10  │ Keeping conv1 in FP32 is 10-year-old practice; KD for quantization is standard; "paradox" is not actually surprising │
  ├───────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Significance / Impact │ 4/10  │ Useful empirical study but limited to small datasets; doesn't advance SOTA understanding                             │
  ├───────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Technical Soundness   │ 6/10  │ Methodology is sound but conclusions overreach; missing key comparisons undermines claims                            │
  ├───────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Clarity / Writing     │ 8/10  │ Well-written with clear structure; figures are informative; narrative is easy to follow                              │
  ├───────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Experimental Rigor    │ 4/10  │ Multiple seeds and architectures, but no ImageNet, no comparisons with TTQ/LSQ/HAWQ, weak baselines                  │
  ├───────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Reproducibility       │ 8/10  │ Code-to-paper pipeline described; deterministic seeds; commendable commitment to reproducibility                     │
  ├───────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Overall               │ 4/10  │ Empirical study validating known practices without novel insights or scale                                           │
  └───────────────────────┴───────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Decision

  Weak Reject

  This paper presents a competent empirical study that validates established practices (FP32 first layer, KD for quantization) on CIFAR-scale datasets with ternary quantization. However, it does not meet the NeurIPS acceptance bar for several reasons:

  1. Insufficient novelty: The main contributions—keeping conv1 in FP32 and using KD—have been standard practice since 2016 and 2019 respectively. The "precise quantification" claim is undercut by HAWQ's existing per-layer sensitivity analysis.
  2. Limited scope: CIFAR and Tiny ImageNet are inadequate for claims about practical deployment. The admitted 26% ImageNet gap suggests the recipe doesn't scale.
  3. Missing context: No comparisons with TTQ (which achieves near-zero gap), LSQ, EWGS, or other QAT methods that the community considers baselines.
  4. Overclaiming: The "augmentation paradox" framing overstates what is a straightforward null result. Claims of exceeding FP32 rely on weak baselines.

  This paper would be a strong fit for the NeurIPS Efficient ML Workshop or a venue like BMVC/WACV, where the scope and novelty expectations are calibrated for empirical studies of this scale.

  What Would Change Your Decision?

  To move from Weak Reject to Borderline/Weak Accept:

  1. Add ImageNet results showing the recipe achieves <5% accuracy gap (currently 26% gap makes the recipe impractical for standard benchmarks)
  2. Include direct comparison table with TTQ, LSQ, and BD-Net on CIFAR-10/100 under matched training conditions
  3. Remove or significantly soften "paradox" framing—rename to "empirical observation" or provide formal prior expectation that was violated
  4. Strengthen FP32 baselines to match published results, or explicitly acknowledge these are matched-compute (not SOTA) comparisons
  5. Provide differentiation from HAWQ: Either show your analysis reveals something HAWQ missed, or acknowledge this is empirical validation of known sensitivity patterns for ternary (vs. general mixed-precision)


