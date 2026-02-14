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

⏺ NeurIPS Review: "When Augmentation Fails: Knowledge Distillation for Ternary CNNs"

  ---
  Summary

  This paper investigates ternary (1.58-bit) quantization of CNNs using the BitNet b1.58 formulation, documenting that data augmentation widens rather than closes the accuracy gap between ternary and FP32 models. The main contribution is a "recipe" combining an FP32 first convolutional layer with knowledge distillation, which recovers 88% of the accuracy gap on CIFAR-10 and reportedly exceeds FP32 accuracy on harder
  tasks. The paper extends experiments to four CNN families and provides reproducibility artifacts.

  ---
  Strengths

  - Systematic experimental coverage: The paper tests 4 augmentation strategies × 2 architectures × 2 datasets × 3 seeds for the augmentation study (Section 4, Tables 1-2), which is methodical even if the scope is limited.
  - Clear quantification of conv1 importance: Table 3 provides precise numbers (58% recovery from 0.08% parameters) that, while not a novel finding, are cleanly presented. The ablation methodology is sound.
  - Honest limitations section: Section 7 acknowledges the 26% ImageNet gap, lack of latency measurements, limited seeds for some claims, and the need for architecture-specific tuning. This transparency is appreciated.
  - Reproducibility effort: Appendix A provides detailed instructions for regenerating results. The determinism guarantees (line 907-915) and repository structure are well-documented.
  - Cross-architecture analysis (Section 6): Testing on EfficientNet, MobileNetV2, and ConvNeXt extends beyond typical ResNet-only studies. The finding that depthwise separable architectures show poor recovery (17-73%) is useful practical guidance.
  - Well-written and organized: The paper is clear, flows logically from problem → failed approach → working solution → extensions.

  ---
  Weaknesses

  [FATAL]

  1. Insufficient novelty for NeurIPS main conference: The core contributions are:
    - Observation that augmentation helps FP32 more than quantized models — this is implicit in nearly all binary/ternary network papers since 2016, which use simpler augmentation than full-precision counterparts
    - FP32 first layer — standard practice since XNOR-Net (2016), explicitly used in TWN, TTQ, DoReFa-Net
    - KD for quantized models — QKD (2019), SQAKD (2024), and numerous others

  The delta is applying these to BitNet b1.58 specifically and quantifying the interaction. This is valuable but represents a systematic application study, not a NeurIPS-level conceptual advance. To address: The paper would need a novel algorithmic contribution (e.g., a new quantization-aware distillation method) or substantially larger-scale validation (full ImageNet, production deployment results).
  2. No ImageNet results with the recipe: The paper acknowledges a 26% baseline gap on ImageNet (Table 11) but does not evaluate the proposed recipe on ImageNet. For a paper about "practical deployment," this is a critical omission. CIFAR/Tiny ImageNet results alone do not establish practical utility. To address: Run the full recipe (conv1 + KD) on ImageNet and report results; alternatively, acknowledge this as a
  CIFAR-scale study and adjust claims accordingly.

  [MAJOR]

  3. Numerical inconsistency in abstract: The abstract claims "80–130% for ResNet" recovery, but Table 8 shows ResNet recovery ranges from 80% to 122%. The 130% figure is for ConvNeXt-Tiny on Tiny ImageNet, not ResNet. This error undermines trust in the paper's precision. To address: Correct to "80–122% for ResNet."
  4. Statistical weakness of "exceeds FP32" claims:
    - CIFAR-100: n=3 seeds, p=0.028 (borderline significance), effect size stated as Cohen's d=3.1 which is implausibly large for this sample size
    - Tiny ImageNet: n=1 seed (acknowledged but still used for claims)

  The paper frames these as key findings ("exceeds FP32 on harder tasks") but the statistical evidence is weak. To address: Run additional seeds (at least 5-10) or remove the "exceeds FP32" framing and present as "competitive with FP32."
  5. Comparison with state-of-the-art quantization methods is inadequate: Table 11 compares against TWN and TTQ but acknowledges these use different architectures (ResNet-56 for TTQ vs ResNet-18 here). LSQ, EWGS, and PACT achieve 1-2% gaps on ImageNet at 4-bit precision with much simpler recipes than what this paper proposes. The paper does not establish why practitioners should use BitNet b1.58 + this recipe over these
  alternatives. To address: Either include head-to-head comparisons on the same architecture/dataset or clearly position this as a study of BitNet specifically rather than a practical recommendation.
  6. Overstated framing of "paradox": Calling the augmentation result a "paradox" implies something contradictory or deeply surprising. However, the explanation is straightforward: quantized models have less capacity and cannot exploit additional training signal. This is consistent with information theory and prior observations. The "paradox" framing feels like marketing. To address: Reframe as "asymmetric augmentation
  benefit" or similar; reserve "paradox" for genuinely counterintuitive findings.

  [MINOR]

  7. Per-architecture hyperparameter tuning undermines generalizability claims: Table 9 shows each architecture required different optimizers and learning rates. The paper claims the recipe "transfers" but the tuning requirements suggest otherwise. This should be more clearly acknowledged as a limitation.
  8. Missing ablation of KD hyperparameters combined with conv1: Section 5.2 mentions that lower α (0.5-0.7) works better in isolation, but the full recipe uses α=0.9. Was the combined recipe tested with optimized α? If α=0.7 + conv1 outperforms α=0.9 + conv1, this matters.
  9. No comparison with progressive or multi-stage training: ReActNet, BNext, and Real-to-Binary use progressive quantization and multi-stage training. The paper uses a simple single-stage 200-epoch recipe. A fairer comparison would match training compute.
  10. Appendix B's information-theoretic arguments are hand-wavy: Claims like "gap scales roughly with log₂(C)" (line 976) are presented without formal derivation. Either formalize or remove.

  ---
  Missing References and Comparisons

  ┌─────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┐
  │                          Paper                          │                             Why it should be cited                             │
  ├─────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ ABC-Net (Lin et al., 2017)                              │ Multiple binary bases approach; alternative to ternary                         │
  ├─────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ LQ-Nets (Zhang et al., 2018)                            │ Learned quantizers; should compare quantization strategies                     │
  ├─────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ ProxylessNAS (Cai et al., 2019)                         │ Hardware-aware architecture search; relevant to "practical deployment" claims  │
  ├─────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ Integer-only quantization (Jacob et al., 2018 - Google) │ Industry-standard quantization; BitNet claims need context                     │
  ├─────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ Once-for-All (Cai et al., 2020)                         │ Elastic deployment; alternative approach to compression                        │
  ├─────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ Data-Free KD (Lopes et al., 2017)                       │ KD without original training data; could strengthen recipe                     │
  ├─────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ FitNets (Romero et al., 2015)                           │ Feature-level distillation; paper mentions this as future work but should cite │
  └─────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────┘

  ---
  Questions for Authors

  1. ImageNet with recipe: Can you run the conv1+KD recipe on ImageNet and report results? The 26% baseline gap makes this critical for any "practical deployment" claim.
  2. Why not learned scales? TTQ's learned scales achieve better results. What is the practical advantage of BitNet's fixed {-1,0,+1} that justifies the larger accuracy gap? Is there actual inference speedup evidence?
  3. Augmentation × architecture interaction: Does the "augmentation widens gap" finding hold for MobileNetV2 and EfficientNet, or only ResNet? The paper's augmentation experiments use only ResNet.
  4. Why does KD help more on harder tasks? The paper offers an information-theoretic intuition (more pairwise relationships in soft labels), but this doesn't explain why the ternary student can exploit this information when it supposedly lacks capacity. Can you reconcile this?
  5. Seed variance for Tiny ImageNet: The "exceeds FP32" claim on Tiny ImageNet rests on a single seed. Can you provide multi-seed results?

  ---
  Scores

  ┌───────────────────────┬───────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │       Criterion       │ Score │                                                                             Justification                                                                              │
  ├───────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Novelty / Originality │ 3/10  │ Core techniques (FP32 first layer, KD) are well-established; the "paradox" is not genuinely surprising; contribution is systematic application rather than new insight │
  ├───────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Significance / Impact │ 4/10  │ CIFAR-scale results with known techniques; no ImageNet validation of recipe; limited practical guidance for practitioners who have access to LSQ/PACT/etc.             │
  ├───────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Technical Soundness   │ 6/10  │ Methodology is reasonable but statistical claims are weak (n=3, single-seed Tiny ImageNet); numerical inconsistency in abstract                                        │
  ├───────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Clarity / Writing     │ 8/10  │ Well-organized, clear exposition, honest limitations; slightly oversells with "paradox" framing                                                                        │
  ├───────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Experimental Rigor    │ 5/10  │ Good coverage within scope but scope is limited (CIFAR + Tiny ImageNet); no latency measurements; architecture-specific tuning undermines generalizability             │
  ├───────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Reproducibility       │ 9/10  │ Excellent documentation, code release, determinism guarantees; this is a strength                                                                                      │
  ├───────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Overall               │ 4/10  │ Solid empirical study that would be valuable as a workshop paper or technical report, but lacks the novelty and scale for NeurIPS main conference                      │
  └───────────────────────┴───────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Decision

  Reject

  This paper is a competent empirical study of applying known techniques (FP32 first layer, knowledge distillation) to a specific quantization scheme (BitNet b1.58) on small-scale benchmarks (CIFAR, Tiny ImageNet). The execution is reasonable and the reproducibility is excellent. However:

  1. The novelty is insufficient for NeurIPS main conference — all core techniques are established practice
  2. The "augmentation paradox" framing oversells what is a predictable capacity-limited phenomenon
  3. The lack of ImageNet results for the proposed recipe is a critical gap for practical relevance
  4. Statistical claims of "exceeding FP32" rest on minimal seeds

  This would be an excellent fit for the Efficient ML Workshop at NeurIPS or a venue like CVPR/ICCV Workshop on Efficient Deep Learning where systematic application studies are valued.

  ---
  What Would Change Your Decision?

  To move from Reject to Weak Accept:

  1. ImageNet results with the full recipe showing competitive accuracy (within 5% of TTQ/TWN) — this alone would substantially strengthen practical relevance
  2. Fix the numerical inconsistency in the abstract (80-130% → 80-122% for ResNet)
  3. Strengthen statistical claims: Either run 5+ seeds for Tiny ImageNet or remove the "exceeds FP32" framing for that dataset
  4. Reframe the "paradox": Present as a systematic quantification of a known phenomenon rather than a surprising discovery
  5. Add at least one novel algorithmic element (e.g., augmentation-adaptive distillation, layer-wise temperature scaling) that goes beyond combining existing techniques

  To move from Weak Accept to Accept would additionally require:

  6. Actual latency/throughput measurements on edge hardware demonstrating the claimed practical benefits
  7. Formal theoretical analysis of why augmentation benefits scale differently with model capacity


