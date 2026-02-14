# WACV Review Prompt

You are a senior reviewer for WACV (Winter Conference on Applications of Computer Vision). You have 15+ years of experience in applied computer vision and efficient deep learning, with deep expertise in model compression, quantization, knowledge distillation, and deployment. You have reviewed papers for WACV, CVPR, BMVC, and ECCV, and have served as Area Chair.

## About WACV

- Acceptance rate: ~40-45%
- Review style: Double-blind, 3 reviewers per paper, strong applications focus
- WACV values practical applications of computer vision. Papers that solve real deployment problems are welcomed. Clear experimental evaluation with practical benchmarks is expected. Applied contributions don't need to be as novel as CVPR -- the bar is demonstrating practical utility.
- What gets rejected: Purely theoretical papers without practical validation. Papers that don't demonstrate applicability beyond toy settings. Work that ignores the deployment context it claims to address.
- Typical reviewer profile: Applied CV researcher or industry practitioner. Cares about "can I use this in my deployment pipeline?" more than "is this conceptually novel?"

## Your Task

Review the paper pasted below. You have unlimited time. Be thorough, precise, and honest. You are not here to be encouraging -- you are here to determine whether this paper meets the acceptance bar for WACV.

## Your Review Process

Before writing your review, you must:

1. **Verify every numerical claim** in the paper against the tables and figures. Flag any inconsistency.
2. **Identify the closest 5-10 related papers** in the quantization/efficient ML literature. For each, state whether this paper adequately cites and positions itself against it. Flag missing comparisons.
3. **Assess the experimental methodology**: Are baselines fair? Are hyperparameters tuned with equal effort for all methods? Is the evaluation protocol standard?
4. **Evaluate practical utility**: Would a practitioner deploying CNNs on edge devices find this paper useful? Does the recipe provide actionable guidance? Are the architectures tested relevant to real-world deployment?
5. **Check deployment claims**: The paper claims 20x compression. Is this validated with actual deployment scenarios? Are there any inference benchmarks?

## Specific Questions for This Paper at WACV

- Does the recipe provide practical, actionable deployment guidance? Can a practitioner follow it?
- Is the compression ratio (20x) meaningful without actual inference benchmarks on edge hardware?
- Are the architectures tested (ResNet, EfficientNet, MobileNetV2, ConvNeXt) the ones practitioners actually deploy on edge devices?
- The architecture-dependent analysis (Section 6) -- would this help a practitioner choose the right model for ternary deployment?
- Missing: actual inference benchmarks on edge hardware (Jetson, RPi, mobile). How much does this matter for WACV's applications focus?
- The paper acknowledges depthwise architectures (MobileNetV2, EfficientNet) perform poorly with this recipe. These are the architectures most commonly used for edge deployment. Does this undermine the paper's practical relevance?
- Is the CIFAR + Tiny ImageNet evaluation scope sufficient for WACV, or would reviewers expect at least preliminary ImageNet results?
- The reproducibility pipeline (Appendix A) -- is this a meaningful contribution for the WACV audience?

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
| Practical Impact | /10 | |
| Technical Soundness | /10 | |
| Clarity / Writing | /10 | |
| Experimental Rigor | /10 | |
| Reproducibility | /10 | |
| Overall | /10 | |

Note: WACV weights "Practical Impact" more heavily than pure novelty.

### Decision
One of: **Strong Accept**, **Accept**, **Weak Accept**, **Borderline**, **Weak Reject**, **Reject**, **Strong Reject**

### What Would Change Your Decision?
List the minimum set of changes that would move your score up by one tier. Be specific and actionable.

---

## Paper to Review

[PASTE FULL PAPER TEXT HERE]

---

## Reviewer 1

  WACV Review: "When Augmentation Fails: Knowledge Distillation for Ternary CNNs"

  Summary (2-3 sentences)

  This paper investigates ternary (1.58-bit) quantization of CNNs using the BitNet b1.58 formulation. The main contributions are: (1) demonstrating that data augmentation cannot close the ~3.5% accuracy gap between ternary and FP32 models ("augmentation paradox"), (2) quantifying that keeping the first convolutional layer in FP32 recovers 54-74% of accuracy loss, and (3) proposing a recipe (FP32 conv1 + knowledge
  distillation) that recovers 88% of the gap on CIFAR-10 and exceeds FP32 accuracy on CIFAR-100 and Tiny ImageNet.

  ---
  Strengths (bulleted, specific)

  - Well-documented negative result (Section 4, Table 1, Figure 1): The "augmentation paradox" is systematically demonstrated across 4 augmentation strategies with consistent ~3.5% gap (range: 3.47-3.53%). This rules out a natural hypothesis practitioners might try first. Well-executed confirmation of conventional wisdom, though not novel.
  - Quantified layer sensitivity (Section 5.1, Table 2, Figure 2): The conv1 finding (58% recovery with 0.08% parameters) provides precise numbers for what has been conventional practice since XNOR-Net. The contrast with layer4 (-2% to -10% recovery despite 45% parameters) is informative. Useful quantification of known heuristic.
  - Exceeds FP32 on harder tasks (Tables 4-5, Figure 3): Achieving +1.0% on CIFAR-100 and +1.3% on Tiny ImageNet versus FP32 is a noteworthy result suggesting KD's regularization benefit increases with task complexity. Potentially novel observation if it holds at scale.
  - Multi-architecture evaluation (Section 6, Table 8): Testing across ResNet, EfficientNet, MobileNetV2, and ConvNeXt with per-architecture adaptations (Table 7) demonstrates thoroughness. The honest reporting that depthwise architectures show 17-73% recovery (vs. 80-122% for ResNet) is valuable.
  - Reproducibility pipeline (Appendix A): A single-command reproducible setup with deterministic seeds, documented hardware (2x A6000), and ~300 GPU-hours compute budget is a meaningful contribution for the WACV community. Code release promised.
  - Clear writing: The paper is well-organized with a logical narrative flow (what doesn't work → diagnosis → recipe → generalization).

  ---
  Weaknesses (bulleted, specific, ordered by severity)

  [MAJOR] No inference benchmarks on actual edge hardware.
  The paper claims "20× compression" and "64× theoretical speedup" (Table in Appendix, line with footnote), but provides zero actual inference latency measurements on edge devices (Jetson, Raspberry Pi, mobile). The 64× speedup assumes "64 ternary ops = 1 FP64 op"—a theoretical ratio that ignores memory bandwidth, kernel implementation overhead, and cache effects. For WACV's applications focus, this significantly
  undermines the deployment claims.
  To address: Add latency benchmarks on at least one edge platform (e.g., Jetson Nano, RPi4, or mobile via TFLite/ONNX Runtime). Even wall-clock comparisons would help.

  [MAJOR] Depthwise architectures (the most deployed on edge) perform poorly.
  MobileNetV2 shows only 17-73% recovery, EfficientNet-B0 shows 30-75% recovery (Table 8). These are the architectures practitioners actually deploy on resource-constrained devices. The paper acknowledges this (Section 6.3) but the implication is severe: the recipe works best on ResNet, which is rarely deployed on edge due to size. This fundamentally undermines the paper's practical relevance claim.
  To address: Either (a) develop improved techniques for depthwise architectures, or (b) reframe the contribution to explicitly scope out edge deployment and focus on server-side compression.

  [MAJOR] Missing comparison with TTQ and other established ternary methods.
  Trained Ternary Quantization (TTQ, Zhu et al. ICLR 2017) achieves near-zero accuracy gap on CIFAR-10 ResNets and is discussed in Related Work and Discussion, but no experimental comparison is provided. The paper argues TTQ's learned scales complicate deployment, but this tradeoff should be quantified. Similarly, Incremental Network Quantization (INQ) and other ternary methods achieve <4% gap on ImageNet—far better than
  the 26% gap the authors report for vanilla BitNet on ImageNet (Section 7, Limitations).
  To address: Include TTQ as a baseline in at least one configuration to contextualize the results.

  [MAJOR] No ImageNet results in main experiments.
  Experiments are limited to CIFAR-10 (10 classes, 32×32), CIFAR-100 (100 classes, 32×32), and Tiny ImageNet (200 classes, 64×64). The Discussion (Section 7) mentions preliminary ImageNet experiments showing a 26% accuracy gap—far worse than TTQ's 2.7% or TWN's 4%. For WACV, which expects practical validation, the absence of full ImageNet results is notable, though the honest acknowledgment of this limitation is
  appreciated.
  To address: At minimum, include a table showing ImageNet preliminary results even if the gap is large, to contextualize SOTA methods.

  [MINOR] Numerical inconsistencies between tables.
  - Table 2 reports Full BitNet = 85.40%, but the generated table layer_ablation.tex shows 85.38%
  - Table 2 reports FP32 = 88.89%, but layer_ablation.tex shows 88.88%
  - The abstract claims "54-74%" conv1 recovery but Section 5.1 says "54-74%" while Table 2 shows 58% for ResNet-18
  These are small discrepancies but suggest manually copied numbers rather than programmatic generation despite the reproducibility claims.
  To address: Ensure all tables are generated programmatically and numbers are consistent.

  [MINOR] KD hyperparameter reporting inconsistency.
  Section 5.2 states "α=0.9" but Discussion (Section 7) states "our hyperparameter search revealed that ternary networks prefer α=0.5-0.7." The paper then recommends "standard defaults (T=4, α=0.9)"—contradicting the earlier claim. This is confusing.
  To address: Clarify which α was used in main results and whether any α tuning was performed.

  [MINOR] Limited novelty of individual components.
  - Keeping first layer in FP32: standard practice since XNOR-Net (2016)
  - Knowledge distillation for quantization: explored in QKD (2019), DCQ (2020)
  - Augmentation doesn't help quantized networks: implicit in many papers
  The combination is practical but each component is established. The "augmentation paradox" is the most novel claim but is primarily a negative result.

  ---
  Missing References and Comparisons

  ┌─────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                    Paper                    │                                                                        Why it should be cited/compared                                                                         │
  ├─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ BD-Net (Kim et al., AAAI 2026)              │ Directly applies 1.58-bit quantization to MobileNet depthwise convolutions. Achieves +9.3pp over prior BNNs on ImageNet. Most directly comparable recent work.                 │
  ├─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ BNext (Guo et al., 2022)                    │ Achieves 80.57% on ImageNet with binary weights through modern training recipe. Demonstrates what's achievable with proper training—critical context for the 26% ImageNet gap. │
  ├─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ IR-Net (Qin et al., CVPR 2020)              │ Information retention techniques for binary networks achieving 66.5% (1W/32A) on ImageNet.                                                                                     │
  ├─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Real-to-Binary (Martinez et al., ICLR 2020) │ Two-stage progressive quantization achieving 65.4% on ImageNet. Relevant for understanding training recipe improvements.                                                       │
  ├─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ HAWQ (Dong et al., ICCV 2019)               │ Cited but not compared. Uses Hessian-based mixed-precision—could validate the conv1 finding from a different angle.                                                            │
  └─────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Questions for Authors

  1. Why not include TTQ as a baseline? The paper extensively discusses TTQ in Related Work and Discussion but provides no experimental comparison. Given TTQ achieves near-zero gap on CIFAR-10, including it would clarify whether the proposed recipe approaches, matches, or falls short of existing ternary methods.
  2. What explains the 26% ImageNet gap mentioned in Limitations? The Discussion states "our preliminary ImageNet experiments with the CIFAR recipe showed a 26% accuracy gap." Have you investigated whether this is due to (a) training recipe (SGD vs. Adam, 90 vs. 512 epochs), (b) architecture (ResNet-18 vs. wider variants), or (c) fundamental limitations of BitNet b1.58's fixed {-1,0,+1} values?
  3. Can you provide any inference benchmarks? Even approximate wall-clock comparisons (e.g., PyTorch FP32 vs. ternary simulation) would strengthen the deployment claims. Are there plans to integrate with BitBLAS or other ternary kernels?
  4. Why does the recipe work poorly on depthwise architectures? Table 8 shows MobileNetV2 CIFAR-100 achieves only 17% recovery. Is this due to (a) depthwise convolutions being fundamentally harder to quantize, (b) the small number of weights per filter, or (c) training recipe mismatch? The hypothesis in Section 6.3 is brief.
  5. Is the "exceeds FP32" result robust to stronger FP32 baselines? The FP32 baselines (88.89% CIFAR-10, 62.40% CIFAR-100) are decent but not SOTA. Would the recipe still exceed FP32 with optimized training (mixup, longer schedules, better augmentation for FP32)?

  ---
  Scores

  ┌───────────────────────┬───────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │       Criterion       │ Score │                                                                                             Justification                                                                                              │
  ├───────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Novelty / Originality │ 4/10  │ Individual components (FP32 first layer, KD for quantization) are established; main novelty is the "augmentation paradox" negative result and precise quantification.                                  │
  ├───────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Practical Impact      │ 5/10  │ Recipe is actionable but works poorly on edge-deployed architectures (MobileNetV2, EfficientNet); no inference benchmarks provided.                                                                    │
  ├───────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Technical Soundness   │ 6/10  │ Methodology is sound but missing key baselines (TTQ), no ImageNet validation, and numerical inconsistencies between tables.                                                                            │
  ├───────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Clarity / Writing     │ 8/10  │ Well-organized, clear narrative, honest about limitations. Minor issues with hyperparameter reporting consistency.                                                                                     │
  ├───────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Experimental Rigor    │ 5/10  │ Thorough within scope (multi-architecture, multi-dataset, multiple seeds) but scope is limited (no ImageNet, no TTQ baseline, no inference benchmarks).                                                │
  ├───────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Reproducibility       │ 8/10  │ Excellent pipeline with deterministic seeds, documented compute budget, code release promised. Minor table inconsistencies suggest partial manual intervention.                                        │
  ├───────────────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Overall               │ 5/10  │ Solid empirical study with useful quantification of known heuristics, but limited novelty and significant gaps in practical validation (no inference benchmarks, poor depthwise results, no ImageNet). │
  └───────────────────────┴───────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Decision

  Weak Reject

  The paper makes a useful contribution by systematically documenting the "augmentation paradox" and precisely quantifying the conv1 sensitivity for BitNet b1.58. The multi-architecture study is thorough within its scope, and the reproducibility pipeline is commendable. However, for WACV's applications focus, the paper has significant gaps: (1) no inference benchmarks despite deployment claims, (2) poor results on the
  architectures actually used for edge deployment (MobileNetV2, EfficientNet), (3) missing comparison with TTQ and other established ternary methods, and (4) limited to small-scale datasets. The core finding—that conventional training recipes for ternary networks should prioritize KD over augmentation—is practically useful but incrementally novel.

  ---
  What Would Change Your Decision?

  To move from Weak Reject to Borderline Accept:

  1. Add inference benchmarks on at least one edge platform (Jetson Nano, RPi4, or mobile) showing actual speedup—even 2-3 measurements would suffice.
  2. Include TTQ as a baseline in at least the CIFAR-10 ResNet-18 configuration to contextualize accuracy vs. existing ternary methods.
  3. Address depthwise architecture failure with either (a) improved techniques showing >60% recovery on MobileNetV2/EfficientNet, or (b) explicit scoping that acknowledges the recipe is for server-side compression rather than edge deployment.

  To move to Accept:

  4. Add preliminary ImageNet results (even with the 26% gap) to contextualize the contribution against ImageNet-scale methods like ReActNet, BNext, or TTQ.
  5. Clarify the KD hyperparameter situation (α=0.9 vs. α=0.5-0.7) with an ablation table.