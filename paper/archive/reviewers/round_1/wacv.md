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

  Summary

  This paper investigates ternary (1.58-bit) quantization of CNNs using the BitNet b1.58 formulation. The main findings are: (1) data augmentation disproportionately benefits FP32 models over ternary models, widening rather than closing the accuracy gap; (2) keeping the first convolutional layer in FP32 recovers 54-74% of accuracy loss; (3) combining FP32 conv1 with knowledge distillation recovers 88% of the gap on
  CIFAR-10 and exceeds FP32 accuracy on harder tasks. The paper tests across four CNN families, finding the recipe works well for standard convolutions but poorly for depthwise separable architectures.

  ---
  Strengths

  - Well-executed "augmentation paradox" analysis (Section 4, Tables 1-2): The systematic evaluation across 4 augmentation strategies × 4 model-dataset configurations provides convincing evidence that augmentation widens the gap. The 10.9× differential benefit for FP32 (Table 2) is a useful quantitative insight for practitioners.
  - Precise quantification of conv1 importance (Section 5.1, Table 3): While keeping early layers in FP32 is established practice, the paper provides the first precise measurement for BitNet b1.58: 58% recovery with 0.08% parameters. This is a well-executed confirmation of known wisdom with actionable numbers.
  - Cross-architecture evaluation (Section 6, Table 8): Testing across ResNet, EfficientNet, MobileNetV2, and ConvNeXt with architecture-specific adaptations (Table 7) provides practical guidance. The finding that depthwise separable architectures recover poorly (17-73%) is valuable negative knowledge.
  - Honest treatment of limitations (Section 7): The paper acknowledges the 26% ImageNet gap, limited statistical power (n=3), lack of latency measurements, and that the recipe isn't plug-and-play. This transparency is appreciated.
  - Reproducibility pipeline (Appendix A): The code-to-paper pipeline with deterministic seeds and clear directory structure enables verification. This is a meaningful contribution for the applied CV community.
  - Clear practical takeaway: "For ternary CNNs, invest in distillation, not augmentation" is an actionable message that practitioners can apply immediately.

  ---
  Weaknesses

  [MAJOR] No inference benchmarks undermine the deployment narrative: The paper claims 20× compression and 64× theoretical speedup but provides zero actual measurements. For a paper positioning itself as enabling "practical ternary CNN deployment" (line 452), this is a significant gap. WACV's applications focus expects validation that the theoretical benefits translate to real hardware.
  - To address: Add inference benchmarks on at least one edge platform (Jetson Nano, RPi 4, or mobile) showing actual memory reduction and speedup. Even preliminary numbers would substantially strengthen the paper.

  [MAJOR] The recipe fails on the architectures practitioners actually deploy: MobileNetV2 and EfficientNet—the dominant architectures for edge deployment—show 17-73% recovery (Table 8). The paper honestly acknowledges this but doesn't adequately address the implications. A recipe that works well for ResNet but poorly for mobile-optimized architectures has limited practical relevance for edge deployment.
  - To address: Either (1) develop architecture-specific variants of the recipe for depthwise separable networks, or (2) reframe the contribution as specifically for ResNet-family deployment and adjust the abstract/claims accordingly.

  [MAJOR] No ImageNet results with the recipe: The paper reports a 26% baseline gap on ImageNet (Table 9) but doesn't apply the recipe to close it. For WACV's applied focus, CIFAR-10/100 and Tiny ImageNet are toy benchmarks. The practical relevance hinges on whether this recipe scales, and that remains unanswered.
  - To address: Run the recipe (conv1 + KD) on ImageNet and report results, even if gap recovery is modest. Alternatively, provide a clear explanation of why the recipe doesn't transfer (beyond "not optimized for scale").

  [MINOR] Statistical significance claims with n=3: The CIFAR-100 "exceeds FP32" claim relies on a one-sided t-test with 3 samples (p=0.028). While the paper notes "limited power," claiming statistical significance with n=3 is borderline. The Tiny ImageNet result is from a single seed.
  - To address: Add 2-3 more seeds for the key claims, or soften the statistical language.

  [MINOR] Comparison with TTQ is incomplete: Table 9 compares with TTQ but notes "ResNet-56" vs "ResNet-18" architectures aren't comparable. The claimed advantage (integer-only inference) is theoretical without actual deployment validation. TTQ achieves near-zero gap; this paper achieves 0.40% with more complexity.
  - To address: Run TTQ on the same architectures/datasets for fair comparison, or acknowledge more directly that BitNet b1.58's advantages are primarily about deployment simplicity, not accuracy.

  [MINOR] Architecture adaptation required (Table 7): The recipe requires per-architecture tuning (different optimizers, learning rates). This reduces the "plug-and-play" value and increases adoption friction.
  - To address: Provide clearer guidance on how practitioners should determine these settings for new architectures, or propose a heuristic.

  ---
  Missing References and Comparisons

  1. Jacob et al. (2018) "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference": Foundational work on integer-only quantization from Google, highly relevant to the deployment claims.
  2. MobileOne (Vasu et al., 2023): Recent efficient architecture for edge deployment; should be discussed in the architecture analysis.
  3. TinyML literature (Banbury et al., 2021): The TinyMLPerf benchmark and related work on edge deployment methodology should be referenced given the deployment framing.
  4. FQ-ViT and quantization for vision transformers: While outside scope, a brief acknowledgment of how this relates to ViT quantization would strengthen positioning.
  5. DepthShrinker (Fu et al., 2022): Directly addresses depthwise separable quantization challenges; relevant to Section 6's findings.

  ---
  Questions for Authors

  1. Why no inference benchmarks? Given WACV's applications focus, what prevented including even preliminary latency/memory measurements on edge hardware?
  2. Why does the recipe fail for depthwise architectures? Section 6 documents the failure but doesn't explain why conv1 + KD doesn't help. Is the bottleneck elsewhere in depthwise networks?
  3. Would the recipe transfer to ImageNet? The 26% baseline gap is much larger than CIFAR. Do you expect the recipe to recover a similar percentage, or are there fundamental scaling issues?
  4. How sensitive is the recipe to KD hyperparameters at scale? You mention T=4, α=0.9 works for up to 200 classes. What happens with 1000 classes?
  5. Practical deployment path: If MobileNetV2/EfficientNet don't work well with this recipe, what would you recommend to a practitioner who needs to deploy on mobile? Use ResNet despite being less efficient? Accept the larger gap?

  ---
  Scores

  ┌───────────────────────┬────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │       Criterion       │ Score  │                                                                   Justification                                                                    │
  ├───────────────────────┼────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Novelty / Originality │ 5/10   │ "Augmentation paradox" is interesting but conv1+KD recipe confirms known practices; no new techniques proposed                                     │
  ├───────────────────────┼────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Practical Impact      │ 5/10   │ Recipe works for ResNet but fails for mobile architectures; no deployment validation                                                               │
  ├───────────────────────┼────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Technical Soundness   │ 7/10   │ Methodology is sound, claims verified, but statistical power limited                                                                               │
  ├───────────────────────┼────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Clarity / Writing     │ 8/10   │ Well-written, clear structure, honest about limitations                                                                                            │
  ├───────────────────────┼────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Experimental Rigor    │ 6/10   │ Thorough within scope but missing ImageNet results and real inference benchmarks                                                                   │
  ├───────────────────────┼────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Reproducibility       │ 9/10   │ Excellent: deterministic seeds, clear pipeline, code release promised                                                                              │
  ├───────────────────────┼────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Overall               │ 5.5/10 │ Useful empirical study with actionable insights, but practical impact limited by failure on mobile architectures and lack of deployment validation │
  └───────────────────────┴────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Decision

  Weak Reject

  The paper provides a useful empirical study with some actionable insights (the augmentation paradox, conv1 importance quantification). However, for a venue that emphasizes practical applications:

  1. The recipe fails on the architectures practitioners actually deploy on edge devices (MobileNetV2, EfficientNet)
  2. No inference benchmarks validate the claimed compression benefits
  3. Results are limited to CIFAR/Tiny ImageNet—no evidence the recipe scales to ImageNet

  The gap between the deployment narrative and the actual validation is too large for WACV's applications-focused bar.

  ---
  What Would Change Your Decision?

  To move to Borderline/Weak Accept, the authors would need to address at least TWO of:

  1. Add inference benchmarks on actual edge hardware (Jetson Nano, RPi 4, or mobile): Show memory footprint and inference latency comparing FP32, full BitNet, and the recipe. Even preliminary numbers on one platform would significantly strengthen practical relevance.
  2. Apply the recipe to ImageNet and report results: Even if gap recovery is lower than CIFAR, showing the recipe provides some benefit at scale would demonstrate practical relevance.
  3. Develop a variant that works for depthwise architectures: The current recipe recovers only 17-73% for MobileNetV2. A simple adaptation (e.g., keeping more layers in FP32, different KD strategy) that improves this to >50% would address the mobile deployment gap.

  Addressing all three would move the paper to Accept.
