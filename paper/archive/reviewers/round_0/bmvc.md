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

## Reviewer 1

Summary (2-3 sentences)

  This paper investigates ternary (1.58-bit) quantization of CNNs using the BitNet b1.58 formulation. The authors claim that data augmentation fails to close the accuracy gap between ternary and FP32 models, identify the first convolutional layer as responsible for 54-74% of accuracy loss, and propose a recipe combining FP32 conv1 with knowledge distillation that recovers 88% of the gap on CIFAR-10 and reportedly exceeds
  FP32 on harder tasks.

  ---
  Strengths (bulleted, specific)

  - Reproducibility commitment (Appendix A): The code-to-paper pipeline with deterministic seeds and regeneration commands is exemplary. BMVC reviewers will appreciate this.
  - Layer-wise ablation quantification (Table 4, Section 5.1): While keeping first/last layers in FP32 is known practice, the precise quantification (58% recovery from 0.08% parameters) provides useful practitioner guidance. This is well-executed confirmation of known results, not novel.
  - Architecture-dependent analysis (Section 6, Table 8): Testing across ResNet, EfficientNet, MobileNetV2, and ConvNeXt with per-architecture tuning is valuable. The finding that depthwise separable architectures recover poorly (17-73% vs 80-122% for ResNet) is actionable for practitioners.
  - Task complexity scaling (Figure 8, Section 7): The observation that KD benefits scale with task complexity (36% recovery on CIFAR-10 vs 57% on CIFAR-100) is interesting and provides useful intuition.
  - Honest limitations section (Section 7): The acknowledgment that ImageNet experiments showed a 26% gap and that the recipe is not plug-and-play demonstrates scientific honesty.

  ---
  Weaknesses (bulleted, specific, ordered by severity)

  [FATAL] The "augmentation paradox" claim is not supported by the data

  The paper claims "the accuracy gap remains constant at approximately 3.5% regardless of augmentation strategy" (Table 1, Section 4.2). However, the generated table augmentation_ablation.tex shows:

  ┌───────────┬───────────┬───────┬─────────┬────────┬────────┐
  │   Model   │  Dataset  │ Basic │ RandAug │ Cutout │  Full  │
  ├───────────┼───────────┼───────┼─────────┼────────┼────────┤
  │ ResNet-18 │ CIFAR-10  │ 2.99% │ 3.44%   │ 3.62%  │ 3.90%  │
  ├───────────┼───────────┼───────┼─────────┼────────┼────────┤
  │ ResNet-18 │ CIFAR-100 │ 3.72% │ 5.47%   │ 4.34%  │ 6.88%  │
  ├───────────┼───────────┼───────┼─────────┼────────┼────────┤
  │ ResNet-50 │ CIFAR-10  │ 3.75% │ 5.25%   │ 5.39%  │ 6.28%  │
  ├───────────┼───────────┼───────┼─────────┼────────┼────────┤
  │ ResNet-50 │ CIFAR-100 │ 9.11% │ 10.38%  │ 9.05%  │ 11.25% │
  └───────────┴───────────┴───────┴─────────┴────────┴────────┘

  The gap varies from 2.99% to 11.25%—a nearly 4× range. On CIFAR-100/ResNet-18, it varies from 3.72% to 6.88% (+85% increase). This directly contradicts the core narrative that augmentation "doesn't matter." The augmentation paradox appears to be an artifact of selective reporting on CIFAR-10/ResNet-18 only.

  To address: The authors must either (a) explain why only CIFAR-10/ResNet-18 results are presented in Table 1 when their own data shows significant variation, or (b) retract the "augmentation paradox" framing and reframe as "augmentation does not reliably close the gap."

  ---
  [MAJOR] Numerical inconsistencies between inline tables and generated data

  The inline Table 1 (line 248-253) reports:
  - Basic: FP32 = 88.89%, BitNet = 85.40%, Gap = -3.49%

  But accuracy_basic.tex shows:
  - ResNet-18/CIFAR-10: Std = 88.88%, Bit = 85.89%, Gap = 2.99%

  These don't match. Either the inline tables are hand-edited estimates or the generated tables are from different runs. This undermines the reproducibility claim.

  To address: Regenerate all tables from the same data source and verify consistency.

  ---
  [MAJOR] Missing baselines that BMVC reviewers would expect

  The paper lacks comparison with standard quantization methods:
  1. QAT (Quantization-Aware Training): The standard baseline for any quantization paper
  2. LSQ (Learned Step Size Quantization): A leading method for learned-scale quantization
  3. TTQ (Trained Ternary Quantization): Mentioned in related work but not compared experimentally. The TTQ research notes show TTQ achieves near-zero gap on CIFAR-10—why is BitNet b1.58 so much worse?

  Without these baselines, it's impossible to assess whether the proposed recipe is competitive with existing methods.

  To address: Add at least one comparison with QAT or LSQ on the same architecture/dataset.

  ---
  [MAJOR] Statistical significance concerns

  - Only 3 seeds used; insufficient for robust statistical claims
  - Several key results lack standard deviations (Table 6: keep_conv1 = 87.40 with no std; Table 7: CIFAR-100 recipe results)
  - The claims of "exceeding FP32" (Table 6: 63.40 vs 62.40, +1.0%) are within typical run-to-run variance and not tested for significance

  To address: Report confidence intervals for all key comparisons; perform t-tests for the "exceeds FP32" claims.

  ---
  [MINOR] The novelty claim about conv1 is overstated

  Section 1 claims "we provide the first precise quantification for BitNet b1.58." However, keeping first/last layers in FP32 has been standard since XNOR-Net (2016) and DoReFa-Net (2016), and TTQ also follows this practice (as noted in the research files). The quantification is useful but not novel—it confirms established practice.

  To address: Reframe as "we validate the established convention with empirical rigor" rather than claiming a first.

  ---
  [MINOR] KD hyperparameter claims are confusing

  Section 7 states "we recommend standard defaults (T=4, α=0.9)" but also mentions "ternary networks prefer α=0.5-0.7." The conclusion (line 582) mentions hyperparameter interactions but provides no resolution. Which should practitioners use?

  To address: Provide clear, actionable guidance.

  ---
  [MINOR] 64× speedup claim is theoretical only

  The efficiency table claims 64× speedup but Section 7 explicitly states "no real latency measurements." This theoretical speedup assumes specialized hardware/kernels that may not exist.

  To address: Either measure actual latency or add prominent caveats about theoretical nature.

  ---
  Missing References and Comparisons

  ┌──────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────┐
  │                  Paper                   │                                 Why It Should Be Cited                                  │
  ├──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ LSQ (Esser et al., ICLR 2020)            │ Leading learned-scale quantization method; direct competitor                            │
  ├──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ PACT (Choi et al., ICML 2018)            │ Parameterized clipping for activation quantization                                      │
  ├──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ ReActNet (Liu et al., ECCV 2020)         │ State-of-the-art binary network achieving 69.4% on ImageNet; mentioned but not compared │
  ├──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ BNext (Guo et al., 2022)                 │ Achieves 80.6% on ImageNet with binary weights; mentioned but not compared              │
  ├──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ Real2Binary (Martinez et al., ICLR 2020) │ Mentioned in limitations but missing from related work                                  │
  ├──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ PROM (Meiner et al., 2025)               │ Recent work on ternary pointwise convolutions                                           │
  └──────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Questions for Authors

  1. Why do the inline tables (Table 1-3) show different values than the generated .tex files? For example, Table 1 shows BitNet=85.40% but accuracy_basic.tex shows 85.89%.
  2. How do you reconcile the "constant ~3.5% gap" claim with the augmentation ablation data showing gaps ranging from 2.99% to 11.25%?
  3. Why is TTQ not included as an experimental baseline, given that your own research notes document it achieves near-zero gap on CIFAR-10 ResNets?
  4. For the "exceeds FP32" claims (Table 6: +1.0% on CIFAR-100), are these differences statistically significant? What is the p-value from a paired t-test?
  5. The paper acknowledges a 26% gap on ImageNet. What prevents including these results, even as negative results, to provide a complete picture?


## Reviewer 2


# Literature gaps and missing baselines in "When Augmentation Fails"

**The paper under review makes several contributions—applying BitNet b1.58 to CNNs, documenting an "augmentation paradox," and combining FP32 first layers with knowledge distillation—but a thorough literature search reveals significant missing citations, absent experimental baselines, and prior work that partially anticipates key claims.** The most critical gaps are the omission of established ternary methods (TTQ, STTN, TRQ) as baselines, insufficient engagement with the rich KD-for-quantization literature post-2020, and incomplete acknowledgment that keeping first/last layers in FP32 is a well-documented convention rather than a novel finding. Below is a detailed accounting organized by research area.

---

## BitNet b1.58 for CNNs remains genuinely underexplored

The paper's core premise—transferring BitNet b1.58's ternary weight scheme from LLMs to standard CNNs—sits in a real gap. Beyond the two works the paper already cites (Nielsen & Schneider-Kamp's "BitNet b1.58 Reloaded," 2024; BD-Net at AAAI 2026), no published work directly applies BitNet b1.58's training recipe to standard CNN convolution layers like ResNet or VGG. The closest related efforts target vision transformers: **TerViT** (2022) introduced ternary weight quantization for DeiT and Swin with progressive training (8-bit → ternary); **TernaryCLIP** (arXiv, October 2025) ternarizes CLIP's ViT-based vision encoder achieving 16.98× compression; and an "Efficient Ternary Weight Embedding Model" (arXiv, November 2024) applies ternary weights to ViT-based image classifiers on CIFAR-10/100 and ImageNet-1k using self-taught knowledge distillation. BD-Net (AAAI 2026) is particularly relevant as it introduces **1.58-bit convolution** for depthwise separable layers, directly addressing the challenge of ternarizing lightweight CNN operations.

However, the paper should cite the **Ternary Quantization Survey** by Dan Liu and Xue Liu (McGill, arXiv 2303.01505, 2023), which provides a comprehensive taxonomy of ternary methods organized around projection functions and proximal operators. The active **Workshop on Binary and Extreme Quantization for Computer Vision** at CVPR 2024/2025 and ICCV 2025 should also be mentioned to contextualize the field's momentum.

## Critical ternary baselines are missing from experimental comparisons

The paper's experimental evaluation omits several established ternary quantization methods that represent stronger baselines than a naive BitNet b1.58 application. The most important missing comparisons:

**TTQ (Trained Ternary Quantization)** by Zhu, Han, Mao, and Dally (ICLR 2017) learns asymmetric ternary values {−Wₙ, 0, +Wₚ} rather than constraining to {−1, 0, +1}. On CIFAR-10, TTQ's ternary ResNet-32/44/56 **exceed full-precision baselines** by +0.04/+0.16/+0.36%, reaching **93.56%** with ResNet-56. The paper under review's **88.48%** with ternary ResNet-18 on CIFAR-10 is substantially lower, and the absence of TTQ as a baseline makes it impossible to assess whether the gap stems from the {−1, 0, +1} constraint, the architecture, or training methodology.

**STTN (Soft Threshold Ternary Networks)** by Xu et al. (IJCAI 2020) removes hard threshold constraints and achieves **66.2% top-1 on ImageNet** with ternary weights AND activations on ResNet-18. **TRQ (Ternary Neural Networks with Residual Quantization)** by Li et al. (AAAI 2021) uses recursive stem-residual quantization, achieving near-lossless ternarization on CIFAR-100 (0.3% drop) and outperforming XNOR by 3.5% on CIFAR-10 with ResNet-18. **INQ (Incremental Network Quantization)** at ICLR 2017 achieved **~69.5% on ImageNet** with 2-bit ternary ResNet-18 weights—nearly matching full precision.

For QAT comparisons, the paper should benchmark against **LSQ** (Esser et al., ICLR 2020), which achieves **67.6% top-1** on ImageNet at 2-bit ResNet-18 through learned step sizes, and **PACT** (Choi et al., 2018), which established parameterized activation clipping. The classic **TWN** (Li & Liu, 2016) and **DoReFa-Net** (Zhou et al., 2016) are essential ternary baselines. For a complete picture, **RTN** (Li et al., AAAI 2020) at ~63.3% top-1 on ImageNet (W2/A2) provides context for full ternary quantization.

A key interpretive issue: if the paper ternarizes only weights (W=ternary, A=FP32), then 88.48% on CIFAR-10 with ResNet-18 is weak compared to TTQ's 93%+ results. If both weights and activations are ternary, the result is more defensible but should be compared to STTN and TRQ.

## The "augmentation paradox" appears novel but needs stronger contextualization

The claim that standard augmentation (RandAugment, CutMix, MixUp) hurts ternary network performance is the paper's most potentially novel contribution. **No prior work explicitly names or systematically studies an "augmentation paradox" for ternary/binary networks.** However, several papers partially anticipate this finding:

**BNext** (Guo et al., 2022) explicitly analyzes the data augmentation pipeline for binary networks as one of three optimization perspectives. The authors found that training recipes from full-precision models need redesign for BNNs and identified a **"counter-intuitive overfitting problem"** when training highly accurate binary models with standard recipes. This is the closest published precedent.

**Rebuffi et al. (NeurIPS 2021)** demonstrated that MixUp is significantly less effective than CutMix/Cutout for adversarial training, finding that "low-level features tend to be destroyed by MixUp." They also showed that specific RandAugment components (Posterize, Invert) are **detrimental** to adversarial robustness—establishing that augmentation effectiveness is highly context-dependent. **KeepAugment** (Gong et al., 2020) showed CutMix can "destroy the main characteristic information" needed for classification, with CutMix dropping ResNet-50 detection from 37.9 to 23.7 mAP.

A 2025 study in Nature Scientific Reports on BNN robustness to fixed-pattern noise showed that strategic noise augmentation during training can help BNNs, but general augmentation effects on binary networks are complex—BNNs with 4-bit batch norm showed "greater sensitivity to all noise types." The paper should cite these works and position its augmentation paradox finding within the broader context of augmentation-regime interactions.

## Knowledge distillation literature for quantized networks has advanced substantially since QKD

The paper's use of KD from an FP32 teacher is well-motivated but its related work appears to stop at QKD (2019). The field has advanced considerably:

**SQAKD** (Zhao & Zhao, AISTATS 2024) is the most directly relevant recent work. It uses the full-precision model as its own teacher in a self-supervised framework, achieving up to **15.86% improvement** over standalone QAT methods (EWGS, PACT, LSQ, DoReFa) across 1–8-bit quantization. Critically, SQAKD **benchmarked 11 different KD methods in the QAT context** and found that relational/structural methods (CRD, RKD, CKTF) outperform simpler feature-matching methods (FitNet, SP) for quantized models. It evaluates on CIFAR-10/100 and Tiny ImageNet with VGG, ResNet, and MobileNetV2—overlapping directly with the paper under review's setup.

**QFD (Quantized Feature Distillation)** at AAAI 2023 discovered that **quantized features are paradoxically better teachers** than full-precision features for quantized students—a finding that should inform the paper's distillation strategy. **FAQD (Feature Affinity Assisted KD)** (2023) achieves near-FP accuracy for binarized students on CIFAR-10/100 and Tiny ImageNet using affinity matrix alignment. **SKD-BNN** (Applied Intelligence, 2024) proposes self-KD for binary networks achieving **93.0% on CIFAR-10** using full-precision weights as feature-level teachers plus historical soft-label banks.

For binary network distillation specifically, the paper should acknowledge that **KD is standard practice**, not a novel addition. BNext uses "Diversified Consecutive KD" with gap-aware teacher selection; ReActNet (ECCV 2020) uses distributional loss as implicit KD; PokeBNN (Google, 2021) found that teacher accuracy beyond a threshold doesn't help—ViT teachers produce similar results to ResNet-50 teachers. **MAD (Multi-bit Adaptive Distillation)** at BMVC 2021 distills from multiple teachers of different bit-widths simultaneously—this paper was published at the same venue and should be cited.

The **Apprentice** framework (Mishra & Marr, 2018) specifically measured KD improvements for ternary networks: joint training with FP teacher improved ternary ResNet ImageNet accuracy from ~33.4% to ~29.7% top-1 error—a **~3.7% improvement** from KD alone, providing a baseline for expected gains.

## First-layer FP32 retention is well-established convention, not a novel finding

The paper's claim about keeping the first convolutional layer in FP32 should be framed as empirical validation of an existing practice rather than a novel contribution. The convention dates to the earliest binary network papers:

**XNOR-Net** (Rastegari et al., ECCV 2016) established the practice, arguing first/last layers constitute a small fraction of computation. **DoReFa-Net** (Zhou et al., 2016) provided empirical justification, stating "we do not quantize the first and last layers" because "for the first layer, the input is often an image with 8-bit features." **BNN** (Hubara et al., NeurIPS 2016) kept the first layer in fixed-point, arguing it was too small to matter computationally. The **Binary Neural Networks Survey** (Yuan et al., Pattern Recognition 2020) codified this as "common sense that the first and last layers should be kept in higher precision."

More rigorous studies exist. **HAWQ** (Dong et al., ICCV 2019) used Hessian eigenvalues to show layers differ by **orders of magnitude** in quantization sensitivity, providing the theoretical foundation for mixed-precision allocation. **PACT/SAWB** (Choi et al., 2018) measured that INT8 first/last layers preserve accuracy but lower precision degrades it. **BNext** (2022) provides an **explicit ablation table (Table 7)** showing that INT8 quantization of first/last layers drops accuracy from 80.57% to 80.47%—a minimal but measurable loss. Most directly, **Ponte** (Xu et al., Sensors 2024) specifically studies feasibility of binarizing first/last layers, identifying three challenges: loss of non-linearity from XNOR replacement, reduced capacity for input features, and information loss at the first-to-second layer transition. **MoPEQ** (2025) confirmed via Hessian trace that "initial layers are highly sensitive to quantization."

- **BNext Table 7**: 80.57% → 80.47% when first/last layers quantized to INT8
- **Ponte (2024)**: Proposes techniques to fully binarize first/last layers, identifying the problem directly
- **HAWQ (2019)**: Orders-of-magnitude Hessian differences between layers
- **Smart Quantization (Razani et al., CVPR-W 2021)**: "Accuracy dropped significantly" when quantizing first/last layers of VGG-16

The paper's contribution here would need to provide significantly more detailed ablation studies than what exists to claim novelty—perhaps per-architecture, per-dataset measurements with modern architectures like ConvNeXt and EfficientNet.

## Recent binary/ternary papers the review should acknowledge

Several 2023–2026 papers advance binary/ternary networks for vision and should appear in the related work:

- **NeurIPS 2024**: "Training Binary Neural Networks via Gaussian Variational Inference and Low-Rank Semidefinite Programming" (Orecchia et al.) replaces heuristic STE with theoretically motivated optimization, consistently outperforming all SOTA BNN methods on CIFAR-10, CIFAR-100, Tiny-ImageNet, and ImageNet
- **ICML 2024**: "Learning 1-bit Tiny Object Detector with Discriminative Feature Refinement"
- **NeurIPS 2023**: "Understanding Neural Network Binarization with Forward and Backward Proximal Quantizers" (Lu et al.)
- **ICML 2023**: **BiBench**—a comprehensive benchmark for evaluating binary neural networks
- **AAAI 2023**: **ReBNN** achieves **66.9% top-1** on ImageNet with binary ResNet-18
- **CVPR 2025**: Normalized binary convolution for multi-spectral image fusion
- **Efficiera Residual Networks** (2024): Fully binary weights with 2-bit activations achieving **72.5% ImageNet** with ResNet-50-compatible architecture—notably quantizes ALL layers including first/last

For lightweight architectures, the paper should note that **ternary results for MobileNetV2 and EfficientNet-B0 are essentially absent** from the literature, making its results on these architectures genuinely novel. ReActNet (ECCV 2020) was the first to successfully binarize a MobileNet-based architecture (69.4% ImageNet). BD-Net (AAAI 2026) addresses depthwise convolution binarization. **TQuant** (arXiv 2306.17442, 2023) attempted ternary weight quantization on MobileNetV2 and EfficientNet-B0 but reported significant accuracy drops requiring special operators.

## Conclusion

The paper occupies a genuine gap at the intersection of BitNet b1.58 and CNN quantization, and its augmentation paradox finding and lightweight architecture results appear novel. However, the literature review has three critical weaknesses. First, **established ternary baselines (TTQ, STTN, TRQ, INQ) are absent**, making it impossible to contextualize the 88.48% CIFAR-10 result—TTQ achieves 93.56% with ternary ResNet-56 on CIFAR-10. Second, **post-2020 KD-for-quantization work is overlooked**: SQAKD (AISTATS 2024), QFD (AAAI 2023), SKD-BNN (2024), and MAD (BMVC 2021) all provide directly comparable methods and benchmarks. Third, **first-layer FP32 retention is a 10-year-old convention** with existing ablation studies (BNext Table 7, HAWQ Hessian analysis, Ponte 2024), so this should not be framed as a novel finding without substantially more rigorous measurement. The paper's strongest novel contributions are the augmentation paradox documentation and ternary results on MobileNetV2/EfficientNet—architectures where essentially no prior ternary benchmarks exist.