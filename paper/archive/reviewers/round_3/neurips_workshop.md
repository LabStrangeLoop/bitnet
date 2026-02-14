NeurIPS Workshop Review: "When Augmentation Fails: Knowledge Distillation for Ternary CNNs"

  ---
  Summary (2-3 sentences)

  This paper investigates training strategies for BitNet b1.58 (ternary) CNNs and demonstrates that data augmentation widens rather than closes the accuracy gap to FP32 models. The authors identify conv1 as responsible for 54-74% of accuracy loss and propose a recipe (FP32 conv1 + knowledge distillation) that recovers 89% of the gap on CIFAR-10 and exceeds FP32 on harder tasks (+1.0% CIFAR-100, +1.7% Tiny ImageNet).

  ---
  Strengths (bulleted, specific)

  - Counterintuitive, discussion-worthy finding: The "augmentation paradox" (Table 1) showing gaps widening by 12-59% with stronger augmentation challenges conventional wisdom and would generate debate at a poster session
  - Quantified conv1 contribution (Table 2): The 58% gap recovery with 0.08% parameters is the first systematic quantification for BitNet-style quantization, moving beyond the common heuristic that "first/last layers matter"
  - Exceeds FP32 on harder tasks with statistical testing (Table 4): Reporting p-values (0.028, 0.004) and Cohen's d is above-average rigor for a workshop paper; the effect sizes are meaningful
  - Architecture-dependent analysis (Table 5): The stark contrast between ResNet (80-130% recovery) and MobileNetV2 (17-73%) provides immediate actionable guidance for practitioners choosing architectures for quantized deployment
  - Crisp takeaway: "Invest in distillation, not augmentation" is memorable and poster-friendly—exactly what workshop attendees will remember
  - Honest limitations section (Section 7): Acknowledging ImageNet gap (26%), Bonferroni correction issues, and lack of latency measurements shows scientific maturity

  ---
  Weaknesses (bulleted, specific, ordered by severity)

  - [MINOR] "Augmentation paradox" framing is imprecise: The paradox isn't that augmentation "fails"—it still improves ternary accuracy (Section 4 notes BitNet gains +0.94 pp on CIFAR-100). The actual finding is differential benefit (FP32 gains 1.5-3.7× more). The title and abstract overstate this as "augmentation fails" when "augmentation helps less" is more accurate. This won't block acceptance but weakens credibility.
  - [MINOR] Missing variance on key ablation results: Tables 2 and 3 report single-run results for several configurations (e.g., "keep_conv1: 87.40" without ±). Given the final recipe claims hinge on additive effects, variance for the ablation components would strengthen the argument.
  - [MINOR] Explanation for why augmentation widens gap is speculative: Section 4 claims "ternary networks lack the capacity to exploit additional training signal" but provides no evidence. An information-theoretic argument (similar to the gap-scaling discussion in Section 7) would be more satisfying. Currently this is a phenomenon without a mechanism.
  - [MINOR] KD hyperparameter narrative is confusing: Table 3 shows T=5 beats default on CIFAR-10 and T=6/α=0.5 beats default on CIFAR-100, yet the paper uses T=4/α=0.9. The explanation ("negative interactions when jointly optimized") is vague. Either simplify to "we use standard KD defaults" or provide the joint optimization data.
  - [MINOR] ConvNeXt results are puzzling without explanation: 32% recovery on CIFAR-10 but 130% on Tiny-IN is a 4× difference for the same architecture. This deserves at least a hypothesis—is it resolution (32×32 vs 64×64)? Dataset size? Leaving it unexplained is a missed opportunity.

  ---
  Questions for Authors / Discussion Points

  1. Why does augmentation widen the gap more on CIFAR-100 than CIFAR-10? The +59% gap increase (ResNet-18/CIFAR-100) vs +12% (ResNet-18/CIFAR-10) suggests task complexity interacts with the phenomenon. Is there a principled explanation?
  2. Would INT8 conv1 provide similar benefits to FP32 conv1? This would maintain integer-only arithmetic on hardware while potentially capturing most of the benefit.
  3. Have you measured actual inference latency? The 64× theoretical speedup assumes optimized ternary kernels. With FP32 conv1, the theoretical benefit is reduced—what's the realistic speedup on edge hardware?
  4. Why does the recipe work so poorly for MobileNetV2 on CIFAR-100 (17% recovery)? You attribute this to depthwise separable convolutions, but MobileNetV2 achieves 73% on Tiny-IN. Is there something specific about CIFAR-100 + MobileNetV2?
  5. What's the training cost overhead of the recipe? KD requires a trained teacher. Is the total compute budget (teacher + student) justified for practitioners?

  ---
  Scores

  ┌─────────────────────────────┬────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │          Criterion          │ Score  │                                                         Justification                                                          │
  ├─────────────────────────────┼────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Interest / Discussion Value │ 8/10   │ The augmentation finding is counterintuitive; architecture comparison is immediately useful. Would generate poster discussion. │
  ├─────────────────────────────┼────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Technical Soundness         │ 7/10   │ Core claims supported but "paradox" framing is overstated; some missing variance reporting.                                    │
  ├─────────────────────────────┼────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Clarity / Writing           │ 8/10   │ Well-structured, clear figures and tables; minor confusion around KD hyperparameters.                                          │
  ├─────────────────────────────┼────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Practical Usefulness        │ 7/10   │ Strong guidance for CIFAR-scale; limited without ImageNet validation or latency numbers.                                       │
  ├─────────────────────────────┼────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Overall                     │ 7.5/10 │ Solid workshop contribution with interesting findings; appropriate scope for gathering feedback.                               │
  └─────────────────────────────┴────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Decision

  Accept

  This paper addresses a practical question (how to train ternary CNNs), provides a counterintuitive finding (augmentation helps FP32 disproportionately), and offers actionable guidance (use KD, keep conv1 FP32, avoid depthwise separable architectures). The limitations are acknowledged honestly, and the scope is appropriate for a workshop where the goal is discussion and feedback rather than completeness. The core message
   is crisp and would make an effective poster presentation.

  ---
  Suggestions for Strengthening (for future main conference submission)

  1. ImageNet-scale validation: The 26% gap vs TTQ's 2.7% is the critical limitation. Adapting techniques from ReActNet/BNext (progressive quantization, learned scaling, longer training) while preserving BitNet's fixed {-1,0,+1} simplicity would be the highest-impact addition.
  2. Mechanistic explanation for augmentation asymmetry: Currently the paper shows that augmentation widens the gap but not why. An analysis of gradient flow, feature diversity, or capacity utilization under different augmentation regimes would make the contribution more fundamental.
  3. Real latency measurements: Theoretical speedup (64×) is compelling, but actual measurements with ternary kernels (BitBLAS, TVM) on edge hardware (Raspberry Pi, Jetson Nano) would dramatically strengthen the practical impact claim—especially with FP32 conv1 in the mix.
