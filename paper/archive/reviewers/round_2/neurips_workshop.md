 Reviewer 2: NeurIPS Workshop Review

  Summary (2-3 sentences)

  This paper systematically studies training strategies for BitNet b1.58 (ternary) CNNs, finding that data augmentation paradoxically widens the accuracy gap versus FP32 baselines. The authors identify the first convolutional layer as a critical bottleneck (54-74% of degradation) and propose a practical recipe combining FP32 conv1 with knowledge distillation that recovers 89% of the gap on CIFAR-10 and surpasses FP32 on
  CIFAR-100/Tiny-ImageNet.

  ---
  Strengths (bulleted, specific)

  - Counterintuitive, discussion-worthy finding (Section 4, Table 1): The observation that stronger augmentation hurts ternary networks relative to FP32 (up to 3.7× differential benefit) challenges conventional wisdom and would generate substantial workshop discussion.
  - Quantitative layer attribution (Table 2): The 58% gap recovery from keeping only conv1 (0.08% parameters) is a crisp, memorable result. Prior work noted first-layer sensitivity heuristically; this paper quantifies it precisely.
  - Statistically rigorous claims for key results (Section 5.3): The authors report p-values, effect sizes (Cohen's d=5.5), and acknowledge Bonferroni correction limitations. This level of statistical transparency is rare and commendable.
  - Actionable architecture guidance (Table 4, Section 6): The 80-130% recovery for standard convolutions vs. 17-73% for depthwise separable is immediately useful for practitioners choosing architectures for edge deployment.
  - Clean scaling narrative (Section 7 Discussion): The information-theoretic explanation—ternary bits (~1.58 bits/weight) vs. task entropy scaling with classes—provides intuitive grounding for why KD helps more on harder tasks.
  - Honest limitations section: Authors explicitly acknowledge ImageNet gap (26%), statistical power limitations, and per-architecture tuning requirements. This transparency strengthens credibility.

  ---
  Weaknesses (bulleted, specific, ordered by severity)

  - [MINOR] Augmentation gap variation undermines "paradox" framing: Table 1 shows gaps ranging from +12% to +59% increase—a 5× variation. The phenomenon is real but inconsistent. Calling it a "paradox" implies universality that the data doesn't fully support. Consider "augmentation asymmetry" or similar softer framing.
  - [MINOR] Missing ablation on KD hyperparameters: The paper uses fixed T=4, α=0.9 without justification. Given that KD is half the recipe, ablating temperature and mixing coefficient would strengthen the contribution. Even a small table showing T∈{2,4,8} and α∈{0.5,0.7,0.9} would help.
  - [MINOR] Unclear if conv1+KD gains are additive or synergistic: Table 3 shows the combined result but not the individual contributions in a way that reveals interaction effects. Is recipe = conv1_gain + KD_gain, or is there positive/negative interaction? This matters for understanding the mechanism.
  - [MINOR] Figure 1 could show variance: The progression plot (Figure 1) appears to show point estimates without error bars. Adding confidence intervals would strengthen the visual narrative and match the statistical rigor in the text.
  - [MINOR] ConvNeXt results missing from Table 4: The abstract mentions testing "four CNN families" but Table 4 only shows ResNet, EfficientNet, and MobileNetV2. Where are ConvNeXt results? (Reviewer 1 mentions 130% recovery on Tiny-IN—is this from supplementary material?)

  ---
  Questions for Authors / Discussion Points

  1. Is the augmentation asymmetry about capacity or gradient flow? You hypothesize capacity limits, but ternary networks also have unusual gradient dynamics (STE). Have you checked whether augmentation changes gradient variance/magnitude differently for ternary vs. FP32?
  2. What happens with learned scaling factors (TTQ-style)? Your recipe keeps conv1 in FP32. TTQ learns per-layer scales. Is there a middle ground—ternary conv1 with learned asymmetric thresholds—that preserves integer-only inference?
  3. Why 89% recovery on CIFAR-10 but >100% on CIFAR-100/Tiny-IN? The KD benefit increasing with task complexity makes sense, but why doesn't CIFAR-10 also exceed FP32? Is there a ceiling effect, or does the teacher not help when the task is "easy"?
  4. How sensitive is the recipe to teacher quality? You use the FP32 baseline as teacher. What if the teacher is also quantized (e.g., INT8)? Or what if the teacher is a larger model (ResNet-50 teaching ResNet-18)?
  5. Practical deployment question: Keeping conv1 in FP32 breaks the "pure integer arithmetic" promise. For edge deployment, is INT8 conv1 sufficient? This would preserve integer-only inference while potentially capturing most of the benefit.

  ---
  Scores

  ┌─────────────────────────────┬────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │          Criterion          │ Score  │                                                    Justification                                                    │
  ├─────────────────────────────┼────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Interest / Discussion Value │ 8/10   │ Augmentation asymmetry is genuinely surprising; architecture findings are immediately actionable for practitioners. │
  ├─────────────────────────────┼────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Technical Soundness         │ 7/10   │ Core claims supported; statistical treatment is good but "paradox" framing oversells consistency.                   │
  ├─────────────────────────────┼────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Clarity / Writing           │ 8/10   │ Well-structured, concise, clear narrative flow. Minor issues with missing ConvNeXt data in main table.              │
  ├─────────────────────────────┼────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Practical Usefulness        │ 7/10   │ Strong for CIFAR-scale; limited until ImageNet validated. Architecture guidance is valuable.                        │
  ├─────────────────────────────┼────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Overall                     │ 7.5/10 │ Solid workshop contribution with clear takeaways; would benefit from workshop feedback before main venue.           │
  └─────────────────────────────┴────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Decision

  Accept

  The paper addresses a practical question relevant to the efficient ML community, provides a counterintuitive finding (augmentation hurts relative performance), and offers a clear, actionable recipe. The weaknesses are minor and appropriate for a workshop paper—incomplete experiments are expected at this venue. The core message ("invest in distillation, not augmentation") is memorable and would make for a strong poster
  presentation. This is exactly the kind of preliminary work that benefits from workshop discussion before targeting a main conference.

  ---
  Suggestions for Strengthening (for future main conference submission)

  1. ImageNet validation with modern training: The 26% gap vs. TTQ's 2.7% is the critical blocker. Adopting progressive quantization schedules, longer training (300+ epochs), and potentially learned activation quantization (as in ReActNet) while maintaining fixed {-1,0,+1} weights would be the highest-impact addition.
  2. Mechanistic understanding of augmentation asymmetry: Move beyond the capacity hypothesis. Analyze gradient flow, feature map statistics, or loss landscape geometry to explain why FP32 benefits more. This would elevate the paper from empirical observation to actionable insight.
  3. Deployment validation: Measure actual latency/energy on edge hardware (Raspberry Pi, Jetson Nano) using ternary kernels (BitBLAS, CUTLASS with ternary extensions). The 64× theoretical speedup is compelling but unvalidated. Even a single hardware platform would significantly strengthen practical claims.
