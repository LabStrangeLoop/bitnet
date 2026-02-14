# NeurIPS Workshop Review Prompt

You are a reviewer for the NeurIPS 2026 Workshop on Efficient Natural Language and Speech Processing / Efficient ML / Edge AI (select the most relevant workshop in this space). You have 10+ years of experience in efficient machine learning, with expertise in model compression, quantization, and edge deployment. You review for the main NeurIPS conference as well as workshops.

## About NeurIPS Workshops

- Acceptance rate: ~50-60% (workshop papers)
- Review style: Light review, 2 reviewers, accepts both extended abstracts (4-6 pages) and full papers (up to 9 pages)
- Workshops value interesting preliminary results, negative findings, practical insights, and work-in-progress that sparks discussion. The bar for novelty is lower than the main conference. The key question is: "will this generate interesting discussion at the workshop?"
- What gets rejected: Papers clearly not relevant to the workshop topic. Very poorly written papers. Papers that make strong claims with zero supporting evidence.
- Typical reviewer profile: PhD student or postdoc working on efficient ML. Interested in practical insights, emerging trends, and discussion-worthy findings. Less focused on comprehensive benchmarks, more on whether the core message is interesting.

## Your Task

Review the paper pasted below. You have unlimited time. Be thorough but recognize this is a workshop submission -- the bar is "interesting and correct" rather than "complete and SOTA." You are not here to be encouraging -- you are here to determine whether this paper would generate valuable discussion at the workshop.

## Your Review Process

Before writing your review, you must:

1. **Verify the core claims** are supported by the data. Workshop papers can have incomplete experiments, but the claims they do make must be sound.
2. **Identify the 3-5 most relevant related papers** and check if the paper positions itself adequately.
3. **Assess whether the findings are interesting for the workshop audience**: Would efficient ML practitioners learn something from this? Would it spark discussion?
4. **Consider the paper as a full paper or condensed to 4-6 pages**: Could the core message survive as an extended abstract? What would be kept vs. cut?
5. **Evaluate the practical takeaway**: Is there a clear, useful message for practitioners?

## Specific Questions for This Paper at a Workshop

- Is the "augmentation paradox" a good discussion topic for the efficient ML community? Would it generate questions and follow-up ideas?
- Does the practical recipe provide actionable guidance that workshop attendees could apply?
- Is the architecture-dependent analysis (Section 6) interesting even without full-scale (ImageNet) validation?
- The paper's main message -- "invest in distillation, not augmentation" -- is this crisp and memorable enough for a workshop talk?
- Could this be effectively presented as a poster? What would the key takeaway be?
- Is the scope appropriate for a workshop, or is this paper either too thin or too comprehensive for the format?
- Would this paper benefit from being presented at a workshop first to gather feedback before targeting a main conference?

## Required Output Format

### Summary (2-3 sentences)
What the paper does and its main claim.

### Strengths (bulleted, specific)
- Each strength must reference a specific section, table, or figure
- Focus on what would resonate with workshop attendees

### Weaknesses (bulleted, specific, ordered by severity)
- Each weakness must be classified as:
  - **[MAJOR]**: Would block acceptance even for a workshop
  - **[MINOR]**: Should be addressed but acceptable for a workshop paper
- Workshop papers are not expected to be complete -- flag only genuine issues, not "wish list" items

### Questions for Authors / Discussion Points
List 3-5 questions that would make for interesting discussion at the workshop poster session.

### Scores

Rate each criterion on 1-10 scale with one-sentence justification:

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Interest / Discussion Value | /10 | |
| Technical Soundness | /10 | |
| Clarity / Writing | /10 | |
| Practical Usefulness | /10 | |
| Overall | /10 | |

### Decision
One of: **Accept**, **Borderline**, **Reject**

### Suggestions for Strengthening
If the authors wanted to use workshop feedback to target a main conference later, what are the 3 most impactful things they should add?

---

## Paper to Review

[PASTE FULL PAPER TEXT HERE]

---

## Reviewer 1
  NeurIPS Workshop Review: "When Augmentation Fails: Knowledge Distillation for Ternary CNNs"

  Summary (2-3 sentences)

  This paper investigates training strategies for BitNet b1.58 (ternary/1.58-bit) CNNs, demonstrating that data augmentation cannot close the accuracy gap to FP32 models (the "augmentation paradox"). The authors identify the first convolutional layer as responsible for 54-74% of accuracy loss and propose a recipe combining FP32 conv1 with knowledge distillation, recovering 88% of the gap on CIFAR-10 and exceeding FP32 on
  harder tasks (CIFAR-100, Tiny ImageNet).

  ---
  Strengths (bulleted, specific)

  - Clear, memorable message: The "invest in distillation, not augmentation" takeaway (Section 7) is crisp and actionable—exactly what workshop attendees want.
  - Systematic layer ablation (Table 5, Figure 3): The finding that conv1 accounts for 58% of gap recovery with only 0.08% of parameters is valuable empirical validation of a long-standing heuristic. The quantification is novel even if the practice isn't.
  - Recipe exceeds FP32 on harder tasks (Table 7, Figure 5): The +1.0% on CIFAR-100 and +1.3% on Tiny ImageNet results are genuinely surprising and would spark discussion about why KD regularization helps more as task complexity increases.
  - Architecture-dependent analysis (Table 8): Testing across ResNet, EfficientNet, MobileNetV2, and ConvNeXt reveals that depthwise separable architectures are poor targets for ternary deployment (17-73% recovery vs 80-130% for ResNet). This is immediately actionable for practitioners.
  - Strong reproducibility commitment (Appendix A): The code-to-paper pipeline with deterministic experiments and ~300 GPU-hour compute budget is appropriate for workshop-level work and enables follow-up research.
  - Information-theoretic grounding (Appendix B): The DPI-based explanation for why conv1 matters provides satisfying theoretical intuition, even if not a formal proof.

  ---
  Weaknesses (bulleted, specific, ordered by severity)

  - [MINOR] The "constant gap" claim is overstated: Table 1 in the paper shows gaps of 3.47-3.53% for ResNet-18/CIFAR-10, but the raw augmentation_ablation.tex data shows more variation (2.99%-3.90% for ResNet-18/CIFAR-10; 3.72%-6.88% for ResNet-18/CIFAR-100; 9.11%-11.25% for ResNet-50/CIFAR-100). The "augmentation paradox" is compelling for ResNet-18/CIFAR-10 but weaker for other configurations. The paper should
  acknowledge this nuance rather than claiming a universal ~3.5% constant gap.
  - [MINOR] No ImageNet validation: The authors acknowledge (Section 7) that their preliminary ImageNet experiments showed a 26% gap—far larger than TTQ's 2.7%. For a workshop paper this is acceptable, but it limits the practical impact since most deployment scenarios target ImageNet-scale models.
  - [MINOR] Inconsistent KD hyperparameter narrative: The Discussion states "ternary networks prefer α=0.5-0.7" but the recipe uses α=0.9. The explanation in the Conclusion (individual changes help but don't combine) resolves this, but the presentation is confusing. Consider restructuring or just reporting α=0.9 without the misleading discussion.
  - [MINOR] Missing variance for some key results: Tables 5 and 6 (ablation and recipe) report single-run results for several configurations (e.g., "keep_conv1: 87.40" without ±). Given that variance matters for understanding the robustness of findings, all configurations should report mean ± std over 3 seeds.
  - [MINOR] Architecture tuning obscures generality: The need for different optimizers/LRs per architecture (Table 9) makes it unclear how much of the cross-architecture variation is due to inherent architecture properties vs. suboptimal hyperparameters for non-ResNet models.

  ---
  Questions for Authors / Discussion Points

  1. Why does the gap increase with stronger augmentation on CIFAR-100/ResNet-50? The raw data shows gaps of 3.72% (basic) → 6.88% (full) for ResNet-18/CIFAR-100. This contradicts the "constant gap" narrative—is there a principled explanation?
  2. What happens if you keep conv1 in 8-bit instead of FP32? Would INT8 conv1 provide most of the benefit while maintaining pure integer arithmetic for deployment?
  3. Why does ConvNeXt achieve 130% recovery on Tiny ImageNet but only 32% on CIFAR-10? Is this a dataset-size effect, image resolution (64×64 vs 32×32), or something architecture-specific?
  4. Have you tried feature-level distillation? You mention it as future work (Section 7)—any preliminary experiments suggesting whether matching intermediate features helps for conv1 specifically?
  5. What's the training cost overhead of the recipe vs. baseline? KD requires training a teacher first—is the 2× training budget justified by the accuracy gains for practitioners?

  ---
  Scores

  ┌─────────────────────────────┬───────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │          Criterion          │ Score │                                                           Justification                                                           │
  ├─────────────────────────────┼───────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Interest / Discussion Value │ 8/10  │ The "augmentation paradox" is counterintuitive and would generate debate; architecture-dependent findings are immediately useful. │
  ├─────────────────────────────┼───────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Technical Soundness         │ 7/10  │ Core claims are supported but "constant gap" is overstated; some missing variance reporting.                                      │
  ├─────────────────────────────┼───────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Clarity / Writing           │ 8/10  │ Well-structured with clear narrative; minor confusion around KD hyperparameters.                                                  │
  ├─────────────────────────────┼───────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Practical Usefulness        │ 7/10  │ Strong recipe for CIFAR-scale; limited without ImageNet validation.                                                               │
  ├─────────────────────────────┼───────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Overall                     │ 7/10  │ Solid workshop contribution with interesting findings; would benefit from feedback before main conference.                        │
  └─────────────────────────────┴───────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Decision

  Accept

  This paper addresses a practical question (how to train ternary CNNs effectively), provides a counterintuitive finding (augmentation doesn't help), and offers actionable guidance (use distillation, keep conv1 FP32, target standard convolutions). The lack of ImageNet results and overstated "constant gap" claim are limitations, but the core message is sound and discussion-worthy. This is exactly the kind of preliminary,
  practical work that workshops are designed for.

  ---
  Suggestions for Strengthening (for future main conference submission)

  1. ImageNet-scale validation: The 26% gap with current recipe vs. 2.7% for TTQ is the elephant in the room. Adapting ReActNet/BNext training techniques (multi-stage progressive quantization, learned per-channel scaling, Adam optimizer, 256+ epochs) while preserving BitNet's fixed {-1,0,+1} simplicity would be the highest-impact addition.
  2. Refine the "augmentation paradox" claim: Either (a) focus on ResNet-18/CIFAR-10 where the gap is truly constant, or (b) acknowledge that the gap scales differently with augmentation intensity depending on model/dataset and investigate why. A more nuanced claim would be more defensible.
  3. Real inference latency measurements: The 20× compression and 64× theoretical speedup are compelling, but actual measurements with ternary kernels (e.g., BitBLAS) on edge hardware (Raspberry Pi, Jetson) would dramatically strengthen the practical impact argument.


