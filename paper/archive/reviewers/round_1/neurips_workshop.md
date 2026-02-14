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


  Summary (2-3 sentences)

  This paper investigates BitNet b1.58 (ternary quantization) applied to CNNs, finding that stronger data augmentation widens rather than closes the accuracy gap with FP32 models. The authors propose a recipe combining FP32 first convolution layer + knowledge distillation that recovers 88% of the gap on CIFAR-10 and exceeds FP32 on harder tasks, while identifying that standard convolution architectures are much better
  suited for this approach than depthwise separable designs.

  ---
  Strengths

  - Counterintuitive finding with systematic evidence (Section 4, Tables 1-2): The "augmentation paradox" is well-documented across 4 model-dataset configurations × 4 augmentation strategies. The 4-11× differential benefit for FP32 vs. ternary models is surprising and the case where augmentation actually hurts the ternary model (ResNet-50/CIFAR-10, -0.89pp) is memorable.
  - Clean ablation methodology (Section 5.1, Table 3): The layer-wise ablation is well-executed. The finding that conv1 accounts for 54-74% of the gap with only 0.08% of parameters is actionable and provides precise quantification of established practice.
  - Practical, memorable takeaway: "Invest in distillation, not augmentation" is crisp messaging that workshop attendees will remember. The 3-step deployment recipe (Section 5.5) is directly actionable.
  - Honest scope and limitations (Section 7): The paper is refreshingly transparent about its limitations—acknowledging the 26% ImageNet gap, limited statistical power (n=3), and that the recipe is not plug-and-play. This builds trust.
  - Architecture-dependent analysis (Section 6, Table 8): The finding that recovery varies from 17% (MobileNetV2/CIFAR-100) to 130% (ConvNeXt/Tiny-IN) is practically useful. The explanation involving depthwise convolution vulnerability is plausible.
  - Reproducibility commitment (Appendix A): The code-to-paper pipeline with determinism guarantees and public repository is commendable for workshop-level work.

  ---
  Weaknesses

  [MAJOR]

  - The "exceeds FP32" claims rest on weak statistical ground: The CIFAR-100 result is significant at p=0.028 with n=3 seeds, but the Tiny ImageNet result (the strongest claim: +1.3%) is from a single seed (Section 5.4, Table 6). The paper acknowledges this but still uses these results prominently in the abstract and contributions. For a workshop paper, the claims should be tempered—"preliminary evidence" rather than
  bolded headline claims.
  - Novelty is overstated: Keeping first/last layers in FP32 is established practice since XNOR-Net (2016) and DoReFa-Net (2016). The paper claims "first precise quantification" (line 132, 164, 348) but the core contribution is measuring something already known, not discovering it. A workshop paper can be an incremental quantification, but the framing should be more humble. The augmentation paradox is the more novel
  contribution.

  [MINOR]

  - Comparison with TTQ/TWN is confusing (Table 10): The paper compares different architectures (ResNet-18 vs. ResNet-56) making the 0.40% vs. -0.36% gap comparison not meaningful. This should either be removed or clearly flagged as illustrative-only.
  - Missing ablation on KD hyperparameters in main text: Section 5.2 mentions that lower α (0.5-0.7) works better in isolation but recommends default (T=4, α=0.9) because combining optimized values "yields no improvement." This is interesting but underexplored—it appears in Discussion (Section 7) rather than as a proper ablation table.
  - Architecture-specific training adaptations (Table 7): The fact that MobileNetV2/EfficientNet need lr=0.01 while ResNet uses lr=0.1 raises the question: could the poor recovery for depthwise architectures partially be a tuning issue? The paper acknowledges training instability but doesn't systematically verify that lr=0.01 is optimal vs. just "stable."
  - The gap scaling claim (Figure 3, Section 7): The paper claims the gap scales "roughly logarithmically" with number of classes based on 3 data points (10, 100, 200 classes). This is speculative and the information-theoretic explanation in Appendix B, while interesting, is post-hoc rationalization.

  ---
  Questions for Authors / Discussion Points

  1. Why does combining optimized T and α degrade performance? You observe that individual improvements (higher T, lower α) don't combine well. Is this a ternary-specific interaction or would this hold for quantized networks generally? This seems like rich territory for workshop discussion.
  2. What happens if you keep more layers in FP32? You show conv1 recovers 58%, but what's the curve? Is there a point of diminishing returns, or does recovery continue linearly with FP32 parameter count?
  3. For depthwise architectures, is there an alternative "first layer equivalent"? MobileNetV2's stem is different from ResNet's conv1. Did you try keeping the entire first inverted residual block in FP32? The low recovery (17-73%) might be fixable with architecture-appropriate intervention.
  4. Could the augmentation paradox be mitigated by training longer? You train 200 epochs uniformly. If ternary networks need more epochs to fit augmented data, could a 400-epoch ternary training recover the differential?
  5. What's driving the "exceeds FP32" phenomenon on harder tasks? Is this purely KD regularization, or is there something specific about ternary networks that makes KD especially effective? This could inform whether the result generalizes to other compression methods.

  ---
  Scores

  ┌─────────────────────────────┬───────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │          Criterion          │ Score │                                                                Justification                                                                │
  ├─────────────────────────────┼───────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Interest / Discussion Value │ 8/10  │ The augmentation paradox is genuinely counterintuitive and will generate discussion; architecture-dependent findings are practically useful │
  ├─────────────────────────────┼───────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Technical Soundness         │ 6/10  │ Core claims are supported but "exceeds FP32" claims lean too heavily on single-seed results; comparison table is misleading                 │
  ├─────────────────────────────┼───────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Clarity / Writing           │ 8/10  │ Well-organized, clear takeaways, honest about limitations; figures and tables are informative                                               │
  ├─────────────────────────────┼───────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Practical Usefulness        │ 7/10  │ Recipe is actionable for CIFAR-scale; unclear if it transfers to ImageNet-scale without significant additional work                         │
  ├─────────────────────────────┼───────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Overall                     │ 7/10  │ Solid workshop paper with interesting findings; would benefit from tempering the strongest claims                                           │
  └─────────────────────────────┴───────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Decision

  Borderline → Accept

  The augmentation paradox finding is interesting and counterintuitive, the layer-wise quantification is useful, and the practical recipe is actionable. The paper would generate good discussion at a workshop. However, the "exceeds FP32" claims need to be tempered (especially Tiny ImageNet which is single-seed), and the novelty framing around conv1 should be more humble given established practice. If the authors can revise
   the abstract/contributions to be more measured about statistical confidence, this is a clear accept. As written, it's borderline due to overclaiming.

  ---
  Suggestions for Strengthening (for main conference submission)

  1. Run ImageNet experiments with modern training recipes: The 26% gap is a showstopper for a main venue. Adopt training techniques from ReActNet/BNext (256+ epochs, progressive quantization, learned scaling) and demonstrate the recipe at ImageNet scale. This is the single most impactful addition.
  2. Increase statistical power on key claims: Run 5+ seeds on CIFAR-100 and Tiny ImageNet to strengthen the "exceeds FP32" claims. Currently, the most exciting results have the weakest statistical support.
  3. Add feature-level distillation comparison: You mention this as future work (Section 7). Adding FitNets-style intermediate feature matching would differentiate from standard KD and potentially improve results further, especially for the depthwise architectures where logit-only KD underperforms.