# Review Score Tracker

Tracks simulated review scores across paper revisions.

## Round 0 — Baseline (Feb 13, 2026)

Paper version: Pre-fix (original draft with augmentation paradox data issue)

| Venue | Decision | Score | Fatal Issues |
|-------|----------|-------|--------------|
| NeurIPS Workshop | **Accept** | **7/10** | None |
| WACV | Weak Reject | 5/10 | Missing inference benchmarks |
| BMVC | Weak Reject | ~5/10 | Augmentation paradox contradicted by data |
| TMLR | Reject | 4/10 | Central claim contradicted, data integrity |
| NeurIPS | Weak Reject | 4/10 | Limited novelty, insufficient scope |
| ICLR | Weak Reject | 4/10 | Augmentation paradox data discrepancy |
| CVPR | Reject | 4/10 | No ImageNet, no baselines |

**Mean score**: 4.7/10
**Accept rate**: 1/7 (14%)

### Key issues flagged (all venues):
1. **FATAL**: Augmentation paradox claim contradicted by own data (cherry-picked Table 1)
2. **FATAL** (top venues): No ImageNet results, no TTQ/LSQ comparison
3. **MAJOR**: Numerical inconsistencies (inline tables vs generated tables)
4. **MAJOR**: "Exceeds FP32" claims lack statistical tests
5. **MAJOR**: KD alpha narrative confusing
6. **MINOR**: Missing references (LSQ, PACT, HAWQ-V2, IR-Net, SQAKD)

---

## Round 1 — Post P0-P5 fixes (Feb 13, 2026)

Paper version: After fixing augmentation paradox, numerical inconsistencies, adding TTQ comparison, statistical tests, alpha clarification, and missing references.

### Fixes applied:
- **P0**: Rewrote Section 4 with full data (4 configs x 4 augmentation levels, gaps 2.99-11.25%)
- **P0**: Reframed narrative: "augmentation widens the gap" (not "constant ~3.5%")
- **P0**: Updated abstract, intro, contributions, conclusion
- **P1**: Fixed FP32 88.89->88.88, BitNet 85.40->85.38 in ablation table
- **P1**: Fixed all inline numbers to match generated tables
- **P3**: Added TTQ/TWN comparison table with published numbers
- **P4**: Added Welch's t-test for CIFAR-100 "exceeds FP32" (p=0.028)
- **P4**: Added statistical caveats for Tiny-IN (single seed)
- **P4**: Clarified KD alpha narrative (defaults recommended, ablation shows negative interaction)
- **P5**: Added 6 missing references (LSQ, PACT, HAWQ-V2, EWGS, IR-Net, SQAKD)

| Venue | Decision | Score | Notes |
|-------|----------|-------|-------|
| NeurIPS Workshop | Borderline→Accept | **7/10** | Temper statistical claims, humble conv1 framing |
| TMLR | **Revise & Resubmit** | **7/10** | Complete layer ablation table, fix abstract numbers, multi-seed Tiny-IN |
| WACV | Weak Reject | 5.5/10 | Needs inference benchmarks, recipe for depthwise architectures |
| BMVC | Weak Reject | 5/10 | Needs QAT/LSQ baseline, soften "exceeds FP32", per-seed results |
| ICLR | Weak Reject | 5/10 | Fix Table 6/8 inconsistency, modern QAT comparison, ImageNet |
| NeurIPS | Reject | 4/10 | Needs ImageNet results, novel algorithmic element, formal theory |
| CVPR | Reject | 4/10 | Needs ImageNet, hardware benchmarks, LSQ/ReActNet comparison |

**Mean score**: 5.4/10
**Accept rate**: 2/7 (29%) — 1 accept + 1 R&R

---

## Score History

| Round | Mean | Accepts | Key Change |
|-------|------|---------|------------|
| 0 | 4.7 | 1/7 | Baseline (pre-fix) |
| 1 | 5.4 | 2/7 | P0-P5 fixes (+0.7 mean, TMLR Reject→R&R) |
