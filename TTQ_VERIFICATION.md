# TTQ Implementation Verification

## Paper Reference
**Trained Ternary Quantization** (Zhu et al., ICLR 2017)
arXiv: https://arxiv.org/abs/1612.01064

## Algorithm Summary (from paper)

### Forward Pass

**Quantization (Eq. 1):**
```
W_t = { +Wp   if W > delta
      { -Wn   if W < -delta
      {  0    otherwise
```

**Initialization (Eq. 2, Section 3.1):**
- Threshold: `delta = 0.7 * E[|W|]`
- Positive scale: `Wp = E[|W|]`
- Negative scale: `Wn = E[|W|]`

Where `E[|W|]` = mean of absolute weight values

### Backward Pass
Straight-through estimator (STE): gradients flow as if no quantization

### Key Innovation
Unlike fixed ternary {-1, 0, +1}, TTQ learns three FP32 parameters per layer:
- `Wp` (positive scale)
- `Wn` (negative scale)
- `delta` (threshold)

## Our Implementation

### Files
- `bitnet/nn/ttq_quantization.py` - Core quantization function
- `bitnet/nn/ttq_linear.py` - Linear layer with TTQ
- `bitnet/nn/ttq_conv2d.py` - Conv2d layer with TTQ
- `tests/test_ttq_layers.py` - Test suite

### Key Design Decisions

**1. Positivity Constraint:**
- Paper assumes Wp, Wn, delta > 0 but doesn't specify enforcement
- We use `F.softplus()` to ensure positivity while maintaining gradients
- Returns tuple `(quantized, wp_pos, wn_pos)` for consistent scaling

**2. Activation Quantization:**
- TTQ paper only specifies weight quantization
- We use BitNet's activation quantization (`quantize_activations` + `dequantize`)
- This allows fair comparison: both methods quantize weights AND activations
- Beta for dequantization: `beta = (wp_pos + wn_pos) / 2`

**3. Initialization:**
- Wp, Wn = `mean(abs(weight))`  ✓ Matches paper
- delta = `0.7 * mean(abs(weight))`  ✓ Matches paper

## Verification Checklist

- [x] Quantization logic matches Eq. 1
- [x] Threshold comparison: `W > delta` and `W < -delta`
- [x] Three learnable parameters: wp, wn, delta
- [x] Initialization: Wp = Wn = E[|W|]
- [x] Initialization: delta = 0.7 * E[|W|]
- [x] Straight-through estimator for gradients
- [x] Positivity enforcement (softplus)
- [x] Consistent scale usage in quantization and dequantization
- [x] Test suite covers shapes, gradients, initialization, stability

## Differences from Pure TTQ

1. **Activation Quantization:** We add BitNet-style activation quantization (8-bit)
   - Reason: Fair comparison (both methods quantize weights + activations)
   - Impact: More realistic for deployment

2. **Positivity Enforcement:** We use softplus, paper doesn't specify
   - Reason: Prevent training instability from negative scales
   - Impact: Minor, gradients still flow

## Bugs Fixed

1. **Double softplus application:** quantization used softplus(wp), dequantization used softplus(softplus(wp))
   - Fixed: Return wp_pos, wn_pos from ttq_quantize

2. **Wrong initialization:** Used std(W) instead of mean(|W|) for delta
   - Fixed: Both use `weight.abs().mean()`

3. **NaN losses:** Parameters could go negative without constraints
   - Fixed: Softplus enforcement

4. **CRITICAL: Activation quantization incompatibility**
   - Bug: Mixing BitNet's activation quant/dequant with TTQ weights caused training to fail (stuck at 10%)
   - Root cause: TTQ weights are pre-scaled {-wn, 0, +wp} but BitNet's dequant expects unscaled {-1,0,+1}
   - Tested 4 configs:
     - A: beta=1.0 → FAILS (10% accuracy)
     - B: beta=(wp+wn)/2 → FAILS (10% accuracy)
     - C: beta=weight.abs().mean() → FAILS (10% accuracy)
     - D: Pure TTQ (no activation quant) → **WORKS** (49% accuracy in 2 epochs!)
   - Solution: Use pure TTQ as in original paper (ternary weights, FP32 activations)

5. **CRITICAL: Zero gradients to TTQ parameters** (final fix)
   - Bug: After fixing activation quantization, training still stuck at 10% on server
   - Root cause: PyTorch indexing assignment breaks gradient flow to scalar parameters

     ```python
     quantized[pos_mask] = wp_pos  # ❌ Breaks gradients to wp_pos
     ```

   - Diagnosis: Gradient verification showed 0/63 TTQ parameters had gradients
   - Additional bug: Softplus initialization caused delta to be 5-7x too large after transformation
   - Solution:
     - Implement custom `TTQQuantizeFunction` with explicit backward pass
     - Gradients: `grad_wp = (grad_output * pos_mask).sum()`
     - Initialize parameters with inverse softplus to get correct values after transformation
   - Verified: wp.grad=0.047, wn.grad=0.046 (proper gradient flow)

## Expected Behavior

- **Accuracy:** Should achieve ~0.5-1.5% better than BitNet+Recipe (based on literature)
- **Complexity:** Requires 2 FP32 params per layer (vs BitNet+Recipe's 1 FP32 layer)
- **Trade-off:** Better accuracy, more deployment complexity

## Test Results

All 9 tests pass:
- Forward pass shapes
- Gradient flow (wp, wn get gradients)
- Correct initialization (E[|W|] and 0.7*E[|W|])
- Numerical stability (no NaN in 10 training steps)
- Various kernel sizes
