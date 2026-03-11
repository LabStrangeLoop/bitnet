import torch
import torch.nn.functional as f
from torch import Tensor
from torch.autograd import Function


class TTQQuantizeFunction(Function):
    """Custom autograd function for TTQ quantization with proper gradient flow."""

    @staticmethod
    def forward(ctx, weight: Tensor, wp_pos: Tensor, wn_pos: Tensor, delta_pos: Tensor) -> Tensor:
        """Forward pass: quantize weights to {-wn, 0, +wp}."""
        # Create masks
        pos_mask = weight > delta_pos
        neg_mask = weight < -delta_pos
        zero_mask = ~(pos_mask | neg_mask)

        # Quantize: +wp for positive, -wn for negative, 0 for middle
        quantized = torch.zeros_like(weight)
        quantized[pos_mask] = wp_pos
        quantized[neg_mask] = -wn_pos

        # Save for backward
        ctx.save_for_backward(weight, wp_pos, wn_pos, pos_mask, neg_mask, zero_mask)

        return quantized

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        """Backward pass: STE for weight, proper gradients for wp/wn/delta."""
        weight, wp_pos, wn_pos, pos_mask, neg_mask, zero_mask = ctx.saved_tensors

        # Gradient w.r.t. weight: straight-through estimator (pass through unchanged)
        grad_weight = grad_output.clone()

        # Gradient w.r.t. wp: sum over all positive weight positions
        grad_wp = (grad_output * pos_mask.float()).sum().reshape_as(wp_pos)

        # Gradient w.r.t. wn: sum over all negative weight positions (note the sign)
        grad_wn = (-grad_output * neg_mask.float()).sum().reshape_as(wn_pos)

        # Gradient w.r.t. delta: None (threshold is non-differentiable)
        grad_delta = None

        return grad_weight, grad_wp, grad_wn, grad_delta


def ttq_quantize(weight: Tensor, wp: Tensor, wn: Tensor, delta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize weights to {-wn, 0, +wp} using Trained Ternary Quantization.

    TTQ (Zhu et al., ICLR 2017) learns per-layer positive/negative scales
    and threshold. Weights are quantized based on learned threshold delta:
        - W > delta  → +wp
        - W < -delta → -wn
        - |W| ≤ delta → 0

    Args:
        weight: FP32 weight tensor
        wp: Learnable positive scale
        wn: Learnable negative scale
        delta: Learnable threshold

    Returns:
        Tuple of (quantized weights, wp_positive, wn_positive)
    """
    # Ensure scales and threshold are positive with softplus (maintains gradients)
    wp_pos = f.softplus(wp)
    wn_pos = f.softplus(wn)
    delta_pos = f.softplus(delta)

    # Apply custom quantization with proper gradient flow
    quantized = TTQQuantizeFunction.apply(weight, wp_pos, wn_pos, delta_pos)

    return quantized, wp_pos, wn_pos
