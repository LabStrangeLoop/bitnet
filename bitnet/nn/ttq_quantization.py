import torch
from torch import Tensor


def ttq_quantize(weight: Tensor, wp: Tensor, wn: Tensor, delta: Tensor) -> Tensor:
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
        Quantized tensor in {-wn, 0, +wp}
    """
    # Apply threshold-based ternary quantization
    pos_mask = weight > delta
    neg_mask = weight < -delta

    quantized = torch.zeros_like(weight)
    quantized[pos_mask] = wp
    quantized[neg_mask] = -wn

    # Straight-through estimator for gradients
    return quantized + (weight - weight.detach())
