import torch
from torch import Tensor


def ste_ternary(weight: Tensor) -> Tensor:
    """Quantize weights to {-1, 0, +1} using straight-through estimator.

    BitNet b1.58 formula: W̃ = RoundClip(W / (γ + ε), -1, 1) where γ = mean(|W|)
    """
    gamma = weight.abs().mean()
    scaled = weight / (gamma + 1e-5)
    quantized = torch.clamp(torch.round(scaled), -1, 1)
    return quantized + (weight - weight.detach())  # STE trick


def quantize_activations(x: Tensor, num_bits: int = 8) -> tuple[Tensor, Tensor]:
    """Absmax quantization of activations to [-Q_b, Q_b].

    Returns quantized tensor and gamma (for dequantization).
    """
    q_b = 2 ** (num_bits - 1)
    gamma = x.abs().amax(dim=tuple(range(1, x.ndim)), keepdim=True).clamp(min=1e-5)
    quantized = torch.clamp(x * q_b / gamma, -q_b + 1e-5, q_b - 1e-5)
    return quantized, gamma


def dequantize(x: Tensor, gamma: Tensor, beta: Tensor, num_bits: int = 8) -> Tensor:
    """Dequantize output: x × βγ / Q_b"""
    q_b = 2 ** (num_bits - 1)
    result: Tensor = x * gamma * beta / q_b
    return result
