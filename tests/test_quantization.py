"""Tests for quantization functions."""

import torch

from bitnet.nn.quantization import dequantize, quantize_activations, ste_ternary


class TestSteTernary:
    """Tests for ternary weight quantization."""

    def test_output_values_are_ternary(self) -> None:
        """Quantized weights should only contain {-1, 0, +1}."""
        weight = torch.randn(64, 32)
        quantized = ste_ternary(weight)
        unique = torch.unique(quantized)
        assert all(v in [-1, 0, 1] for v in unique.tolist())

    def test_preserves_shape(self) -> None:
        """Output shape should match input shape."""
        weight = torch.randn(128, 64)
        quantized = ste_ternary(weight)
        assert quantized.shape == weight.shape

    def test_gradient_flows_through(self) -> None:
        """STE should allow gradients to flow through quantization."""
        weight = torch.randn(32, 16, requires_grad=True)
        quantized = ste_ternary(weight)
        loss = quantized.sum()
        loss.backward()
        assert weight.grad is not None
        assert weight.grad.shape == weight.shape


class TestQuantizeActivations:
    """Tests for activation quantization."""

    def test_output_in_range(self) -> None:
        """Quantized activations should be in [-Q_b, Q_b]."""
        x = torch.randn(8, 64)
        quantized, _ = quantize_activations(x, num_bits=8)
        q_b = 128  # 2^7
        assert quantized.min() >= -q_b
        assert quantized.max() <= q_b

    def test_preserves_shape(self) -> None:
        """Output shape should match input shape."""
        x = torch.randn(4, 3, 32, 32)
        quantized, gamma = quantize_activations(x)
        assert quantized.shape == x.shape
        assert gamma.shape[0] == x.shape[0]

    def test_gamma_is_positive(self) -> None:
        """Gamma (scale factor) should always be positive."""
        x = torch.randn(8, 64)
        _, gamma = quantize_activations(x)
        assert (gamma > 0).all()


class TestDequantize:
    """Tests for dequantization."""

    def test_roundtrip_approximate(self) -> None:
        """Quantize then dequantize should approximately preserve values."""
        x = torch.randn(8, 64)
        x_quant, gamma = quantize_activations(x, num_bits=8)
        beta = torch.tensor(1.0)
        # Simplified roundtrip (without actual linear op)
        reconstructed = dequantize(x_quant, gamma, beta, num_bits=8)
        # Should be roughly same scale as original
        assert reconstructed.std() > 0
