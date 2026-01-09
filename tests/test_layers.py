"""Tests for BitLinear and BitConv2d layers."""

import torch

from bitnet.nn.bitconv2d import BitConv2d
from bitnet.nn.bitlinear import BitLinear


class TestBitLinear:
    """Tests for BitLinear layer."""

    def test_forward_shape(self) -> None:
        """Output shape should be (batch, out_features)."""
        layer = BitLinear(64, 32)
        x = torch.randn(8, 64)
        out = layer(x)
        assert out.shape == (8, 32)

    def test_gradient_flows(self) -> None:
        """Gradients should flow through the layer."""
        layer = BitLinear(64, 32)
        x = torch.randn(8, 64, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert layer.weight.grad is not None

    def test_weights_are_quantized_during_forward(self) -> None:
        """During forward, effective weights should be ternary."""
        layer = BitLinear(64, 32)
        # Access quantized weights indirectly via the quantization function
        from bitnet.nn.quantization import ste_ternary

        w_quant = ste_ternary(layer.weight)
        unique = torch.unique(w_quant)
        assert all(v in [-1, 0, 1] for v in unique.tolist())


class TestBitConv2d:
    """Tests for BitConv2d layer."""

    def test_forward_shape(self) -> None:
        """Output shape should follow conv2d formula."""
        layer = BitConv2d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(4, 3, 32, 32)
        out = layer(x)
        assert out.shape == (4, 16, 32, 32)

    def test_gradient_flows(self) -> None:
        """Gradients should flow through the layer."""
        layer = BitConv2d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(4, 3, 32, 32, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert layer.weight.grad is not None

    def test_different_kernel_sizes(self) -> None:
        """Layer should work with various kernel sizes."""
        for kernel_size in [1, 3, 5, 7]:
            layer = BitConv2d(3, 8, kernel_size=kernel_size, padding=kernel_size // 2)
            x = torch.randn(2, 3, 16, 16)
            out = layer(x)
            assert out.shape == (2, 8, 16, 16)
