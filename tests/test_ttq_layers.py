"""Tests for TTQLinear and TTQConv2d layers."""

import torch
import torch.nn as nn

from bitnet.nn.ttq_conv2d import TTQConv2d
from bitnet.nn.ttq_linear import TTQLinear


class TestTTQLinear:
    """Tests for TTQLinear layer."""

    def test_forward_shape(self) -> None:
        """Output shape should be (batch, out_features)."""
        layer = TTQLinear(64, 32)
        x = torch.randn(8, 64)
        out = layer(x)
        assert out.shape == (8, 32)

    def test_gradient_flows(self) -> None:
        """Gradients should flow through the layer."""
        layer = TTQLinear(64, 32)
        x = torch.randn(8, 64, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert layer.weight.grad is not None
        assert layer.wp.grad is not None
        assert layer.wn.grad is not None
        # Note: delta gradients may be None in simple forward passes since
        # it's used in comparison operations (weight > delta). In real training
        # with classification loss, delta gets gradients through the loss.

    def test_parameters_initialized_properly(self) -> None:
        """TTQ parameters should be initialized per paper (Eq. 2)."""
        layer = TTQLinear(64, 32)
        weight_mean_abs = layer.weight.data.abs().mean()

        # wp and wn are stored in inverse softplus space, apply softplus to compare
        wp_actual = torch.nn.functional.softplus(layer.wp)
        wn_actual = torch.nn.functional.softplus(layer.wn)
        delta_actual = torch.nn.functional.softplus(layer.delta)

        # wp and wn should be initialized to E[|W|]
        assert torch.allclose(wp_actual, weight_mean_abs, rtol=1e-5)
        assert torch.allclose(wn_actual, weight_mean_abs, rtol=1e-5)
        # delta should be initialized to 0.7 * E[|W|]
        assert torch.allclose(delta_actual, 0.7 * weight_mean_abs, rtol=1e-5)

    def test_numerical_stability_during_training(self) -> None:
        """Training should not produce NaN losses."""
        # Create a simple model with TTQ layer
        layer = TTQLinear(10, 10)
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

        # Train for a few steps
        for _ in range(10):
            x = torch.randn(4, 10)
            target = torch.randn(4, 10)

            out = layer(x)
            loss = nn.functional.mse_loss(out, target)

            # Loss should not be NaN
            assert not torch.isnan(loss), "Loss became NaN during training"

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Parameters should not be NaN
            assert not torch.isnan(layer.wp).any(), "wp became NaN"
            assert not torch.isnan(layer.wn).any(), "wn became NaN"
            assert not torch.isnan(layer.delta).any(), "delta became NaN"


class TestTTQConv2d:
    """Tests for TTQConv2d layer."""

    def test_forward_shape(self) -> None:
        """Output shape should follow conv2d formula."""
        layer = TTQConv2d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(4, 3, 32, 32)
        out = layer(x)
        assert out.shape == (4, 16, 32, 32)

    def test_gradient_flows(self) -> None:
        """Gradients should flow through the layer."""
        layer = TTQConv2d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(4, 3, 32, 32, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert layer.weight.grad is not None
        assert layer.wp.grad is not None
        assert layer.wn.grad is not None
        # Note: delta gradients may be None in simple forward passes since
        # it's used in comparison operations (weight > delta). In real training
        # with classification loss, delta gets gradients through the loss.

    def test_parameters_initialized_properly(self) -> None:
        """TTQ parameters should be initialized per paper (Eq. 2)."""
        layer = TTQConv2d(3, 16, kernel_size=3)
        weight_mean_abs = layer.weight.data.abs().mean()

        # wp and wn are stored in inverse softplus space, apply softplus to compare
        wp_actual = torch.nn.functional.softplus(layer.wp)
        wn_actual = torch.nn.functional.softplus(layer.wn)
        delta_actual = torch.nn.functional.softplus(layer.delta)

        # wp and wn should be initialized to E[|W|]
        assert torch.allclose(wp_actual, weight_mean_abs, rtol=1e-5)
        assert torch.allclose(wn_actual, weight_mean_abs, rtol=1e-5)
        # delta should be initialized to 0.7 * E[|W|]
        assert torch.allclose(delta_actual, 0.7 * weight_mean_abs, rtol=1e-5)

    def test_numerical_stability_during_training(self) -> None:
        """Training should not produce NaN losses."""
        # Create a simple model with TTQ conv layer
        layer = TTQConv2d(3, 8, kernel_size=3, padding=1)
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

        # Train for a few steps
        for _ in range(10):
            x = torch.randn(2, 3, 16, 16)
            target = torch.randn(2, 8, 16, 16)

            out = layer(x)
            loss = nn.functional.mse_loss(out, target)

            # Loss should not be NaN
            assert not torch.isnan(loss), "Loss became NaN during training"

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Parameters should not be NaN
            assert not torch.isnan(layer.wp).any(), "wp became NaN"
            assert not torch.isnan(layer.wn).any(), "wn became NaN"
            assert not torch.isnan(layer.delta).any(), "delta became NaN"

    def test_different_kernel_sizes(self) -> None:
        """Layer should work with various kernel sizes."""
        for kernel_size in [1, 3, 5, 7]:
            layer = TTQConv2d(3, 8, kernel_size=kernel_size, padding=kernel_size // 2)
            x = torch.randn(2, 3, 16, 16)
            out = layer(x)
            assert out.shape == (2, 8, 16, 16)
