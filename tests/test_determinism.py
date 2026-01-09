"""Tests for training reproducibility with fixed seeds."""

import torch
from torch import nn, optim

from bitnet.nn.bitconv2d import BitConv2d
from bitnet.nn.bitlinear import BitLinear
from experiments.train import set_seed


def _create_model() -> nn.Module:
    """Create a small model for testing."""
    return nn.Sequential(
        BitConv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        BitLinear(8, 10),
    )


def _run_training_step(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one training step and return loss and model output."""
    set_seed(seed)
    model = _create_model()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Fixed random input (but generated after seed is set)
    x = torch.randn(4, 3, 32, 32)
    targets = torch.tensor([0, 1, 2, 3])

    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    return loss.detach(), output.detach()


class TestDeterminism:
    """Tests for reproducible training."""

    def test_same_seed_same_initialization(self) -> None:
        """Same seed should produce identical model initialization."""
        set_seed(42)
        model1 = _create_model()

        set_seed(42)
        model2 = _create_model()

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1, p2), "Model initialization differs with same seed"

    def test_different_seed_different_initialization(self) -> None:
        """Different seeds should produce different initialization."""
        set_seed(42)
        model1 = _create_model()

        set_seed(123)
        model2 = _create_model()

        params_differ = False
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if not torch.equal(p1, p2):
                params_differ = True
                break
        assert params_differ, "Different seeds produced identical models"

    def test_same_seed_same_training_step(self) -> None:
        """Same seed should produce identical training results."""
        loss1, output1 = _run_training_step(seed=42)
        loss2, output2 = _run_training_step(seed=42)

        assert torch.equal(loss1, loss2), f"Losses differ: {loss1} vs {loss2}"
        assert torch.equal(output1, output2), "Outputs differ with same seed"

    def test_different_seed_different_training_step(self) -> None:
        """Different seeds should produce different training results."""
        loss1, output1 = _run_training_step(seed=42)
        loss2, output2 = _run_training_step(seed=123)

        assert not torch.equal(output1, output2), "Different seeds produced same output"

    def test_multiple_steps_reproducible(self) -> None:
        """Multiple training steps should be reproducible."""
        def run_multiple_steps(seed: int, num_steps: int = 3) -> list[float]:
            set_seed(seed)
            model = _create_model()
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            losses = []
            for _ in range(num_steps):
                x = torch.randn(4, 3, 32, 32)
                targets = torch.randint(0, 10, (4,))
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            return losses

        losses1 = run_multiple_steps(seed=42)
        losses2 = run_multiple_steps(seed=42)

        for i, (l1, l2) in enumerate(zip(losses1, losses2)):
            assert l1 == l2, f"Step {i}: losses differ {l1} vs {l2}"
