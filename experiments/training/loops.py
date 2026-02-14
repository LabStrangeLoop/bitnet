"""Training and evaluation loops."""

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LRScheduler, SequentialLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_scheduler(
    optimizer: optim.Optimizer, name: str, epochs: int, warmup_epochs: int = 0, min_lr: float = 0.0
) -> LRScheduler:
    """Create a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        name: Scheduler type ("cosine", "step", or "none").
        epochs: Total training epochs.
        warmup_epochs: Number of warmup epochs.
        min_lr: Minimum learning rate for cosine scheduler.
    """

    def make_main_scheduler() -> LRScheduler:
        main_epochs = epochs - warmup_epochs
        schedulers = {
            "cosine": lambda: CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=min_lr),
            "step": lambda: StepLR(optimizer, step_size=30, gamma=0.1),
            "none": lambda: LambdaLR(optimizer, lambda _: 1.0),
        }
        if name not in schedulers:
            raise ValueError(f"Unknown scheduler: {name}")
        return schedulers[name]()

    if warmup_epochs > 0:
        warmup = LambdaLR(optimizer, lambda e: (e + 1) / warmup_epochs)
        return SequentialLR(optimizer, [warmup, make_main_scheduler()], milestones=[warmup_epochs])
    return make_main_scheduler()


def mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply mixup augmentation to a batch.

    Returns:
        mixed_x: Mixed inputs
        y_a: Original targets
        y_b: Shuffled targets
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float
) -> torch.Tensor:
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    *,
    mixup_alpha: float = 0.0,
    quiet: bool = False,
) -> tuple[float, float]:
    """Run one training epoch.

    Args:
        model: Model to train.
        loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to use.
        mixup_alpha: Mixup alpha parameter (0 = disabled).
        quiet: Suppress progress bars.

    Returns:
        Tuple of (average loss, accuracy percentage).
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    iterator = loader if quiet else tqdm(loader, desc="Training", leave=False)

    for inputs, targets in iterator:
        inputs, targets = inputs.to(device), targets.to(device)

        # Apply mixup if enabled
        if mixup_alpha > 0:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            # For accuracy, use the dominant target (lam > 0.5 → targets_a, else targets_b)
            dominant_targets = targets_a if lam > 0.5 else targets_b
            correct += outputs.argmax(1).eq(dominant_targets).sum().item()
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            correct += outputs.argmax(1).eq(targets).sum().item()

        total += targets.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    quiet: bool = False,
) -> tuple[float, float]:
    """Evaluate model on a dataset.

    Returns:
        Tuple of (average loss, accuracy percentage).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    iterator = loader if quiet else tqdm(loader, desc="Evaluating", leave=False)
    for inputs, targets in iterator:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)
    return total_loss / total, 100.0 * correct / total
