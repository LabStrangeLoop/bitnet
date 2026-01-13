"""Training and evaluation loops."""

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LRScheduler, SequentialLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_scheduler(optimizer: optim.Optimizer, name: str, epochs: int, warmup_epochs: int = 0) -> LRScheduler:
    """Create a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        name: Scheduler type ("cosine", "step", or "none").
        epochs: Total training epochs.
        warmup_epochs: Number of warmup epochs.
    """

    def make_main_scheduler() -> LRScheduler:
        main_epochs = epochs - warmup_epochs
        schedulers = {
            "cosine": lambda: CosineAnnealingLR(optimizer, T_max=main_epochs),
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


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    *,
    quiet: bool = False,
) -> tuple[float, float]:
    """Run one training epoch.

    Returns:
        Tuple of (average loss, accuracy percentage).
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    iterator = loader if quiet else tqdm(loader, desc="Training", leave=False)
    for inputs, targets in iterator:
        inputs, targets = inputs.to(device), targets.to(device)
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
