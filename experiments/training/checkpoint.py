"""Checkpoint management and training state persistence."""

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn, optim

log = logging.getLogger(__name__)


def save(
    output_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    best_acc: float,
    history: dict,
) -> None:
    """Save training checkpoint."""
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc": best_acc,
        "history": history,
    }
    torch.save(checkpoint, output_dir / "checkpoint.pth")


def load(
    output_dir: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
) -> tuple[int, float, dict]:
    """Load checkpoint if exists, otherwise return defaults."""
    checkpoint_path = output_dir / "checkpoint.pth"
    if not checkpoint_path.exists():
        return 0, 0.0, {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    log.info("Resumed from epoch %d (best_acc=%.2f%%)", checkpoint["epoch"], checkpoint["best_acc"])
    return checkpoint["epoch"], checkpoint["best_acc"], checkpoint["history"]


def exists(output_dir: Path) -> bool:
    """Check if checkpoint exists."""
    return (output_dir / "checkpoint.pth").exists()


def cleanup(output_dir: Path) -> None:
    """Remove checkpoint after successful training."""
    (output_dir / "checkpoint.pth").unlink(missing_ok=True)
