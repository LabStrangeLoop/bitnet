"""Logging configuration for training experiments."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.utils.tensorboard.writer import SummaryWriter

    from experiments.config import EpochMetrics, TrainConfig

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

log = logging.getLogger(__name__)


def setup(output_dir: Path, *, resume: bool = False, quiet: bool = False) -> None:
    """Configure logging to both console and file."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    console.setLevel(logging.WARNING if quiet else logging.INFO)
    root.addHandler(console)

    log_file = output_dir / "train.log"
    file_mode = "a" if resume else "w"
    file_handler = logging.FileHandler(log_file, mode=file_mode)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root.addHandler(file_handler)


def log_experiment_header(config: TrainConfig, device: torch.device) -> None:
    """Log experiment configuration header."""
    log.info("=" * 60)
    log.info("Experiment: %s %s on %s (seed=%d)", config.model, config.version.value, config.dataset, config.seed)
    log.info("Device: %s (GPUs: %d)", device, torch.cuda.device_count())
    log.info(
        "Hyperparams: lr=%.4f, wd=%.0e, batch=%d, epochs=%d",
        config.lr,
        config.weight_decay,
        config.batch_size,
        config.epochs,
    )
    log.info("=" * 60)


# Milestones for quiet mode progress (percentage of total epochs)
_PROGRESS_MILESTONES = {25, 50, 75}


def log_epoch(
    epoch: int,
    metrics: EpochMetrics,
    writer: SummaryWriter | None = None,
    *,
    total_epochs: int = 0,
) -> None:
    """Log epoch metrics to console and optionally TensorBoard.

    When total_epochs > 0, also logs progress at 25/50/75% milestones using WARNING level
    (visible in quiet mode).
    """
    log.info(
        "Epoch %3d | Train: %.4f / %.2f%% | Test: %.4f / %.2f%%",
        epoch,
        metrics.train_loss,
        metrics.train_acc,
        metrics.test_loss,
        metrics.test_acc,
    )
    if writer:
        for name, val in metrics.as_dict().items():
            writer.add_scalar(f"epoch/{name}", val, epoch)

    # Progress milestones for quiet mode
    if total_epochs > 0:
        pct = (epoch * 100) // total_epochs
        prev_pct = ((epoch - 1) * 100) // total_epochs
        if pct in _PROGRESS_MILESTONES and prev_pct < pct:
            log.warning("Progress: %d%% (%d/%d epochs, acc=%.2f%%)", pct, epoch, total_epochs, metrics.test_acc)
