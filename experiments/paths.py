"""Centralized experiment path generation.

Single source of truth for directory naming. Prevents accidental overwrites
and ensures consistent naming across all training scripts.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ExperimentType(Enum):
    """Type of experiment determines the base results directory."""

    STANDARD = "raw"
    KD = "raw_kd"


@dataclass(frozen=True)
class KDDefaults:
    """Default values for knowledge distillation hyperparameters."""

    temperature: float = 4.0
    alpha: float = 0.9


KD_DEFAULTS = KDDefaults()


def get_experiment_dir(
    experiment_type: ExperimentType,
    dataset: str,
    model: str,
    seed: int,
    *,
    version: str = "bit",
    augment: str = "basic",
    ablation: str = "none",
    kd_temperature: float | None = None,
    kd_alpha: float | None = None,
) -> Path:
    """Generate experiment directory path based on configuration.

    Only includes non-default values in the path name to keep paths readable.

    Args:
        experiment_type: Type of experiment (standard or KD).
        dataset: Dataset name (cifar10, cifar100, imagenet).
        model: Model architecture (resnet18, resnet50, etc.).
        seed: Random seed.
        version: Model version (std or bit).
        augment: Augmentation strategy (basic, cutout, randaug, full).
        ablation: Ablation mode (none, keep_conv1, keep_layer1, etc.).
        kd_temperature: KD temperature (only for KD experiments).
        kd_alpha: KD alpha (only for KD experiments).

    Returns:
        Path to experiment directory.
    """
    base_dir = Path("results") / experiment_type.value / dataset / model
    parts: list[str] = []

    if experiment_type == ExperimentType.STANDARD:
        parts.append(version)
        if augment != "basic":
            parts.append(augment)
        if ablation != "none":
            parts.append(ablation)
    else:
        parts.append("bit_kd")
        if ablation != "none":
            parts.append(ablation)
        if kd_temperature is not None and kd_temperature != KD_DEFAULTS.temperature:
            parts.append(f"t{kd_temperature:g}")
        if kd_alpha is not None and kd_alpha != KD_DEFAULTS.alpha:
            parts.append(f"a{kd_alpha:g}")

    parts.append(f"s{seed}")
    run_name = "_".join(parts)
    return base_dir / run_name


def experiment_exists(experiment_dir: Path) -> bool:
    """Check if an experiment has already been completed."""
    return (experiment_dir / "results.json").exists()


def should_skip_experiment(experiment_dir: Path, *, force: bool = False) -> bool:
    """Check if an experiment should be skipped.

    Args:
        experiment_dir: Path to experiment directory.
        force: If True, allow overwriting existing results.

    Returns:
        True if the experiment should be skipped (already exists and not forcing).
    """
    if experiment_exists(experiment_dir) and not force:
        print(f"\n[SKIP] Experiment already exists: {experiment_dir}")
        print("       Use --force to overwrite.\n")
        return True
    return False
