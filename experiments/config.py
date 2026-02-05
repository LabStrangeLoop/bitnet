"""Experiment configuration: models, datasets, and seeds."""

from dataclasses import dataclass, field
from enum import Enum

from .paths import ExperimentType, get_experiment_dir


class Version(Enum):
    """Model version: standard (FP32) or bit-quantized (1.58-bit)."""

    STD = "std"
    BIT = "bit"

    @classmethod
    def from_bool(cls, bit_version: bool) -> "Version":
        """Convert boolean flag to Version enum."""
        return cls.BIT if bit_version else cls.STD


class AblationMode(Enum):
    """Layer-wise ablation: which layer(s) to keep in FP32 while quantizing rest.

    For ResNet architectures:
    - conv1: First convolutional layer (stem)
    - layer1-4: Residual block groups
    - fc: Final classifier layer
    """

    NONE = "none"  # Full BitNet (all layers quantized)
    KEEP_CONV1 = "keep_conv1"  # Keep first conv in FP32
    KEEP_LAYER1 = "keep_layer1"  # Keep first residual block in FP32
    KEEP_LAYER4 = "keep_layer4"  # Keep last residual block in FP32
    KEEP_FC = "keep_fc"  # Keep classifier in FP32


# Mapping from ablation mode to layer name prefixes to skip
ABLATION_SKIP_LAYERS: dict[AblationMode, set[str]] = {
    AblationMode.NONE: set(),
    AblationMode.KEEP_CONV1: {"conv1"},
    AblationMode.KEEP_LAYER1: {"layer1"},
    AblationMode.KEEP_LAYER4: {"layer4"},
    AblationMode.KEEP_FC: {"fc", "head", "classifier"},  # Different model naming
}


@dataclass
class TrainConfig:
    """Training configuration with typed fields and defaults."""

    model: str = "resnet18"
    dataset: str = "cifar10"
    version: Version = Version.STD
    ablation: AblationMode = AblationMode.NONE
    pretrained: bool = False
    epochs: int = 200
    batch_size: int = 128
    lr: float = 0.1
    weight_decay: float = 5e-4
    optimizer: str = "sgd"
    scheduler: str = "cosine"
    warmup_epochs: int = 0
    augment: str = "basic"
    seed: int = 42
    num_workers: int = 4
    data_dir: str = "./data"
    output_dir: str = field(default="")  # Set dynamically based on other fields
    tensorboard: bool = True
    quiet: bool = False

    def __post_init__(self) -> None:
        """Set output_dir if not provided."""
        if not self.output_dir:
            experiment_dir = get_experiment_dir(
                ExperimentType.STANDARD,
                self.dataset,
                self.model,
                self.seed,
                version=self.version.value,
                augment=self.augment,
                ablation=self.ablation.value,
            )
            self.output_dir = str(experiment_dir)


# Convenience: default values for argparse (frozen to prevent modification)
@dataclass(frozen=True)
class Defaults:
    """Default values for argparse. Use TrainConfig for runtime configuration."""

    model: str = "resnet18"
    dataset: str = "cifar10"
    epochs: int = 200
    batch_size: int = 128
    lr: float = 0.1
    weight_decay: float = 5e-4
    scheduler: str = "cosine"
    seed: int = 42
    num_workers: int = 4
    data_dir: str = "./data"
    output_dir: str = "results/raw"


DEFAULTS = Defaults()


@dataclass
class EpochMetrics:
    """Metrics for a single training epoch."""

    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float

    def as_dict(self) -> dict[str, float]:
        """Return metrics as a dictionary for logging."""
        return {
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "test_loss": self.test_loss,
            "test_acc": self.test_acc,
        }


# Experiment matrix
MODELS = ["resnet18", "resnet50", "vgg16", "mobilenetv2_100", "efficientnet_b0"]
DATASETS = ["cifar10", "cifar100", "imagenet"]
SEEDS = [42, 123, 456]

# Dataset-specific settings
DATASET_NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "imagenet": 1000}
DATASET_EPOCHS = {"cifar10": 200, "cifar100": 200, "imagenet": 90}
