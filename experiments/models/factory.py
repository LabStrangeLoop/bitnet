"""Model factory for creating standard and bit-quantized models."""

import timm
from torch import nn

from bitnet.layer_swap import replace_layers
from bitnet.layer_swap_selective import replace_layers_selective


def adapt_resnet_stem_for_small_images(model: nn.Module) -> None:
    """Adapt ResNet stem for small images (CIFAR-10/100, Tiny-ImageNet).

    Replaces 7x7 stride-2 conv + maxpool with 3x3 stride-1 conv and removes
    maxpool. This preserves spatial resolution for 32x32 and 64x64 images.

    Standard ResNet stem destroys spatial information on small images:
    - 32x32 → 16x16 → 8x8 (loses most information)
    - 64x64 → 32x32 → 16x16 (significant loss)

    CIFAR-adapted stem preserves resolution:
    - 32x32 → 32x32 (no loss)
    - 64x64 → 64x64 (no loss)

    Args:
        model: ResNet model with standard ImageNet stem.

    Raises:
        AttributeError: If model doesn't have conv1 or maxpool attributes.
    """
    if not hasattr(model, "conv1") or not hasattr(model, "maxpool"):
        raise AttributeError("Model must have conv1 and maxpool attributes")

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()


def get_model(
    name: str,
    num_classes: int,
    bit_version: bool = False,
    pretrained: bool = True,
    skip_layers: set[str] | None = None,
    use_cifar_stem: bool = False,
) -> nn.Module:
    """Get a model using timm with optional bit quantization.

    Args:
        name: Model architecture name (e.g., 'resnet18', 'mobilenetv3_small_100')
        num_classes: Number of output classes
        bit_version: If True, replace layers with 1.58-bit versions
        pretrained: If True, load pretrained weights
        skip_layers: Layer prefixes to keep in FP32 (for ablation experiments)
        use_cifar_stem: If True, adapt ResNet stem for small images (CIFAR/Tiny-ImageNet)

    Returns:
        PyTorch model
    """
    model = timm.create_model(name, num_classes=num_classes, pretrained=pretrained)

    if use_cifar_stem and "resnet" in name.lower():
        adapt_resnet_stem_for_small_images(model)

    if bit_version:
        if skip_layers:
            replace_layers_selective(model, skip_layers)
        else:
            replace_layers(model)

    return model


def list_available_models(pattern: str = "*") -> list[str]:
    """List available models from timm."""
    return timm.list_models(pattern)
