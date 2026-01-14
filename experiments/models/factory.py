"""Model factory for creating standard and bit-quantized models."""

import timm
from torch import nn

from bitnet.layer_swap import replace_layers
from bitnet.layer_swap_selective import replace_layers_selective


def get_model(
    name: str,
    num_classes: int,
    bit_version: bool = False,
    pretrained: bool = True,
    skip_layers: set[str] | None = None,
) -> nn.Module:
    """Get a model using timm with optional bit quantization.

    Args:
        name: Model architecture name (e.g., 'resnet18', 'mobilenetv3_small_100')
        num_classes: Number of output classes
        bit_version: If True, replace layers with 1.58-bit versions
        pretrained: If True, load pretrained weights
        skip_layers: Layer prefixes to keep in FP32 (for ablation experiments)

    Returns:
        PyTorch model
    """
    model = timm.create_model(name, num_classes=num_classes, pretrained=pretrained)

    if bit_version:
        if skip_layers:
            replace_layers_selective(model, skip_layers)
        else:
            replace_layers(model)

    return model


def list_available_models(pattern: str = "*") -> list[str]:
    """List available models from timm."""
    return timm.list_models(pattern)
