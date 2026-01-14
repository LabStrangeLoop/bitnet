"""Selective layer replacement for layer-wise ablation experiments.

This module extends the basic layer_swap functionality to support
keeping specific layers in FP32 while quantizing others.
"""

from torch import nn

from bitnet.nn import BitConv2d, BitLinear


def _should_skip(full_name: str, skip_prefixes: set[str]) -> bool:
    """Check if a layer should be skipped based on its full hierarchical name."""
    if not skip_prefixes:
        return False
    for prefix in skip_prefixes:
        if full_name == prefix or full_name.startswith(f"{prefix}."):
            return True
    return False


def _replace_linear_selective(
    model: nn.Module,
    skip_prefixes: set[str],
    parent_name: str = "",
) -> None:
    """Recursively replace Linear layers, skipping those matching skip_prefixes."""
    for name, module in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        if isinstance(module, nn.Linear):
            if not _should_skip(full_name, skip_prefixes):
                setattr(
                    model,
                    name,
                    BitLinear(module.in_features, module.out_features, bias=module.bias is not None),
                )
        else:
            _replace_linear_selective(module, skip_prefixes, full_name)


def _replace_conv2d_selective(
    model: nn.Module,
    skip_prefixes: set[str],
    parent_name: str = "",
) -> None:
    """Recursively replace Conv2d layers, skipping those matching skip_prefixes."""
    for name, module in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        if isinstance(module, nn.Conv2d):
            if not _should_skip(full_name, skip_prefixes):
                setattr(
                    model,
                    name,
                    BitConv2d(
                        module.in_channels,
                        module.out_channels,
                        module.kernel_size,
                        module.stride,
                        module.padding,
                        module.dilation,
                        module.groups,
                        bias=module.bias is not None,
                    ),
                )
        else:
            _replace_conv2d_selective(module, skip_prefixes, full_name)


def replace_layers_selective(model: nn.Module, skip_prefixes: set[str]) -> None:
    """Replace layers with BitNet versions, keeping specified layers in FP32.

    Args:
        model: PyTorch model to modify in-place
        skip_prefixes: Set of layer name prefixes to keep as FP32.
            For ResNet: "conv1", "layer1", "layer4", "fc"
            Layers matching these prefixes (or starting with prefix + ".")
            will NOT be quantized.

    Example:
        >>> model = timm.create_model("resnet18", num_classes=10)
        >>> replace_layers_selective(model, {"conv1"})  # Keep first conv FP32
    """
    _replace_linear_selective(model, skip_prefixes)
    _replace_conv2d_selective(model, skip_prefixes)
