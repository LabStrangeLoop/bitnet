"""1.58-bit Neural Network Implementation"""

from bitnet.layer_swap import (
    replace_conv2d_layers,
    replace_layers,
    replace_linear_layers,
)

__all__ = ["replace_layers", "replace_linear_layers", "replace_conv2d_layers"]
