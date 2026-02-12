import torch.nn.functional as f
from torch import Tensor, nn

from bitnet.nn.quantization import dequantize, quantize_activations, ste_ternary


class BitConv2d(nn.Conv2d):
    """Conv2d layer with 1.58-bit weights (ternary: {-1, 0, +1})."""

    def __init__(self, *args, num_bits: int = 8, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.num_bits = num_bits

    def forward(self, x: Tensor) -> Tensor:
        x = f.layer_norm(x, x.shape[1:])
        x_quant, gamma = quantize_activations(x, self.num_bits)

        w_quant = ste_ternary(self.weight)
        beta = self.weight.abs().mean()

        out = f.conv2d(x_quant, w_quant, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return dequantize(out, gamma, beta, self.num_bits)
