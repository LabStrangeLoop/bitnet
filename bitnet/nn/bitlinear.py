import torch.nn.functional as f
from torch import Tensor, nn

from bitnet.nn.quantization import dequantize, quantize_activations, ste_ternary


class BitLinear(nn.Linear):
    """Linear layer with 1.58-bit weights (ternary: {-1, 0, +1})."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_bits: int = 8,
    ):
        super().__init__(in_features, out_features, bias)
        self.num_bits = num_bits

    def forward(self, x: Tensor) -> Tensor:
        x = f.layer_norm(x, x.shape[1:])
        x_quant, gamma = quantize_activations(x, self.num_bits)

        w_quant = ste_ternary(self.weight)
        beta = self.weight.abs().mean()

        out = f.linear(x_quant, w_quant, self.bias)
        return dequantize(out, gamma, beta, self.num_bits)
