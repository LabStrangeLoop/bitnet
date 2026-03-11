import torch
import torch.nn.functional as f
from torch import Tensor, nn

from bitnet.nn.quantization import dequantize, quantize_activations
from bitnet.nn.ttq_quantization import ttq_quantize


class TTQLinear(nn.Linear):
    """Linear layer with TTQ (Trained Ternary Quantization).

    TTQ learns per-layer positive/negative scales (Wp, Wn) and threshold (delta)
    during training, achieving near-FP32 accuracy at the cost of 2 additional
    FP32 parameters per layer.

    Reference: Zhu et al., "Trained Ternary Quantization", ICLR 2017
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_bits: int = 8,
    ):
        super().__init__(in_features, out_features, bias)
        self.num_bits = num_bits

        # Learnable positive/negative scales (init to 1.0)
        self.register_parameter("wp", nn.Parameter(torch.ones(1)))
        self.register_parameter("wn", nn.Parameter(torch.ones(1)))

        # Learnable threshold (init to 0.7 * std as in paper)
        weight_std = self.weight.data.std()
        self.register_parameter("delta", nn.Parameter(torch.ones(1) * 0.7 * weight_std))

    def forward(self, x: Tensor) -> Tensor:
        # Activation quantization (same as BitNet)
        x = f.layer_norm(x, x.shape[1:])
        x_quant, gamma = quantize_activations(x, self.num_bits)

        # TTQ weight quantization with learned scales
        w_quant = ttq_quantize(self.weight, self.wp, self.wn, self.delta)

        # Use average of positive scales as beta for dequantization
        beta = (self.wp.abs() + self.wn.abs()) / 2

        out = f.linear(x_quant, w_quant, self.bias)
        return dequantize(out, gamma, beta, self.num_bits)
