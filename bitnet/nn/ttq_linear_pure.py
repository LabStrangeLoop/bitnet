"""Pure TTQ Linear layer - weight quantization only, no activation quant."""

import torch
from torch import Tensor, nn

from bitnet.nn.ttq_quantization import ttq_quantize


class TTQLinearPure(nn.Linear):
    """Linear layer with pure TTQ (weight-only quantization).

    This is TTQ as described in the paper - no activation quantization,
    no dequantization. Just ternary weights with learned scales.

    Use this to test if the issue is with weight quantization vs
    the BitNet-style activation quant/dequant we added.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__(in_features, out_features, bias)

        # Initialize scales and threshold per TTQ paper
        weight_mean_abs = self.weight.data.abs().mean()
        self.register_parameter("wp", nn.Parameter(torch.ones(1) * weight_mean_abs))
        self.register_parameter("wn", nn.Parameter(torch.ones(1) * weight_mean_abs))
        self.register_parameter("delta", nn.Parameter(torch.ones(1) * 0.7 * weight_mean_abs))

    def forward(self, x: Tensor) -> Tensor:
        # Pure TTQ: just quantize weights and apply linear
        # No layer norm, no activation quant, no dequantization
        w_quant, _, _ = ttq_quantize(self.weight, self.wp, self.wn, self.delta)
        return nn.functional.linear(x, w_quant, self.bias)
