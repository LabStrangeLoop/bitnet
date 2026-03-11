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

        # Initialize scales and threshold per TTQ paper (Zhu et al., ICLR 2017)
        # Eq. 2: threshold = 0.7 * E[|W|], scales = E[|W|]
        weight_mean_abs = self.weight.data.abs().mean()
        self.register_parameter("wp", nn.Parameter(torch.ones(1) * weight_mean_abs))
        self.register_parameter("wn", nn.Parameter(torch.ones(1) * weight_mean_abs))
        self.register_parameter("delta", nn.Parameter(torch.ones(1) * 0.7 * weight_mean_abs))

    def forward(self, x: Tensor) -> Tensor:
        # Activation quantization (same as BitNet)
        x = f.layer_norm(x, x.shape[1:])
        x_quant, gamma = quantize_activations(x, self.num_bits)

        # TTQ weight quantization with learned scales (already scaled!)
        w_quant, wp_pos, wn_pos = ttq_quantize(self.weight, self.wp, self.wn, self.delta)

        # Beta = 1.0 because quantized weights are already scaled by wp/wn
        # Unlike BitNet which scales {-1,0,+1} with beta in dequant, TTQ pre-scales
        beta = torch.ones_like(wp_pos)

        out = f.linear(x_quant, w_quant, self.bias)
        return dequantize(out, gamma, beta, self.num_bits)
