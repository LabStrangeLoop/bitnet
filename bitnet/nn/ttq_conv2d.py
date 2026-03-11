import torch
import torch.nn.functional as f
from torch import Tensor, nn

from bitnet.nn.ttq_quantization import ttq_quantize


class TTQConv2d(nn.Conv2d):
    """Conv2d layer with TTQ (Trained Ternary Quantization).

    Pure TTQ as described in the paper: ternary weight quantization with FP32 activations.
    Testing showed that mixing BitNet's activation quantization with TTQ weights fails to train.

    TTQ learns per-layer positive/negative scales (wp, wn) and threshold (delta)
    during training, achieving near-FP32 accuracy at the cost of 2 additional
    FP32 parameters per layer.

    Reference: Zhu et al., "Trained Ternary Quantization", ICLR 2017
    """

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)

        # Initialize scales and threshold per TTQ paper (Zhu et al., ICLR 2017)
        # Eq. 2: threshold = 0.7 * E[|W|], scales = E[|W|]
        weight_mean_abs = self.weight.data.abs().mean()
        self.register_parameter("wp", nn.Parameter(torch.ones(1) * weight_mean_abs))
        self.register_parameter("wn", nn.Parameter(torch.ones(1) * weight_mean_abs))
        self.register_parameter("delta", nn.Parameter(torch.ones(1) * 0.7 * weight_mean_abs))

    def forward(self, x: Tensor) -> Tensor:
        # Pure TTQ: Only quantize weights, use FP32 activations
        # This is TTQ as described in the original paper
        w_quant, _, _ = ttq_quantize(self.weight, self.wp, self.wn, self.delta)
        return f.conv2d(x, w_quant, self.bias, self.stride, self.padding, self.dilation, self.groups)
