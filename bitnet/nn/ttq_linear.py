import torch
import torch.nn.functional as f
from torch import Tensor, nn

from bitnet.nn.ttq_quantization import ttq_quantize


class TTQLinear(nn.Linear):
    """Linear layer with TTQ (Trained Ternary Quantization).

    Pure TTQ as described in the paper: ternary weight quantization with FP32 activations.
    Testing showed that mixing BitNet's activation quantization with TTQ weights fails to train.

    TTQ learns per-layer positive/negative scales (wp, wn) and threshold (delta)
    during training, achieving near-FP32 accuracy at the cost of 2 additional
    FP32 parameters per layer.

    Reference: Zhu et al., "Trained Ternary Quantization", ICLR 2017
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__(in_features, out_features, bias)

        # Initialize scales and threshold per TTQ paper (Zhu et al., ICLR 2017)
        # Eq. 2: threshold = 0.7 * E[|W|], scales = E[|W|]
        # Note: We use softplus to ensure positivity, so initialize with inverse softplus
        # to get the desired values AFTER softplus is applied
        weight_mean_abs = self.weight.data.abs().mean()

        # Inverse softplus: softplus_inv(x) = log(exp(x) - 1)
        def inv_softplus(x: float) -> float:
            if x < 1e-3:
                return x  # For very small values, softplus(x) ≈ x
            return float(torch.log(torch.exp(torch.tensor(x)) - 1.0))

        self.register_parameter("wp", nn.Parameter(torch.ones(1) * inv_softplus(weight_mean_abs.item())))
        self.register_parameter("wn", nn.Parameter(torch.ones(1) * inv_softplus(weight_mean_abs.item())))
        self.register_parameter("delta", nn.Parameter(torch.ones(1) * inv_softplus((0.7 * weight_mean_abs).item())))

    def forward(self, x: Tensor) -> Tensor:
        # Pure TTQ: Only quantize weights, use FP32 activations
        # This is TTQ as described in the original paper
        # Type assertions for mypy - these are always Tensors after register_parameter
        assert isinstance(self.wp, Tensor)
        assert isinstance(self.wn, Tensor)
        assert isinstance(self.delta, Tensor)

        w_quant, _, _ = ttq_quantize(self.weight, self.wp, self.wn, self.delta)
        return f.linear(x, w_quant, self.bias)
