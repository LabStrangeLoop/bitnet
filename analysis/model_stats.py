"""Model statistics: parameter counts and memory footprint."""

from torch import nn

from experiments.config import MODELS
from experiments.models.factory import get_model

FP32_BITS = 32
BITNET_WEIGHT_BITS = 1.58  # log2(3) for ternary {-1, 0, +1}


def count_parameters(model: nn.Module) -> int:
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())


def format_params(n: int) -> str:
    """Format parameter count as human-readable string."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def get_model_stats(model_name: str, num_classes: int = 1000) -> dict:
    """Get stats for a model."""
    model = get_model(model_name, num_classes, bit_version=False, pretrained=False)
    params = count_parameters(model)
    return {
        "model": model_name,
        "params": params,
        "params_str": format_params(params),
        "fp32_mb": params * FP32_BITS / 8 / 1e6,
        "bitnet_mb": params * BITNET_WEIGHT_BITS / 8 / 1e6,
    }


def print_model_table() -> None:
    """Print model statistics table."""
    print(f"{'Model':<20} {'Params':<10} {'FP32 (MB)':<12} {'BitNet (MB)':<12}")
    print("-" * 54)
    for model_name in MODELS:
        stats = get_model_stats(model_name)
        print(
            f"{stats['model']:<20} {stats['params_str']:<10} "
            f"{stats['fp32_mb']:<12.2f} {stats['bitnet_mb']:<12.2f}"
        )

    compression = FP32_BITS / BITNET_WEIGHT_BITS
    print(f"\nTheoretical weight compression: {compression:.1f}x (32-bit -> 1.58-bit)")


if __name__ == "__main__":
    print_model_table()
