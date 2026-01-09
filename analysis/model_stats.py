"""Model statistics: parameter counts and compression ratios."""

from torch import nn

from experiments.models.factory import get_model

MODELS = ["resnet18", "resnet50"]
FP32_BITS = 32
BITNET_WEIGHT_BITS = 1.58  # log2(3) for ternary {-1, 0, +1}
BITNET_ACTIVATION_BITS = 8


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def compute_compression_ratio(num_params: int) -> float:
    """Compute theoretical compression ratio for BitNet vs FP32."""
    fp32_bits = num_params * FP32_BITS
    bitnet_bits = num_params * BITNET_WEIGHT_BITS
    return fp32_bits / bitnet_bits


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
        "params": params["total"],
        "params_str": format_params(params["total"]),
        "fp32_mb": params["total"] * FP32_BITS / 8 / 1e6,
        "bitnet_mb": params["total"] * BITNET_WEIGHT_BITS / 8 / 1e6,
        "compression": compute_compression_ratio(params["total"]),
    }


def print_model_table() -> None:
    """Print model statistics table."""
    print(f"{'Model':<12} {'Params':<10} {'FP32 (MB)':<12} {'BitNet (MB)':<12} {'Compression':<12}")
    print("-" * 58)
    for model_name in MODELS:
        stats = get_model_stats(model_name)
        print(
            f"{stats['model']:<12} {stats['params_str']:<10} "
            f"{stats['fp32_mb']:<12.2f} {stats['bitnet_mb']:<12.2f} {stats['compression']:<12.1f}x"
        )


if __name__ == "__main__":
    print_model_table()
