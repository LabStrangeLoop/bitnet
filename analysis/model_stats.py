"""Model statistics: parameter counts, memory footprint, and computational cost."""

import torch
from thop import profile
from torch import nn

from experiments.config import MODELS
from experiments.models.factory import get_model

FP32_BITS = 32
BITNET_WEIGHT_BITS = 1.58  # log2(3) for ternary {-1, 0, +1}
TERNARY_PACKING_FACTOR = 64  # 64 ternary ops pack into one 64-bit operation


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


def format_flops(n: float) -> str:
    """Format FLOPs as human-readable string."""
    if n >= 1e12:
        return f"{n / 1e12:.2f}T"
    if n >= 1e9:
        return f"{n / 1e9:.2f}G"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return f"{n:.2f}"


def get_flops(model: nn.Module, input_size: tuple[int, int]) -> float:
    """Calculate FLOPs for a model given input size (H, W)."""
    h, w = input_size
    dummy_input = torch.randn(1, 3, h, w)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    return float(flops)


def get_model_stats(model_name: str, num_classes: int = 1000, input_size: tuple[int, int] = (224, 224)) -> dict:
    """Get comprehensive stats for a model."""
    model = get_model(model_name, num_classes, bit_version=False, pretrained=False)
    model.eval()

    params = count_parameters(model)
    flops = get_flops(model, input_size)

    # BOPs = same operation count as FLOPs, but with ternary weights
    bops = flops
    # Effective operations after packing: 64 ternary ops = 1 64-bit op
    effective_ops = bops / TERNARY_PACKING_FACTOR

    return {
        "model": model_name,
        "params": params,
        "params_str": format_params(params),
        "fp32_mb": params * FP32_BITS / 8 / 1e6,
        "bitnet_mb": params * BITNET_WEIGHT_BITS / 8 / 1e6,
        "flops": flops,
        "flops_str": format_flops(flops),
        "bops": bops,
        "effective_ops": effective_ops,
        "effective_ops_str": format_flops(effective_ops),
        "compute_reduction": flops / effective_ops,
    }


def print_model_table() -> None:
    """Print model statistics table."""
    print(f"{'Model':<20} {'Params':<10} {'FP32 (MB)':<12} {'BitNet (MB)':<12}")
    print("-" * 54)
    for model_name in MODELS:
        stats = get_model_stats(model_name)
        print(f"{stats['model']:<20} {stats['params_str']:<10} {stats['fp32_mb']:<12.2f} {stats['bitnet_mb']:<12.2f}")

    compression = FP32_BITS / BITNET_WEIGHT_BITS
    print(f"\nTheoretical weight compression: {compression:.1f}x (32-bit -> 1.58-bit)")


def print_efficiency_table(input_size: tuple[int, int] = (32, 32)) -> None:
    """Print comprehensive efficiency table with BOPs."""
    h, w = input_size
    print(f"\n{'=' * 80}")
    print(f"EFFICIENCY METRICS (input: {h}x{w})")
    print(f"{'=' * 80}\n")

    # Memory efficiency
    print("MEMORY EFFICIENCY")
    print("-" * 70)
    print(f"{'Model':<18} {'Params':<10} {'FP32 (MB)':<12} {'BitNet (MB)':<12} {'Compress':<10}")
    print("-" * 70)

    all_stats = []
    for model_name in MODELS:
        try:
            stats = get_model_stats(model_name, num_classes=10, input_size=input_size)
            all_stats.append(stats)
            compression = stats["fp32_mb"] / stats["bitnet_mb"]
            print(
                f"{stats['model']:<18} {stats['params_str']:<10} "
                f"{stats['fp32_mb']:<12.2f} {stats['bitnet_mb']:<12.2f} "
                f"{compression:<10.1f}x"
            )
        except Exception as e:
            print(f"{model_name:<18} Error: {e}")

    # Compute efficiency
    print("\n\nCOMPUTE EFFICIENCY (BOPs)")
    print("-" * 70)
    print(f"{'Model':<18} {'FP32 FLOPs':<12} {'BitNet BOPs':<12} {'Effective':<12} {'Speedup':<10}")
    print("-" * 70)

    for stats in all_stats:
        print(
            f"{stats['model']:<18} {stats['flops_str']:<12} "
            f"{stats['flops_str']:<12} {stats['effective_ops_str']:<12} "
            f"{stats['compute_reduction']:<10.1f}x"
        )

    print("\n\nNotes:")
    print("  - BOPs = Binary/Ternary Operations (same count as FLOPs)")
    print("  - Effective = BOPs / 64 (64 ternary ops pack into one 64-bit op)")
    print("  - Speedup = FLOPs / Effective ops (theoretical max)")
    print(f"  - Weight compression: {FP32_BITS / BITNET_WEIGHT_BITS:.1f}x")


def print_ablation_efficiency(model_name: str = "resnet18") -> None:
    """Show efficiency for ablation modes (conv1 FP32 vs full BitNet)."""
    print(f"\n{'=' * 80}")
    print(f"MIXED PRECISION EFFICIENCY ({model_name})")
    print(f"{'=' * 80}\n")

    model = get_model(model_name, num_classes=10, bit_version=False, pretrained=False)

    # Count conv1 parameters specifically
    conv1_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if "conv1" in name and "layer" not in name:  # First conv only
            conv1_params += param.numel()

    other_params = total_params - conv1_params

    # Memory calculations
    full_fp32_mb = total_params * FP32_BITS / 8 / 1e6
    full_bitnet_mb = total_params * BITNET_WEIGHT_BITS / 8 / 1e6
    mixed_mb = (conv1_params * FP32_BITS + other_params * BITNET_WEIGHT_BITS) / 8 / 1e6

    conv1_pct = 100 * conv1_params / total_params
    other_pct = 100 * other_params / total_params
    fp32_compress = full_fp32_mb / full_bitnet_mb
    mixed_compress = full_fp32_mb / mixed_mb
    size_increase_pct = 100 * (mixed_mb - full_bitnet_mb) / full_bitnet_mb

    print("Parameter breakdown:")
    print(f"  - conv1: {format_params(conv1_params)} ({conv1_pct:.2f}%)")
    print(f"  - rest:  {format_params(other_params)} ({other_pct:.2f}%)")
    print(f"  - total: {format_params(total_params)}")

    print("\nMemory footprint:")
    print(f"  - Full FP32:              {full_fp32_mb:.2f} MB")
    print(f"  - Full BitNet:            {full_bitnet_mb:.2f} MB ({fp32_compress:.1f}x compression)")
    print(f"  - Mixed (conv1 FP32):     {mixed_mb:.2f} MB ({mixed_compress:.1f}x compression)")

    print("\nAccuracy vs Efficiency trade-off:")
    print(f"  - Full BitNet:        85.40% acc, {full_bitnet_mb:.2f} MB")
    print(f"  - Mixed (conv1 FP32): 87.40% acc, {mixed_mb:.2f} MB (+2.0% acc, +{size_increase_pct:.1f}% size)")


if __name__ == "__main__":
    # CIFAR-sized inputs
    print_efficiency_table(input_size=(32, 32))

    # Ablation efficiency analysis
    print_ablation_efficiency("resnet18")
