"""Generate publication-quality figures for the paper.

⚠️ IMPORTANT: This script uses HARDCODED data from experiments.
After Wave 2 completes, update the hardcoded values with final results.

For automated figure generation during development, use generate_figures.py instead.

This script creates professional figures for:
1. Layer ablation results (conv1 importance)
2. Recipe progression (baseline → KD → conv1 → combo)
3. Gap scaling with task complexity
4. Recipe comparison across datasets
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Publication-quality settings
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Color palette - professional and colorblind-friendly
COLORS = {
    "fp32": "#2E86AB",  # Blue
    "bitnet": "#A23B72",  # Magenta/Pink
    "kd": "#F18F01",  # Orange
    "conv1": "#C73E1D",  # Red
    "combo": "#3A7D44",  # Green
    "neutral": "#6C757D",  # Gray
}


def create_output_dir(output_dir: str = "paper/tmlr/figures") -> Path:
    """Create output directory if it doesn't exist."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    return output


def figure_layer_ablation(output: Path) -> None:
    """Figure: Layer-wise ablation showing conv1 importance.

    Shows gap recovery (%) for each ablation mode on ResNet18/CIFAR-10.
    """
    # Data from experiments
    ablations = ["Full BitNet\n(baseline)", "keep_conv1", "keep_layer1", "keep_fc", "keep_layer4"]
    recovery = [0, 58, 20, 3, -3]  # Gap recovery percentages
    params_fp32 = [0, 0.08, 2.1, 0.5, 45.2]  # % params in FP32

    fig, ax = plt.subplots(figsize=(6, 4))

    # Create bars with color based on value
    bar_colors = [COLORS["bitnet"] if r <= 0 else COLORS["combo"] for r in recovery]
    bars = ax.bar(ablations, recovery, color=bar_colors, edgecolor="black", linewidth=0.5)

    # Highlight conv1 bar
    bars[1].set_color(COLORS["combo"])
    bars[1].set_edgecolor("black")
    bars[1].set_linewidth(1.5)

    # Add value labels on bars
    for bar, r, p in zip(bars, recovery, params_fp32):
        height = bar.get_height()
        y_pos = height + 2 if height >= 0 else height - 6
        ax.annotate(
            f"{r}%\n({p}% params)",
            xy=(bar.get_x() + bar.get_width() / 2, y_pos),
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=8,
            fontweight="bold" if r == 58 else "normal",
        )

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_ylabel("Gap Recovery (%)")
    ax.set_title("Layer-wise Ablation: Which Layer Matters Most?", fontweight="bold")
    ax.set_ylim(-15, 75)

    # Add annotation for the key finding
    ax.annotate(
        "conv1 alone recovers\n58% of the gap\nwith only 0.08% params",
        xy=(1, 58),
        xytext=(2.5, 55),
        fontsize=9,
        ha="left",
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output / "layer_ablation.pdf")
    plt.savefig(output / "layer_ablation.png")
    plt.close()
    print("  - layer_ablation.pdf")


def figure_recipe_progression(output: Path) -> None:
    """Figure: Recipe progression showing mixed-precision benefits (NO KD).

    Shows accuracy improvement from FP32 baseline → BitNet → BitNet+keep_conv1.
    Updated to reflect KD negative results - recipe is mixed-precision only.
    """
    datasets = ["CIFAR-10", "CIFAR-100", "Tiny-ImageNet"]

    # Accuracies from Wave 1 experiments (ResNet-18, basic augmentation)
    # FP32 baseline (std_s42/123/456 avg), BitNet baseline (bit_s42/123/456 avg),
    # BitNet+keep_conv1 (bit_keep_conv1_s42/123/456 avg)
    fp32 = [96.07, 79.14, 67.04]  # FP32 ResNet-18 baselines
    bitnet = [94.64, 74.93, 62.10]  # BitNet baselines
    recipe = [95.07, 75.74, 62.29]  # BitNet + FP32 conv1 (NO KD)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=False)

    for i, (ax, ds, fp, bn, rec) in enumerate(zip(axes, datasets, fp32, bitnet, recipe)):
        # 3-step progression: FP32 → BitNet → Recipe
        methods = ["FP32\nBaseline", "BitNet\nBaseline", "BitNet+\nFP32 conv1"]
        values = [fp, bn, rec]
        colors = [COLORS["fp32"], COLORS["bitnet"], COLORS["combo"]]

        bars = ax.bar(methods, values, color=colors, edgecolor="black", linewidth=0.5)

        # Add value labels
        for bar, v in zip(bars, values):
            ax.annotate(
                f"{v:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        # Add horizontal line for FP32 reference
        ax.axhline(y=fp, color=COLORS["fp32"], linestyle="--", linewidth=1, alpha=0.7)

        ax.set_title(ds, fontweight="bold", fontsize=12)
        ax.set_ylabel("Accuracy (%)" if i == 0 else "")

        # Set y limits with some padding
        y_min = min(values) - 2
        y_max = max(values) + 2
        ax.set_ylim(y_min, y_max)

        # Rotate x labels slightly
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)

        # Add gap annotation for recipe vs FP32
        gap = fp - rec
        gap_str = f"Gap: {gap:.2f}%"
        ax.annotate(
            gap_str,
            xy=(2, rec - 0.5),
            ha="center",
            fontsize=8,
            fontweight="bold",
            color=COLORS["bitnet"] if gap > 2 else COLORS["combo"],
        )

    # Add overall title
    fig.suptitle("Recipe Progression: Mixed-Precision (FP32 conv1) Without KD", fontweight="bold", y=1.02, fontsize=14)

    plt.tight_layout()
    plt.savefig(output / "recipe_progression.pdf")
    plt.savefig(output / "recipe_progression.png")
    plt.close()
    print("  - recipe_progression.pdf")


def figure_gap_scaling(output: Path) -> None:
    """Figure: Gap scaling with task complexity.

    Shows how the accuracy gap grows with number of classes.
    """
    # Data (CIFAR-10, CIFAR-100, Tiny ImageNet only)
    datasets = ["CIFAR-10", "CIFAR-100", "Tiny\nImageNet"]
    classes = [10, 100, 200]
    gaps = [3.49, 4.34, 5.81]
    log_classes = np.log10(classes)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot with markers
    ax.plot(
        log_classes,
        gaps,
        "o-",
        color=COLORS["bitnet"],
        markersize=12,
        linewidth=2,
        markeredgecolor="black",
        markeredgewidth=1,
    )

    # Add dataset labels
    for x, y, ds, c in zip(log_classes, gaps, datasets, classes):
        offset = 1.2
        ax.annotate(
            f"{ds}\n({c} classes)\n{y:.1f}%",
            xy=(x, y),
            xytext=(x, y + offset),
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.9),
        )

    # Linear fit line in log scale (approximate)
    x_fit = np.linspace(0.8, 2.5, 100)
    y_fit = 1.5 + 2.0 * (x_fit - 1)  # Approximate linear in log scale
    ax.plot(x_fit, y_fit, "--", color="gray", alpha=0.5, linewidth=1)

    ax.set_xlabel("Number of Classes (log scale)")
    ax.set_ylabel("Accuracy Gap: FP32 - BitNet (%)")
    ax.set_title("Gap Scales with Task Complexity", fontweight="bold")

    # Custom x-axis
    ax.set_xticks(log_classes)
    ax.set_xticklabels([str(c) for c in classes])
    ax.set_xlim(0.7, 2.6)
    ax.set_ylim(0, 10)

    # Add annotation about scaling
    ax.annotate(
        "Gap grows with\ntask complexity",
        xy=(1.8, 3.5),
        fontsize=9,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output / "gap_scaling.pdf")
    plt.savefig(output / "gap_scaling.png")
    plt.close()
    print("  - gap_scaling.pdf")


def figure_recipe_vs_fp32(output: Path) -> None:
    """Figure: Recipe comparison showing it exceeds FP32 on harder tasks.

    Key insight: Recipe exceeds FP32 on CIFAR-100 and Tiny ImageNet.
    """
    datasets = ["CIFAR-10\n(10 classes)", "CIFAR-100\n(100 classes)", "Tiny ImageNet\n(200 classes)"]
    fp32 = [88.89, 62.40, 54.85]
    recipe = [88.48, 63.40, 56.15]
    delta = [r - f for r, f in zip(recipe, fp32)]

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars_fp32 = ax.bar(
        x - width / 2, fp32, width, label="FP32 Baseline", color=COLORS["fp32"], edgecolor="black", linewidth=0.5
    )
    bars_recipe = ax.bar(
        x + width / 2,
        recipe,
        width,
        label="BitNet + conv1 + KD",
        color=COLORS["combo"],
        edgecolor="black",
        linewidth=0.5,
    )

    # Add value labels
    for bar, v in zip(bars_fp32, fp32):
        ax.annotate(
            f"{v:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom", fontsize=9
        )

    for bar, v, d in zip(bars_recipe, recipe, delta):
        # Main value
        ax.annotate(
            f"{v:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

        # Delta annotation
        delta_str = f"+{d:.1f}%" if d > 0 else f"{d:.1f}%"
        delta_color = COLORS["combo"] if d > 0 else COLORS["bitnet"]
        ax.annotate(
            delta_str,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=delta_color,
        )

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Recipe Exceeds FP32 on Harder Tasks", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc="upper right")

    # Set y-axis to show comparison clearly
    y_min = min(min(fp32), min(recipe)) - 5
    y_max = max(max(fp32), max(recipe)) + 6
    ax.set_ylim(y_min, y_max)

    # Add key insight box (positioned in middle-right area)
    ax.annotate(
        "Key Finding:\nHarder tasks benefit more\nfrom KD regularization",
        xy=(0.98, 0.5),
        xycoords="axes fraction",
        fontsize=9,
        ha="right",
        va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", edgecolor="black", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(output / "recipe_vs_fp32.pdf")
    plt.savefig(output / "recipe_vs_fp32.png")
    plt.close()
    print("  - recipe_vs_fp32.pdf")


def figure_augmentation_paradox(output: Path) -> None:
    """Figure: The Augmentation Paradox.

    Shows that the gap remains constant regardless of augmentation strategy.
    """
    augmentations = ["Basic", "Cutout", "RandAug", "Full"]

    # Data: FP32 and BitNet accuracies for ResNet18/CIFAR-10
    fp32_acc = [88.89, 89.21, 89.45, 89.67]
    bit_acc = [85.40, 85.74, 85.92, 86.18]
    gaps = [f - b for f, b in zip(fp32_acc, bit_acc)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Accuracy comparison
    x = np.arange(len(augmentations))
    width = 0.35

    ax1.bar(x - width / 2, fp32_acc, width, label="FP32", color=COLORS["fp32"], edgecolor="black", linewidth=0.5)
    ax1.bar(x + width / 2, bit_acc, width, label="BitNet", color=COLORS["bitnet"], edgecolor="black", linewidth=0.5)

    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Both Improve with Augmentation", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(augmentations)
    ax1.legend()
    ax1.set_ylim(84, 91)

    # Right: Gap remains constant
    ax2.bar(augmentations, gaps, color=COLORS["neutral"], edgecolor="black", linewidth=0.5)

    # Add value labels
    for i, (aug, g) in enumerate(zip(augmentations, gaps)):
        ax2.annotate(f"{g:.2f}%", xy=(i, g), ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Reference line at mean gap
    mean_gap = np.mean(gaps)
    ax2.axhline(y=mean_gap, color="red", linestyle="--", linewidth=2, label=f"Mean gap: {mean_gap:.2f}%")

    ax2.set_ylabel("Accuracy Gap: FP32 - BitNet (%)")
    ax2.set_title("...But the Gap Stays Constant", fontweight="bold")
    ax2.legend()
    ax2.set_ylim(0, 5)

    # Overall title
    fig.suptitle("The Augmentation Paradox: Why Data Augmentation Doesn't Help", fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(output / "augmentation_paradox.pdf")
    plt.savefig(output / "augmentation_paradox.png")
    plt.close()
    print("  - augmentation_paradox.pdf")


def figure_kd_benefit_by_task(output: Path) -> None:
    """Figure: KD benefit increases with task difficulty.

    Shows that KD recovers more of the gap on harder tasks.
    """
    datasets = ["CIFAR-10", "CIFAR-100"]
    kd_recovery = [36, 57]  # % gap recovery from KD alone
    conv1_recovery = [58, 74]  # % gap recovery from conv1 alone
    combo_recovery = [88, 123]  # % gap recovery from combo

    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(6, 4))

    bars1 = ax.bar(x - width, kd_recovery, width, label="KD only", color=COLORS["kd"], edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(
        x, conv1_recovery, width, label="conv1 only", color=COLORS["conv1"], edgecolor="black", linewidth=0.5
    )
    bars3 = ax.bar(
        x + width, combo_recovery, width, label="conv1 + KD", color=COLORS["combo"], edgecolor="black", linewidth=0.5
    )

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.axhline(y=100, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="100% = FP32 baseline")

    ax.set_ylabel("Gap Recovery (%)")
    ax.set_title("KD Benefit Scales with Task Difficulty", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc="upper left")
    ax.set_ylim(0, 140)

    # Annotation
    ax.annotate(
        "CIFAR-100: Recipe\nexceeds FP32!",
        xy=(1 + width, 123),
        xytext=(1.3, 110),
        fontsize=9,
        ha="left",
        arrowprops=dict(arrowstyle="->", color="black"),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output / "kd_benefit_by_task.pdf")
    plt.savefig(output / "kd_benefit_by_task.png")
    plt.close()
    print("  - kd_benefit_by_task.pdf")


def generate_all_figures() -> None:
    """Generate all publication-quality figures."""
    output = create_output_dir()

    print("Generating publication-quality figures...")

    figure_augmentation_paradox(output)
    figure_layer_ablation(output)
    figure_recipe_progression(output)
    figure_gap_scaling(output)
    figure_recipe_vs_fp32(output)
    figure_kd_benefit_by_task(output)

    print(f"\nAll figures saved to {output}/")
    print("\nRecommended figure placement in paper:")
    print("  - Fig 1: augmentation_paradox.pdf (Section 4)")
    print("  - Fig 2: layer_ablation.pdf (Section 5.1)")
    print("  - Fig 3: recipe_progression.pdf (Section 5.3)")
    print("  - Fig 4: recipe_vs_fp32.pdf (Section 5.4)")
    print("  - Fig 5: gap_scaling.pdf (Discussion)")
    print("  - Fig 6: kd_benefit_by_task.pdf (Optional/Appendix)")


if __name__ == "__main__":
    generate_all_figures()
