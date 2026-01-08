"""Generate figures for paper."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def accuracy_comparison_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Bar plot comparing standard vs bit accuracy for each model/dataset."""
    _, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    datasets = df["dataset"].unique()

    for ax, dataset in zip(axes, datasets):
        data = df[df["dataset"] == dataset]
        pivot = data.pivot_table(
            values="best_acc", index="model", columns="version", aggfunc="mean"
        )

        pivot.plot(kind="bar", ax=ax, rot=45)
        ax.set_title(dataset.upper())
        ax.set_xlabel("")
        ax.set_ylabel("Accuracy (%)" if ax == axes[0] else "")
        ax.legend(title="Version")

    plt.tight_layout()
    plt.savefig(output_path / "accuracy_comparison.pdf", bbox_inches="tight")
    plt.savefig(output_path / "accuracy_comparison.png", bbox_inches="tight", dpi=150)
    plt.close()


def training_curves_plot(
    df: pd.DataFrame,
    output_path: Path,  # noqa: ARG001
) -> None:
    """Plot training curves from history (if available in results)."""
    # Placeholder - implement when results with history are available
    _ = df, output_path


def accuracy_delta_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Heatmap showing accuracy difference (std - bit) for each model/dataset."""
    pivot_std = df[df["version"] == "std"].pivot_table(
        values="best_acc", index="model", columns="dataset", aggfunc="mean"
    )
    pivot_bit = df[df["version"] == "bit"].pivot_table(
        values="best_acc", index="model", columns="dataset", aggfunc="mean"
    )
    delta = pivot_std - pivot_bit

    plt.figure(figsize=(8, 6))
    sns.heatmap(delta, annot=True, fmt=".2f", cmap="RdYlGn_r", center=0)
    plt.title("Accuracy Drop: Standard - Bit (%)")
    plt.tight_layout()
    plt.savefig(output_path / "accuracy_delta_heatmap.pdf", bbox_inches="tight")
    plt.savefig(output_path / "accuracy_delta_heatmap.png", bbox_inches="tight", dpi=150)
    plt.close()


def save_figures(df: pd.DataFrame, output_dir: str = "paper/figures") -> None:
    """Generate and save all figures."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    accuracy_comparison_plot(df, output)
    accuracy_delta_heatmap(df, output)
    print(f"Figures saved to {output}")


if __name__ == "__main__":
    from analysis.aggregate_results import load_results

    df = load_results()
    if len(df) > 0:
        save_figures(df)
