"""Generate figures for paper."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from experiments.datasets.factory import AUGMENT_CHOICES


def accuracy_comparison_plot(df: pd.DataFrame, output_path: Path, augment: str = "basic") -> None:
    """Bar plot comparing standard vs bit accuracy for each model/dataset."""
    if "augment" in df.columns:
        df = df[df["augment"] == augment]

    datasets = sorted(df["dataset"].unique())
    n_datasets = len(datasets)
    if n_datasets == 0:
        return

    _, axes = plt.subplots(1, max(n_datasets, 1), figsize=(4 * n_datasets, 4), sharey=True)
    if n_datasets == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        data = df[df["dataset"] == dataset]
        pivot = data.pivot_table(values="best_acc", index="model", columns="version", aggfunc="mean")

        pivot.plot(kind="bar", ax=ax, rot=45)
        ax.set_title(dataset.upper())
        ax.set_xlabel("")
        ax.set_ylabel("Accuracy (%)" if ax == axes[0] else "")
        ax.legend(title="Version")

    plt.tight_layout()
    plt.savefig(output_path / f"accuracy_comparison_{augment}.pdf", bbox_inches="tight")
    plt.savefig(output_path / f"accuracy_comparison_{augment}.png", bbox_inches="tight", dpi=150)
    plt.close()


def accuracy_delta_heatmap(df: pd.DataFrame, output_path: Path, augment: str = "basic") -> None:
    """Heatmap showing accuracy difference (std - bit) for each model/dataset."""
    if "augment" in df.columns:
        df = df[df["augment"] == augment]

    pivot_std = df[df["version"] == "std"].pivot_table(
        values="best_acc", index="model", columns="dataset", aggfunc="mean"
    )
    pivot_bit = df[df["version"] == "bit"].pivot_table(
        values="best_acc", index="model", columns="dataset", aggfunc="mean"
    )
    delta = pivot_std - pivot_bit

    if delta.empty:
        return

    plt.figure(figsize=(8, 6))
    sns.heatmap(delta, annot=True, fmt=".2f", cmap="RdYlGn_r", center=0)
    plt.title(f"Accuracy Drop: Standard - Bit (%) [{augment}]")
    plt.tight_layout()
    plt.savefig(output_path / f"accuracy_delta_heatmap_{augment}.pdf", bbox_inches="tight")
    plt.savefig(output_path / f"accuracy_delta_heatmap_{augment}.png", bbox_inches="tight", dpi=150)
    plt.close()


def augmentation_gap_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Line plot showing how accuracy gap changes with augmentation level."""
    if "augment" not in df.columns:
        return

    augments = [a for a in AUGMENT_CHOICES if a in df["augment"].unique()]
    if len(augments) < 2:
        return

    # Compute gap for each model/dataset/augment
    gaps = []
    for (model, dataset, augment), group in df.groupby(["model", "dataset", "augment"]):
        std_acc = group[group["version"] == "std"]["best_acc"].mean()
        bit_acc = group[group["version"] == "bit"]["best_acc"].mean()
        if pd.notna(std_acc) and pd.notna(bit_acc):
            gaps.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "augment": augment,
                    "gap": std_acc - bit_acc,
                }
            )

    if not gaps:
        return

    gap_df = pd.DataFrame(gaps)

    # Create subplot for each dataset
    datasets = sorted(gap_df["dataset"].unique())
    n_datasets = len(datasets)

    _, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 4), sharey=True)
    if n_datasets == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        data = gap_df[gap_df["dataset"] == dataset]
        for model in data["model"].unique():
            model_data = data[data["model"] == model]
            # Sort by augment order
            model_data = model_data.set_index("augment").loc[
                [a for a in AUGMENT_CHOICES if a in model_data["augment"].values]
            ]
            ax.plot(model_data.index, model_data["gap"], marker="o", label=model)

        ax.set_title(dataset.upper())
        ax.set_xlabel("Augmentation Level")
        ax.set_ylabel("Accuracy Gap (FP32 - BitNet)" if ax == axes[0] else "")
        ax.legend()
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path / "augmentation_gap.pdf", bbox_inches="tight")
    plt.savefig(output_path / "augmentation_gap.png", bbox_inches="tight", dpi=150)
    plt.close()


def save_figures(df: pd.DataFrame, output_dir: str = "paper/tmlr/figures") -> None:
    """Generate and save all figures."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # Generate plots for each augmentation level present
    augments = df["augment"].unique() if "augment" in df.columns else ["basic"]
    for augment in augments:
        accuracy_comparison_plot(df, output, augment=augment)
        accuracy_delta_heatmap(df, output, augment=augment)

    # Augmentation ablation plot (exclude imagenet)
    df_no_imagenet = df[df["dataset"] != "imagenet"] if "dataset" in df.columns else df
    augmentation_gap_plot(df_no_imagenet, output)

    print(f"Figures saved to {output}")


if __name__ == "__main__":
    from analysis.aggregate_results import load_results

    df = load_results()
    if len(df) > 0:
        save_figures(df)
