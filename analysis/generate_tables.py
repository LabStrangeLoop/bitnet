"""Generate LaTeX tables for paper."""

from pathlib import Path

import pandas as pd

from analysis.model_stats import get_model_stats
from experiments.config import MODELS
from experiments.datasets.factory import AUGMENT_CHOICES


def accuracy_table(df: pd.DataFrame, augment: str = "basic") -> str:
    """Generate LaTeX table comparing standard vs bit accuracy by model and dataset."""
    if "augment" in df.columns:
        df = df[df["augment"] == augment]
    if "ablation" in df.columns:
        df = df[df["ablation"] == "none"]

    versions = df["version"].unique()
    if len(versions) < 2:
        return f"% Insufficient data: only {list(versions)} version(s) available"

    # Check if TTQ data exists
    has_ttq = "ttq" in versions

    pivot = df.pivot_table(
        values="best_acc",
        index=["model", "dataset"],
        columns="version",
        aggfunc=["mean", "std"],
    ).round(2)

    # Build table header dynamically based on available versions
    if has_ttq:
        caption = rf"Test accuracy (\%) for standard, BitNet, and TTQ models ({augment} augmentation)"
        tabular_cols = "llcccccc"
        header = r"Model & Dataset & Std (mean) & Std (std) & Bit (mean) & Bit (std) & TTQ (mean) & TTQ (std) \\"
    else:
        caption = rf"Test accuracy (\%) for standard and 1.58-bit models ({augment} augmentation)"
        tabular_cols = "llcccc"
        header = r"Model & Dataset & Std (mean) & Std (std) & Bit (mean) & Bit (std) \\"

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:accuracy_{augment}}}",
        rf"\begin{{tabular}}{{{tabular_cols}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    for idx, row in pivot.iterrows():
        model, dataset = idx  # type: ignore[misc]
        std_mean = row.get(("mean", "std"), float("nan"))
        std_std = row.get(("std", "std"), float("nan"))
        bit_mean = row.get(("mean", "bit"), float("nan"))
        bit_std = row.get(("std", "bit"), float("nan"))

        std_mean_str = f"{std_mean:.2f}" if pd.notna(std_mean) else "-"
        std_std_str = f"{std_std:.2f}" if pd.notna(std_std) else "-"
        bit_mean_str = f"{bit_mean:.2f}" if pd.notna(bit_mean) else "-"
        bit_std_str = f"{bit_std:.2f}" if pd.notna(bit_std) else "-"

        cols = [model, dataset, std_mean_str, std_std_str, bit_mean_str, bit_std_str]

        if has_ttq:
            ttq_mean = row.get(("mean", "ttq"), float("nan"))
            ttq_std = row.get(("std", "ttq"), float("nan"))
            ttq_mean_str = f"{ttq_mean:.2f}" if pd.notna(ttq_mean) else "-"
            ttq_std_str = f"{ttq_std:.2f}" if pd.notna(ttq_std) else "-"
            cols.extend([ttq_mean_str, ttq_std_str])

        lines.append(" & ".join(cols) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def augmentation_ablation_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table showing accuracy gap across augmentation levels."""
    if "augment" not in df.columns:
        return "% No augmentation data available"

    augments = [a for a in AUGMENT_CHOICES if a in df["augment"].unique()]
    if len(augments) < 2:
        return f"% Insufficient augmentation data: only {augments} available"

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Accuracy gap (FP32 - BitNet) across augmentation levels}",
        r"\label{tab:augmentation_ablation}",
        r"\begin{tabular}{ll" + "c" * len(augments) + "}",
        r"\toprule",
        "Model & Dataset & " + " & ".join(augments) + r" \\",
        r"\midrule",
    ]

    # Exclude ablation experiments from gap calculation
    if "ablation" in df.columns:
        df = df[df["ablation"] == "none"]

    for (model, dataset), group in df.groupby(["model", "dataset"]):
        row_vals = [str(model), str(dataset)]
        for augment in augments:
            aug_data = group[group["augment"] == augment]
            if len(aug_data) == 0:
                row_vals.append("-")
                continue

            std_acc = aug_data[aug_data["version"] == "std"]["best_acc"].mean()
            bit_acc = aug_data[aug_data["version"] == "bit"]["best_acc"].mean()
            gap = std_acc - bit_acc
            row_vals.append(f"{gap:.2f}" if pd.notna(gap) else "-")

        lines.append(" & ".join(row_vals) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def statistical_table(comparisons: pd.DataFrame) -> str:
    """Generate LaTeX table with statistical test results."""
    valid = comparisons[~comparisons.get("error", pd.Series([False] * len(comparisons))).notna()]
    if len(valid) == 0:
        return "% No valid statistical comparisons available"

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Statistical comparison of standard vs 1.58-bit models}",
        r"\label{tab:statistics}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Model & Dataset & $\Delta$ Acc & $t$ & $p$ & Cohen's $d$ \\",
        r"\midrule",
    ]

    for _, row in valid.iterrows():
        sig = "*" if row["significant"] else ""
        lines.append(
            f"{row['model']} & {row['dataset']} & {row['diff']:.2f} & "
            f"{row['t_stat']:.2f} & {row['p_value']:.3f}{sig} & {row['cohens_d']:.2f} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\multicolumn{6}{l}{\footnotesize * $p < 0.05$} \\",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def efficiency_table(input_size: tuple[int, int] = (32, 32)) -> str:
    """Generate LaTeX table with model efficiency metrics (size, FLOPs, compression)."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Model efficiency: memory and computational cost}",
        r"\label{tab:efficiency}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & Params & FP32 (MB) & BitNet (MB) & Compress & FLOPs & Speedup \\",
        r"\midrule",
    ]

    for model_name in MODELS:
        try:
            stats = get_model_stats(model_name, num_classes=10, input_size=input_size)
            compress = stats["fp32_mb"] / stats["bitnet_mb"]
            lines.append(
                f"{model_name} & {stats['params_str']} & {stats['fp32_mb']:.2f} & "
                f"{stats['bitnet_mb']:.2f} & {compress:.1f}$\\times$ & "
                f"{stats['flops_str']} & {stats['compute_reduction']:.0f}$\\times$ \\\\"
            )
        except Exception:
            continue

    lines.extend(
        [
            r"\bottomrule",
            r"\multicolumn{7}{l}{\footnotesize Speedup assumes 64 ternary ops = 1 FP64 op} \\",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def layer_ablation_table(df: pd.DataFrame, dataset: str = "cifar10", augment: str = "basic") -> str:
    """Generate LaTeX table showing layer-wise ablation results for a specific dataset."""
    if "ablation" not in df.columns:
        return "% No ablation data available"

    # Filter by dataset and augmentation (ablation experiments use basic augment)
    df = df[(df["dataset"] == dataset) & (df["augment"] == augment)]

    # Filter to BitNet runs with ablation
    ablation_df = df[(df["version"] == "bit") | (df["ablation"] != "none")]
    if len(ablation_df) == 0:
        return "% No ablation experiments found"

    # Get FP32 baseline
    fp32_df = df[(df["version"] == "std") & (df["ablation"] == "none")]

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{Layer-wise ablation on {dataset.upper()}: accuracy when keeping specific layers in FP32}}",
        rf"\label{{tab:layer_ablation_{dataset}}}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Model & Ablation & Accuracy (\%) & Gap Recovery \\",
        r"\midrule",
    ]

    for model in ablation_df["model"].unique():
        model_data = ablation_df[ablation_df["model"] == model]
        fp32_acc = fp32_df[fp32_df["model"] == model]["best_acc"].mean()
        bitnet_baseline = model_data[model_data["ablation"] == "none"]["best_acc"].mean()
        total_gap = fp32_acc - bitnet_baseline if pd.notna(fp32_acc) else 0

        for ablation in ["none", "keep_conv1", "keep_layer1", "keep_layer4", "keep_fc"]:
            abl_data = model_data[model_data["ablation"] == ablation]
            if len(abl_data) == 0:
                continue

            acc = abl_data["best_acc"].mean()
            if ablation == "none":
                recovery = "0\\%"
                label = "Full BitNet"
            else:
                gap_recovered = acc - bitnet_baseline
                recovery_pct = 100 * gap_recovered / total_gap if total_gap > 0 else 0
                recovery = f"{recovery_pct:.0f}\\%"
                label = ablation.replace("_", "\\_")

            lines.append(f"{model} & {label} & {acc:.2f} & {recovery} \\\\")

        # Add FP32 baseline row
        if pd.notna(fp32_acc):
            lines.append(f"{model} & FP32 baseline & {fp32_acc:.2f} & 100\\% \\\\")
        lines.append(r"\midrule")

    # Remove last midrule
    if lines[-1] == r"\midrule":
        lines.pop()

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def kd_statistics_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table with paired t-tests for KD failure (BitNet vs BitNet+KD)."""
    from analysis.statistical_analysis import kd_effect_ttest

    configs = [
        ("resnet18", "cifar10"),
        ("resnet18", "cifar100"),
        ("resnet18", "tiny-imagenet"),
        ("resnet50", "cifar10"),
        ("resnet50", "cifar100"),
        ("resnet50", "tiny-imagenet"),
    ]

    results = []
    for model, dataset in configs:
        stats = kd_effect_ttest(df, model, dataset, ablation="none")
        if "error" not in stats:
            results.append(stats)

    if len(results) == 0:
        return "% No KD comparison data available"

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Paired t-tests: BitNet vs BitNet+KD (Knowledge Distillation degradation)}",
        r"\label{tab:kd_statistics}",
        r"\begin{tabular}{llrrrrc}",
        r"\toprule",
        r"Model & Dataset & BitNet & +KD & $\Delta$ & $p$ & Cohen's $d$ \\",
        r"\midrule",
    ]

    for r in results:
        sig = "*" if r["significant"] else ""
        lines.append(
            f"{r['model']} & {r['dataset']} & {r['baseline_mean']:.2f} & "
            f"{r['kd_mean']:.2f} & {r['mean_diff']:+.2f} & "
            f"{r['p_value']:.4f}{sig} & {r['cohens_d']:.2f} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\multicolumn{7}{l}{\footnotesize * $p < 0.05$} \\",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def save_tables(df: pd.DataFrame, comparisons: pd.DataFrame, output_dir: str = "paper/tmlr/tables") -> None:
    """Save all tables to files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    (output / "accuracy_basic.tex").write_text(accuracy_table(df, augment="basic"))
    (output / "accuracy_full.tex").write_text(accuracy_table(df, augment="full"))
    (output / "augmentation_ablation.tex").write_text(augmentation_ablation_table(df))
    (output / "statistics.tex").write_text(statistical_table(comparisons))
    (output / "kd_statistics.tex").write_text(kd_statistics_table(df))
    (output / "efficiency.tex").write_text(efficiency_table())
    (output / "layer_ablation.tex").write_text(layer_ablation_table(df, dataset="cifar10"))
    print(f"Tables saved to {output}")


if __name__ == "__main__":
    from analysis.aggregate_results import load_results
    from analysis.statistical_analysis import run_all_comparisons

    # Load from both regular and KD experiment directories
    df = load_results(["results/raw", "results/raw_kd"])
    if len(df) == 0:
        print("No results found")
        raise SystemExit(0)

    comparisons = run_all_comparisons(df)
    save_tables(df, comparisons)
