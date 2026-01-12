"""Generate LaTeX tables for paper."""

from pathlib import Path

import pandas as pd

from experiments.datasets.factory import AUGMENT_CHOICES


def accuracy_table(df: pd.DataFrame, augment: str = "basic") -> str:
    """Generate LaTeX table comparing standard vs bit accuracy by model and dataset."""
    if "augment" in df.columns:
        df = df[df["augment"] == augment]

    versions = df["version"].unique()
    if len(versions) < 2:
        return f"% Insufficient data: only {list(versions)} version(s) available"

    pivot = df.pivot_table(
        values="best_acc",
        index=["model", "dataset"],
        columns="version",
        aggfunc=["mean", "std"],
    ).round(2)

    caption = rf"Test accuracy (\%) for standard and 1.58-bit models ({augment} augmentation)"
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:accuracy_{augment}}}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Model & Dataset & Std (mean) & Std (std) & Bit (mean) & Bit (std) \\",
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


def save_tables(
    df: pd.DataFrame, comparisons: pd.DataFrame, output_dir: str = "paper/tables"
) -> None:
    """Save all tables to files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    (output / "accuracy_basic.tex").write_text(accuracy_table(df, augment="basic"))
    (output / "accuracy_full.tex").write_text(accuracy_table(df, augment="full"))
    (output / "augmentation_ablation.tex").write_text(augmentation_ablation_table(df))
    (output / "statistics.tex").write_text(statistical_table(comparisons))
    print(f"Tables saved to {output}")


if __name__ == "__main__":
    from analysis.aggregate_results import load_results
    from analysis.statistical_analysis import run_all_comparisons

    df = load_results()
    if len(df) == 0:
        print("No results found")
        raise SystemExit(0)

    comparisons = run_all_comparisons(df)
    save_tables(df, comparisons)
