"""Generate LaTeX tables for paper."""

from pathlib import Path

import pandas as pd


def accuracy_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table comparing standard vs bit accuracy by model and dataset."""
    pivot = df.pivot_table(
        values="best_acc",
        index=["model", "dataset"],
        columns="version",
        aggfunc=["mean", "std"],
    ).round(2)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Test accuracy (\%) for standard and 1.58-bit models}",
        r"\label{tab:accuracy}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Model & Dataset & Std (mean) & Std (std) & Bit (mean) & Bit (std) \\",
        r"\midrule",
    ]

    for idx, row in pivot.iterrows():
        model, dataset = idx  # type: ignore[misc]
        std_mean = row[("mean", "std")]
        std_std = row[("std", "std")]
        bit_mean = row[("mean", "bit")]
        bit_std = row[("std", "bit")]
        lines.append(
            f"{model} & {dataset} & {std_mean:.2f} & {std_std:.2f} & "
            f"{bit_mean:.2f} & {bit_std:.2f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def statistical_table(comparisons: pd.DataFrame) -> str:
    """Generate LaTeX table with statistical test results."""
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

    for _, row in comparisons.iterrows():
        if "error" in row and pd.notna(row.get("error")):
            continue
        sig = "*" if row["significant"] else ""
        lines.append(
            f"{row['model']} & {row['dataset']} & {row['diff']:.2f} & "
            f"{row['t_stat']:.2f} & {row['p_value']:.3f}{sig} & {row['cohens_d']:.2f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\multicolumn{6}{l}{\footnotesize * $p < 0.05$} \\",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def save_tables(
    df: pd.DataFrame, comparisons: pd.DataFrame, output_dir: str = "paper/tables"
) -> None:
    """Save all tables to files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    (output / "accuracy.tex").write_text(accuracy_table(df))
    (output / "statistics.tex").write_text(statistical_table(comparisons))
    print(f"Tables saved to {output}")


if __name__ == "__main__":
    from analysis.aggregate_results import load_results
    from analysis.statistical_analysis import run_all_comparisons

    df = load_results()
    if len(df) > 0:
        comparisons = run_all_comparisons(df)
        save_tables(df, comparisons)
