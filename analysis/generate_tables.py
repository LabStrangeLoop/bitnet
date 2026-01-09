"""Generate LaTeX tables for paper."""

from pathlib import Path

import pandas as pd


def accuracy_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table comparing standard vs bit accuracy by model and dataset."""
    versions = df["version"].unique()
    if len(versions) < 2:
        return f"% Insufficient data: only {list(versions)} version(s) available"

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

    (output / "accuracy.tex").write_text(accuracy_table(df))
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
