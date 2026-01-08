"""Statistical analysis: t-tests and effect sizes for comparing standard vs bit models."""

import pandas as pd
from scipy import stats


def paired_ttest(df: pd.DataFrame, model: str, dataset: str) -> dict:
    """Run paired t-test comparing standard vs bit versions across seeds."""
    std_results = df[(df["model"] == model) & (df["dataset"] == dataset) & (df["version"] == "std")]
    bit_results = df[(df["model"] == model) & (df["dataset"] == dataset) & (df["version"] == "bit")]

    std_acc = std_results.sort_values("seed")["best_acc"].values
    bit_acc = bit_results.sort_values("seed")["best_acc"].values

    if len(std_acc) < 2 or len(bit_acc) < 2 or len(std_acc) != len(bit_acc):
        return {"model": model, "dataset": dataset, "error": "insufficient data"}

    t_stat, p_value = stats.ttest_rel(std_acc, bit_acc)
    diff = std_acc.mean() - bit_acc.mean()
    pooled_std = ((std_acc.std() ** 2 + bit_acc.std() ** 2) / 2) ** 0.5
    cohens_d = diff / pooled_std if pooled_std > 0 else 0

    return {
        "model": model,
        "dataset": dataset,
        "std_mean": std_acc.mean(),
        "bit_mean": bit_acc.mean(),
        "diff": diff,
        "t_stat": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "significant": p_value < 0.05,
    }


def run_all_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Run t-tests for all model/dataset combinations."""
    results = []
    for model in df["model"].unique():
        for dataset in df["dataset"].unique():
            results.append(paired_ttest(df, model, dataset))
    return pd.DataFrame(results)


if __name__ == "__main__":
    from analysis.aggregate_results import load_results

    df = load_results()
    if len(df) > 0:
        comparisons = run_all_comparisons(df)
        print(comparisons.to_string())
