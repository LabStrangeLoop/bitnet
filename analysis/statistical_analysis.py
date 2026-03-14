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


def kd_effect_ttest(df: pd.DataFrame, model: str, dataset: str, ablation: str = "none") -> dict:
    """
    Paired t-test: BitNet vs BitNet+KD (isolate KD effect).

    Compares:
    - BitNet baseline (bit=True, ablation=ablation, kd_alpha=NaN)
    - BitNet+KD (bit=True, ablation=ablation, kd_alpha=0.9)

    Paired by seed [42, 123, 456].
    """
    import numpy as np

    # Filter to target model, dataset, ablation, and bit version
    mask = (df["model"] == model) & (df["dataset"] == dataset) & (df["ablation"] == ablation) & df["bit_version"]
    subset = df[mask].copy()

    # Separate non-KD vs KD (handle NaN for kd_alpha)
    no_kd = subset[subset["kd_alpha"].isna()].sort_values("seed")
    with_kd = subset[subset["kd_alpha"] == 0.9].sort_values("seed")

    if len(no_kd) != len(with_kd) or len(no_kd) < 2:
        return {
            "model": model,
            "dataset": dataset,
            "error": f"Mismatched data: {len(no_kd)} non-KD, {len(with_kd)} KD seeds",
        }

    # Verify seeds match for pairing
    if not (no_kd["seed"].values == with_kd["seed"].values).all():
        return {"model": model, "dataset": dataset, "error": "Seeds do not match for pairing"}

    # Paired t-test
    baseline_accs = no_kd["best_acc"].values
    kd_accs = with_kd["best_acc"].values

    t_stat, p_value = stats.ttest_rel(baseline_accs, kd_accs)

    # Effect size (Cohen's d for paired samples)
    diff = baseline_accs - kd_accs
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else np.inf

    return {
        "model": model,
        "dataset": dataset,
        "baseline_mean": np.mean(baseline_accs),
        "baseline_std": np.std(baseline_accs, ddof=1),
        "kd_mean": np.mean(kd_accs),
        "kd_std": np.std(kd_accs, ddof=1),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_stat": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "significant": p_value < 0.05,
        "n": len(baseline_accs),
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
