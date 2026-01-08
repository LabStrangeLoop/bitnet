"""Aggregate results from all experiments into a single DataFrame."""

import json
from pathlib import Path

import pandas as pd


def load_results(results_dir: str = "results/raw") -> pd.DataFrame:
    """Load all results.json files from experiment directories."""
    results_path = Path(results_dir)
    rows = []

    for run_dir in results_path.iterdir():
        if not run_dir.is_dir():
            continue

        results_file = run_dir / "results.json"
        if not results_file.exists():
            continue

        with open(results_file) as f:
            data = json.load(f)

        config = data.get("config", {})
        rows.append({
            "model": config.get("model"),
            "dataset": config.get("dataset"),
            "seed": config.get("seed"),
            "bit_version": config.get("bit_version", False),
            "version": "bit" if config.get("bit_version") else "std",
            "best_acc": data.get("best_acc"),
            "final_test_acc": data.get("final_test_acc"),
            "epochs": config.get("epochs"),
            "lr": config.get("lr"),
            "run_dir": str(run_dir),
        })

    return pd.DataFrame(rows)


def get_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std across seeds for each model/dataset/version combination."""
    return df.groupby(["model", "dataset", "version"]).agg({
        "best_acc": ["mean", "std", "count"],
        "final_test_acc": ["mean", "std"],
    }).round(2)


def save_aggregated(
    df: pd.DataFrame, output_path: str = "results/processed/aggregated.csv"
) -> None:
    """Save aggregated results to CSV."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved to {output}")


if __name__ == "__main__":
    df = load_results()
    print(f"Loaded {len(df)} experiment results")
    print(df.head())

    if len(df) > 0:
        summary = get_summary(df)
        print("\nSummary:")
        print(summary)

        save_aggregated(df)
