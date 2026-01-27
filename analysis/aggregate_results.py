"""Aggregate results from all experiments into a single DataFrame."""

import json
from pathlib import Path

import pandas as pd


def load_results(results_dir: str = "results/raw") -> pd.DataFrame:
    """Load all results.json files from experiment directories.

    Supports both flat and hierarchical directory structures:
    - Flat: results/raw/{run_name}/results.json
    - Hierarchical: results/raw/{dataset}/{model}/{run_name}/results.json
    """
    results_path = Path(results_dir)
    rows = []

    for results_file in results_path.rglob("results.json"):
        with open(results_file) as f:
            data = json.load(f)

        # Config may be embedded in results.json or in separate config.json
        config = data.get("config", {})
        if not config:
            config_file = results_file.parent / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
        # Handle both old format (bit_version: bool) and new format (version: str)
        if "version" in config:
            version = config["version"]
        else:
            version = "bit" if config.get("bit_version") else "std"
        rows.append(
            {
                "model": config.get("model"),
                "dataset": config.get("dataset"),
                "seed": config.get("seed"),
                "bit_version": version == "bit",
                "version": version,
                "augment": config.get("augment", "basic"),
                "ablation": config.get("ablation", "none"),
                "best_acc": data.get("best_acc"),
                "final_test_acc": data.get("final_test_acc"),
                "epochs": config.get("epochs"),
                "lr": config.get("lr"),
                "run_dir": str(results_file.parent),
            }
        )

    return pd.DataFrame(rows)


def get_summary(df: pd.DataFrame, include_augment: bool = True) -> pd.DataFrame:
    """Compute mean and std across seeds for each model/dataset/version/augment combination."""
    group_cols = ["model", "dataset", "version"]
    if include_augment and "augment" in df.columns:
        group_cols.append("augment")

    return (
        df.groupby(group_cols)
        .agg(
            {
                "best_acc": ["mean", "std", "count"],
                "final_test_acc": ["mean", "std"],
            }
        )
        .round(2)
    )


def compute_accuracy_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy gap (std - bit) for each model/dataset/augment combination."""
    group_cols = ["model", "dataset", "augment"]
    available_cols = [c for c in group_cols if c in df.columns]

    std_df = df[df["version"] == "std"].groupby(available_cols)["best_acc"].mean()
    bit_df = df[df["version"] == "bit"].groupby(available_cols)["best_acc"].mean()

    gap = (std_df - bit_df).reset_index()
    gap.columns = [*available_cols, "gap"]
    return gap


def save_aggregated(df: pd.DataFrame, output_path: str = "results/processed/aggregated.csv") -> None:
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

        gap = compute_accuracy_gap(df)
        print("\nAccuracy Gap (std - bit):")
        print(gap)

        save_aggregated(df)
