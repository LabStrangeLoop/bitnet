"""Run systematic experiment sweeps across models, datasets, and seeds."""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

from experiments.config import DATASETS, MODELS, SEEDS
from experiments.datasets.factory import AUGMENT_CHOICES

VERSIONS = [False, True]  # standard, bit


def get_experiment_configs(augments: list[str]):
    """Generate all experiment configurations."""
    for model, dataset, seed, bit_version, augment in itertools.product(
        MODELS, DATASETS, SEEDS, VERSIONS, augments
    ):
        yield {
            "model": model,
            "dataset": dataset,
            "seed": seed,
            "bit_version": bit_version,
            "augment": augment,
        }


def run_experiment(
    config: dict, output_dir: str, epochs: int, dry_run: bool = False
) -> bool:
    """Run a single experiment."""
    version = "bit" if config["bit_version"] else "std"
    aug_suffix = f"_{config['augment']}" if config["augment"] != "basic" else ""
    run_name = f"{config['model']}_{version}_{config['dataset']}{aug_suffix}_s{config['seed']}"
    run_dir = Path(output_dir) / run_name

    if (run_dir / "results.json").exists():
        print(f"Skipping {run_name} (already completed)")
        return True

    cmd = [
        sys.executable,
        "-m",
        "experiments.train",
        "--model",
        config["model"],
        "--dataset",
        config["dataset"],
        "--seed",
        str(config["seed"]),
        "--epochs",
        str(epochs),
        "--augment",
        config["augment"],
        "--output-dir",
        output_dir,
    ]
    if config["bit_version"]:
        cmd.append("--bit-version")

    print(f"Running: {run_name}")
    if dry_run:
        print(f"  Command: {' '.join(cmd)}")
        return True

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/raw")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument(
        "--augments", nargs="+", default=["basic"], choices=AUGMENT_CHOICES
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    configs = [
        c
        for c in get_experiment_configs(args.augments)
        if c["model"] in args.models
        and c["dataset"] in args.datasets
        and c["seed"] in args.seeds
    ]

    print(f"Total experiments: {len(configs)}")
    print(f"Models: {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"Augments: {args.augments}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")
    print()

    completed, failed = 0, 0
    for config in configs:
        success = run_experiment(config, args.output_dir, args.epochs, args.dry_run)
        if success:
            completed += 1
        else:
            failed += 1

    print(f"\nCompleted: {completed}, Failed: {failed}")


if __name__ == "__main__":
    main()
