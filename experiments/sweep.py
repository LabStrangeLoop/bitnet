"""Run systematic experiment sweeps across models, datasets, and seeds."""

import argparse
import itertools
import signal
import subprocess
import sys
from pathlib import Path

from experiments.config import DATASETS, MODELS, SEEDS
from experiments.datasets.factory import AUGMENT_CHOICES

VERSIONS = [False, True]  # standard, bit

# Global state for interrupt handler
_sweep_state = {"completed": 0, "failed": 0, "skipped": 0, "total": 0}


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
    config: dict,
    output_dir: str,
    epochs: int,
    index: int,
    total: int,
    dry_run: bool = False,
) -> str:
    """Run a single experiment. Returns 'completed', 'skipped', or 'failed'."""
    version = "bit" if config["bit_version"] else "std"
    aug_suffix = f"_{config['augment']}" if config["augment"] != "basic" else ""
    run_name = f"{config['model']}_{version}_{config['dataset']}{aug_suffix}_s{config['seed']}"

    # Check hierarchical output structure
    run_dir = (
        Path(output_dir)
        / config["dataset"]
        / config["model"]
        / f"{version}{aug_suffix}_s{config['seed']}"
    )

    prefix = f"[{index}/{total}]"

    if (run_dir / "results.json").exists():
        print(f"{prefix} Skipping {run_name} (already completed)")
        return "skipped"

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
        "--quiet",
    ]
    if config["bit_version"]:
        cmd.append("--bit-version")

    print(f"{prefix} Running {run_name}...", end=" ", flush=True)
    if dry_run:
        print(f"\n  Command: {' '.join(cmd)}")
        return "completed"

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        # Extract final accuracy from output
        for line in result.stdout.splitlines()[::-1]:
            if "Best accuracy" in line:
                print(line.split("Best accuracy:")[-1].strip())
                break
        else:
            print("done")
        return "completed"
    else:
        print("FAILED")
        if result.stderr:
            print(f"  Error: {result.stderr.splitlines()[-1]}")
        return "failed"


def print_summary() -> None:
    """Print sweep summary."""
    state = _sweep_state
    print(f"\n{'=' * 40}")
    print(
        f"Sweep Summary: {state['completed']} completed, "
        f"{state['skipped']} skipped, {state['failed']} failed"
    )
    print(f"{'=' * 40}")


def handle_interrupt(_signum: int, _frame: object) -> None:
    """Handle Ctrl+C gracefully."""
    print("\n\nInterrupted by user.")
    print_summary()
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/raw")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--augments", nargs="+", default=["basic"], choices=AUGMENT_CHOICES)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, handle_interrupt)

    configs = [
        c
        for c in get_experiment_configs(args.augments)
        if c["model"] in args.models and c["dataset"] in args.datasets and c["seed"] in args.seeds
    ]

    total = len(configs)
    _sweep_state["total"] = total

    print(f"Total experiments: {total}")
    print(f"Models: {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"Augments: {args.augments}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")
    print()

    for i, config in enumerate(configs, 1):
        result = run_experiment(config, args.output_dir, args.epochs, i, total, args.dry_run)
        _sweep_state[result] += 1

    print_summary()


if __name__ == "__main__":
    main()
