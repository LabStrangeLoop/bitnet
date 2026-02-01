"""Run Knowledge Distillation experiments across multiple seeds."""

import argparse
import signal
import subprocess
import sys
from pathlib import Path

from experiments.config import DATASETS, DEFAULTS, MODELS, SEEDS

# Global state for interrupt handler
_sweep_state = {"completed": 0, "failed": 0, "skipped": 0, "total": 0}


def find_teacher_path(model: str, dataset: str, results_dir: Path, teacher_seed: int) -> Path | None:
    """Find the FP32 teacher checkpoint for a given model/dataset."""
    teacher_path = results_dir / dataset / model / f"std_s{teacher_seed}" / "best_model.pth"
    return teacher_path if teacher_path.exists() else None


def run_kd_experiment(
    model: str,
    dataset: str,
    seed: int,
    teacher_path: Path,
    output_dir: Path,
    epochs: int,
    temperature: float,
    alpha: float,
    index: int,
    total: int,
    dry_run: bool = False,
) -> str:
    """Run a single KD experiment. Returns 'completed', 'skipped', or 'failed'."""
    run_name = f"{model}_kd_{dataset}_s{seed}"
    run_dir = output_dir / dataset / model / f"bit_kd_s{seed}"
    prefix = f"[{index}/{total}]"

    if (run_dir / "results.json").exists():
        print(f"{prefix} Skipping {run_name} (already completed)")
        return "skipped"

    cmd = [
        sys.executable,
        "-m",
        "experiments.train_kd",
        "--model",
        model,
        "--dataset",
        dataset,
        "--teacher-path",
        str(teacher_path),
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--temperature",
        str(temperature),
        "--alpha",
        str(alpha),
        "--output-dir",
        str(run_dir),
        "--quiet",
    ]

    print(f"{prefix} Running {run_name} (teacher: {teacher_path.name})...", flush=True)
    if dry_run:
        print(f"  Command: {' '.join(cmd)}")
        return "completed"

    best_acc_line = None
    error_line = None
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            if "Progress:" in line or "Training complete" in line or "Skipping" in line:
                print(f"  {line}")
            if "Best accuracy" in line:
                best_acc_line = line
        proc.wait()
        if proc.returncode != 0 and proc.stderr:
            error_line = proc.stderr.read().strip().splitlines()[-1] if proc.stderr else None

    if proc.returncode == 0:
        if best_acc_line:
            print(f"  Done: {best_acc_line.split('Best accuracy:')[-1].strip()}")
        else:
            print("  Done")
        return "completed"
    else:
        print("  FAILED")
        if error_line:
            print(f"  Error: {error_line}")
        return "failed"


def print_summary() -> None:
    """Print sweep summary."""
    state = _sweep_state
    print(f"\n{'=' * 40}")
    print(f"KD Sweep: {state['completed']} completed, {state['skipped']} skipped, {state['failed']} failed")
    print(f"{'=' * 40}")


def handle_interrupt(_signum: int, _frame: object) -> None:
    """Handle Ctrl+C gracefully."""
    print("\n\nInterrupted by user.")
    print_summary()
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KD experiments across seeds")
    parser.add_argument("--output-dir", default="results/raw_kd")
    parser.add_argument("--results-dir", default="results/raw", help="Dir with FP32 teacher checkpoints")
    parser.add_argument("--epochs", type=int, default=DEFAULTS.epochs)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Student seeds")
    parser.add_argument("--teacher-seed", type=int, default=42, help="Teacher seed (default: 42)")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, handle_interrupt)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    # Build experiment list
    experiments = []
    for model in args.models:
        for dataset in args.datasets:
            teacher_path = find_teacher_path(model, dataset, results_dir, args.teacher_seed)
            if teacher_path is None:
                print(f"Warning: No teacher found for {model}/{dataset} (seed {args.teacher_seed}), skipping")
                continue
            for seed in args.seeds:
                experiments.append((model, dataset, seed, teacher_path))

    total = len(experiments)
    _sweep_state["total"] = total

    print(f"Total KD experiments: {total}")
    print(f"Models: {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"Student seeds: {args.seeds}")
    print(f"Teacher seed: {args.teacher_seed}")
    print(f"KD params: T={args.temperature}, alpha={args.alpha}")
    print(f"Epochs: {args.epochs}")
    print()

    for i, (model, dataset, seed, teacher_path) in enumerate(experiments, 1):
        result = run_kd_experiment(
            model,
            dataset,
            seed,
            teacher_path,
            output_dir,
            args.epochs,
            args.temperature,
            args.alpha,
            i,
            total,
            args.dry_run,
        )
        _sweep_state[result] += 1

    print_summary()


if __name__ == "__main__":
    main()
