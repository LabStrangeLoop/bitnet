"""Fix experiment directory naming to match current convention.

Scans results/ directories, reads each config.json, computes the canonical
directory name from get_experiment_dir(), and generates rename commands.

Produces two shell scripts:
  - rename_local.sh:  renames dirs locally (JSON files only)
  - rename_server.sh: renames dirs on server (full experiments)

Usage:
    uv run python -m scripts.fix_naming             # dry-run (print mismatches)
    uv run python -m scripts.fix_naming --write      # write shell scripts
    uv run python -m scripts.fix_naming --apply      # apply renames directly (local)
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

# Defaults must match experiments/paths.py
TRAINING_DEFAULTS = {"lr": 0.1, "augment": "basic", "ablation": "none", "optimizer": "sgd"}
KD_DEFAULTS = {"temperature": 4.0, "alpha": 0.9}

RESULTS_DIRS = ["results/raw", "results/raw_kd"]
SERVER_BASE = "/home/dcazzani/code/lab-strange-loop/bitnet"


@dataclass(frozen=True)
class Rename:
    """A single directory rename operation."""

    old_path: Path
    new_path: Path
    has_conflict: bool


def get_version(config: dict) -> str:
    """Handle both old (bit_version: bool) and new (version: str) formats."""
    if "version" in config:
        return str(config["version"])
    if "bit_version" in config:
        return "bit" if config["bit_version"] else "std"
    return "bit"


def canonical_name(config: dict, experiment_type: str) -> str:
    """Reproduce naming logic from experiments/paths.py:get_experiment_dir."""
    parts: list[str] = []
    version = get_version(config)
    augment = config.get("augment", "basic")
    ablation = config.get("ablation", "none")
    optimizer = config.get("optimizer", "sgd")
    lr = config.get("lr", 0.1)
    seed = config.get("seed", 42)
    kd = config.get("kd", {})

    if experiment_type == "raw":
        parts.append(version)
        if augment != TRAINING_DEFAULTS["augment"]:
            parts.append(augment)
        if ablation != TRAINING_DEFAULTS["ablation"]:
            parts.append(ablation)
        if optimizer != TRAINING_DEFAULTS["optimizer"]:
            parts.append(optimizer)
        if lr != TRAINING_DEFAULTS["lr"]:
            parts.append(f"lr{lr:g}")
    else:  # raw_kd
        parts.append("bit_kd")
        if ablation != TRAINING_DEFAULTS["ablation"]:
            parts.append(ablation)
        if optimizer != TRAINING_DEFAULTS["optimizer"]:
            parts.append(optimizer)
        if lr != TRAINING_DEFAULTS["lr"]:
            parts.append(f"lr{lr:g}")
        if kd:
            temp = kd.get("temperature", 4.0)
            alpha = kd.get("alpha", 0.9)
            if temp != KD_DEFAULTS["temperature"]:
                parts.append(f"t{temp:g}")
            if alpha != KD_DEFAULTS["alpha"]:
                parts.append(f"a{alpha:g}")

    parts.append(f"s{seed}")
    return "_".join(parts)


def find_mismatches() -> list[Rename]:
    """Scan experiment directories and find naming mismatches."""
    mismatches: list[Rename] = []

    for results_dir in RESULTS_DIRS:
        base = Path(results_dir)
        if not base.exists():
            continue

        exp_type = "raw" if "raw_kd" not in results_dir else "raw_kd"
        for config_file in base.rglob("config.json"):
            exp_dir = config_file.parent
            with open(config_file) as f:
                config = json.load(f)

            current_name = exp_dir.name
            expected_name = canonical_name(config, exp_type)

            if current_name != expected_name:
                new_path = exp_dir.parent / expected_name
                mismatches.append(
                    Rename(
                        old_path=exp_dir,
                        new_path=new_path,
                        has_conflict=new_path.exists(),
                    )
                )

    return sorted(mismatches, key=lambda r: str(r.old_path))


def update_json_paths(directory: Path, old_name: str, new_name: str) -> None:
    """Update output_dir references inside config.json and results.json."""
    for json_file in ["config.json", "results.json"]:
        path = directory / json_file
        if not path.exists():
            continue

        text = path.read_text()
        if old_name in text:
            text = text.replace(old_name, new_name)
            path.write_text(text)


def apply_renames(renames: list[Rename]) -> None:
    """Apply renames directly on the local filesystem."""
    for r in renames:
        if r.has_conflict:
            print(f"  SKIP (conflict): {r.old_path} -> {r.new_path}")
            continue

        old_name = r.old_path.name
        new_name = r.new_path.name
        shutil.move(str(r.old_path), str(r.new_path))
        update_json_paths(r.new_path, old_name, new_name)
        print(f"  RENAMED: {r.old_path} -> {r.new_path}")


def write_shell_scripts(renames: list[Rename]) -> None:
    """Write rename_local.sh and rename_server.sh."""
    _write_script(renames, Path("scripts/rename_local.sh"), prefix="")
    _write_script(renames, Path("scripts/rename_server.sh"), prefix=SERVER_BASE + "/")
    print(f"Wrote scripts/rename_local.sh ({len(renames)} renames)")
    print(f"Wrote scripts/rename_server.sh ({len(renames)} renames)")


def _write_script(renames: list[Rename], output: Path, prefix: str) -> None:
    """Write a shell script with mv commands and JSON fixups."""
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "# Auto-generated by scripts/fix_naming.py",
        "# Renames experiment directories to match current naming convention.",
        "set -euo pipefail",
        "",
    ]

    for r in renames:
        old = f"{prefix}{r.old_path}"
        new = f"{prefix}{r.new_path}"
        old_name = r.old_path.name
        new_name = r.new_path.name

        if r.has_conflict:
            lines.append("# CONFLICT — target exists, skipping:")
            lines.append(f"#   {old} -> {new}")
            lines.append("")
            continue

        lines.append(f"# {old_name} -> {new_name}")
        lines.append(f'mv "{old}" "{new}"')
        # Fix output_dir inside JSON files
        for jf in ["config.json", "results.json"]:
            lines.append(f'[ -f "{new}/{jf}" ] && ' f"sed -i'' -e 's/{old_name}/{new_name}/g' \"{new}/{jf}\"")
        lines.append("")

    output.write_text("\n".join(lines) + "\n")
    output.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix experiment directory naming")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true", help="Write shell scripts")
    group.add_argument("--apply", action="store_true", help="Apply renames locally")
    args = parser.parse_args()

    renames = find_mismatches()

    if not renames:
        print("All experiment directories match the current naming convention.")
        return

    print(f"Found {len(renames)} naming mismatches:\n")
    for r in renames:
        conflict = " [CONFLICT]" if r.has_conflict else ""
        print(f"  {r.old_path.name:45s} -> {r.new_path.name}{conflict}")
        print(f"    in {r.old_path.parent}")

    if args.write:
        print()
        write_shell_scripts(renames)
    elif args.apply:
        print("\nApplying renames...")
        apply_renames(renames)
        print("Done.")
    else:
        print("\nDry run. Use --write to generate shell scripts or --apply to rename locally.")


if __name__ == "__main__":
    main()
