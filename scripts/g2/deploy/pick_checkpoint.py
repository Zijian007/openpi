#!/usr/bin/env python3
"""Pick a G2 checkpoint step dir for official ``scripts/serve_policy.py``.

Empty --policy-dir → newest experiment (by latest step mtime), then its highest
numeric step. Prints the resolved path on stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_under_project_root(path: str | Path, root: Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved.resolve()
    return (root / resolved).resolve()


def _numeric_step_directories(experiment_directory: Path) -> list[Path]:
    if not experiment_directory.is_dir():
        return []
    return sorted(
        (entry for entry in experiment_directory.iterdir() if entry.is_dir() and entry.name.isdigit()),
        key=lambda path: int(path.name),
    )


def select_highest_numeric_step_directory(experiment_directory: Path) -> Path | None:
    step_directories = _numeric_step_directories(experiment_directory)
    return step_directories[-1] if step_directories else None


def autoselect_policy_step_directory(config_checkpoint_root: Path) -> tuple[Path, str] | None:
    if not config_checkpoint_root.is_dir():
        return None

    step_directories: list[Path] = []
    for experiment_directory in config_checkpoint_root.iterdir():
        if not experiment_directory.is_dir():
            continue
        step_directories.extend(_numeric_step_directories(experiment_directory))

    if not step_directories:
        return None

    newest_step_directory = max(step_directories, key=lambda path: path.stat().st_mtime)
    experiment_directory = newest_step_directory.parent
    selected_step_directory = select_highest_numeric_step_directory(experiment_directory)
    if selected_step_directory is None:
        return None
    return selected_step_directory, experiment_directory.name


def assert_policy_checkpoint_ready(policy_step_directory: Path) -> None:
    if not policy_step_directory.is_dir():
        raise FileNotFoundError(
            f"checkpoint step directory not found: {policy_step_directory}\n"
            "Set --policy-dir to …/<exp>/<step> or train a checkpoint first."
        )
    has_jax_params = (policy_step_directory / "params").is_dir()
    has_pytorch_weights = (policy_step_directory / "model.safetensors").is_file()
    if not has_jax_params and not has_pytorch_weights:
        raise FileNotFoundError(
            f"{policy_step_directory} looks incomplete (need params/ or model.safetensors)"
        )


def resolve_policy_step_directory(
    train_config_name: str,
    project_root_dir: Path,
    explicit_step_directory: str | Path | None = None,
    checkpoint_base_dir: str | Path = "./checkpoints",
) -> Path:
    if explicit_step_directory:
        policy_step_directory = resolve_under_project_root(explicit_step_directory, project_root_dir)
        assert_policy_checkpoint_ready(policy_step_directory)
        return policy_step_directory

    base = Path(checkpoint_base_dir).expanduser()
    if not base.is_absolute():
        base = project_root_dir / base
    config_checkpoint_root = (base / train_config_name).resolve()
    selection = autoselect_policy_step_directory(config_checkpoint_root)
    if selection is None:
        raise FileNotFoundError(
            f"No checkpoint found under {config_checkpoint_root}. "
            "Train first or pass --policy-dir."
        )
    policy_step_directory, experiment_name = selection
    print(f"auto policy: newest exp {experiment_name}", file=sys.stderr)
    assert_policy_checkpoint_ready(policy_step_directory)
    return policy_step_directory


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--policy-dir", default="")
    parser.add_argument("--checkpoint-base-dir", default="./checkpoints")
    args = parser.parse_args(argv)
    path = resolve_policy_step_directory(
        train_config_name=args.config_name,
        project_root_dir=project_root(),
        explicit_step_directory=args.policy_dir or None,
        checkpoint_base_dir=args.checkpoint_base_dir,
    )
    print(path)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
