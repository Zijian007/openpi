#!/usr/bin/env python3
"""Run G2 normalization and training from one YAML experiment configuration."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

import yaml

from openpi.adapters.g2 import config as g2_config

PROJECT_ROOT = g2_config.PROJECT_ROOT


def resolve_under_project(path: str, project_root: Path = PROJECT_ROOT) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()


def should_skip_norm(norm_mode: str, norm_stats_file: Path) -> bool:
    return norm_mode == "skip" or (norm_mode == "auto" and norm_stats_file.is_file())


def _parse_argv(argv: list[str]):
    if "--" in argv:
        split_at = argv.index("--")
        argv, extra_args = argv[:split_at], tuple(argv[split_at + 1 :])
    else:
        extra_args = ()
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument(
        "--config", type=Path, default=g2_config.DEFAULT_CONFIG, help="YAML file; paths are relative to repo root"
    )
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, help="short run; preserves all module modes")
    parser.add_argument("--arm", choices=("both", "left", "right"))
    for module in ("vision", "language", "action-expert"):
        parser.add_argument(f"--{module}-mode", choices=("frozen", "lora", "full"))
    for flag in ("exp-name", "repo-id", "data-root", "checkpoint-base-dir", "assets-base-dir", "params-path"):
        parser.add_argument(f"--{flag}")
    for flag in (
        "batch-size",
        "num-workers",
        "num-train-steps",
        "log-interval",
        "save-interval",
        "seed",
        "fsdp-devices",
        "action-horizon",
        "max-frames",
        "warmup-steps",
        "decay-steps",
    ):
        parser.add_argument(f"--{flag}", type=int)
    for flag in ("peak-lr", "decay-lr"):
        parser.add_argument(f"--{flag}", type=float)
    parser.add_argument("--keep-period", type=lambda value: None if value.lower() in ("none", "null") else int(value))
    for flag in ("overwrite", "resume"):
        parser.add_argument(f"--{flag}", action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb", nargs="?", const="on", choices=("auto", "on", "off"))
    parser.add_argument("--no-wandb", dest="wandb", action="store_const", const="off")
    parser.add_argument("--wandb-project", "--project-name", dest="wandb_project")
    parser.add_argument("--norm", choices=("auto", "force", "skip"))
    parser.add_argument("--skip-norm", dest="norm", action="store_const", const="skip")
    parser.add_argument("--force-norm", dest="norm", action="store_const", const="force")
    parser.add_argument("--norm-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage", choices=("norm", "train"), help=argparse.SUPPRESS)
    parser.add_argument("--resolved-config", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv), extra_args


def _resolve(args):
    overrides = vars(args).copy()
    if "arm" not in overrides and "ARM" in os.environ:
        overrides["arm"] = os.environ["ARM"]
    settings = g2_config.resolve_settings(g2_config.load_settings(args.config), overrides)
    if not settings["experiment"]["name"]:
        if settings["checkpoints"]["resume"]:
            raise ValueError("Resume requires an explicit experiment.name; use --exp-name for a smoke experiment")
        prefix = "g2_vr_smoke" if settings["run"]["smoke"] else "g2_vr_pi05"
        stamp = dt.datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
        settings["experiment"]["name"] = f"{prefix}_{stamp}"
    for section, key in (("data", "root"), ("checkpoints", "base_dir"), ("checkpoints", "assets_dir")):
        settings[section][key] = str(resolve_under_project(settings[section][key]))
    if "://" not in settings["model"]["params_path"]:
        settings["model"]["params_path"] = str(resolve_under_project(settings["model"]["params_path"]))
    settings["runtime"]["mem_fraction"] = float(
        os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", settings["runtime"]["mem_fraction"])
    )
    g2_config.validate_settings(settings)
    return settings


def _setup_runtime(settings):
    os.environ["HF_LEROBOT_HOME"] = settings["data"]["root"]
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(map(str, settings["runtime"]["gpus"])))
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(settings["runtime"]["mem_fraction"])


def _check_dataset(settings):
    root = Path(settings["data"]["root"]) / settings["data"]["repo_id"]
    for name in ("info.json", "episodes_stats.jsonl"):
        if not (root / "meta" / name).is_file():
            raise FileNotFoundError(f"LeRobot v2.1 dataset file missing: {root / 'meta' / name}")
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"Expected LeRobot v2.1 in {root}, got {info.get('codebase_version')!r}")
    from g2_policy_tools.data_contract import validate_dataset_contract

    validate_dataset_contract(
        info.get("features", {}), info, settings["policy"],
        tuple(f"observation.images.{camera}" for camera in g2_config.selected_cameras(settings)),
    )


def _run_stage(stage: str, snapshot: Path):
    payload = g2_config.read_yaml(snapshot)
    settings = g2_config.normalize_settings(payload["settings"])
    _setup_runtime(settings)  # must precede importing JAX/LeRobot
    config = g2_config.build_train_config(
        settings, tuple(payload["extra_args"]),
        resume_checkpoint=Path(payload["resume_checkpoint"]) if payload.get("resume_checkpoint") else None,
        resume_training=payload.get("training_state_config"),
    )
    # scripts/ contains the official entrypoints, without needing to be a package.
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    if stage == "norm":
        from compute_norm_stats import compute_norm_stats

        compute_norm_stats(config, max_frames=settings["normalization"]["max_frames"])
        signature = config.assets_dirs / settings["data"]["repo_id"] / "g2_norm_config.yaml"
        signature.write_text(yaml.safe_dump(_norm_signature(settings), sort_keys=False))
    else:
        from train import main as train_main

        train_main(config)


def normalize_child_status(return_code: int) -> int:
    """Map a negative wait status (-signal) to the conventional 128+signal exit code."""
    if return_code < 0:
        return 128 + (-return_code)
    return return_code


def _run_command(command: list[str]) -> None:
    # Ignore SIGINT/SIGTERM here so Ctrl+C reaches train.py, which saves the current
    # step. This process keeps waiting until that checkpoint write finishes.
    process = subprocess.Popen([sys.executable, *command], cwd=PROJECT_ROOT)
    previous = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        return_code = process.wait()
    finally:
        signal.signal(signal.SIGINT, previous[signal.SIGINT])
        signal.signal(signal.SIGTERM, previous[signal.SIGTERM])
    return_code = normalize_child_status(return_code)
    if return_code in (128 + signal.SIGINT, 128 + signal.SIGTERM):
        print("\nChild process interrupted; check its logs for checkpoint completion.", file=sys.stderr)
        raise SystemExit(return_code)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, [sys.executable, *command])


def _norm_signature(settings):
    return {
        "data_root": settings["data"]["root"],
        "repo_id": settings["data"]["repo_id"],
        "arm": settings["experiment"]["arm"],
        "policy": g2_config.policy_semantics(settings),
        "cameras": list(g2_config.selected_cameras(settings)),
    }


def main(argv: list[str] | None = None) -> None:
    args, extra_args = _parse_argv(list(argv if argv is not None else sys.argv[1:]))
    if hasattr(args, "stage"):
        _run_stage(args.stage, args.resolved_config)
        return
    settings = _resolve(args)
    g2_config.validate_extra_args(extra_args)
    _setup_runtime(settings)
    name = g2_config.training_config_name(settings)
    norm_path = Path(settings["checkpoints"]["assets_dir"]) / name / settings["data"]["repo_id"] / "norm_stats.json"
    ckpt_dir = Path(settings["checkpoints"]["base_dir"]) / name / settings["experiment"]["name"]
    norm_only = getattr(args, "norm_only", False)
    dry_run = getattr(args, "dry_run", False)
    resume_checkpoint = None
    resume_training = None
    if settings["checkpoints"]["resume"]:
        if norm_only:
            raise ValueError("Resume cannot be combined with --norm-only; use --no-resume for a new stats computation")
        from openpi.adapters.g2.resume import resolve_resume

        settings, extra_args, resume_checkpoint, resume_training = resolve_resume(settings, extra_args, ckpt_dir)
        norm_path = resume_checkpoint / "assets" / settings["data"]["repo_id"] / "norm_stats.json"
        skip_norm = True
    else:
        skip_norm = should_skip_norm(settings["normalization"]["mode"], norm_path)
        if skip_norm and settings["normalization"]["mode"] == "auto":
            signature_path = norm_path.with_name("g2_norm_config.yaml")
            skip_norm = signature_path.is_file() and g2_config.read_yaml(signature_path) == _norm_signature(settings)
    payload = {"settings": g2_config.public_settings(settings), "extra_args": list(extra_args)}
    if resume_checkpoint is not None:
        payload.update(resume_checkpoint=str(resume_checkpoint), training_state_config=resume_training)
        print(f"Resume checkpoint: {resume_checkpoint} (step={resume_checkpoint.name})")
        print(f"Target total steps: {settings['training']['num_train_steps']}; decay_steps: {settings['training']['decay_steps']}")
        print("Normalization: checkpoint assets (normalization.mode only applies to new training)")
    print(f"YAML: {args.config.resolve()}\nTrainConfig: {name}\nCheckpoint: {ckpt_dir}")
    print(f"Norm stats: {norm_path} (skip={skip_norm})")
    print("Resolved experiment configuration:")
    print(yaml.safe_dump(payload["settings"], sort_keys=False, allow_unicode=True), flush=True)
    print("Derived policy contract:")
    print(yaml.safe_dump(g2_config.describe_policy(settings), sort_keys=False, allow_unicode=True), flush=True)
    if dry_run:
        print("[dry-run] No data access, downloads, normalization, or training.")
        print(f"[1/2] {'reuse' if skip_norm else 'compute'} norm stats using the resolved config above")
        if not norm_only:
            print("[2/2] train using the same resolved config")
        return
    _check_dataset(settings)
    if (
        not norm_only
        and ckpt_dir.exists()
        and not (settings["checkpoints"]["resume"] or settings["checkpoints"]["overwrite"])
    ):
        raise FileExistsError(f"Experiment exists: {ckpt_dir}; choose a new experiment.name or use --resume")
    if skip_norm and not norm_only and not norm_path.is_file():
        raise FileNotFoundError(f"Cannot skip normalization: {norm_path} does not exist")
    with tempfile.TemporaryDirectory(prefix="g2-train-", dir="/tmp") as temporary:
        snapshot = Path(temporary) / g2_config.CHECKPOINT_CONFIG
        snapshot.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
        worker = [str(Path(__file__).resolve()), "--resolved-config", str(snapshot)]
        if not skip_norm:
            print("[1/2] Computing norm stats", flush=True)
            _run_command([*worker, "--stage", "norm"])
        else:
            print(f"[1/2] Reusing {norm_path}", flush=True)
        if not norm_only:
            print("[2/2] Training", flush=True)
            _run_command([*worker, "--stage", "train"])
    print(f"Done: {norm_path if norm_only else ckpt_dir}")


if __name__ == "__main__":
    main()
