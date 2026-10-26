#!/usr/bin/env python3
"""Run G2 normalization and training from one YAML experiment configuration."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

import yaml

from openpi.training import g2_config

PROJECT_ROOT = g2_config.PROJECT_ROOT


def resolve_under_project(path: str, project_root: Path = PROJECT_ROOT) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()


def should_skip_norm(norm_mode: str, norm_stats_file: Path) -> bool:
    return norm_mode == "skip" or (norm_mode == "auto" and norm_stats_file.is_file())


def wandb_enabled(wandb_mode: str, preset: str) -> bool:
    return wandb_mode == "on" or (wandb_mode == "auto" and preset != "smoke")


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
    parser.add_argument("--preset", choices=("low_mem", "full_ft", "smoke"))
    parser.add_argument("--smoke", dest="preset", action="store_const", const="smoke")
    parser.add_argument("--full", dest="preset", action="store_const", const="full_ft")
    parser.add_argument("--config-name", help="Compatibility alias for a built-in G2 preset/arm")
    parser.add_argument("--arm", choices=("both", "left", "right"))
    for flag in ("exp-name", "repo-id", "data-root", "checkpoint-base-dir", "assets-base-dir", "params-path"):
        parser.add_argument(f"--{flag}")
    parser.add_argument("--action-mode", choices=("relative", "absolute"))
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
    for flag in ("overwrite", "resume", "use-proxy"):
        parser.add_argument(f"--{flag}", action=argparse.BooleanOptionalAction)
    parser.add_argument("--no-proxy", dest="use_proxy", action="store_false")
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
    name = overrides.pop("config_name", None)
    if name:
        base = name.removesuffix("_left").removesuffix("_right")
        presets = {"pi05_g2_vr": "full_ft", "pi05_g2_vr_low_mem": "low_mem"}
        if base not in presets:
            raise ValueError("--config-name only accepts built-in pi05_g2_vr configurations; use YAML for overrides")
        if name.endswith(("_left", "_right")):
            arm = "left" if name.endswith("_left") else "right"
            if "arm" in overrides and overrides["arm"] != arm:
                raise ValueError("--arm disagrees with --config-name")
            overrides["arm"] = arm
        overrides["preset"] = presets[base]
    if "arm" not in overrides and "ARM" in os.environ:
        overrides["arm"] = os.environ["ARM"]
    settings = g2_config.resolve_settings(g2_config.load_settings(args.config), overrides)
    if not settings["experiment"]["name"]:
        prefix = "g2_vr_smoke" if settings["experiment"]["preset"] == "smoke" else "g2_vr_pi05"
        stamp = dt.datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
        settings["experiment"]["name"] = f"{prefix}_{stamp}"
    for section, key in (("data", "root"), ("checkpoints", "base_dir"), ("checkpoints", "assets_dir")):
        settings[section][key] = str(resolve_under_project(settings[section][key]))
    if "://" not in settings["model"]["params_path"]:
        settings["model"]["params_path"] = str(resolve_under_project(settings["model"]["params_path"]))
    settings["runtime"]["gpu"] = os.environ.get("CUDA_VISIBLE_DEVICES", settings["runtime"]["gpu"])
    settings["runtime"]["mem_fraction"] = float(
        os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", settings["runtime"]["mem_fraction"])
    )
    g2_config.validate_settings(settings)
    return settings


def _setup_runtime(settings):
    os.environ["HF_LEROBOT_HOME"] = settings["data"]["root"]
    os.environ["CUDA_VISIBLE_DEVICES"] = settings["runtime"]["gpu"]
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(settings["runtime"]["mem_fraction"])
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        if settings["runtime"]["use_proxy"]:
            os.environ[key] = settings["runtime"]["proxy"]
        else:
            os.environ.pop(key, None)


def _check_dataset(settings):
    root = Path(settings["data"]["root"]) / settings["data"]["repo_id"]
    for name in ("info.json", "episodes_stats.jsonl"):
        if not (root / "meta" / name).is_file():
            raise FileNotFoundError(f"LeRobot v2.1 dataset file missing: {root / 'meta' / name}")
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"Expected LeRobot v2.1 in {root}, got {info.get('codebase_version')!r}")
    features = info.get("features", {})
    state = "state.pose" if settings["data"]["action_mode"] == "relative" else "state.joint"
    required = {
        state,
        "action.pose",
        *(f"observation.images.{camera}" for camera in g2_config.selected_cameras(settings)),
    }
    missing = required - features.keys()
    if missing:
        raise ValueError(f"Dataset is missing required features: {sorted(missing)}")


def _run_stage(stage: str, snapshot: Path):
    payload = g2_config.read_yaml(snapshot)
    settings = payload["settings"]
    _setup_runtime(settings)  # must precede importing JAX/LeRobot
    config = g2_config.build_train_config(settings, tuple(payload["extra_args"]))
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


def _run_command(command: list[str], *, dry_run: bool) -> None:
    if dry_run:
        print(shlex.join(["uv", "run", *command]))
        return
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
        "action_mode": settings["data"]["action_mode"],
        "cameras": list(g2_config.selected_cameras(settings)),
        "action_horizon": settings["model"]["action_horizon"],
    }


def main(argv: list[str] | None = None) -> None:
    args, extra_args = _parse_argv(list(argv if argv is not None else sys.argv[1:]))
    if hasattr(args, "stage"):
        _run_stage(args.stage, args.resolved_config)
        return
    settings = _resolve(args)
    if any(arg.startswith("--") and not arg.startswith(("--optimizer.", "--ema-decay")) for arg in extra_args):
        raise ValueError("Configure model/data/training in YAML; '--' only accepts optimizer/EMA overrides")
    _setup_runtime(settings)
    name = g2_config.config_name(settings)
    norm_path = Path(settings["checkpoints"]["assets_dir"]) / name / settings["data"]["repo_id"] / "norm_stats.json"
    ckpt_dir = Path(settings["checkpoints"]["base_dir"]) / name / settings["experiment"]["name"]
    skip_norm = should_skip_norm(settings["normalization"]["mode"], norm_path)
    if skip_norm and settings["normalization"]["mode"] == "auto":
        signature_path = norm_path.with_name("g2_norm_config.yaml")
        skip_norm = signature_path.is_file() and g2_config.read_yaml(signature_path) == _norm_signature(settings)
    norm_only = getattr(args, "norm_only", False)
    dry_run = getattr(args, "dry_run", False)
    payload = {"settings": settings, "extra_args": list(extra_args)}
    print(f"YAML: {args.config.resolve()}\nTrainConfig: {name}\nCheckpoint: {ckpt_dir}")
    print(f"Norm stats: {norm_path} (skip={skip_norm})")
    print("Resolved experiment configuration:")
    print(yaml.safe_dump(settings, sort_keys=False, allow_unicode=True), flush=True)
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
    if settings["checkpoints"]["resume"] and not norm_only:
        previous = ckpt_dir / g2_config.CHECKPOINT_CONFIG
        if previous.is_file():
            old = g2_config.read_yaml(previous)
            # Resume may change total steps / logging / LR, but must preserve input/output semantics.
            for section in ("data", "model"):
                if old["settings"][section] != settings[section]:
                    raise ValueError(f"Cannot resume with changed {section} settings; use a new experiment.name")
    with tempfile.TemporaryDirectory(prefix="g2-train-", dir="/tmp") as temporary:
        snapshot = Path(temporary) / g2_config.CHECKPOINT_CONFIG
        snapshot.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
        worker = [str(Path(__file__).resolve()), "--resolved-config", str(snapshot)]
        if not skip_norm:
            print("[1/2] Computing norm stats", flush=True)
            _run_command([*worker, "--stage", "norm"], dry_run=False)
        else:
            print(f"[1/2] Reusing {norm_path}", flush=True)
        if not norm_only:
            print("[2/2] Training", flush=True)
            _run_command([*worker, "--stage", "train"], dry_run=False)
    print(f"Done: {norm_path if norm_only else ckpt_dir}")


if __name__ == "__main__":
    main()
