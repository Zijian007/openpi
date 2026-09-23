#!/usr/bin/env python3
"""G2 wrapper: compute_norm_stats → train via official OpenPI scripts.

Daily knobs live in ``train_pipeline.sh`` (exported as ``G2_TRAIN_*``).
This file only applies presets and shells out to::

    uv run scripts/compute_norm_stats.py ...
    uv run scripts/train.py ...
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys
from pathlib import Path

PRESET_TO_CONFIG = {
    "low_mem": "pi05_g2_vr_low_mem",
    "full_ft": "pi05_g2_vr",
    "smoke": "pi05_g2_vr_low_mem",
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, "")
    return int(raw) if raw else None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"true", "1", "yes"}


def _flag_present(argv: list[str], flag: str) -> bool:
    return any(token == flag or token.startswith(f"{flag}=") for token in argv)


def resolve_under_project(path: str, project_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()


def resolve_config_name(
    *,
    preset: str,
    config_name: str,
    preset_from_cli: bool,
    config_name_from_cli: bool,
) -> str:
    if config_name_from_cli:
        return config_name
    if preset_from_cli or not config_name:
        return PRESET_TO_CONFIG[preset]
    return config_name


def should_skip_norm(norm_mode: str, norm_stats_file: Path) -> bool:
    if norm_mode == "force":
        return False
    if norm_mode == "skip":
        return True
    return norm_stats_file.is_file()


def wandb_enabled(wandb_mode: str, preset: str) -> bool:
    if wandb_mode == "on":
        return True
    if wandb_mode == "off":
        return False
    return preset != "smoke"


def _parse_argv(argv: list[str]) -> tuple[argparse.Namespace, list[str], tuple[str, ...]]:
    if "--" in argv:
        split_at = argv.index("--")
        pipeline_argv, passthru = argv[:split_at], tuple(argv[split_at + 1 :])
    else:
        pipeline_argv, passthru = argv, ()

    rewritten: list[str] = []
    index = 0
    while index < len(pipeline_argv):
        token = pipeline_argv[index]
        if token == "--smoke":
            rewritten.extend(["--preset", "smoke"])
        elif token == "--full":
            rewritten.extend(["--preset", "full_ft"])
        elif token == "--no-wandb":
            rewritten.extend(["--wandb", "off"])
        elif token == "--wandb":
            if index + 1 < len(pipeline_argv) and pipeline_argv[index + 1] in {"auto", "on", "off"}:
                rewritten.append(token)
            else:
                rewritten.extend(["--wandb", "on"])
        elif token == "--project-name" and index + 1 < len(pipeline_argv):
            rewritten.extend(["--wandb-project", pipeline_argv[index + 1]])
            index += 2
            continue
        elif token == "--no-proxy":
            rewritten.append("--no-use-proxy")
        else:
            rewritten.append(token)
        index += 1

    parser = argparse.ArgumentParser(
        description="G2 wrapper around OpenPI compute_norm_stats.py + train.py"
    )
    parser.add_argument("--preset", choices=sorted(PRESET_TO_CONFIG), default="low_mem")
    parser.add_argument("--config-name", default=_env("G2_TRAIN_CONFIG_NAME"))
    parser.add_argument("--exp-name", default=_env("G2_TRAIN_EXP_NAME"))
    parser.add_argument("--repo-id", default=_env("G2_TRAIN_REPO_ID", "g2_vr_lerobot_v21"))
    parser.add_argument("--norm", choices=("auto", "skip", "force"), default=_env("G2_TRAIN_NORM", "auto"))
    parser.add_argument("--max-frames", type=int, default=_env_int("G2_TRAIN_MAX_FRAMES"))
    parser.add_argument("--checkpoint-base-dir", default=_env("G2_TRAIN_CHECKPOINT_BASE_DIR", "./checkpoints"))
    parser.add_argument("--assets-base-dir", default=_env("G2_TRAIN_ASSETS_BASE_DIR", "./assets"))
    parser.add_argument("--batch-size", type=int, default=_env_int("G2_TRAIN_BATCH_SIZE"))
    parser.add_argument("--num-workers", type=int, default=_env_int("G2_TRAIN_NUM_WORKERS"))
    parser.add_argument("--num-train-steps", type=int, default=_env_int("G2_TRAIN_NUM_TRAIN_STEPS"))
    parser.add_argument("--log-interval", type=int, default=_env_int("G2_TRAIN_LOG_INTERVAL"))
    parser.add_argument("--save-interval", type=int, default=_env_int("G2_TRAIN_SAVE_INTERVAL"))
    parser.add_argument("--keep-period", default=_env("G2_TRAIN_KEEP_PERIOD") or None)
    parser.add_argument("--seed", type=int, default=_env_int("G2_TRAIN_SEED"))
    parser.add_argument("--fsdp-devices", type=int, default=_env_int("G2_TRAIN_FSDP_DEVICES"))
    parser.add_argument("--overwrite", dest="overwrite", action="store_true")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.set_defaults(overwrite=_env_bool("G2_TRAIN_OVERWRITE", True))
    parser.add_argument("--resume", action="store_true", default=_env_bool("G2_TRAIN_RESUME", False))
    parser.add_argument("--wandb", choices=("auto", "on", "off"), default=_env("G2_TRAIN_WANDB", "auto"))
    parser.add_argument("--wandb-project", default=_env("G2_TRAIN_WANDB_PROJECT", "G2_openpi"))
    parser.add_argument(
        "--use-proxy",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("G2_TRAIN_USE_PROXY", True),
    )
    parser.add_argument("--norm-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-norm", action="store_true")
    parser.add_argument("--force-norm", action="store_true")
    args = parser.parse_args(rewritten)
    return args, rewritten, passthru


def _build_norm_command(args: argparse.Namespace, config_name: str) -> list[str]:
    command = [
        "scripts/compute_norm_stats.py",
        "--config-name",
        config_name,
        f"--repo-id={args.repo_id}",
    ]
    if args.max_frames is not None:
        command.append(f"--max-frames={args.max_frames}")
    return command


def _build_train_command(
    args: argparse.Namespace,
    *,
    config_name: str,
    exp_name: str,
    overwrite: bool,
    wandb_on: bool,
    passthru: tuple[str, ...],
) -> list[str]:
    command = [
        "scripts/train.py",
        config_name,
        f"--exp-name={exp_name}",
        f"--checkpoint-base-dir={args.checkpoint_base_dir}",
        f"--assets-base-dir={args.assets_base_dir}",
        f"--project-name={args.wandb_project}",
        f"--data.repo-id={args.repo_id}",
    ]
    for value, flag in (
        (args.batch_size, "--batch-size"),
        (args.num_workers, "--num-workers"),
        (args.num_train_steps, "--num-train-steps"),
        (args.log_interval, "--log-interval"),
        (args.save_interval, "--save-interval"),
        (args.seed, "--seed"),
        (args.fsdp_devices, "--fsdp-devices"),
    ):
        if value is not None:
            command.append(f"{flag}={value}")
    if args.keep_period:
        if args.keep_period.lower() in {"none", "null"}:
            command.append("--keep-period=None")
        else:
            command.append(f"--keep-period={args.keep_period}")
    command.append("--overwrite" if overwrite else "--no-overwrite")
    command.append("--resume" if args.resume else "--no-resume")
    command.append("--wandb-enabled" if wandb_on else "--no-wandb-enabled")
    command.extend(passthru)
    return command


def _run_command(command: list[str], dry_run: bool) -> None:
    if dry_run:
        print(shlex.join(["uv", "run", *command]))
        return
    subprocess.run([sys.executable, *command], cwd=PROJECT_ROOT, check=True)


def main(argv: list[str] | None = None) -> None:
    raw = list(argv if argv is not None else sys.argv[1:])
    args, rewritten, passthru = _parse_argv(raw)

    preset_from_cli = _flag_present(rewritten, "--preset") or _flag_present(raw, "--smoke") or _flag_present(raw, "--full")
    config_name_from_cli = _flag_present(rewritten, "--config-name")
    config_name = resolve_config_name(
        preset=args.preset,
        config_name=args.config_name,
        preset_from_cli=preset_from_cli,
        config_name_from_cli=config_name_from_cli,
    )

    if args.preset == "smoke":
        if args.num_train_steps is None:
            args.num_train_steps = 20
        if args.batch_size is None:
            args.batch_size = 1

    wandb_mode = args.wandb
    if wandb_mode == "auto":
        wandb_mode = "off" if args.preset == "smoke" else "on"

    exp_name = args.exp_name
    if not exp_name:
        prefix = "g2_vr_smoke" if args.preset == "smoke" else "g2_vr_pi05"
        exp_name = f"{prefix}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    overwrite = False if args.resume else args.overwrite

    norm_mode = args.norm
    if args.force_norm:
        norm_mode = "force"
    elif args.skip_norm:
        norm_mode = "skip"

    os.chdir(PROJECT_ROOT)

    if config_name == "pi05_g2_vr":
        visible_gpu_count = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
        if visible_gpu_count <= 1:
            print(
                f"WARN: {config_name} full fine-tune usually OOMs on a single 24GB GPU.",
                file=sys.stderr,
            )
            print("      Use --preset low_mem, or add GPUs and --fsdp-devices.", file=sys.stderr)

    norm_stats_path = resolve_under_project(
        f"{args.assets_base_dir}/{config_name}/{args.repo_id}/norm_stats.json",
        PROJECT_ROOT,
    )
    skip_norm = should_skip_norm(norm_mode, norm_stats_path)
    ckpt_dir = resolve_under_project(
        f"{args.checkpoint_base_dir}/{config_name}/{exp_name}",
        PROJECT_ROOT,
    )
    wandb_on = wandb_enabled(wandb_mode, args.preset)

    print("========================================")
    print(f"pipeline:   norm_stats -> train (preset={args.preset})")
    print(f"project:    {PROJECT_ROOT}")
    print(f"config:     {config_name}")
    print(f"exp:        {exp_name}")
    print(f"data:       {os.environ.get('HF_LEROBOT_HOME', '<unset>')}/{args.repo_id}")
    print(f"hf_cache:   {os.environ.get('HF_DATASETS_CACHE', '<unset>')}")
    print(f"openpi:     {os.environ.get('OPENPI_DATA_HOME', '<unset>')}")
    print(f"norm:       {norm_mode} (skip={skip_norm})")
    print(f"max_frames: {args.max_frames if args.max_frames is not None else '<full>'}")
    print(f"ckpt_dir:   {ckpt_dir}")
    print(f"batch/steps:{args.batch_size if args.batch_size is not None else '<cfg>'} / {args.num_train_steps if args.num_train_steps is not None else '<cfg>'}")
    print(f"save/log:   {args.save_interval if args.save_interval is not None else '<cfg>'} / {args.log_interval if args.log_interval is not None else '<cfg>'}")
    print(f"wandb:      {wandb_on}  project={args.wandb_project}")
    print(f"overwrite:  {overwrite}  resume={args.resume}")
    print(f"GPU:        CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")
    if args.dry_run:
        print("dry_run:    true")
    if passthru:
        print(f"passthru:   {' '.join(passthru)}")
    print("========================================")

    norm_cmd = _build_norm_command(args, config_name)
    train_cmd = _build_train_command(
        args,
        config_name=config_name,
        exp_name=exp_name,
        overwrite=overwrite,
        wandb_on=wandb_on,
        passthru=passthru,
    )

    if args.dry_run:
        if not skip_norm:
            _run_command(norm_cmd, dry_run=True)
        else:
            print(f"[dry-run] skip norm stats (reuse {norm_stats_path})")
        if not args.norm_only:
            _run_command(train_cmd, dry_run=True)
        return

    if not skip_norm:
        print("\n[1/2] compute_norm_stats ...")
        _run_command(norm_cmd, dry_run=False)
    else:
        print(f"\n[1/2] compute_norm_stats skipped (reuse {norm_stats_path})")

    if args.norm_only:
        print(f"\ndone. norm stats: {norm_stats_path}")
        return

    print("\n[2/2] train ...")
    _run_command(train_cmd, dry_run=False)
    print(f"\ndone. checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
