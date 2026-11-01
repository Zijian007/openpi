#!/usr/bin/env python3
"""Run G2 normalization and training from one YAML experiment configuration."""

from __future__ import annotations

import argparse
import codecs
from contextlib import ExitStack
import datetime as dt
import errno
import json
import os
from pathlib import Path
import pty
import re
import signal
from string import Formatter
import subprocess
import sys
import tempfile
import termios
from zoneinfo import ZoneInfo

import yaml

from openpi.adapters.g2 import config as g2_config
from openpi.adapters.g2.training_config import to_native
from openpi.shared.wandb_preflight import check_wandb

PROJECT_ROOT = g2_config.PROJECT_ROOT


def resolve_under_project(path: str, project_root: Path = PROJECT_ROOT) -> Path:
    return (project_root / Path(path).expanduser()).resolve()


def _space_name(policy: dict, group: str) -> str:
    parts = []
    for branch in ("pose", "joint"):
        spec = policy[group][branch]
        if not spec["enabled"]:
            continue
        representation = ("dse3" if spec["method"] == "se3" else "dcomp") if spec["delta"] else "abs"
        parts.append(f"{group[0]}{branch}_{representation}")
    return "_".join(parts)


def _resolve_experiment_name(settings: dict, config_path: Path) -> str:
    template = settings["experiment"]["name"]
    if template == "auto":
        template = "{task}_{arm}_{state}_{action}_{date}_{version}"
    if "{" not in template and "}" not in template:
        return template
    if settings["checkpoints"]["resume"]:
        raise ValueError("Resume requires an explicit experiment.name; pass the generated name with --exp-name")
    values = {
        "task": re.sub(r"[^\w-]+", "_", config_path.stem.lower()).strip("_"),
        "arm": settings["experiment"]["arm"],
        "state": _space_name(settings["policy"], "state"),
        "action": _space_name(settings["policy"], "action"),
        "date": dt.datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y%m%d"),
        "version": "v01",
    }
    try:
        tokens = list(Formatter().parse(template))
        for _, field, format_spec, conversion in tokens:
            if field is not None and (field not in values or format_spec or conversion):
                raise ValueError(f"Unsupported experiment.name placeholder: {field!r}")
    except ValueError as error:
        raise ValueError(f"Invalid experiment.name template: {error}") from error
    if any(field == "version" for _, field, _, _ in tokens):
        # Match only runs from this template and date.
        pattern = "".join(
            re.escape(literal)
            + (r"v(\d+)" if field == "version" else re.escape(values[field]) if field is not None else "")
            for literal, field, _, _ in tokens
        )
        root = Path(settings["checkpoints"]["base_dir"])
        versions = []
        if root.is_dir():
            for entry in root.iterdir():
                match = re.fullmatch(pattern, entry.name)
                if match:
                    versions.append(int(match[1]))
        values["version"] = f"v{max(versions, default=0) + 1:02d}"
    name = template.format_map(values)
    if not name or name in (".", "..") or "/" in name or "\\" in name or "{" in name or "}" in name:
        raise ValueError("Generated experiment.name must be a nonempty directory name without braces")
    return name


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
    parser.add_argument("--run", type=Path, help="resume a run using its saved configuration")
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
    parser.add_argument("--steps", dest="num_train_steps", type=int)
    parser.add_argument("--learning-rate", dest="peak_lr", type=float)
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
    if hasattr(args, "run"):
        from openpi.adapters.g2.resume import latest_checkpoint

        run = resolve_under_project(str(args.run))
        if overrides.get("resume") is False:
            raise ValueError("--run resumes an existing experiment; cannot combine with --no-resume")
        snapshot = latest_checkpoint(run / "checkpoints") / "assets" / g2_config.CHECKPOINT_CONFIG
        settings = g2_config.normalize_settings(g2_config.read_yaml(snapshot)["settings"])
        overrides.update(resume=True, exp_name=run.name, checkpoint_base_dir=str(run.parent))
        args.config = snapshot
    else:
        settings = g2_config.load_settings(args.config)
    settings["runtime"]["mem_fraction"] = float(
        os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", settings["runtime"]["mem_fraction"])
    )
    settings = g2_config.resolve_settings(settings, overrides)
    if not settings["experiment"]["name"]:
        if settings["checkpoints"]["resume"]:
            raise ValueError("Resume requires an explicit experiment.name; use --exp-name for a smoke experiment")
        prefix = "g2_vr_smoke" if settings["run"]["smoke"] else "g2_vr_pi05"
        stamp = dt.datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y%m%d_%H%M%S_%f")
        settings["experiment"]["name"] = f"{prefix}_{stamp}"
    for section, key in (("data", "root"), ("checkpoints", "base_dir"), ("checkpoints", "assets_dir")):
        settings[section][key] = str(resolve_under_project(settings[section][key]))
    if "://" not in settings["model"]["params_path"]:
        settings["model"]["params_path"] = str(resolve_under_project(settings["model"]["params_path"]))
    settings["experiment"]["name"] = _resolve_experiment_name(settings, args.config)
    return settings


def _setup_runtime(settings):
    os.environ["HF_LEROBOT_HOME"] = settings["data"]["root"]
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(map(str, settings["runtime"]["gpus"])))
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(settings["runtime"]["mem_fraction"])


def _check_dataset(settings):
    from g2_policy_protocol.data.sources import resolve_sources

    for source in resolve_sources(settings["data"], base_dir=PROJECT_ROOT):
        try:
            _check_source(Path(source["path"]), settings)
        except (OSError, ValueError) as error:
            raise type(error)(f"Source {source['name']}: {error}") from error


def _check_source(root, settings):
    for name in ("info.json", "episodes_stats.jsonl"):
        if not (root / "meta" / name).is_file():
            raise FileNotFoundError(f"LeRobot v2.1 dataset file missing: {root / 'meta' / name}")
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"Expected LeRobot v2.1 in {root}, got {info.get('codebase_version')!r}")
    from g2_policy_protocol.data.dataset import validate_dataset_contract

    validate_dataset_contract(
        info.get("features", {}), info, settings["policy"],
        tuple(f"observation.images.{camera}" for camera in g2_config.selected_cameras(settings)),
    )


def _run_stage(stage: str, snapshot: Path):
    payload = g2_config.read_yaml(snapshot)
    settings = to_native(payload["settings"])
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
        signature = config.assets_dirs / g2_config.asset_id(settings) / "g2_norm_config.yaml"
        signature.write_text(yaml.safe_dump(g2_config.normalization_signature(settings), sort_keys=False))
    else:
        from train import main as train_main

        try:
            train_main(config)
        finally:
            # Only publish the authoritative record after native initialization/restoration wrote it.
            record = config.checkpoint_dir / g2_config.CHECKPOINT_CONFIG
            if record.is_file():
                (g2_config.run_dir(settings) / g2_config.CHECKPOINT_CONFIG).write_text(record.read_text())


def normalize_child_status(return_code: int) -> int:
    """Map a negative wait status (-signal) to the conventional 128+signal exit code."""
    if return_code < 0:
        return 128 + (-return_code)
    return return_code


def _run_command(command: list[str], *, log_path: Path | None = None) -> None:
    # Ignore SIGINT/SIGTERM here so Ctrl+C reaches train.py, which saves the current
    # step. This process keeps waiting until that checkpoint write finishes.
    use_pty = log_path is not None and sys.stdout.isatty()
    with ExitStack() as resources:
        log = resources.enter_context(log_path.open("ab")) if log_path else None
        master_fd = slave_fd = None
        if use_pty:
            master_fd, slave_fd = pty.openpty()
            resources.callback(os.close, master_fd)

            def resize_terminal(*_):
                try:
                    size = os.get_terminal_size(sys.stdout.fileno())
                    termios.tcsetwinsize(master_fd, (size.lines, size.columns))
                except OSError:
                    pass

            resize_terminal()
        try:
            process = subprocess.Popen(
                [sys.executable, *command], cwd=PROJECT_ROOT,
                stdout=slave_fd if use_pty else subprocess.PIPE if log else None,
                stderr=subprocess.STDOUT if log else None,
            )
        finally:
            if slave_fd is not None:
                os.close(slave_fd)
        if process.stdout is not None:
            resources.enter_context(process.stdout)
        previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if use_pty:
            previous[signal.SIGWINCH] = signal.getsignal(signal.SIGWINCH)
            signal.signal(signal.SIGWINCH, resize_terminal)
        try:
            if log:
                output_fd = master_fd if use_pty else process.stdout.fileno()
                binary_console = getattr(sys.stdout, "buffer", None)
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                while True:
                    try:
                        chunk = os.read(output_fd, 8192)
                    except OSError as error:
                        if use_pty and error.errno == errno.EIO:
                            break
                        raise
                    if not chunk:
                        break
                    log.write(chunk)
                    log.flush()
                    if binary_console is not None:
                        binary_console.write(chunk)
                        binary_console.flush()
                    else:
                        sys.stdout.write(decoder.decode(chunk))
                        sys.stdout.flush()
                if binary_console is None:
                    sys.stdout.write(decoder.decode(b"", final=True))
                    sys.stdout.flush()
            return_code = process.wait()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    return_code = normalize_child_status(return_code)
    if return_code in (128 + signal.SIGINT, 128 + signal.SIGTERM):
        print("\nChild process interrupted; check its logs for checkpoint completion.", file=sys.stderr)
        raise SystemExit(return_code)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, [sys.executable, *command])


def main(argv: list[str] | None = None) -> None:
    args, extra_args = _parse_argv(list(argv if argv is not None else sys.argv[1:]))
    if hasattr(args, "stage"):
        _run_stage(args.stage, args.resolved_config)
        return
    settings = _resolve(args)
    g2_config.validate_extra_args(extra_args)
    _setup_runtime(settings)
    if "sources" in settings["data"]:
        from openpi.adapters.g2.mixture import prepare_mixture

        sources = prepare_mixture(settings, base_dir=PROJECT_ROOT)
        mixture = settings["mixture"]
        for source, ratio, lengths, repeats in zip(
            sources, mixture["proportions"], mixture["lengths"], mixture["plan"], strict=True
        ):
            samples = sum(lengths)
            virtual_samples = sum(length * count for length, count in zip(lengths, repeats, strict=True))
            print(
                f"Source {source['name']}: {source['path']} (actual share {ratio:.2%}) | "
                f"episodes: {len(lengths)} | samples: {samples} | virtual samples: {virtual_samples} | "
                f"episode copies: {min(repeats)}-{max(repeats)}"
            )
    name = g2_config.training_config_name(settings)
    norm_path = g2_config.normalization_cache_dir(settings) / g2_config.asset_id(settings) / "norm_stats.json"
    run_dir = g2_config.run_dir(settings)
    ckpt_dir = g2_config.checkpoint_dir(settings)
    norm_only = getattr(args, "norm_only", False)
    dry_run = getattr(args, "dry_run", False)
    resume_checkpoint = None
    resume_training = None
    if settings["checkpoints"]["resume"]:
        if norm_only:
            raise ValueError("Resume cannot be combined with --norm-only; use --no-resume for a new stats computation")
        from openpi.adapters.g2.resume import resolve_resume

        settings, extra_args, resume_checkpoint, resume_training = resolve_resume(settings, extra_args, ckpt_dir)
        norm_path = resume_checkpoint / "assets" / g2_config.asset_id(settings) / "norm_stats.json"
        skip_norm = True
    else:
        mode = settings["normalization"]["mode"]
        signature_path = norm_path.with_name("g2_norm_config.yaml")
        skip_norm = mode == "skip" or (
            mode == "auto" and norm_path.is_file() and signature_path.is_file()
            and g2_config.read_yaml(signature_path) == g2_config.normalization_signature(settings)
        )
    payload = {"settings": g2_config.public_settings(settings), "extra_args": list(extra_args)}
    if resume_checkpoint is not None:
        payload.update(resume_checkpoint=str(resume_checkpoint), training_state_config=resume_training)
        print(f"Resume checkpoint: {resume_checkpoint} (step={resume_checkpoint.name})")
        print(f"Target total steps: {settings['training']['num_train_steps']}; decay_steps: {settings['training']['decay_steps']}")
        print("Normalization: checkpoint assets (normalization.mode only applies to new training)")
    print(f"YAML: {args.config.resolve()}\nTrainConfig: {name}\nRun: {run_dir}\nCheckpoint: {ckpt_dir}")
    print(f"Norm stats: {norm_path} (skip={skip_norm})")
    visible = os.environ["CUDA_VISIBLE_DEVICES"]
    gpu_count = len([gpu for gpu in visible.split(",") if gpu.strip() and gpu.strip() != "-1"])
    batch = settings["training"]["batch_size"]
    fsdp = settings["training"]["fsdp_devices"]
    per_gpu = str(batch // gpu_count) if gpu_count and batch % gpu_count == 0 else "invalid division"
    groups = str(gpu_count // fsdp) if gpu_count and gpu_count % fsdp == 0 else "invalid division"
    print(
        f"GPU selection: {visible} | FSDP devices/group: {fsdp} | data-parallel groups: {groups} | "
        f"global batch: {batch} | batch/GPU: {per_gpu} (actual devices validated by JAX at training startup)"
    )
    print("Resolved experiment configuration:")
    # Only shorten the console view; workers and checkpoints retain the full snapshot.
    console_settings = dict(payload["settings"])
    if "mixture" in console_settings.get("resolved", {}):
        console_settings["resolved"] = dict(console_settings["resolved"])
        console_settings["resolved"]["mixture"] = {
            key: value
            for key, value in console_settings["resolved"]["mixture"].items()
            if key not in ("lengths", "plan")
        }
        print("Per-episode lengths and repeat plan omitted here; full details are saved in checkpoint assets.")
    print(yaml.safe_dump(console_settings, sort_keys=False, allow_unicode=True), flush=True)
    print("Derived policy contract:")
    print(yaml.safe_dump(g2_config.describe_policy(settings), sort_keys=False, allow_unicode=True), flush=True)
    if dry_run:
        if "sources" in settings["data"]:
            print("[dry-run] Read mixture metadata only; no video decoding, downloads, normalization, or training.")
        else:
            print("[dry-run] No data access, downloads, normalization, or training.")
        print(f"[1/2] {'reuse' if skip_norm else 'compute'} norm stats using the resolved config above")
        if not norm_only:
            print("[2/2] train using the same resolved config")
        return
    _check_dataset(settings)
    if (
        not norm_only
        and run_dir.exists()
        and not (settings["checkpoints"]["resume"] or settings["checkpoints"]["overwrite"])
    ):
        raise FileExistsError(f"Experiment exists: {run_dir}; choose a new experiment.name or use --run {run_dir}")
    if skip_norm and not norm_only and not norm_path.is_file():
        raise FileNotFoundError(f"Cannot skip normalization: {norm_path} does not exist")
    snapshot_text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    if not norm_only:
        check_wandb(
            enabled=settings["wandb"]["mode"] == "on"
            or (settings["wandb"]["mode"] == "auto" and not settings["run"]["smoke"]),
        )
        run_dir.mkdir(parents=True, exist_ok=settings["checkpoints"]["resume"] or settings["checkpoints"]["overwrite"])
        # Record the attempt immediately; workers and each checkpoint retain their own snapshot.
        record = run_dir / g2_config.CHECKPOINT_CONFIG
        if record.exists() and not (run_dir / f"initial_{g2_config.CHECKPOINT_CONFIG}").exists():
            (run_dir / f"initial_{g2_config.CHECKPOINT_CONFIG}").write_text(record.read_text())
        record.write_text(snapshot_text)
        with (run_dir / "train.log").open("a") as log:
            log.write(f"\nYAML: {args.config.resolve()}\nRun: {run_dir}\nCheckpoint: {ckpt_dir}\n")
            log.write(snapshot_text)
    with tempfile.TemporaryDirectory(prefix="g2-train-", dir="/tmp") as temporary:
        snapshot = Path(temporary) / g2_config.CHECKPOINT_CONFIG
        snapshot.write_text(snapshot_text)
        worker = [str(Path(__file__).resolve()), "--resolved-config", str(snapshot)]
        if not skip_norm:
            print("[1/2] Computing norm stats", flush=True)
            _run_command([*worker, "--stage", "norm"], log_path=None if norm_only else run_dir / "train.log")
        else:
            print(f"[1/2] Reusing {norm_path}", flush=True)
        if not norm_only:
            print("[2/2] Training", flush=True)
            _run_command([*worker, "--stage", "train"], log_path=run_dir / "train.log")
    print(f"Done: {norm_path if norm_only else ckpt_dir}")


if __name__ == "__main__":
    main()
