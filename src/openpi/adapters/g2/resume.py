"""Resolve G2 resume metadata once, before data access or model initialization."""

import copy
import json
from pathlib import Path

from openpi.adapters.g2 import config as g2_config

RUN_FIELDS = {"num_train_steps", "num_workers", "log_interval", "save_interval", "keep_period", "fsdp_devices"}


def resolve_resume(settings, extra_args, checkpoint_dir: Path):
    candidates = sorted(
        (path for path in checkpoint_dir.glob("*") if path.is_dir() and path.name.isdecimal()),
        key=lambda path: int(path.name),
        reverse=True,
    )
    for checkpoint in candidates:
        required = ("_CHECKPOINT_METADATA", "train_state/_METADATA", "params/_METADATA")
        if not all((checkpoint / name).is_file() for name in required):
            continue
        metadata = json.loads((checkpoint / "_CHECKPOINT_METADATA").read_text())
        if metadata.get("commit_timestamp_nsecs"):
            break
    else:
        raise FileNotFoundError(
            f"Resume requires a complete checkpoint in {checkpoint_dir}; no new training was started"
        )

    saved = g2_config.read_yaml(checkpoint / "assets" / g2_config.CHECKPOINT_CONFIG)
    old = g2_config.normalize_settings(saved["settings"])
    differences = []
    if g2_config.finetuning_modes(old) != g2_config.finetuning_modes(settings):
        differences.append("finetuning modes")
    if g2_config.model_semantics(old) != g2_config.model_semantics(settings):
        differences.append("model settings")
    if old["data"] != settings["data"]:
        differences.append("data settings")
    if (
        g2_config.policy_semantics(old) != g2_config.policy_semantics(settings)
        or old["experiment"]["arm"] != settings["experiment"]["arm"]
    ):
        differences.append("policy semantics")
    state_config = saved.get("training_state_config")
    decay_steps = old["training"]["decay_steps"] or (
        max(30_000, old["training"]["warmup_steps"] + 1) if old["run"]["smoke"] else old["training"]["num_train_steps"]
    )
    if state_config is not None:
        decay_steps = state_config["lr_schedule"]["decay_steps"]
    for key, value in settings["training"].items():
        if key in RUN_FIELDS:
            continue
        # null means keep the checkpoint's resolved schedule, even when total steps change.
        if key == "decay_steps":
            if value is not None and value != decay_steps:
                differences.append("training.decay_steps")
        elif value != old["training"][key]:
            differences.append(f"training.{key}")
    saved_extra = tuple(saved.get("extra_args", ()))
    if extra_args and tuple(extra_args) != saved_extra:
        differences.append("optimizer/EMA overrides")
    if settings["wandb"]["project"] != old["wandb"]["project"]:
        differences.append("wandb.project")
    if settings["run"]["smoke"] != old["run"]["smoke"]:
        differences.append("run.smoke")
    if differences:
        raise ValueError(f"Cannot resume with changed {', '.join(differences)}; use a new experiment.name")
    if settings["training"]["num_train_steps"] <= int(checkpoint.name):
        raise ValueError(f"Target num_train_steps must exceed checkpoint step {checkpoint.name}")
    norm_path = checkpoint / "assets" / old["data"]["repo_id"] / "norm_stats.json"
    if not norm_path.is_file():
        raise FileNotFoundError(f"Resume requires checkpoint normalization stats: {norm_path}")
    resolved = copy.deepcopy(old)
    resolved["training"].update({key: settings["training"][key] for key in RUN_FIELDS})
    resolved["training"]["decay_steps"] = decay_steps
    resolved["runtime"] = copy.deepcopy(settings["runtime"])
    resolved["checkpoints"] = copy.deepcopy(settings["checkpoints"])
    # Disabling external logging is independent of the training state. Enabling it
    # on a run without a W&B ID is rejected before model initialization by init_wandb.
    resolved["wandb"]["mode"] = settings["wandb"]["mode"]
    return resolved, saved_extra, checkpoint, state_config
