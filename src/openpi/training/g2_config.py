"""G2 YAML settings shared by normalization, training, and checkpoint loading.

Keep imports lightweight: reading/previewing settings must not initialize JAX.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from pathlib import Path
import re
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "scripts/g2/train/config.yaml"
CHECKPOINT_CONFIG = "g2_train_config.yaml"

# A single list maps compatibility CLI arguments to the YAML fields they override.
CLI_FIELDS = {
    "preset": ("experiment", "preset"),
    "exp_name": ("experiment", "name"),
    "arm": ("experiment", "arm"),
    "repo_id": ("data", "repo_id"),
    "data_root": ("data", "root"),
    "action_mode": ("data", "action_mode"),
    "action_horizon": ("model", "action_horizon"),
    "params_path": ("model", "params_path"),
    "norm": ("normalization", "mode"),
    "max_frames": ("normalization", "max_frames"),
    "checkpoint_base_dir": ("checkpoints", "base_dir"),
    "assets_base_dir": ("checkpoints", "assets_dir"),
    "overwrite": ("checkpoints", "overwrite"),
    "resume": ("checkpoints", "resume"),
    "wandb": ("wandb", "mode"),
    "wandb_project": ("wandb", "project"),
    "use_proxy": ("runtime", "use_proxy"),
    **{
        key: ("training", key)
        for key in (
            "batch_size",
            "num_workers",
            "num_train_steps",
            "log_interval",
            "save_interval",
            "keep_period",
            "seed",
            "fsdp_devices",
            "peak_lr",
            "warmup_steps",
            "decay_steps",
            "decay_lr",
        )
    },
}

SCHEMA = {
    "experiment": {"name", "preset", "arm"},
    "data": {"root", "repo_id", "action_space", "action_mode", "cameras"},
    "model": {"action_horizon", "discrete_state_input", "params_path"},
    "training": {
        "batch_size",
        "num_workers",
        "num_train_steps",
        "seed",
        "fsdp_devices",
        "log_interval",
        "save_interval",
        "keep_period",
        "peak_lr",
        "warmup_steps",
        "decay_steps",
        "decay_lr",
    },
    "checkpoints": {"base_dir", "assets_dir", "overwrite", "resume"},
    "normalization": {"mode", "max_frames"},
    "runtime": {"gpu", "mem_fraction", "use_proxy", "proxy"},
    "wandb": {"mode", "project"},
}


class _UniqueLoader(yaml.SafeLoader):
    pass


# Treat on/off as strings (W&B modes), and accept scientific notation such as 1e-5.
# Copy resolvers so importing this module never changes PyYAML's global behavior.
_UniqueLoader.yaml_implicit_resolvers = copy.deepcopy(yaml.SafeLoader.yaml_implicit_resolvers)
for _initial, _resolvers in _UniqueLoader.yaml_implicit_resolvers.items():
    _UniqueLoader.yaml_implicit_resolvers[_initial] = [
        (tag, pattern) for tag, pattern in _resolvers if tag != "tag:yaml.org,2002:bool"
    ]
_UniqueLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|false|True|False|TRUE|FALSE)$"), list("tTfF")
)
_UniqueLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float", re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+$"), list("-+0123456789.")
)


def _unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        result = yaml.load(stream, Loader=_UniqueLoader)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return result


def load_settings(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Custom YAML may override a subset of the canonical config's fields."""
    settings = read_yaml(DEFAULT_CONFIG)
    if path.resolve() != DEFAULT_CONFIG.resolve():
        overrides = read_yaml(path)
        for section, values in overrides.items():
            if section not in settings or not isinstance(values, dict):
                raise ValueError(f"Unknown YAML section or invalid mapping: {section}")
            unknown = values.keys() - settings[section].keys()
            if unknown:
                raise ValueError(f"Unknown YAML fields in {section}: {sorted(unknown)}")
            settings[section].update(values)
    return settings


def resolve_settings(settings: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(settings)
    preset = overrides.get("preset", result["experiment"]["preset"])
    if preset == "smoke":
        result["training"].update(batch_size=1, num_train_steps=20)
        result["experiment"]["name"] = ""  # keep smoke checkpoints out of the regular experiment
    for key, value in overrides.items():
        if key in CLI_FIELDS:
            section, field = CLI_FIELDS[key]
            result[section][field] = value
    if result["checkpoints"]["resume"]:
        result["checkpoints"]["overwrite"] = False
    validate_settings(result)
    return result


def selected_cameras(settings: dict[str, Any]) -> tuple[str, ...]:
    arm = settings["experiment"]["arm"]
    available = (
        ("head_color", "hand_left_color", "hand_right_color")
        if arm == "both"
        else (
            "head_color",
            f"hand_{arm}_color",
        )
    )
    cameras = settings["data"]["cameras"]
    return available if cameras is None else tuple(cameras)


def config_name(settings: dict[str, Any]) -> str:
    preset = settings["experiment"]["preset"]
    name = "pi05_g2_vr" if preset == "full_ft" else "pi05_g2_vr_low_mem"
    arm = settings["experiment"]["arm"]
    return name if arm == "both" else f"{name}_{arm}"


def validate_settings(settings: dict[str, Any]) -> None:
    if settings.keys() != SCHEMA.keys():
        raise ValueError(f"YAML sections must be {sorted(SCHEMA)}")
    for section, fields in SCHEMA.items():
        if not isinstance(settings[section], dict) or settings[section].keys() != fields:
            raise ValueError(f"YAML fields in {section} must be {sorted(fields)}")

    def choice(section, key, choices):
        value = settings[section][key]
        if value not in choices:
            raise ValueError(f"{section}.{key} must be one of {choices}, got {value!r}")

    choice("experiment", "preset", ("low_mem", "full_ft", "smoke"))
    choice("experiment", "arm", ("both", "left", "right"))
    choice("data", "action_space", ("pose",))
    choice("data", "action_mode", ("relative", "absolute"))
    choice("normalization", "mode", ("auto", "force", "skip"))
    choice("wandb", "mode", ("auto", "on", "off"))
    for section, key in (
        ("checkpoints", "overwrite"),
        ("checkpoints", "resume"),
        ("runtime", "use_proxy"),
        ("model", "discrete_state_input"),
    ):
        if type(settings[section][key]) is not bool:
            raise ValueError(f"{section}.{key} must be a YAML boolean (true/false)")
    for section, key, allow_zero, nullable in (
        ("model", "action_horizon", False, False),
        *(
            ("training", key, key in ("num_workers", "seed", "warmup_steps"), key in ("keep_period", "decay_steps"))
            for key in (
                "batch_size",
                "num_workers",
                "num_train_steps",
                "seed",
                "fsdp_devices",
                "log_interval",
                "save_interval",
                "keep_period",
                "warmup_steps",
                "decay_steps",
            )
        ),
        ("normalization", "max_frames", False, True),
    ):
        value = settings[section][key]
        if value is None and nullable:
            continue
        if type(value) is not int or value < (0 if allow_zero else 1):
            raise ValueError(f"{section}.{key} must be {'nonnegative' if allow_zero else 'positive'} integer")
    for key in ("peak_lr", "decay_lr"):
        value = settings["training"][key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.{key} must be a positive number (e.g. 2.5e-5)")
    fraction = settings["runtime"]["mem_fraction"]
    if type(fraction) not in (int, float) or not 0 < fraction <= 1:
        raise ValueError("runtime.mem_fraction must be in (0, 1]")
    for section, key in (
        ("data", "root"),
        ("data", "repo_id"),
        ("model", "params_path"),
        ("checkpoints", "base_dir"),
        ("checkpoints", "assets_dir"),
        ("wandb", "project"),
        ("runtime", "gpu"),
        ("runtime", "proxy"),
    ):
        if not isinstance(settings[section][key], str) or not settings[section][key].strip():
            raise ValueError(f"{section}.{key} must be a nonempty string (quote GPU IDs)")
    name = settings["experiment"]["name"]
    if not isinstance(name, str) or (name and (name in (".", "..") or "/" in name or "\\" in name)):
        raise ValueError("experiment.name must be a directory name, or an empty string for an automatic name")
    repo = Path(settings["data"]["repo_id"])
    if repo.is_absolute() or ".." in repo.parts:
        raise ValueError("data.repo_id must be relative to data.root without '..'")
    arm = settings["experiment"]["arm"]
    if arm != "both" and settings["data"]["action_mode"] == "absolute":
        raise ValueError("Single-arm training currently requires data.action_mode: relative")
    cameras = settings["data"]["cameras"]
    if cameras is not None and (
        not isinstance(cameras, list) or not cameras or any(not isinstance(item, str) for item in cameras)
    ):
        raise ValueError("data.cameras must be null or a list of camera names")
    cameras = selected_cameras(settings)
    allowed = (
        {"head_color", "hand_left_color", "hand_right_color"}
        if arm == "both"
        else {
            "head_color",
            f"hand_{arm}_color",
        }
    )
    if "head_color" not in cameras or len(cameras) != len(set(cameras)) or not set(cameras) <= allowed:
        raise ValueError(
            f"data.cameras must include head_color, use unique names, and be a subset of {sorted(allowed)}"
        )
    decay_steps = settings["training"]["decay_steps"] or settings["training"]["num_train_steps"]
    # Smoke deliberately runs only a few updates of the normal schedule.
    if settings["experiment"]["preset"] != "smoke" and decay_steps <= settings["training"]["warmup_steps"]:
        raise ValueError("training.decay_steps (or num_train_steps) must exceed warmup_steps")


def build_train_config(settings: dict[str, Any], extra_args: tuple[str, ...] = ()):
    """Build exactly the same TrainConfig for both stages and for inference."""
    from openpi.training import config as config_lib
    from openpi.training import optimizer
    from openpi.training import weight_loaders

    validate_settings(settings)
    base = config_lib.get_config(config_name(settings))
    train = settings["training"]
    model = dataclasses.replace(
        base.model,
        action_horizon=settings["model"]["action_horizon"],
        discrete_state_input=settings["model"]["discrete_state_input"],
    )
    data = dataclasses.replace(
        base.data,
        repo_id=settings["data"]["repo_id"],
        use_delta_actions=settings["data"]["action_mode"] == "relative",
        cameras=selected_cameras(settings),
    )
    # Preserve the normal warmup/decay schedule in a 20-step smoke run.
    decay_steps = train["decay_steps"] or (
        max(30_000, train["warmup_steps"] + 1)
        if settings["experiment"]["preset"] == "smoke"
        else train["num_train_steps"]
    )
    metadata = {
        **base.policy_metadata,
        "action_horizon": model.action_horizon,
    }
    config = dataclasses.replace(
        base,
        model=model,
        data=data,
        exp_name=settings["experiment"]["name"],
        weight_loader=weight_loaders.CheckpointWeightLoader(settings["model"]["params_path"]),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=train["warmup_steps"],
            peak_lr=train["peak_lr"],
            decay_steps=decay_steps,
            decay_lr=train["decay_lr"],
        ),
        **{
            key: train[key]
            for key in (
                "batch_size",
                "num_workers",
                "num_train_steps",
                "log_interval",
                "save_interval",
                "keep_period",
                "seed",
                "fsdp_devices",
            )
        },
        assets_base_dir=settings["checkpoints"]["assets_dir"],
        checkpoint_base_dir=settings["checkpoints"]["base_dir"],
        overwrite=settings["checkpoints"]["overwrite"],
        resume=settings["checkpoints"]["resume"],
        wandb_enabled=settings["wandb"]["mode"] == "on"
        or (settings["wandb"]["mode"] == "auto" and settings["experiment"]["preset"] != "smoke"),
        project_name=settings["wandb"]["project"],
        policy_metadata=metadata,
    )
    if extra_args:
        import tyro

        # Structural settings must be visible in YAML, not hidden behind a second parser.
        allowed = ("--optimizer.", "--ema-decay")
        if any(arg.startswith("--") and not arg.startswith(allowed) for arg in extra_args):
            raise ValueError(
                "Use YAML/normal CLI options for model, data and training; '--' only accepts optimizer/EMA"
            )
        config = tyro.cli(config_lib.TrainConfig, default=config, args=list(extra_args))
    return dataclasses.replace(config, g2_training_config={"settings": settings, "extra_args": list(extra_args)})
