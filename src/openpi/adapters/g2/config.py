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

from g2_policy_protocol import protocol
from g2_policy_protocol.data import policy as policy_config
from g2_policy_protocol.data.dataset import action_label_column
from g2_policy_protocol.data.encoding import action_dimensions
import yaml

from openpi.adapters.g2 import training_config as training_schema

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = PROJECT_ROOT / "scripts/g2/configs"
DEFAULT_CONFIG = CONFIG_DIR / "train.yaml"
DEFAULT_DEPLOY_CONFIG = CONFIG_DIR / "deploy.yaml"
CHECKPOINT_CONFIG = "g2_train_config.yaml"
MODEL_CONFIG_NAMES = {
    ("lora", "lora"): "pi05_g2_vr_low_mem",
    ("lora", "full"): "pi05_g2_vr_vlm_lora_action_full",
    ("full", "full"): "pi05_g2_vr",
}

# A single list maps CLI arguments to the YAML fields they override.
CLI_FIELDS = {
    "smoke": ("run", "smoke"),
    "exp_name": ("experiment", "name"),
    "arm": ("experiment", "arm"),
    "repo_id": ("data", "repo_id"),
    "data_root": ("data", "root"),
    "action_horizon": ("policy", "timing", "action_horizon"),
    "params_path": ("model", "params_path"),
    "norm": ("normalization", "mode"),
    "max_frames": ("normalization", "max_frames"),
    "checkpoint_base_dir": ("checkpoints", "base_dir"),
    "assets_base_dir": ("checkpoints", "assets_dir"),
    "overwrite": ("checkpoints", "overwrite"),
    "resume": ("checkpoints", "resume"),
    "wandb": ("wandb", "mode"),
    "wandb_project": ("wandb", "project"),
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

# Default numerical contract consumed by the shared dataset and geometry tools.

POLICY_DEFAULTS = policy_config.resolve_policy(
    {
        "action": {
            "pose": {
                "delta": True,
                "method": "component",
                "reference": {"source": "state", "scope": "chunk"},
            }
        }
    },
    horizon=50,
)

SCHEMA = {
    "schema_version": None,
    "experiment": {"name", "arm"},
    "finetuning": {
        module: {"mode": None, "lora": {"rank", "alpha", "targets", "rslora"}}
        for module in ("vision", "language", "action_expert")
    },
    "run": {"smoke"},
    "data": {"root", "repo_id", "cameras"},
    "policy": None,
    "model": {"discrete_state_input", "params_path"},
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
    "runtime": {"gpus", "mem_fraction"},
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


def normalize_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Validate a self-contained public checkpoint at its loading boundary."""
    result = training_schema.to_native(settings)
    for group in ("state", "action"):
        coordinates = result["policy"][group]["pose"]["coordinates"]
        for key, value in policy_config.COORDINATES.items():
            coordinates.setdefault(key, copy.deepcopy(value))
    validate_settings(result)
    return result


def public_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return training_schema.openpi_snapshot(settings)


def load_settings(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    raw = read_yaml(DEFAULT_CONFIG)
    # Calibration stays in the resolved contract, so experiment YAML can select frames only.
    for group in ("state", "action"):
        coordinates = copy.deepcopy(policy_config.COORDINATES)
        policy_config.merge(
            coordinates, raw["policy"][group]["pose"]["coordinates"], f"policy.{group}.pose.coordinates"
        )
        raw["policy"][group]["pose"]["coordinates"] = coordinates
    if path.resolve() != DEFAULT_CONFIG.resolve():
        override = read_yaml(path)
        policy_config.merge(raw, override)
    return training_schema.to_native(raw)


def resolve_settings(settings: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(settings)
    overrides = dict(overrides)
    for module in ("vision", "language", "action_expert"):
        if f"{module}_mode" in overrides:
            result["finetuning"][module]["mode"] = overrides.pop(f"{module}_mode")
    if overrides.get("smoke", result["run"]["smoke"]):
        if result["training"]["decay_steps"] is None:
            # Keep the selected experiment's LR schedule while shortening its run.
            result["training"]["decay_steps"] = max(
                result["training"]["num_train_steps"], result["training"]["warmup_steps"] + 1
            )
        result["training"].update(batch_size=1, num_train_steps=20)
        result["experiment"]["name"] = ""  # keep smoke checkpoints out of the regular experiment
    for key, value in overrides.items():
        if key in CLI_FIELDS:
            target = result
            *parents, field = CLI_FIELDS[key]
            for parent in parents:
                target = target[parent]
            target[field] = value
    if result["checkpoints"]["resume"]:
        result["checkpoints"]["overwrite"] = False
    validate_settings(result)
    return result


def cameras_for_arm(arm: str, cameras: list | tuple | None = None) -> tuple[str, ...]:
    available = protocol.cameras_for_arm(arm)
    selected = available if cameras is None else tuple(cameras)
    if len(set(selected)) != len(selected) or not set(selected) <= set(available):
        raise ValueError(f"cameras must be unique names from {available} for arm={arm}, got {selected!r}")
    return selected


def needs_joint_state(policy: dict[str, Any]) -> bool:
    return policy["state"]["joint"]["enabled"] or policy["action"]["joint"]["enabled"]


def selected_cameras(settings: dict[str, Any]) -> tuple[str, ...]:
    return cameras_for_arm(settings["experiment"]["arm"], settings["data"]["cameras"])


def training_config_name(settings: dict[str, Any]) -> str:
    modes = finetuning_modes(settings)
    name = MODEL_CONFIG_NAMES.get(
        (modes["language"], modes["action_expert"]),
        f"pi05_g2_vr_v{modes['vision']}_l{modes['language']}_a{modes['action_expert']}",
    )
    arm = settings["experiment"]["arm"]
    return name if arm == "both" else f"{name}_{arm}"


def _validate_fields(value, schema, path="settings"):
    if not isinstance(value, dict) or value.keys() != schema.keys():
        raise ValueError(f"{path} fields must be {sorted(schema)}")
    for key, fields in schema.items():
        if isinstance(fields, dict):
            _validate_fields(value[key], fields, f"{path}.{key}")
        elif isinstance(fields, set) and (not isinstance(value[key], dict) or value[key].keys() != fields):
            raise ValueError(f"{path}.{key} fields must be {sorted(fields)}")


def validate_settings(settings: dict[str, Any]) -> None:
    _validate_fields(settings, SCHEMA)
    if settings["schema_version"] != training_schema.VERSION:
        raise ValueError("New training configs require schema_version: 7")
    policy_config.validate_policy(settings["policy"])
    if state_dimension(settings) > 32 and not settings["model"]["discrete_state_input"]:
        raise ValueError("Combined/history states wider than 32 require discrete_state_input: true")

    def choice(section, key, choices):
        value = settings[section][key]
        if value not in choices:
            raise ValueError(f"{section}.{key} must be one of {choices}, got {value!r}")

    for module, spec in settings["finetuning"].items():
        mode, options = spec["mode"], spec["lora"]
        path = f"finetuning.{module}"
        if mode not in ("frozen", "lora", "full"):
            raise ValueError(f"{path}.mode must be frozen, lora or full")
        if mode != "lora":
            continue
        if type(options["rank"]) is not int or options["rank"] <= 0:
            raise ValueError(f"{path}.lora.rank must be a positive integer")
        if type(options["alpha"]) not in (int, float) or not math.isfinite(options["alpha"]) or options["alpha"] <= 0:
            raise ValueError(f"{path}.lora.alpha must be a positive finite number")
        if type(options["rslora"]) is not bool:
            raise ValueError(f"{path}.lora.rslora must be a YAML boolean")
        targets = options["targets"]
        allowed = ("attention",) if module == "vision" else ("attention", "ffn")
        if (
            not isinstance(targets, list)
            or not targets
            or any(not isinstance(target, str) or target not in allowed for target in targets)
            or len(set(targets)) != len(targets)
        ):
            raise ValueError(f"{path}.lora.targets must be a nonempty list of unique values from {allowed}")
    if all(mode == "frozen" for mode in finetuning_modes(settings).values()):
        raise ValueError("At least one finetuning module must be trainable")
    if type(settings["run"]["smoke"]) is not bool:
        raise ValueError("run.smoke must be a YAML boolean")
    choice("experiment", "arm", ("both", "left", "right"))
    choice("normalization", "mode", ("auto", "force", "skip"))
    choice("wandb", "mode", ("auto", "on", "off"))
    for section, key in (
        ("checkpoints", "overwrite"),
        ("checkpoints", "resume"),
        ("model", "discrete_state_input"),
    ):
        if type(settings[section][key]) is not bool:
            raise ValueError(f"{section}.{key} must be a YAML boolean (true/false)")
    for section, key, allow_zero, nullable in (
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
    gpus = settings["runtime"]["gpus"]
    if (
        not isinstance(gpus, list)
        or not gpus
        or any(type(gpu) is not int or gpu < 0 for gpu in gpus)
        or len(set(gpus)) != len(gpus)
    ):
        raise ValueError("runtime.gpus must be a nonempty list of unique nonnegative GPU IDs")
    for section, key in (
        ("data", "root"),
        ("data", "repo_id"),
        ("model", "params_path"),
        ("checkpoints", "base_dir"),
        ("checkpoints", "assets_dir"),
        ("wandb", "project"),
    ):
        if not isinstance(settings[section][key], str) or not settings[section][key].strip():
            raise ValueError(f"{section}.{key} must be a nonempty string (quote GPU IDs)")
    name = settings["experiment"]["name"]
    if not isinstance(name, str) or (name and (name in (".", "..") or "/" in name or "\\" in name)):
        raise ValueError("experiment.name must be a directory name, or an empty string for an automatic name")
    repo = Path(settings["data"]["repo_id"])
    if repo.is_absolute() or ".." in repo.parts:
        raise ValueError("data.repo_id must be relative to data.root without '..'")
    cameras = settings["data"]["cameras"]
    if cameras is not None and (not isinstance(cameras, list) or any(not isinstance(item, str) for item in cameras)):
        raise ValueError("data.cameras must be null or a list of camera names")
    selected_cameras(settings)
    decay_steps = settings["training"]["decay_steps"] or settings["training"]["num_train_steps"]
    # Smoke deliberately runs only a few updates of the normal schedule.
    if (not settings["run"]["smoke"] or settings["training"]["decay_steps"] is not None) and decay_steps <= settings[
        "training"
    ]["warmup_steps"]:
        raise ValueError("training.decay_steps (or num_train_steps) must exceed warmup_steps")


def build_train_config(
    settings: dict[str, Any], extra_args: tuple[str, ...] = (), *,
    resume_checkpoint: Path | None = None, resume_training: dict | None = None,
):
    """Build exactly the same TrainConfig for both stages and for inference."""
    from openpi.adapters.g2 import presets
    from openpi.models.pi0_config import LoRAOptions
    from openpi.training import optimizer
    from openpi.training import weight_loaders

    name = training_config_name(settings)
    arm = settings["experiment"]["arm"]
    # Native model variants and masks come only from the explicit module modes.
    base = presets.base_config(arm)
    train = settings["training"]
    policy = settings["policy"]
    modes = finetuning_modes(settings)
    adapters = {
        f"{module}_lora": LoRAOptions(
            rank=spec["lora"]["rank"],
            alpha=spec["lora"]["alpha"],
            targets=tuple(spec["lora"]["targets"]),
            rslora=spec["lora"]["rslora"],
        )
        if spec["mode"] == "lora"
        else None
        for module, spec in settings["finetuning"].items()
    }
    model = dataclasses.replace(
        base.model,
        paligemma_variant="gemma_2b_lora" if modes["language"] == "lora" else "gemma_2b",
        action_expert_variant="gemma_300m_lora" if modes["action_expert"] == "lora" else "gemma_300m",
        action_horizon=policy["timing"]["action_horizon"],
        discrete_state_input=settings["model"]["discrete_state_input"],
        **adapters,
        max_token_len=max(base.model.max_token_len, 6 * state_dimension(settings) + 96)
        if state_dimension(settings) > 32
        else base.model.max_token_len,
    )
    data = dataclasses.replace(
        base.data,
        repo_id=settings["data"]["repo_id"],
        policy=copy.deepcopy(policy),
        cameras=selected_cameras(settings),
    )
    # Preserve the normal warmup/decay schedule in a 20-step smoke run.
    decay_steps = train["decay_steps"] or (
        max(30_000, train["warmup_steps"] + 1) if settings["run"]["smoke"] else train["num_train_steps"]
    )
    metadata = {
        **base.policy_metadata,
        "action_horizon": model.action_horizon,
        "policy_hz": policy["timing"]["sample_hz"] / policy["timing"]["stride_steps"],
        "policy_contract": copy.deepcopy(policy),
    }
    config = dataclasses.replace(
        base,
        name=name,
        model=model,
        freeze_filter=presets.finetuning_filter(model, modes),
        ema_decay=None if "lora" in (modes["language"], modes["action_expert"]) else base.ema_decay,
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
        or (settings["wandb"]["mode"] == "auto" and not settings["run"]["smoke"]),
        project_name=settings["wandb"]["project"],
        policy_metadata=metadata,
    )
    if extra_args:
        import tyro

        # Parse only the advertised advanced fields. Traversing the entire TrainConfig
        # also visits adapter datasets and their policy dicts, which aren't CLI types.
        advanced_type = dataclasses.make_dataclass(
            "AdvancedOverrides", [("optimizer", type(config.optimizer)), ("ema_decay", float | None)]
        )
        advanced = tyro.cli(
            advanced_type, default=advanced_type(config.optimizer, config.ema_decay), args=list(extra_args)
        )
        config = dataclasses.replace(config, optimizer=advanced.optimizer, ema_decay=advanced.ema_decay)
    if resume_training is not None:
        config = dataclasses.replace(
            config, optimizer=dataclasses.replace(config.optimizer, **resume_training["optimizer"]),
            ema_decay=resume_training["ema_decay"],
            lr_schedule=dataclasses.replace(config.lr_schedule, **resume_training["lr_schedule"]),
        )
    if resume_checkpoint is not None:
        config = dataclasses.replace(
            config, resume_step=int(resume_checkpoint.name),
            data=dataclasses.replace(config.data, assets=dataclasses.replace(
                config.data.assets, assets_dir=str(resume_checkpoint / "assets"),
            )),
        )
    snapshot = public_settings(settings)
    return dataclasses.replace(
        config,
        checkpoint_assets={
            CHECKPOINT_CONFIG: yaml.safe_dump(
                {
                    "settings": snapshot, "extra_args": list(extra_args),
                    "training_state_config": {
                        "optimizer": dataclasses.asdict(config.optimizer), "ema_decay": config.ema_decay,
                        "lr_schedule": dataclasses.asdict(config.lr_schedule),
                    },
                }, sort_keys=False
            )
        },
    )


def validate_extra_args(extra_args: tuple[str, ...]) -> None:
    """Keep structural overrides out of the optimizer/EMA-only advanced CLI."""
    if any(arg.startswith("--") and not arg.startswith(("--optimizer.", "--ema-decay")) for arg in extra_args):
        raise ValueError("Configure model/data/training in YAML; '--' only accepts optimizer/EMA overrides")


def policy_semantics(settings: dict[str, Any]) -> dict[str, Any]:
    """Numerical input/label semantics; disabled branches do not affect statistics or resume."""
    policy = copy.deepcopy(settings["policy"])
    for group in ("state", "action"):
        policy[group] = {kind: spec for kind, spec in policy[group].items() if spec["enabled"]}
    return policy


def finetuning_modes(settings: dict[str, Any]) -> dict[str, str]:
    return {module: spec["mode"] for module, spec in settings["finetuning"].items()}


def model_semantics(settings: dict[str, Any]) -> dict[str, Any]:
    model = copy.deepcopy(settings["model"])
    model["lora"] = {
        module: {**copy.deepcopy(spec["lora"]), "targets": sorted(spec["lora"]["targets"])}
        for module, spec in settings["finetuning"].items()
        if spec["mode"] == "lora"
    }
    return model


def state_dimension(settings):
    policy = settings["policy"]
    arms = 2 if settings["experiment"]["arm"] == "both" else 1
    total = 0
    if policy["state"]["pose"]["enabled"]:
        rotation = policy["state"]["pose"]["coordinates"]["rotation"]
        total += arms * (4 + {"rot6d": 6, "quaternion": 4, "axis_angle": 3}[rotation])
    if policy["state"]["joint"]["enabled"]:
        total += arms * 8
    return total * policy["timing"]["history_steps"]


def describe_policy(settings: dict[str, Any]) -> dict[str, Any]:
    """Derived dimensions and timestamps, without loading model dependencies."""
    policy = settings["policy"]
    timing = policy["timing"]
    action_dim, _ = action_dimensions(policy, settings["experiment"]["arm"])
    first = timing["target_offset_steps"]
    last = first + (timing["action_horizon"] - 1) * timing["stride_steps"]
    columns = [f"state.{kind}" for kind in ("pose", "joint") if policy["state"][kind]["enabled"]]
    return {
        "state": copy.deepcopy(policy["state"]),
        "state_column": columns[0] if len(columns) == 1 else columns,
        "state_dim": state_dimension(settings),
        "label_column": action_label_column(policy),
        "action_dim": action_dim,
        "prediction_shape": [timing["action_horizon"], action_dim],
        "target_offsets_steps": [first, last],
        "last_target_time_seconds": last / timing["sample_hz"],
        "action_sample_hz": timing["sample_hz"] / timing["stride_steps"],
        "action": copy.deepcopy(policy["action"]),
        "gripper_encoding": "absolute_close_ratio",
    }
