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

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = PROJECT_ROOT / "scripts/g2/train/config.yaml"
CHECKPOINT_CONFIG = "g2_train_config.yaml"

# A single list maps compatibility CLI arguments to the YAML fields they override.
CLI_FIELDS = {
    "preset": ("experiment", "preset"),
    "exp_name": ("experiment", "name"),
    "arm": ("experiment", "arm"),
    "repo_id": ("data", "repo_id"),
    "data_root": ("data", "root"),
    "state_space": ("policy", "state", "space"),
    "action_representation": ("policy", "action", "representation"),
    "prediction_steps": ("policy", "timing", "prediction_steps"),
    "action_horizon": ("policy", "timing", "prediction_steps"),  # legacy CLI alias
    "execution_steps": ("policy", "timing", "execution_steps"),
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

# Defaults are versioned in code so old checkpoints never inherit today's YAML.
POLICY_DEFAULTS = {
    "pose": {
        "parent_frame": "base_link",
        "ee_frame": "flange",
        "rotation": "rot6d",
        "base_to_parent": None,
        "flange_to_tcp": None,
    },
    "state": {"space": "eef", "representation": "absolute", "relative_to": None},
    "action": {
        "space": "eef",
        "control_quantity": "pose_target",
        "label_source": "command",
        "representation": "component_delta",
        "relative_to": "current_measured_pose",
        "delta_scope": "chunk",
    },
    "timing": {
        "sample_hz": 30.0,
        "history_steps": 1,
        "prediction_steps": 50,
        "execution_steps": None,
        "target_offset_steps": 0,
        "stride_steps": 1,
    },
}

SCHEMA = {
    "schema_version": None,
    "experiment": {"name", "preset", "arm"},
    "data": {"root", "repo_id", "cameras"},
    "policy": {key: set(fields) for key, fields in POLICY_DEFAULTS.items()},
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


def upgrade_settings(settings: dict[str, Any], *, partial: bool = False) -> dict[str, Any]:
    """Translate v1 fields without reading a template; preserve old state semantics."""
    result = copy.deepcopy(settings)
    version = result.get("schema_version", 1)
    if type(version) is not int or version not in (1, 2):
        raise ValueError("schema_version must be 1 or 2")
    data = result.get("data", {})
    model = result.get("model", {})
    if not isinstance(data, dict) or not isinstance(model, dict):
        raise ValueError("data and model must be YAML mappings")
    legacy = {"action_space", "action_mode", "state_space"}.intersection(data) or "action_horizon" in model
    if legacy and (version == 2 or "policy" in result):
        raise ValueError(
            "Cannot mix legacy data.action_mode/action_space/state_space or model.action_horizon with v2 policy"
        )
    if not partial and "policy" in result and isinstance(result["policy"], dict):
        pose = result["policy"].get("pose", {})
        if isinstance(pose, dict):
            pose.setdefault("base_to_parent", None)
            pose.setdefault("flange_to_tcp", None)
    if not legacy:
        if "schema_version" in result or not partial:
            result["schema_version"] = 2
        return result

    policy = {} if partial else copy.deepcopy(POLICY_DEFAULTS)
    if "action_space" in data:
        space = data.pop("action_space")
        policy.setdefault("action", {})["space"] = "eef" if space == "pose" else space
    if "action_mode" in data:
        mode = data.pop("action_mode")
        if mode not in ("relative", "absolute"):
            raise ValueError("legacy data.action_mode must be relative or absolute")
        relative = mode == "relative"
        policy.setdefault("state", {})["space"] = "eef" if relative else "joint"
        policy.setdefault("action", {}).update(
            representation="component_delta" if relative else "absolute",
            relative_to="current_measured_pose" if relative else None,
            delta_scope="chunk" if relative else None,
        )
    if "state_space" in data:
        space = data.pop("state_space")
        policy.setdefault("state", {})["space"] = "eef" if space == "pose" else space
    if "action_horizon" in model:
        policy.setdefault("timing", {})["prediction_steps"] = model.pop("action_horizon")
    result.update(schema_version=2, policy=policy)
    return result


def _merge_settings(base: dict, overrides: dict, path: str = "") -> None:
    for key, value in overrides.items():
        name = f"{path}.{key}" if path else key
        if key not in base:
            raise ValueError(f"Unknown YAML field: {name}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a YAML mapping")
            _merge_settings(base[key], value, name)
        else:
            base[key] = copy.deepcopy(value)


def load_settings(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Recursively merge a partial YAML; v1 files are translated before merging."""
    settings = upgrade_settings(read_yaml(DEFAULT_CONFIG))
    if path.resolve() != DEFAULT_CONFIG.resolve():
        overrides = upgrade_settings(read_yaml(path), partial=True)
        policy = overrides.get("policy", {})
        if not isinstance(policy, dict):
            raise ValueError("policy must be a YAML mapping")
        action = policy.get("action", {})
        if not isinstance(action, dict):
            raise ValueError("policy.action must be a YAML mapping")
        if action.get("representation") == "absolute":
            action.setdefault("relative_to", None)
            action.setdefault("delta_scope", None)
        _merge_settings(settings, overrides)
    return settings


def resolve_settings(settings: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = upgrade_settings(settings)
    preset = overrides.get("preset", result["experiment"]["preset"])
    if preset == "smoke":
        result["training"].update(batch_size=1, num_train_steps=20)
        result["experiment"]["name"] = ""  # keep smoke checkpoints out of the regular experiment
    overrides = dict(overrides)
    if "action_mode" in overrides:  # CLI alias changes actions only, not the state input.
        mode = overrides.pop("action_mode")
        if mode not in ("relative", "absolute"):
            raise ValueError("--action-mode must be relative or absolute")
        representation = "component_delta" if mode == "relative" else "absolute"
        if overrides.get("action_representation", representation) != representation:
            raise ValueError("--action-mode disagrees with --action-representation")
        overrides["action_representation"] = representation
    if (
        "action_horizon" in overrides
        and "prediction_steps" in overrides
        and overrides["action_horizon"] != overrides["prediction_steps"]
    ):
        raise ValueError("--action-horizon disagrees with --prediction-steps")
    if "action_representation" in overrides:
        action = result["policy"]["action"]
        absolute = overrides["action_representation"] == "absolute"
        action.update(
            relative_to=None if absolute else "current_measured_pose", delta_scope=None if absolute else "chunk"
        )
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


def _validate_fields(value, schema, path="settings"):
    if not isinstance(value, dict) or value.keys() != schema.keys():
        raise ValueError(f"{path} fields must be {sorted(schema)}")
    for key, fields in schema.items():
        if isinstance(fields, dict):
            _validate_fields(value[key], fields, f"{path}.{key}")
        elif isinstance(fields, set) and (not isinstance(value[key], dict) or value[key].keys() != fields):
            raise ValueError(f"{path}.{key} fields must be {sorted(fields)}")


def _validate_policy(policy: dict[str, Any]) -> None:
    pose, state, action, timing = (policy[key] for key in ("pose", "state", "action", "timing"))

    def choice(group, key, choices):
        if policy[group][key] not in choices:
            raise ValueError(f"policy.{group}.{key} must be one of {choices}")

    for key in ("parent_frame", "ee_frame", "rotation"):
        if not isinstance(pose[key], str) or not pose[key].strip():
            raise ValueError(f"policy.pose.{key} must be a nonempty string")
    choice("state", "space", ("joint", "eef", "both"))
    choice("state", "representation", ("absolute", "relative"))
    choice("action", "space", ("joint", "eef"))
    choice(
        "action", "control_quantity", ("pose_target", "joint_position", "joint_velocity", "eef_twist", "joint_torque")
    )
    choice("action", "label_source", ("command", "measured"))
    choice(
        "action",
        "representation",
        ("absolute", "component_delta", "se3_body_delta", "se3_spatial_delta", "hybrid_delta"),
    )
    for group in ("state", "action"):
        relative_to = policy[group]["relative_to"]
        if relative_to is not None and (not isinstance(relative_to, str) or not relative_to.strip()):
            raise ValueError(f"policy.{group}.relative_to must be null or a nonempty string")
    if state["representation"] == "absolute" and state["relative_to"] is not None:
        raise ValueError("policy.state.relative_to must be null for absolute state")
    if state["representation"] == "relative" and state["relative_to"] is None:
        raise ValueError("policy.state.relative_to must specify a reference for relative state")
    if action["representation"] == "absolute":
        if action["relative_to"] is not None or action["delta_scope"] is not None:
            raise ValueError("policy.action.relative_to and delta_scope must be null for absolute actions")
    elif action["relative_to"] is None or action["delta_scope"] not in ("chunk", "step"):
        raise ValueError("Delta actions must specify relative_to and delta_scope: chunk or step")
    for key in ("history_steps", "prediction_steps", "execution_steps", "target_offset_steps", "stride_steps"):
        value = timing[key]
        if key == "execution_steps" and value is None:
            continue
        if type(value) is not int or value < (0 if key == "target_offset_steps" else 1):
            raise ValueError(f"policy.timing.{key} must be a valid integer step count")
    hz = timing["sample_hz"]
    if type(hz) not in (int, float) or not math.isfinite(hz) or hz <= 0:
        raise ValueError("policy.timing.sample_hz must be a positive finite number")
    if timing["execution_steps"] is not None and timing["execution_steps"] > timing["prediction_steps"]:
        raise ValueError("policy.timing.execution_steps must not exceed prediction_steps")

    choice("pose", "rotation", ("rot6d", "quaternion", "axis_angle"))
    references = (
        "current_measured_pose",
        "current_command_pose",
        "previous_command_pose",
        "episode_start_pose",
        "object_pose",
    )
    if state["representation"] == "relative" and state["relative_to"] not in references:
        raise ValueError(f"policy.state.relative_to must be one of {references}")
    if (
        state["representation"] == "relative"
        and state["space"] in ("joint", "both")
        and state["relative_to"] not in ("episode_start_pose", "object_pose")
    ):
        raise ValueError("Relative joint state must use episode_start_pose or object_pose (with object_joint)")
    if action["representation"] != "absolute" and action["relative_to"] not in references:
        raise ValueError(f"policy.action.relative_to must be one of {references}")
    quantity = action["control_quantity"]
    expected_space = "eef" if quantity in ("pose_target", "eef_twist") else "joint"
    if action["space"] != expected_space:
        raise ValueError(f"{quantity} requires action.space: {expected_space}")
    if quantity in ("joint_velocity", "eef_twist", "joint_torque") and action["representation"] != "absolute":
        raise ValueError("Velocity/twist/torque control values must use absolute representation")
    if quantity == "joint_position" and action["representation"] not in ("absolute", "component_delta"):
        raise ValueError("Joint positions support absolute or component_delta representation")
    if (
        quantity == "joint_position"
        and action["representation"] == "component_delta"
        and action["relative_to"] != "current_measured_pose"
    ):
        raise ValueError("Joint deltas use current measured joints (current_measured_pose reference)")
    matrices = [("base_to_parent", pose["base_to_parent"])]
    if isinstance(pose["flange_to_tcp"], dict) and set(pose["flange_to_tcp"]) == {"left", "right"}:
        matrices.extend((f"flange_to_tcp.{side}", pose["flange_to_tcp"][side]) for side in ("left", "right"))
    for name, matrix in matrices:
        if name == "base_to_parent" and (matrix is None or (isinstance(matrix, str) and matrix == "observation")):
            continue
        _validate_rigid_transform(matrix, f"policy.pose.{name}")
    if pose["flange_to_tcp"] is not None and (
        not isinstance(pose["flange_to_tcp"], dict) or set(pose["flange_to_tcp"]) != {"left", "right"}
    ):
        raise ValueError("policy.pose.flange_to_tcp must contain left and right 4x4 transforms")
    if pose["parent_frame"] != "base_link" and pose["base_to_parent"] is None:
        raise ValueError("A different parent_frame requires base_to_parent calibration or 'observation'")
    if pose["ee_frame"] != "flange" and pose["flange_to_tcp"] is None:
        raise ValueError("A different ee_frame requires flange_to_tcp calibration")
    if pose["ee_frame"] == "flange" and pose["flange_to_tcp"] is not None:
        raise ValueError("flange_to_tcp requires a named TCP ee_frame")
    if quantity == "eef_twist" and pose["flange_to_tcp"] is not None:
        # A TCP twist requires per-target lever arms, not just the observation pose.
        raise ValueError("eef_twist uses flange; TCP velocity transport requires recorded TCP twist labels")


def _validate_rigid_transform(matrix, name):
    if (
        not isinstance(matrix, list)
        or len(matrix) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in matrix)
    ):
        raise ValueError(f"{name} must be a 4x4 rigid transform")
    if any(type(v) not in (int, float) or not math.isfinite(v) for row in matrix for v in row):
        raise ValueError(f"{name} must contain finite numbers")
    if any(abs(a - b) > 1e-6 for a, b in zip(matrix[3], [0, 0, 0, 1], strict=True)):
        raise ValueError(f"{name} must have homogeneous last row [0,0,0,1]")
    for i in range(3):
        for j in range(3):
            if abs(sum(matrix[k][i] * matrix[k][j] for k in range(3)) - int(i == j)) > 1e-5:
                raise ValueError(f"{name} rotation must be orthonormal")
    r = matrix
    determinant = (
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    )
    if abs(determinant - 1) > 1e-5:
        raise ValueError(f"{name} rotation determinant must be +1")


def validate_settings(settings: dict[str, Any]) -> None:
    _validate_fields(settings, SCHEMA)
    if type(settings["schema_version"]) is not int or settings["schema_version"] != 2:
        raise ValueError("schema_version must be 2 after migration")
    _validate_policy(settings["policy"])
    if state_dimension(settings) > 32 and not settings["model"]["discrete_state_input"]:
        raise ValueError("Combined/history states wider than 32 require discrete_state_input: true")

    def choice(section, key, choices):
        value = settings[section][key]
        if value not in choices:
            raise ValueError(f"{section}.{key} must be one of {choices}, got {value!r}")

    choice("experiment", "preset", ("low_mem", "full_ft", "smoke"))
    choice("experiment", "arm", ("both", "left", "right"))
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
    settings = upgrade_settings(settings)
    validate_settings(settings)

    from openpi.adapters.g2 import presets as config_lib
    from openpi.training import optimizer
    from openpi.training import weight_loaders

    base = config_lib.get_config(config_name(settings))
    train = settings["training"]
    policy = settings["policy"]
    model = dataclasses.replace(
        base.model,
        action_horizon=policy["timing"]["prediction_steps"],
        discrete_state_input=settings["model"]["discrete_state_input"],
        max_token_len=max(base.model.max_token_len, 6 * state_dimension(settings) + 96)
        if state_dimension(settings) > 32
        else base.model.max_token_len,
    )
    data = dataclasses.replace(
        base.data,
        repo_id=settings["data"]["repo_id"],
        state_space=policy["state"]["space"],
        policy=copy.deepcopy(policy),
        use_delta_actions=policy["action"]["representation"] == "component_delta",
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
        "policy_hz": policy["timing"]["sample_hz"] / policy["timing"]["stride_steps"],
        "policy_contract": copy.deepcopy(policy),
    }
    if policy["timing"]["execution_steps"] is not None:
        metadata["execution_steps"] = policy["timing"]["execution_steps"]
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
    return dataclasses.replace(
        config,
        checkpoint_assets={
            CHECKPOINT_CONFIG: yaml.safe_dump({"settings": settings, "extra_args": list(extra_args)}, sort_keys=False)
        },
    )


def policy_semantics(settings: dict[str, Any]) -> dict[str, Any]:
    """Input/label semantics used for statistics and resume; K is deployment-only."""
    policy = copy.deepcopy(settings["policy"])
    policy["timing"].pop("execution_steps")
    return policy


def state_dimension(settings):
    policy = settings["policy"]
    arms = 2 if settings["experiment"]["arm"] == "both" else 1
    width = 4 + {"rot6d": 6, "quaternion": 4, "axis_angle": 3}[policy["pose"]["rotation"]]
    per_arm = {"eef": width, "joint": 8, "both": width + 8}[policy["state"]["space"]]
    return arms * per_arm * policy["timing"]["history_steps"]


def describe_policy(settings: dict[str, Any]) -> dict[str, Any]:
    """Derived dimensions and timestamps, without loading model dependencies."""
    policy = settings["policy"]
    action, timing = policy["action"], policy["timing"]
    arms = 2 if settings["experiment"]["arm"] == "both" else 1
    width = 4 + {"rot6d": 6, "quaternion": 4, "axis_angle": 3}[policy["pose"]["rotation"]]
    action_width = {"pose_target": width, "joint_position": 8, "joint_velocity": 8, "eef_twist": 7, "joint_torque": 8}[
        action["control_quantity"]
    ]
    suffix = {
        "pose_target": "pose",
        "joint_position": "joint",
        "joint_velocity": "joint_velocity",
        "eef_twist": "eef_twist",
        "joint_torque": "joint_torque",
    }[action["control_quantity"]]
    first = timing["target_offset_steps"]
    last = first + (timing["prediction_steps"] - 1) * timing["stride_steps"]
    return {
        "state_column": {"eef": "state.pose", "joint": "state.joint", "both": ["state.pose", "state.joint"]}[
            policy["state"]["space"]
        ],
        "state_dim": state_dimension(settings),
        "label_column": ("action" if action["label_source"] == "command" else "state") + "." + suffix,
        "action_dim": arms * action_width,
        "prediction_shape": [timing["prediction_steps"], arms * action_width],
        "target_offsets_steps": [first, last],
        "last_target_time_seconds": last / timing["sample_hz"],
        "action_sample_hz": timing["sample_hz"] / timing["stride_steps"],
        "label_encoding": f"{action['representation']} / {action['relative_to']} / {action['delta_scope']}; gripper absolute",
        "execution_steps": timing["execution_steps"],
    }
