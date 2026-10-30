"""OpenPI training YAML and checkpoint snapshots; independent of the wire package."""

import copy

VERSION = 7

TRAINING_FIELDS = {
    "batch_size": "batch_size", "steps": "num_train_steps", "num_workers": "num_workers",
    "seed": "seed", "log_interval": "log_interval", "save_interval": "save_interval",
    "learning_rate": "peak_lr",
}

OPENPI_OPTIONS = {
    "finetuning": None,
    "run": {"smoke"},
    "model": {"discrete_state_input"},
    "training": {"fsdp_devices", "keep_period", "warmup_steps", "decay_steps", "decay_lr"},
    "checkpoints": {"base_dir", "assets_dir", "overwrite"},
    "normalization": {"mode", "max_frames"},
    "runtime": {"mem_fraction"},
}


SOURCE_FIELDS = {
    "schema_version": None,
    "backend": None,
    "experiment": None,
    "data": None,
    "policy": None,
    "model": {"type", "pretrained_path"},
    "training": TRAINING_FIELDS.keys(),
    "runtime": {"gpus", "resume"},
    "wandb": {"mode", "project"},
    "backend_options": {"openpi"},
}


def _validate_fields(value, allowed, path):
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a YAML mapping")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown YAML field: {path}.{min(unknown)}")


def to_native(source):
    _validate_fields(source, SOURCE_FIELDS, "settings")
    if source.get("schema_version") != VERSION:
        raise ValueError(f"openpi training requires schema_version: {VERSION}")
    if source.get("backend") != "openpi":
        raise ValueError("backend must be openpi for this training entry point")
    for section, allowed in SOURCE_FIELDS.items():
        if allowed is not None:
            _validate_fields(source.get(section, {}), allowed, section)
    options = source.get("backend_options", {}).get("openpi", {})
    _validate_fields(options, OPENPI_OPTIONS.keys(), "backend_options.openpi")
    for section, allowed in OPENPI_OPTIONS.items():
        if allowed is not None:
            _validate_fields(options.get(section, {}), allowed, f"backend_options.openpi.{section}")
    model_type = source.get("model", {}).get("type", "pi05")
    if model_type != "pi05":
        raise ValueError("The OpenPI G2 training entry point supports model.type: pi05")
    result = copy.deepcopy(options)
    result["schema_version"] = VERSION
    for section in ("experiment", "data", "policy"):
        if section in source:
            result[section] = copy.deepcopy(source[section])
    training = result.setdefault("training", {})
    for key, value in source.get("training", {}).items():
        training[TRAINING_FIELDS[key]] = value
    if "pretrained_path" in source.get("model", {}):
        result.setdefault("model", {})["params_path"] = source["model"]["pretrained_path"]
    runtime = source.get("runtime", {})
    if "gpus" in runtime:
        result.setdefault("runtime", {})["gpus"] = copy.deepcopy(runtime["gpus"])
    if "resume" in runtime:
        result.setdefault("checkpoints", {})["resume"] = runtime["resume"]
    if "wandb" in source:
        result["wandb"] = copy.deepcopy(source["wandb"])
    return result


def openpi_snapshot(settings):
    """Serialize resolved OpenPI settings as a self-contained public snapshot."""
    native = copy.deepcopy(settings)
    result = {"schema_version": VERSION, "backend": "openpi"}
    for section in ("experiment", "data", "policy"):
        if section in native:
            result[section] = native.pop(section)
    runtime = native.pop("runtime", {})
    result["runtime"] = {"gpus": runtime.pop("gpus"), "resume": native["checkpoints"].pop("resume")}
    training = {}
    native.pop("schema_version", None)
    model = native["model"]
    result["model"] = {"type": "pi05", "pretrained_path": model.pop("params_path")}
    for key, name in TRAINING_FIELDS.items():
        if name in native["training"]:
            training[key] = native["training"].pop(name)
    native["runtime"] = runtime
    result["wandb"] = native.pop("wandb", {})
    result["training"] = training
    result["backend_options"] = {"openpi": native}
    return result
