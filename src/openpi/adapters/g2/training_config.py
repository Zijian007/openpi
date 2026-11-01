"""Versioned G2 training YAML and snapshots; no model/framework imports."""

import copy

VERSION = 8
LORA_DEFAULTS = {
    "vision": {"rank": 16, "alpha": 16.0, "targets": ["attention"], "rslora": False},
    "language": {"rank": 16, "alpha": 16.0, "targets": ["attention", "ffn"], "rslora": False},
    "action_expert": {"rank": 32, "alpha": 32.0, "targets": ["attention", "ffn"], "rslora": False},
}
FINETUNING_FIELDS = {module: {"mode": None, "lora": {"rank", "alpha", "targets", "rslora"}} for module in LORA_DEFAULTS}

SOURCE_FIELDS = {
    "schema_version": None,
    "backend": None,
    "experiment": None,
    "runtime": {"gpus", "fsdp_devices", "mem_fraction"},
    "model": {"type", "pretrained_path", "discrete_state_input"},
    "finetuning": FINETUNING_FIELDS,
    "data": None,
    "training": {
        "batch_size": None,
        "steps": None,
        "seed": None,
        "num_workers": None,
        "smoke": None,
        "lr_schedule": {"peak_lr", "warmup_steps", "decay_steps", "decay_lr"},
    },
    "checkpoints": {"base_dir", "assets_dir", "resume", "overwrite", "save_interval", "keep_period"},
    "normalization": {"mode", "max_frames"},
    "logging": {"log_interval": None, "wandb": {"mode", "project"}},
    "policy": None,
    # Computed by the mixture planner; persisted for self-contained snapshots.
    "resolved": {"mixture": {"asset_id", "fingerprint", "proportions", "plan", "lengths"}},
}

# Public path -> native path. Native TrainConfig/resume semantics stay unchanged.
V8_PATHS = {
    **{(name,): (name,) for name in ("experiment", "finetuning", "data", "normalization", "policy")},
    **{("runtime", key): ("runtime", key) for key in ("gpus", "mem_fraction")},
    ("runtime", "fsdp_devices"): ("training", "fsdp_devices"),
    ("model", "pretrained_path"): ("model", "params_path"),
    ("model", "discrete_state_input"): ("model", "discrete_state_input"),
    **{("training", key): ("training", key) for key in ("batch_size", "seed", "num_workers")},
    ("training", "steps"): ("training", "num_train_steps"),
    ("training", "smoke"): ("run", "smoke"),
    **{
        ("training", "lr_schedule", key): ("training", key)
        for key in ("peak_lr", "warmup_steps", "decay_steps", "decay_lr")
    },
    **{("checkpoints", key): ("checkpoints", key) for key in ("base_dir", "assets_dir", "resume", "overwrite")},
    **{("checkpoints", key): ("training", key) for key in ("save_interval", "keep_period")},
    ("logging", "log_interval"): ("training", "log_interval"),
    ("logging", "wandb"): ("wandb",),
    ("resolved", "mixture"): ("mixture",),
}
_MISSING = object()


def _validate_fields(value, schema, path):
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a YAML mapping")
    unknown = value.keys() - schema.keys() if isinstance(schema, dict) else value.keys() - schema
    if unknown:
        raise ValueError(f"Unknown YAML field: {path}.{min(unknown)}")
    if isinstance(schema, dict):
        for key, child in schema.items():
            if child is not None and key in value:
                _validate_fields(value[key], child, f"{path}.{key}")


def _get(tree, path):
    for key in path:
        if not isinstance(tree, dict) or key not in tree:
            return _MISSING
        tree = tree[key]
    return tree


def _put(tree, path, value):
    for key in path[:-1]:
        tree = tree.setdefault(key, {})
    tree[path[-1]] = copy.deepcopy(value)


def to_native(source):
    """Translate explicit v8 fields, without inserting inherited defaults."""
    if not isinstance(source, dict):
        raise TypeError("settings must be a YAML mapping")
    version = source.get("schema_version")
    if type(version) is not int or version != VERSION:
        raise ValueError("openpi training requires schema_version: 8")
    _validate_fields(source, SOURCE_FIELDS, "settings")
    if source.get("backend") != "openpi":
        raise ValueError("backend must be openpi for this training entry point")
    if source.get("model", {}).get("type", "pi05") != "pi05":
        raise ValueError("The OpenPI G2 training entry point supports model.type: pi05")
    result = {"schema_version": VERSION}
    for public, native in V8_PATHS.items():
        value = _get(source, public)
        if value is not _MISSING:
            _put(result, native, value)
    return result


def experiment_to_native(source):
    """Allow omitted version/backend in partial v8 overrides."""
    source = copy.deepcopy(source)
    source.setdefault("schema_version", VERSION)
    source.setdefault("backend", "openpi")
    return to_native(source)


def complete_finetuning(settings):
    """Fill omitted LoRA options after inheritance, including inactive modules."""
    for module, spec in settings.get("finetuning", {}).items():
        if module in LORA_DEFAULTS:
            options = copy.deepcopy(LORA_DEFAULTS[module])
            options.update(spec.get("lora", {}))
            spec["lora"] = options
    return settings


def openpi_snapshot(settings):
    """Serialize complete native settings as a self-contained v8 snapshot."""
    result = {"schema_version": VERSION, "backend": "openpi"}
    for public, native in V8_PATHS.items():
        value = _get(settings, native)
        if value is not _MISSING:
            _put(result, public, value)
    result.setdefault("model", {})["type"] = "pi05"
    order = (
        "schema_version",
        "backend",
        "experiment",
        "runtime",
        "model",
        "finetuning",
        "data",
        "training",
        "checkpoints",
        "normalization",
        "logging",
        "policy",
        "resolved",
    )
    return {key: result[key] for key in order if key in result}
