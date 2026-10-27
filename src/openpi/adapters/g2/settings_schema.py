"""Public G2 YAML vocabulary, translated to the existing policy/wire contract.

This module stays dependency-free so config previews never initialize a model.
"""

import copy

REFERENCES = {
    "current_measured": "current_measured_pose",
    "current_command": "current_command_pose",
    "previous_command": "previous_command_pose",
    "episode_start": "episode_start_pose",
    "object": "object_pose",
}
PUBLIC_REFERENCES = {value: key for key, value in REFERENCES.items()}
FINETUNING_MODES = ("frozen", "lora", "full")
ACTION_QUANTITIES = {
    "eef_pose": "pose_target",
    "joint_position": "joint_position",
    "joint_velocity": "joint_velocity",
    "eef_twist": "eef_twist",
    "joint_torque": "joint_torque",
}
ACTION_METHODS = {
    "component_difference": "component_delta",
    "se3_body": "se3_body_delta",
    "se3_spatial": "se3_spatial_delta",
    "hybrid": "hybrid_delta",
}
STATE_METHODS = {**ACTION_METHODS, "component_difference": "relative"}
REFERENCE_STRATEGIES = {"fixed": "chunk", "previous_target": "step"}
REFERENCE_SCOPES = {"chunk": "chunk", "step": "step"}


def _mapping(value, fields, path):
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a YAML mapping")
    unknown = value.keys() - fields
    if unknown:
        raise ValueError(f"Unknown YAML field: {path}.{sorted(unknown)[0]}")
    return value


def _translate(value, choices, path):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{path} must be one of {tuple(choices)}")
    return choices[value]


_MISSING = object()


def _compile_group(spec, group):
    """Parse State/Action once; legacy names are resolved only at the input boundary."""
    path = f"policy.{group}"
    current = bool(spec.keys() & {"spatial", "temporal"})
    identity = "space" if group == "state" else "target"
    allowed = {identity, "spatial", "temporal"} if current else {identity, "encoding"}
    if spec.keys() - allowed:
        raise ValueError(f"Cannot mix {path} fields; allowed fields are {sorted(allowed)}")
    result = {}
    if group == "state" and "space" in spec:
        result["space"] = spec["space"]
    if group == "action" and "target" in spec:
        target = _mapping(spec["target"], {"quantity", "source"}, f"{path}.target")
        for key, internal, choices in (
            ("quantity", "control_quantity", ACTION_QUANTITIES),
            ("source", "label_source", {"command": "command", "measured": "measured"}),
        ):
            if key in target:
                result[internal] = _translate(target[key], choices, f"{path}.target.{key}")
    if current:
        spatial = _mapping(spec.get("spatial", {}), {"mode", "method"}, f"{path}.spatial")
        mode, method = spatial.get("mode", _MISSING), spatial.get("method", _MISSING)
        temporal = spec.get("temporal", _MISSING)
        mode_path, method_path, temporal_path = f"{path}.spatial.mode", f"{path}.spatial.method", f"{path}.temporal"
        scope_key, scopes = "reference_scope", REFERENCE_SCOPES
    else:
        if "encoding" not in spec:
            return result
        fields = {"mode", "spatial", "temporal"}
        if group == "action":
            fields |= {"method", "reference"}
        encoding = _mapping(spec["encoding"], fields, f"{path}.encoding")
        aliases = bool(encoding.keys() & {"method", "reference"})
        if aliases and encoding.keys() & {"spatial", "temporal"}:
            raise ValueError("Cannot mix encoding.spatial/temporal with legacy encoding.method/reference")
        method_key, temporal_key = ("method", "reference") if aliases else ("spatial", "temporal")
        mode, method = encoding.get("mode", _MISSING), encoding.get(method_key, _MISSING)
        temporal = encoding.get(temporal_key, _MISSING)
        mode_path, method_path = f"{path}.encoding.mode", f"{path}.encoding.{method_key}"
        temporal_path = f"{path}.encoding.{temporal_key}"
        scope_key, scopes = ("strategy", REFERENCE_STRATEGIES) if aliases else ("reference_scope", REFERENCE_SCOPES)

    temporal_fields = {"anchor", scope_key} if group == "action" else {"anchor"}
    if current and temporal is not _MISSING and temporal is not None:
        temporal = _mapping(temporal, temporal_fields, temporal_path)
    if mode is not _MISSING and mode not in ("absolute", "relative"):
        raise ValueError(f"{mode_path} must be absolute or relative")
    if mode == "absolute":
        null_reference = (
            temporal is _MISSING or temporal is None or (current and all(value is None for value in temporal.values()))
        )
        if (method is not _MISSING and method is not None) or not null_reference:
            raise ValueError(f"{method_path} and {temporal_path} must be null for absolute mode")
        result.update(representation="absolute", relative_to=None)
        if group == "action":
            result["delta_scope"] = None
        return result
    if mode == "relative" and method is _MISSING:
        raise ValueError(f"{method_path} is required when mode=relative")
    if method is not _MISSING:
        methods = STATE_METHODS if group == "state" else ACTION_METHODS
        result["representation"] = _translate(method, methods, method_path)
    if temporal is _MISSING:
        return result
    if temporal is None:
        result["relative_to"] = None
        if group == "action":
            result["delta_scope"] = None
        return result
    temporal = _mapping(temporal, temporal_fields, temporal_path)
    for key, internal, choices in (
        ("anchor", "relative_to", REFERENCES),
        (scope_key, "delta_scope", scopes),
    ):
        if key in temporal:
            value = temporal[key]
            result[internal] = (
                None if current and value is None else _translate(value, choices, f"{temporal_path}.{key}")
            )
    return result


def compile_input(settings):
    """Translate new field names, rejecting conflicting or misspelled encodings."""
    result = copy.deepcopy(settings)
    version = result.get("schema_version", 1)
    if type(version) is not int or version not in (1, 2, 3, 4):
        raise ValueError("schema_version must be 1, 2, 3 or 4")
    if version >= 3:
        if isinstance(result.get("experiment"), dict) and "preset" in result["experiment"]:
            raise ValueError(f"schema v{version} uses finetuning module modes; remove experiment.preset")
        result["schema_version"] = 2
    policy = result.get("policy")
    if not isinstance(policy, dict):
        return result
    for group in ("state", "action"):
        spec = policy.get(group)
        if not isinstance(spec, dict):
            continue
        if (
            spec.keys() & {"spatial", "temporal"}
            or isinstance(spec.get("encoding"), dict)
            or (group == "action" and "target" in spec)
        ):
            policy[group] = _compile_group(spec, group)
            continue
        if group == "action" and version == 4 and spec:
            raise ValueError(
                "schema v4 action uses target/spatial/temporal mappings; legacy fields require schema v1/v2/v3"
            )
        for public, internal in (("encoding", "representation"), ("reference", "relative_to")):
            if public not in spec:
                continue
            if internal in spec:
                raise ValueError(f"Cannot mix policy.{group}.{public} and {internal}")
            value = spec.pop(public)
            if public == "reference" and isinstance(value, str):
                if value not in REFERENCES and value not in PUBLIC_REFERENCES:
                    raise ValueError(f"policy.{group}.reference must be one of {tuple(REFERENCES)} or null")
                value = REFERENCES.get(value, value)
            if public == "encoding" and group == "state":
                if value not in ("absolute", "component_delta"):
                    raise ValueError("policy.state.encoding must be absolute or component_delta")
                value = "relative" if value == "component_delta" else value
            spec[internal] = value
    action = policy.get("action")
    if isinstance(action, dict) and "control_quantity" in action and "space" not in action:
        action["space"] = "eef" if action["control_quantity"] in ("pose_target", "eef_twist") else "joint"
    return result


def legacy_finetuning(preset, vision_rank):
    """Recover old freezing semantics without consulting today's template."""
    modes = {
        "low_mem": {"vision": "full", "language": "lora", "action_expert": "lora"},
        "smoke": {"vision": "full", "language": "lora", "action_expert": "lora"},
        "full_ft": {"vision": "full", "language": "full", "action_expert": "full"},
        "vlm_lora_action_full": {"vision": "frozen", "language": "lora", "action_expert": "full"},
    }
    if not isinstance(preset, str) or preset not in modes:
        raise ValueError(f"Unknown legacy preset: {preset!r}")
    result = modes[preset].copy()
    if isinstance(vision_rank, int) and vision_rank > 0:
        result["vision"] = "lora"
    return result


def preset_for_modes(modes):
    """Keep existing checkpoint namespaces for the three established model variants."""
    pair = (modes.get("language"), modes.get("action_expert"))
    if not all(isinstance(mode, str) for mode in pair):
        return "custom"
    return {
        ("full", "full"): "full_ft",
        ("lora", "lora"): "low_mem",
        ("lora", "full"): "vlm_lora_action_full",
    }.get(pair, "custom")


def add_training_sections(settings, *, partial):
    """Separate legacy smoke presets and derive a compatibility namespace."""
    experiment = settings.get("experiment")
    if not isinstance(experiment, dict):
        return settings
    preset = experiment.get("preset")
    if "finetuning" not in settings and preset is not None:
        rank = settings.get("model", {}).get("vision_lora_rank", 0)
        settings["finetuning"] = legacy_finetuning(preset, rank)
    if preset == "smoke":
        if "run" in settings and not isinstance(settings["run"], dict):
            raise ValueError("run must be a YAML mapping")
        settings.setdefault("run", {}).setdefault("smoke", True)
    if not partial:
        settings.setdefault("run", {"smoke": False})
        if isinstance(settings.get("finetuning"), dict):
            experiment["preset"] = preset_for_modes(settings["finetuning"])
    return settings


def public_action(action):
    """Describe target selection and arm encoding separately; grippers stay absolute."""
    quantities = {value: key for key, value in ACTION_QUANTITIES.items()}
    methods = {value: key for key, value in ACTION_METHODS.items()}
    relative = action["representation"] != "absolute"
    return {
        "target": {"quantity": quantities[action["control_quantity"]], "source": action["label_source"]},
        "spatial": {
            "mode": "relative" if relative else "absolute",
            "method": methods[action["representation"]] if relative else None,
        },
        "temporal": {
            "anchor": PUBLIC_REFERENCES[action["relative_to"]] if relative else None,
            "reference_scope": action["delta_scope"] if relative else None,
        },
    }


def public_state(state):
    """State pose coordinates come from policy.pose; all history frames use one anchor."""
    methods = {value: key for key, value in STATE_METHODS.items()}
    relative = state["representation"] != "absolute"
    return {
        "space": state["space"],
        "spatial": {
            "mode": "relative" if relative else "absolute",
            "method": methods[state["representation"]] if relative else None,
        },
        "temporal": {"anchor": PUBLIC_REFERENCES[state["relative_to"]] if relative else None},
    }


def public_settings(settings):
    """Export a self-contained schema v4 snapshot/preview from normalized settings."""
    result = copy.deepcopy(settings)
    result["schema_version"] = 4
    result["experiment"].pop("preset", None)
    result["policy"]["state"] = public_state(result["policy"]["state"])
    result["policy"]["action"] = public_action(result["policy"]["action"])
    if result["policy"]["timing"].get("execution_steps") is None:
        result["policy"]["timing"].pop("execution_steps", None)
    return result
