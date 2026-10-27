import copy
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from openpi.adapters.g2 import config as g2_config


def legacy_settings(mode="relative"):
    settings = g2_config.load_settings()
    settings.pop("schema_version")
    policy = settings.pop("policy")
    settings["data"].update(action_space="pose", action_mode=mode)
    settings["model"]["action_horizon"] = policy["timing"]["action_horizon"]
    return settings


@pytest.mark.parametrize(("mode", "space"), [("relative", "eef"), ("absolute", "joint")])
def test_legacy_checkpoint_migration_is_self_contained(mode, space):
    legacy = legacy_settings(mode)
    original = copy.deepcopy(legacy)
    with patch.object(g2_config, "load_settings", side_effect=AssertionError("must not read today's template")):
        config = g2_config.build_train_config(legacy)
    assert config.data.resolved_state_space == space
    assert config.data.use_delta_actions == (mode == "relative")
    assert config.model.action_horizon == 50
    assert yaml.safe_load(config.checkpoint_assets[g2_config.CHECKPOINT_CONFIG])["settings"]["schema_version"] == 2
    assert legacy == original


def test_partial_legacy_yaml_and_nested_v2_yaml(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text("data:\n  action_mode: absolute\nmodel:\n  action_horizon: 16\n")
    settings = g2_config.resolve_settings(g2_config.load_settings(path), {})
    assert settings["policy"]["state"]["space"] == "joint"
    assert settings["policy"]["action"]["representation"] == "absolute"
    assert settings["policy"]["timing"]["action_horizon"] == 16
    path.write_text("policy:\n  action:\n    representation: absolute\n  timing:\n    action_horizon: 16\n")
    settings = g2_config.resolve_settings(g2_config.load_settings(path), {})
    assert settings["policy"]["state"]["space"] == "eef"
    assert settings["policy"]["action"]["relative_to"] is None
    assert settings["policy"]["action"]["delta_scope"] is None
    assert settings["policy"]["timing"]["sample_hz"] == 30.0
    path.write_text("policy:\n  timing:\n    prediction_steps: 12\n")
    settings = g2_config.resolve_settings(g2_config.load_settings(path), {})
    assert settings["policy"]["timing"]["action_horizon"] == 12
    assert "prediction_steps" not in settings["policy"]["timing"]


@pytest.mark.parametrize(
    "text",
    [
        "policy:\n  state:\n    sapce: joint\n",
        "policy:\n  timing:\n    action_horizon: 16\n    action_horizon: 20\n",
        "policy:\n  state: joint\n",
        "policy: null\n",
        "policy:\n  action: null\n",
        "schema_version: 2\ndata:\n  action_mode: absolute\n",
        "policy:\n  state:\n    space: eef\nmodel:\n  action_horizon: 16\n",
    ],
)
def test_rejects_nested_typo_duplicate_and_mixed_schema(tmp_path, text):
    path = tmp_path / "invalid.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match="Unknown|Duplicate|mapping|mix"):
        g2_config.load_settings(path)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("pose", "rotation"), "quaternion"),
        (("state", "space"), "both"),
        (("action", "representation"), "se3_body_delta"),
        (("action", "representation"), "se3_spatial_delta"),
        (("action", "delta_scope"), "step"),
        (("action", "label_source"), "measured"),
        (("timing", "history_steps"), 2),
        (("timing", "target_offset_steps"), 1),
        (("timing", "stride_steps"), 2),
    ],
)
def test_extended_semantics_are_supported(path, value):
    settings = g2_config.load_settings()
    settings["policy"][path[0]][path[1]] = value
    g2_config.validate_settings(settings)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("sample_hz", 0),
        ("sample_hz", float("nan")),
        ("sample_hz", float("inf")),
        ("history_steps", True),
        ("action_horizon", 0),
        ("execution_steps", 51),
        ("execution_steps", True),
        ("target_offset_steps", -1),
        ("stride_steps", 0),
    ],
)
def test_timing_units_ranges_and_types_are_validated(key, value):
    settings = g2_config.load_settings()
    settings["policy"]["timing"][key] = value
    with pytest.raises(ValueError, match="must"):
        g2_config.validate_settings(settings)


def test_joint_state_delta_supported_and_absolute_reference_rejected():
    settings = g2_config.load_settings()
    settings["policy"]["state"]["space"] = "joint"
    g2_config.validate_settings(settings)
    settings = g2_config.load_settings()
    settings["policy"]["action"]["representation"] = "absolute"
    with pytest.raises(ValueError, match="must be null"):
        g2_config.validate_settings(settings)


@pytest.mark.parametrize("arm", ["both", "left", "right"])
@pytest.mark.parametrize(
    ("space", "representation"),
    [("eef", "component_delta"), ("eef", "absolute"), ("joint", "absolute"), ("joint", "component_delta")],
)
def test_state_action_encoding_round_trip(tmp_path, monkeypatch, arm, space, representation):
    from openpi import transforms
    from openpi.adapters.g2.transforms import arm_slice
    from openpi.training import config as config_lib

    settings = g2_config.resolve_settings(
        g2_config.load_settings(),
        {
            "arm": arm,
            "state_space": space,
            "action_representation": representation,
        },
    )
    settings["data"]["cameras"] = ["head_color"]
    config = g2_config.build_train_config(settings)
    monkeypatch.setattr(config_lib.DataConfigFactory, "_load_norm_stats", lambda *args: None)
    monkeypatch.setattr(config_lib.ModelTransformFactory, "__call__", lambda *args: transforms.Group())
    data_config = config.data.create(tmp_path, config.model)
    state = np.arange(20 if space == "eef" else 16, dtype=np.float32) / 10
    actions = np.arange(60, dtype=np.float32).reshape(3, 20) / 20
    original = actions.copy()
    pose = np.arange(20, dtype=np.float32) / 10
    state_key = "state.pose" if space == "eef" else "state.joint"
    sample = {
        "observation.images.head_color": np.zeros((224, 224, 3), np.uint8),
        state_key: state,
        "state.pose": pose,
        "action.pose": actions,
    }
    # The real dataset repacker expects a prompt from the task table.
    sample["prompt"] = "tea"
    for transform in (*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs):
        sample = transform(sample)
    state_selection = arm_slice(arm, len(state))
    action_selection = arm_slice(arm, 20)
    np.testing.assert_array_equal(sample["state"], state[state_selection])
    expected = original[:, action_selection].copy()
    if representation == "component_delta":
        mask = np.array(([True] * 9 + [False]) * (2 if arm == "both" else 1))
        expected[:, mask] -= pose[action_selection][mask]
    np.testing.assert_allclose(sample["actions"], expected)
    from openpi.adapters.g2.transforms import ActionCodec

    decoded = ActionCodec(settings["policy"], arm).decode(sample["actions"], {"observation/pose": pose})
    np.testing.assert_allclose(decoded[:, action_selection], original[:, action_selection], atol=1e-6)
    np.testing.assert_array_equal(actions, original)




def test_derived_preview_is_consistent_with_prediction_and_state_choice():
    settings = g2_config.resolve_settings(
        g2_config.load_settings(),
        {
            "state_space": "joint",
            "action_representation": "absolute",
            "action_horizon": 16,
        },
    )
    preview = g2_config.describe_policy(settings)
    assert preview["state_column"] == "state.joint"
    assert preview["state_dim"] == 8
    assert preview["prediction_shape"] == [16, 10]
    assert preview["target_offsets_steps"] == [0, 15]
    assert preview["last_target_time_seconds"] == 0.5


def test_schema_version_is_not_a_boolean():
    settings = g2_config.load_settings()
    settings["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version"):
        g2_config.build_train_config(settings)


def test_checkpoint_yaml_contains_resolved_v2_only(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(legacy_settings()))
    settings = g2_config.resolve_settings(g2_config.load_settings(path), {})
    config = g2_config.build_train_config(settings)
    snapshot = yaml.safe_load(yaml.safe_dump(yaml.safe_load(config.checkpoint_assets[g2_config.CHECKPOINT_CONFIG])))
    assert "action_mode" not in snapshot["settings"]["data"]
    assert "action_horizon" not in snapshot["settings"]["model"]
    assert snapshot["settings"]["policy"] == settings["policy"]
