"""v6 branch selection, independent state encodings and LeRobot supervision."""

import copy
from types import SimpleNamespace

from g2_policy_tools.geometry import ActionCodec
from g2_policy_tools.geometry import matrix_pose
import numpy as np
import pytest
import yaml

from openpi.adapters.g2 import config
from openpi.adapters.g2.data import G2Dataset
from openpi.adapters.g2.data import RepackG2
from openpi.adapters.g2.policy import create_trained_policy
from openpi.adapters.g2.server import connection_metadata
from openpi.adapters.g2.transforms import G2Inputs


def resolve(tmp_path, policy):
    path = tmp_path / "branches.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 6, "policy": policy}))
    return config.resolve_settings(config.load_settings(path), {})


def test_default_tree_and_self_contained_snapshot():
    resolved = config.resolve_settings(config.load_settings(), {})
    public = config.public_settings(resolved)
    assert public["schema_version"] == 6
    assert public["policy"].keys() == {"state", "action", "timing"}
    for group in ("state", "action"):
        assert public["policy"][group].keys() == {"joint", "pose"}
        assert public["policy"][group]["joint"]["enabled"] is False
        assert public["policy"][group]["pose"]["enabled"] is True
        assert "coordinates" in public["policy"][group]["pose"]
    assert config.normalize_settings(public) == resolved
    model = config.build_train_config(resolved)
    saved = yaml.safe_load(model.checkpoint_assets[config.CHECKPOINT_CONFIG])["settings"]
    assert saved == public
    preview = config.describe_policy(resolved)
    assert preview["state_dim"] == 10
    assert preview["prediction_shape"] == [50, 10]
    assert preview["label_column"] == "action.pose"


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5, 7, True, "6"])
def test_other_schemas_are_rejected(tmp_path, version):
    path = tmp_path / "version.yaml"
    path.write_text(yaml.safe_dump({"schema_version": version}))
    with pytest.raises(ValueError, match="schema_version: 6"):
        config.load_settings(path)


@pytest.mark.parametrize(
    "policy",
    [
        {"state": {"space": "eef"}},
        {"action": {"target": {"quantity": "eef_pose"}}},
        {"pose": {"rotation": "rot6d"}},
        {"action": {"pose": {"spatial": {"method": "se3_body"}}}},
        {"state": {"joint": {"coordinates": {}}}},
        {"action": {"pose": {"enabled": False}, "joint": {"spatial": {"method": "component_difference"}}}},
        {"action": {"pose": {"enabled": False}, "joint": {"enabled": True, "quantity": "joint_position"}}},
    ],
)
def test_removed_fields_aliases_and_inapplicable_branch_options_are_rejected(tmp_path, policy):
    with pytest.raises(ValueError, match="Unknown|must be one of"):
        resolve(tmp_path, policy)


@pytest.mark.parametrize(
    "policy",
    [
        {"state": {"joint": {"enabled": False}, "pose": {"enabled": False}}},
        {"action": {"joint": {"enabled": False}, "pose": {"enabled": False}}},
        {"action": {"joint": {"enabled": True}}},
    ],
)
def test_invalid_branch_counts_fail_before_dataset_access(tmp_path, policy):
    with pytest.raises(ValueError, match="active branch"):
        resolve(tmp_path, policy)


def test_absolute_override_clears_inherited_relative_options(tmp_path):
    settings = resolve(tmp_path, {"action": {"pose": {"delta": False}}})
    spec = config.public_settings(settings)["policy"]["action"]["pose"]
    assert spec["delta"] is False
    assert spec["reference"] == {"source": None, "scope": None}
    with pytest.raises(ValueError, match="reference requires"):
        resolve(tmp_path, {"action": {"pose": {"delta": False, "reference": {"source": "state"}}}})


@pytest.mark.parametrize("arm", ["both", "left", "right"])
@pytest.mark.parametrize("pose_relative", [False, True])
@pytest.mark.parametrize("joint_relative", [False, True])
@pytest.mark.parametrize("joint_source", ["state", "action"])
def test_combined_state_encodes_each_branch_independently(tmp_path, arm, pose_relative, joint_relative, joint_source):
    settings = resolve(
        tmp_path,
        {
            "state": {
                "pose": {
                    "delta": pose_relative,
                    "method": "se3" if pose_relative else None,
                    "reference": {"source": "state", "scope": "chunk"} if pose_relative else {"source": None, "scope": None},
                },
                "joint": {
                    "enabled": True,
                    "delta": joint_relative,
                    "method": "component" if joint_relative else None,
                    "reference": {"source": joint_source, "scope": "chunk"} if joint_relative else {"source": None, "scope": None},
                },
            },
            "timing": {"history_steps": 2},
        },
    )
    settings = config.resolve_settings(settings, {"arm": arm})
    pose = np.tile(matrix_pose(np.eye(4), 0.6), 2)
    history = np.stack([pose.copy(), pose.copy()])
    history[0, [0, 10]] = 0.2
    history[1, [0, 10]] = 0.5
    joints = np.arange(32, dtype=np.float32).reshape(2, 16) / 10
    reference = np.full(16, 0.3, np.float32)
    obs = {
        "observation/state": history[-1],
        "observation/pose": history[-1],
        "observation/pose_history": history,
        "observation/joint": joints[-1],
        "observation/joint_history": joints,
        "observation/command_joint": reference,
    }
    actual = G2Inputs(policy=settings["policy"], arm=arm, cameras=())(obs)["state"]
    expected_pose = history.copy()
    if pose_relative:
        expected_pose[:, [0, 10]] -= 0.5
    expected_joint = joints.copy()
    if joint_relative:
        mask = np.tile([True] * 7 + [False], 2)
        joint_ref = joints[-1] if joint_source == "state" else reference
        expected_joint[:, mask] -= joint_ref[mask]
    if arm != "both":
        p, q = (slice(0, 10), slice(0, 8)) if arm == "left" else (slice(10, 20), slice(8, 16))
        expected_pose, expected_joint = expected_pose[:, p], expected_joint[:, q]
    np.testing.assert_allclose(actual, np.concatenate([expected_pose, expected_joint], axis=-1).reshape(-1), atol=1e-6)
    assert actual.size == config.state_dimension(settings)
    assert config.normalize_settings(config.public_settings(settings)) == settings
    metadata = connection_metadata({"policy_contract": settings["policy"], "required_cameras": []})
    context = metadata["observation_spec"]["context_keys"]
    if joint_relative and joint_source == "action":
        assert "command_joint" in context
    assert "object_pose" not in context


@pytest.mark.parametrize(
    ("kind", "quantity", "suffix", "width"),
    [
        ("joint", "position", "joint", 8),
        ("joint", "velocity", "joint_velocity", 8),
        ("joint", "torque", "joint_torque", 8),
        ("pose", "pose", "pose", 10),
    ],
)
@pytest.mark.parametrize("source", ["action", "state"])
def test_branch_quantity_and_source_select_lerobot_labels(tmp_path, kind, quantity, suffix, width, source):
    spec = {"enabled": True, "quantity": quantity, "source": source, "delta": False}
    settings = resolve(
        tmp_path,
        {
            "action": {
                "joint": spec if kind == "joint" else {"enabled": False},
                "pose": spec if kind == "pose" else {"enabled": False},
            }
        },
    )
    column = f"{source}.{suffix}"
    rows = []
    for i in range(4):
        row = {}
        for prefix in ("state", "action"):
            value = i + (10 if prefix == "action" else 0)
            pose = np.tile(matrix_pose(np.eye(4), 0.4), 2)
            pose[[0, 10]] = value
            row[f"{prefix}.pose"] = pose
            for name, dim in (("joint", 16), ("joint_velocity", 16), ("joint_torque", 16)):
                row[f"{prefix}.{name}"] = np.full(dim, value, np.float32)
        rows.append(row)

    class Dataset:
        hf_dataset = rows

        def __init__(self):
            self.meta = SimpleNamespace(fps=30, features={key: {} for key in rows[0]})
            self.episode_data_index = {"from": [0], "to": [4]}

        def __getitem__(self, index):
            return copy.deepcopy(rows[index])

    sample = G2Dataset(Dataset(), settings["policy"], (), 3)[0]
    packed = RepackG2(settings["policy"], ())(sample)
    target = np.stack([row[column] for row in rows[:3]])
    np.testing.assert_array_equal(packed["actions"], target)
    codec = ActionCodec(settings["policy"], "right")
    encoded = codec.encode(packed["actions"], packed)
    np.testing.assert_array_equal(encoded, target[:, width:])
    np.testing.assert_allclose(codec.decode(encoded, packed)[:, width:], target[:, width:], atol=1e-6)
    preview = config.describe_policy(settings)
    assert preview["label_column"] == column
    assert preview["prediction_shape"] == [50, width]


@pytest.mark.parametrize("scope", ["chunk", "step"])
def test_joint_position_delta_preserves_grippers_and_round_trips(tmp_path, scope):
    settings = resolve(
        tmp_path,
        {
            "action": {
                "pose": {"enabled": False},
                "joint": {
                    "enabled": True,
                    "delta": True,
                    "method": "component",
                    "reference": {"source": "state", "scope": scope},
                },
            }
        },
    )
    obs = {"observation/joint": np.arange(16, dtype=np.float32)}
    targets = np.arange(48, dtype=np.float32).reshape(3, 16) / 10
    codec = ActionCodec(settings["policy"], "both")
    encoded = codec.encode(targets, obs)
    np.testing.assert_array_equal(encoded[:, [7, 15]], targets[:, [7, 15]])
    np.testing.assert_allclose(codec.decode(encoded, obs), targets, atol=1e-6)


@pytest.mark.parametrize("quantity", ["velocity", "torque"])
def test_relative_joint_velocity_and_torque_are_rejected(tmp_path, quantity):
    with pytest.raises(ValueError, match="require delta: false"):
        resolve(
            tmp_path,
            {
                "action": {
                    "pose": {"enabled": False},
                    "joint": {
                        "enabled": True,
                        "quantity": quantity,
                        "delta": True,
                        "method": "component",
                        "reference": {"source": "state", "scope": "chunk"},
                    },
                }
            },
        )


def test_checkpoint_without_settings_snapshot_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("openpi.adapters.g2.policy.download.maybe_download", lambda path: tmp_path)
    with pytest.raises(ValueError, match="g2_train_config.yaml"):
        create_trained_policy(tmp_path)


def test_disabled_options_survive_preview_snapshot_and_activation(tmp_path):
    baseline = config.resolve_settings(config.load_settings(), {})
    configured = resolve(
        tmp_path,
        {
            "state": {"joint": {"delta": True, "method": "component", "reference": {"source": "action", "scope": "chunk"}}},
            "action": {
                "joint": {
                    "delta": True,
                    "method": "component",
                    "reference": {"source": "state", "scope": "step"},
                    "source": "state",
                }
            },
        },
    )
    assert config.policy_semantics(configured) == config.policy_semantics(baseline)
    assert config.describe_policy(configured)["prediction_shape"] == [50, 10]
    public = config.public_settings(configured)
    for group in ("state", "action"):
        spec = public["policy"][group]["joint"]
        assert spec["enabled"] is False
        assert spec["delta"] is True
    assert public["policy"]["action"]["joint"]["source"] == "state"
    assert config.describe_policy(configured)["action"]["joint"] == public["policy"]["action"]["joint"]
    built = config.build_train_config(configured)
    saved = yaml.safe_load(built.checkpoint_assets[config.CHECKPOINT_CONFIG])["settings"]
    assert saved == public
    assert config.normalize_settings(saved) == configured

    for group in ("state", "action"):
        public["policy"][group]["joint"]["enabled"] = True
        public["policy"][group]["pose"]["enabled"] = False
    activated = config.resolve_settings(config.normalize_settings(public), {})
    assert activated["policy"]["state"]["joint"]["reference"]["source"] == "action"
    assert activated["policy"]["action"]["joint"]["reference"]["scope"] == "step"
    preview = config.describe_policy(activated)
    assert preview["state_dim"] == 8
    assert preview["label_column"] == "state.joint"
    assert preview["prediction_shape"] == [50, 8]
    assert config.policy_semantics(activated) != config.policy_semantics(baseline)


@pytest.mark.parametrize("group", ["state", "action"])
@pytest.mark.parametrize("kind", ["joint", "pose"])
@pytest.mark.parametrize("enabled", ["true", 1, 0, None, [], {}])
def test_enabled_requires_a_yaml_boolean(tmp_path, group, kind, enabled):
    with pytest.raises(ValueError, match="enabled must be a YAML boolean"):
        resolve(tmp_path, {group: {kind: {"enabled": enabled}}})


@pytest.mark.parametrize("group", ["state", "action"])
@pytest.mark.parametrize("kind", ["joint", "pose"])
def test_null_branch_is_rejected_instead_of_hiding_its_children(tmp_path, group, kind):
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        resolve(tmp_path, {group: {kind: None}})


def test_disabled_coordinates_do_not_require_calibration_until_enabled(tmp_path):
    settings = resolve(
        tmp_path,
        {
            group: {
                "joint": {"enabled": True},
                "pose": {"enabled": False, "coordinates": {"ee_frame": "tcp"}},
            }
            for group in ("state", "action")
        },
    )
    metadata = connection_metadata({"policy_contract": settings["policy"]})
    assert not metadata["observation_spec"]["pose"]
    assert "base_to_parent" not in metadata["observation_spec"]["context_keys"]
    public = config.public_settings(settings)
    assert public["policy"]["state"]["pose"]["coordinates"]["ee_frame"] == "tcp"
    assert public["policy"]["action"]["pose"]["coordinates"]["ee_frame"] == "tcp"
    assert config.normalize_settings(public) == settings
    public["policy"]["state"]["pose"]["enabled"] = True
    with pytest.raises(ValueError, match="requires flange_to_tcp"):
        config.resolve_settings(config.normalize_settings(public), {})


@pytest.mark.parametrize("method", [None, "component", "se3"])
def test_v6_pose_combinations_build_and_restore_without_current_yaml(tmp_path, method, monkeypatch):
    spec = {"delta": method is not None, "method": method}
    if method is not None:
        spec["reference"] = {"source": "state", "scope": "step"}
    settings = resolve(tmp_path, {"action": {"pose": spec}})
    public = config.public_settings(settings)
    assert public["policy"]["action"]["pose"]["delta"] is (method is not None)
    assert public["policy"]["action"]["pose"]["method"] == method
    assert public["policy"]["action"]["pose"]["coordinates"]["ee_frame"] == "flange"
    monkeypatch.setattr(config, "load_settings", lambda *a, **kw: pytest.fail("snapshot must be self contained"))
    built = config.build_train_config(config.normalize_settings(public))
    assert yaml.safe_load(built.checkpoint_assets[config.CHECKPOINT_CONFIG])["settings"] == public


def test_removed_twist_is_rejected():
    saved = config.public_settings(config.load_settings())
    saved["policy"]["action"]["pose"]["quantity"] = "twist"
    with pytest.raises(ValueError, match="quantity must be one of"):
        config.normalize_settings(saved)
