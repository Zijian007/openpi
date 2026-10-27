"""State taxonomy, shared coordinates and observation-time history geometry."""

import copy
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from openpi.adapters.g2 import config as g2_config
from openpi.adapters.g2 import settings_schema
from openpi.adapters.g2.transforms import G2Inputs
from openpi.adapters.g2.transforms import matrix_pose
from openpi.adapters.g2.transforms import pose_matrix
from openpi.models.model import ModelType


def resolve_state(tmp_path, state, *, history=3, pose=None, arm="both"):
    path = tmp_path / "state.yaml"
    policy = {"state": state, "timing": {"history_steps": history}}
    if pose:
        policy["pose"] = pose
    path.write_text(yaml.safe_dump({"schema_version": 4, "policy": policy}))
    return g2_config.resolve_settings(g2_config.load_settings(path), {"arm": arm})


def transform(index):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec([0.13 * index, 0.2, -0.17 * index]).as_matrix()
    result[:3, 3] = [0.2 * index, -0.1, 0.3]
    return result


def pose_pair(index):
    return np.concatenate([matrix_pose(transform(index), 0.1 * index), matrix_pose(transform(index + 1), 0.2)])


def observation():
    history = np.stack([pose_pair(i) for i in (1, 2, 3)])
    return {
        "observation/state": history[-1],
        "observation/pose": history[-1],
        "observation/pose_history": history,
        "observation/image": np.zeros((8, 8, 3), np.uint8),
        "observation/command_pose": pose_pair(4),
        "observation/previous_command_pose": pose_pair(5),
        "observation/start_pose": pose_pair(6),
        "observation/object_pose": pose_pair(7),
    }


@pytest.mark.parametrize("method", ["absolute", *settings_schema.STATE_METHODS])
@pytest.mark.parametrize("rotation", ["rot6d", "quaternion", "axis_angle"])
@pytest.mark.parametrize("arm", ["both", "left", "right"])
@pytest.mark.parametrize("anchor", ["current_measured", "object"])
def test_state_history_uses_shared_pose_coordinates_and_one_fixed_anchor(tmp_path, method, rotation, arm, anchor):
    parent, left_tool, right_tool = (transform(i) for i in (0.5, 0.3, -0.2))
    spatial = {"mode": "absolute"} if method == "absolute" else {"mode": "relative", "method": method}
    temporal = {"anchor": None} if method == "absolute" else {"anchor": anchor}
    settings = resolve_state(
        tmp_path,
        {"space": "eef", "spatial": spatial, "temporal": temporal},
        pose={
            "rotation": rotation,
            "parent_frame": "calibrated_parent",
            "ee_frame": "tool_tcp",
            "base_to_parent": parent.tolist(),
            "flange_to_tcp": {"left": left_tool.tolist(), "right": right_tool.tolist()},
        },
        arm=arm,
    )
    obs = observation()
    original = copy.deepcopy(obs)
    result = G2Inputs(ModelType.PI05, policy=settings["policy"], arm=arm, cameras=("head_color",))(obs)
    reference = obs["observation/pose" if anchor == "current_measured" else "observation/object_pose"]
    expected_frames = []
    for frame in obs["observation/pose_history"]:
        parts = []
        for side, tool in enumerate((left_tool, right_tool)):
            if arm != "both" and side != (0 if arm == "left" else 1):
                continue
            start = side * 10
            target = parent @ pose_matrix(frame[start : start + 10]) @ tool
            ref = parent @ pose_matrix(reference[start : start + 10])
            # A reference object is a coordinate frame, not a robot flange.
            if anchor != "object":
                ref = ref @ tool
            if method == "absolute":
                value = matrix_pose(target, frame[start + 9], rotation)
            elif method == "component_difference":
                value = matrix_pose(target, frame[start + 9], rotation)
                value[:-1] -= matrix_pose(ref, 0.0, rotation)[:-1]
            else:
                if method == "se3_body":
                    delta = np.linalg.inv(ref) @ target
                elif method == "se3_spatial":
                    delta = target @ np.linalg.inv(ref)
                else:
                    delta = np.eye(4)
                    delta[:3, 3] = target[:3, 3] - ref[:3, 3]
                    delta[:3, :3] = ref[:3, :3].T @ target[:3, :3]
                value = matrix_pose(delta, frame[start + 9], rotation)
            parts.append(value)
        expected_frames.append(np.concatenate(parts))
    np.testing.assert_allclose(result["state"], np.asarray(expected_frames).reshape(-1), atol=2e-6)
    for key, value in original.items():
        np.testing.assert_array_equal(obs[key], value)
    assert g2_config.describe_policy(settings)["state_dim"] == len(result["state"])
    assert g2_config.upgrade_settings(g2_config.public_settings(settings)) == settings


@pytest.mark.parametrize("method", list(settings_schema.STATE_METHODS))
def test_combined_state_uses_pose_geometry_and_joint_component_differences(tmp_path, method):
    settings = resolve_state(
        tmp_path,
        {"space": "both", "encoding": {"spatial": method, "temporal": {"anchor": "episode_start"}}},
    )
    obs = observation()
    joints = np.arange(48, dtype=np.float32).reshape(3, 16) / 50
    obs.update(
        {"observation/joint": joints[-1], "observation/joint_history": joints, "observation/start_joint": joints[0]}
    )
    result = G2Inputs(ModelType.PI05, policy=settings["policy"], cameras=("head_color",))(obs)["state"].reshape(3, 36)
    expected = joints - joints[0]
    expected[:, [7, 15]] = joints[:, [7, 15]]
    np.testing.assert_array_equal(result[:, 20:], expected)
    np.testing.assert_array_equal(result[:, [9, 19]], obs["observation/pose_history"][:, [9, 19]])


def test_state_encoding_is_independent_of_action_quantity_and_chunk_scope(tmp_path):
    settings = resolve_state(tmp_path, {"encoding": {"spatial": "se3_body", "temporal": {"anchor": "current_command"}}})
    policy = settings["policy"]
    before = G2Inputs(ModelType.PI05, policy=policy, cameras=("head_color",))(observation())["state"]
    changed = copy.deepcopy(policy)
    changed["action"].update(
        space="joint", control_quantity="joint_velocity", representation="absolute", relative_to=None, delta_scope=None
    )
    after = G2Inputs(ModelType.PI05, policy=changed, cameras=("head_color",))(observation())["state"]
    np.testing.assert_array_equal(before, after)
    changed["action"].update(
        space="eef",
        control_quantity="pose_target",
        representation="hybrid_delta",
        relative_to="object_pose",
        delta_scope="step",
    )
    after = G2Inputs(ModelType.PI05, policy=changed, cameras=("head_color",))(observation())["state"]
    np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("space", ["eef", "joint", "both"])
def test_legacy_component_state_aliases_restore_the_same_policy_and_statistics(tmp_path, space):
    old = resolve_state(tmp_path, {"space": space, "encoding": "component_delta", "reference": "episode_start"})
    new = resolve_state(
        tmp_path,
        {"space": space, "encoding": {"spatial": "component_difference", "temporal": {"anchor": "episode_start"}}},
    )
    assert old == new
    assert old["policy"]["state"]["representation"] == "relative"
    assert g2_config.policy_semantics(old) == g2_config.policy_semantics(new)


@pytest.mark.parametrize("method", list(settings_schema.STATE_METHODS))
def test_state_snapshot_restores_without_reading_the_current_yaml(tmp_path, method):
    settings = resolve_state(tmp_path, {"encoding": {"spatial": method, "temporal": {"anchor": "episode_start"}}})
    config = g2_config.build_train_config(settings)
    snapshot = yaml.safe_load(config.checkpoint_assets[g2_config.CHECKPOINT_CONFIG])["settings"]
    assert snapshot["policy"]["state"]["spatial"]["method"] == method
    with patch.object(g2_config, "load_settings", side_effect=AssertionError("must not reload template")):
        restored = g2_config.build_train_config(snapshot)
    assert restored.data.policy == config.data.policy
    assert restored.model == config.model


@pytest.mark.parametrize(
    ("state", "match"),
    [
        ({"encoding": {"mode": "relative"}}, "spatial is required"),
        ({"encoding": {"mode": "delta"}}, "mode must be absolute or relative"),
        ({"encoding": {"spatial": "typo"}}, "spatial must be one of"),
        ({"encoding": {"mode": "absolute", "spatial": "hybrid"}}, "must be null"),
        ({"encoding": {"mode": "absolute", "temporal": {}}}, "must be null"),
        ({"encoding": {"temporal": "episode_start"}}, "temporal must be a YAML mapping"),
        ({"encoding": {"temporal": {"anchor": "typo"}}}, "anchor must be one of"),
        ({"encoding": {"spatial": "hybrid", "temporal": None}}, "Relative state requires"),
        ({"encoding": {"temporal": {"reference_scope": "chunk"}}}, "Unknown YAML field"),
        ({"encoding": {}, "reference": "episode_start"}, "Cannot mix"),
        (
            {"space": "joint", "encoding": {"spatial": "se3_body", "temporal": {"anchor": "episode_start"}}},
            "Joint state supports",
        ),
        (
            {"space": "both", "encoding": {"spatial": "hybrid", "temporal": {"anchor": "current_measured"}}},
            "Relative joint state",
        ),
    ],
)
def test_invalid_or_action_only_state_options_are_rejected(tmp_path, state, match):
    with pytest.raises(ValueError, match=match):
        resolve_state(tmp_path, state)


def test_absolute_state_override_clears_its_reference(tmp_path):
    relative = resolve_state(tmp_path, {"encoding": {"spatial": "hybrid", "temporal": {"anchor": "episode_start"}}})
    settings = g2_config.resolve_settings(relative, {"state_mode": "absolute"})
    assert settings["policy"]["state"] == {"space": "eef", "representation": "absolute", "relative_to": None}
    public = g2_config.public_settings(settings)["policy"]["state"]
    assert public["spatial"] == {"mode": "absolute", "method": None}
    assert public["temporal"] == {"anchor": None}


def test_geometric_single_frame_self_reference_warns_and_encodes_identity(tmp_path):
    with pytest.warns(UserWarning, match="identity relative rotation"):
        settings = resolve_state(
            tmp_path, {"encoding": {"spatial": "se3_body", "temporal": {"anchor": "current_measured"}}}, history=1
        )
    obs = observation()
    obs["observation/pose_history"] = obs["observation/pose"][None]
    result = G2Inputs(ModelType.PI05, policy=settings["policy"], cameras=("head_color",))(obs)["state"]
    for offset in (0, 10):
        np.testing.assert_allclose(pose_matrix(result[offset : offset + 10]), np.eye(4), atol=1e-6)
        assert result[offset + 9] == obs["observation/pose"][offset + 9]


def test_inference_wrapper_encodes_explicit_history_like_training(tmp_path):
    from openpi.adapters.g2.policy import G2Policy

    settings = resolve_state(tmp_path, {"encoding": {"spatial": "se3_body", "temporal": {"anchor": "episode_start"}}})
    inputs = G2Inputs(ModelType.PI05, policy=settings["policy"], cameras=("head_color",))

    class CapturingPolicy:
        def __init__(self):
            self.metadata = {"arm": "both"}

        def infer(self, obs):
            encoded_state = inputs(obs)["state"]
            return {"encoded_state": encoded_state, "actions": np.zeros((2, 20), np.float32)}

    wrapped = G2Policy(CapturingPolicy(), settings["policy"])
    result = wrapped.infer(observation())
    np.testing.assert_array_equal(result["encoded_state"], inputs(observation())["state"])
