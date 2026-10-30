"""Independent State/Action coordinates, canonical references and self-contained snapshots."""

import copy
from unittest.mock import patch

from g2_policy_protocol.data.encoding import ActionCodec
from g2_policy_protocol.data.encoding import matrix_pose
from g2_policy_protocol.data.encoding import pose_matrix
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from openpi.adapters.g2 import config as g2_config
from openpi.adapters.g2.policy import G2Policy
from openpi.adapters.g2.server import connection_metadata
from openpi.adapters.g2.transforms import G2Inputs


def resolve(tmp_path, policy):
    path = tmp_path / "pose.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 7, "policy": policy}))
    return g2_config.resolve_settings(g2_config.load_settings(path), {})


def transform(index):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec([0.17 * index, -0.23, 0.31 * index]).as_matrix()
    result[:3, 3] = [0.13 * index, -0.1, 0.2]
    return result


def coordinates(index, rotation):
    return {
        "ee_frame": f"tcp_{index}",
        "rotation": rotation,
        "parent_frame": "arm_base_link" if index > 0 else "base_link",
        "flange_to_tcp": {"left": transform(index + 1).tolist(), "right": transform(index + 2).tolist()},
    }


def pair(index):
    return np.concatenate([matrix_pose(transform(index), 0.2), matrix_pose(transform(index + 1), 0.7)])


@pytest.mark.parametrize("method", ["component", "se3"])
@pytest.mark.parametrize("scope", ["chunk", "step"])
@pytest.mark.parametrize("state_relative", [False, True])
@pytest.mark.parametrize("rotations", [("quaternion", "rot6d"), ("axis_angle", "quaternion"), ("rot6d", "axis_angle")])
def test_independent_coordinates_match_manual_geometry_and_decode_canonical_targets(
    tmp_path, method, scope, state_relative, rotations
):
    state_rotation, action_rotation = rotations
    state_pose, action_pose = coordinates(0.4, state_rotation), coordinates(-0.6, action_rotation)
    settings = resolve(
        tmp_path,
        {
            "state": {
                "pose": {
                    "coordinates": state_pose,
                    "delta": state_relative,
                    "method": "se3" if state_relative else None,
                    "reference": {"source": "state", "scope": "chunk"} if state_relative else {"source": None, "scope": None},
                },
            },
            "action": {
                "pose": {
                    "coordinates": action_pose,
                    "delta": True,
                    "method": method,
                    "reference": {"source": "state", "scope": scope},
                }
            },
            "timing": {"history_steps": 2},
        },
    )
    history = np.stack([pair(1), pair(2)])
    targets = np.stack([pair(3), pair(4), pair(5)])
    obs = {
        "observation/state": history[-1], "observation/pose": history[-1],
        "observation/pose_history": history, "observation/base_to_parent": transform(0.4),
    }
    inputs = G2Inputs(policy=settings["policy"], arm="right", cameras=())
    encoded = inputs(dict(obs, actions=targets))

    state_parent = obs["observation/base_to_parent"]
    state_tool = np.asarray(state_pose["flange_to_tcp"]["right"])
    state_ref = state_parent @ pose_matrix(history[-1, 10:]) @ state_tool
    expected_state = []
    for frame in history:
        state_target = state_parent @ pose_matrix(frame[10:]) @ state_tool
        if state_relative:
            state_target = np.linalg.inv(state_ref) @ state_target
        expected_state.append(matrix_pose(state_target, frame[-1], state_rotation))
    np.testing.assert_allclose(encoded["state"], np.concatenate(expected_state), atol=2e-6)
    assert len(encoded["state"]) == g2_config.state_dimension(settings)

    action_parent = np.eye(4)
    action_tool = np.asarray(action_pose["flange_to_tcp"]["right"])
    ref = action_parent @ pose_matrix(history[-1, 10:]) @ action_tool
    expected_actions = []
    for frame in targets:
        target = action_parent @ pose_matrix(frame[10:]) @ action_tool
        if method == "component":
            value = matrix_pose(target, frame[-1], action_rotation)
            value[:-1] -= matrix_pose(ref, 0, action_rotation)[:-1]
        else:
            value = matrix_pose(np.linalg.inv(ref) @ target, frame[-1], action_rotation)
        expected_actions.append(value)
        if scope == "step":
            ref = target
    np.testing.assert_allclose(encoded["actions"], expected_actions, atol=2e-6)
    codec = ActionCodec(settings["policy"], "right")
    decoded = codec.decode(encoded["actions"], obs)
    np.testing.assert_allclose(decoded[:, 10:], targets[:, 10:], atol=3e-6)
    np.testing.assert_array_equal(decoded[:, :10], np.tile(history[-1, :10], (3, 1)))

    class NativePolicy:
        def __init__(self):
            self.metadata = {"arm": "right"}

        def infer(self, context):
            np.testing.assert_allclose(inputs(context)["state"], encoded["state"], atol=2e-6)
            return {"actions": encoded["actions"]}

    deployed = G2Policy(NativePolicy(), settings["policy"]).infer(obs)
    np.testing.assert_allclose(deployed["actions"], decoded, atol=3e-6)


def test_independent_pose_overrides_and_checkpoint_restoration(tmp_path):
    state_pose, action_pose = coordinates(1, "quaternion"), coordinates(2, "axis_angle")
    settings = resolve(
        tmp_path, {"state": {"pose": {"coordinates": state_pose}}, "action": {"pose": {"coordinates": action_pose}}}
    )
    public = g2_config.public_settings(settings)
    assert "pose" not in public["policy"]
    assert "state_pose" not in public["policy"]
    assert public["policy"]["state"]["pose"]["coordinates"] == state_pose
    assert public["policy"]["action"]["pose"]["coordinates"] == action_pose
    assert g2_config.normalize_settings(public) == settings
    with patch.object(g2_config, "load_settings", side_effect=AssertionError("must not read current YAML")):
        restored = g2_config.build_train_config(g2_config.normalize_settings(public))
    assert restored.data.policy == settings["policy"]
    saved = yaml.safe_load(restored.checkpoint_assets[g2_config.CHECKPOINT_CONFIG])["settings"]
    assert saved == public
    assert g2_config.describe_policy(settings)["prediction_shape"] == [50, 7]
    assert g2_config.state_dimension(settings) == 8

    state_only = resolve(tmp_path, {"state": {"pose": {"coordinates": {"rotation": "quaternion"}}}})
    assert state_only["policy"]["action"]["pose"]["coordinates"]["rotation"] == "rot6d"
    action_only = resolve(tmp_path, {"action": {"pose": {"coordinates": {"rotation": "axis_angle"}}}})
    assert action_only["policy"]["state"]["pose"]["coordinates"]["rotation"] == "rot6d"


@pytest.mark.parametrize("group", ["state", "action"])
@pytest.mark.parametrize(
    ("pose", "message"),
    [
        (None, "must be a YAML mapping"),
        ({"parent_fram": "world"}, "Unknown YAML field"),
        ({"parent_frame": "world"}, "parent_frame must be one of"),
        ({"ee_frame": "tcp"}, "requires flange_to_tcp"),
        ({"rotation": "euler"}, "rotation must be"),
        ({"base_to_parent": [[1]]}, "Unknown YAML field"),
    ],
)
def test_invalid_pose_settings_report_the_group(tmp_path, group, pose, message):
    with pytest.raises(ValueError, match=message):
        resolve(tmp_path, {group: {"pose": {"coordinates": pose}}})


@pytest.mark.parametrize("group", ["state", "action"])
def test_dynamic_parent_in_either_group_is_advertised_and_used(tmp_path, group):
    settings = resolve(
        tmp_path, {group: {"pose": {"coordinates": {"parent_frame": "arm_base_link"}}}}
    )
    metadata = connection_metadata({"policy_contract": settings["policy"], "required_cameras": []})
    assert "base_to_parent" in metadata["observation_spec"]["context_keys"]
    obs = {"observation/state": pair(2), "observation/pose": pair(2), "observation/base_to_parent": transform(0.3)}
    inputs = G2Inputs(policy=settings["policy"], arm="right", cameras=())
    result = inputs(dict(obs, actions=pair(3)[None]))
    static = copy.deepcopy(settings)
    static["policy"][group]["pose"]["coordinates"]["parent_frame"] = "base_link"
    canonical = dict(obs, actions=pair(3)[None])
    for key in ("observation/state", "observation/pose", "actions"):
        value = canonical[key]
        canonical[key] = np.concatenate([
            matrix_pose(transform(0.3) @ pose_matrix(value[..., offset:offset + 10]), value[..., offset + 9])
            for offset in (0, 10)
        ], axis=-1)
    expected = G2Inputs(policy=static["policy"], arm="right", cameras=())(canonical)
    key = "state" if group == "state" else "actions"
    np.testing.assert_allclose(result[key], expected[key], atol=2e-6)
    obs.pop("observation/base_to_parent")
    with pytest.raises(ValueError, match="arm_base_link requires"):
        inputs(dict(obs, actions=pair(3)[None]))


def test_two_dynamic_parents_share_the_same_observation_transform(tmp_path):
    settings = resolve(
        tmp_path,
        {
            "state": {"pose": {"coordinates": {"parent_frame": "arm_base_link"}}},
            "action": {"pose": {"coordinates": {"parent_frame": "arm_base_link"}}},
        },
    )
    metadata = connection_metadata({"policy_contract": settings["policy"], "required_cameras": []})
    assert "base_to_parent" in metadata["observation_spec"]["context_keys"]
