import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from openpi.adapters.g2.config import POLICY_DEFAULTS
from openpi.adapters.g2.config import _validate_rigid_transform
from openpi.adapters.g2.transforms import ActionCodec
from openpi.adapters.g2.transforms import matrix_pose
from openpi.adapters.g2.transforms import pose_matrix


def poses(seed, count=None):
    rng = np.random.default_rng(seed)
    shape = () if count is None else (count,)
    parts = []
    for _ in range(2):
        t = np.broadcast_to(np.eye(4), (*shape, 4, 4)).copy()
        t[..., :3, :3] = (
            Rotation.random(1 if count is None else count, random_state=rng).as_matrix().reshape(*shape, 3, 3)
        )
        t[..., :3, 3] = rng.normal(size=(*shape, 3))
        parts.append(matrix_pose(t, rng.uniform(size=shape)))
    return np.concatenate(parts, axis=-1)


@pytest.mark.parametrize("rotation", ["rot6d", "quaternion", "axis_angle"])
@pytest.mark.parametrize(
    "representation", ["absolute", "component_delta", "se3_body_delta", "se3_spatial_delta", "hybrid_delta"]
)
@pytest.mark.parametrize("scope", ["chunk", "step"])
@pytest.mark.parametrize("arm", ["both", "left", "right"])
def test_noncommuting_pose_codec_round_trip(rotation, representation, scope, arm):
    policy = copy.deepcopy(POLICY_DEFAULTS)
    policy["pose"].update(rotation=rotation)
    # Include nontrivial tool and parent transforms; identity-only tests miss multiplication-order bugs.
    parent = pose_matrix(poses(13)[:10])
    tools = {side: pose_matrix(poses(seed)[:10]).tolist() for side, seed in (("left", 14), ("right", 15))}
    policy["pose"].update(base_to_parent=parent.tolist(), flange_to_tcp=tools)
    policy["action"].update(
        representation=representation,
        delta_scope=None if representation == "absolute" else scope,
        relative_to=None if representation == "absolute" else "current_measured_pose",
    )
    obs = {"observation/pose": poses(1)}
    absolute = poses(2, 6)
    original = absolute.copy()
    codec = ActionCodec(policy, arm)
    encoded = codec.encode(absolute, obs)
    result = codec.decode(encoded, obs)
    chosen = slice(None) if arm == "both" else slice(0, 10) if arm == "left" else slice(10, 20)
    np.testing.assert_allclose(result[:, chosen], absolute[:, chosen], atol=3e-5)
    if arm != "both":
        other = slice(10, 20) if arm == "left" else slice(0, 10)
        np.testing.assert_array_equal(result[:, other], np.tile(obs["observation/pose"][other], (6, 1)))
    np.testing.assert_array_equal(absolute, original)


def test_body_and_spatial_deltas_use_distinct_frames():
    pose = matrix_pose(np.eye(4), 0.3)
    reference = pose.copy()
    reference[3:9] = [0, 1, 0, -1, 0, 0]  # +90 degrees around Z.
    target = reference.copy()
    target[0] = 1
    policy = copy.deepcopy(POLICY_DEFAULTS)
    obs = {"observation/pose": np.tile(reference, 2)}
    policy["action"]["representation"] = "se3_body_delta"
    body = ActionCodec(policy).encode(np.tile(target, 2)[None], obs)
    np.testing.assert_allclose(body[0, :3], [0, -1, 0], atol=1e-6)
    policy["action"]["representation"] = "hybrid_delta"
    hybrid = ActionCodec(policy).encode(np.tile(target, 2)[None], obs)
    np.testing.assert_allclose(hybrid[0, :3], [1, 0, 0], atol=1e-6)


@pytest.mark.parametrize("bad", [np.zeros((4, 4)).tolist(), np.diag([-1, 1, 1, 1]).tolist()])
def test_invalid_calibration_is_rejected(bad):
    with pytest.raises(ValueError, match="calibration"):
        _validate_rigid_transform(bad, "calibration")


@pytest.mark.parametrize(
    "reference", ["current_command_pose", "previous_command_pose", "episode_start_pose", "object_pose"]
)
def test_reference_is_explicit_and_required(reference):
    policy = copy.deepcopy(POLICY_DEFAULTS)
    policy["action"]["relative_to"] = reference
    codec = ActionCodec(policy)
    with pytest.raises(ValueError, match="requires"):
        codec.encode(poses(0, 3), {})
    key = {
        "current_command_pose": "command_pose",
        "previous_command_pose": "previous_command_pose",
        "episode_start_pose": "start_pose",
        "object_pose": "object_pose",
    }[reference]
    obs = {f"observation/{key}": poses(1), "observation/pose": poses(2)}
    target = poses(3, 3)
    np.testing.assert_allclose(codec.decode(codec.encode(target, obs), obs), target, atol=1e-6)


def test_dynamic_parent_frame_matches_calibration_and_round_trips():
    policy = copy.deepcopy(POLICY_DEFAULTS)
    policy["pose"].update(parent_frame="object", base_to_parent="observation")
    obs = {"observation/pose": poses(1), "observation/base_to_parent": pose_matrix(poses(2)[:10])}
    absolute = poses(3, 3)
    codec = ActionCodec(policy)
    np.testing.assert_allclose(codec.decode(codec.encode(absolute, obs), obs), absolute, atol=2e-6)
    with pytest.raises(ValueError, match="Dynamic parent"):
        codec.encode(absolute, {"observation/pose": poses(1)})


def test_object_reference_does_not_inherit_robot_tcp_offset():
    policy = copy.deepcopy(POLICY_DEFAULTS)
    tool = np.eye(4)
    tool[0, 3] = 1
    policy["pose"].update(ee_frame="tcp", flange_to_tcp={"left": tool.tolist(), "right": tool.tolist()})
    policy["action"].update(representation="se3_body_delta", relative_to="object_pose")
    origin = np.tile(matrix_pose(np.eye(4), 0.5), 2)
    obs = {"observation/pose": origin, "observation/object_pose": origin}
    codec = ActionCodec(policy)
    delta = codec.encode(origin[None], obs)
    np.testing.assert_allclose(delta[0, [0, 10]], [1, 1])
    np.testing.assert_allclose(codec.decode(delta, obs), origin[None], atol=1e-6)
