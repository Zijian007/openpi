import copy

from g2_policy_protocol.data.encoding import ActionCodec
from g2_policy_protocol.data.encoding import matrix_pose
from g2_policy_protocol.data.encoding import pose_matrix
from g2_policy_protocol.validation import validate_rigid_transform
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from openpi.adapters.g2.config import POLICY_DEFAULTS


def set_encoding(spec, representation="absolute", *, scope=None, source="state"):
    if representation in (None, "absolute"):
        spec.update(delta=False, method=None, reference={"source": None, "scope": None})
        return
    spec.update(
        delta=True,
        method="se3" if representation == "se3_body_delta" else "component",
        reference={"source": source, "scope": scope},
    )


def set_state_space(policy, space):
    policy["state"]["pose"]["enabled"] = space in ("eef", "both")
    policy["state"]["joint"]["enabled"] = space in ("joint", "both")


def set_joint_action(policy, quantity="position", representation="absolute", *, scope=None, source="action"):
    policy["action"]["pose"]["enabled"] = False
    spec = policy["action"]["joint"]
    spec["enabled"] = True
    spec["quantity"] = {
        "joint_position": "position",
        "joint_velocity": "velocity",
        "joint_torque": "torque",
        "position": "position",
        "velocity": "velocity",
        "torque": "torque",
    }[quantity]
    spec["source"] = source
    set_encoding(spec, representation, scope=scope)


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
    "representation", ["absolute", "component_delta", "se3_body_delta"]
)
@pytest.mark.parametrize("scope", ["chunk", "step"])
@pytest.mark.parametrize("arm", ["both", "left", "right"])
def test_noncommuting_pose_codec_round_trip(rotation, representation, scope, arm):
    policy = copy.deepcopy(POLICY_DEFAULTS)
    coordinates = policy["action"]["pose"]["coordinates"]
    coordinates["rotation"] = rotation
    # Include nontrivial tool and parent transforms; identity-only tests miss multiplication-order bugs.
    parent = pose_matrix(poses(13)[:10])
    tools = {side: pose_matrix(poses(seed)[:10]).tolist() for side, seed in (("left", 14), ("right", 15))}
    coordinates.update(parent_frame="arm_base_link", flange_to_tcp=tools)
    set_encoding(
        policy["action"]["pose"],
        representation,
        scope=None if representation == "absolute" else scope,
    )
    obs = {"observation/pose": poses(1), "observation/base_to_parent": parent}
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


def test_se3_and_component_use_distinct_frames():
    pose = matrix_pose(np.eye(4), 0.3)
    reference = pose.copy()
    reference[3:9] = [0, 1, 0, -1, 0, 0]  # +90 degrees around Z.
    target = reference.copy()
    target[0] = 1
    policy = copy.deepcopy(POLICY_DEFAULTS)
    obs = {"observation/pose": np.tile(reference, 2)}
    set_encoding(policy["action"]["pose"], "se3_body_delta", scope="chunk")
    body = ActionCodec(policy).encode(np.tile(target, 2)[None], obs)
    np.testing.assert_allclose(body[0, :3], [0, -1, 0], atol=1e-6)
    set_encoding(policy["action"]["pose"], "component_delta", scope="chunk")
    component = ActionCodec(policy).encode(np.tile(target, 2)[None], obs)
    np.testing.assert_allclose(component[0, :3], [1, 0, 0], atol=1e-6)


@pytest.mark.parametrize("bad", [np.zeros((4, 4)).tolist(), np.diag([-1, 1, 1, 1]).tolist()])
def test_invalid_calibration_is_rejected(bad):
    with pytest.raises(ValueError, match="calibration"):
        validate_rigid_transform(bad, "calibration")


@pytest.mark.parametrize("source,key", [("state", "pose"), ("action", "command_pose")])
def test_reference_is_explicit_and_required(source, key):
    policy = copy.deepcopy(POLICY_DEFAULTS)
    policy["action"]["pose"]["reference"]["source"] = source
    codec = ActionCodec(policy)
    with pytest.raises(ValueError, match="requires"):
        codec.encode(poses(0, 3), {})
    obs = {f"observation/{key}": poses(1), "observation/pose": poses(2)}
    target = poses(3, 3)
    np.testing.assert_allclose(codec.decode(codec.encode(target, obs), obs), target, atol=1e-6)


def test_dynamic_parent_frame_matches_calibration_and_round_trips():
    policy = copy.deepcopy(POLICY_DEFAULTS)
    policy["action"]["pose"]["coordinates"]["parent_frame"] = "arm_base_link"
    obs = {"observation/pose": poses(1), "observation/base_to_parent": pose_matrix(poses(2)[:10])}
    absolute = poses(3, 3)
    codec = ActionCodec(policy)
    np.testing.assert_allclose(codec.decode(codec.encode(absolute, obs), obs), absolute, atol=2e-6)
    with pytest.raises(ValueError, match="arm_base_link requires"):
        codec.encode(absolute, {"observation/pose": poses(1)})


def test_tcp_offset_applies_to_robot_state_and_action_references():
    policy = copy.deepcopy(POLICY_DEFAULTS)
    tool = np.eye(4)
    tool[0, 3] = 1
    policy["action"]["pose"]["coordinates"].update(
        ee_frame="tcp", flange_to_tcp={"left": tool.tolist(), "right": tool.tolist()}
    )
    set_encoding(policy["action"]["pose"], "se3_body_delta", scope="chunk")
    origin = np.tile(matrix_pose(np.eye(4), 0.5), 2)
    obs = {"observation/pose": origin}
    codec = ActionCodec(policy)
    delta = codec.encode(origin[None], obs)
    np.testing.assert_allclose(delta[0, :3], 0, atol=1e-6)
    np.testing.assert_allclose(codec.decode(delta, obs), origin[None], atol=1e-6)
