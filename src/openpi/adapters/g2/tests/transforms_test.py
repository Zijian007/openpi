import copy
import pickle

from g2_policy_protocol.data.encoding import ActionCodec
from g2_policy_protocol.data.policy import resolve_policy
import numpy as np
import pytest

from openpi.adapters.g2.config import POLICY_DEFAULTS
from openpi.adapters.g2.transforms import G2Inputs
from openpi.adapters.g2.transforms import prepare_inputs


@pytest.mark.parametrize(("arm", "start"), [("left", 0), ("right", 10), ("both", 0)])
def test_selected_state_actions_and_camera_masks(arm, start):
    size = 20 if arm == "both" else 10
    keys = ["observation/image"] + [
        f"observation/wrist_image_{side}" for side in ("left", "right") if arm in ("both", side)
    ]
    state = np.arange(20, dtype=np.float32)
    actions = np.arange(40, dtype=np.float32).reshape(2, 20)
    data = {
        "observation/state": state,
        "actions": actions,
        "prompt": b"demo",
        **{key: np.ones((224, 224, 3), np.uint8) for key in keys},
    }
    out = prepare_inputs(data, arm)
    np.testing.assert_array_equal(out["state"], state[start : start + size])
    np.testing.assert_array_equal(out["actions"], actions[:, start : start + size])
    assert bool(out["image_mask"]["left_wrist_0_rgb"]) == (arm != "right")
    assert bool(out["image_mask"]["right_wrist_0_rgb"]) == (arm != "left")
    assert out["prompt"] == "demo"
    # DeltaActions mutates its action input. Training data and RTC prefix must remain unchanged.
    out["actions"] -= 1
    np.testing.assert_array_equal(actions, np.arange(40).reshape(2, 20))


@pytest.mark.parametrize(("arm", "start"), [("left", 0), ("right", 10)])
def test_single_arm_absolute_output_is_mapped_into_dual_arm_wire(arm, start):
    state = np.arange(20, dtype=np.float32)
    actions = np.full((3, 10), 42, np.float32)
    from openpi.adapters.g2.tests.geometry_test import set_encoding

    policy = copy.deepcopy(POLICY_DEFAULTS)
    set_encoding(policy["action"]["pose"])
    out = ActionCodec(policy, arm).decode(actions, {"observation/pose": state})
    np.testing.assert_array_equal(out[:, start : start + 10], actions)
    other = slice(10, 20) if arm == "left" else slice(0, 10)
    np.testing.assert_array_equal(out[:, other], np.tile(state[other], (3, 1)))


def test_head_only_camera_config_masks_both_wrists_without_reading_them():
    data = {"observation/state": np.zeros(20, np.float32), "observation/image": np.ones((224, 224, 3), np.uint8)}
    out = prepare_inputs(data, "both", ("head_color",))
    assert out["image_mask"]["base_0_rgb"]
    assert not out["image_mask"]["left_wrist_0_rgb"]
    assert not out["image_mask"]["right_wrist_0_rgb"]
    assert not out["image"]["left_wrist_0_rgb"].any()
    assert not out["image"]["right_wrist_0_rgb"].any()


def test_wrist_only_camera_config_masks_head_without_reading_it():
    data = {
        "observation/state": np.zeros(20, np.float32),
        "observation/wrist_image_right": np.ones((224, 224, 3), np.uint8),
    }
    out = prepare_inputs(data, "right", ("hand_right_color",))
    assert not out["image_mask"]["base_0_rgb"]
    assert not out["image_mask"]["left_wrist_0_rgb"]
    assert out["image_mask"]["right_wrist_0_rgb"]
    assert not out["image"]["base_0_rgb"].any()


@pytest.mark.parametrize("with_actions", [False, True])
def test_combined_history_uses_canonical_observations_and_survives_worker_serialization(with_actions):
    policy = resolve_policy({
        "state": {"joint": {"enabled": True}},
        "action": {"pose": {"enabled": False}, "joint": {
            "enabled": True, "delta": True, "method": "component",
            "reference": {"source": "state", "scope": "chunk"},
        }},
        "timing": {"history_steps": 2},
    }, horizon=3)
    pose = np.tile([0, 0, 0, 1, 0, 0, 0, 1, 0, 0.4], 2).astype(np.float32)
    pose_history = np.stack([pose, pose])
    joint_history = np.arange(32, dtype=np.float32).reshape(2, 16)
    data = {
        "observation/pose": pose,
        "observation/pose_history": pose_history,
        "observation/joint": joint_history[-1],
        "observation/joint_history": joint_history,
        "observation/wrist_image_right": np.stack([
            np.full((3, 8, 12), 0.2, np.float32), np.full((3, 8, 12), 0.8, np.float32),
        ]),
        "prompt": b"demo",
    }
    if with_actions:
        data["actions"] = np.tile(joint_history[-1] + 1, (3, 1))
    original = copy.deepcopy(data)
    transform = G2Inputs(policy=policy, arm="right", cameras=("hand_right_color",))
    result = transform(data)
    expected_state = np.concatenate([pose_history[:, 10:], joint_history[:, 8:]], axis=-1).reshape(-1)
    np.testing.assert_allclose(result["state"], expected_state)
    assert result["prompt"] == "demo"
    image = result["image"]["right_wrist_0_rgb"]
    assert image.shape == (224, 448, 3)
    np.testing.assert_array_equal(image[112, 112], [51] * 3)
    np.testing.assert_array_equal(image[112, 336], [204] * 3)
    assert not result["image_mask"]["base_0_rgb"]
    assert not result["image_mask"]["left_wrist_0_rgb"]
    assert set(result) == {"state", "image", "image_mask", "prompt"} | ({"actions"} if with_actions else set())
    if with_actions:
        expected = np.ones((3, 8), np.float32)
        expected[:, -1] = data["actions"][:, -1]  # Grippers remain absolute.
        np.testing.assert_allclose(result["actions"], expected)
    # Native data loaders may serialize transforms when starting worker processes.
    restored = pickle.loads(pickle.dumps(transform))
    replay = restored(data)
    np.testing.assert_array_equal(replay["state"], result["state"])
    if with_actions:
        np.testing.assert_array_equal(replay["actions"], result["actions"])
        result["actions"][:] = 0
    result["state"][:] = 0
    for key, value in original.items():
        np.testing.assert_array_equal(data[key], value)
