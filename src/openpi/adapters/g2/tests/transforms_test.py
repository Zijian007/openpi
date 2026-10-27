import numpy as np
import pytest

from openpi.adapters.g2.transforms import expand_arm_actions
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
    out = expand_arm_actions(actions, state, arm)
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
