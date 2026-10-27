from types import SimpleNamespace

import numpy as np
import pytest

from openpi.adapters.g2.policy import G2Policy


@pytest.mark.parametrize("arm", ["both", "left", "right"])
def test_wrapper_only_expands_selected_arm_and_preserves_native_result(arm):
    native_actions = np.full((2, 20 if arm == "both" else 10), 3.0, np.float32)
    native_result = {"actions": native_actions, "rtc_info": {"prefix_steps": 2}, "policy_timing": {"infer_ms": 1}}
    seen = []
    native = SimpleNamespace(
        metadata={"arm": arm},
        infer=lambda obs, **kwargs: seen.append(kwargs) or native_result,
        reset=lambda: seen.append("reset"),
    )
    wrapper = G2Policy(native)
    measured = np.arange(20, dtype=np.float32)
    obs = {"observation/state": np.zeros(16, np.float32), "observation/pose": measured}
    noise = np.ones((2, 32), np.float32)
    result = wrapper.infer(obs, noise=noise)
    assert result["actions"].shape == (2, 20)
    if arm != "both":
        selection = slice(0, 10) if arm == "left" else slice(10, 20)
        other = slice(10, 20) if arm == "left" else slice(0, 10)
        np.testing.assert_array_equal(result["actions"][:, selection], native_actions)
        np.testing.assert_array_equal(result["actions"][:, other], np.tile(measured[other], (2, 1)))
    assert result["rtc_info"] == native_result["rtc_info"]
    assert result["policy_timing"] == native_result["policy_timing"]
    assert native_result["actions"] is native_actions
    assert seen[0]["noise"] is noise
    wrapper.reset()
    assert seen[-1] == "reset"


def test_episode_reference_is_isolated_per_session_and_reset():
    import copy

    from openpi.adapters.g2.config import POLICY_DEFAULTS
    from openpi.adapters.g2.tests.geometry_test import poses

    settings = copy.deepcopy(POLICY_DEFAULTS)
    settings["action"]["relative_to"] = "episode_start_pose"
    native = SimpleNamespace(
        metadata={"arm": "both"}, infer=lambda obs: {"actions": np.zeros((2, 20), np.float32)}, reset=lambda: None
    )
    original = G2Policy(native, settings)
    a, b = original.new_session(), original.new_session()
    first, second, third = poses(1), poses(2), poses(3)
    np.testing.assert_allclose(a.infer({"observation/state": first})["actions"][0, :3], first[:3])
    np.testing.assert_allclose(b.infer({"observation/state": second})["actions"][0, :3], second[:3])
    np.testing.assert_allclose(a.infer({"observation/state": third})["actions"][0, :3], first[:3])
    a.reset()
    np.testing.assert_allclose(a.infer({"observation/state": third})["actions"][0, :3], third[:3])


def test_history_fallback_preserves_order_and_copies_observation_frames():
    import copy

    from openpi.adapters.g2.config import POLICY_DEFAULTS
    from openpi.adapters.g2.tests.geometry_test import poses

    settings = copy.deepcopy(POLICY_DEFAULTS)
    settings["timing"]["history_steps"] = 3
    seen = []

    def infer(obs):
        seen.append(obs)
        return {"actions": np.zeros((2, 20), np.float32)}

    wrapper = G2Policy(SimpleNamespace(metadata={"arm": "both"}, infer=infer), settings)
    frames = [poses(i) for i in range(4)]
    expected = [frame.copy() for frame in frames]
    for i, frame in enumerate(frames):
        wrapper.infer({"observation/state": frame, "observation/image": np.full((8, 8, 3), i, np.uint8)})
        indices = [max(0, j) for j in range(i - 2, i + 1)]
        np.testing.assert_array_equal(seen[-1]["observation/pose_history"], np.stack([expected[j] for j in indices]))
        np.testing.assert_array_equal(seen[-1]["observation/image"][:, 0, 0, 0], indices)
        frame[:] = -1  # Subsequent history must use the captured values, not caller-owned buffers.
