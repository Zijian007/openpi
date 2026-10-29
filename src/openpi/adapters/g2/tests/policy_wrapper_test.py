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
    import copy

    from openpi.adapters.g2.config import POLICY_DEFAULTS

    from openpi.adapters.g2.tests.geometry_test import set_encoding
    from openpi.adapters.g2.tests.geometry_test import set_state_space

    settings = copy.deepcopy(POLICY_DEFAULTS)
    set_state_space(settings, "joint")
    set_encoding(settings["action"]["pose"])
    wrapper = G2Policy(native, settings)
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


def test_command_reference_comes_from_the_request_not_session_state():
    import copy

    from openpi.adapters.g2.config import POLICY_DEFAULTS
    from openpi.adapters.g2.tests.geometry_test import poses

    from openpi.adapters.g2.tests.geometry_test import set_encoding

    settings = copy.deepcopy(POLICY_DEFAULTS)
    set_encoding(settings["action"]["pose"], "component_delta", scope="chunk", source="action")
    native = SimpleNamespace(
        metadata={"arm": "both"}, infer=lambda obs: {"actions": np.zeros((2, 20), np.float32)}, reset=lambda: None
    )
    original = G2Policy(native, settings)
    a, b = original.new_session(), original.new_session()
    first, second, third = poses(1), poses(2), poses(3)
    np.testing.assert_allclose(
        a.infer({"observation/state": first, "observation/pose": first, "observation/command_pose": first})["actions"][0, :3],
        first[:3],
    )
    np.testing.assert_allclose(
        b.infer({"observation/state": second, "observation/pose": second, "observation/command_pose": second})["actions"][0, :3],
        second[:3],
    )
    np.testing.assert_allclose(
        a.infer({"observation/state": third, "observation/pose": third, "observation/command_pose": first})["actions"][0, :3],
        first[:3],
    )
    a.reset()
    np.testing.assert_allclose(
        a.infer({"observation/state": third, "observation/pose": third, "observation/command_pose": third})["actions"][0, :3],
        third[:3],
    )


def test_server_does_not_invent_observation_history():
    import copy
    from openpi.adapters.g2.config import POLICY_DEFAULTS

    settings = copy.deepcopy(POLICY_DEFAULTS)
    settings["timing"]["history_steps"] = 3
    wrapper = G2Policy(SimpleNamespace(metadata={"arm": "both"}), settings)
    context = wrapper.prepare_observation({"observation/state": np.zeros(20)})
    assert "observation/pose_history" not in context
    assert "observation/start_pose" not in context


def test_explicit_histories_are_used_without_reconstructing_incomplete_cached_history():
    import copy

    from openpi.adapters.g2.config import POLICY_DEFAULTS
    from openpi.adapters.g2.tests.geometry_test import poses

    settings = copy.deepcopy(POLICY_DEFAULTS)
    settings["timing"]["history_steps"] = 3
    native = SimpleNamespace(metadata={"arm": "both"}, infer=lambda obs: {"actions": np.zeros((2, 20), np.float32)})
    wrapper = G2Policy(native, settings)
    first, current = poses(1), poses(2)
    pose_history = np.stack([first, first, current])
    joint_history = np.arange(48, dtype=np.float32).reshape(3, 16)
    context = wrapper.prepare_observation(
        {
            "observation/state": current,
            "observation/pose": current,
            "observation/joint": joint_history[-1],
            "observation/pose_history": pose_history,
            "observation/joint_history": joint_history,
        }
    )
    assert context["observation/pose_history"] is pose_history
    assert context["observation/joint_history"] is joint_history
    assert "observation/start_joint" not in context  # No first-frame joints were observed.
