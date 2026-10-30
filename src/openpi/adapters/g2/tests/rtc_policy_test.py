import copy
from types import SimpleNamespace

from g2_policy_protocol.data.encoding import action_dimensions
import numpy as np
import pytest
import torch

pytest.importorskip("flax")

from openpi import transforms
from openpi.adapters.g2 import transforms as g2_transforms
from openpi.adapters.g2.policy import G2Policy
from openpi.models import model
from openpi.models.rtc import RTCConfig
from openpi.policies.policy import Policy as NativePolicy


def _absolute_settings():
    from openpi.adapters.g2.config import POLICY_DEFAULTS
    from openpi.adapters.g2.tests.geometry_test import set_encoding

    settings = copy.deepcopy(POLICY_DEFAULTS)
    set_encoding(settings["action"]["pose"])
    return settings


class EchoModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(action_horizon=50, action_dim=32)
        self.kwargs = None

    def sample_actions(self, device, observation, **kwargs):
        self.kwargs = kwargs
        return kwargs["rtc_prefix"]


@pytest.mark.parametrize(("arm", "start"), [("left", 0), ("right", 10)])
@pytest.mark.parametrize("state_space", ["eef", "joint"])
def test_absolute_single_arm_rtc_uses_pose_for_inactive_arm(tmp_path, monkeypatch, arm, start, state_space):
    from openpi.adapters.g2 import presets as g2_configs
    from openpi.training import config as config_lib

    monkeypatch.setattr(config_lib.DataConfigFactory, "_load_norm_stats", lambda *args: None)
    monkeypatch.setattr(config_lib.ModelTransformFactory, "__call__", lambda *args: transforms.Group())
    from openpi.adapters.g2.config import POLICY_DEFAULTS
    from openpi.adapters.g2.tests.geometry_test import set_encoding
    from openpi.adapters.g2.tests.geometry_test import set_state_space

    settings = copy.deepcopy(POLICY_DEFAULTS)
    set_state_space(settings, state_space)
    set_encoding(settings["action"]["pose"])
    data = g2_configs.LeRobotG2DataConfig(repo_id="test", arm=arm, policy=settings, cameras=("head_color",)).create(
        tmp_path, SimpleNamespace(model_type=model.ModelType.PI05)
    )
    stats = {"actions": transforms.NormStats(mean=np.ones(10), std=np.full(10, 2.0))}
    policy = G2Policy(
        NativePolicy(
            EchoModel(),
            is_pytorch=True,
            rtc_config=RTCConfig(),
            metadata={"robot": "g2", "arm": arm, "action_dim": 20, "model_action_dim": 10},
            transforms=[*data.data_transforms.inputs, transforms.Normalize(stats), transforms.PadStatesAndActions(32)],
            output_transforms=[transforms.Unnormalize(stats), *data.data_transforms.outputs],
        ),
        settings,
    )
    pose = np.arange(20, dtype=np.float32) / 10
    obs = {
        "observation/state": pose if state_space == "eef" else np.zeros(16, np.float32),
        "observation/pose": pose,
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "rtc_enabled": True,
        "prev_chunk_left_over": np.full((6, 20), 3.0, np.float32),
        "inference_delay": 2,
    }
    actions = policy.infer(obs)["actions"]
    np.testing.assert_allclose(actions[:6, start : start + 10], 3.0, atol=1e-6)
    other = slice(10, 20) if arm == "left" else slice(0, 10)
    np.testing.assert_array_equal(actions[:, other], np.tile(pose[other], (50, 1)))


@pytest.mark.parametrize(("arm", "start"), [("left", 0), ("right", 10)])
def test_single_arm_rtc_round_trip_keeps_correct_side_and_ignores_inactive_prefix(arm, start):
    echo_model = EchoModel()
    stats = {"actions": transforms.NormStats(mean=np.ones(10), std=np.full(10, 2.0))}
    policy = G2Policy(
        NativePolicy(
            echo_model,
            is_pytorch=True,
            rtc_config=RTCConfig(),
            metadata={"robot": "g2", "arm": arm, "action_dim": 20, "model_action_dim": 10},
            transforms=[
                g2_transforms.G2Inputs(arm=arm),
                transforms.DeltaActions([True] * 9 + [False]),
                transforms.Normalize(stats),
                transforms.PadStatesAndActions(32),
            ],
            output_transforms=[
                transforms.Unnormalize(stats),
                transforms.AbsoluteActions([True] * 9 + [False]),
                g2_transforms.G2Outputs(action_dim=10),
            ],
        ),
        _absolute_settings(),
    )
    obs = g2_transforms.make_g2_example()
    obs.pop("prompt")  # Numeric echo model has no language tokenizer.
    inactive = "left" if arm == "right" else "right"
    obs.pop(f"observation/wrist_image_{inactive}")
    obs["observation/state"] = np.arange(20, dtype=np.float32) / 10
    prefix = np.full((6, 20), 3.0, dtype=np.float32)
    obs.update(rtc_enabled=True, prev_chunk_left_over=prefix, inference_delay=2)
    result = policy.infer(obs)["actions"]
    np.testing.assert_allclose(result[:6, start : start + 10], 3.0, atol=1e-6)
    other = slice(10, 20) if arm == "left" else slice(0, 10)
    np.testing.assert_allclose(result[:, other], np.tile(obs["observation/state"][other], (50, 1)))
    assert not echo_model.kwargs["rtc_weights"].numpy()[..., 10:].any()


@pytest.mark.parametrize("quantiles", [False, True])
def test_absolute_prefix_reanchors_normalizes_and_round_trips(quantiles):
    stats = {
        "actions": transforms.NormStats(
            mean=np.ones(20),
            std=np.full(20, 2.0),
            q01=np.full(20, -2.0),
            q99=np.full(20, 4.0),
        )
    }
    mask = ([True] * 9 + [False]) * 2
    echo_model = EchoModel()
    policy = G2Policy(
        NativePolicy(
            echo_model,
            is_pytorch=True,
            rtc_config=RTCConfig(),
            metadata={"action_dim": 20},
            transforms=[
                g2_transforms.G2Inputs(),
                transforms.DeltaActions(mask),
                transforms.Normalize(stats, use_quantiles=quantiles),
                transforms.PadStatesAndActions(32),
            ],
            output_transforms=[
                transforms.Unnormalize(stats, use_quantiles=quantiles),
                transforms.AbsoluteActions(mask),
                g2_transforms.G2Outputs(),
            ],
        ),
        _absolute_settings(),
    )
    obs = g2_transforms.make_g2_example()
    obs.pop("prompt")
    obs["observation/state"] = np.arange(20, dtype=np.float32) / 10
    absolute = np.full((6, 20), 3.0, dtype=np.float32)
    obs.update(rtc_enabled=True, prev_chunk_left_over=absolute, inference_delay=2)
    out = policy.infer(obs)
    assert policy.metadata["rtc_soft_overlap_steps"] == 5
    assert policy.metadata["rtc_prefix_weighting"] == "delay_plus_soft_overlap"
    assert out["rtc_info"]["soft_overlap_applied"] == 4
    assert out["rtc_info"]["soft_overlap_requested"] == 5
    expected = np.array([1, 1, 5 / 6, 4 / 6, 3 / 6, 2 / 6, 0], dtype=np.float32)
    np.testing.assert_allclose(echo_model.kwargs["rtc_weights"].cpu().numpy()[0, :7, 0], expected, atol=1e-7)
    np.testing.assert_allclose(out["actions"][:6], absolute, atol=1e-6)
    # New measured state must change the model-space prefix, not its absolute result.
    old_prefix = echo_model.kwargs["rtc_prefix"].clone()
    obs["observation/state"] += 0.5
    out = policy.infer(obs)
    np.testing.assert_allclose(out["actions"][:6], absolute, atol=1e-6)
    new_prefix = echo_model.kwargs["rtc_prefix"]
    assert not torch.equal(old_prefix[..., 0], new_prefix[..., 0])
    torch.testing.assert_close(old_prefix[..., 9], new_prefix[..., 9])
    weights = echo_model.kwargs["rtc_weights"].numpy()
    assert not weights[:, 6:].any()
    assert not weights[..., 20:].any()
    np.testing.assert_array_equal(absolute, 3)
    # First request has the same compiled shapes, but zero guidance.
    obs.pop("prev_chunk_left_over")
    policy.infer(obs)
    assert echo_model.kwargs["rtc_prefix"].shape == (1, 50, 32)
    assert not echo_model.kwargs["rtc_weights"].any()


def test_server_rejects_rtc_when_disabled():
    policy = NativePolicy(EchoModel(), is_pytorch=True)
    with pytest.raises(ValueError, match="not started"):
        policy.infer({"rtc_enabled": True})


@pytest.mark.parametrize("arm", ["both", "left", "right"])
@pytest.mark.parametrize("representation", ["component_delta", "se3_body_delta"])
@pytest.mark.parametrize("scope", ["chunk", "step"])
def test_joint_state_with_independent_pose_reference_round_trips_rtc(tmp_path, monkeypatch, arm, representation, scope):
    import copy

    from openpi.adapters.g2.config import POLICY_DEFAULTS
    from openpi.adapters.g2.data import LeRobotG2DataConfig
    from openpi.adapters.g2.tests.geometry_test import poses
    from openpi.adapters.g2.tests.geometry_test import set_encoding
    from openpi.adapters.g2.tests.geometry_test import set_state_space
    from openpi.training import config as native_config

    settings = copy.deepcopy(POLICY_DEFAULTS)
    set_state_space(settings, "joint")
    set_encoding(settings["action"]["pose"], representation, scope=scope)
    settings["action"]["pose"]["coordinates"]["rotation"] = "quaternion"
    data = LeRobotG2DataConfig(repo_id="test", policy=settings, arm=arm, cameras=("head_color",))
    monkeypatch.setattr(native_config.DataConfigFactory, "_load_norm_stats", lambda *args: None)
    monkeypatch.setattr(native_config.ModelTransformFactory, "__call__", lambda *args: transforms.Group())
    dc = data.create(tmp_path, SimpleNamespace(model_type=model.ModelType.PI05))
    dim, wire_dim = action_dimensions(settings, arm)
    echo = EchoModel()
    stats = {"actions": transforms.NormStats(mean=np.ones(dim), std=np.full(dim, 2.0))}
    native = NativePolicy(
        echo,
        is_pytorch=True,
        rtc_config=RTCConfig(),
        metadata={"arm": arm, "action_dim": wire_dim, "model_action_dim": dim},
        transforms=[*dc.data_transforms.inputs, transforms.Normalize(stats), transforms.PadStatesAndActions(32)],
        output_transforms=[transforms.Unnormalize(stats), *dc.data_transforms.outputs],
    )
    policy = G2Policy(native, settings)
    prefix = poses(3, 6)
    obs = {
        "observation/state": np.arange(16, dtype=np.float32),
        "observation/pose": poses(1),
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "rtc_enabled": True,
        "prev_chunk_left_over": prefix,
        "inference_delay": 2,
    }
    result = policy.infer(obs)["actions"]
    selection = slice(None) if arm == "both" else slice(0, 10) if arm == "left" else slice(10, 20)
    np.testing.assert_allclose(result[:6, selection], prefix[:, selection], atol=3e-5)
    before = echo.kwargs["rtc_prefix"].clone()
    obs["observation/pose"] = poses(2)
    result = policy.infer(obs)["actions"]
    np.testing.assert_allclose(result[:6, selection], prefix[:, selection], atol=3e-5)
    assert not torch.equal(before, echo.kwargs["rtc_prefix"])


@pytest.mark.parametrize("arm", ["both", "left", "right"])
@pytest.mark.parametrize(
    ("quantity", "representation", "scope"),
    [
        ("joint_position", "absolute", None),
        ("joint_position", "component_delta", "chunk"),
        ("joint_position", "component_delta", "step"),
        ("joint_velocity", "absolute", None),
        ("joint_torque", "absolute", None),
    ],
)
def test_rtc_other_control_quantities_keep_wire_units(tmp_path, monkeypatch, arm, quantity, representation, scope):
    import copy

    from openpi.adapters.g2.config import POLICY_DEFAULTS
    from openpi.adapters.g2.data import LeRobotG2DataConfig
    from openpi.adapters.g2.tests.geometry_test import set_joint_action
    from openpi.adapters.g2.tests.geometry_test import set_state_space
    from openpi.training import config as native_config

    settings = copy.deepcopy(POLICY_DEFAULTS)
    set_state_space(settings, "joint")
    set_joint_action(settings, quantity, representation, scope=scope)
    monkeypatch.setattr(native_config.DataConfigFactory, "_load_norm_stats", lambda *args: None)
    monkeypatch.setattr(native_config.ModelTransformFactory, "__call__", lambda *args: transforms.Group())
    dc = LeRobotG2DataConfig(repo_id="test", policy=settings, arm=arm, cameras=("head_color",)).create(
        tmp_path, SimpleNamespace(model_type=model.ModelType.PI05)
    )
    dim, wire_dim = action_dimensions(settings, arm)
    stats = {"actions": transforms.NormStats(mean=np.ones(dim), std=np.full(dim, 2.0))}
    policy = G2Policy(
        NativePolicy(
            EchoModel(),
            is_pytorch=True,
            rtc_config=RTCConfig(),
            metadata={"arm": arm, "action_dim": wire_dim, "model_action_dim": dim},
            transforms=[*dc.data_transforms.inputs, transforms.Normalize(stats), transforms.PadStatesAndActions(32)],
            output_transforms=[transforms.Unnormalize(stats), *dc.data_transforms.outputs],
        ),
        settings,
    )
    prefix = np.arange(6 * wire_dim, dtype=np.float32).reshape(6, wire_dim) / 10
    obs = {
        "observation/state": np.arange(16, dtype=np.float32),
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "rtc_enabled": True,
        "prev_chunk_left_over": prefix,
        "inference_delay": 2,
    }
    output = policy.infer(obs)["actions"]
    chosen = (
        slice(None) if arm == "both" else slice(0, wire_dim // 2) if arm == "left" else slice(wire_dim // 2, wire_dim)
    )
    np.testing.assert_allclose(output[:6, chosen], prefix[:, chosen], atol=1e-5)
    assert output.shape == (50, wire_dim)
