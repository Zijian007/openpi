from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("flax")

from openpi import transforms
from openpi.models import model
from openpi.models.rtc import RTCConfig
from openpi.policies import g2_policy
from openpi.policies.policy import Policy


class EchoModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(action_horizon=50, action_dim=32)
        self.kwargs = None

    def sample_actions(self, device, observation, **kwargs):
        self.kwargs = kwargs
        return kwargs["rtc_prefix"]


@pytest.mark.parametrize("arm,start", [("left", 0), ("right", 10)])
def test_single_arm_rtc_round_trip_keeps_correct_side_and_ignores_inactive_prefix(arm, start):
    echo_model = EchoModel()
    stats = {"actions": transforms.NormStats(mean=np.ones(10), std=np.full(10, 2.0))}
    policy = Policy(
        echo_model,
        is_pytorch=True,
        rtc_config=RTCConfig(),
        metadata={"robot": "g2", "arm": arm, "action_dim": 20, "model_action_dim": 10},
        transforms=[
            g2_policy.G2Inputs(model.ModelType.PI05, arm=arm),
            transforms.DeltaActions([True] * 9 + [False]),
            transforms.Normalize(stats),
            transforms.PadStatesAndActions(32),
        ],
        output_transforms=[
            transforms.Unnormalize(stats),
            transforms.AbsoluteActions([True] * 9 + [False]),
            g2_policy.G2Outputs(action_dim=10),
        ],
    )
    obs = g2_policy.make_g2_example()
    inactive = "left" if arm == "right" else "right"
    obs.pop(f"observation/wrist_image_{inactive}")
    obs["observation/state"] = np.arange(20, dtype=np.float32) / 10
    prefix = np.full((6, 20), 3.0, dtype=np.float32)
    obs.update(rtc_enabled=True, prev_chunk_left_over=prefix, inference_delay=2)
    result = policy.infer(obs)["actions"]
    np.testing.assert_allclose(result[:6, start:start + 10], 3.0, atol=1e-6)
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
    policy = Policy(
        echo_model,
        is_pytorch=True,
        rtc_config=RTCConfig(),
        metadata={"action_dim": 20},
        transforms=[
            g2_policy.G2Inputs(model.ModelType.PI05),
            transforms.DeltaActions(mask),
            transforms.Normalize(stats, use_quantiles=quantiles),
            transforms.PadStatesAndActions(32),
        ],
        output_transforms=[
            transforms.Unnormalize(stats, use_quantiles=quantiles),
            transforms.AbsoluteActions(mask),
            g2_policy.G2Outputs(),
        ],
    )
    obs = g2_policy.make_g2_example()
    obs["observation/state"] = np.arange(20, dtype=np.float32) / 10
    absolute = np.full((6, 20), 3.0, dtype=np.float32)
    obs.update(rtc_enabled=True, prev_chunk_left_over=absolute, inference_delay=2)
    out = policy.infer(obs)
    assert policy.metadata["rtc_soft_overlap_steps"] == 5
    assert policy.metadata["rtc_prefix_weighting"] == "delay_plus_soft_overlap"
    assert out["rtc_info"]["soft_overlap_applied"] == 4
    assert out["rtc_info"]["soft_overlap_requested"] == 5
    expected = np.array([1, 1, 5/6, 4/6, 3/6, 2/6, 0], dtype=np.float32)
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
    policy = Policy(EchoModel(), is_pytorch=True)
    with pytest.raises(ValueError, match="not started"):
        policy.infer({"rtc_enabled": True})
