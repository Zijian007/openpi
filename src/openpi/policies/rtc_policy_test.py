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
