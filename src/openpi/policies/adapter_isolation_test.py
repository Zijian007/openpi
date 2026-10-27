from types import SimpleNamespace

import numpy as np
import torch

from openpi.policies.policy import Policy


def test_native_inference_does_not_dispatch_on_robot_metadata():
    class Model(torch.nn.Module):
        config = SimpleNamespace(action_horizon=3, action_dim=10)

        def sample_actions(self, device, observation, **kwargs):
            return torch.zeros((1, 3, 10), device=device)

    policy = Policy(Model(), is_pytorch=True, metadata={"robot": "g2", "arm": "right"})
    result = policy.infer({"state": np.zeros(8, np.float32), "image": {}, "image_mask": {}})
    assert result["actions"].shape == (3, 10)
