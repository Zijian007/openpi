"""Build synthetic wire inputs without connecting to a server or loading weights."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy'))
import smoke_infer_client


@pytest.mark.parametrize('zeros', [False, True])
def test_make_observation_supports_default_and_zero_smoke(zeros):
    args = smoke_infer_client.Args(zeros=zeros, prompt='test')
    obs = smoke_infer_client._make_obs(args)
    assert obs['observation/state'].shape == (20,)
    assert np.isfinite(obs['observation/state']).all()
    assert obs['prompt'] == 'test'
    for key in ('observation/image', 'observation/wrist_image_left', 'observation/wrist_image_right'):
        assert obs[key].shape == (224, 224, 3)
        assert obs[key].dtype == np.uint8
