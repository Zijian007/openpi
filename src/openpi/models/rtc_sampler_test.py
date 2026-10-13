"""Exercise the real JAX Pi0 sampler with a small differentiable action expert."""

import numpy as np
import pytest

pytest.importorskip("flax")

from flax import nnx
import flax.linen as nn
from flax.nnx import bridge
import jax
import jax.numpy as jnp

from openpi.models.model import Observation
from openpi.models.pi0 import Pi0
from openpi.shared.nnx_utils import module_jit


class ToyLLM(nn.Module):
    @nn.compact
    def __call__(self, tokens, **kwargs):
        scale = self.param("scale", nn.initializers.constant(0.1), (2,))
        prefix, suffix = tokens
        return (prefix, None if suffix is None else (suffix * scale).astype(jnp.bfloat16)), None


class ToyPi0(nnx.Module):
    sample_actions = Pi0.sample_actions

    def __init__(self):
        self.action_dim = 2
        self.action_horizon = 8
        llm = bridge.ToNNX(ToyLLM())
        llm.lazy_init([jnp.zeros((1, 1, 2)), None], rngs=nnx.Rngs(0))
        self.PaliGemma = nnx.Dict(llm=llm)
        self.action_out_proj = nnx.Linear(2, 2, rngs=nnx.Rngs(1))
        self.action_out_proj.kernel.value = self.action_out_proj.kernel.value.astype(jnp.bfloat16)
        self.action_out_proj.bias.value = self.action_out_proj.bias.value.astype(jnp.bfloat16)

    def embed_prefix(self, observation):
        batch = observation.state.shape[0]
        return jnp.zeros((batch, 1, 2)), jnp.ones((batch, 1), dtype=bool), jnp.zeros(1, dtype=bool)

    def embed_suffix(self, observation, actions, time):
        batch, steps, _ = actions.shape
        return actions, jnp.ones((batch, steps), dtype=bool), jnp.zeros(steps, dtype=bool), None


def test_jitted_sampler_zero_mask_parity_and_guided_prefix():
    model = ToyPi0()
    sample = module_jit(model.sample_actions)
    key = jax.random.key(0)
    obs = Observation(
        images={name: jnp.zeros((1, 224, 224, 3)) for name in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
        image_masks={name: jnp.ones(1, dtype=bool) for name in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
        state=jnp.zeros((1, 2)),
    )
    noise = jax.random.normal(key, (1, 8, 2))
    prefix = jnp.ones_like(noise)
    zero_weights = jnp.zeros_like(noise)
    plain = sample(key, obs, noise=noise)
    no_prefix = sample(key, obs, noise=noise, rtc_prefix=prefix, rtc_weights=zero_weights)
    np.testing.assert_allclose(no_prefix, plain, atol=1e-6)
    weights = zero_weights.at[:, :3].set(1)
    guided = sample(key, obs, noise=noise, rtc_prefix=prefix, rtc_weights=weights)
    assert np.linalg.norm(np.asarray(guided[:, :3]) - 1) < np.linalg.norm(np.asarray(plain[:, :3]) - 1)
    np.testing.assert_allclose(guided[:, 3:], plain[:, 3:], atol=1e-6)
    changed = sample(key, obs, noise=noise, rtc_prefix=-prefix, rtc_weights=weights)
    assert not np.allclose(changed[:, :3], guided[:, :3])
