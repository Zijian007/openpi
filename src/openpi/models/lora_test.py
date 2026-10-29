import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

import openpi.models.lora as lora


def test_lora_einsum_params_shape():
    shape = (3, 8, 32, 4)  # (3KDH)
    einsum = lora.Einsum(shape)
    lora0 = lora.Einsum(shape, lora_config=lora.LoRAConfig(rank=2))
    lora1 = lora.Einsum(shape, lora_config=lora.LoRAConfig(rank=2, axes=(1, 2)))

    key = jax.random.key(0)
    x = jax.random.normal(key, (8, 64, 32))  # (BSD)
    eqn = "BSD,3KDH->3BSKH"

    # Ensure that lora parameters are not initialized when LoRA is not used.
    params = einsum.init(key, eqn, x)
    assert "lora_a" not in params["params"]
    assert "lora_b" not in params["params"]

    # Check that default axes work.
    params_lora0 = lora0.init(key, eqn, x)
    assert params_lora0["params"]["lora_a"].shape == (3, 8, 32, 2)
    assert params_lora0["params"]["lora_b"].shape == (3, 8, 2, 4)

    # Check that user provided axes work.
    params_lora1 = lora1.init(key, eqn, x)
    assert params_lora1["params"]["lora_a"].shape == (3, 8, 2, 4)
    assert params_lora1["params"]["lora_b"].shape == (3, 2, 32, 4)


def test_lora_einsum_same_output():
    shape = (3, 8, 32, 4)  # (3KDH)
    einsum = lora.Einsum(shape)
    einsum_lora = lora.Einsum(shape, lora_config=lora.LoRAConfig(rank=2, init_fn=nn.initializers.zeros))

    key = jax.random.key(0)
    x = jax.random.normal(key, (8, 64, 32))  # (BSD)
    eqn = "BSD,3KDH->3BSKH"

    params = einsum.init(key, eqn, x)
    output = einsum.apply(params, eqn, x)

    params_lora = einsum_lora.init(key, eqn, x)
    output_lora = einsum_lora.apply(params_lora, eqn, x)

    # Results are the same since the LoRA parameters are initialized to zeros.
    assert jnp.allclose(output, output_lora)


def test_lora_ffn_params_shape():
    ffn = lora.FeedForward(features=8, hidden_dim=32)
    ffn_lora = lora.FeedForward(
        features=8,
        hidden_dim=32,
        lora_config=lora.LoRAConfig(rank=2),
    )

    key = jax.random.key(0)
    x = jax.random.normal(key, (2, 8))

    params = ffn.init(key, x)
    assert params["params"]["gating_einsum"].shape == (2, 8, 32)
    assert params["params"]["linear"].shape == (32, 8)

    params_lora = ffn_lora.init(key, x)
    assert params_lora["params"]["gating_einsum"].shape == (2, 8, 32)
    assert params_lora["params"]["linear"].shape == (32, 8)
    assert params_lora["params"]["gating_einsum_lora_a"].shape == (2, 8, 2)
    assert params_lora["params"]["gating_einsum_lora_b"].shape == (2, 2, 32)
    assert params_lora["params"]["linear_lora_a"].shape == (32, 2)
    assert params_lora["params"]["linear_lora_b"].shape == (2, 8)


def test_lora_ffn_same_output():
    ffn = lora.FeedForward(features=8, hidden_dim=32)
    ffn_lora = lora.FeedForward(
        features=8,
        hidden_dim=32,
        lora_config=lora.LoRAConfig(rank=2, init_fn=nn.initializers.zeros),
    )

    key = jax.random.key(0)
    x = jax.random.normal(key, (2, 8))

    params = ffn.init(key, x)
    output = ffn.apply(params, x)

    params_lora = ffn_lora.init(key, x)
    output_lora = ffn_lora.apply(params_lora, x)

    assert jnp.allclose(output, output_lora)


@pytest.mark.parametrize("rslora", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_ffn_applies_alpha_and_rank_stabilized_scaling(rslora, dtype):
    options = lora.LoRAConfig(rank=2, alpha=6.0, rslora=rslora)
    ffn = lora.FeedForward(features=2, hidden_dim=3, lora_config=options)
    x = jnp.array([[0.2, -0.1], [0.4, 0.3]], dtype=dtype)
    variables = ffn.init(jax.random.key(0), x)
    params = {key: jnp.full_like(value, 0.25) if "lora" in key else value for key, value in variables["params"].items()}
    weights = {key: value.astype(dtype) for key, value in params.items()}
    scale = jnp.asarray(6.0 / (2.0**0.5 if rslora else 2.0), dtype=dtype)
    gate = nn.gelu(
        x @ weights["gating_einsum"][0]
        + (x @ weights["gating_einsum_lora_a"][0] @ weights["gating_einsum_lora_b"][0]) * scale
    )
    up = (
        x @ weights["gating_einsum"][1]
        + (x @ weights["gating_einsum_lora_a"][1] @ weights["gating_einsum_lora_b"][1]) * scale
    )
    hidden = gate * up
    expected = hidden @ weights["linear"] + (hidden @ weights["linear_lora_a"] @ weights["linear_lora_b"]) * scale
    actual = ffn.apply({"params": params}, x)
    assert actual.dtype == dtype
    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)
