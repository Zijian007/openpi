import numpy as np
import pytest
import torch

from openpi.models import rtc


def test_prefix_keeps_committed_steps_and_masks_padding():
    prefix, weights = rtc.prepare_prefix(
        np.ones((5, 32)),
        action_horizon=50,
        action_dim=32,
        wire_action_dim=20,
        inference_delay=2,
        config=rtc.RTCConfig(soft_overlap_steps=2),
    )
    np.testing.assert_allclose(weights[:2, :20], 1)
    np.testing.assert_allclose(weights[2:4, 0], [2 / 3, 1 / 3])
    assert not weights[4:].any()
    assert not weights[:, 20:].any()
    assert not prefix[5:].any()


@pytest.mark.parametrize("delay", [-1, 1.5, True, 6])
def test_invalid_delay_is_rejected(delay):
    with pytest.raises(ValueError, match="RTC"):
        rtc.prepare_prefix(
            np.ones((5, 32)),
            action_horizon=50,
            action_dim=32,
            wire_action_dim=20,
            inference_delay=delay,
            config=rtc.RTCConfig(),
        )


@pytest.mark.parametrize("delay", [0, 7, 13, 18])
def test_soft_overlap_width_is_independent_of_inference_delay(delay):
    _, weights = rtc.prepare_prefix(
        np.ones((30, 32)),
        action_horizon=50,
        action_dim=32,
        wire_action_dim=20,
        inference_delay=delay,
        config=rtc.RTCConfig(soft_overlap_steps=5),
    )
    np.testing.assert_allclose(weights[:delay, :20], 1)
    np.testing.assert_allclose(weights[delay:delay + 5, 0], [5/6, 4/6, 3/6, 2/6, 1/6], atol=1e-7)
    assert not weights[delay + 5:].any()
    assert not weights[:, 20:].any()


def test_short_prefix_masks_missing_steps_without_changing_decay_slope():
    _, weights = rtc.prepare_prefix(np.ones((15, 32)), action_horizon=50, action_dim=32,
                                   wire_action_dim=20, inference_delay=13,
                                   config=rtc.RTCConfig(soft_overlap_steps=5))
    np.testing.assert_allclose(weights[13:15, 0], [5/6, 4/6], atol=1e-7)
    assert not weights[15:].any()


def test_zero_soft_overlap_is_explicitly_supported():
    _, weights = rtc.prepare_prefix(np.ones((20, 32)), action_horizon=50, action_dim=32,
                                   wire_action_dim=20, inference_delay=13,
                                   config=rtc.RTCConfig(soft_overlap_steps=0))
    np.testing.assert_array_equal(weights[:13, :20], 1)
    assert not weights[13:].any()


@pytest.mark.parametrize("steps", [-1, 1.5, True])
def test_invalid_soft_overlap_is_rejected(steps):
    with pytest.raises(ValueError, match="soft_overlap_steps"):
        rtc.RTCConfig(soft_overlap_steps=steps)


def _guidance_example():
    # Coupled velocity makes a masked prefix influence another action via VJP.
    matrix = np.array([[0.2, 0.3], [0.1, -0.4]], dtype=np.float32)
    x = np.array([[[0.5, -0.2]]], dtype=np.float32)
    prefix = np.array([[[1.0, 0.7]]], dtype=np.float32)
    weights = np.array([[[1.0, 0.0]]], dtype=np.float32)
    time = 0.4
    velocity = x @ matrix.T
    jacobian = np.eye(2) - time * matrix
    endpoint = x - time * velocity
    error = (prefix - endpoint) * weights
    gain = (time**2 + (1 - time) ** 2) / (time * (1 - time))
    expected = velocity - gain * (error @ jacobian)
    return matrix, x, prefix, weights, time, expected


def test_torch_guidance_includes_velocity_jacobian():
    matrix, x, prefix, weights, time, expected = _guidance_example()
    with torch.no_grad():
        torch_velocity = rtc.guided_velocity_torch(
            lambda v: v @ torch.from_numpy(matrix).T,
            torch.from_numpy(x),
            torch.tensor(time),
            torch.from_numpy(prefix),
            torch.from_numpy(weights),
            10.0,
        )
    np.testing.assert_allclose(torch_velocity.numpy(), expected, atol=1e-6)


def test_jax_guidance_includes_velocity_jacobian():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    matrix, x, prefix, weights, time, expected = _guidance_example()
    result = jax.jit(
        lambda value: rtc.guided_velocity_jax(
            lambda v: v @ jnp.asarray(matrix).T,
            value,
            jnp.asarray(time),
            jnp.asarray(prefix),
            jnp.asarray(weights),
            10.0,
        )
    )(jnp.asarray(x))
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("time", [0.0, 1.0])
def test_guidance_endpoint_times_are_finite(time):
    jnp = pytest.importorskip("jax.numpy")
    x = jnp.ones((1, 3, 2))
    result = rtc.guided_velocity_jax(lambda v: v * 0.1, x, time, x * 2, x, 10.0)
    assert np.isfinite(result).all()


def test_guidance_improves_prefix_without_touching_unmasked_steps():
    jnp = pytest.importorskip("jax.numpy")
    x = jnp.zeros((1, 10, 2))
    prefix = jnp.ones_like(x)
    weights = jnp.zeros_like(x).at[:, :3].set(1)
    for i in range(10):
        velocity = rtc.guided_velocity_jax(jnp.zeros_like, x, 1 - i / 10, prefix, weights, 5.0)
        x = x - 0.1 * velocity
    assert np.max(np.abs(np.asarray(x[:, :3]) - 1)) < 0.05
    np.testing.assert_array_equal(x[:, 3:], 0)


def test_torch_guidance_improves_prefix_without_touching_unmasked_steps():
    x = torch.zeros((1, 10, 2))
    prefix = torch.ones_like(x)
    weights = torch.zeros_like(x)
    weights[:, :3] = 1
    with torch.no_grad():
        for i in range(10):
            velocity = rtc.guided_velocity_torch(
                torch.zeros_like,
                x,
                torch.tensor(1 - i / 10),
                prefix,
                weights,
                5.0,
            )
            x = x - 0.1 * velocity
    assert torch.max(torch.abs(x[:, :3] - 1)) < 0.05
    torch.testing.assert_close(x[:, 3:], torch.zeros_like(x[:, 3:]))


@pytest.mark.parametrize("time", [0.0, 1.0])
def test_torch_endpoint_times_are_finite(time):
    x = torch.ones((1, 3, 2))
    result = rtc.guided_velocity_torch(lambda v: v * 0.1, x, torch.tensor(time), x * 2, x, 10.0)
    assert torch.isfinite(result).all()


def test_torch_zero_mask_preserves_ordinary_velocity():
    x = torch.ones((1, 3, 2))
    result = rtc.guided_velocity_torch(lambda v: v * 0.1, x, torch.tensor(0.4), x * 2, x * 0, 10.0)
    torch.testing.assert_close(result, x * 0.1)
