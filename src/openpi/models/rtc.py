"""Inference-time RTC for OpenPI's noise-at-1, actions-at-0 flow convention.

Pseudoinverse guidance follows Physical-Intelligence/real-time-chunking-kinetix.
The prefix is supplied in *model action space*, with fixed shapes and explicit
weights so changing leftover lengths does not trigger JAX recompilation.
"""

from collections.abc import Callable
import dataclasses

import numpy as np

PROTOCOL = "openpi.rtc.v1"


@dataclasses.dataclass(frozen=True)
class RTCConfig:
    execution_horizon: int = 10
    max_guidance_weight: float = 10.0

    def __post_init__(self):
        if (
            isinstance(self.execution_horizon, bool)
            or not isinstance(self.execution_horizon, (int, np.integer))
            or self.execution_horizon <= 0
        ):
            raise ValueError("RTC execution_horizon must be positive")
        if not np.isfinite(self.max_guidance_weight) or self.max_guidance_weight <= 0:
            raise ValueError("RTC max_guidance_weight must be finite and positive")


def prepare_prefix(
    actions: np.ndarray | None,
    *,
    action_horizon: int,
    action_dim: int,
    wire_action_dim: int,
    inference_delay: int,
    config: RTCConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad already transformed actions; never guide padded time or dimensions."""
    if isinstance(inference_delay, bool) or not isinstance(inference_delay, (int, np.integer)) or inference_delay < 0:
        raise ValueError("RTC inference_delay must be a nonnegative integer (policy steps)")
    prefix = np.zeros((action_horizon, action_dim), dtype=np.float32)
    weights = np.zeros_like(prefix)
    if actions is None:
        return prefix, weights
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != action_dim or not np.isfinite(actions).all():
        raise ValueError("RTC transformed actions must be finite (T, model_action_dim)")
    valid = min(len(actions), action_horizon)
    if inference_delay >= action_horizon or inference_delay > valid:
        raise ValueError("RTC delay exceeds the available prefix/prediction horizon")
    prefix[:valid] = actions[:valid]
    end = min(valid, max(config.execution_horizon, inference_delay))
    # Keep committed steps at full weight, then decay over the soft overlap.
    temporal = np.zeros(action_horizon, dtype=np.float32)
    temporal[:inference_delay] = 1.0
    if end > inference_delay:
        temporal[inference_delay:end] = np.linspace(1, 0, end - inference_delay + 2)[1:-1]
    weights[:, :wire_action_dim] = temporal[:, None]
    return prefix, weights


def guided_velocity_jax(velocity_fn: Callable, x_t, time, prefix, weights, max_guidance_weight):
    """VJP of the endpoint estimate, including the velocity's input Jacobian."""
    import jax  # noqa: PLC0415 -- keep the NumPy wire helpers usable without JAX
    import jax.numpy as jnp  # noqa: PLC0415

    def denoiser(x):
        velocity = velocity_fn(x)
        return x - time * velocity, velocity

    endpoint, pullback, velocity = jax.vjp(denoiser, x_t, has_aux=True)
    error = (prefix - endpoint) * weights
    correction = pullback(error)[0]
    # tau=1-time is the paper's noise-to-actions time. Clamp endpoint divisions.
    time = jnp.clip(time, 1e-6, 1.0)
    tau = jnp.maximum(1.0 - time, 1e-6)
    gain = jnp.minimum((time**2 + tau**2) / (time * tau), max_guidance_weight)
    return velocity - gain * correction


def guided_velocity_torch(velocity_fn: Callable, x_t, time, prefix, weights, max_guidance_weight):
    import torch  # noqa: PLC0415 -- PyTorch is optional for the JAX guidance helper

    # sample_actions uses no_grad; enable gradients only for the action expert.
    with torch.enable_grad():
        x = x_t.detach().requires_grad_()
        velocity = velocity_fn(x)
        endpoint = x - time * velocity
        error = ((prefix - endpoint) * weights).detach()
        correction = torch.autograd.grad(endpoint, x, grad_outputs=error)[0]
    time = time.clamp(1e-6, 1.0)
    tau = (1.0 - time).clamp_min(1e-6)
    gain = ((time.square() + tau.square()) / (time * tau)).clamp_max(max_guidance_weight)
    return (velocity - gain * correction).detach()
