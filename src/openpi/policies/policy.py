from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import rtc as _rtc
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        rtc_config: _rtc.RTCConfig | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = dict(metadata or {})
        self._rtc_config = rtc_config
        self._metadata.update(rtc_supported=rtc_config is not None)
        if rtc_config is not None:
            self._metadata.update(
                rtc_protocol=_rtc.PROTOCOL,
                rtc_soft_overlap_steps=rtc_config.soft_overlap_steps,
                rtc_prefix_weighting="delay_plus_soft_overlap",
                rtc_max_guidance_weight=rtc_config.max_guidance_weight,
            )
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        shape_config = model.config if is_pytorch else model
        self._action_dim = shape_config.action_dim
        self._action_horizon = shape_config.action_horizon

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        rtc_enabled = inputs.pop("rtc_enabled", False)
        if not isinstance(rtc_enabled, (bool, np.bool_)):
            raise ValueError("rtc_enabled must be boolean")
        leftover = inputs.pop("prev_chunk_left_over", None)
        inference_delay = inputs.pop("inference_delay", 0)
        if rtc_enabled and self._rtc_config is None:
            raise ValueError("RTC requested but this policy was not started with RTC enabled")
        wire_action_dim = 0
        if rtc_enabled and leftover is not None:
            leftover = np.asarray(leftover, dtype=np.float32)
            expected_dim = self._metadata.get("action_dim", self._action_dim)
            if (
                leftover.ndim != 2
                or len(leftover) == 0
                or leftover.shape[1] != expected_dim
                or not np.isfinite(leftover).all()
            ):
                raise ValueError("RTC leftover must be finite absolute actions (T, wire_action_dim)")
            wire_action_dim = leftover.shape[1]
            # The SAME training input pipeline reanchors DeltaActions to this
            # observation, normalizes using checkpoint stats, and pads dimensions.
            inputs["actions"] = leftover.copy()
        inputs = self._input_transform(inputs)
        transformed_prefix = inputs.pop("actions", None)
        sample_kwargs = dict(self._sample_kwargs)
        rtc_info = None
        if self._rtc_config is not None:
            prefix, weights = _rtc.prepare_prefix(
                transformed_prefix if rtc_enabled else None,
                action_horizon=self._action_horizon,
                action_dim=self._action_dim,
                wire_action_dim=wire_action_dim,
                inference_delay=inference_delay if rtc_enabled else 0,
                config=self._rtc_config,
            )
            valid = 0 if transformed_prefix is None or not rtc_enabled else min(len(transformed_prefix), self._action_horizon)
            guided_delay = int(inference_delay) if valid else 0
            rtc_info = {
                "delay_steps": guided_delay,
                "prefix_steps": valid,
                "soft_overlap_requested": self._rtc_config.soft_overlap_steps,
                "soft_overlap_applied": max(0, min(self._rtc_config.soft_overlap_steps, valid - guided_delay)),
            }
            # Fixed shapes even for the first chunk; warmup compiles the guided branch.
            if self._is_pytorch_model:
                prefix = torch.from_numpy(prefix).to(self._pytorch_device)[None, ...]
                weights = torch.from_numpy(weights).to(self._pytorch_device)[None, ...]
            else:
                prefix, weights = jnp.asarray(prefix)[None, ...], jnp.asarray(weights)[None, ...]
            sample_kwargs.update(
                rtc_prefix=prefix,
                rtc_weights=weights,
                rtc_max_guidance_weight=self._rtc_config.max_guidance_weight,
            )
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if rtc_info is not None:
            outputs["rtc_info"] = rtc_info
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
