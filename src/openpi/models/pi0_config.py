import dataclasses
import math
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import lora as _lora
from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class LoRAOptions:
    """Serializable adapter settings; initialization details stay in the numerical module."""

    rank: int
    alpha: float
    targets: tuple[str, ...] = ("attention", "ffn")
    rslora: bool = False

    def __post_init__(self):
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")
        if type(self.alpha) not in (int, float) or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("LoRA alpha must be a positive finite number")
        if type(self.rslora) is not bool:
            raise ValueError("LoRA rslora must be a boolean")
        if (
            not isinstance(self.targets, tuple)
            or not self.targets
            or any(target not in ("attention", "ffn") for target in self.targets)
            or len(set(self.targets)) != len(self.targets)
        ):
            raise ValueError("LoRA targets must be a nonempty tuple of unique attention/ffn targets")

    def numerical_config(self) -> _lora.LoRAConfig:
        return _lora.LoRAConfig(rank=self.rank, alpha=self.alpha, rslora=self.rslora)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    # Explicit adapter settings override the Gemma variant's default adapters when supplied.
    vision_lora: LoRAOptions | None = None
    language_lora: LoRAOptions | None = None
    action_expert_lora: LoRAOptions | None = None

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.vision_lora is not None and self.vision_lora.targets != ("attention",):
            raise ValueError("Vision LoRA only supports the attention target (Q/K/V/O)")
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    def gemma_configs(self) -> tuple[_gemma.Config, _gemma.Config]:
        """Resolve text/expert adapters once, preserving native variants when not overridden."""
        configs = []
        for variant, options in (
            (self.paligemma_variant, self.language_lora),
            (self.action_expert_variant, self.action_expert_lora),
        ):
            config = _gemma.get_config(variant)
            if options is not None:
                targets = {"attention": "attn", "ffn": "ffn"}
                config = dataclasses.replace(
                    config, lora_configs={targets[target]: options.numerical_config() for target in options.targets}
                )
            configs.append(config)
        return tuple(configs)

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        language_lora = self.language_lora is not None or "lora" in self.paligemma_variant
        expert_lora = self.action_expert_lora is not None or "lora" in self.action_expert_variant
        if language_lora:
            filters.append(
                gemma_params_filter,
            )
            if not expert_lora:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif expert_lora:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        freeze_filter = nnx.All(*filters) if filters else nnx.Nothing
        if self.vision_lora is not None:
            freeze_filter = nnx.Any(
                freeze_filter,
                nnx.All(nnx_utils.PathRegex("PaliGemma/img/.*"), nnx.Not(nnx_utils.PathRegex(".*lora.*"))),
            )
        return freeze_filter
