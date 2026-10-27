"""G2 presets, separate from the native OpenPI configuration registry."""

import dataclasses

import flax.nnx as nnx

from openpi.adapters.g2.data import LeRobotG2DataConfig
from openpi.adapters.g2.transforms import G2_ACTION_DIM
from openpi.models import pi0_config
from openpi.shared import nnx_utils
from openpi.training import weight_loaders
from openpi.training.config import DataConfig
from openpi.training.config import TrainConfig

# Versioned fallback for checkpoints without a training snapshot. Current YAML
# experiments replace repo_id in build_train_config; importing presets needs no file I/O.
_LEGACY_REPO_ID = "vr/Sing_TeaPi/lerobot_v21"
_BASE = TrainConfig(
    name="pi05_g2_vr",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
    data=LeRobotG2DataConfig(
        repo_id=_LEGACY_REPO_ID,
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=30_000,
    batch_size=8,
    wandb_enabled=True,
    project_name="G2_openpi",
    policy_metadata={
        "backend": "openpi",
        "robot": "g2",
        "action_dim": G2_ACTION_DIM,
        "action_horizon": 50,
        "policy_hz": 30.0,
    },
)
_LOW_MEM_MODEL = dataclasses.replace(
    _BASE.model,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
)
_VLM_LORA_ACTION_FULL_MODEL = dataclasses.replace(
    _BASE.model,
    paligemma_variant="gemma_2b_lora",
    vision_lora_rank=16,
    vision_lora_alpha=16.0,
)
_CONFIGS = [
    _BASE,
    dataclasses.replace(
        _BASE,
        name="pi05_g2_vr_low_mem",
        model=_LOW_MEM_MODEL,
        freeze_filter=_LOW_MEM_MODEL.get_freeze_filter(),
        ema_decay=None,
        batch_size=2,
    ),
    dataclasses.replace(
        _BASE,
        name="pi05_g2_vr_vlm_lora_action_full",
        model=_VLM_LORA_ACTION_FULL_MODEL,
        # Also preserves the original frozen vision backbone when old snapshots
        # restore this preset without visual adapters.
        freeze_filter=nnx.Any(
            _VLM_LORA_ACTION_FULL_MODEL.get_freeze_filter(),
            nnx.All(nnx_utils.PathRegex("PaliGemma/img/.*"), nnx.Not(nnx_utils.PathRegex(".*lora.*"))),
        ),
        ema_decay=None,
        batch_size=2,
    ),
]

# Separate names isolate norm assets and checkpoints between arm selections.
_CONFIGS.extend(
    [
        dataclasses.replace(
            config,
            name=f"{config.name}_{arm}",
            data=dataclasses.replace(config.data, arm=arm),
        )
        for config in tuple(_CONFIGS)
        for arm in ("left", "right")
    ]
)

_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def finetuning_filter(model: pi0_config.Pi0Config, modes: dict):
    """Freeze explicit module groups; LoRA modes keep their adapters trainable."""
    filters = [model.get_freeze_filter()]
    if modes["vision"] == "frozen":
        filters.append(nnx_utils.PathRegex("PaliGemma/img/.*"))
    if modes["language"] == "frozen":
        filters.append(nnx.All(nnx_utils.PathRegex("PaliGemma/llm/.*"), nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*"))))
    if modes["action_expert"] == "frozen":
        filters.extend(
            (
                nnx_utils.PathRegex(".*llm.*_1.*"),
                nnx_utils.PathRegex("(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)/.*"),
            )
        )
    return nnx.Any(*filters)


def get_config(name: str) -> TrainConfig:
    try:
        return _CONFIGS_DICT[name]
    except KeyError:
        raise ValueError(f"Unknown G2 preset {name!r}; available: {sorted(_CONFIGS_DICT)}") from None
