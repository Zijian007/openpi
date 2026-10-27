"""G2 presets, separate from the native OpenPI configuration registry."""

import dataclasses

from openpi.adapters.g2 import config as g2_config
from openpi.adapters.g2.transforms import G2_ACTION_DIM
from openpi.adapters.g2.data import LeRobotG2DataConfig
from openpi.models import pi0_config
from openpi.training import weight_loaders
from openpi.training.config import DataConfig
from openpi.training.config import TrainConfig

_REPO_ID = g2_config.read_yaml(g2_config.DEFAULT_CONFIG)["data"]["repo_id"]

_CONFIGS = [
    #
    # G2 wholebody (VR pose). Dataset id comes from scripts/g2/train/config.yaml.
    #
    TrainConfig(
        name="pi05_g2_vr",
        # Pi05 defaults to encoding the measured state in the prompt.
        # 50 action steps at 30 Hz cover approximately 1.7 seconds.
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=LeRobotG2DataConfig(
            repo_id=_REPO_ID,
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        # Single 4090: start small; raise if VRAM allows.
        batch_size=8,
        wandb_enabled=True,
        project_name="G2_openpi",
        # Handshake fields, plus action_dim for in-process RTC. G2PolicyServer drops action_dim.
        policy_metadata={
            "backend": "openpi",
            "robot": "g2",
            "action_dim": G2_ACTION_DIM,
            "action_horizon": 50,
            "policy_hz": 30.0,
        },
    ),
    TrainConfig(
        name="pi05_g2_vr_low_mem",
        # LoRA variants for smoke / 24GB cards (mirrors pi0_libero_low_mem_finetune pattern).
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotG2DataConfig(
            repo_id=_REPO_ID,
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=30_000,
        batch_size=2,
        wandb_enabled=True,
        project_name="G2_openpi",
        policy_metadata={
            "backend": "openpi",
            "robot": "g2",
            "action_dim": G2_ACTION_DIM,
            "action_horizon": 50,
            "policy_hz": 30.0,
        },
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
        if config.name in ("pi05_g2_vr", "pi05_g2_vr_low_mem")
        for arm in ("left", "right")
    ]
)

_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def get_config(name: str) -> TrainConfig:
    try:
        return _CONFIGS_DICT[name]
    except KeyError:
        raise ValueError(f"Unknown G2 preset {name!r}; available: {sorted(_CONFIGS_DICT)}") from None
