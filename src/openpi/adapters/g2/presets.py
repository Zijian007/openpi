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

# Base model template; explicit YAML module modes determine the actual training model.
_DEFAULT_REPO_ID = "vr/Sing_TeaPi/lerobot_v21"
_BASE = TrainConfig(
    name="pi05_g2_vr",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
    data=LeRobotG2DataConfig(
        repo_id=_DEFAULT_REPO_ID,
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


def base_config(arm: str = "both") -> TrainConfig:
    """Build the single model template for the selected arms."""
    if arm not in ("both", "left", "right"):
        raise ValueError("arm must be both, left or right")
    return dataclasses.replace(
        _BASE,
        name=_BASE.name if arm == "both" else f"{_BASE.name}_{arm}",
        data=dataclasses.replace(_BASE.data, arm=arm),
    )
