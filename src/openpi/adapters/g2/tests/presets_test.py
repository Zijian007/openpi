"""Check training masks against the real Pi05 parameter tree without allocating weights."""

import runpy
from unittest.mock import patch

import flax.nnx as nnx
import jax
import pytest
import yaml

from openpi.adapters.g2 import config as g2_config


def test_base_template_imports_without_reading_training_yaml():
    path = g2_config.PROJECT_ROOT / "src/openpi/adapters/g2/presets.py"
    with patch.object(g2_config, "read_yaml", side_effect=AssertionError("base template must not depend on live YAML")):
        namespace = runpy.run_path(str(path))
    config = namespace["base_config"]("right")
    assert config.data.repo_id == "vr/Sing_TeaPi/lerobot_v21"
    assert config.data.arm == "right"


def test_experiments_override_base_template_dataset():
    settings = g2_config.load_settings()
    settings["data"]["repo_id"] = "vr/another_task/lerobot_v21"
    config = g2_config.build_train_config(settings)
    assert config.data.repo_id == settings["data"]["repo_id"]


@pytest.mark.parametrize("arm", ["both", "left", "right"])
def test_vlm_lora_action_full_restores_model_and_arm_from_snapshot(arm):
    settings = g2_config.resolve_settings(g2_config.load_settings(), {"arm": arm})
    config = g2_config.build_train_config(settings)
    payload = yaml.safe_load(config.checkpoint_assets[g2_config.CHECKPOINT_CONFIG])
    restored = g2_config.build_train_config(
        g2_config.normalize_settings(payload["settings"]), tuple(payload["extra_args"])
    )

    assert restored.name == config.name
    assert restored.data.arm == arm
    assert restored.model.pi05
    assert restored.model.paligemma_variant == "gemma_2b_lora"
    assert restored.model.action_expert_variant == "gemma_300m"
    assert restored.model.vision_lora.rank == 16
    assert restored.model.vision_lora.alpha == 16.0
    assert restored.ema_decay is None


def test_vlm_lora_action_full_trains_only_vlm_adapters_and_full_action_path():
    settings = g2_config.resolve_settings(g2_config.load_settings(), {})
    config = g2_config.build_train_config(settings)
    model = nnx.eval_shape(config.model.create, jax.random.key(0))
    params = nnx.state(model, nnx.Param)
    paths = set(params.flat_state())
    trainable = set(params.filter(config.trainable_filter).flat_state())
    frozen = set(params.filter(config.freeze_filter).flat_state())

    vision = {path for path in paths if path[:2] == ("PaliGemma", "img")}
    vision_lora = {path for path in vision if "lora" in "/".join(map(str, path))}
    llm = {path for path in paths if path[:2] == ("PaliGemma", "llm")}
    action_expert = {path for path in llm if any(str(part).endswith("_1") for part in path)}
    vlm = llm - action_expert
    lora = {path for path in vlm if "lora" in "/".join(map(str, path))}
    action_projections = {path for path in paths if path[0] != "PaliGemma"}

    # Guard against vacuous checks if upstream module names change.
    assert vision
    assert vision_lora
    assert {path[5] for path in vision_lora} == {"query", "key", "value", "out"}
    assert action_expert
    assert vlm - lora
    assert lora
    assert {path[0] for path in action_projections} == {
        "action_in_proj",
        "action_out_proj",
        "time_mlp_in",
        "time_mlp_out",
    }
    assert trainable.isdisjoint(frozen)
    assert trainable | frozen == paths
    assert vision - vision_lora <= frozen
    assert vision_lora <= trainable
    assert vlm - lora <= frozen
    assert lora <= trainable
    assert action_expert <= trainable
    assert action_projections <= trainable
    assert not any("lora" in "/".join(map(str, path)) for path in action_expert)
