"""YAML adapter settings reach real parameter shapes, masks and self-contained snapshots."""

import copy
import dataclasses
import json
from unittest.mock import patch

import flax.nnx as nnx
import jax
import pytest
import yaml

from openpi.adapters.g2 import config


def resolve(tmp_path, finetuning):
    path = tmp_path / "lora.yaml"
    path.write_text(yaml.safe_dump({"finetuning": finetuning}))
    return config.resolve_settings(config.load_settings(path), {})


def test_custom_yaml_controls_real_adapter_shapes_targets_and_restoration(tmp_path):
    specs = {
        "vision": {"mode": "lora", "lora": {"rank": 4, "alpha": 12.0, "targets": ["attention"], "rslora": True}},
        "language": {"mode": "lora", "lora": {"rank": 3, "alpha": 9.0, "targets": ["attention"], "rslora": False}},
        "action_expert": {"mode": "lora", "lora": {"rank": 5, "alpha": 10.0, "targets": ["ffn"], "rslora": True}},
    }
    settings = resolve(tmp_path, specs)
    built = config.build_train_config(settings)
    language, expert = built.model.gemma_configs()
    assert set(language.lora_configs) == {"attn"}
    assert set(expert.lora_configs) == {"ffn"}
    assert language.lora_configs["attn"].rank == 3
    assert language.lora_configs["attn"].alpha == 9.0
    assert expert.lora_configs["ffn"].rslora is True
    assert built.model.vision_lora.rslora is True
    assert built.model.vision_lora.alpha == 12.0
    # Model settings remain serializable, including nested LoRA options.
    json.dumps(dataclasses.asdict(built.model))
    abstract = nnx.eval_shape(built.model.create, jax.random.key(0))
    params = nnx.state(abstract, nnx.Param)
    paths = set(params.flat_state())
    trainable = set(params.filter(built.trainable_filter).flat_state())
    groups = {
        "vision": {p for p in paths if p[:2] == ("PaliGemma", "img")},
        "language": {p for p in paths if p[:2] == ("PaliGemma", "llm") and not any(str(x).endswith("_1") for x in p)},
        "action_expert": {p for p in paths if p[:2] == ("PaliGemma", "llm") and any(str(x).endswith("_1") for x in p)},
    }
    flat = dict(params.flat_state())
    for module, group in groups.items():
        adapters = {p for p in group if "lora" in "/".join(map(str, p))}
        assert adapters
        assert trainable & group == adapters
        for path in adapters:
            if str(path[-1]).endswith("lora_a"):
                assert flat[path].value.shape[-1] == specs[module]["lora"]["rank"]
        if module == "language":
            assert all("attn" in path for path in adapters)
        elif module == "action_expert":
            assert all("mlp_1" in path for path in adapters)
    projections = {p for p in paths if p[0] != "PaliGemma"}
    assert projections <= trainable

    payload = yaml.safe_load(built.checkpoint_assets[config.CHECKPOINT_CONFIG])
    assert payload["settings"]["finetuning"] == specs
    with patch.object(config, "load_settings", side_effect=AssertionError("must not reload current YAML")):
        restored = config.build_train_config(config.normalize_settings(payload["settings"]))
    assert restored.model == built.model
    assert config.model_semantics(config.normalize_settings(payload["settings"])) == config.model_semantics(settings)


def test_ffn_only_language_adapter_does_not_create_attention_adapters(tmp_path):
    settings = resolve(tmp_path, {"language": {"lora": {"rank": 7, "alpha": 21.0, "targets": ["ffn"]}}})
    built = config.build_train_config(settings)
    language, _ = built.model.gemma_configs()
    assert set(language.lora_configs) == {"ffn"}
    assert language.lora_configs["ffn"].rank == 7
    abstract = nnx.eval_shape(built.model.create, jax.random.key(0))
    paths = set(nnx.state(abstract, nnx.Param).flat_state())
    adapters = {
        p
        for p in paths
        if p[:2] == ("PaliGemma", "llm")
        and "lora" in "/".join(map(str, p))
        and not any(str(x).endswith("_1") for x in p)
    }
    assert adapters
    assert all("mlp" in path for path in adapters)


@pytest.mark.parametrize("module", ["vision", "language", "action_expert"])
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("rank", 0),
        ("rank", -1),
        ("rank", True),
        ("rank", 2.5),
        ("alpha", 0),
        ("alpha", float("inf")),
        ("alpha", True),
        ("rslora", "true"),
        ("targets", []),
        ("targets", "attention"),
        ("targets", ["attention", "attention"]),
        ("targets", ["unknown"]),
    ],
)
def test_invalid_active_adapter_parameters_fail_before_model_creation(tmp_path, module, key, value):
    with pytest.raises(ValueError, match=f"finetuning.{module}.lora.{key}"):
        resolve(tmp_path, {module: {"mode": "lora", "lora": {key: value}}})


def test_vision_rejects_unsupported_ffn_target(tmp_path):
    with pytest.raises(ValueError, match="targets"):
        resolve(tmp_path, {"vision": {"lora": {"targets": ["ffn"]}}})


def test_module_mode_cli_override_preserves_detailed_yaml_options(tmp_path):
    settings = resolve(tmp_path, {"language": {"mode": "full", "lora": {"rank": 6, "alpha": 18.0}}})
    active = config.resolve_settings(settings, {"language_mode": "lora"})
    assert active["finetuning"]["language"]["lora"] == settings["finetuning"]["language"]["lora"]
    assert config.build_train_config(active).model.language_lora.rank == 6
    assert config.build_train_config(settings).model.language_lora is None


@pytest.mark.parametrize("mode", ["full", "frozen"])
def test_inactive_lora_settings_do_not_change_the_model_or_resume_semantics(tmp_path, mode):
    settings = resolve(tmp_path, {"action_expert": {"mode": mode}})
    changed = copy.deepcopy(settings)
    changed["finetuning"]["action_expert"]["lora"].update(rank=7, alpha=99.0, targets=["attention"], rslora=True)
    assert config.build_train_config(settings).model == config.build_train_config(changed).model
    assert config.model_semantics(settings) == config.model_semantics(changed)
    assert config.public_settings(changed)["finetuning"]["action_expert"]["lora"]["rank"] == 7


@pytest.mark.parametrize(("key", "value"), [("rank", 8), ("alpha", 24.0), ("targets", ["ffn"]), ("rslora", True)])
def test_active_lora_settings_participate_in_resume_semantics(tmp_path, key, value):
    settings = resolve(tmp_path, {})
    changed = copy.deepcopy(settings)
    changed["finetuning"]["language"]["lora"][key] = value
    assert config.model_semantics(settings) != config.model_semantics(changed)
    assert config.policy_semantics(settings) == config.policy_semantics(changed)


def test_target_order_does_not_change_resume_semantics(tmp_path):
    settings = resolve(tmp_path, {})
    changed = copy.deepcopy(settings)
    changed["finetuning"]["language"]["lora"]["targets"].reverse()
    assert config.model_semantics(settings) == config.model_semantics(changed)
