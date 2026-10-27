"""Public YAML behavior, legacy preservation, and independent module training modes."""

import dataclasses
import warnings

import flax.nnx as nnx
import jax
import numpy as np
import pytest
import yaml

from openpi.adapters.g2 import config as g2_config
from openpi.adapters.g2 import presets
from openpi.adapters.g2.transforms import ActionCodec


def load_partial(tmp_path, text):
    path = tmp_path / "experiment.yaml"
    path.write_text(text)
    return g2_config.resolve_settings(g2_config.load_settings(path), {})


def test_joint_quantity_infers_space_and_preserves_wire_round_trip(tmp_path):
    settings = load_partial(
        tmp_path,
        "schema_version: 3\nexperiment:\n  arm: both\npolicy:\n"
        "  state:\n    space: joint\n  action:\n    control_quantity: joint_position\n"
        "    encoding: component_delta\n    reference: current_measured\n",
    )
    assert settings["policy"]["action"]["space"] == "joint"
    assert g2_config.describe_policy(settings)["label_column"] == "action.joint"
    obs = {"observation/joint": np.arange(16, dtype=np.float32)}
    targets = np.arange(32, dtype=np.float32).reshape(2, 16) / 10
    codec = ActionCodec(settings["policy"], "both")
    encoded = codec.encode(targets, obs)
    np.testing.assert_array_equal(encoded[:, [7, 15]], targets[:, [7, 15]])
    np.testing.assert_allclose(codec.decode(encoded, obs), targets, atol=1e-6)


def test_state_and_action_use_the_same_explicit_component_delta_vocabulary(tmp_path):
    settings = load_partial(
        tmp_path,
        "policy:\n  state:\n    encoding: component_delta\n    reference: episode_start\n"
        "  action:\n    encoding: se3_body_delta\n    reference: current_command\n",
    )
    assert settings["policy"]["state"]["representation"] == "relative"
    assert settings["policy"]["state"]["relative_to"] == "episode_start_pose"
    assert settings["policy"]["action"]["relative_to"] == "current_command_pose"
    assert g2_config.upgrade_settings(g2_config.public_settings(settings)) == settings


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("policy:\n  state:\n    encoding: relative\n", "must be absolute or component_delta"),
        ("policy:\n  state:\n    encoding: absolute\n    representation: absolute\n", "Cannot mix"),
        ("policy:\n  action:\n    reference: object\n    relative_to: object_pose\n", "Cannot mix"),
        ("policy:\n  action:\n    reference: typo\n", "must be one of"),
        ("finetuning:\n  vision: typo\n", "must be one of"),
        ("finetuning:\n  language: []\n", "must be one of"),
        ("experiment:\n  preset: []\n", "Unknown legacy preset"),
        ("finetuning:\n  vision: frozen\n  language: frozen\n  action_expert: frozen\n", "At least one"),
        ("run:\n  smoke: 'true'\n", "must be a YAML boolean"),
        ("schema_version: 3\nexperiment:\n  preset: full_ft\n", "schema v3 uses"),
    ],
)
def test_rejects_ambiguous_or_invalid_public_settings(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        load_partial(tmp_path, text)


def test_absolute_override_clears_only_omitted_reference(tmp_path):
    settings = load_partial(tmp_path, "policy:\n  action:\n    encoding: absolute\n")
    assert settings["policy"]["action"]["relative_to"] is None
    assert settings["policy"]["action"]["delta_scope"] is None
    with pytest.raises(ValueError, match="must be null"):
        load_partial(tmp_path, "policy:\n  action:\n    encoding: absolute\n    reference: object\n")


@pytest.mark.parametrize("legacy_preset", ["low_mem", "full_ft", "vlm_lora_action_full", "smoke"])
@pytest.mark.parametrize("vision_rank", [0, 16])
def test_legacy_snapshots_keep_model_freezing_namespace_and_ema(legacy_preset, vision_rank):
    old = g2_config.load_settings()
    old.pop("finetuning")
    old.pop("run")
    old["experiment"]["preset"] = legacy_preset
    old["model"]["vision_lora_rank"] = vision_rank
    if legacy_preset == "smoke":
        old["training"]["num_train_steps"] = 20
    name = g2_config.PRESET_CONFIG_NAMES[legacy_preset] + "_right"
    original = presets.get_config(name)
    expected_model = dataclasses.replace(original.model, vision_lora_rank=vision_rank)
    expected_filter = nnx.Any(original.freeze_filter, expected_model.get_freeze_filter())
    restored = g2_config.build_train_config(old)
    assert restored.name == name
    assert restored.model == expected_model
    assert restored.ema_decay == original.ema_decay
    for path in (
        ("PaliGemma", "img", "embedding", "kernel"),
        ("PaliGemma", "img", "Transformer", "lora_a"),
        ("PaliGemma", "llm", "layers", "attn", "q_einsum", "w"),
        ("PaliGemma", "llm", "layers", "attn", "q_einsum", "lora_a"),
        ("PaliGemma", "llm", "layers", "attn", "q_einsum_1", "w"),
        ("PaliGemma", "llm", "layers", "attn", "q_einsum_1", "lora_a"),
        ("action_out_proj", "kernel"),
    ):
        # Compare only adapter paths that exist in this legacy model variant.
        if "lora" in path[-1]:
            if path[:2] == ("PaliGemma", "img") and not vision_rank:
                continue
            if "q_einsum_1" in path and "lora" not in expected_model.action_expert_variant:
                continue
            if "q_einsum" in path and "lora" not in expected_model.paligemma_variant:
                continue
        assert restored.freeze_filter(path, None) == expected_filter(path, None)
    public = yaml.safe_load(restored.checkpoint_assets[g2_config.CHECKPOINT_CONFIG])["settings"]
    again = g2_config.build_train_config(public)
    assert again.model == restored.model
    assert again.name == restored.name
    assert again.ema_decay == restored.ema_decay


@pytest.mark.parametrize(
    "modes",
    [
        {"vision": "full", "language": "full", "action_expert": "full"},
        {"vision": "frozen", "language": "lora", "action_expert": "full"},
        {"vision": "full", "language": "frozen", "action_expert": "lora"},
        {"vision": "lora", "language": "full", "action_expert": "frozen"},
    ],
)
def test_module_modes_select_the_actual_parameters_and_smoke_preserves_them(modes):
    settings = g2_config.load_settings()
    settings["finetuning"] = modes
    normal = g2_config.build_train_config(settings)
    smoke_settings = g2_config.resolve_settings(settings, {"smoke": True})
    smoke = g2_config.build_train_config(smoke_settings)
    assert smoke.model == normal.model
    assert smoke_settings["finetuning"] == modes
    assert smoke.lr_schedule == normal.lr_schedule
    assert smoke.num_train_steps == 20
    assert smoke.batch_size == 1
    assert not smoke.wandb_enabled
    model = nnx.eval_shape(normal.model.create, jax.random.key(0))
    params = nnx.state(model, nnx.Param)
    paths = set(params.flat_state())
    trainable = set(params.filter(normal.trainable_filter).flat_state())
    assert trainable == set(params.filter(smoke.trainable_filter).flat_state())
    groups = {
        "vision": {p for p in paths if p[:2] == ("PaliGemma", "img")},
        "language": {p for p in paths if p[:2] == ("PaliGemma", "llm") and not any(str(x).endswith("_1") for x in p)},
        "action_expert": {p for p in paths if p[:2] == ("PaliGemma", "llm") and any(str(x).endswith("_1") for x in p)},
    }
    for module, group in groups.items():
        assert group
        adapters = {p for p in group if "lora" in "/".join(map(str, p))}
        if modes[module] == "lora":
            assert adapters
            assert group & trainable == adapters
        else:
            assert not adapters
            assert group & trainable == (group if modes[module] == "full" else set())
    projections = {p for p in paths if p[0] != "PaliGemma"}
    assert projections
    assert projections & trainable == (set() if modes["action_expert"] == "frozen" else projections)
    if set(modes.values()) == {"full"}:
        assert paths == trainable


def test_full_cli_disables_vision_lora_despite_rank_in_yaml():
    settings = g2_config.resolve_settings(g2_config.load_settings(), {"preset": "full_ft"})
    config = g2_config.build_train_config(settings)
    assert settings["finetuning"] == {"vision": "full", "language": "full", "action_expert": "full"}
    assert config.model.vision_lora_rank == 0


def test_optimizer_override_is_applied_and_saved():
    extra = ("--optimizer.clip-gradient-norm=0.5", "--ema-decay=0.9")
    config = g2_config.build_train_config(g2_config.load_settings(), extra)
    assert config.optimizer.clip_gradient_norm == 0.5
    assert config.ema_decay == 0.9
    payload = yaml.safe_load(config.checkpoint_assets[g2_config.CHECKPOINT_CONFIG])
    restored = g2_config.build_train_config(payload["settings"], tuple(payload["extra_args"]))
    assert restored.optimizer == config.optimizer
    assert restored.ema_decay == config.ema_decay


def test_single_frame_self_reference_warns_but_history_does_not(tmp_path):
    text = "policy:\n  state:\n    encoding: component_delta\n    reference: current_measured\n"
    with pytest.warns(UserWarning, match="zeros the EEF"):
        load_partial(tmp_path, text)
    with warnings.catch_warnings(record=True) as caught:
        load_partial(tmp_path, text + "  timing:\n    history_steps: 2\n")
    assert not caught
