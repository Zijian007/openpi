"""Independent module training modes and advanced optimizer settings."""

import flax.nnx as nnx
import jax
import pytest
import yaml

from openpi.adapters.g2 import config as g2_config


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
    for module, mode in modes.items():
        settings["finetuning"][module]["mode"] = mode
    normal = g2_config.build_train_config(settings)
    smoke_settings = g2_config.resolve_settings(settings, {"smoke": True})
    smoke = g2_config.build_train_config(smoke_settings)
    assert smoke.model == normal.model
    assert g2_config.finetuning_modes(smoke_settings) == modes
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


def test_optimizer_override_is_applied_and_saved():
    extra = ("--optimizer.clip-gradient-norm=0.5", "--ema-decay=0.9")
    config = g2_config.build_train_config(g2_config.load_settings(), extra)
    assert config.optimizer.clip_gradient_norm == 0.5
    assert config.ema_decay == 0.9
    payload = yaml.safe_load(config.checkpoint_assets[g2_config.CHECKPOINT_CONFIG])
    restored = g2_config.build_train_config(
        g2_config.normalize_settings(payload["settings"]), tuple(payload["extra_args"])
    )
    assert restored.optimizer == config.optimizer
    assert restored.ema_decay == config.ema_decay
