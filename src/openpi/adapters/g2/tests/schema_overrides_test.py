"""v8 overrides and strict configuration validation."""

import pytest
import yaml

from openpi.adapters.g2 import config
from openpi.adapters.g2 import training_config as schema


def test_overrides_take_precedence_over_defaults(tmp_path):
    source = {
        "runtime": {"gpus": [1, 2], "fsdp_devices": 2, "mem_fraction": 0.7},
        "checkpoints": {"resume": True, "save_interval": 2000},
        "logging": {"log_interval": 20},
        "training": {"batch_size": 4, "lr_schedule": {"peak_lr": 1e-5, "warmup_steps": 200}},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(source))
    settings = config.resolve_settings(config.load_settings(path), {})
    assert settings["runtime"] == {"gpus": [1, 2], "mem_fraction": 0.7}
    assert settings["checkpoints"]["resume"] is True
    expected = {
        "batch_size": 4,
        "peak_lr": 1e-5,
        "fsdp_devices": 2,
        "warmup_steps": 200,
        "save_interval": 2000,
        "log_interval": 20,
    }
    assert {key: settings["training"][key] for key in expected} == expected


@pytest.mark.parametrize(
    ("section", "override"),
    [
        ("runtime", {"resume": False}),
        ("training", {"learning_rate": 1e-5}),
        ("wandb", {"mode": "off"}),
        ("backend_options", {"openpi": {}}),
        ("logging", {"wandb": {"typo": True}}),
        ("training", {"lr_schedule": {"warmup_step": 100}}),
        ("finetuning", {"vision": {"lora": {"typo": 1}}}),
    ],
)
def test_v8_rejects_old_paths_and_nested_typos(section, override):
    with pytest.raises(ValueError, match="Unknown YAML field"):
        schema.to_native({section: override})


def test_omitted_inactive_lora_options_allow_later_cli_activation(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("finetuning:\n  action_expert:\n    mode: frozen\n")
    settings = config.resolve_settings(config.load_settings(path), {"action_expert_mode": "lora"})
    assert settings["finetuning"]["action_expert"]["lora"] == schema.LORA_DEFAULTS["action_expert"]
    config.validate_settings(settings)


