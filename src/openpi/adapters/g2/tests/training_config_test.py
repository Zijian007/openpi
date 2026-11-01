"""OpenPI owns its YAML schema, translation and configuration import boundary."""

import json
import subprocess
import sys

import pytest

from openpi.adapters.g2.training_config import openpi_snapshot
from openpi.adapters.g2.training_config import to_native


@pytest.fixture
def source():
    return {
        "schema_version": 8,
        "backend": "openpi",
        "experiment": {"name": "test", "arm": "right"},
        "data": {"root": "data", "repo_id": "org/task", "cameras": None},
        "policy": {"timing": {"action_horizon": 50}},
        "model": {"type": "pi05", "pretrained_path": "base"},
        "training": {
            "batch_size": 8,
            "steps": 10000,
            "num_workers": 4,
            "seed": 42,
            "lr_schedule": {"peak_lr": 2.5e-5},
        },
        "runtime": {"gpus": [0]},
        "checkpoints": {"resume": False, "save_interval": 1000},
        "logging": {"log_interval": 50, "wandb": {"mode": "off", "project": "test"}},
    }


def test_openpi_snapshot_round_trips_all_training_settings(source):
    source["training"]["lr_schedule"]["warmup_steps"] = 1000
    source["model"]["discrete_state_input"] = True
    source["checkpoints"].update(base_dir="checkpoints", assets_dir="assets", overwrite=False, keep_period=None)
    source["runtime"]["mem_fraction"] = 0.9
    native = to_native(source)
    assert native["training"]["num_train_steps"] == 10000
    assert native["training"]["peak_lr"] == 2.5e-5
    assert native["checkpoints"]["resume"] is False
    assert native["model"]["params_path"] == "base"
    assert openpi_snapshot(native)["schema_version"] == 8
    assert to_native(openpi_snapshot(native)) == native


@pytest.mark.parametrize("version", [None, 5, 6, 7, 9, 7.0, 8.0, "7", "8", True])
def test_unsupported_training_version_is_rejected(source, version):
    source["schema_version"] = version
    with pytest.raises(ValueError, match="schema_version: 8"):
        to_native(source)


def test_other_backend_is_rejected(source):
    source["backend"] = "lerobot"
    with pytest.raises(ValueError, match="backend must be openpi"):
        to_native(source)


@pytest.mark.parametrize("section", [None, "model", "training", "runtime", "logging", "checkpoints"])
def test_unknown_entry_fields_are_rejected(source, section):
    target = source if section is None else source[section]
    target["typo"] = 1
    with pytest.raises(ValueError, match="Unknown YAML field"):
        to_native(source)


@pytest.mark.parametrize("section", ["model", "training", "runtime", "logging", "checkpoints"])
def test_entry_sections_must_be_mappings(source, section):
    source[section] = None
    with pytest.raises(TypeError, match="must be a YAML mapping"):
        to_native(source)


def test_training_translation_needs_no_wire_package_or_model_framework(source):
    code = """
import importlib.abc
import json
import sys

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in (
            'g2_policy_protocol', 'jax', 'torch', 'lerobot', 'wholebody',
        ):
            raise ImportError('training configuration must not import ' + fullname)

sys.meta_path.insert(0, Block())
from openpi.adapters.g2.training_config import to_native
result = to_native(json.loads(sys.argv[1]))
assert result
"""
    result = subprocess.run(
        [sys.executable, "-c", code, json.dumps(source)], capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stderr
