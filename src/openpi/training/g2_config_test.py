import dataclasses
from unittest.mock import patch

import pytest
import yaml

from openpi.training import g2_config


@pytest.mark.parametrize("overrides", [{}, {"arm": "right"}, {"action_mode": "absolute"}, {"preset": "full_ft"}])
def test_build_config_preserves_model_data_and_lr_settings(overrides):
    settings = g2_config.resolve_settings(g2_config.load_settings(), overrides)
    config = g2_config.build_train_config(settings)
    assert config.model.action_horizon == 50
    assert config.data.repo_id == settings["data"]["repo_id"]
    assert config.data.use_delta_actions == (settings["data"]["action_mode"] == "relative")
    assert config.lr_schedule.decay_steps == config.num_train_steps == 50000
    assert config.g2_training_config["settings"] == settings


def test_checkpoint_snapshot_is_self_contained_after_yaml_is_edited(tmp_path):
    settings = g2_config.resolve_settings(g2_config.load_settings(), {"action_horizon": 30, "arm": "right"})
    settings["data"]["cameras"] = ["head_color"]
    config = g2_config.build_train_config(settings)
    snapshot = tmp_path / g2_config.CHECKPOINT_CONFIG
    snapshot.write_text(yaml.safe_dump(config.g2_training_config))
    # Loading a checkpoint must not merge in today's template settings.
    with patch("openpi.training.g2_config.load_settings", side_effect=AssertionError("must not reload template")):
        payload = g2_config.read_yaml(snapshot)
        restored = g2_config.build_train_config(payload["settings"], tuple(payload["extra_args"]))
    assert restored.model.action_horizon == 30
    assert restored.data.arm == "right"
    assert restored.data.cameras == ("head_color",)
    assert restored.policy_metadata["action_horizon"] == 30


def test_norm_and_train_workers_use_identical_resolved_settings(tmp_path):
    # Exercise the actual stage dispatch without reading a dataset or initializing a model.
    import importlib.util
    import sys
    import types

    script = g2_config.PROJECT_ROOT / "scripts/g2/train/train_pipeline.py"
    spec = importlib.util.spec_from_file_location("g2_pipeline_worker_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = g2_config.resolve_settings(g2_config.load_settings(), {"arm": "left", "action_horizon": 25})
    settings["checkpoints"]["assets_dir"] = str(tmp_path / "custom-assets")
    settings["training"]["batch_size"] = 3
    settings["data"]["cameras"] = ["head_color"]
    snapshot = tmp_path / "resolved.yaml"
    snapshot.write_text(yaml.safe_dump({"settings": settings, "extra_args": []}))
    captured = []

    def compute(config, max_frames=None):
        captured.append(config)
        (config.assets_dirs / config.data.repo_id).mkdir(parents=True)

    fake_norm = types.ModuleType("compute_norm_stats")
    fake_norm.compute_norm_stats = compute
    fake_train = types.ModuleType("train")
    fake_train.main = captured.append
    with patch.dict(sys.modules, {"compute_norm_stats": fake_norm, "train": fake_train}), patch.dict("os.environ"):
        module.main(["--stage", "norm", "--resolved-config", str(snapshot)])
        module.main(["--stage", "train", "--resolved-config", str(snapshot)])
    norm, train = captured
    assert dataclasses.asdict(norm.model) == dataclasses.asdict(train.model)
    assert norm.data == train.data
    assert norm.assets_dirs == train.assets_dirs
    assert norm.batch_size == train.batch_size == 3
    assert norm.data.cameras == ("head_color",)
    assert norm.model.action_horizon == 25


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("data", "action_space", "joint"),
        ("training", "batch_size", 0),
        ("training", "batch_size", True),
        ("training", "num_train_steps", 5),
        ("checkpoints", "overwrite", "false"),
        ("data", "cameras", ["hand_left_color"]),
        ("data", "cameras", ["head_color", "head_color"]),
        ("data", "repo_id", "../outside"),
    ],
)
def test_invalid_yaml_fails_before_training(section, key, value):
    settings = g2_config.load_settings()
    settings[section][key] = value
    with pytest.raises(ValueError, match="must|exceed|relative"):
        g2_config.validate_settings(settings)


def test_smoke_uses_valid_schedule_and_disables_wandb():
    settings = g2_config.resolve_settings(g2_config.load_settings(), {"preset": "smoke"})
    config = g2_config.build_train_config(settings)
    assert config.num_train_steps == 20
    assert config.batch_size == 1
    assert config.lr_schedule.decay_steps > config.lr_schedule.warmup_steps
    assert not config.wandb_enabled


def test_yaml_accepts_wandb_on_and_scientific_learning_rate(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("wandb:\n  mode: on\ntraining:\n  peak_lr: 1e-5\n")
    settings = g2_config.resolve_settings(g2_config.load_settings(path), {})
    assert settings["wandb"]["mode"] == "on"
    assert settings["training"]["peak_lr"] == 1e-5


def test_serving_restores_checkpoint_settings_before_loading_model(tmp_path, monkeypatch):
    import types

    from openpi.models import model as model_lib
    from openpi.models import pi0_config
    from openpi.policies import policy as policy_lib
    from openpi.policies import policy_config
    from openpi.training import config as config_lib

    settings = g2_config.resolve_settings(g2_config.load_settings(), {"arm": "right", "action_horizon": 30})
    settings["data"]["cameras"] = ["head_color"]
    snapshot = tmp_path / "assets" / g2_config.CHECKPOINT_CONFIG
    snapshot.parent.mkdir()
    snapshot.write_text(yaml.safe_dump({"settings": settings, "extra_args": []}))
    loaded_horizons = []

    def load(model_config, params):
        loaded_horizons.append(model_config.action_horizon)
        return object()

    monkeypatch.setattr(policy_config.download, "maybe_download", lambda path: tmp_path)
    monkeypatch.setattr(model_lib, "restore_params", lambda *args, **kwargs: {})
    monkeypatch.setattr(pi0_config.Pi0Config, "load", load)
    monkeypatch.setattr(config_lib.LeRobotG2DataConfig, "create", lambda *args: config_lib.DataConfig(asset_id="test"))
    monkeypatch.setattr(policy_lib, "Policy", lambda *args, **kwargs: types.SimpleNamespace(**kwargs))
    policy = policy_config.create_trained_policy(
        config_lib.get_config("pi05_g2_vr_low_mem_right"), tmp_path, norm_stats={}
    )
    assert loaded_horizons == [30]
    assert policy.metadata["action_horizon"] == 30
    assert policy.metadata["camera_ids"] == ["head_color"]


@pytest.mark.parametrize("arm", ["both", "right"])
def test_yaml_camera_selection_reaches_dataset_transforms(tmp_path, monkeypatch, arm):
    import numpy as np

    from openpi import transforms
    from openpi.training import config as config_lib

    settings = g2_config.resolve_settings(g2_config.load_settings(), {"arm": arm})
    settings["data"]["cameras"] = ["head_color"]
    config = g2_config.build_train_config(settings)
    monkeypatch.setattr(config_lib.DataConfigFactory, "_load_norm_stats", lambda *args: None)
    monkeypatch.setattr(config_lib.ModelTransformFactory, "__call__", lambda *args: transforms.Group())
    data_config = config.data.create(tmp_path, config.model)
    sample = {
        "observation.images.head_color": np.ones((224, 224, 3), np.uint8),
        "state.pose": np.arange(20, dtype=np.float32),
        "action.pose": np.zeros((50, 20), np.float32),
        "prompt": "tea",
    }
    for transform in (*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs):
        sample = transform(sample)
    assert sample["state"].shape == ((20,) if arm == "both" else (10,))
    assert not sample["image_mask"]["left_wrist_0_rgb"]
    assert not sample["image_mask"]["right_wrist_0_rgb"]
