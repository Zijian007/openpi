import dataclasses
from unittest.mock import patch

import pytest
import yaml

from openpi.adapters.g2 import config as g2_config


@pytest.mark.parametrize(
    "overrides",
    [{}, {"arm": "right"}, {"action_representation": "absolute"}, {"preset": "full_ft"}, {"preset": "low_mem"}],
)
def test_build_config_preserves_model_data_and_lr_settings(overrides):
    settings = g2_config.resolve_settings(g2_config.load_settings(), overrides)
    config = g2_config.build_train_config(settings)
    assert config.model.action_horizon == 50
    assert config.data.repo_id == settings["data"]["repo_id"]
    assert config.data.use_delta_actions == (settings["policy"]["action"]["representation"] == "component_delta")
    assert config.lr_schedule.decay_steps == config.num_train_steps == 50000
    assert (
        g2_config.upgrade_settings(yaml.safe_load(config.checkpoint_assets[g2_config.CHECKPOINT_CONFIG])["settings"])
        == settings
    )


@pytest.mark.parametrize("preset", ["low_mem", "vlm_lora_action_full"])
@pytest.mark.parametrize("vision_rank", [0, 4, 16])
def test_checkpoint_snapshot_is_self_contained_after_yaml_is_edited(tmp_path, preset, vision_rank):
    settings = g2_config.resolve_settings(
        g2_config.load_settings(), {"action_horizon": 30, "arm": "right", "preset": preset}
    )
    settings["data"]["cameras"] = ["head_color"]
    settings["model"].update(vision_lora_rank=vision_rank, vision_lora_alpha=8.0)
    settings["finetuning"]["vision"] = "lora" if vision_rank else "frozen"
    config = g2_config.build_train_config(settings)
    snapshot = tmp_path / g2_config.CHECKPOINT_CONFIG
    snapshot.write_text(yaml.safe_dump(yaml.safe_load(config.checkpoint_assets[g2_config.CHECKPOINT_CONFIG])))
    # Loading a checkpoint must not merge in today's template settings.
    with patch("openpi.adapters.g2.config.load_settings", side_effect=AssertionError("must not reload template")):
        payload = g2_config.read_yaml(snapshot)
        restored = g2_config.build_train_config(payload["settings"], tuple(payload["extra_args"]))
    assert restored.model.action_horizon == 30
    assert restored.data.arm == "right"
    assert restored.data.cameras == ("head_color",)
    assert restored.policy_metadata["action_horizon"] == 30
    assert restored.model == config.model


@pytest.mark.parametrize("preset", ["low_mem", "vlm_lora_action_full", "full_ft"])
def test_old_checkpoint_keeps_vision_lora_disabled(preset):
    settings = g2_config.resolve_settings(g2_config.load_settings(), {"preset": preset})
    settings["model"].pop("vision_lora_rank")
    settings["model"].pop("vision_lora_alpha")
    settings.pop("finetuning")
    settings.pop("run")
    with patch("openpi.adapters.g2.config.load_settings", side_effect=AssertionError("must not reload template")):
        config = g2_config.build_train_config(settings)
    assert config.model.vision_lora_rank == 0
    assert config.model.vision_lora_alpha == 16.0


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
    ("path", "value"),
    [
        (("policy", "action", "space"), "joint"),
        (("training", "batch_size"), 0),
        (("training", "batch_size"), True),
        (("training", "num_train_steps"), 5),
        (("checkpoints", "overwrite"), "false"),
        (("model", "vision_lora_rank"), -1),
        (("model", "vision_lora_rank"), True),
        (("model", "vision_lora_rank"), 2.5),
        (("model", "vision_lora_alpha"), 0),
        (("model", "vision_lora_alpha"), float("inf")),
        (("data", "cameras"), ["hand_left_color"]),
        (("data", "cameras"), ["head_color", "head_color"]),
        (("data", "repo_id"), "../outside"),
    ],
)
def test_invalid_yaml_fails_before_training(path, value):
    settings = g2_config.load_settings()
    target = settings
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match="must|exceed|requires"):
        g2_config.validate_settings(settings)


def test_upgrade_drops_legacy_proxy_runtime_fields():
    settings = g2_config.load_settings()
    settings["runtime"]["use_proxy"] = False
    settings["runtime"]["proxy"] = "socks5h://127.0.0.1:1080"
    upgraded = g2_config.upgrade_settings(settings)
    assert "use_proxy" not in upgraded["runtime"]
    assert "proxy" not in upgraded["runtime"]
    g2_config.validate_settings(upgraded)


def test_wrist_only_cameras_are_valid_for_the_selected_arm():
    settings = g2_config.load_settings()
    settings["experiment"]["arm"] = "right"
    settings["data"]["cameras"] = ["hand_right_color"]
    g2_config.validate_settings(settings)
    assert g2_config.selected_cameras(settings) == ("hand_right_color",)


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


@pytest.mark.parametrize(
    ("state_space", "representation"), [("eef", "component_delta"), ("eef", "absolute"), ("joint", "absolute")]
)
@pytest.mark.parametrize("preset", ["low_mem", "vlm_lora_action_full"])
def test_serving_restores_checkpoint_settings_before_loading_model(
    tmp_path, monkeypatch, state_space, representation, preset
):
    import types

    from openpi.adapters.g2 import policy as policy_config
    from openpi.adapters.g2 import presets as g2_configs
    from openpi.models import model as model_lib
    from openpi.models import pi0_config
    from openpi.policies import policy as policy_lib
    from openpi.training import config as config_lib

    settings = g2_config.resolve_settings(
        g2_config.load_settings(),
        {
            "arm": "right",
            "action_horizon": 30,
            "state_space": state_space,
            "action_representation": representation,
            "preset": preset,
        },
    )
    settings["policy"]["timing"]["execution_steps"] = 10
    settings["data"]["cameras"] = ["head_color"]
    snapshot = tmp_path / "assets" / g2_config.CHECKPOINT_CONFIG
    snapshot.parent.mkdir()
    snapshot.write_text(yaml.safe_dump({"settings": settings, "extra_args": []}))
    loaded_models = []

    def load(model_config, params):
        loaded_models.append(model_config)
        return object()

    monkeypatch.setattr(policy_config.download, "maybe_download", lambda path: tmp_path)
    monkeypatch.setattr(model_lib, "restore_params", lambda *args, **kwargs: {})
    monkeypatch.setattr(pi0_config.Pi0Config, "load", load)
    monkeypatch.setattr(g2_configs.LeRobotG2DataConfig, "create", lambda *args: config_lib.DataConfig(asset_id="test"))
    monkeypatch.setattr(policy_lib, "Policy", lambda *args, **kwargs: types.SimpleNamespace(**kwargs))
    policy = policy_config.create_trained_policy(
        g2_configs.get_config(g2_config.preset_name(settings)), tmp_path, norm_stats={}
    )
    assert len(loaded_models) == 1
    assert loaded_models[0] == g2_config.build_train_config(settings).model
    assert policy.metadata["action_horizon"] == 30
    assert policy.metadata["required_cameras"] == ["head_color"]
    assert policy.metadata["requires_joint_state"] == (state_space == "joint")
    assert "execution_steps" not in policy.metadata
    assert policy.metadata["policy_contract"] == settings["policy"]


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
