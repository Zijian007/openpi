import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from openpi import transforms
from openpi.adapters.g2 import config as g2_config
from openpi.adapters.g2.data import DatasetFactory
from openpi.adapters.g2.data import RepackG2
from openpi.adapters.g2.mixture import prepare_mixture
from openpi.adapters.g2.resume import resolve_resume
from openpi.adapters.g2.tests.dataset_test import RecordedDataset
from openpi.adapters.g2.transforms import G2Inputs
from openpi.training.config import DataConfig


def settings_for(tmp_path):
    settings = g2_config.resolve_settings(g2_config.load_settings(), {"resume": False})
    settings["data"] = {"root": str(tmp_path), "cameras": ["head_color"], "image_size": [224, 224],
                        "sources": [{"path": "a", "weight": 3}, {"path": "b", "weight": 1}]}
    settings["policy"]["timing"].update(sample_hz=4, action_horizon=3)
    for name in ("a", "b"):
        meta = tmp_path / name / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text(json.dumps({
            "codebase_version": "v2.1", "fps": 4,
            "features": {"state.pose": {"shape": [20]}, "action.pose": {"shape": [20]},
                         "observation.images.head_color": {"shape": [3, 8, 8]}},
        }))
        (meta / "episodes.jsonl").write_text("\n".join(
            json.dumps({"episode_index": i, "length": 4}) for i in range(2)))
        (meta / "episodes_stats.jsonl").write_text("")
    prepare_mixture(settings, base_dir=tmp_path)
    return settings


def test_sources_override_default_selection_and_reject_cli_paths(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("data:\n  sources:\n    - path: a\n    - path: b\n  image_size: [224, 224]\n")
    settings = g2_config.load_settings(path)
    assert "repo_id" not in settings["data"]
    assert settings["data"]["root"] == "./data"
    g2_config.validate_settings(settings)
    for flag in ("repo_id", "data_root"):
        with pytest.raises(ValueError, match="YAML"):
            g2_config.resolve_settings(settings, {flag: "other"})
    settings["data"]["image_size"] = [480, 640]
    with pytest.raises(ValueError, match="224"):
        g2_config.validate_settings(settings)


def test_mixture_asset_survives_without_training_data(tmp_path, monkeypatch):
    settings = settings_for(tmp_path)
    expected = settings["mixture"]["asset_id"]
    assert settings["mixture"]["proportions"] == [0.75, 0.25]
    assert expected.startswith("mix-")
    config = g2_config.build_train_config(settings)
    payload = yaml.safe_load(config.checkpoint_assets[g2_config.CHECKPOINT_CONFIG])
    for name in ("a", "b"):
        (tmp_path / name).rename(tmp_path / (name + "-hidden"))
    restored = g2_config.build_train_config(g2_config.normalize_settings(payload["settings"]))
    assert restored.data.assets.asset_id == expected
    assert restored.data.sources == config.data.sources
    # Native policy restoration must find the saved asset before creating transforms.
    monkeypatch.setattr("openpi.training.config.DataConfigFactory._load_norm_stats", lambda self, root, asset: asset)
    monkeypatch.setattr("openpi.training.config.ModelTransformFactory.__call__", lambda *args: transforms.Group())
    data = restored.data.create(tmp_path, restored.model)
    assert data.asset_id == data.norm_stats == expected


def test_fingerprint_changes_with_metadata(tmp_path):
    settings = settings_for(tmp_path)
    previous = settings["mixture"]["fingerprint"]
    info = tmp_path / "b/meta/info.json"
    value = json.loads(info.read_text())
    value["note"] = "changed source"
    info.write_text(json.dumps(value))
    prepare_mixture(settings, base_dir=tmp_path)
    assert settings["mixture"]["fingerprint"] != previous


def test_multi_source_resume_uses_saved_asset_and_checks_identity(tmp_path):
    settings = settings_for(tmp_path)
    checkpoint = tmp_path / "checkpoints/1000"
    for name in ("params", "train_state", "assets"):
        (checkpoint / name).mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text(json.dumps({"commit_timestamp_nsecs": 1}))
    for name in ("params", "train_state"):
        (checkpoint / name / "_METADATA").write_text("{}")
    (checkpoint / "assets" / g2_config.CHECKPOINT_CONFIG).write_text(yaml.safe_dump({
        "settings": g2_config.public_settings(settings), "extra_args": [],
    }))
    norm = checkpoint / "assets" / settings["mixture"]["asset_id"] / "norm_stats.json"
    norm.parent.mkdir()
    norm.write_text("{}")
    resolved, _, selected, _ = resolve_resume(settings, (), checkpoint.parent)
    assert selected == checkpoint
    assert g2_config.asset_id(resolved) == settings["mixture"]["asset_id"]
    settings["mixture"]["fingerprint"] = "changed"
    with pytest.raises(ValueError, match="mixture fingerprint"):
        resolve_resume(settings, (), checkpoint.parent)


def test_multi_source_dry_run_reports_realized_proportions(tmp_path, capsys):
    settings = settings_for(tmp_path)
    source = g2_config.public_settings(settings)
    source["backend_options"]["openpi"].pop("mixture")
    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump(source))
    spec = importlib.util.spec_from_file_location("g2_source_pipeline_test",
                                               g2_config.PROJECT_ROOT / "scripts/g2/train/train_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(["--config", str(config), "--dry-run"])
    output = capsys.readouterr().out
    assert "75.00%" in output
    assert "25.00%" in output
    assert settings["mixture"]["asset_id"] in output


def test_norm_mixture_uses_all_samples_without_decoding_videos(tmp_path):
    settings = settings_for(tmp_path)
    sources = g2_config.resolve_sources(settings["data"], base_dir=tmp_path)

    class Raw(RecordedDataset):
        def __init__(self, repo_id, root):
            super().__init__()
            self.meta.info = {}
            if Path(root).name == "b":
                for row in self.hf_dataset:
                    row["state.pose"][[0, 10]] += 20

        def __getitem__(self, index):
            raise AssertionError("normalization must not decode images")

    data = DataConfig(
        repo_id="mixture", asset_id=settings["mixture"]["asset_id"],
        dataset_factory=DatasetFactory(settings["policy"], ("head_color",), sources,
                                       mixture=settings["mixture"], seed=settings["training"]["seed"]),
        repack_transforms=transforms.Group(inputs=[RepackG2(settings["policy"], ("head_color",))]),
        data_transforms=transforms.Group(inputs=[G2Inputs(settings["policy"], "right", ("head_color",))]),
    )
    script = g2_config.PROJECT_ROOT / "scripts/compute_norm_stats.py"
    spec = importlib.util.spec_from_file_location("g2_mixture_norm_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = SimpleNamespace(data=SimpleNamespace(create=lambda *args: data), assets_dirs=tmp_path / "assets",
                             model=SimpleNamespace(action_horizon=3), batch_size=7, num_workers=0)
    with patch("lerobot.common.datasets.lerobot_dataset.LeRobotDataset", Raw):
        loader, batches = module.create_torch_dataloader(data, 3, 7, config.model, 0)
        arrays = [batch["state"] for batch in loader]
        assert sum(len(value) for value in arrays) == 32
        assert batches == 5
        module.compute_norm_stats(config)
        capped, batches = module.create_torch_dataloader(data, 3, 7, config.model, 0, max_frames=3)
        assert batches == 1
        assert sum(len(batch["state"]) for batch in capped) == 3
    result = json.loads((config.assets_dirs / data.asset_id / "norm_stats.json").read_text())
    np.testing.assert_allclose(result["norm_stats"]["state"]["mean"][0], 56.5)
    np.testing.assert_allclose(result["norm_stats"]["state"]["mean"], np.concatenate(arrays).mean(axis=0), atol=1e-5)
