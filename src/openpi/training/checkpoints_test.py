from pathlib import Path

import pytest

from openpi.training.checkpoints import find_norm_stats_dir
from openpi.training.checkpoints import load_norm_stats


def _write_norm_stats(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "norm_stats.json").write_text('{"norm_stats": {}}')


def test_find_norm_stats_dir_prefers_asset_id(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    _write_norm_stats(assets / "g2_vr_lerobot_v21")
    _write_norm_stats(assets / "vr" / "Sing_TeaPi" / "lerobot_v21")

    found = find_norm_stats_dir(assets, "g2_vr_lerobot_v21")
    assert Path(str(found)) == assets / "g2_vr_lerobot_v21"


def test_find_norm_stats_dir_nested_repo_id_when_config_default_missing(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    nested = assets / "vr" / "Sing_TeaPi" / "lerobot_v21"
    _write_norm_stats(nested)

    found = find_norm_stats_dir(assets, "g2_vr_lerobot_v21")
    assert Path(str(found)) == nested
    loaded = load_norm_stats(assets, "g2_vr_lerobot_v21")
    assert loaded == {}


def test_find_norm_stats_dir_ambiguous_copies_raise(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    _write_norm_stats(assets / "dataset_a")
    _write_norm_stats(assets / "dataset_b")

    with pytest.raises(FileNotFoundError, match="Multiple copies"):
        find_norm_stats_dir(assets, "g2_vr_lerobot_v21")


def test_save_state_captures_yaml_settings_with_checkpoint_assets(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import yaml

    from openpi.training import checkpoints
    from openpi.training.g2_config import CHECKPOINT_CONFIG

    snapshot = {"settings": {"model": {"action_horizon": 30}}, "extra_args": []}
    saved = {}

    def save(step, items):
        saved["step"] = step
        items["assets"](tmp_path)

    monkeypatch.setattr(checkpoints, "_split_params", lambda state: (state, {}))
    loader = SimpleNamespace(data_config=lambda: SimpleNamespace(norm_stats=None, asset_id="test"))
    checkpoints.save_state(SimpleNamespace(save=save), object(), loader, 5000, g2_training_config=snapshot)
    assert saved["step"] == 5000
    assert yaml.safe_load((tmp_path / CHECKPOINT_CONFIG).read_text()) == snapshot
