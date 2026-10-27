from pathlib import Path

import pytest

from openpi.training.checkpoints import find_norm_stats_dir
from openpi.training.checkpoints import load_norm_stats


def _write_norm_stats(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "norm_stats.json").write_text('{"norm_stats": {}}')


def test_find_norm_stats_dir_prefers_asset_id(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    _write_norm_stats(assets / "default_dataset")
    _write_norm_stats(assets / "vr" / "Sing_TeaPi" / "lerobot_v21")

    found = find_norm_stats_dir(assets, "default_dataset")
    assert Path(str(found)) == assets / "default_dataset"


def test_find_norm_stats_dir_nested_repo_id_when_config_default_missing(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    nested = assets / "vr" / "Sing_TeaPi" / "lerobot_v21"
    _write_norm_stats(nested)

    found = find_norm_stats_dir(assets, "default_dataset")
    assert Path(str(found)) == nested
    loaded = load_norm_stats(assets, "default_dataset")
    assert loaded == {}


def test_find_norm_stats_dir_ambiguous_copies_raise(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    _write_norm_stats(assets / "dataset_a")
    _write_norm_stats(assets / "dataset_b")

    with pytest.raises(FileNotFoundError, match="Multiple copies"):
        find_norm_stats_dir(assets, "default_dataset")


def test_save_state_captures_yaml_settings_with_checkpoint_assets(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from openpi.training import checkpoints

    snapshot = {"model": {"action_horizon": 30}}
    name = "run_config.json"
    saved = {}

    def save(step, items):
        saved["step"] = step
        items["assets"](tmp_path)

    monkeypatch.setattr(checkpoints, "_split_params", lambda state: (state, {}))
    loader = SimpleNamespace(data_config=lambda: SimpleNamespace(norm_stats=None, asset_id="test"))
    checkpoints.save_state(
        SimpleNamespace(save=save), object(), loader, 5000, checkpoint_assets={name: json.dumps(snapshot)}
    )
    assert saved["step"] == 5000
    assert json.loads((tmp_path / name).read_text()) == snapshot


@pytest.mark.parametrize("name", ["../outside.yaml", "/tmp/outside.yaml", "nested/config.yaml", "..", ""])
def test_checkpoint_extra_assets_require_plain_filenames(tmp_path, name):
    from openpi.training.checkpoints import write_checkpoint_assets

    with pytest.raises(ValueError, match="filename"):
        write_checkpoint_assets(tmp_path, {name: "contents"})
