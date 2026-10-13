from pathlib import Path

import pytest

from openpi.training.checkpoints import find_norm_stats_dir, load_norm_stats


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
