from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_pipeline import _check_dataset
from train_pipeline import _norm_signature
from train_pipeline import _parse_argv
from train_pipeline import _resolve
from train_pipeline import main
from train_pipeline import normalize_child_status

from openpi.adapters.g2 import config as g2_config


def resolve(argv):
    args, _ = _parse_argv(argv)
    with patch.dict(os.environ, {}, clear=True):
        return _resolve(args)


def test_yaml_defaults_and_cli_override():
    settings = resolve(["--batch-size", "3", "--num-train-steps", "40000"])
    assert settings["experiment"]["name"] == "Sing_TeaPi"
    assert settings["training"]["batch_size"] == 3
    assert settings["training"]["num_train_steps"] == 40000
    assert settings["checkpoints"]["overwrite"] is False


def test_smoke_overrides_yaml_but_explicit_cli_wins():
    settings = resolve(["--smoke"])
    assert settings["training"]["batch_size"] == 1
    assert settings["training"]["num_train_steps"] == 20
    assert settings["experiment"]["name"].startswith("g2_vr_smoke_")
    settings = resolve(["--smoke", "--batch-size", "2", "--num-train-steps", "5", "--exp-name", "my_smoke"])
    assert settings["training"]["batch_size"] == 2
    assert settings["training"]["num_train_steps"] == 5
    assert settings["experiment"]["name"] == "my_smoke"


def test_custom_yaml_merges_without_repeating_all_fields(tmp_path):
    config = tmp_path / "train.yaml"
    config.write_text("experiment:\n  name: new_run\ntraining:\n  batch_size: 2\n")
    settings = resolve(["--config", str(config)])
    assert settings["experiment"]["name"] == "new_run"
    assert settings["training"]["batch_size"] == 2
    assert settings["training"]["num_train_steps"] == 50000


@pytest.mark.parametrize("text", ["training:\n  batch_szie: 2\n", "training:\n  batch_size: 2\n  batch_size: 3\n"])
def test_rejects_typo_and_duplicate_yaml_keys(tmp_path, text):
    config = tmp_path / "bad.yaml"
    config.write_text(text)
    with pytest.raises(ValueError, match="Unknown|Duplicate"):
        resolve(["--config", str(config)])


def test_right_arm_and_resume():
    settings = resolve(["--arm", "right", "--resume", "--overwrite"])
    assert g2_config.config_name(settings) == "pi05_g2_vr_low_mem_right"
    assert g2_config.selected_cameras(settings) == ("head_color", "hand_right_color")
    assert settings["checkpoints"]["resume"]
    assert not settings["checkpoints"]["overwrite"]
    with pytest.raises(ValueError, match="disagrees"):
        resolve(["--arm", "left", "--config-name", "pi05_g2_vr_right"])


def test_runtime_environment_overrides_yaml():
    args, _ = _parse_argv([])
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.7"}, clear=True):
        settings = _resolve(args)
    assert settings["runtime"]["gpu"] == "0,1"
    assert settings["runtime"]["mem_fraction"] == 0.7


def test_dry_run_does_not_check_data_or_launch_children():
    out = StringIO()
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("train_pipeline._check_dataset") as check,
        patch("train_pipeline._run_command") as run,
        redirect_stdout(out),
    ):
        main(["--dry-run", "--repo-id", "does/not/exist"])
    check.assert_not_called()
    run.assert_not_called()
    assert "does/not/exist" in out.getvalue()


def test_shell_smoke_entry_uses_yaml_not_old_shell_defaults(tmp_path):
    # Capture the launcher command without starting training or relying on missing data.
    root = tmp_path / "project"
    entry = root / "scripts/g2/train/train_smoke.sh"
    entry.parent.mkdir(parents=True)
    entry.write_text((g2_config.PROJECT_ROOT / "scripts/g2/train/train_smoke.sh").read_text())
    (root / "scripts/g2/_env.sh").write_text("")
    interpreter = root / ".venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    interpreter.chmod(0o755)
    result = subprocess.run(
        ["bash", str(entry)],
        cwd=root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == [
        "scripts/g2/train/train_pipeline.py",
        "--config",
        str(root / "scripts/g2/train/config.yaml"),
        "--smoke",
    ]


def test_dataset_check_uses_cli_resolved_path_and_camera_selection(tmp_path):
    config = tmp_path / "head.yaml"
    config.write_text("data:\n  cameras: [head_color]\n")
    settings = resolve(["--config", str(config), "--data-root", str(tmp_path), "--repo-id", "other/dataset"])
    meta = tmp_path / "other/dataset/meta"
    meta.mkdir(parents=True)
    features = {name: {} for name in ("state.pose", "action.pose", "observation.images.head_color")}
    (meta / "info.json").write_text(json.dumps({"codebase_version": "v2.1", "features": features, "fps": 30.0}))
    (meta / "episodes_stats.jsonl").touch()
    _check_dataset(settings)
    del features["action.pose"]
    (meta / "info.json").write_text(json.dumps({"codebase_version": "v2.1", "features": features, "fps": 30.0}))
    with pytest.raises(ValueError, match="action.pose"):
        _check_dataset(settings)


def test_changed_action_settings_invalidate_auto_norm(tmp_path):
    settings = resolve(["--assets-base-dir", str(tmp_path)])
    norm = tmp_path / g2_config.config_name(settings) / settings["data"]["repo_id"] / "norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text("{}")
    signature = norm.with_name("g2_norm_config.yaml")
    signature.write_text(yaml.safe_dump(_norm_signature(settings)))
    out = StringIO()
    with patch.dict(os.environ, {}, clear=True), redirect_stdout(out):
        main(["--dry-run", "--assets-base-dir", str(tmp_path), "--action-mode", "absolute"])
    assert "skip=False" in out.getvalue()


@pytest.mark.parametrize(("return_code", "expected"), [(0, 0), (130, 130), (-2, 130), (-15, 143)])
def test_normalize_child_status(return_code, expected):
    assert normalize_child_status(return_code) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fps", 0.0),
        ("pose_frame", "world"),
        ("rotation", "quaternion"),
        ("action_supervision", "state"),
    ],
)
def test_dataset_contract_rejects_incompatible_declarations(tmp_path, field, value):
    settings = resolve(["--data-root", str(tmp_path)])
    meta = tmp_path / settings["data"]["repo_id"] / "meta"
    meta.mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "fps": 30.0,
        "features": {
            name: {}
            for name in (
                "state.pose",
                "action.pose",
                "observation.images.head_color",
                "observation.images.hand_right_color",
            )
        },
        "wholebody_export": {"pose_frame": "base_link", "rotation": "rot6d", "action_supervision": "command"},
    }
    if field == "fps":
        info[field] = value
    else:
        info["wholebody_export"][field] = value
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "episodes_stats.jsonl").touch()
    with pytest.raises(ValueError, match="match|must"):
        _check_dataset(settings)


def test_absolute_eef_state_only_requires_pose_column(tmp_path):
    settings = resolve(["--data-root", str(tmp_path), "--action-representation", "absolute"])
    settings["data"]["cameras"] = ["head_color"]
    meta = tmp_path / settings["data"]["repo_id"] / "meta"
    meta.mkdir(parents=True)
    features = {name: {} for name in ("state.pose", "action.pose", "observation.images.head_color")}
    (meta / "info.json").write_text(json.dumps({"codebase_version": "v2.1", "features": features, "fps": 30.0}))
    (meta / "episodes_stats.jsonl").touch()
    _check_dataset(settings)
    settings["policy"]["state"]["space"] = "joint"
    with pytest.raises(ValueError, match="state.joint"):
        _check_dataset(settings)


def test_state_changes_invalidate_norm_cache_but_execution_length_does_not(tmp_path):
    args = ["--assets-base-dir", str(tmp_path), "--action-representation", "absolute"]
    settings = resolve(args)
    norm = tmp_path / g2_config.config_name(settings) / settings["data"]["repo_id"] / "norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text("{}")
    norm.with_name("g2_norm_config.yaml").write_text(yaml.safe_dump(_norm_signature(settings)))
    for extra, expected in ((["--execution-steps", "10"], "skip=True"), (["--state-space", "joint"], "skip=False")):
        out = StringIO()
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(out):
            main([*args, *extra, "--dry-run"])
        assert expected in out.getvalue()


@pytest.mark.parametrize("legacy", [False, True])
def test_resume_preserves_semantics_and_allows_execution_override(tmp_path, legacy):
    args = [
        "--checkpoint-base-dir",
        str(tmp_path / "checkpoints"),
        "--assets-base-dir",
        str(tmp_path / "assets"),
        "--resume",
        "--skip-norm",
    ]
    settings = resolve(args)
    root = tmp_path / "checkpoints" / g2_config.config_name(settings) / settings["experiment"]["name"]
    root.mkdir(parents=True)
    # Store resolved absolute paths just as train.py does.
    saved = settings.copy()
    if legacy:
        import copy

        saved = copy.deepcopy(settings)
        saved.pop("schema_version")
        policy = saved.pop("policy")
        saved["data"].update(action_space="pose", action_mode="relative")
        saved["model"]["action_horizon"] = policy["timing"]["prediction_steps"]
    (root / g2_config.CHECKPOINT_CONFIG).write_text(yaml.safe_dump({"settings": saved, "extra_args": []}))
    norm = tmp_path / "assets" / g2_config.config_name(settings) / settings["data"]["repo_id"] / "norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text("{}")
    with patch("train_pipeline._check_dataset"), patch("train_pipeline._run_command") as run:
        with patch.dict(os.environ, {}, clear=True):
            main([*args, "--execution-steps", "10"])
        assert run.call_count == 1  # mocked worker, no actual training
    with patch("train_pipeline._check_dataset"), patch("train_pipeline._run_command") as run:
        with patch.dict(os.environ, {}, clear=True), pytest.raises(ValueError, match="policy semantics"):
            main([*args, "--prediction-steps", "16"])
        run.assert_not_called()
