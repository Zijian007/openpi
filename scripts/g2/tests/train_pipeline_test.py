from contextlib import redirect_stdout
import copy
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

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


@pytest.mark.parametrize("legacy_flags", [False, True])
def test_cli_action_taxonomy_overrides_target_and_encoding_independently(legacy_flags):
    settings = resolve(
        [
            "--action-source",
            "measured",
            "--action-method" if legacy_flags else "--action-spatial",
            "se3_body",
            "--action-anchor",
            "episode_start",
            "--action-strategy" if legacy_flags else "--action-reference-scope",
            "previous_target" if legacy_flags else "step",
        ]
    )
    preview = g2_config.describe_policy(settings)
    assert preview["label_column"] == "state.pose"
    assert preview["action"] == {
        "target": {"quantity": "eef_pose", "source": "measured"},
        "spatial": {"mode": "relative", "method": "se3_body"},
        "temporal": {"anchor": "episode_start", "reference_scope": "step"},
    }
    assert preview["gripper_encoding"] == "absolute_close_ratio"


def test_cli_absolute_quantity_clears_inherited_relative_encoding():
    settings = resolve(["--action-quantity", "joint_velocity", "--action-mode", "absolute"])
    assert g2_config.describe_policy(settings)["label_column"] == "action.joint_velocity"
    assert settings["policy"]["action"]["space"] == "joint"
    action = g2_config.public_settings(settings)["policy"]["action"]
    assert action["spatial"] == {"mode": "absolute", "method": None}
    assert action["temporal"] == {"anchor": None, "reference_scope": None}


def test_new_cli_action_fields_take_precedence_over_legacy_encoding_selector():
    settings = resolve(["--action-encoding", "hybrid_delta", "--action-mode", "absolute"])
    assert settings["policy"]["action"]["representation"] == "absolute"
    settings = resolve(["--action-representation", "component_delta", "--action-method", "se3_spatial"])
    assert settings["policy"]["action"]["representation"] == "se3_spatial_delta"


def test_cli_relative_mode_requires_an_explicit_spatial_method():
    with pytest.raises(ValueError, match="method is required"):
        resolve(["--action-mode", "relative"])


def test_cli_state_encoding_uses_shared_coordinates_and_does_not_change_actions():
    settings = resolve(["--state-mode", "relative", "--state-spatial", "se3_body", "--state-anchor", "episode_start"])
    preview = g2_config.describe_policy(settings)
    assert preview["state"] == {
        "space": "eef",
        "spatial": {"mode": "relative", "method": "se3_body"},
        "temporal": {"anchor": "episode_start"},
    }
    assert preview["pose"]["parent_frame"] == "base_link"
    assert preview["pose"]["ee_frame"] == "flange"
    assert settings["policy"]["action"] == resolve([])["policy"]["action"]
    absolute = resolve(["--state-mode", "absolute"])
    assert absolute["policy"]["state"]["relative_to"] is None


@pytest.mark.parametrize("scope_args", [["--action-reference-scope", "chunk"], ["--action-reference-scope", "step"]])
def test_new_cli_reference_scope_takes_precedence_over_legacy_strategy(scope_args):
    settings = resolve([*scope_args, "--action-strategy", "previous_target"])
    assert settings["policy"]["action"]["delta_scope"] == scope_args[1]


@pytest.mark.parametrize("text", ["training:\n  batch_szie: 2\n", "training:\n  batch_size: 2\n  batch_size: 3\n"])
def test_rejects_typo_and_duplicate_yaml_keys(tmp_path, text):
    config = tmp_path / "bad.yaml"
    config.write_text(text)
    with pytest.raises(ValueError, match="Unknown|Duplicate"):
        resolve(["--config", str(config)])


@pytest.mark.parametrize(
    ("preset", "config_name"),
    [
        ("low_mem", "pi05_g2_vr_low_mem_right"),
        ("vlm_lora_action_full", "pi05_g2_vr_vlm_lora_action_full_right"),
    ],
)
def test_right_arm_and_resume(preset, config_name):
    settings = resolve(["--preset", preset, "--arm", "right", "--resume", "--overwrite"])
    assert g2_config.preset_name(settings) == config_name
    assert g2_config.selected_cameras(settings) == ("head_color", "hand_right_color")
    assert settings["checkpoints"]["resume"]
    assert not settings["checkpoints"]["overwrite"]


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


@pytest.mark.parametrize(
    ("launcher", "mode_args"),
    [
        ("train_pipeline.sh", []),
        ("train_smoke.sh", ["--smoke"]),
        ("train_full.sh", ["--full"]),
        ("compute_norm_stats.sh", ["--norm-only"]),
    ],
)
@pytest.mark.parametrize("custom_config", [False, True])
def test_shell_training_entries_forward_selected_yaml_and_overrides(tmp_path, launcher, mode_args, custom_config):
    # Capture the launcher command without starting training or relying on missing data.
    root = tmp_path / "project"
    entry = root / "scripts/g2/train" / launcher
    entry.parent.mkdir(parents=True)
    entry.write_text((g2_config.PROJECT_ROOT / "scripts/g2/train" / launcher).read_text())
    (root / "scripts/g2/_env.sh").write_text((g2_config.PROJECT_ROOT / "scripts/g2/_env.sh").read_text())
    interpreter = root / ".venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    interpreter.chmod(0o755)
    cli_args = (
        ["--config", str(tmp_path / "custom experiment.yaml"), "--batch-size", "3", "--dry-run"]
        if custom_config
        else []
    )
    result = subprocess.run(
        ["bash", str(entry), *cli_args],
        cwd=tmp_path,
        env={
            **os.environ,
            "HOME": str(tmp_path / "home"),
            "OPENPI_DATA_HOME": str(tmp_path / "cache"),
            "JAX_COMPILATION_CACHE_DIR": str(tmp_path / "jax_cache"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == [
        "scripts/g2/train/train_pipeline.py",
        "--config",
        "scripts/g2/configs/train.yaml",
        *mode_args,
        *cli_args,
    ]


def test_shell_custom_yaml_preview_from_outside_repo(tmp_path):
    config = tmp_path / "custom experiment.yaml"
    config.write_text("experiment:\n  name: selected_yaml\ntraining:\n  batch_size: 3\n")
    result = subprocess.run(
        [
            "bash",
            str(g2_config.PROJECT_ROOT / "scripts/g2/train/train_pipeline.sh"),
            "--config",
            str(config),
            "--dry-run",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "HOME": str(tmp_path / "home"),
            "OPENPI_DATA_HOME": str(tmp_path / "cache"),
            "JAX_COMPILATION_CACHE_DIR": str(tmp_path / "jax_cache"),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"YAML: {config}" in result.stdout
    assert "name: selected_yaml" in result.stdout
    assert "batch_size: 3" in result.stdout
    assert "[dry-run] No data access" in result.stdout


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
    norm = tmp_path / g2_config.preset_name(settings) / settings["data"]["repo_id"] / "norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text("{}")
    signature = norm.with_name("g2_norm_config.yaml")
    signature.write_text(yaml.safe_dump(_norm_signature(settings)))
    out = StringIO()
    with patch.dict(os.environ, {}, clear=True), redirect_stdout(out):
        main(["--dry-run", "--assets-base-dir", str(tmp_path), "--action-representation", "absolute"])
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


def test_geometric_combined_state_requires_recorded_object_joint_reference(tmp_path):
    settings = resolve(
        [
            "--data-root",
            str(tmp_path),
            "--state-space",
            "both",
            "--state-spatial",
            "se3_body",
            "--state-anchor",
            "object",
        ]
    )
    settings["data"]["cameras"] = ["head_color"]
    meta = tmp_path / settings["data"]["repo_id"] / "meta"
    meta.mkdir(parents=True)
    features = {
        name: {}
        for name in (
            "state.pose",
            "state.joint",
            "action.pose",
            "reference.object_pose",
            "observation.images.head_color",
        )
    }
    (meta / "info.json").write_text(json.dumps({"codebase_version": "v2.1", "features": features, "fps": 30.0}))
    (meta / "episodes_stats.jsonl").touch()
    with pytest.raises(ValueError, match="reference.object_joint"):
        _check_dataset(settings)


def test_state_changes_invalidate_norm_cache_but_execution_length_does_not(tmp_path):
    args = ["--assets-base-dir", str(tmp_path), "--action-representation", "absolute"]
    settings = resolve(args)
    norm = tmp_path / g2_config.preset_name(settings) / settings["data"]["repo_id"] / "norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text("{}")
    norm.with_name("g2_norm_config.yaml").write_text(yaml.safe_dump(_norm_signature(settings)))
    for extra, expected in ((["--log-interval", "10"], "skip=True"), (["--state-space", "joint"], "skip=False")):
        out = StringIO()
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(out):
            main([*args, *extra, "--dry-run"])
        assert expected in out.getvalue()


@pytest.mark.parametrize("snapshot_format", ["v1", "v2", "v4_legacy", "v4"])
def test_resume_preserves_semantics_across_snapshot_formats(tmp_path, snapshot_format):
    args = [
        "--checkpoint-base-dir",
        str(tmp_path / "checkpoints"),
        "--assets-base-dir",
        str(tmp_path / "assets"),
        "--resume",
        "--skip-norm",
    ]
    settings = resolve(args)
    root = tmp_path / "checkpoints" / g2_config.preset_name(settings) / settings["experiment"]["name"]
    root.mkdir(parents=True)
    # Store resolved absolute paths just as train.py does.
    saved = copy.deepcopy(settings)
    if snapshot_format == "v1":
        saved.pop("schema_version")
        policy = saved.pop("policy")
        saved["data"].update(action_space="pose", action_mode="relative")
        saved["model"]["action_horizon"] = policy["timing"]["action_horizon"]
    elif snapshot_format in ("v4", "v4_legacy"):
        saved = g2_config.public_settings(settings)
        if snapshot_format == "v4_legacy":
            saved["policy"]["state"] = {"space": "eef", "encoding": "absolute", "reference": None}
            saved["policy"]["action"].pop("spatial")
            saved["policy"]["action"].pop("temporal")
            saved["policy"]["action"]["encoding"] = {
                "mode": "relative",
                "method": "component_difference",
                "reference": {"anchor": "current_measured", "strategy": "fixed"},
            }
    assert _norm_signature(g2_config.upgrade_settings(saved)) == _norm_signature(settings)
    (root / g2_config.CHECKPOINT_CONFIG).write_text(yaml.safe_dump({"settings": saved, "extra_args": []}))
    norm = tmp_path / "assets" / g2_config.preset_name(settings) / settings["data"]["repo_id"] / "norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text("{}")
    with patch("train_pipeline._check_dataset"), patch("train_pipeline._run_command") as run:
        with patch.dict(os.environ, {}, clear=True):
            main(args)
        assert run.call_count == 1  # mocked worker, no actual training
    with patch("train_pipeline._check_dataset"), patch("train_pipeline._run_command") as run:
        with patch.dict(os.environ, {}, clear=True), pytest.raises(ValueError, match="policy semantics"):
            main([*args, "--action-horizon", "16"])
        run.assert_not_called()
    with (
        patch("train_pipeline._check_dataset"),
        patch("train_pipeline._run_command") as run,
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(ValueError, match="policy semantics"),
    ):
        main([*args, "--action-reference-scope", "step"])
    run.assert_not_called()
    with (
        patch("train_pipeline._check_dataset"),
        patch("train_pipeline._run_command") as run,
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(ValueError, match="policy semantics"),
    ):
        main([*args, "--state-spatial", "hybrid", "--state-anchor", "episode_start"])
    run.assert_not_called()


@pytest.mark.parametrize(
    ("current_mode", "previous_mode", "previous_rank", "error"),
    [
        ("frozen", "full", 16, "finetuning modes"),
        ("full", "full", 0, None),
        ("lora", "lora", 4, "model settings"),
    ],
)
def test_resume_checks_freezing_and_only_active_lora_settings(
    tmp_path, current_mode, previous_mode, previous_rank, error
):
    config = tmp_path / "experiment.yaml"
    config.write_text(f"finetuning:\n  vision: {current_mode}\n")
    args = [
        "--config",
        str(config),
        "--checkpoint-base-dir",
        str(tmp_path / "checkpoints"),
        "--assets-base-dir",
        str(tmp_path / "assets"),
        "--resume",
        "--skip-norm",
    ]
    settings = resolve(args)
    previous = copy.deepcopy(settings)
    previous["finetuning"]["vision"] = previous_mode
    previous["model"]["vision_lora_rank"] = previous_rank
    checkpoint = tmp_path / "checkpoints" / g2_config.preset_name(settings) / settings["experiment"]["name"]
    checkpoint.mkdir(parents=True)
    (checkpoint / g2_config.CHECKPOINT_CONFIG).write_text(
        yaml.safe_dump({"settings": g2_config.public_settings(previous), "extra_args": []})
    )
    norm = tmp_path / "assets" / g2_config.preset_name(settings) / settings["data"]["repo_id"] / "norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text("{}")
    with (
        patch("train_pipeline._check_dataset"),
        patch("train_pipeline._run_command") as run,
        patch.dict(os.environ, {}, clear=True),
    ):
        if error:
            with pytest.raises(ValueError, match=error):
                main(args)
            run.assert_not_called()
        else:
            main(args)
            assert run.call_count == 1
