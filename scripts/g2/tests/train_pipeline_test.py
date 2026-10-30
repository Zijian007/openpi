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
    args, _ = _parse_argv(["--no-resume", *argv])
    with patch.dict(os.environ, {}, clear=True):
        return _resolve(args)



def write_resume_checkpoint(root, settings, step=1400, extra_args=()):
    checkpoint = root / str(step)
    for item in ("params", "train_state", "assets"):
        (checkpoint / item).mkdir(parents=True, exist_ok=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text(json.dumps({"commit_timestamp_nsecs": 1}))
    for item in ("params", "train_state"):
        (checkpoint / item / "_METADATA").write_text("{}")
    (checkpoint / "assets" / g2_config.CHECKPOINT_CONFIG).write_text(yaml.safe_dump({
        "settings": g2_config.public_settings(settings), "extra_args": list(extra_args),
    }))
    norm = checkpoint / "assets" / settings["data"]["repo_id"] / "norm_stats.json"
    norm.parent.mkdir(parents=True, exist_ok=True)
    norm.write_text('{"norm_stats": {}}')
    return checkpoint

def test_yaml_defaults_and_cli_override():
    settings = resolve(["--batch-size", "3", "--num-train-steps", "40000"])
    assert settings["experiment"]["name"] == "Sing_TeaPi"
    assert settings["training"]["batch_size"] == 3
    assert settings["training"]["num_train_steps"] == 40000
    assert settings["checkpoints"]["overwrite"] is False


def test_common_cli_and_single_entry_validation():
    with patch.object(g2_config, "validate_settings", wraps=g2_config.validate_settings) as validate:
        settings = resolve(["--steps", "40000", "--learning-rate", "1e-5"])
    validate.assert_called_once()
    assert settings["training"]["num_train_steps"] == 40000
    assert settings["training"]["peak_lr"] == 1e-5


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


def test_runtime_environment_overrides_yaml():
    from train_pipeline import _setup_runtime

    args, _ = _parse_argv([])
    with patch.dict(os.environ, {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.7"}, clear=True):
        settings = _resolve(args)
        assert settings["runtime"]["gpus"] == [3]
        assert settings["runtime"]["mem_fraction"] == 0.7
        _setup_runtime(settings)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=False):
        _setup_runtime(settings)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_dry_run_does_not_check_data_or_launch_children():
    out = StringIO()
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("train_pipeline._check_dataset") as check,
        patch("train_pipeline._run_command") as run,
        redirect_stdout(out),
    ):
        main(["--no-resume", "--dry-run", "--repo-id", "does/not/exist"])
    check.assert_not_called()
    run.assert_not_called()
    assert "does/not/exist" in out.getvalue()


@pytest.mark.parametrize(
    ("launcher", "mode_args", "script_config"),
    [
        ("train_pipeline.sh", [], None),
        ("train_pipeline.sh", [], "scripts/g2/configs/selected experiment.yaml"),
        ("train_smoke.sh", ["--smoke"], None),
        ("train_full.sh", ["--vision-mode", "full", "--language-mode", "full", "--action-expert-mode", "full"], None),
        ("compute_norm_stats.sh", ["--norm-only"], None),
    ],
)
@pytest.mark.parametrize("custom_config", [False, True])
def test_shell_training_entries_forward_selected_yaml_and_overrides(
    tmp_path, launcher, mode_args, script_config, custom_config
):
    # Capture the launcher command without starting training or relying on missing data.
    root = tmp_path / "project"
    entry = root / "scripts/g2/train" / launcher
    entry.parent.mkdir(parents=True)
    script = (g2_config.PROJECT_ROOT / "scripts/g2/train" / launcher).read_text()
    if script_config is not None:
        script = script.replace('G2_TRAIN_CONFIG="scripts/g2/configs/train.yaml"', f'G2_TRAIN_CONFIG="{script_config}"')
    entry.write_text(script)
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
        script_config or "scripts/g2/configs/train.yaml",
        *mode_args,
        *cli_args,
    ]


@pytest.mark.parametrize("config_style", ["flag", "positional", "relative_positional"])
def test_shell_custom_yaml_preview_from_outside_repo(tmp_path, config_style):
    config = tmp_path / "custom experiment.yaml"
    config.write_text("experiment:\n  name: selected_yaml\ntraining:\n  batch_size: 3\n")
    if config_style == "flag":
        config_args = ["--config", str(config)]
    elif config_style == "relative_positional":
        config_args = [os.path.relpath(config, g2_config.PROJECT_ROOT)]
    else:
        config_args = [str(config)]
    result = subprocess.run(
        [
            "bash",
            str(g2_config.PROJECT_ROOT / "scripts/g2/train/train_pipeline.sh"),
            *config_args,
            "--no-resume",
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


@pytest.mark.parametrize("group", ["state", "action"])
def test_dynamic_parent_in_either_pose_group_requires_recorded_transform(tmp_path, group):
    config = tmp_path / "dynamic.yaml"
    config.write_text(
        yaml.safe_dump(
            {"policy": {group: {"pose": {"coordinates": {"parent_frame": "arm_base_link"}}}}}
        )
    )
    settings = resolve(["--config", str(config), "--data-root", str(tmp_path)])
    settings["data"]["cameras"] = ["head_color"]
    meta = tmp_path / settings["data"]["repo_id"] / "meta"
    meta.mkdir(parents=True)
    features = {name: {} for name in ("state.pose", "action.pose", "observation.images.head_color")}
    info = {"codebase_version": "v2.1", "features": features, "fps": 30.0}
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "episodes_stats.jsonl").touch()
    with pytest.raises(ValueError, match="state.arm_base"):
        _check_dataset(settings)
    features["state.arm_base"] = {"shape": [9]}
    (meta / "info.json").write_text(json.dumps(info))
    _check_dataset(settings)


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
    config.write_text(f"backend_options:\n  openpi:\n    finetuning:\n      vision:\n        mode: {current_mode}\n")
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
    previous["finetuning"]["vision"]["mode"] = previous_mode
    previous["finetuning"]["vision"]["lora"]["rank"] = previous_rank
    checkpoint = tmp_path / "checkpoints" / g2_config.training_config_name(settings) / settings["experiment"]["name"]
    checkpoint.mkdir(parents=True)
    write_resume_checkpoint(checkpoint, previous)
    norm = (
        tmp_path / "assets" / g2_config.training_config_name(settings) / settings["data"]["repo_id"] / "norm_stats.json"
    )
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


def test_worker_snapshot_uses_public_tree_for_both_stages(tmp_path):
    captured = []

    def run(command):
        path = Path(command[command.index("--resolved-config") + 1])
        payload = g2_config.read_yaml(path)
        assert payload["settings"]["schema_version"] == 7
        assert payload["settings"]["policy"]["state"].keys() == {"joint", "pose"}
        assert payload["settings"]["policy"]["action"].keys() == {"joint", "pose"}
        captured.append(g2_config.normalize_settings(payload["settings"]))

    with patch("train_pipeline._check_dataset"), patch("train_pipeline._run_command", side_effect=run):
        main(["--no-resume", "--checkpoint-base-dir", str(tmp_path / "checkpoints"), "--assets-base-dir", str(tmp_path / "assets")])
    assert len(captured) == 2
    assert captured[0] == captured[1]


@pytest.mark.parametrize("override", [
    {"action": {"pose": {"reference": {"scope": "step"}}}},
    {"action": {"pose": {"coordinates": {"parent_frame": "arm_base_link"}}}},
])
def test_resume_rejects_changed_branch_semantics_before_launching_worker(tmp_path, override):
    args = [
        "--checkpoint-base-dir",
        str(tmp_path / "checkpoints"),
        "--assets-base-dir",
        str(tmp_path / "assets"),
        "--resume",
        "--skip-norm",
    ]
    settings = resolve(args)
    root = tmp_path / "checkpoints" / g2_config.training_config_name(settings) / settings["experiment"]["name"]
    root.mkdir(parents=True)
    write_resume_checkpoint(root, settings)
    norm = (
        tmp_path / "assets" / g2_config.training_config_name(settings) / settings["data"]["repo_id"] / "norm_stats.json"
    )
    norm.parent.mkdir(parents=True)
    norm.write_text("{}")
    with patch("train_pipeline._check_dataset"), patch("train_pipeline._run_command") as run:
        main(args)
        assert run.call_count == 1
    selected = tmp_path / "step.yaml"
    selected.write_text(yaml.safe_dump({"policy": override}))
    with patch("train_pipeline._check_dataset"), patch("train_pipeline._run_command") as run:
        with pytest.raises(ValueError, match="policy semantics"):
            main([*args, "--config", str(selected)])
        run.assert_not_called()


@pytest.mark.parametrize("override", [
    {"state": {"joint": {"enabled": True}}},
    {"state": {"pose": {"coordinates": {"parent_frame": "arm_base_link"}}}},
])
def test_branch_changes_invalidate_normalization_cache(tmp_path, override):
    args = ["--no-resume", "--assets-base-dir", str(tmp_path), "--norm", "auto"]
    settings = resolve(args)
    norm = tmp_path / g2_config.training_config_name(settings) / settings["data"]["repo_id"] / "norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text("{}")
    norm.with_name("g2_norm_config.yaml").write_text(yaml.safe_dump(_norm_signature(settings)))
    selected = tmp_path / "combined.yaml"
    selected.write_text(yaml.safe_dump({"policy": override}))
    for extra, expected in ((["--log-interval", "10"], "skip=True"), (["--config", str(selected)], "skip=False")):
        out = StringIO()
        with redirect_stdout(out):
            main([*args, *extra, "--dry-run"])
        assert expected in out.getvalue()


@pytest.fixture
def resume_experiment(tmp_path):
    args = ["--resume", "--checkpoint-base-dir", str(tmp_path / "checkpoints"),
            "--assets-base-dir", str(tmp_path / "assets")]
    settings = resolve(args)
    root = Path(settings["checkpoints"]["base_dir"]) / g2_config.training_config_name(settings) / settings["experiment"]["name"]
    checkpoint = write_resume_checkpoint(root, settings)
    return args, settings, checkpoint


def test_resume_force_uses_checkpoint_without_shared_stats(resume_experiment):
    args, settings, checkpoint = resume_experiment
    captured = []
    def run(command):
        assert command[-1] == "train"
        payload = g2_config.read_yaml(Path(command[command.index("--resolved-config") + 1]))
        assert payload["resume_checkpoint"] == str(checkpoint)
        assert payload["settings"]["backend_options"]["openpi"]["training"]["decay_steps"] == settings["training"]["num_train_steps"]
        captured.append(payload)
    with patch("train_pipeline._check_dataset"), patch("train_pipeline._run_command", side_effect=run):
        main([*args, "--num-train-steps", "60000"])
    assert len(captured) == 1
    assert captured[0]["settings"]["training"]["steps"] == 60000


@pytest.mark.parametrize("missing", ["configuration", "statistics", "complete checkpoint"])
def test_resume_missing_assets_never_launches(resume_experiment, missing):
    args, settings, checkpoint = resume_experiment
    target = {
        "configuration": checkpoint / "assets" / g2_config.CHECKPOINT_CONFIG,
        "statistics": checkpoint / "assets" / settings["data"]["repo_id"] / "norm_stats.json",
        "complete checkpoint": checkpoint / "params/_METADATA",
    }[missing]
    target.unlink()
    with patch("train_pipeline._run_command") as run, pytest.raises(FileNotFoundError):
        main(args)
    run.assert_not_called()


@pytest.mark.parametrize(("overrides", "message"), [
    (["--batch-size", "2"], "batch_size"), (["--seed", "99"], "seed"),
    (["--peak-lr", "0.001"], "peak_lr"), (["--warmup-steps", "100"], "warmup_steps"),
    (["--decay-steps", "70000"], "decay_steps"),
    (["--", "--ema-decay", "0.9"], "optimizer/EMA"),
])
def test_resume_rejects_training_definition_changes(resume_experiment, overrides, message):
    args, _, _ = resume_experiment
    with patch("train_pipeline._run_command") as run, pytest.raises(ValueError, match=message):
        main([*args, *overrides])
    run.assert_not_called()


def test_resume_uses_latest_complete_checkpoint_and_pins_it(resume_experiment):
    from openpi.adapters.g2.resume import resolve_resume
    args, settings, checkpoint = resume_experiment
    newer = write_resume_checkpoint(checkpoint.parent, settings, step=1500)
    incomplete = write_resume_checkpoint(checkpoint.parent, settings, step=1600)
    (incomplete / "train_state/_METADATA").unlink()
    _, _, selected, _ = resolve_resume(settings, (), checkpoint.parent)
    assert selected == newer


def test_resume_loads_checkpoint_stats_not_shared_assets(resume_experiment):
    import numpy as np

    from openpi.adapters.g2.resume import resolve_resume
    from openpi.shared import normalize
    _, settings, checkpoint = resume_experiment
    shared = Path(settings["checkpoints"]["assets_dir"]) / g2_config.training_config_name(settings) / settings["data"]["repo_id"]
    for path, mean in ((shared, 99.), (checkpoint / "assets" / settings["data"]["repo_id"], 1.)):
        normalize.save(path, {"state": normalize.NormStats(mean=np.array([mean]), std=np.array([1.]))})
    resolved, extra, selected, state_config = resolve_resume(settings, (), checkpoint.parent)
    cfg = g2_config.build_train_config(resolved, extra, resume_checkpoint=selected, resume_training=state_config)
    assert cfg.resume_step == 1400
    actual = cfg.data.create(cfg.assets_dirs, cfg.model).norm_stats
    assert actual["state"].mean.tolist() == [1.]
    assert normalize.load(shared)["state"].mean.tolist() == [99.]


def test_saved_snapshot_freezes_effective_schedule_and_optimizer(resume_experiment):
    _, settings, _ = resume_experiment
    settings["checkpoints"]["resume"] = False
    cfg = g2_config.build_train_config(settings, ("--optimizer.b1", "0.8"))
    saved = yaml.safe_load(cfg.checkpoint_assets[g2_config.CHECKPOINT_CONFIG])
    assert saved["settings"]["backend_options"]["openpi"]["training"]["decay_steps"] is None
    assert saved["training_state_config"]["lr_schedule"]["decay_steps"] == 50000
    assert saved["training_state_config"]["optimizer"]["b1"] == .8
    cfg2 = g2_config.build_train_config(settings, resume_training=saved["training_state_config"])
    assert cfg2.optimizer.b1 == .8


def test_resume_rejects_already_completed_target(resume_experiment):
    args, _, _ = resume_experiment
    with pytest.raises(ValueError, match="must exceed checkpoint step"):
        main([*args, "--num-train-steps", "1400"])


def test_resume_preview_reads_checkpoint_but_not_dataset(resume_experiment):
    args, _, _ = resume_experiment
    with patch("train_pipeline._check_dataset") as data, patch("train_pipeline._run_command") as run:
        main([*args, "--dry-run"])
    data.assert_not_called()
    run.assert_not_called()


