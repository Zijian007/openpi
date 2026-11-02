"""Automatic experiment names must identify the resolved policy and preserve resume identity."""

import copy
import datetime as dt
from pathlib import Path
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

from train_pipeline import _parse_argv
from train_pipeline import _resolve
from train_pipeline import _space_name
from train_pipeline import main

from openpi.adapters.g2 import config as g2_config


@pytest.fixture
def experiment(tmp_path):
    config = tmp_path / "ChaPai_layer3.yaml"
    config.write_text("experiment:\n  name: auto\n")
    with patch("train_pipeline.dt") as clock:
        clock.datetime.now.return_value = dt.datetime(2026, 10, 31, 14, 30, tzinfo=ZoneInfo("Asia/Taipei"))
        yield config, tmp_path / "checkpoints"


def resolve(config, root, *flags):
    args, _ = _parse_argv([
        "--config", str(config), "--checkpoint-base-dir", str(root), "--no-resume", *flags,
    ])
    return _resolve(args)


def test_auto_name_uses_config_stem_and_resolved_arm(experiment):
    config, root = experiment
    settings = resolve(config, root, "--arm", "left")
    assert settings["experiment"]["name"] == "chapai_layer3_left_spose_abs_apose_dcomp_20261031_1430_v01"
    assert not root.exists()


def test_template_versions_skip_existing_runs_and_preview_does_not_reserve(experiment):
    config, root = experiment
    template = "chapai_{arm}_{state}_{action}_{date}_{version}"
    for version in (1, 3):
        (root / f"chapai_right_spose_abs_apose_dcomp_20261031_1430_v{version:02d}").mkdir(parents=True)
    (root / "chapai_right_spose_abs_apose_dcomp_20261030_1429_v99").mkdir()
    (root / "chapai_right_spose_abs_apose_dcomp_20261031_1429_v99").mkdir()
    (root / "chapai_right_spose_abs_apose_dcomp_20261031_v99").mkdir()
    second = resolve(config, root, "--exp-name", template)
    assert second["experiment"]["name"] == "chapai_right_spose_abs_apose_dcomp_20261031_1430_v04"
    assert resolve(config, root, "--exp-name", template)["experiment"] == second["experiment"]
    assert not (root / second["experiment"]["name"]).exists()


def test_resume_requires_concrete_name_but_cli_override_is_supported(experiment):
    config, root = experiment
    with pytest.raises(ValueError, match="Resume requires an explicit"):
        resolve(config, root, "--resume")
    settings = resolve(config, root)
    name = settings["experiment"]["name"]
    assert resolve(config, root, "--resume", "--exp-name", name)["experiment"]["name"] == name
    old_name = "chapai_right_spose_abs_apose_dcomp_20261031_v04"
    assert resolve(config, root, "--resume", "--exp-name", old_name)["experiment"]["name"] == old_name


@pytest.mark.parametrize("name", ["{unknown}", "{arm.__class__}", "{date:8}", "{arm!r}", "run_{", "run_}"])
def test_invalid_templates_fail_before_training(experiment, name):
    config, root = experiment
    with pytest.raises(ValueError, match="Invalid experiment.name template"):
        resolve(config, root, "--exp-name", name)


def test_policy_names_distinguish_delta_methods_and_joint_positions():
    policy = copy.deepcopy(g2_config.load_settings()["policy"])
    assert _space_name(policy, "state") == "spose_abs"
    assert _space_name(policy, "action") == "apose_dcomp"
    policy["action"]["pose"]["method"] = "se3"
    assert _space_name(policy, "action") == "apose_dse3"
    policy["action"]["pose"]["delta"] = False
    assert _space_name(policy, "action") == "apose_abs"
    policy["action"]["pose"]["enabled"] = False
    policy["action"]["joint"].update(enabled=True, quantity="position")
    assert _space_name(policy, "action") == "ajoint_abs"
    policy["state"]["joint"]["enabled"] = True
    assert _space_name(policy, "state") == "spose_abs_sjoint_abs"


def test_dry_run_prints_concrete_name_without_writing_or_training(experiment, capsys):
    config, root = experiment
    with patch("train_pipeline._check_dataset") as check, patch("train_pipeline._run_command") as run:
        main(["--config", str(config), "--checkpoint-base-dir", str(root), "--dry-run"])
    output = capsys.readouterr().out
    assert "name: chapai_layer3_right_spose_abs_apose_dcomp_20261031_1430_v01" in output
    assert "name: auto" not in output
    assert not root.exists()
    check.assert_not_called()
    run.assert_not_called()


def test_smoke_keeps_separate_identity_and_manual_names_still_work(experiment):
    config, root = experiment
    assert resolve(config, root, "--smoke")["experiment"]["name"].startswith("g2_vr_smoke_")
    assert resolve(config, root, "--exp-name", "my_existing_run")["experiment"]["name"] == "my_existing_run"
