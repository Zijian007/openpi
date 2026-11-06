from types import SimpleNamespace
from unittest.mock import Mock

from openpi.adapters.g2 import wandb_logging


def _config(tmp_path, *, project="G2_openpi", exp_name="exp"):
    checkpoint_dir = tmp_path / "exp" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    return SimpleNamespace(
        checkpoint_dir=checkpoint_dir,
        project_name=project,
        exp_name=exp_name,
    )


def test_experiment_wandb_dir_sits_next_to_checkpoints(tmp_path):
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    assert wandb_logging.experiment_wandb_dir(checkpoint_dir) == tmp_path / "run"


def test_install_converts_must_resume_and_writes_id(tmp_path, monkeypatch):
    config = _config(tmp_path)
    run = SimpleNamespace(id="newid", step=0)
    init = Mock(return_value=run)
    monkeypatch.setattr(wandb_logging, "_original_init", init)
    monkeypatch.setattr(wandb_logging.wandb, "run", run)
    monkeypatch.setattr(wandb_logging, "backfill_history", Mock(return_value=0))
    wandb_logging.uninstall()
    wandb_logging.install(config)
    wandb_logging._patched_init(id="old", resume="must", project="G2_openpi")
    assert init.call_args.kwargs["resume"] == "allow"
    assert init.call_args.kwargs["dir"] == str(tmp_path / "exp")
    assert (config.checkpoint_dir / "wandb_id.txt").read_text() == "newid"
    wandb_logging.uninstall()


def test_resume_failure_starts_a_new_run(tmp_path, monkeypatch):
    config = _config(tmp_path)
    run = SimpleNamespace(id="fresh", step=0)
    init = Mock(side_effect=[RuntimeError("deleted"), run])
    monkeypatch.setattr(wandb_logging, "_original_init", init)
    monkeypatch.setattr(wandb_logging.wandb, "run", run)
    monkeypatch.setattr(wandb_logging, "backfill_history", Mock(return_value=0))
    wandb_logging.uninstall()
    wandb_logging.install(config)
    wandb_logging._patched_init(id="gone", resume="must", project="G2_openpi")
    assert init.call_count == 2
    assert "id" not in init.call_args.kwargs
    assert (config.checkpoint_dir / "wandb_id.txt").read_text() == "fresh"
    wandb_logging.uninstall()


def test_log_remaps_step_zero_after_resume(monkeypatch):
    log = Mock()
    monkeypatch.setattr(wandb_logging, "_original_log", log)
    monkeypatch.setattr(wandb_logging.wandb, "run", SimpleNamespace(id="r", step=3701))
    wandb_logging._patched_log({"camera_views": "img"}, step=0)
    assert log.call_args.kwargs["step"] == 3701
