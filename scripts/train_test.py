import dataclasses
import os
import pathlib
import signal

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

from . import train


def test_checkpoint_step_matches_completed_step():
    assert train.checkpoint_step_to_save(1, start_step=0, save_interval=5000, num_train_steps=50000, interrupt=False) is None
    assert train.checkpoint_step_to_save(5000, start_step=0, save_interval=5000, num_train_steps=50000, interrupt=False) == 5000
    assert train.checkpoint_step_to_save(50000, start_step=0, save_interval=5000, num_train_steps=50000, interrupt=False) == 50000
    # Final step is kept even when it is not on the save interval.
    assert train.checkpoint_step_to_save(10, start_step=0, save_interval=100, num_train_steps=10, interrupt=False) == 10
    # Resume does not rewrite the step that was just loaded.
    assert (
        train.checkpoint_step_to_save(35001, start_step=35001, save_interval=5000, num_train_steps=50000, interrupt=False)
        is None
    )
    assert (
        train.checkpoint_step_to_save(37300, start_step=35001, save_interval=5000, num_train_steps=50000, interrupt=True)
        == 37300
    )
    assert (
        train.checkpoint_step_to_save(40000, start_step=35001, save_interval=5000, num_train_steps=50000, interrupt=False)
        == 40000
    )


def test_first_interrupt_requests_save_after_this_step():
    stop = train._StopRequest(armed=True)
    previous = train._install_stop_handlers(stop)
    try:
        os.kill(os.getpid(), signal.SIGINT)
        assert stop.requested
        assert stop.signum == signal.SIGINT
    finally:
        train._restore_stop_handlers(previous)


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
