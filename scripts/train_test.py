import dataclasses
import os
import pathlib
import signal

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from flax import nnx
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.training import config as _config

from . import train


def test_checkpoint_step_matches_completed_step():
    assert (
        train.checkpoint_step_to_save(1, start_step=0, save_interval=5000, num_train_steps=50000, interrupt=False)
        is None
    )
    assert (
        train.checkpoint_step_to_save(5000, start_step=0, save_interval=5000, num_train_steps=50000, interrupt=False)
        == 5000
    )
    assert (
        train.checkpoint_step_to_save(50000, start_step=0, save_interval=5000, num_train_steps=50000, interrupt=False)
        == 50000
    )
    # Final step is kept even when it is not on the save interval.
    assert train.checkpoint_step_to_save(10, start_step=0, save_interval=100, num_train_steps=10, interrupt=False) == 10
    # Resume does not rewrite the step that was just loaded.
    assert (
        train.checkpoint_step_to_save(
            35001, start_step=35001, save_interval=5000, num_train_steps=50000, interrupt=False
        )
        is None
    )
    assert (
        train.checkpoint_step_to_save(
            37300, start_step=35001, save_interval=5000, num_train_steps=50000, interrupt=True
        )
        == 37300
    )
    assert (
        train.checkpoint_step_to_save(
            40000, start_step=35001, save_interval=5000, num_train_steps=50000, interrupt=False
        )
        == 40000
    )


def test_first_interrupt_requests_save_after_this_step():
    stop = train._StopRequest(armed=True)  # noqa: SLF001
    previous = train._install_stop_handlers(stop)  # noqa: SLF001
    try:
        os.kill(os.getpid(), signal.SIGINT)
        assert stop.requested
        assert stop.signum == signal.SIGINT
    finally:
        train._restore_stop_handlers(previous)  # noqa: SLF001


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


class TinyModel(_model.BaseModel):
    def __init__(self):
        super().__init__(2, 2, 1)
        self.kernel = nnx.Param(jnp.full((2, 2), 0.1))

    def compute_loss(self, rng, observation, actions, *, train=False):
        del rng, train
        prediction = (observation.state @ self.kernel.value)[:, None, :]
        return jnp.mean((prediction - actions) ** 2, axis=-1)

    def sample_actions(self, rng, observation, **kwargs):
        del rng, kwargs
        value = (observation.state @ self.kernel.value)[:, None, :]
        return jnp.repeat(value, 2, axis=1)


@dataclasses.dataclass(frozen=True)
class TinyModelConfig(_model.BaseModelConfig):
    """Only used to exercise the real trainer without loading a vision model."""

    action_dim: int = 2
    action_horizon: int = 2
    max_token_len: int = 1

    @property
    def model_type(self):
        return _model.ModelType.PI0

    def create(self, rng):
        del rng

        return TinyModel()

    def inputs_spec(self, *, batch_size=1):
        import jax
        import jax.numpy as jnp

        from openpi.models.model import Observation
        from openpi.shared import array_typing as at

        with at.disable_typechecking():
            return (
                Observation(
                    images={"test": jax.ShapeDtypeStruct((batch_size, 8, 8, 3), jnp.float32)},
                    image_masks={"test": jax.ShapeDtypeStruct((batch_size,), jnp.bool_)},
                    state=jax.ShapeDtypeStruct((batch_size, 2), jnp.float32),
                ),
                jax.ShapeDtypeStruct((batch_size, 2, 2), jnp.float32),
            )


@pytest.fixture
def tiny_train_config(tmp_path):
    from openpi.training import optimizer

    return dataclasses.replace(
        _config.get_config("debug"),
        model=TinyModelConfig(),
        batch_size=2,
        num_workers=0,
        checkpoint_base_dir=str(tmp_path),
        exp_name="tiny",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        save_interval=1,
        log_interval=1,
        ema_decay=None,
        checkpoint_assets={"test_config.txt": "original"},
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=0, peak_lr=0.001, decay_steps=10),
    )


def test_tiny_training_resume_and_failed_restore_preserve_record(tiny_train_config, monkeypatch):
    from openpi.training import checkpoints

    config = tiny_train_config
    train.main(config)
    assert (config.checkpoint_dir / "2/params/_METADATA").is_file()
    resume = dataclasses.replace(
        config, resume=True, resume_step=2, num_train_steps=4, checkpoint_assets={"test_config.txt": "resumed"}
    )

    def fail(*args, **kwargs):
        raise RuntimeError("simulated failed restore")

    with monkeypatch.context() as patcher:
        patcher.setattr(checkpoints, "restore_state", fail)
        with pytest.raises(RuntimeError, match="simulated failed restore"):
            train.main(resume)
    assert (config.checkpoint_dir / "test_config.txt").read_text() == "original"
    train.main(resume)
    assert (config.checkpoint_dir / "4/train_state/_METADATA").is_file()
    assert (config.checkpoint_dir / "test_config.txt").read_text() == "resumed"
    assert (config.checkpoint_dir / "initial_test_config.txt").read_text() == "original"


def test_tiny_interrupt_checkpoint_can_resume(tiny_train_config, monkeypatch):
    config = tiny_train_config
    native_step = train.train_step

    def interrupt_after_step(*args, **kwargs):
        result = native_step(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGINT)
        return result

    with monkeypatch.context() as patcher:
        patcher.setattr(train, "train_step", interrupt_after_step)
        with pytest.raises(SystemExit) as interrupted:
            train.main(config)
    assert interrupted.value.code == 128 + signal.SIGINT
    assert (config.checkpoint_dir / "1/train_state/_METADATA").is_file()
    train.main(dataclasses.replace(config, resume=True, resume_step=1, num_train_steps=3))
    assert (config.checkpoint_dir / "3/train_state/_METADATA").is_file()
