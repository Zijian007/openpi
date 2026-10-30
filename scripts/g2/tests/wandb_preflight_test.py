"""W&B failures must stop preparation without creating an experiment run."""

import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openpi.shared import wandb_preflight as preflight


@pytest.mark.parametrize(("enabled", "mode"), [(False, None), (True, "offline"), (True, "disabled")])
def test_local_logging_never_contacts_wandb(monkeypatch, enabled, mode):
    run = Mock()
    monkeypatch.setattr(preflight.subprocess, "run", run)
    preflight.check_wandb(enabled=enabled, mode=mode)
    run.assert_not_called()


def test_environment_offline_is_overridden_by_explicit_online_mode(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    run = Mock(return_value=SimpleNamespace(stdout="connected"))
    monkeypatch.setattr(preflight.subprocess, "run", run)
    preflight.check_wandb(enabled=True)
    run.assert_not_called()
    preflight.check_wandb(enabled=True, mode="online")
    assert run.call_args.kwargs["timeout"] == 20
    assert run.call_args.kwargs["stdin"] == subprocess.DEVNULL


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.TimeoutExpired("probe", 20),
        subprocess.CalledProcessError(1, "probe", stderr="AuthenticationError: invalid credentials"),
    ],
)
def test_failed_probe_aborts_with_actionable_error(monkeypatch, failure):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(preflight.subprocess, "run", Mock(side_effect=failure))
    with pytest.raises(RuntimeError, match="W&B preflight failed.*HTTP_PROXY.*tmux.*wandb login"):
        preflight.check_wandb(enabled=True)


def test_probe_errors_do_not_expose_proxy_passwords_or_api_keys(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "secret-api-key")
    error = subprocess.CalledProcessError(
        1,
        "probe",
        stderr="Failure: http://user:password@localhost:7890 secret-api-key",
    )
    monkeypatch.setattr(preflight.subprocess, "run", Mock(side_effect=error))
    with pytest.raises(RuntimeError) as caught:
        preflight.check_wandb(enabled=True, mode="online")
    assert "password" not in str(caught.value)
    assert "secret-api-key" not in str(caught.value)


def test_probe_checks_authenticated_account_without_creating_run(monkeypatch):
    import sys

    wandb = SimpleNamespace(
        sdk=SimpleNamespace(
            wandb_setup=SimpleNamespace(
                _setup=Mock(
                    return_value=SimpleNamespace(
                        settings=SimpleNamespace(mode="online", base_url="https://wandb.test")
                    ),
                )
            )
        ),
        Api=Mock(return_value=SimpleNamespace(viewer=SimpleNamespace(id="user-id", username="test-user"))),
        init=Mock(),
    )
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    preflight.probe_wandb(None)
    wandb.Api.assert_called_once_with(timeout=10, overrides={"base_url": "https://wandb.test"})
    wandb.init.assert_not_called()
    wandb.Api.reset_mock()
    wandb.Api.return_value.viewer = SimpleNamespace(id=None)
    with pytest.raises(RuntimeError, match="authenticated user"):
        preflight.probe_wandb(None)
