"""Build synthetic wire inputs without connecting to a server or loading weights."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import smoke_infer_client


@pytest.mark.parametrize("zeros", [False, True])
def test_make_observation_supports_default_and_zero_smoke(zeros):
    args = smoke_infer_client.Args(zeros=zeros, prompt="test")
    obs = smoke_infer_client._make_obs(args)  # noqa: SLF001
    assert obs["observation/state"].shape == (20,)
    assert np.isfinite(obs["observation/state"]).all()
    assert obs["prompt"] == "test"
    for key in ("observation/image", "observation/wrist_image_left", "observation/wrist_image_right"):
        assert obs[key].shape == (224, 224, 3)
        assert obs[key].dtype == np.uint8


@pytest.mark.parametrize("flag", ["--config", "--deploy-config"])
@pytest.mark.parametrize(
    ("host", "client_host"), [("0.0.0.0", "127.0.0.1"), ("192.0.2.10", "192.0.2.10"), ("::", "::1")]
)
def test_client_selects_deployment_address(tmp_path, monkeypatch, flag, host, client_host):
    config = tmp_path / "custom deployment.yaml"
    config.write_text(f'server:\n  host: "{host}"\n  port: 8111\n')
    seen = []
    monkeypatch.setattr(smoke_infer_client, "main", lambda args: seen.append(args) or 0)
    assert smoke_infer_client.cli([flag, str(config)]) == 0
    assert seen[0].host == client_host
    assert seen[0].port == 8111
    address = smoke_infer_client._address(client_host, 8111)  # noqa: SLF001
    assert address == ("[::1]:8111" if client_host == "::1" else f"{client_host}:8111")


def test_client_uses_default_yaml_and_allows_connection_overrides(tmp_path, monkeypatch):
    config = tmp_path / "default.yaml"
    config.write_text("server:\n  host: 0.0.0.0\n  port: 8111\n")
    monkeypatch.setattr(smoke_infer_client, "DEFAULT_DEPLOY_CONFIG", config)
    seen = []
    monkeypatch.setattr(smoke_infer_client, "main", lambda args: seen.append(args) or 0)
    smoke_infer_client.cli([])
    assert seen[-1].port == 8111
    smoke_infer_client.cli(["--host", "localhost", "--port", "8123"])
    assert seen[-1].host == "localhost"
    assert seen[-1].port == 8123
