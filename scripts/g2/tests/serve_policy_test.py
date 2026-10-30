import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from openpi.adapters.g2 import config as g2_config

G2_SCRIPTS = Path(__file__).resolve().parents[1]
DEPLOY = G2_SCRIPTS / "deploy"
sys.path.insert(0, str(DEPLOY))

import serve_policy  # noqa: E402


@pytest.mark.parametrize(
    ("launcher", "cli_args"),
    [
        ("serve_policy.sh", ["--no-warmup", "--no-rtc"]),
        ("serve_policy.sh", ["--config", "configs/deploy right.yaml", "--port", "8123"]),
        ("smoke_infer_client.sh", ["--config", "configs/deploy right.yaml", "--port", "8123"]),
    ],
)
def test_shell_launches_adapter_with_optional_dependency(tmp_path, launcher, cli_args):
    root = tmp_path / "project"
    entry = root / "scripts/g2/deploy" / launcher
    entry.parent.mkdir(parents=True)
    entry.write_text((DEPLOY / launcher).read_text())
    (root / "scripts/g2/_env.sh").write_text(f'G2_OPENPI_PROJECT_ROOT="{root}"\n')
    binary = tmp_path / "bin"
    binary.mkdir()
    uv = binary / "uv"
    uv.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "if 'scripts/g2/deploy/serve_policy.py' in sys.argv: assert 'VIRTUAL_ENV' not in os.environ\n"
        "print('LAUNCH=' + json.dumps(sys.argv[1:]))\n"
    )
    uv.chmod(0o755)
    result = subprocess.run(
        ["bash", str(entry), *cli_args],
        cwd=root,
        env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}", "VIRTUAL_ENV": "/other/active-venv"},
        capture_output=True,
        text=True,
        check=True,
    )
    command = json.loads(
        next(line.removeprefix("LAUNCH=") for line in result.stdout.splitlines() if line.startswith("LAUNCH="))
    )
    assert command == ["run", "--extra", "g2", f"scripts/g2/deploy/{Path(launcher).stem}.py", *cli_args]


@pytest.mark.parametrize("arm", ["both", "right"])
def test_joint_state_warmup_preserves_pose_for_single_arm_expansion(arm):
    seen = []

    def infer(obs):
        seen.append(obs.copy())
        return {"actions": np.zeros((50, 20), np.float32)}

    from openpi.adapters.g2.tests.geometry_test import set_state_space

    contract = copy.deepcopy(g2_config.POLICY_DEFAULTS)
    set_state_space(contract, "joint")
    policy = SimpleNamespace(
        metadata={"requires_joint_state": True, "arm": arm, "policy_contract": contract},
        infer=infer,
        reset=lambda: None,
    )
    args = serve_policy.Args(
        warmup=True,
        rtc=False,
        policy=serve_policy.Checkpoint(dir="unused"),
    )
    serve_policy._warmup(policy, args)  # noqa: SLF001
    assert len(seen) == 2
    for obs in seen:
        assert obs["observation/joint"].shape == (16,)
        if arm != "both":
            assert obs["observation/pose"].shape == (20,)


def test_deployment_yaml_restores_checkpoint_without_duplicate_model_selection():
    config = serve_policy.load_deploy_config(g2_config.DEFAULT_DEPLOY_CONFIG)
    assert config.policy.dir
    assert config.port == 8001


def test_deploy_runtime_sets_cuda_without_overriding_the_environment(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    deploy_config = g2_config.DEFAULT_DEPLOY_CONFIG
    serve_policy.load_deploy_config(deploy_config, apply_runtime=True)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ",".join(
        map(str, g2_config.read_yaml(deploy_config)["runtime"]["gpus"])
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    serve_policy.load_deploy_config(deploy_config, apply_runtime=True)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_cli_checkpoint_override_uses_yaml_default_without_a_subcommand():
    import tyro

    defaults = serve_policy.load_deploy_config(g2_config.DEFAULT_DEPLOY_CONFIG)
    args = tyro.cli(serve_policy.Args, args=["--policy.dir", "/fake/20", "--no-warmup"], default=defaults)
    assert args.policy.dir == "/fake/20"
    assert not args.warmup


@pytest.mark.parametrize("flag", ["--config", "--deploy-config"])
@pytest.mark.parametrize("extra_args", [[], ["--port", "8123", "--policy.dir", "/override/20"]])
def test_cli_selects_deployment_yaml_and_applies_overrides(tmp_path, monkeypatch, flag, extra_args):
    raw = g2_config.read_yaml(g2_config.DEFAULT_DEPLOY_CONFIG)
    raw["checkpoint"] = "/selected/50000"
    raw["server"].update(port=8111, warmup=False)
    raw["runtime"]["gpus"] = [5]
    path = tmp_path / "custom deployment.yaml"
    import yaml

    path.write_text(yaml.safe_dump(raw))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    seen = []
    monkeypatch.setattr(serve_policy, "main", seen.append)
    serve_policy.cli([flag, str(path), *extra_args])
    assert len(seen) == 1
    assert seen[0].policy.dir == ("/override/20" if extra_args else "/selected/50000")
    assert seen[0].port == (8123 if extra_args else 8111)
    assert not seen[0].warmup


def test_contract_warmup_omits_undeclared_pose():
    seen = []

    def infer(obs):
        seen.append(obs.copy())
        return {"actions": np.zeros((8, 16), np.float32)}

    from g2_policy_protocol.data.policy import resolve_policy

    contract = resolve_policy(
        {
            "state": {"pose": {"enabled": False}, "joint": {"enabled": True}},
            "action": {"pose": {"enabled": False}, "joint": {"enabled": True}},
        },
        horizon=50,
    )
    policy = SimpleNamespace(
        metadata={
            "requires_joint_state": True,
            "arm": "both",
            "required_cameras": ["head_color"],
            "policy_contract": contract,
            "requires_prompt": True,
        },
        infer=infer,
        reset=lambda: None,
    )
    args = serve_policy.Args(warmup=True, rtc=False, policy=serve_policy.Checkpoint(dir="unused"))
    serve_policy._warmup(policy, args)  # noqa: SLF001
    assert seen
    assert "observation/pose" not in seen[0]
    assert seen[0]["observation/joint"].shape == (16,)
    assert "observation/image" in seen[0]
    assert "observation/wrist_image_left" not in seen[0]


def test_warmup_requires_the_checkpoint_contract():
    with pytest.raises(ValueError, match="policy_contract"):
        serve_policy._warmup_observation(SimpleNamespace(metadata={}))  # noqa: SLF001
