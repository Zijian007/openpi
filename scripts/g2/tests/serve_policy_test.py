import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

G2_SCRIPTS = Path(__file__).resolve().parents[1]
DEPLOY = G2_SCRIPTS / "deploy"
sys.path.insert(0, str(DEPLOY))

import serve_policy


def test_shell_launches_adapter_with_optional_dependency(tmp_path):
    root = tmp_path / "project"
    entry = root / "scripts/g2/deploy/serve_policy.sh"
    entry.parent.mkdir(parents=True)
    entry.write_text((DEPLOY / "serve_policy.sh").read_text())
    (root / "scripts/g2/_env.sh").write_text(f'G2_OPENPI_PROJECT_ROOT="{root}"\n')
    binary = tmp_path / "bin"
    binary.mkdir()
    uv = binary / "uv"
    uv.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                  "assert 'VIRTUAL_ENV' not in os.environ\n"
                  "print('LAUNCH=' + json.dumps(sys.argv[1:]))\n")
    uv.chmod(0o755)
    result = subprocess.run(
        ["bash", str(entry), "--no-warmup", "--no-rtc"],
        cwd=root,
        env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}", "VIRTUAL_ENV": "/other/active-venv"},
        capture_output=True,
        text=True,
        check=True,
    )
    command = json.loads(
        next(line.removeprefix("LAUNCH=") for line in result.stdout.splitlines() if line.startswith("LAUNCH="))
    )
    assert command == ["run", "--extra", "g2", "scripts/g2/deploy/serve_policy.py", "--no-warmup", "--no-rtc"]


@pytest.mark.parametrize("arm", ["both", "right"])
def test_joint_state_warmup_preserves_pose_for_single_arm_expansion(arm):
    seen = []

    def infer(obs):
        seen.append(obs.copy())
        return {"actions": np.zeros((50, 20), np.float32)}

    policy = SimpleNamespace(metadata={"requires_joint_state": True, "arm": arm}, infer=infer, reset=lambda: None)
    args = serve_policy.Args(
        warmup=True,
        rtc=False,
        policy=serve_policy.Checkpoint(config="pi05_g2_vr_low_mem_right", dir="unused"),
    )
    serve_policy._warmup(policy, args)  # noqa: SLF001
    assert len(seen) == 2
    for obs in seen:
        assert obs["observation/state"].shape == (16,)
        if arm != "both":
            assert obs["observation/pose"].shape == (20,)


def test_deployment_yaml_restores_checkpoint_without_duplicate_model_selection():
    config = serve_policy.load_deploy_config(DEPLOY / "config.yaml")
    assert config.policy.dir and config.policy.config is None
    assert config.port == 8001


def test_deploy_runtime_sets_cuda_without_overriding_the_environment(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    serve_policy.load_deploy_config(DEPLOY / "config.yaml", apply_runtime=True)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    serve_policy.load_deploy_config(DEPLOY / "config.yaml", apply_runtime=True)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_cli_checkpoint_override_uses_yaml_default_without_a_subcommand():
    import tyro

    defaults = serve_policy.load_deploy_config(DEPLOY / "config.yaml")
    args = tyro.cli(serve_policy.Args, args=["--policy.dir", "/fake/20", "--no-warmup"], default=defaults)
    assert args.policy.dir == "/fake/20" and not args.warmup


def test_contract_warmup_omits_undeclared_pose():
    seen = []

    def infer(obs):
        seen.append(obs.copy())
        return {"actions": np.zeros((8, 16), np.float32)}

    contract = {
        "state": {"space": "joint", "relative_to": None},
        "action": {"space": "joint", "control_quantity": "joint_position", "relative_to": "current_measured_pose"},
        "pose": {"base_to_parent": None},
        "timing": {"history_steps": 1, "sample_hz": 30.0, "stride_steps": 1, "target_offset_steps": 0},
    }
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
    assert seen and "observation/pose" not in seen[0]
    assert seen[0]["observation/joint"].shape == (16,)
    assert "observation/image" in seen[0]
    assert "observation/wrist_image_left" not in seen[0]
