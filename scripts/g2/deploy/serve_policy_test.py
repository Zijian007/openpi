import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import serve_policy


def test_shell_launches_adapter_with_optional_dependency(tmp_path):
    root = tmp_path / "project"
    entry = root / "scripts/g2/deploy/serve_policy.sh"
    entry.parent.mkdir(parents=True)
    entry.write_text(Path(__file__).with_name("serve_policy.sh").read_text())
    (root / "scripts/g2/_env.sh").write_text(
        f'G2_OPENPI_PROJECT_ROOT="{root}"\ng2_openpi_clear_proxy() {{ :; }}\ng2_openpi_setup_proxy() {{ :; }}\n'
    )
    (root / ".venv").mkdir()
    binary = tmp_path / "bin"
    binary.mkdir()
    uv = binary / "uv"
    uv.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        "if any(arg.endswith('pick_checkpoint.py') for arg in sys.argv):\n"
        "    print('/fake/checkpoint/10000')\n"
        "else:\n"
        "    print('LAUNCH=' + json.dumps(sys.argv[1:]))\n"
    )
    uv.chmod(0o755)
    result = subprocess.run(
        ["bash", str(entry), "--no-warmup", "--no-rtc"],
        cwd=root,
        env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}", "ARM": "both"},
        capture_output=True,
        text=True,
        check=True,
    )
    command = json.loads(
        next(line.removeprefix("LAUNCH=") for line in result.stdout.splitlines() if line.startswith("LAUNCH="))
    )
    assert command[:4] == ["run", "--extra", "g2", "scripts/g2/deploy/serve_policy.py"]
    assert "policy:checkpoint" in command
    assert "--policy.dir=/fake/checkpoint/10000" in command


@pytest.mark.parametrize("arm", ["both", "right"])
def test_joint_state_warmup_preserves_pose_for_single_arm_expansion(arm):
    seen = []

    def infer(obs):
        seen.append(obs.copy())
        return {"actions": np.zeros((50, 20), np.float32)}

    policy = SimpleNamespace(metadata={"requires_joint_state": True, "arm": arm}, infer=infer, reset=lambda: None)
    args = serve_policy.Args(
        warmup=True, policy=serve_policy.Checkpoint(config="pi05_g2_vr_low_mem_right", dir="unused")
    )
    serve_policy._warmup(policy, args)  # noqa: SLF001
    assert len(seen) == 2
    for obs in seen:
        assert obs["observation/state"].shape == (16,)
        if arm != "both":
            assert obs["observation/pose"].shape == (20,)
