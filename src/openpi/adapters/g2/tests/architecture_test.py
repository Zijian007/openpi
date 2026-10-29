"""Guard the dependency direction: robot adapters may depend on OpenPI, never the reverse."""

import ast
from pathlib import Path
import subprocess
import sys
import textwrap
import tomllib

from openpi.adapters.g2.config import PROJECT_ROOT


def test_native_modules_do_not_import_robot_adapters():
    paths = [*Path(PROJECT_ROOT / "src/openpi").rglob("*.py")]
    paths.extend(PROJECT_ROOT / "scripts" / name for name in ("train.py", "serve_policy.py", "compute_norm_stats.py"))
    forbidden = []
    for path in paths:
        if "adapters" in path.parts or path.name.endswith("_test.py"):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            forbidden.extend(
                (str(path), module)
                for module in modules
                if module.startswith(("openpi.adapters", "g2_policy_protocol", "g2_policy_tools"))
            )
    assert forbidden == []


def test_native_entrypoints_import_without_g2_dependencies():
    script = textwrap.dedent("""
        import importlib.abc
        import runpy
        import sys

        class BlockAdapters(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith(("openpi.adapters", "g2_policy_protocol", "g2_policy_tools")):
                    raise AssertionError(f"Native OpenPI tried to import {fullname}")

        sys.meta_path.insert(0, BlockAdapters())
        from openpi.training import config
        from openpi.policies import policy_config
        from openpi.serving import websocket_policy_server
        assert config.get_config("debug").name == "debug"
        assert not hasattr(config, "LeRobotG2DataConfig")
        try:
            config.get_config("pi05_g2_vr")
        except ValueError:
            pass
        else:
            raise AssertionError("G2 preset leaked into native registry")
        runpy.run_path("scripts/train.py")
        runpy.run_path("scripts/serve_policy.py")
        runpy.run_path("scripts/compute_norm_stats.py")
        print("Native imports are isolated")
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=45, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "Native imports are isolated" in result.stdout


def test_protocol_is_an_optional_dependency():
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())["project"]
    assert all(not dependency.startswith("g2-policy-protocol") for dependency in project["dependencies"])
    assert "g2-policy-protocol[geometry]==0.6.0" in project["optional-dependencies"]["g2"]
