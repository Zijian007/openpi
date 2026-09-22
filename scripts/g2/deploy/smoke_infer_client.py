#!/usr/bin/env python3
"""GPU-local smoke: connect to OpenPI serve_policy and run fake G2 observations.

Prerequisite (another terminal)::

    bash scripts/g2/deploy/serve_policy.sh \\
      --config-name pi05_g2_vr_low_mem \\
      --policy-dir checkpoints/pi05_g2_vr_low_mem/<exp>/<step>

Then::

    uv run scripts/g2/deploy/smoke_infer_client.py
    # or: bash scripts/g2/deploy/smoke_infer_client.sh

Observation keys match ``openpi.policies.g2_policy.make_g2_example`` / G2Inputs.
Not compatible with G2_pi policy_server or wholebody lerobot_pi05_client protocol.
"""

from __future__ import annotations

import dataclasses
import logging
import time
import urllib.error
import urllib.request

import numpy as np
import tyro
from openpi_client import websocket_client_policy

from openpi.policies import g2_policy

logger = logging.getLogger("g2.smoke_infer")

# G2 TrainConfig: action_horizon=10; G2Outputs truncates padded 32 → 20.
EXPECTED_HORIZON = 10
EXPECTED_ACTION_DIM = g2_policy.G2_ACTION_DIM


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    """Must match serve_policy --port (default 8000; use 8001 if LeRobot also on :8000)."""
    prompt: str = "Pick up the drink and put it in the box"
    num_infer: int = 3
    """How many infer round-trips (first often includes JIT / warmup)."""
    zeros: bool = False
    """If true, send zero state/images instead of random (deterministic shape check)."""
    connect_timeout_s: float = 60.0
    """Fail if /healthz is not OK within this many seconds."""


def _make_obs(args: Args) -> dict:
    if args.zeros:
        return {
            "observation/state": np.zeros(EXPECTED_ACTION_DIM, dtype=np.float32),
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_right": np.zeros((224, 224, 3), dtype=np.uint8),
            "prompt": args.prompt,
        }
    obs = g2_policy.make_g2_example()
    obs["prompt"] = args.prompt
    return obs


def _wait_healthz(host: str, port: int, timeout_s: float) -> None:
    """serve_policy exposes HTTP GET /healthz on the same port."""
    url = f"http://{host}:{port}/healthz"
    deadline = time.monotonic() + timeout_s
    last_err: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if 200 <= resp.status < 300:
                    logger.info("healthz OK (%s)", url)
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_err = exc
            logger.info("waiting for %s (%s)", url, exc)
            time.sleep(1.0)
    raise ConnectionError(f"healthz not ready at {url} within {timeout_s}s: {last_err}")


def _check_actions(actions: np.ndarray, *, step: int) -> None:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise AssertionError(f"step {step}: actions ndim={arr.ndim}, expected 2, shape={arr.shape}")
    horizon, dim = arr.shape
    if horizon != EXPECTED_HORIZON:
        raise AssertionError(f"step {step}: horizon={horizon}, expected {EXPECTED_HORIZON}")
    # Server may return padded 32 or G2Outputs-truncated 20.
    if dim < EXPECTED_ACTION_DIM:
        raise AssertionError(f"step {step}: action dim={dim}, expected >= {EXPECTED_ACTION_DIM}")
    if not np.isfinite(arr).all():
        raise AssertionError(f"step {step}: actions contain non-finite values")
    pose = arr[:, :EXPECTED_ACTION_DIM]
    logger.info(
        "step %d: actions shape=%s  pose[:20] finite  xyz_L=%s grip_L=%.3f grip_R=%.3f",
        step,
        arr.shape,
        np.array2string(pose[0, :3], precision=3),
        float(pose[0, 9]),
        float(pose[0, 19]),
    )


def main(args: Args) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    # Always print a one-liner so success is visible even if logging is muted.
    print(f"smoke_infer: ws://{args.host}:{args.port}  num_infer={args.num_infer}", flush=True)
    logger.info("waiting for serve_policy at ws://%s:%s ...", args.host, args.port)
    _wait_healthz(args.host, args.port, args.connect_timeout_s)

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    meta = client.get_server_metadata()
    logger.info("server metadata: %s", meta)

    for i in range(args.num_infer):
        obs = _make_obs(args)
        t0 = time.monotonic()
        out = client.infer(obs)
        dt_ms = (time.monotonic() - t0) * 1000.0
        if not isinstance(out, dict) or "actions" not in out:
            raise AssertionError(
                f"step {i}: unexpected response keys={list(out) if isinstance(out, dict) else type(out)}"
            )
        _check_actions(out["actions"], step=i)
        timing = out.get("server_timing") or out.get("policy_timing") or {}
        logger.info("step %d: roundtrip=%.1f ms  timing=%s", i, dt_ms, timing)

    logger.info(
        "OK: %d infer(s) passed (horizon=%d, action_dim>=%d)",
        args.num_infer,
        EXPECTED_HORIZON,
        EXPECTED_ACTION_DIM,
    )
    print(
        f"OK: {args.num_infer} infer(s) passed "
        f"(horizon={EXPECTED_HORIZON}, action_dim>={EXPECTED_ACTION_DIM})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(tyro.cli(Args)))
    except (AssertionError, ConnectionError, RuntimeError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
