#!/usr/bin/env python3
"""GPU-local smoke: connect to OpenPI serve_policy and run fake G2 observations.

Prerequisite (another terminal)::

    bash scripts/g2/deploy/serve_policy.sh \\
      --config-name pi05_g2_vr_low_mem \\
      --policy-dir checkpoints/pi05_g2_vr_low_mem/<exp>/<step>

Then::

    uv run scripts/g2/deploy/smoke_infer_client.py
    # or: bash scripts/g2/deploy/smoke_infer_client.sh

Uses g2.policy.v2, identical to G2_pi and the G2 robot client.
"""

from __future__ import annotations

import dataclasses
import logging
import time
import urllib.error
import urllib.request

import numpy as np
import tyro
from g2_policy_protocol import codec, protocol
from websockets.sync.client import connect

from openpi.policies import g2_policy

logger = logging.getLogger("g2.smoke_infer")

# G2 TrainConfig: action_horizon=50; G2Outputs truncates padded 32 → 20.
EXPECTED_HORIZON = 50
EXPECTED_ACTION_DIM = g2_policy.G2_ACTION_DIM


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    """Must match serve_policy --port (default 8000; use 8001 if LeRobot also on :8000)."""
    prompt: str = "Pick up the drink and put it in the box"
    num_infer: int = 3
    """How many infer round-trips. With serve_policy.sh warmup, all should be steady (~100ms)."""
    connect_timeout_s: float = 180.0
    """Fail if /healthz is not OK within this many seconds (covers weight load + JIT warmup)."""
    zeros: bool = False
    """If true, send zero state/images instead of random (deterministic shape check)."""


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
    """serve_policy exposes HTTP GET /healthz on the same port (after warmup)."""
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

    with connect(f"ws://{args.host}:{args.port}", compression=None, max_size=None) as client:
        meta = codec.loads(client.recv(), encoding="msgpack")
        protocol.validate_metadata(meta)
        logger.info("server metadata: %s", meta)
        for i in range(args.num_infer):
            native = _make_obs(args)
            mapping = dict(zip(protocol.CAMERA_IDS, (
                "observation/image", "observation/wrist_image_left", "observation/wrist_image_right")))
            images = {key: native[mapping[key]] for key in meta["required_cameras"]}
            if meta["image_encoding"] == "jpeg":
                import cv2
                images = {key: cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tobytes()
                          for key, image in images.items()}
            request = protocol.build_request(request_id=i, prompt=args.prompt,
                state=native["observation/state"], images=images, metadata=meta)
            t0 = time.monotonic()
            client.send(codec.dumps(request))
            out = codec.loads(client.recv())
            actions = protocol.parse_response(out, request_id=i)
            _check_actions(actions, step=i)
            logger.info("step %d: roundtrip=%.1f ms timing=%s", i,
                        (time.monotonic()-t0)*1000, out.get("server_timing", {}))

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
