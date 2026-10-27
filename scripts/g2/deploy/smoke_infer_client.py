#!/usr/bin/env python3
"""GPU-local smoke: connect to a running G2 serve_policy and send fake observations.

Prerequisite (another terminal)::

    bash scripts/g2/deploy/serve_policy.sh
    # or override the YAML checkpoint:
    bash scripts/g2/deploy/serve_policy.sh --policy.dir checkpoints/<config>/<exp>/<step>

Then::

    bash scripts/g2/deploy/smoke_infer_client.sh

Follows the server handshake (g2.policy.v3 when the checkpoint has a training snapshot).
"""

from __future__ import annotations

import dataclasses
import logging
import time
import urllib.error
import urllib.request

from g2_policy_protocol import codec
from g2_policy_protocol import protocol
from g2_policy_protocol.samples import synthetic_context
import numpy as np
import tyro
from websockets.sync.client import connect

from openpi.adapters.g2.transforms import G2_ACTION_DIM

logger = logging.getLogger("g2.smoke_infer")

# G2 TrainConfig: action_horizon=50; G2Outputs truncates padded 32 → 20.
EXPECTED_HORIZON = 50
EXPECTED_ACTION_DIM = G2_ACTION_DIM


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8001
    """Must match deploy/config.yaml server.port."""
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


def _check_actions(actions: np.ndarray, *, step: int, metadata: dict | None = None) -> None:
    metadata = metadata or {}
    expected_dim = protocol.action_contract(metadata)["dim"]
    expected_horizon = metadata.get("action_horizon", EXPECTED_HORIZON)
    arr = np.asarray(actions, dtype=np.float32)
    if arr.shape != (expected_horizon, expected_dim) or not np.isfinite(arr).all():
        raise AssertionError(
            f"step {step}: expected finite {(expected_horizon, expected_dim)} actions, got {arr.shape}"
        )
    logger.info(
        "step %d: actions=%s quantity=%s first=%s",
        step,
        arr.shape,
        protocol.action_contract(metadata)["control_quantity"],
        np.array2string(arr[0, :3], precision=3),
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
            mapping = dict(
                zip(
                    protocol.CAMERA_IDS,
                    ("observation/image", "observation/wrist_image_left", "observation/wrist_image_right"),
                    strict=True,
                )
            )
            images = {key: native[mapping[key]] for key in meta["required_cameras"]}
            if meta["image_encoding"] == "jpeg":
                import cv2

                images = {
                    key: cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tobytes()
                    for key, image in images.items()
                }
            context = None
            if protocol.is_v3(meta):
                context = synthetic_context(dict(state=native["observation/state"],
                                                  joint_state=np.zeros(16, np.float32),
                                                  images={key: native[mapping[key]] for key in meta["required_cameras"]}), meta)
            request = protocol.build_request(
                request_id=i, prompt=args.prompt, state=native["observation/state"], images=images,
                metadata=meta, joint_state=np.zeros(16, np.float32) if meta["requires_joint_state"] else None,
                observation_time_ns=time.monotonic_ns() if protocol.is_v3(meta) else None, context=context,
            )
            t0 = time.monotonic()
            client.send(codec.dumps(request))
            out = codec.loads(client.recv())
            actions = protocol.parse_response(out, request_id=i, metadata=meta)
            _check_actions(actions, step=i, metadata=meta)
            logger.info(
                "step %d: roundtrip=%.1f ms timing=%s", i, (time.monotonic() - t0) * 1000, out.get("server_timing", {})
            )

    logger.info(
        "OK: %d infer(s) passed (horizon=%d, action_dim>=%d)",
        args.num_infer,
        meta["action_horizon"],
        protocol.action_contract(meta)["dim"],
    )
    print(
        f"OK: {args.num_infer} infer(s) passed "
        f"(horizon={meta['action_horizon']}, action_dim={protocol.action_contract(meta)['dim']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(tyro.cli(Args)))
    except (AssertionError, ConnectionError, RuntimeError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
