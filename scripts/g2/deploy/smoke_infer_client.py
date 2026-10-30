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

import argparse
import dataclasses
import logging
from pathlib import Path
import time
import urllib.error
import urllib.request

from g2_policy_protocol import codec
from g2_policy_protocol import protocol
from g2_policy_protocol.samples import synthetic_context
import numpy as np
import tyro
from websockets.sync.client import connect

from openpi.adapters.g2.config import DEFAULT_DEPLOY_CONFIG
from openpi.adapters.g2.config import read_yaml
from openpi.adapters.g2.contract import CAMERA_OBSERVATION_KEYS
from openpi.adapters.g2.transforms import G2_ACTION_DIM
from openpi.adapters.g2.transforms import make_g2_example

logger = logging.getLogger("g2.smoke_infer")

# G2 TrainConfig: action_horizon=50; G2Outputs truncates padded 32 → 20.


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8001
    """Defaults to the selected deployment YAML's server.port."""
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
            "observation/state": np.zeros(G2_ACTION_DIM, dtype=np.float32),
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_right": np.zeros((224, 224, 3), dtype=np.uint8),
            "prompt": args.prompt,
        }
    obs = make_g2_example()
    obs["prompt"] = args.prompt
    return obs


def _address(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host and not host.startswith("[") else f"{host}:{port}"


def _wait_healthz(host: str, port: int, timeout_s: float) -> None:
    """serve_policy exposes HTTP GET /healthz on the same port (after warmup)."""
    url = f"http://{_address(host, port)}/healthz"
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


def _check_actions(actions: np.ndarray, *, step: int, metadata: dict) -> None:
    expected_dim = metadata["action_spec"]["dim"]
    expected_horizon = metadata["action_horizon"]
    arr = np.asarray(actions, dtype=np.float32)
    if arr.shape != (expected_horizon, expected_dim) or not np.isfinite(arr).all():
        raise AssertionError(
            f"step {step}: expected finite {(expected_horizon, expected_dim)} actions, got {arr.shape}"
        )
    logger.info(
        "step %d: actions=%s quantity=%s first=%s",
        step,
        arr.shape,
        metadata["action_spec"]["control_quantity"],
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
    url = f"ws://{_address(args.host, args.port)}"
    print(f"smoke_infer: {url}  num_infer={args.num_infer}", flush=True)
    logger.info("waiting for serve_policy at %s ...", url)
    _wait_healthz(args.host, args.port, args.connect_timeout_s)

    with connect(url, compression=None, max_size=None) as client:
        meta = codec.loads(client.recv(), encoding="msgpack")
        protocol.validate_metadata(meta)
        logger.info("server metadata: %s", meta)
        for i in range(args.num_infer):
            native = _make_obs(args)
            raw_images = {key: native[CAMERA_OBSERVATION_KEYS[key]] for key in meta["required_cameras"]}
            images = raw_images
            if meta["image_encoding"] == "jpeg":
                import cv2

                images = {
                    key: cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tobytes()
                    for key, image in images.items()
                }
            context = synthetic_context(
                {
                    "state": native["observation/state"],
                    "joint_state": np.zeros(16, np.float32),
                    "images": raw_images,
                },
                meta,
            )
            request = protocol.build_request(
                request_id=i,
                prompt=args.prompt,
                state=native["observation/state"],
                images=images,
                metadata=meta,
                joint_state=np.zeros(16, np.float32) if meta["requires_joint_state"] else None,
                observation_time_ns=time.monotonic_ns(),
                context=context,
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
        meta["action_spec"]["dim"],
    )
    print(
        f"OK: {args.num_infer} infer(s) passed "
        f"(horizon={meta['action_horizon']}, action_dim={meta['action_spec']['dim']})",
        flush=True,
    )
    return 0


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--config", "--deploy-config", dest="deploy_config", type=Path, default=DEFAULT_DEPLOY_CONFIG)
    known, remaining = parser.parse_known_args(argv)
    server = read_yaml(known.deploy_config)["server"]
    host, port = server["host"], server["port"]
    if not isinstance(host, str) or not host.strip():
        raise ValueError("server.host must be a nonempty string")
    if type(port) is not int or not 0 < port < 65536:
        raise ValueError("server.port must be an integer in [1, 65535]")
    # A server bind address is not necessarily a suitable local client destination.
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    return main(
        tyro.cli(
            Args,
            args=remaining,
            default=Args(host=host, port=port),
            description="G2 smoke client. Select deployment YAML with --config PATH (alias: --deploy-config).",
        )
    )


if __name__ == "__main__":
    try:
        raise SystemExit(cli())
    except (AssertionError, ConnectionError, RuntimeError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
