#!/usr/bin/env python3
"""G2-only deployment entrypoint; the native OpenPI server remains robot agnostic."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from pathlib import Path
import time


@dataclasses.dataclass
class Checkpoint:
    dir: str
    config: str | None = None  # only for checkpoints predating the saved training snapshot


@dataclasses.dataclass
class Args:
    policy: Checkpoint = dataclasses.field(default_factory=lambda: Checkpoint(dir=""))
    default_prompt: str | None = None
    host: str = "0.0.0.0"
    port: int = 8001
    record: bool = False
    warmup: bool = True
    rtc: bool = True
    rtc_soft_overlap_steps: int = 5
    rtc_guidance: float = 10.0


def create_policy(args: Args):
    from openpi.adapters.g2 import policy as policy_adapter
    from openpi.adapters.g2 import presets
    from openpi.models.rtc import RTCConfig

    if args.policy is None or not args.policy.dir:
        raise ValueError("G2 serving requires a checkpoint path (--policy.dir or deploy config)")
    return policy_adapter.create_trained_policy(
        presets.get_config(args.policy.config) if args.policy.config is not None else None,
        args.policy.dir,
        default_prompt=args.default_prompt,
        rtc_config=RTCConfig(soft_overlap_steps=args.rtc_soft_overlap_steps, max_guidance_weight=args.rtc_guidance)
        if args.rtc
        else None,
    )


def _warmup_observation(policy) -> dict:
    """Synthetic observation using only fields the checkpoint contract declares."""
    import numpy as np
    from openpi.adapters.g2.server import connection_metadata
    from openpi.adapters.g2.transforms import make_g2_example

    if not policy.metadata.get("policy_contract"):
        obs = make_g2_example()
        if policy.metadata.get("requires_joint_state", False):
            obs["observation/pose"] = obs["observation/state"]
            obs["observation/state"] = np.zeros(16, dtype=np.float32)
            obs["observation/joint"] = np.zeros(16, np.float32)
        return obs
    wire = connection_metadata(policy.metadata)
    spec = wire["observation_spec"]
    native = {
        "head_color": "observation/image",
        "hand_left_color": "observation/wrist_image_left",
        "hand_right_color": "observation/wrist_image_right",
    }
    blank = np.zeros((224, 224, 3), np.uint8)
    obs = {native[camera]: blank.copy() for camera in wire["required_cameras"]}
    if spec["prompt"]:
        obs["prompt"] = "warmup"
    pose = np.zeros(20, np.float32)
    joint = np.zeros(16, np.float32)
    if spec["pose"]:
        obs["observation/pose"] = pose.copy()
        obs["observation/state"] = pose.copy()
    if spec["joint"]:
        obs["observation/joint"] = joint.copy()
        obs.setdefault("observation/state", joint.copy())
    count = spec["history_steps"]
    for key in spec["context_keys"]:
        if key in ("command_pose", "previous_command_pose", "object_pose", "start_pose"):
            obs[f"observation/{key}"] = pose.copy()
        elif key in ("object_joint", "start_joint"):
            obs[f"observation/{key}"] = joint.copy()
        elif key == "base_to_parent":
            obs["observation/base_to_parent"] = np.eye(4, dtype=np.float32)
        elif key == "pose_history":
            obs["observation/pose_history"] = np.stack([pose] * count)
        elif key == "joint_history":
            obs["observation/joint_history"] = np.stack([joint] * count)
        elif key == "image_history":
            for camera in wire["required_cameras"]:
                obs[native[camera]] = np.stack([blank] * count)
    return obs


def _warmup(policy, args: Args) -> None:
    if not args.warmup:
        return
    obs = _warmup_observation(policy)
    previous_actions = None
    for step in range(4 if args.rtc else 2):
        if args.rtc and step == 2:
            obs.update(rtc_enabled=True, prev_chunk_left_over=previous_actions, inference_delay=2)
        start = time.monotonic()
        result = policy.infer(obs)
        previous_actions = result["actions"]
        logging.info(
            "G2 warmup step=%d infer_ms=%.1f actions=%s",
            step,
            (time.monotonic() - start) * 1000,
            previous_actions.shape,
        )

    policy.reset()


def main(args: Args) -> None:
    from openpi.adapters.g2 import policy as policy_adapter
    from openpi.adapters.g2.server import G2PolicyServer

    policy = create_policy(args)
    metadata = policy.metadata
    _warmup(policy, args)
    if args.record:
        policy = policy_adapter.G2PolicyRecorder(policy, "policy_records")
    G2PolicyServer(policy=policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


def load_deploy_config(path: Path, *, apply_runtime: bool = False) -> Args:
    """Deployment owns process settings; model semantics come from checkpoint assets."""
    from openpi.adapters.g2.config import read_yaml

    raw = read_yaml(path)
    allowed = {"checkpoint", "legacy_preset", "server", "runtime"}
    if set(raw) != allowed:
        raise ValueError(f"deployment config fields must be {sorted(allowed)}")
    server, runtime = raw["server"], raw["runtime"]
    server_keys = {"host", "port", "warmup", "record", "rtc", "rtc_soft_overlap_steps", "rtc_guidance"}
    if not isinstance(server, dict) or set(server) != server_keys:
        raise ValueError(f"server config fields must be {sorted(server_keys)}")
    gpu = runtime.get("gpu") if isinstance(runtime, dict) else None
    if not isinstance(runtime, dict) or set(runtime) != {"gpu"} or not isinstance(gpu, str) or not gpu.strip():
        raise ValueError("runtime config fields must be gpu")
    checkpoint = raw["checkpoint"]
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("checkpoint must be a nonempty path")
    if raw["legacy_preset"] is not None and not isinstance(raw["legacy_preset"], str):
        raise ValueError("legacy_preset must be a string or null")
    if type(server["port"]) is not int or not 0 < server["port"] < 65536:
        raise ValueError("port must be an integer in [1, 65535]")
    for field in ("warmup", "record", "rtc"):
        if type(server[field]) is not bool:
            raise ValueError(f"{field} must be boolean")
    if apply_runtime:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)
    return Args(policy=Checkpoint(dir=checkpoint, config=raw["legacy_preset"]), **server)


def cli() -> None:
    import tyro

    logging.basicConfig(level=logging.INFO, force=True)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--deploy-config", type=Path, default=Path(__file__).with_name("config.yaml"))
    known, remaining = parser.parse_known_args()
    defaults = load_deploy_config(known.deploy_config, apply_runtime=True)
    main(tyro.cli(Args, args=remaining, default=defaults))


if __name__ == "__main__":
    cli()
