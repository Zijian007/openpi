#!/usr/bin/env python3
"""G2-only deployment entrypoint; the native OpenPI server remains robot agnostic."""

import dataclasses
import logging
import time

import numpy as np
import tyro

from openpi.adapters.g2 import policy as policy_adapter
from openpi.adapters.g2 import presets
from openpi.adapters.g2.server import G2PolicyServer
from openpi.adapters.g2.transforms import make_g2_example
from openpi.models.rtc import RTCConfig


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Args:
    policy: Checkpoint | None = None
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False
    warmup: bool = False
    rtc: bool = False
    rtc_soft_overlap_steps: int = 5
    rtc_guidance: float = 10.0


def create_policy(args: Args):
    if args.policy is None:
        raise ValueError("G2 serving requires policy:checkpoint with --policy.config and --policy.dir")
    return policy_adapter.create_trained_policy(
        presets.get_config(args.policy.config),
        args.policy.dir,
        default_prompt=args.default_prompt,
        rtc_config=RTCConfig(soft_overlap_steps=args.rtc_soft_overlap_steps, max_guidance_weight=args.rtc_guidance)
        if args.rtc
        else None,
    )


def _warmup(policy, args: Args) -> None:
    if not args.warmup:
        return
    obs = make_g2_example()
    if policy.metadata.get("requires_joint_state", False):
        obs["observation/pose"] = obs["observation/state"]
        obs["observation/state"] = np.zeros(16, dtype=np.float32)
    pose = obs.get("observation/pose", obs["observation/state"])
    obs.update(
        {
            f"observation/{name}": pose.copy()
            for name in ("command_pose", "previous_command_pose", "object_pose", "start_pose")
        }
    )
    obs["observation/object_joint"] = np.zeros(16, np.float32)
    obs["observation/base_to_parent"] = np.eye(4, dtype=np.float32)
    if policy.metadata.get("requires_joint_state", False):
        obs["observation/joint"] = np.zeros(16, np.float32)
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
    policy = create_policy(args)
    metadata = policy.metadata
    _warmup(policy, args)
    if args.record:
        policy = policy_adapter.G2PolicyRecorder(policy, "policy_records")
    G2PolicyServer(policy=policy, host="0.0.0.0", port=args.port, metadata=metadata).serve_forever()


def cli() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))


if __name__ == "__main__":
    cli()
