import dataclasses
import enum
import logging
import socket
import time

import tyro

from openpi.models.rtc import RTCConfig
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False
    # Run a dummy infer before listen so JAX JIT is not paid by the robot's first chunk.
    warmup: bool = False
    # Inference-time guidance; uses the same checkpoint as ordinary inference.
    rtc: bool = False
    # Width of decaying prefix guidance AFTER the estimated inference delay.
    rtc_soft_overlap_steps: int = 5
    rtc_guidance: float = 10.0

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            if args.rtc and not args.policy.config.startswith("pi05_g2"):
                raise ValueError("This RTC wire deployment requires a pi05_g2 configuration")
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                rtc_config=RTCConfig(soft_overlap_steps=args.rtc_soft_overlap_steps,
                                     max_guidance_weight=args.rtc_guidance) if args.rtc else None,
            )
        case Default():
            if args.rtc:
                raise ValueError("RTC requires an explicit G2 checkpoint")
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def _g2_warmup_obs() -> dict:
    from openpi.policies import g2_policy

    return g2_policy.make_g2_example()


def _warmup(policy: _policy.Policy, args: Args) -> None:
    """Compile ``sample_actions`` before binding the port (healthz stays down until done)."""
    if not args.warmup:
        return
    config_name = args.policy.config if isinstance(args.policy, Checkpoint) else None
    if not config_name or not str(config_name).startswith("pi05_g2"):
        logging.info("warmup skipped (not a G2 config: %s)", config_name)
        return

    obs = _g2_warmup_obs()
    previous_actions = None
    logging.info("warmup start (JIT) images=224x224 prompt=%r", obs.get("prompt"))
    for step in range(4 if args.rtc else 2):
        if args.rtc and step == 2:
            obs = {**obs, "rtc_enabled": True, "prev_chunk_left_over": previous_actions, "inference_delay": 2}
        t0 = time.monotonic()
        out = policy.infer(obs)
        previous_actions = out["actions"]
        infer_ms = (time.monotonic() - t0) * 1000.0
        policy_ms = None
        timing = out.get("policy_timing") if isinstance(out, dict) else None
        if isinstance(timing, dict) and "infer_ms" in timing:
            try:
                policy_ms = float(timing["infer_ms"])
            except (TypeError, ValueError):
                policy_ms = None
        actions = out.get("actions") if isinstance(out, dict) else None
        shape = getattr(actions, "shape", None)
        logging.info(
            "warmup step=%d infer_ms=%.1f policy_ms=%s actions=%s",
            step,
            infer_ms,
            "-" if policy_ms is None else f"{policy_ms:.1f}",
            shape,
        )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    _warmup(policy, args)

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = "unknown"
    logging.info("Creating server (host: %s, ip: %s, port: %s)", hostname, local_ip, args.port)

    server_type = websocket_policy_server.WebsocketPolicyServer
    if policy_metadata.get("robot") == "g2":
        from openpi.serving.g2_policy_server import G2PolicyServer
        server_type = G2PolicyServer
    server = server_type(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
