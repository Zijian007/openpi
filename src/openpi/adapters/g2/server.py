"""G2 protocol boundary; model-facing OpenPI observations remain unchanged."""

import logging
import time

from g2_policy_protocol import protocol
from g2_policy_protocol.data.policy import policy_capabilities
from g2_policy_protocol.images import decode_image_history, decode_images
from g2_policy_protocol.images import encode_jpeg
from g2_policy_protocol.server import PolicyServer
import numpy as np

from openpi.adapters.g2.contract import CAMERA_OBSERVATION_KEYS

logger = logging.getLogger(__name__)


def _warmup_jpeg_transport(metadata) -> None:
    """Pay JPEG codec initialization before the robot's first observation arrives."""
    if metadata["image_encoding"] != "jpeg" or not metadata["required_cameras"]:
        return
    start = time.monotonic()
    pixels = np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    images = dict.fromkeys(metadata["required_cameras"], encode_jpeg(pixels))
    for step in range(2):
        decode_start = time.monotonic()
        decode_images(images, {**metadata, "image_sizes": {}})
        logger.info(
            "JPEG transport warmup step=%d decode_ms=%.3f images=%dx224x224",
            step + 1,
            (time.monotonic() - decode_start) * 1000,
            len(images),
        )
    logger.info("JPEG transport warmup complete total_ms=%.3f", (time.monotonic() - start) * 1000)


def connection_metadata(policy_metadata, image_encoding="jpeg"):
    """Translate checkpoint semantics and deployment transport into shared wire capabilities."""
    native = dict(policy_metadata or {})
    arm = native.get("arm", "both")
    required = native.get("required_cameras")
    required = list(protocol.cameras_for_arm(arm) if required is None else required)
    settings = native.get("policy_contract")
    result = {
        "policy_hz": native.get("policy_hz", 30.0),
        "action_horizon": native.get("action_horizon", 50),
        "arm": arm,
        "image_encoding": image_encoding,
        "required_cameras": required,
        "image_resize": "letterbox",
        "image_sizes": {key: [224, 224] for key in required},
        "rtc_supported": bool(native.get("rtc_supported", False)),
        "requires_joint_state": bool(native.get("requires_joint_state", False)),
        "extensions": {"backend": "openpi"},
    }
    if settings:
        result.update(policy_capabilities(
            settings, required, prompt=bool(native.get("requires_prompt", True)),
            reset=bool(native.get("reset_supported", False)),
        ))
        result["extensions"]["response_mode"] = "full_chunk"
    return result


class OpenPIBackend:
    """Convert canonical G2 observations to OpenPI keys; no wire dialect here."""

    def __init__(self, policy, metadata):
        self.policy, self.metadata = policy, metadata
        self._observation_spec = metadata["observation_spec"]

    def new_session(self):
        policy = self.policy.new_session() if hasattr(self.policy, "new_session") else self.policy
        return type(self)(policy, self.metadata)

    def reset(self):
        self.policy.reset()

    def infer(self, request):
        spec = self._observation_spec
        obs = {"prompt": request["prompt"]}
        if spec["joint"]:
            obs["observation/joint"] = request["joint_state"]
            obs["observation/state"] = request["joint_state"]
        if spec["pose"]:
            obs["observation/pose"] = request["state"]
            obs.setdefault("observation/state", request["state"])
        if "observation_time_ns" in request:
            obs["observation/time_ns"] = request["observation_time_ns"]
        for name, value in request["context"].items():
            if name != "image_history":
                obs[f"observation/{name}"] = value
        if request["rtc_enabled"]:
            obs.update(rtc_enabled=True, inference_delay=request["inference_delay"])
            if request["prev_chunk_left_over"] is not None:
                obs["prev_chunk_left_over"] = request["prev_chunk_left_over"]
        # Histories include the current frame, so the separate current images are unused.
        decode_start = time.monotonic()
        history = request["context"].get("image_history")
        images = (decode_images(request["images"], self.metadata) if history is None
                  else decode_image_history(history, self.metadata))
        decode_ms = (time.monotonic() - decode_start) * 1000
        obs.update({CAMERA_OBSERVATION_KEYS[camera]: image for camera, image in images.items()})
        infer_start = time.monotonic()
        result = dict(self.policy.infer(obs))
        infer_ms = (time.monotonic() - infer_start) * 1000
        # G2Policy already restored the full wire layout; PolicyServer validates it.
        result["actions"] = np.asarray(result["actions"], np.float32)
        result["server_timing"] = {"decode_ms": decode_ms, "infer_ms": infer_ms}
        return result


class G2PolicyServer(PolicyServer):
    def __init__(self, policy, host="0.0.0.0", port=None, metadata=None, image_encoding="jpeg"):
        native = dict(metadata or {})
        native["reset_supported"] = callable(getattr(policy, "reset", None))
        shared = protocol.build_metadata(**connection_metadata(native, image_encoding))
        super().__init__(OpenPIBackend(policy, shared), host=host, port=port, metadata=shared)

    async def run(self):
        _warmup_jpeg_transport(self._metadata)
        await super().run()
