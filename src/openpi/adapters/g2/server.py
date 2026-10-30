"""G2 protocol boundary; model-facing OpenPI observations remain unchanged."""

import logging
import time

from g2_policy_protocol import protocol
from g2_policy_protocol.data.policy import policy_capabilities
from g2_policy_protocol.server import PolicyServer
import numpy as np
from openpi_client import msgpack_numpy

from openpi.adapters.g2.contract import CAMERA_OBSERVATION_KEYS

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)
_JPEG_IMAGE_KEYS = tuple(CAMERA_OBSERVATION_KEYS.values())


def _warmup_jpeg_transport() -> None:
    """Pay codec initialization before the robot's first observation arrives."""
    if cv2 is None:
        logger.info("JPEG transport warmup skipped (OpenCV unavailable; raw images only)")
        return
    start = time.monotonic()
    bgr = np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("JPEG transport warmup encode failed")
    packed = msgpack_numpy.packb(dict.fromkeys(_JPEG_IMAGE_KEYS, buffer.tobytes()))
    for step in range(2):
        obs = msgpack_numpy.unpackb(packed)
        decode_start = time.monotonic()
        _decode_jpeg_images(obs)
        logger.info(
            "JPEG transport warmup step=%d decode_ms=%.3f images=3x224x224",
            step + 1,
            (time.monotonic() - decode_start) * 1000,
        )
    logger.info("JPEG transport warmup complete total_ms=%.3f", (time.monotonic() - start) * 1000)


def _decode_jpeg_images(obs: object) -> None:
    """Replace JPEG bytes with RGB uint8 HWC. Raw arrays are left unchanged.

    The client encodes RGB via OpenCV, which stores JPEG as BGR, so decode swaps back.
    """
    if not isinstance(obs, dict):
        return
    for key in _JPEG_IMAGE_KEYS:
        value = obs.get(key)
        if not isinstance(value, bytes | bytearray | memoryview):
            continue
        if cv2 is None:
            raise RuntimeError("JPEG observation received but OpenCV is not installed")
        bgr = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"failed to decode JPEG observation {key}")
        obs[key] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def connection_metadata(policy_metadata):
    """Translate checkpoint semantics into shared wire capabilities."""
    native = dict(policy_metadata or {})
    arm = native.get("arm", "both")
    required = native.get("required_cameras")
    required = list(protocol.cameras_for_arm(arm) if required is None else required)
    settings = native.get("policy_contract")
    result = {
        "policy_hz": native.get("policy_hz", 30.0),
        "action_horizon": native.get("action_horizon", 50),
        "arm": arm,
        "image_encoding": "jpeg" if cv2 is not None else "raw",
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
        self._action_dim = metadata["action_spec"]["dim"]

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
        for name, value in request.get("context", {}).items():
            if name == "image_history":
                obs.update({CAMERA_OBSERVATION_KEYS[camera]: frames for camera, frames in value.items()})
            else:
                obs[f"observation/{name}"] = value
        for camera, image in request["images"].items():
            obs.setdefault(CAMERA_OBSERVATION_KEYS[camera], image)
        if request["rtc_enabled"]:
            obs.update(rtc_enabled=True, inference_delay=request["inference_delay"])
            if request["prev_chunk_left_over"] is not None:
                obs["prev_chunk_left_over"] = request["prev_chunk_left_over"]
        decode_start = time.monotonic()
        _decode_jpeg_images(obs)
        # Histories were already validated as raw RGB by the shared protocol.
        if spec["history_steps"] == 1:
            protocol.validate_images(
                {camera: obs[CAMERA_OBSERVATION_KEYS[camera]] for camera in request["images"]},
                metadata=self.metadata,
                decoded=True,
            )
        decode_ms = (time.monotonic() - decode_start) * 1000
        infer_start = time.monotonic()
        result = dict(self.policy.infer(obs))
        infer_ms = (time.monotonic() - infer_start) * 1000
        actions = np.asarray(result["actions"], np.float32)
        dim = self._action_dim
        # Legacy native policies may still return padded outputs; adaptation ends here.
        if actions.ndim != 2 or actions.shape[1] not in (dim, 32):
            raise ValueError(f"OpenPI adapter requires expanded {dim}-D actions")
        result["actions"] = actions[:, :dim]
        result["server_timing"] = {"decode_ms": decode_ms, "infer_ms": infer_ms}
        return result


class G2PolicyServer(PolicyServer):
    def __init__(self, policy, host="0.0.0.0", port=None, metadata=None):
        native = dict(metadata or {})
        native["reset_supported"] = callable(getattr(policy, "reset", None))
        shared = protocol.build_metadata(**connection_metadata(native))
        super().__init__(OpenPIBackend(policy, shared), host=host, port=port, metadata=shared)

    async def run(self):
        _warmup_jpeg_transport()
        await super().run()
