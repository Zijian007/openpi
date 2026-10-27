"""G2 protocol boundary; model-facing OpenPI observations remain unchanged."""

import logging
import time

from g2_policy_protocol import codec
from g2_policy_protocol import protocol
import numpy as np
from openpi_client import msgpack_numpy
import websockets

from openpi.adapters.g2.transforms import action_dimensions
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)
# Wire camera id → OpenPI observation key. ``prepare_inputs`` renames those again
# to base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb for the model.
_NATIVE_IMAGES = dict(
    zip(
        protocol.CAMERA_IDS,
        (
            "observation/image",
            "observation/wrist_image_left",
            "observation/wrist_image_right",
        ),
        strict=True,
    )
)


_JPEG_IMAGE_KEYS = tuple(_NATIVE_IMAGES.values())


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
    """Build handshake fields and explicit adapter capabilities from the checkpoint.

    ``action_dim`` and ``model_action_dim`` stay on the policy for RTC prefix width.
    Pose clients receive (T, 20); the extended protocol declares other control widths.
    """
    native = dict(policy_metadata or {})
    arm = native.get("arm", "both")
    required = list(native.get("required_cameras") or protocol.cameras_for_arm(arm))
    settings = native.get("policy_contract")
    extensions = {"backend": "openpi", "policy_type": "pi05"}
    if settings:
        extensions["policy_contract"] = settings
        extensions["wire_pose"] = {"parent_frame": "base_link", "ee_frame": "flange", "rotation": "rot6d"}
        extensions["wire_action_dim"] = action_dimensions(settings)[1]
        extensions["action_units"] = {
            "pose_target": ["m"] * 3 + ["dimensionless"] * 6 + ["close_ratio"],
            "joint_position": ["rad"] * 7 + ["close_ratio"],
            "joint_velocity": ["rad/s"] * 7 + ["close_ratio"],
            "eef_twist": ["m/s"] * 3 + ["rad/s"] * 3 + ["close_ratio"],
            "joint_torque": ["N*m"] * 7 + ["close_ratio"],
        }[settings["action"]["control_quantity"]] * 2
    return dict(
        policy_hz=native.get("policy_hz", 30.0),
        action_horizon=native.get("action_horizon", 50),
        arm=arm,
        image_encoding="jpeg" if cv2 is not None else "raw",
        required_cameras=required,
        image_resize="letterbox",
        image_sizes={key: [224, 224] for key in required},
        rtc_supported=bool(native.get("rtc_supported", False)),
        requires_joint_state=bool(native.get("requires_joint_state", False)),
        **({"execution_steps": native["execution_steps"]} if native.get("execution_steps") is not None else {}),
        extensions=extensions,
    )


class G2PolicyServer(WebsocketPolicyServer):
    def __init__(self, policy, host="0.0.0.0", port=None, metadata=None):
        super().__init__(policy, host=host, port=port, metadata=metadata)
        self._contract = connection_metadata(metadata)
        self._requires_joint_state = self._contract["requires_joint_state"]
        self._settings = (metadata or {}).get("policy_contract")
        self._advanced = bool(
            self._settings
            and (
                action_dimensions(self._settings)[1] != 20
                or self._settings["pose"].get("base_to_parent") == "observation"
                or {self._settings["action"]["relative_to"], self._settings["state"]["relative_to"]}.intersection(
                    {"current_command_pose", "previous_command_pose", "episode_start_pose", "object_pose"}
                )
                or self._settings["timing"]["history_steps"] > 1
                or self._settings["timing"]["target_offset_steps"] != 0
                or self._settings["timing"]["stride_steps"] != 1
            )
        )
        protocol.build_metadata(**self._contract)  # Fail before listening on invalid capabilities.

    async def run(self):
        _warmup_jpeg_transport()
        await super().run()

    async def _handler(self, websocket):
        path = getattr(getattr(websocket, "request", None), "path", "/")
        encoding = codec.encoding_from_path(path)
        metadata = protocol.build_metadata(**self._contract)
        if self._advanced:
            metadata["protocol"] = "g2.openpi.v1"
        policy = self._policy.new_session() if hasattr(self._policy, "new_session") else self._policy
        await websocket.send(codec.dumps(metadata, encoding=encoding))
        while True:
            payload = None
            try:
                raw = await websocket.recv()
                start = time.monotonic()
                payload = codec.loads(raw, encoding=encoding)
                extra = {}
                parse_payload = payload
                if self._advanced:
                    if payload.get("protocol") != "g2.openpi.v1":
                        raise ValueError("This policy requires explicit g2.openpi.v1 client support")
                    parse_payload = dict(payload)
                    parse_payload.pop("protocol")
                    extra = parse_payload.pop("context", {})
                    if not isinstance(extra, dict):
                        raise ValueError("context must be an object")
                    prefix = parse_payload.pop("prev_chunk_left_over", None)
                    request = protocol.parse_request(parse_payload, metadata=metadata)
                    if prefix is not None:
                        expected_dim = action_dimensions(self._settings)[1]
                        prefix = np.asarray(prefix, np.float32)
                        if (
                            not request["rtc_enabled"]
                            or prefix.ndim != 2
                            or not len(prefix)
                            or prefix.shape[1] != expected_dim
                            or not np.isfinite(prefix).all()
                        ):
                            raise ValueError(f"RTC prefix requires finite (T, {expected_dim}) actions")
                        request["prev_chunk_left_over"] = prefix
                else:
                    request = protocol.parse_request(payload, metadata=metadata)
                if self._advanced and self._settings["timing"]["history_steps"] > 1:
                    space = self._settings["state"]["space"]
                    needed = {"image_history"}
                    if space in ("eef", "both"):
                        needed.add("pose_history")
                    if space in ("joint", "both"):
                        needed.add("joint_history")
                    if (
                        not needed <= extra.keys()
                        or not set(self._contract["required_cameras"]) <= extra.get("image_history", {}).keys()
                    ):
                        raise ValueError("History policy requires all configured state/image histories at sample_hz")
                if self._settings:
                    refs = {self._settings["state"]["relative_to"], self._settings["action"]["relative_to"]}
                    for reference, key in (
                        ("current_command_pose", "command_pose"),
                        ("previous_command_pose", "previous_command_pose"),
                    ):
                        if reference in refs and key not in extra:
                            raise ValueError(f"{reference} requires explicit context.{key} at this sample time")
                unpack_ms = (time.monotonic() - start) * 1000
                state = request["joint_state"] if self._requires_joint_state else request["state"]
                obs = {"observation/state": state, "prompt": request["prompt"]}
                if self._requires_joint_state:
                    obs["observation/pose"] = request["state"]
                    obs["observation/joint"] = request["joint_state"]
                for name, value in extra.items():
                    if name not in (
                        "command_pose",
                        "previous_command_pose",
                        "object_pose",
                        "episode_start_pose",
                        "object_joint",
                        "base_to_parent",
                        "pose_history",
                        "joint_history",
                        "image_history",
                        "start_pose",
                        "start_joint",
                    ):
                        raise ValueError(f"Unknown observation context {name}")
                    if name == "image_history":
                        for camera, history_frames in value.items():
                            if camera not in self._contract["required_cameras"]:
                                raise ValueError("Unexpected history camera")
                            frames = np.asarray(history_frames)
                            if frames.dtype != np.uint8 or frames.shape != (
                                self._settings["timing"]["history_steps"],
                                224,
                                224,
                                3,
                            ):
                                raise ValueError("Invalid image_history shape")
                            obs[_NATIVE_IMAGES[camera] + "_history"] = frames
                    else:
                        arr = np.asarray(value, np.float32)
                        width = 16 if name in ("joint_history", "object_joint", "start_joint") else 20
                        expected = (
                            (4, 4)
                            if name == "base_to_parent"
                            else (self._settings["timing"]["history_steps"], width)
                            if name.endswith("_history")
                            else (width,)
                        )
                        if arr.shape != expected or not np.isfinite(arr).all():
                            raise ValueError(f"Invalid context {name}; expected finite {expected}")
                        obs[f"observation/{name}"] = arr
                obs.update({_NATIVE_IMAGES[key]: image for key, image in request["images"].items()})
                if request["rtc_enabled"]:
                    obs.update(rtc_enabled=True, inference_delay=request["inference_delay"])
                    if request["prev_chunk_left_over"] is not None:
                        obs["prev_chunk_left_over"] = request["prev_chunk_left_over"]
                decode_start = time.monotonic()
                _decode_jpeg_images(obs)
                protocol.validate_images(
                    {key: obs[_NATIVE_IMAGES[key]] for key in request["images"]}, metadata=metadata, decoded=True
                )
                decode_ms = (time.monotonic() - decode_start) * 1000
                infer_start = time.monotonic()
                if self._advanced and self._settings["timing"]["history_steps"] > 1:
                    for key in _NATIVE_IMAGES.values():
                        if key + "_history" in obs:
                            obs[key] = obs.pop(key + "_history")
                result = policy.infer(obs)
                infer_ms = (time.monotonic() - infer_start) * 1000
                actions = np.asarray(result["actions"], dtype=np.float32)
                expected_dim = action_dimensions(self._settings)[1] if self._settings else 20
                if (
                    actions.ndim != 2
                    or not len(actions)
                    or actions.shape[1] not in (expected_dim, 32)
                    or not np.isfinite(actions).all()
                ):
                    raise ValueError(f"OpenPI adapter requires finite expanded {expected_dim}-D actions")
                timing = {
                    "infer_ms": infer_ms,
                    "unpack_ms": unpack_ms,
                    "decode_ms": decode_ms,
                    "process_ms": (time.monotonic() - start) * 1000,
                }
                if self._advanced:
                    response = {
                        "type": protocol.ACTION_TYPE,
                        "protocol": "g2.openpi.v1",
                        "request_id": request["request_id"],
                        "actions": actions[:, :expected_dim],
                        "server_timing": timing,
                        "control_quantity": self._settings["action"]["control_quantity"],
                    }
                    for key in ("policy_timing", "rtc_info"):
                        if key in result:
                            response[key] = result[key]
                else:
                    response = protocol.build_response(
                        request_id=request["request_id"],
                        actions=actions[:, :20],
                        server_timing=timing,
                        policy_timing=result.get("policy_timing"),
                        rtc_info=result.get("rtc_info"),
                    )
                await websocket.send(codec.dumps(response, encoding=encoding))
            except websockets.ConnectionClosed:
                return
            except Exception as exc:
                logger.exception("G2 inference failed")
                request_id = payload.get("request_id", 0) if isinstance(payload, dict) else 0
                if type(request_id) is not int or request_id < 0:
                    request_id = 0
                try:
                    await websocket.send(
                        codec.dumps(
                            protocol.build_error(request_id=request_id, message=f"{type(exc).__name__}: {exc}"),
                            encoding=encoding,
                        )
                    )
                except websockets.ConnectionClosed:
                    return
