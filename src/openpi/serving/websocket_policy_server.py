import asyncio
import http
import logging
import time
import traceback

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

try:
    import cv2
except Exception:  # pragma: no cover - serving env always has opencv
    cv2 = None

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = dict(metadata or {})
        # Encoding is a transport capability, independent of the loaded policy.
        self._metadata["image_encoding"] = "jpeg" if cv2 is not None else "raw"
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                raw = await websocket.recv()
                obs_bytes = len(raw) if isinstance(raw, (bytes, bytearray)) else 0
                process_start = time.monotonic()
                obs = msgpack_numpy.unpackb(raw)
                decode_start = time.monotonic()
                unpack_ms = (decode_start - process_start) * 1000
                _decode_jpeg_images(obs)
                decode_ms = (time.monotonic() - decode_start) * 1000

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                    "unpack_ms": unpack_ms,
                    "decode_ms": decode_ms,
                    "process_ms": (time.monotonic() - process_start) * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                policy_ms = None
                timing = action.get("policy_timing")
                if isinstance(timing, dict) and "infer_ms" in timing:
                    try:
                        policy_ms = float(timing["infer_ms"])
                    except (TypeError, ValueError):
                        policy_ms = None
                image_shapes = _obs_image_shapes(obs)
                logger.info(
                    "infer infer_ms=%.1f policy_ms=%s obs_kb=%.1f images=%s decode_ms=%.1f",
                    infer_time * 1000.0,
                    "-" if policy_ms is None else f"{policy_ms:.1f}",
                    obs_bytes / 1024.0,
                    image_shapes or "-",
                    decode_ms,
                )

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


_JPEG_IMAGE_KEYS = (
    "observation/image",
    "observation/wrist_image_left",
    "observation/wrist_image_right",
)


def _decode_jpeg_images(obs: object) -> None:
    """Replace JPEG bytes with RGB uint8 HWC. Raw arrays are left unchanged.

    The client encodes RGB via OpenCV, which stores JPEG as BGR, so decode swaps back.
    """
    if not isinstance(obs, dict):
        return
    for key in _JPEG_IMAGE_KEYS:
        value = obs.get(key)
        if not isinstance(value, (bytes, bytearray, memoryview)):
            continue
        if cv2 is None:
            raise RuntimeError("JPEG observation received but OpenCV is not installed")
        bgr = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"failed to decode JPEG observation {key}")
        obs[key] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _obs_image_shapes(obs: object) -> str:
    if not isinstance(obs, dict):
        return ""
    parts: list[str] = []
    for key, tag in (
        ("observation/image", "H"),
        ("observation/wrist_image_left", "L"),
        ("observation/wrist_image_right", "R"),
    ):
        shape = getattr(obs.get(key), "shape", None)
        if shape is not None and len(shape) >= 2:
            parts.append(f"{tag}{int(shape[0])}x{int(shape[1])}")
    return ",".join(parts)


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
