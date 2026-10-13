import asyncio
from unittest.mock import patch

import cv2
import numpy as np
from openpi_client import msgpack_numpy
import pytest
import websockets.asyncio.server
from websockets.sync.client import connect

from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.serving.websocket_policy_server import _warmup_jpeg_transport


def test_jpeg_warmup_decodes_three_images_twice_without_policy():
    from openpi.serving.websocket_policy_server import _decode_jpeg_images

    decoded = []

    def decode(obs):
        assert len(obs) == 3
        assert all(isinstance(image, bytes) for image in obs.values())
        _decode_jpeg_images(obs)
        decoded.append(obs)

    with patch("openpi.serving.websocket_policy_server._decode_jpeg_images", side_effect=decode):
        _warmup_jpeg_transport()
    assert len(decoded) == 2
    for obs in decoded:
        assert all(image.shape == (224, 224, 3) and image.dtype == np.uint8 for image in obs.values())


def test_jpeg_warmup_skips_without_opencv():
    with (
        patch("openpi.serving.websocket_policy_server.cv2", None),
        patch("openpi.serving.websocket_policy_server._decode_jpeg_images") as decode,
    ):
        _warmup_jpeg_transport()
    decode.assert_not_called()


def test_transport_warmup_finishes_before_binding_port():
    events = []

    class Listener:
        async def __aenter__(self):
            events.append("listen")
            return self

        async def __aexit__(self, *args):
            pass

        async def serve_forever(self):
            events.append("serve")

    def bind(*args, **kwargs):
        events.append("bind")
        return Listener()

    with (
        patch("openpi.serving.websocket_policy_server._warmup_jpeg_transport",
              side_effect=lambda: events.append("warmup")),
        patch("openpi.serving.websocket_policy_server._server.serve", side_effect=bind),
    ):
        asyncio.run(WebsocketPolicyServer(None).run())
    assert events == ["warmup", "bind", "listen", "serve"]


def test_transport_owns_encoding_capability():
    metadata = {"robot": "g2", "image_encoding": "stale"}
    server = WebsocketPolicyServer(None, metadata=metadata)
    assert server._metadata["image_encoding"] == "jpeg"
    assert metadata["image_encoding"] == "stale"
    with patch("openpi.serving.websocket_policy_server.cv2", None):
        assert WebsocketPolicyServer(None)._metadata["image_encoding"] == "raw"


def test_real_server_decodes_jpeg_before_policy_and_accepts_raw():
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    rgb[..., 0] = 220
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    keys = ("observation/image", "observation/wrist_image_left", "observation/wrist_image_right")
    seen = []

    class Policy:
        def infer(self, obs):
            for key in keys:
                assert obs[key].shape == rgb.shape
                assert obs[key].dtype == np.uint8
                assert obs[key][..., 0].mean() > 200
                assert obs[key][..., 2].mean() < 10
            seen.append(obs)
            return {"actions": np.zeros((2, 20), dtype=np.float32)}

    server = WebsocketPolicyServer(Policy())

    def client(port):
        with connect(f"ws://127.0.0.1:{port}", compression=None, open_timeout=3) as ws:
            meta = msgpack_numpy.unpackb(ws.recv(timeout=3))
            assert meta["image_encoding"] == "jpeg"
            for image in (buf.tobytes(), rgb):
                ws.send(msgpack_numpy.packb(dict.fromkeys(keys, image)))
                response = msgpack_numpy.unpackb(ws.recv(timeout=3))
                timing = response["server_timing"]
                assert timing["decode_ms"] >= 0
                assert timing["unpack_ms"] >= 0
                assert timing["process_ms"] >= timing["infer_ms"] + timing["decode_ms"]

    async def run():
        async with websockets.asyncio.server.serve(server._handler, "127.0.0.1", 0, compression=None) as listener:
            await asyncio.wait_for(asyncio.to_thread(client, listener.sockets[0].getsockname()[1]), 10)

    asyncio.run(run())
    assert len(seen) == 2


def test_corrupt_jpeg_is_rejected():
    from openpi.serving.websocket_policy_server import _decode_jpeg_images

    with pytest.raises(ValueError, match="failed to decode"):
        _decode_jpeg_images({"observation/image": b"not a jpeg"})
