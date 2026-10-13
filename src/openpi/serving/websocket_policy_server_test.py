import asyncio
from unittest.mock import patch

import cv2
import numpy as np
from openpi_client import msgpack_numpy
import pytest
import websockets.asyncio.server
from websockets.sync.client import connect

from openpi.serving.websocket_policy_server import WebsocketPolicyServer


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
