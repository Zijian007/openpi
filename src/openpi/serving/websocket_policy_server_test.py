import asyncio

import numpy as np
from openpi_client import msgpack_numpy
import websockets.asyncio.server
from websockets.sync.client import connect

from openpi.serving.websocket_policy_server import WebsocketPolicyServer


def test_native_server_preserves_metadata_and_passes_observations_through():
    seen = []

    class Policy:
        def infer(self, obs):
            seen.append(obs)
            return {"actions": np.ones((2, 7), np.float32)}

    metadata = {"application": "native-test"}
    server = WebsocketPolicyServer(Policy(), metadata=metadata)

    def client(port):
        with connect(f"ws://127.0.0.1:{port}", compression=None, open_timeout=3) as websocket:
            assert msgpack_numpy.unpackb(websocket.recv(timeout=3)) == metadata
            observation = {"state": np.zeros(7, np.float32), "custom_payload": b"passed through unchanged"}
            websocket.send(msgpack_numpy.packb(observation))
            response = msgpack_numpy.unpackb(websocket.recv(timeout=3))
            assert response["actions"].shape == (2, 7)
            assert response["server_timing"]["infer_ms"] >= 0

    async def run():
        async with websockets.asyncio.server.serve(
            server._handler,  # noqa: SLF001
            "127.0.0.1",
            0,
            compression=None,
        ) as listener:
            await asyncio.wait_for(asyncio.to_thread(client, listener.sockets[0].getsockname()[1]), 10)

    asyncio.run(run())
    assert seen[0]["custom_payload"] == b"passed through unchanged"
    assert metadata == {"application": "native-test"}
