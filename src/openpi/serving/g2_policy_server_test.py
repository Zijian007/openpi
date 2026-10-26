import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from g2_policy_protocol import codec
from g2_policy_protocol import protocol
import numpy as np
import pytest
import websockets

from openpi.serving.g2_policy_server import G2PolicyServer


@pytest.mark.parametrize("requires_joint_state", [False, True])
def test_configured_camera_subset_and_state_source_reach_policy(requires_joint_state):
    seen = []

    class Policy:
        def infer(self, obs):
            seen.append(obs)
            return {"actions": np.zeros((2, 20), np.float32)}

    class Websocket:
        request = SimpleNamespace(path="/")

        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(codec.loads(payload, encoding=codec.encoding_from_path("/")))

        async def recv(self):
            if len(self.sent) > 1:
                raise websockets.ConnectionClosed(None, None)
            payload = protocol.build_request(
                request_id=1,
                prompt="tea",
                state=np.ones(20, np.float32),
                joint_state=np.full(16, 2, np.float32) if requires_joint_state else None,
                images={"head_color": np.zeros((224, 224, 3), np.uint8)},
                metadata=self.sent[0],
            )
            return codec.dumps(payload, encoding=codec.encoding_from_path("/"))

    with patch("openpi.serving.g2_policy_server.cv2", None):
        server = G2PolicyServer(
            Policy(), metadata={"required_cameras": ["head_color"], "requires_joint_state": requires_joint_state}
        )
    websocket = Websocket()
    asyncio.run(server._handler(websocket))  # noqa: SLF001
    metadata, response = websocket.sent
    assert metadata["required_cameras"] == ["head_color"]
    assert metadata["requires_joint_state"] == requires_joint_state
    assert response["type"] == protocol.ACTION_TYPE
    assert set(seen[0]) == {"observation/state", "observation/image", "prompt"}
    expected = np.full(16, 2, np.float32) if requires_joint_state else np.ones(20, np.float32)
    np.testing.assert_array_equal(seen[0]["observation/state"], expected)
