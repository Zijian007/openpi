import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from g2_policy_protocol import codec
from g2_policy_protocol import protocol
import numpy as np
import pytest
import websockets

from openpi.adapters.g2.server import G2PolicyServer


@pytest.mark.parametrize("requires_joint_state", [False, True])
@pytest.mark.parametrize("arm", ["both", "right"])
def test_configured_camera_subset_and_state_source_reach_policy(requires_joint_state, arm):
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

    with patch("openpi.adapters.g2.server.cv2", None):
        server = G2PolicyServer(
            Policy(),
            metadata={
                "arm": arm,
                "required_cameras": ["head_color"],
                "requires_joint_state": requires_joint_state,
                "execution_steps": 10,
            },
        )
    websocket = Websocket()
    asyncio.run(server._handler(websocket))  # noqa: SLF001
    metadata, response = websocket.sent
    assert metadata["required_cameras"] == ["head_color"]
    assert metadata["requires_joint_state"] == requires_joint_state
    assert metadata["execution_steps"] == 10
    assert response["type"] == protocol.ACTION_TYPE
    keys = {"observation/state", "observation/image", "prompt"}
    if requires_joint_state:
        keys.update({"observation/pose", "observation/joint"})
        np.testing.assert_array_equal(seen[0]["observation/pose"], np.ones(20, np.float32))
    assert set(seen[0]) == keys
    expected = np.full(16, 2, np.float32) if requires_joint_state else np.ones(20, np.float32)
    np.testing.assert_array_equal(seen[0]["observation/state"], expected)


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize(
    ("quantity", "space", "dim"),
    [
        ("joint_position", "joint", 16),
        ("joint_velocity", "joint", 16),
        ("joint_torque", "joint", 16),
        ("eef_twist", "eef", 14),
    ],
)
def test_non_pose_wire_contract_requires_explicit_client_opt_in(accepted, quantity, space, dim):
    import copy

    from openpi.adapters.g2.config import POLICY_DEFAULTS

    settings = copy.deepcopy(POLICY_DEFAULTS)
    settings["action"].update(
        control_quantity=quantity, space=space, representation="absolute", relative_to=None, delta_scope=None
    )
    calls = []

    class Policy:
        def infer(self, obs):
            calls.append(obs)
            return {"actions": np.ones((3, dim), np.float32)}

    class Websocket:
        request = SimpleNamespace(path="/")

        def __init__(self):
            self.sent = []

        async def send(self, message):
            self.sent.append(codec.loads(message))

        async def recv(self):
            if len(self.sent) > 1:
                raise websockets.ConnectionClosed(None, None)
            message = protocol.build_request(
                request_id=3,
                prompt="tea",
                state=np.zeros(20, np.float32),
                joint_state=np.zeros(16, np.float32),
                images={"head_color": np.zeros((224, 224, 3), np.uint8)},
                metadata=self.sent[0],
            )
            if accepted:
                message["protocol"] = "g2.openpi.v1"
            return codec.dumps(message)

    with patch("openpi.adapters.g2.server.cv2", None):
        server = G2PolicyServer(
            Policy(),
            metadata={"policy_contract": settings, "requires_joint_state": True, "required_cameras": ["head_color"]},
        )
    ws = Websocket()
    asyncio.run(server._handler(ws))  # noqa: SLF001
    meta, response = ws.sent
    assert meta["protocol"] == "g2.openpi.v1"
    if accepted:
        assert response["actions"].shape == (3, dim)
        assert response["control_quantity"] == quantity
        assert response["protocol"] == "g2.openpi.v1"
        assert len(calls) == 1
    else:
        assert response["type"] == protocol.ERROR_TYPE
        assert calls == []
