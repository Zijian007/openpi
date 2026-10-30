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
    assert "execution_steps" not in metadata
    assert response["type"] == protocol.ACTION_TYPE
    keys = {"observation/state", "observation/pose", "observation/image", "prompt"}
    np.testing.assert_array_equal(seen[0]["observation/pose"], np.ones(20, np.float32))
    if requires_joint_state:
        keys.add("observation/joint")
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
    ],
)
def test_non_pose_wire_uses_shared_contract_and_rejects_private_dialect(accepted, quantity, space, dim):
    import copy

    from openpi.adapters.g2.config import POLICY_DEFAULTS

    settings = copy.deepcopy(POLICY_DEFAULTS)
    from openpi.adapters.g2.tests.geometry_test import set_joint_action

    set_joint_action(settings, quantity)
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
            metadata={"policy_contract": settings, "requires_joint_state": True, "required_cameras": ["head_color"], "action_horizon": 3},
        )
    ws = Websocket()
    asyncio.run(server._handler(ws))  # noqa: SLF001
    meta, response = ws.sent
    assert meta["protocol"] == "g2.policy.v3"
    if not accepted:
        assert response["actions"].shape == (3, dim)
        assert meta["action_spec"]["control_quantity"] == quantity
        protocol.parse_response(response, request_id=3, metadata=meta)
        assert len(calls) == 1
    else:
        assert response["type"] == protocol.ERROR_TYPE
        assert calls == []


def test_joint_only_checkpoint_does_not_require_wire_pose():
    import copy

    from openpi.adapters.g2.config import POLICY_DEFAULTS
    from openpi.adapters.g2.server import connection_metadata

    settings = copy.deepcopy(POLICY_DEFAULTS)
    from openpi.adapters.g2.tests.geometry_test import set_joint_action
    from openpi.adapters.g2.tests.geometry_test import set_state_space

    set_state_space(settings, "joint")
    set_joint_action(settings, "position")
    meta = connection_metadata(
        {
            "policy_contract": settings,
            "requires_joint_state": True,
            "required_cameras": ["head_color"],
            "requires_prompt": True,
        }
    )
    assert meta["observation_spec"]["pose"] is False
    assert meta["observation_spec"]["joint"] is True
    assert meta["observation_spec"]["prompt"] is True
    assert "pose_history" not in meta["observation_spec"]["context_keys"]
    assert "execution_steps" not in meta


def test_v3_handshake_does_not_advertise_client_execution_budget():
    import copy

    from openpi.adapters.g2.config import POLICY_DEFAULTS
    from openpi.adapters.g2.server import connection_metadata

    settings = copy.deepcopy(POLICY_DEFAULTS)
    contract = connection_metadata({"policy_contract": settings, "execution_steps": 10, "action_horizon": 50})
    assert "execution_steps" not in contract
    handshake = protocol.build_metadata(**contract)
    assert handshake["protocol"] == protocol.PROTOCOL
    assert "execution_steps" not in handshake


@pytest.mark.parametrize("space", ["joint", "eef", "both"])
def test_relative_state_only_declares_references_used_by_its_space(space):
    import copy

    from openpi.adapters.g2.config import POLICY_DEFAULTS
    from openpi.adapters.g2.server import OpenPIBackend
    from openpi.adapters.g2.server import connection_metadata

    settings = copy.deepcopy(POLICY_DEFAULTS)
    from openpi.adapters.g2.tests.geometry_test import set_encoding
    from openpi.adapters.g2.tests.geometry_test import set_joint_action
    from openpi.adapters.g2.tests.geometry_test import set_state_space

    set_state_space(settings, space)
    for kind in ("pose", "joint"):
        if settings["state"][kind]["enabled"]:
            set_encoding(settings["state"][kind], "component_delta", scope="chunk", source="action")
    set_joint_action(settings, "position")
    meta = protocol.build_metadata(**connection_metadata({"policy_contract": settings, "required_cameras": []}))
    expected = ({"command_pose"} if space in ("eef", "both") else set()) | (
        {"command_joint"} if space in ("joint", "both") else set()
    )
    assert set(meta["observation_spec"]["context_keys"]) == expected
    context = {key: np.zeros(16 if key == "command_joint" else 20, np.float32) for key in expected}
    request = protocol.build_request(
        request_id=1,
        prompt="tea",
        images={},
        metadata=meta,
        state=np.zeros(20) if meta["observation_spec"]["pose"] else None,
        joint_state=np.zeros(16),
        context=context,
    )
    observed = []

    class Policy:
        def infer(self, obs):
            observed.append(obs)
            return {"actions": np.zeros((3, 16), np.float32)}

    result = OpenPIBackend(Policy(), meta).infer(request)
    assert result["actions"].shape == (3, 16)
    assert {f"observation/{key}" for key in expected} <= observed[0].keys()
    if space == "joint":
        assert "observation/pose" not in observed[0]
        assert "observation/command_pose" not in observed[0]
