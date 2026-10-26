"""G2 protocol boundary; model-facing OpenPI observations remain unchanged."""
import logging
import time

import numpy as np
import websockets
from g2_policy_protocol import codec, protocol

from openpi.serving.websocket_policy_server import (
    WebsocketPolicyServer, _decode_jpeg_images, cv2,
)

logger = logging.getLogger(__name__)
# Wire camera id → OpenPI observation key. ``prepare_inputs`` renames those again
# to base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb for the model.
_NATIVE_IMAGES = dict(zip(protocol.CAMERA_IDS, (
    'observation/image', 'observation/wrist_image_left', 'observation/wrist_image_right',
)))


def connection_metadata(policy_metadata):
    """Copy the checkpoint fields that belong on the g2.policy.v2 handshake.

    ``action_dim`` and ``model_action_dim`` stay on the policy for RTC prefix width.
    The client always receives actions of shape (T, 20).
    """
    native = dict(policy_metadata or {})
    arm = native.get('arm', 'both')
    required = list(native.get('required_cameras') or protocol.cameras_for_arm(arm))
    return dict(
        policy_hz=native.get('policy_hz', 30.0),
        action_horizon=native.get('action_horizon', 50),
        arm=arm,
        image_encoding='jpeg' if cv2 is not None else 'raw',
        required_cameras=required,
        image_resize='letterbox',
        image_sizes={key: [224, 224] for key in required},
        rtc_supported=bool(native.get('rtc_supported', False)),
        requires_joint_state=bool(native.get('requires_joint_state', False)),
        extensions={'backend': 'openpi', 'policy_type': 'pi05'},
    )


class G2PolicyServer(WebsocketPolicyServer):
    def __init__(self, policy, host='0.0.0.0', port=None, metadata=None):
        super().__init__(policy, host=host, port=port, metadata=metadata)
        self._contract = connection_metadata(metadata)
        self._requires_joint_state = self._contract['requires_joint_state']
        protocol.build_metadata(**self._contract)  # Fail before listening on invalid capabilities.

    async def _handler(self, websocket):
        path = getattr(getattr(websocket, 'request', None), 'path', '/')
        encoding = codec.encoding_from_path(path)
        metadata = protocol.build_metadata(**self._contract)
        await websocket.send(codec.dumps(metadata, encoding=encoding))
        while True:
            payload = None
            try:
                raw = await websocket.recv()
                start = time.monotonic()
                payload = codec.loads(raw, encoding=encoding)
                request = protocol.parse_request(payload, metadata=metadata)
                unpack_ms = (time.monotonic() - start) * 1000
                state = request['joint_state'] if self._requires_joint_state else request['state']
                obs = {'observation/state': state, 'prompt': request['prompt']}
                obs.update({_NATIVE_IMAGES[key]: image for key, image in request['images'].items()})
                if request['rtc_enabled']:
                    obs.update(rtc_enabled=True, inference_delay=request['inference_delay'])
                    if request['prev_chunk_left_over'] is not None:
                        obs['prev_chunk_left_over'] = request['prev_chunk_left_over']
                decode_start = time.monotonic()
                _decode_jpeg_images(obs)
                protocol.validate_images({key: obs[_NATIVE_IMAGES[key]] for key in request["images"]},
                                         metadata=metadata, decoded=True)
                decode_ms = (time.monotonic() - decode_start) * 1000
                infer_start = time.monotonic()
                result = self._policy.infer(obs)
                infer_ms = (time.monotonic() - infer_start) * 1000
                actions = np.asarray(result['actions'], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] not in (protocol.POSE_DIM, 32):
                    raise ValueError('OpenPI adapter requires expanded 20-D or padded 32-D actions')
                response = protocol.build_response(
                    request_id=request['request_id'], actions=actions[:, :protocol.POSE_DIM],
                    server_timing=dict(infer_ms=infer_ms, unpack_ms=unpack_ms, decode_ms=decode_ms,
                                       process_ms=(time.monotonic() - start) * 1000),
                    policy_timing=result.get('policy_timing'), rtc_info=result.get('rtc_info'),
                )
                await websocket.send(codec.dumps(response, encoding=encoding))
            except websockets.ConnectionClosed:
                return
            except Exception as exc:
                logger.exception('G2 inference failed')
                request_id = payload.get('request_id', 0) if isinstance(payload, dict) else 0
                if type(request_id) is not int or request_id < 0:
                    request_id = 0
                try:
                    await websocket.send(codec.dumps(protocol.build_error(
                        request_id=request_id, message=f'{type(exc).__name__}: {exc}'), encoding=encoding))
                except websockets.ConnectionClosed:
                    return
