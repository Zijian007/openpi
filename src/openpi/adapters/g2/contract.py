"""Shared G2 disk contract and OpenPI observation key mapping."""

from g2_policy_tools.contract import G2
from g2_policy_tools.contract import SIDES
from g2_policy_tools.contract import CameraKeys
from g2_policy_tools.contract import DiskColumnKeys
from g2_policy_tools.contract import G2LerobotContract
from g2_policy_tools.contract import VectorLayout

CAMERA_OBSERVATION_KEYS = {
    "head_color": "observation/image",
    "hand_left_color": "observation/wrist_image_left",
    "hand_right_color": "observation/wrist_image_right",
}

__all__ = [
    "CAMERA_OBSERVATION_KEYS",
    "G2",
    "SIDES",
    "CameraKeys",
    "DiskColumnKeys",
    "G2LerobotContract",
    "VectorLayout",
]
