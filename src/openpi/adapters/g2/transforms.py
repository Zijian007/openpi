"""Prepare G2 cameras and encode state/action semantics for native OpenPI models."""

from __future__ import annotations

import dataclasses

from g2_policy_protocol.data.contract import G2
from g2_policy_protocol.data.encoding import ActionCodec
from g2_policy_protocol.data.encoding import encode_state
import numpy as np

from openpi import transforms
from openpi.adapters.g2.config import cameras_for_arm
from openpi.adapters.g2.contract import CAMERA_OBSERVATION_KEYS
from openpi.shared import image_tools

__all__ = ["G2_ACTION_DIM", "G2Inputs", "G2Outputs", "arm_slice", "make_g2_example", "prepare_inputs"]

# Full dual-arm pose width; native models pad encoded actions to their own width.
G2_ACTION_DIM = G2.pose.dim
_MODEL_IMAGE_KEYS = {
    "head_color": "base_0_rgb",
    "hand_left_color": "left_wrist_0_rgb",
    "hand_right_color": "right_wrist_0_rgb",
}
_BLANK_RGB = np.zeros((224, 224, 3), dtype=np.uint8)


def arm_slice(arm: str, full_dim: int) -> slice:
    if arm not in ("both", "left", "right"):
        raise ValueError(f"invalid G2 arm {arm!r}")
    if full_dim not in (16, 20):
        raise ValueError(f"G2 input must be a full dual-arm 16/20-D vector, got {full_dim}")
    half = full_dim // 2
    return slice(None) if arm == "both" else slice(0, half) if arm == "left" else slice(half, full_dim)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


def _prepare_images(data, arm, cameras, *, history=False):
    selected = cameras_for_arm(arm, cameras)
    images, masks = {}, {}
    for camera, key in _MODEL_IMAGE_KEYS.items():
        present = camera in selected
        masks[key] = np.bool_(present)
        if not present:
            images[key] = _BLANK_RGB.copy()
            continue
        image = np.asarray(data[CAMERA_OBSERVATION_KEYS[camera]])
        if history and image.ndim == 4:
            # Oldest-to-newest horizontal contact sheet; native image encoder is unchanged.
            images[key] = np.concatenate(
                [image_tools.resize_with_pad(_parse_image(frame), 224, 224) for frame in image], axis=1
            )
        else:
            images[key] = _parse_image(image)
    result = {"image": images, "image_mask": masks}
    if "prompt" in data:
        prompt = data["prompt"]
        result["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
    return result


def prepare_inputs(data: dict, arm: str = "both", cameras: tuple[str, ...] | None = None) -> dict:
    """Prepare canonical vectors for legacy transform chains without a policy contract."""
    state = np.asarray(data["observation/state"])
    selection = arm_slice(arm, state.shape[-1])
    result = _prepare_images(data, arm, cameras)
    result["state"] = state[..., selection].copy()
    if "actions" in data:
        actions = np.asarray(data["actions"])
        result["actions"] = actions[..., arm_slice(arm, actions.shape[-1])].copy()
    return result


def make_g2_example() -> dict:
    """Random observation dict matching G2Inputs / serve_policy client keys."""
    return {
        "observation/state": np.random.rand(G2_ACTION_DIM).astype(np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "Pick up the drink and put it in the box",
    }


@dataclasses.dataclass(frozen=True)
class G2Inputs(transforms.DataTransformFn):
    """Encode state, labels and camera history without exposing auxiliary references."""

    policy: dict | None = None
    arm: str = "both"
    cameras: tuple[str, ...] | None = None
    _codec: ActionCodec | None = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_codec", ActionCodec(self.policy, self.arm) if self.policy is not None else None)

    def __call__(self, data: dict) -> dict:
        if self.policy is None:
            return prepare_inputs(data, self.arm, self.cameras)
        result = _prepare_images(data, self.arm, self.cameras, history=True)
        result["state"] = encode_state(self.policy, data, self.arm)
        if "actions" in data:
            result["actions"] = self._codec.encode(data["actions"], data)
        return result


@dataclasses.dataclass(frozen=True)
class G2Outputs(transforms.DataTransformFn):
    """Trim padded model actions to the configured encoded action width."""

    action_dim: int = G2_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
