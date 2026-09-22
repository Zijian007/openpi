"""G2 wholebody (VR pose) transforms for OpenPI π₀.₅.

Expected training keys after RepackTransform (see LeRobotG2DataConfig):

- observation/image              ← head camera
- observation/wrist_image_left   ← left wrist
- observation/wrist_image_right  ← right wrist
- observation/state              ← 20-D EE when using delta actions, else 16-D joints
- actions                        ← 20-D absolute pose commands (chunked by dataloader)
- prompt                         ← language (from LeRobot task when prompt_from_task=True)

Action layout (20-D): L/R each xyz(3) + rot6d(6) + gripper(1). Gripper close ∈ [0,1], 0=open.
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Effective G2 pose action dim (padded to model action_dim=32 by PadStatesAndActions).
G2_ACTION_DIM = 20


def make_g2_example() -> dict:
    """Random observation dict matching G2Inputs / serve_policy client keys."""
    return {
        "observation/state": np.random.rand(G2_ACTION_DIM).astype(np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "Pick up the drink and put it in the box",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class G2Inputs(transforms.DataTransformFn):
    """Map G2 observation dict → π₀.₅ model inputs (three cameras)."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        left_wrist = _parse_image(data["observation/wrist_image_left"])
        right_wrist = _parse_image(data["observation/wrist_image_right"])

        inputs = {
            "state": np.asarray(data["observation/state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class G2Outputs(transforms.DataTransformFn):
    """Truncate padded model actions back to G2 20-D pose (inference)."""

    action_dim: int = G2_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
