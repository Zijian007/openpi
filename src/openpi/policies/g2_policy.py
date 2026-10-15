"""G2 wholebody (VR pose) transforms for OpenPI π₀.₅.

Expected training keys after RepackTransform (see LeRobotG2DataConfig):

- observation/image              ← head camera
- observation/wrist_image_left   ← left wrist
- observation/wrist_image_right  ← right wrist
- observation/state              ← 20-D state.pose when using delta actions, else 16-D state.joint
- actions                        ← 20-D absolute action.pose (chunked by dataloader)
- prompt                         ← language (from LeRobot task when prompt_from_task=True)

Action layout (20-D): L/R each xyz(3) + rot6d(6) + gripper(1). Gripper close ∈ [0,1], 0=open.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from openpi.policies.g2_arm_contract import arm_slice as arm_slice
from openpi.policies.g2_arm_contract import expand_arm_actions as expand_arm_actions
from openpi.policies.g2_arm_contract import prepare_inputs
from openpi import transforms
from openpi.g2_lerobot_contract import G2
from openpi.models import model as _model

# Effective G2 pose action dim (padded to model action_dim=32 by PadStatesAndActions).
G2_ACTION_DIM = G2.pose.dim


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
    """Map G2 observation dict → π₀.₅ model inputs (three cameras)."""

    model_type: _model.ModelType
    arm: str = "both"

    def __call__(self, data: dict) -> dict:
        return prepare_inputs(data, self.arm)


@dataclasses.dataclass(frozen=True)
class G2Outputs(transforms.DataTransformFn):
    """Truncate padded model actions back to G2 20-D pose (inference)."""

    action_dim: int = G2_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
