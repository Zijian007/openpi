"""G2 wholebody (VR pose) transforms for OpenPI π₀.₅.

Expected training keys after RepackTransform (see LeRobotG2DataConfig):

- observation/image              ← head camera
- observation/wrist_image_left   ← left wrist
- observation/wrist_image_right  ← right wrist
- observation/state              ← 20-D state.pose or 16-D state.joint, selected independently
- actions                        ← 20-D absolute action.pose (chunked by dataloader)
- prompt                         ← language (from LeRobot task when prompt_from_task=True)

Action layout (20-D): L/R each xyz(3) + rot6d(6) + gripper(1). Gripper close ∈ [0,1], 0=open.
"""

from __future__ import annotations

import dataclasses

from g2_policy_tools.geometry import ActionCodec
from g2_policy_tools.geometry import action_dimensions
from g2_policy_tools.geometry import convert_pose
from g2_policy_tools.geometry import matrix_pose
from g2_policy_tools.geometry import policy_reference
from g2_policy_tools.geometry import pose_matrix
from g2_policy_tools.geometry import pose_settings
from g2_policy_tools.geometry import reference_pose

# Compatibility exports; the implementation has no OpenPI dependencies.
from g2_policy_tools.geometry import rotation_matrix
from g2_policy_tools.geometry import rotation_vector
import numpy as np

from openpi import transforms
from openpi.adapters.g2.contract import G2
from openpi.models import model as _model
from openpi.shared import image_tools

__all__ = [
    "ActionCodec",
    "EncodeActions",
    "G2Inputs",
    "G2Outputs",
    "action_dimensions",
    "arm_slice",
    "convert_pose",
    "expand_arm_actions",
    "make_g2_example",
    "matrix_pose",
    "policy_reference",
    "pose_matrix",
    "pose_settings",
    "prepare_inputs",
    "reference_pose",
    "rotation_matrix",
    "rotation_vector",
]

# Effective G2 pose action dim (padded to model action_dim=32 by PadStatesAndActions).
G2_ACTION_DIM = G2.pose.dim


def arm_slice(arm: str, full_dim: int) -> slice:
    if arm not in ("both", "left", "right"):
        raise ValueError(f"invalid G2 arm {arm!r}")
    if full_dim not in (16, 20):
        raise ValueError(f"G2 input must be a full dual-arm 16/20-D vector, got {full_dim}")
    half = full_dim // 2
    return slice(None) if arm == "both" else slice(0, half) if arm == "left" else slice(half, full_dim)


def expand_arm_actions(actions, measured_pose, arm: str) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    measured = np.asarray(measured_pose, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] != 10 or measured.shape != (20,) or not np.isfinite(measured).all():
        raise ValueError("single-arm G2 output needs (T, 10) actions and a finite full (20,) EE observation")
    result = np.broadcast_to(measured, (len(actions), 20)).copy()
    result[:, arm_slice(arm, 20)] = actions
    return result


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


_BLANK_RGB = np.zeros((224, 224, 3), dtype=np.uint8)


def prepare_inputs(data: dict, arm: str = "both", cameras: tuple[str, ...] | None = None) -> dict:
    from openpi.adapters.g2.config import cameras_for_arm

    selected = cameras_for_arm(arm, cameras)
    selection = arm_slice(arm, np.asarray(data["observation/state"]).shape[-1])

    def _image(key: str, *, present: bool) -> np.ndarray:
        if not present:
            return _BLANK_RGB.copy()
        return _parse_image(data[key])

    base_image = _image("observation/image", present="head_color" in selected)
    left_wrist = _image("observation/wrist_image_left", present="hand_left_color" in selected)
    right_wrist = _image("observation/wrist_image_right", present="hand_right_color" in selected)

    inputs = {
        "state": np.asarray(data["observation/state"])[..., selection].copy(),
        "image": {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": left_wrist,
            "right_wrist_0_rgb": right_wrist,
        },
        "image_mask": {
            "base_0_rgb": np.bool_("head_color" in selected),
            "left_wrist_0_rgb": np.bool_("hand_left_color" in selected),
            "right_wrist_0_rgb": np.bool_("hand_right_color" in selected),
        },
    }

    if "actions" in data:
        actions = np.asarray(data["actions"])
        inputs["actions"] = actions[..., arm_slice(arm, actions.shape[-1])].copy()

    if "prompt" in data:
        prompt = data["prompt"]
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        inputs["prompt"] = prompt

    return inputs


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
    policy: dict | None = None
    arm: str = "both"
    cameras: tuple[str, ...] | None = None

    def __call__(self, data: dict) -> dict:
        if self.policy is None:
            return prepare_inputs(data, self.arm, self.cameras)
        obs = dict(data)
        joint = np.asarray(obs.get("observation/joint", obs["observation/state"]))
        pose = np.asarray(obs.get("observation/pose", obs["observation/state"]))
        state_settings = self.policy["state"]
        history = self.policy["timing"]["history_steps"]
        pose = np.asarray(obs.get("observation/pose_history", pose))
        joint = np.asarray(obs.get("observation/joint_history", joint))
        if pose.ndim == 1:
            pose = np.tile(pose, (history, 1))
        if joint.ndim == 1:
            joint = np.tile(joint, (history, 1))
        space = state_settings["space"]
        if space == "both" and "observation/joint" not in obs:
            raise ValueError("Combined state requires explicit observation/joint")
        if space in ("eef", "both") and pose.shape != (history, 20):
            raise ValueError(f"EEF history must have shape {(history, 20)}")
        if space in ("joint", "both") and joint.shape != (history, 16):
            raise ValueError(f"Joint history must have shape {(history, 16)}")
        pose_state = _encode_pose_state(self.policy, pose, obs) if space in ("eef", "both") else None
        if state_settings["representation"] != "absolute":
            name = state_settings["relative_to"]
            if space in ("joint", "both"):
                key = "observation/start_joint" if name == "episode_start_pose" else "observation/object_joint"
                if key not in obs:
                    raise ValueError(f"Relative joint state requires {key}")
                joint = joint.copy()
                mask = np.tile([True] * 7 + [False], 2)
                joint[..., mask] -= np.asarray(obs[key])[mask]
        states = []
        if space in ("eef", "both"):
            states.append(pose_state[..., self._select(pose_state.shape[-1])])
        if space in ("joint", "both"):
            states.append(joint[..., self._select(16)])
        state = np.concatenate(states, axis=-1).reshape(-1)
        # prepare_inputs retains the legacy camera mapping and masks.
        images = dict(obs, **{"observation/state": np.zeros(20, np.float32)})
        images.pop("actions", None)
        for key in ("observation/image", "observation/wrist_image_left", "observation/wrist_image_right"):
            image = np.asarray(images[key]) if key in images else None
            if image is not None and image.ndim == 4:
                # Fixed oldest-to-newest horizontal contact sheet; native image encoder is unchanged.
                images[key] = np.concatenate(
                    [image_tools.resize_with_pad(_parse_image(frame), 224, 224) for frame in image], axis=1
                )
        result = prepare_inputs(images, self.arm, self.cameras)
        result["state"] = state.astype(np.float32)
        result.update({key: value for key, value in obs.items() if key.startswith("observation/")})
        if "actions" in obs:
            result["actions"] = np.asarray(obs["actions"]).copy()
        return EncodeActions(ActionCodec(self.policy, self.arm))(result)

    def _select(self, dim):
        return slice(None) if self.arm == "both" else slice(0, dim // 2) if self.arm == "left" else slice(dim // 2, dim)


def _encode_pose_state(policy, history, obs):
    """Reuse pose geometry for measured history, fixed to one observation-time reference.

    State encoding is independent of the actual action quantity, representation and scope.
    Canonical base/flange history and the reference both undergo policy.pose conversion.
    """
    state = policy["state"]
    codec_policy = {
        "pose": policy["pose"],
        "action": {
            "space": "eef",
            "control_quantity": "pose_target",
            "label_source": "measured",
            "representation": "component_delta" if state["representation"] == "relative" else state["representation"],
            "relative_to": state["relative_to"],
            "delta_scope": None if state["representation"] == "absolute" else "chunk",
        },
    }
    return ActionCodec(codec_policy).encode(history, obs)


@dataclasses.dataclass(frozen=True)
class G2Outputs(transforms.DataTransformFn):
    """Truncate padded model actions back to G2 20-D pose (inference)."""

    action_dim: int = G2_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}


@dataclasses.dataclass(frozen=True)
class EncodeActions(transforms.DataTransformFn):
    codec: ActionCodec

    def __call__(self, data):
        if "actions" in data:
            data = dict(data, actions=self.codec.encode(data["actions"], data))
        # Auxiliary references never enter normalization, tokenization, or the native model.
        return {key: value for key, value in data.items() if not key.startswith("observation/")}
