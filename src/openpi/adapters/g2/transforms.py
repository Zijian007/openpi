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

import numpy as np

from openpi import transforms
from openpi.adapters.g2.contract import G2
from openpi.models import model as _model
from openpi.shared import image_tools

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


def selected_camera_ids(arm: str = "both", cameras: tuple[str, ...] | None = None) -> tuple[str, ...]:
    arm_slice(arm, 20)
    available = (
        ("head_color", "hand_left_color", "hand_right_color")
        if arm == "both"
        else (
            "head_color",
            f"hand_{arm}_color",
        )
    )
    selected = available if cameras is None else tuple(cameras)
    if "head_color" not in selected or len(set(selected)) != len(selected) or not set(selected) <= set(available):
        raise ValueError(
            f"Invalid cameras {selected!r} for arm={arm}; include head_color and use a subset of {available}"
        )
    return selected


def prepare_inputs(data: dict, arm: str = "both", cameras: tuple[str, ...] | None = None) -> dict:
    selected = selected_camera_ids(arm, cameras)
    base_image = _parse_image(data["observation/image"])
    selection = arm_slice(arm, np.asarray(data["observation/state"]).shape[-1])
    left_wrist = (
        _parse_image(data["observation/wrist_image_left"])
        if "hand_left_color" in selected
        else np.zeros_like(base_image)
    )
    right_wrist = (
        _parse_image(data["observation/wrist_image_right"])
        if "hand_right_color" in selected
        else np.zeros_like(base_image)
    )

    inputs = {
        "state": np.asarray(data["observation/state"])[..., selection].copy(),
        "image": {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": left_wrist,
            "right_wrist_0_rgb": right_wrist,
        },
        "image_mask": {
            "base_0_rgb": np.True_,
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
        pose_state = convert_pose(pose, pose_settings(self.policy, obs)) if space in ("eef", "both") else None
        if state_settings["representation"] == "relative":
            name = state_settings["relative_to"]
            if space in ("eef", "both"):
                ref = policy_reference(self.policy, obs, name)
                width = len(ref) // 2
                mask = np.tile([True] * (width - 1) + [False], 2)
                pose_state[..., mask] -= ref[mask]
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


@dataclasses.dataclass(frozen=True)
class G2Outputs(transforms.DataTransformFn):
    """Truncate padded model actions back to G2 20-D pose (inference)."""

    action_dim: int = G2_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}


def rotation_matrix(value, representation="rot6d"):
    """Project rot6d onto SO(3); quaternions use xyzw, axis angles use radians."""
    from scipy.spatial.transform import Rotation

    value = np.asarray(value, dtype=np.float64)
    shape = value.shape[:-1]
    if representation == "rot6d":
        a, b = value[..., :3], value[..., 3:6]
        norm = np.linalg.norm(a, axis=-1, keepdims=True)
        x = np.where(norm > 1e-8, a / np.maximum(norm, 1e-8), np.array([1.0, 0.0, 0.0]))
        b = b - np.sum(x * b, axis=-1, keepdims=True) * x
        fallback = np.eye(3)[np.argmin(np.abs(x), axis=-1)]
        fallback -= np.sum(x * fallback, axis=-1, keepdims=True) * x
        b = np.where(np.linalg.norm(b, axis=-1, keepdims=True) > 1e-8, b, fallback)
        y = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
        return np.stack([x, y, np.cross(x, y)], axis=-1)
    if representation == "quaternion":
        q = value.reshape(-1, 4).copy()
        q[np.linalg.norm(q, axis=-1) < 1e-8] = [0, 0, 0, 1]
        return Rotation.from_quat(q).as_matrix().reshape(*shape, 3, 3)
    return Rotation.from_rotvec(value.reshape(-1, 3)).as_matrix().reshape(*shape, 3, 3)


def rotation_vector(matrix, representation="rot6d"):
    from scipy.spatial.transform import Rotation

    matrix = np.asarray(matrix)
    shape = matrix.shape[:-2]
    if representation == "rot6d":
        return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)
    r = Rotation.from_matrix(matrix.reshape(-1, 3, 3))
    result = r.as_quat(canonical=True) if representation == "quaternion" else r.as_rotvec()
    return result.reshape(*shape, result.shape[-1])


def pose_matrix(pose, representation="rot6d"):
    pose = np.asarray(pose)
    result = np.broadcast_to(np.eye(4), (*pose.shape[:-1], 4, 4)).copy()
    result[..., :3, :3] = rotation_matrix(pose[..., 3:-1], representation)
    result[..., :3, 3] = pose[..., :3]
    return result


def matrix_pose(matrix, gripper, representation="rot6d"):
    return np.concatenate(
        [matrix[..., :3, 3], rotation_vector(matrix[..., :3, :3], representation), np.asarray(gripper)[..., None]],
        axis=-1,
    ).astype(np.float32)


def convert_pose(pose, settings, *, inverse=False):
    """Canonical base/flange rot6d ↔ configured parent/EE/rotation, per selected arm."""
    pose = np.asarray(pose)
    if not np.isfinite(pose).all():
        raise ValueError("Pose contains non-finite values")
    if (
        settings["rotation"] == "rot6d"
        and settings.get("base_to_parent") is None
        and settings.get("flange_to_tcp") is None
    ):
        return pose.copy()
    width = 10 if not inverse else 4 + {"rot6d": 6, "quaternion": 4, "axis_angle": 3}[settings["rotation"]]
    arms = pose.shape[-1] // width
    pieces = []
    for i in range(arms):
        p = pose[..., i * width : (i + 1) * width]
        parent = np.asarray(settings["base_to_parent"] if settings.get("base_to_parent") is not None else np.eye(4))
        tcp = settings.get("flange_to_tcp")
        tcp = np.eye(4) if tcp is None else np.asarray(tcp["left" if i == 0 else "right"])
        if inverse:
            t = np.linalg.inv(parent) @ pose_matrix(p, settings["rotation"]) @ np.linalg.inv(tcp)
            pieces.append(matrix_pose(t, p[..., -1]))
        else:
            t = parent @ pose_matrix(p) @ tcp
            pieces.append(matrix_pose(t, p[..., -1], settings["rotation"]))
    return np.concatenate(pieces, axis=-1)


def pose_settings(policy, obs):
    settings = dict(policy["pose"])
    if isinstance(settings.get("base_to_parent"), str) and settings["base_to_parent"] == "observation":
        if "observation/base_to_parent" not in obs:
            raise ValueError("Dynamic parent frame requires observation/base_to_parent")
        settings["base_to_parent"] = np.asarray(obs["observation/base_to_parent"]).tolist()
        from openpi.adapters.g2.config import _validate_rigid_transform

        _validate_rigid_transform(settings["base_to_parent"], "observation/base_to_parent")
    return settings


def action_dimensions(policy, arm="both"):
    action = policy["action"]
    width = {
        "pose_target": 4 + {"rot6d": 6, "quaternion": 4, "axis_angle": 3}[policy["pose"]["rotation"]],
        "joint_position": 8,
        "joint_velocity": 8,
        "eef_twist": 7,
        "joint_torque": 8,
    }[action["control_quantity"]]
    wire = (
        20 if action["control_quantity"] == "pose_target" else 14 if action["control_quantity"] == "eef_twist" else 16
    )
    return width * (2 if arm == "both" else 1), wire


def reference_pose(data, name):
    keys = {
        "current_measured_pose": "observation/pose",
        "current_command_pose": "observation/command_pose",
        "previous_command_pose": "observation/previous_command_pose",
        "episode_start_pose": "observation/start_pose",
        "object_pose": "observation/object_pose",
    }
    key = keys[name]
    if key not in data:
        raise ValueError(f"Reference {name} requires {key}")
    result = np.asarray(data[key], dtype=np.float32)
    if result.shape != (20,) or not np.isfinite(result).all():
        raise ValueError(f"{key} must be a finite full (20,) EEF pose")
    return result


def policy_reference(policy, obs, name):
    settings = pose_settings(policy, obs)
    if name == "object_pose":
        # An object is not a flange: its origin must not be moved by tool calibration.
        settings["flange_to_tcp"] = None
    return convert_pose(reference_pose(obs, name), settings)


@dataclasses.dataclass(frozen=True)
class ActionCodec:
    """The same codec encodes labels/RTC prefixes and decodes policy outputs."""

    policy: dict
    arm: str = "both"

    @property
    def dimensions(self):
        return action_dimensions(self.policy, self.arm)

    def _selection(self, dim):
        half = dim // 2
        return slice(None) if self.arm == "both" else slice(0, half) if self.arm == "left" else slice(half, dim)

    def encode(self, actions, obs):
        action = self.policy["action"]
        values = np.asarray(actions, dtype=np.float32).copy()
        if values.ndim != 2 or values.shape[-1] != self.dimensions[1] or not np.isfinite(values).all():
            raise ValueError(f"Actions must be finite (T, {self.dimensions[1]}) wire values")
        if action["control_quantity"] == "pose_target":
            values = convert_pose(values, pose_settings(self.policy, obs))
            if action["representation"] != "absolute":
                ref = policy_reference(self.policy, obs, action["relative_to"])
                values = self._pose_delta(values, ref, inverse=False)
        elif action["control_quantity"] in ("joint_position", "joint_velocity", "joint_torque"):
            if action["representation"] == "component_delta":
                ref = np.asarray(obs["observation/joint"])
                refs = np.concatenate([ref[None], values[:-1]], axis=0) if action["delta_scope"] == "step" else ref
                mask = np.tile([True] * 7 + [False], 2)
                values[..., mask] -= refs[..., mask]
        elif self.policy["pose"].get("base_to_parent") is not None:
            r = np.asarray(pose_settings(self.policy, obs)["base_to_parent"])[:3, :3]
            for offset in (0, 7):
                values[..., offset : offset + 3] = values[..., offset : offset + 3] @ r.T
                values[..., offset + 3 : offset + 6] = values[..., offset + 3 : offset + 6] @ r.T
        return values[..., self._selection(values.shape[-1])].copy()

    def decode(self, actions, obs):
        action = self.policy["action"]
        encoded_dim, wire_dim = self.dimensions
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or not len(values) or values.shape[-1] < encoded_dim or not np.isfinite(values).all():
            raise ValueError(f"Model actions must be finite with at least {encoded_dim} columns")
        values = values[..., :encoded_dim].copy()
        if self.arm != "both":
            if action["control_quantity"] == "pose_target":
                full = convert_pose(reference_pose(obs, "current_measured_pose"), pose_settings(self.policy, obs))
            elif action["control_quantity"] == "joint_position":
                full = np.asarray(obs["observation/joint"])
            else:
                full = np.zeros(wire_dim, dtype=np.float32)
            values_full = np.broadcast_to(full, (len(values), len(full))).copy()
            values_full[..., self._selection(len(full))] = values
            values = values_full
        if action["control_quantity"] == "pose_target":
            if action["representation"] != "absolute":
                ref = policy_reference(self.policy, obs, action["relative_to"])
                values = self._pose_delta(values, ref, inverse=True)
            values = convert_pose(values, pose_settings(self.policy, obs), inverse=True)
            if self.arm != "both":
                other = slice(10, 20) if self.arm == "left" else slice(0, 10)
                values[..., other] = reference_pose(obs, "current_measured_pose")[other]
        elif action["representation"] == "component_delta":
            ref = np.asarray(obs["observation/joint"])
            mask = np.tile([True] * 7 + [False], 2)
            if action["delta_scope"] == "step":
                values[..., mask] = np.cumsum(values[..., mask], axis=0) + ref[mask]
            else:
                values[..., mask] += ref[mask]
            if self.arm != "both":
                other = slice(8, 16) if self.arm == "left" else slice(0, 8)
                values[..., other] = ref[other]
        elif action["control_quantity"] == "eef_twist" and self.policy["pose"].get("base_to_parent") is not None:
            r = np.asarray(pose_settings(self.policy, obs)["base_to_parent"])[:3, :3]
            for offset in (0, 7):
                values[..., offset : offset + 3] = values[..., offset : offset + 3] @ r
                values[..., offset + 3 : offset + 6] = values[..., offset + 3 : offset + 6] @ r
        return values

    def _pose_delta(self, values, ref, *, inverse):
        representation = self.policy["action"]["representation"]
        rotation = self.policy["pose"]["rotation"]
        width = len(ref) // 2
        step = self.policy["action"]["delta_scope"] == "step"
        if representation == "component_delta":
            mask = np.tile([True] * (width - 1) + [False], 2)
            out = values.copy()
            if inverse:
                out[..., mask] = (np.cumsum(values[..., mask], axis=0) if step else values[..., mask]) + ref[mask]
            else:
                refs = np.concatenate([ref[None], values[:-1]], axis=0) if step else ref
                out[..., mask] -= refs[..., mask]
            return out
        out = []
        current = ref.copy()
        for value in values:
            parts = []
            for offset in (0, width):
                p, r = value[offset : offset + width], current[offset : offset + width]
                tp, tr = pose_matrix(p, rotation), pose_matrix(r, rotation)
                if representation == "se3_body_delta":
                    t = tr @ tp if inverse else np.linalg.inv(tr) @ tp
                elif representation == "se3_spatial_delta":
                    t = tp @ tr if inverse else tp @ np.linalg.inv(tr)
                else:  # Translation in parent coordinates, orientation relative to the reference EE.
                    t = tp.copy()
                    t[:3, :3] = tr[:3, :3] @ tp[:3, :3] if inverse else tr[:3, :3].T @ tp[:3, :3]
                    t[:3, 3] = p[:3] + r[:3] if inverse else p[:3] - r[:3]
                parts.append(matrix_pose(t, p[-1], rotation))
            result = np.concatenate(parts)
            out.append(result)
            if step:
                current = result if inverse else value
        return np.asarray(out, dtype=np.float32)


@dataclasses.dataclass(frozen=True)
class EncodeActions(transforms.DataTransformFn):
    codec: ActionCodec

    def __call__(self, data):
        if "actions" in data:
            data = dict(data, actions=self.codec.encode(data["actions"], data))
        # Auxiliary references never enter normalization, tokenization, or the native model.
        return {key: value for key, value in data.items() if not key.startswith("observation/")}
