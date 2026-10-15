"""NumPy-only G2 arm contract, shared by dataset adapters and wire output."""

from __future__ import annotations

import numpy as np


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


def prepare_inputs(data: dict, arm: str = "both") -> dict:
    base_image = _parse_image(data["observation/image"])
    selection = arm_slice(arm, np.asarray(data["observation/state"]).shape[-1])
    left_wrist = _parse_image(data["observation/wrist_image_left"]) if arm != "right" else np.zeros_like(base_image)
    right_wrist = _parse_image(data["observation/wrist_image_right"]) if arm != "left" else np.zeros_like(base_image)

    inputs = {
        "state": np.asarray(data["observation/state"])[..., selection].copy(),
        "image": {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": left_wrist,
            "right_wrist_0_rgb": right_wrist,
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.bool_(arm != "right"),
            "right_wrist_0_rgb": np.bool_(arm != "left"),
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
