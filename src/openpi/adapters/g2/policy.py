"""Checkpoint restoration and robot output adaptation around the native Policy."""

from collections import deque
import dataclasses
from pathlib import Path

import numpy as np
from openpi_client import base_policy

from openpi.adapters.g2 import config as g2_config
from openpi.adapters.g2.data import LeRobotG2DataConfig
from openpi.adapters.g2.data import policy_for_data
from openpi.adapters.g2.transforms import ActionCodec
from openpi.adapters.g2.transforms import expand_arm_actions
from openpi.adapters.g2.transforms import selected_camera_ids
from openpi.policies import policy_config as native_policy_config
from openpi.policies.policy import PolicyRecorder
from openpi.shared import download
from openpi.training.config import TrainConfig


class G2Policy(base_policy.BasePolicy):
    """Adapt decoded model actions to the full G2 wire layout, outside native inference."""

    def __init__(self, policy, settings=None):
        self._policy = policy
        self._settings = settings
        self._history = deque(maxlen=settings["timing"]["history_steps"] if settings else 1)
        self._start = None
        self._previous_command = None

    @property
    def metadata(self) -> dict:
        return self._policy.metadata

    def reset(self) -> None:
        self._history.clear()
        self._start = None
        self._previous_command = None
        self._policy.reset()

    def infer(self, obs: dict, *, noise=None) -> dict:
        if self._settings is None:
            result = self._policy.infer(obs) if noise is None else self._policy.infer(obs, noise=noise)
            if self.metadata.get("arm", "both") != "both":
                result = dict(result)
                pose = obs.get("observation/pose", obs["observation/state"])
                result["actions"] = expand_arm_actions(result["actions"], pose, self.metadata["arm"])
            return result
        context = self.prepare_observation(obs)
        result = self._policy.infer(context) if noise is None else self._policy.infer(context, noise=noise)
        result = dict(result)
        result["actions"] = ActionCodec(self._settings, self.metadata.get("arm", "both")).decode(
            result["actions"], context
        )
        self._history.append(
            {
                key: np.asarray(obs.get(key, value))[-1].copy()
                if np.asarray(obs.get(key, value)).ndim == 4
                else np.asarray(obs.get(key, value)).copy()
                for key, value in context.items()
                if key
                in (
                    "observation/pose",
                    "observation/joint",
                    "observation/image",
                    "observation/wrist_image_left",
                    "observation/wrist_image_right",
                )
            }
        )
        if self._start is None:
            self._start = {
                key: np.asarray(value).copy()
                for key, value in context.items()
                if key in ("observation/pose", "observation/joint")
            }
        if "observation/command_pose" in context:
            self._previous_command = np.asarray(context["observation/command_pose"]).copy()
        return result

    def new_session(self):
        return G2Policy(self._policy, self._settings)

    def prepare_observation(self, obs):
        context = dict(obs)
        space = self._settings["state"]["space"]
        if space != "joint":
            context.setdefault("observation/pose", context["observation/state"])
        if space == "joint":
            context.setdefault("observation/joint", context["observation/state"])
        for suffix in ("pose", "joint"):
            key = f"observation/{suffix}"
            if key in context:
                start = (self._start or context)[key]
                context.setdefault(f"observation/start_{suffix}", start)
        if self._previous_command is not None:
            context.setdefault("observation/previous_command_pose", self._previous_command)
        elif "observation/command_pose" in context:
            context.setdefault("observation/previous_command_pose", context["observation/command_pose"])
        count = self._settings["timing"]["history_steps"]
        frames = [*self._history, context][-count:]
        frames = [frames[0]] * (count - len(frames)) + frames
        for suffix in ("pose", "joint"):
            key = f"observation/{suffix}"
            if key in context:
                context.setdefault(f"observation/{suffix}_history", np.stack([frame[key] for frame in frames]))
        for key in ("observation/image", "observation/wrist_image_left", "observation/wrist_image_right"):
            if key in context and count > 1 and np.asarray(context[key]).ndim == 3:
                context[key] = np.stack([frame[key] for frame in frames])
        return context


def create_trained_policy(train_config: TrainConfig, checkpoint_dir, **kwargs) -> G2Policy:
    """Restore adapter settings, then let native OpenPI load weights and transforms."""
    if not isinstance(train_config.data, LeRobotG2DataConfig):
        raise ValueError("The G2 policy loader requires a G2 data adapter")
    checkpoint_dir = Path(download.maybe_download(str(checkpoint_dir)))
    snapshot = checkpoint_dir / "assets" / g2_config.CHECKPOINT_CONFIG
    if snapshot.is_file():
        payload = g2_config.read_yaml(snapshot)
        restored = g2_config.build_train_config(payload["settings"], tuple(payload["extra_args"]))
        if restored.name != train_config.name:
            raise ValueError(f"Checkpoint config is {restored.name}, requested {train_config.name}")
        train_config = restored

    arm = train_config.data.arm
    settings = policy_for_data(train_config.data)
    action_dim, wire_dim = ActionCodec(settings, arm).dimensions
    metadata = {
        **(train_config.policy_metadata or {}),
        "arm": arm,
        "action_dim": wire_dim,
        "model_action_dim": action_dim,
        "policy_contract": settings,
        "required_cameras": list(selected_camera_ids(arm, train_config.data.cameras)),
        "requires_joint_state": train_config.data.resolved_state_space in ("joint", "both")
        or settings["action"]["space"] == "joint",
    }
    configured = dataclasses.replace(train_config, policy_metadata=metadata)
    return G2Policy(native_policy_config.create_trained_policy(configured, checkpoint_dir, **kwargs), settings)


class G2PolicyRecorder(PolicyRecorder):
    """Keep reference/history state and record paths separate for each connection."""

    def new_session(self):
        import uuid

        return G2PolicyRecorder(self._policy.new_session(), self._record_dir / uuid.uuid4().hex)
