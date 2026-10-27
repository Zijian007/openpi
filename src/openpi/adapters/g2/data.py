"""G2 dataset construction, episode-safe temporal sampling, and native config integration."""

import copy
import dataclasses
import pathlib

import numpy as np
from typing_extensions import override

from openpi import transforms as native_transforms
from openpi.adapters.g2 import transforms as g2
from openpi.adapters.g2.config import POLICY_DEFAULTS
from openpi.models import model as _model
from openpi.training.config import DataConfig
from openpi.training.config import DataConfigFactory
from openpi.training.config import ModelTransformFactory


def policy_for_data(data):
    if data.policy is not None:
        return copy.deepcopy(data.policy)
    policy = copy.deepcopy(POLICY_DEFAULTS)
    policy["state"]["space"] = data.resolved_state_space
    if not data.use_delta_actions:
        policy["action"].update(representation="absolute", relative_to=None, delta_scope=None)
    return policy


def label_column(policy):
    action = policy["action"]
    prefix = "action" if action["label_source"] == "command" else "state"
    suffix = {
        "pose_target": "pose",
        "joint_position": "joint",
        "joint_velocity": "joint_velocity",
        "eef_twist": "eef_twist",
        "joint_torque": "joint_torque",
    }[action["control_quantity"]]
    return f"{prefix}.{suffix}"


@dataclasses.dataclass(frozen=True)
class RepackG2(native_transforms.DataTransformFn):
    policy: dict
    cameras: tuple[str, ...]

    def __call__(self, data):
        state_key = "state.joint" if self.policy["state"]["space"] == "joint" else "state.pose"
        result = {
            "observation/state": data[state_key],
            "actions": data.get("g2.actions", data.get(label_column(self.policy))),
        }
        for source, dest in (
            ("state.pose", "pose"),
            ("state.joint", "joint"),
            ("g2.pose_history", "pose_history"),
            ("g2.joint_history", "joint_history"),
            ("g2.start_pose", "start_pose"),
            ("g2.start_joint", "start_joint"),
            ("g2.command_pose", "command_pose"),
            ("g2.previous_command_pose", "previous_command_pose"),
            ("reference.object_pose", "object_pose"),
            ("reference.object_joint", "object_joint"),
            ("reference.base_to_parent", "base_to_parent"),
        ):
            if source in data:
                result[f"observation/{dest}"] = np.asarray(data[source])
        mapping = {
            "head_color": "image",
            "hand_left_color": "wrist_image_left",
            "hand_right_color": "wrist_image_right",
        }
        for camera in self.cameras:
            result[f"observation/{mapping[camera]}"] = data[f"observation.images.{camera}"]
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        elif "task" in data:
            result["prompt"] = data["task"]
        if result["actions"] is None:
            result.pop("actions")
        return result


class G2Dataset:
    """Sample in seconds within episodes; clamp boundary targets, repeat initial history.

    Low-dimensional signals interpolate linearly, poses interpolate on SO(3).
    Images use the nearest recorded frame. No sample crosses an episode boundary.
    """

    def __init__(self, dataset, policy, cameras, horizon):
        self.dataset, self.policy, self.cameras, self.horizon = dataset, policy, cameras, horizon
        self.fps = float(dataset.meta.fps)
        self.hz = float(policy["timing"]["sample_hz"])
        self.anchors = []
        self.bounds = []
        for first, last in zip(dataset.episode_data_index["from"], dataset.episode_data_index["to"], strict=True):
            start, stop = int(first), int(last)
            count = int(np.floor((stop - start - 1) * self.hz / self.fps + 1e-7)) + 1
            for i in range(count):
                self.anchors.append(start + i * self.fps / self.hz)
                self.bounds.append((start, stop))

    def __len__(self):
        return len(self.anchors)

    def _row(self, index):
        return self.dataset.hf_dataset[int(index)]

    def _signal(self, key, positions, bounds):
        start, stop = bounds
        samples = []
        for position in positions:
            x = float(np.clip(position, start, stop - 1))
            low, high = int(np.floor(x)), int(np.ceil(x))
            a = np.asarray(self._row(low)[key], dtype=np.float32)
            b = np.asarray(self._row(high)[key], dtype=np.float32)
            f = x - low
            value = a * (1 - f) + b * f
            if key in ("state.pose", "action.pose", "reference.object_pose") and high != low:
                from scipy.spatial.transform import Rotation
                from scipy.spatial.transform import Slerp

                for offset in (0, 10):
                    rotations = g2.rotation_matrix(np.stack([a[offset + 3 : offset + 9], b[offset + 3 : offset + 9]]))
                    value[offset + 3 : offset + 9] = g2.rotation_vector(
                        Slerp([0, 1], Rotation.from_matrix(rotations))([f]).as_matrix()[0]
                    )
            if key == "reference.base_to_parent" and high != low:
                from scipy.spatial.transform import Rotation
                from scipy.spatial.transform import Slerp

                value[:3, :3] = Slerp([0, 1], Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]])))([f]).as_matrix()[
                    0
                ]
            samples.append(value)
        return np.asarray(samples)

    def __getitem__(self, index):
        anchor, bounds = self.anchors[index], self.bounds[index]
        timing, action = self.policy["timing"], self.policy["action"]
        current = dict(self.dataset[round(anchor)])
        fields = self.dataset.meta.features
        for key in (
            "state.pose",
            "state.joint",
            "reference.object_pose",
            "reference.object_joint",
            "reference.base_to_parent",
        ):
            if key in fields:
                current[key] = self._signal(key, [anchor], bounds)[0]
        for key, dest in (("state.pose", "pose"), ("state.joint", "joint")):
            if key in fields:
                current[f"g2.start_{dest}"] = self._signal(key, [bounds[0]], bounds)[0]
                history = anchor + np.arange(1 - timing["history_steps"], 1) * self.fps / self.hz
                current[f"g2.{dest}_history"] = self._signal(key, history, bounds)
        if "action.pose" in fields:
            current["g2.command_pose"] = self._signal("action.pose", [anchor], bounds)[0]
            current["g2.previous_command_pose"] = self._signal("action.pose", [anchor - self.fps / self.hz], bounds)[0]
        if timing["history_steps"] > 1:
            frames = [self.dataset[int(np.clip(round(x), bounds[0], bounds[1] - 1))] for x in history]
            for camera in self.cameras:
                key = f"observation.images.{camera}"
                current[key] = np.stack([np.asarray(frame[key]) for frame in frames])
        targets = (
            anchor
            + (timing["target_offset_steps"] + np.arange(self.horizon) * timing["stride_steps"]) * self.fps / self.hz
        )
        column = label_column(self.policy)
        if column in fields:
            labels = self._signal(column, targets, bounds)
        else:
            # Velocities can be derived from position/pose trajectories; torque must be recorded.
            suffix = "pose" if action["control_quantity"] == "eef_twist" else "joint"
            source = ("action" if action["label_source"] == "command" else "state") + "." + suffix
            if action["control_quantity"] not in ("joint_velocity", "eef_twist"):
                raise ValueError(f"Dataset requires recorded {column}")
            poses = self._signal(source, targets, bounds)
            next_targets = np.minimum(targets + self.fps / self.hz, bounds[1] - 1)
            previous = np.maximum(np.minimum(targets, bounds[1] - 1) - self.fps / self.hz, bounds[0])
            # Forward differences, backward at the endpoint; a one-frame episode has zero velocity.
            begins = np.where(
                next_targets > np.minimum(targets, bounds[1] - 1), np.minimum(targets, bounds[1] - 1), previous
            )
            ends = np.where(
                next_targets > np.minimum(targets, bounds[1] - 1), next_targets, np.minimum(targets, bounds[1] - 1)
            )
            a, b = self._signal(source, begins, bounds), self._signal(source, ends, bounds)
            dt = np.maximum((ends - begins) / self.fps, 1e-8)
            if suffix == "joint":
                labels = (b - a) / dt[:, None]
                labels[:, [7, 15]] = poses[:, [7, 15]]
            else:
                from scipy.spatial.transform import Rotation

                parts = []
                for offset in (0, 10):
                    angular = (
                        Rotation.from_matrix(
                            g2.rotation_matrix(b[:, offset + 3 : offset + 9])
                            @ np.swapaxes(g2.rotation_matrix(a[:, offset + 3 : offset + 9]), -1, -2)
                        ).as_rotvec()
                        / dt[:, None]
                    )
                    parts.append(
                        np.concatenate(
                            [
                                (b[:, offset : offset + 3] - a[:, offset : offset + 3]) / dt[:, None],
                                angular,
                                poses[:, offset + 9 : offset + 10],
                            ],
                            axis=-1,
                        )
                    )
                labels = np.concatenate(parts, axis=-1)
        current["g2.actions"] = labels.astype(np.float32)
        return current


@dataclasses.dataclass(frozen=True)
class DatasetFactory:
    policy: dict
    cameras: tuple[str, ...]

    def __call__(self, data_config, horizon, model_config):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(data_config.repo_id)
        return G2Dataset(dataset, self.policy, self.cameras, horizon)


@dataclasses.dataclass(frozen=True)
class LeRobotG2DataConfig(DataConfigFactory):
    use_delta_actions: bool = True
    state_space: str | None = None
    arm: str = "both"
    cameras: tuple[str, ...] | None = None
    policy: dict | None = None

    @property
    def resolved_state_space(self):
        return (
            self.policy["state"]["space"]
            if self.policy
            else self.state_space or ("eef" if self.use_delta_actions else "joint")
        )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        policy = policy_for_data(self)
        selected = g2.selected_camera_ids(self.arm, self.cameras)
        dim, _ = g2.action_dimensions(policy, self.arm)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            dataset_factory=DatasetFactory(policy, selected),
            repack_transforms=native_transforms.Group(inputs=[RepackG2(policy, selected)]),
            data_transforms=native_transforms.Group(
                inputs=[g2.G2Inputs(model_config.model_type, policy=policy, arm=self.arm, cameras=selected)],
                outputs=[g2.G2Outputs(action_dim=dim)],
            ),
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=(label_column(policy),),
        )
