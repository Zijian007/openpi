"""G2 dataset construction, episode-safe temporal sampling, and native config integration."""

import copy
import dataclasses
import pathlib

from g2_policy_tools.dataset import G2Dataset
from g2_policy_tools.dataset import label_column
import numpy as np
from typing_extensions import override

from openpi import transforms as native_transforms
from openpi.adapters.g2 import transforms as g2
from openpi.adapters.g2.config import POLICY_DEFAULTS
from openpi.adapters.g2.config import cameras_for_arm
from openpi.models import model as _model
from openpi.training.config import DataConfig
from openpi.training.config import DataConfigFactory
from openpi.training.config import ModelTransformFactory

__all__ = ["DatasetFactory", "G2Dataset", "LeRobotG2DataConfig", "RepackG2", "label_column", "policy_for_data"]


def policy_for_data(data):
    if data.policy is not None:
        return copy.deepcopy(data.policy)
    policy = copy.deepcopy(POLICY_DEFAULTS)
    policy["state"]["space"] = data.resolved_state_space
    if not data.use_delta_actions:
        policy["action"].update(representation="absolute", relative_to=None, delta_scope=None)
    return policy


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
        selected = cameras_for_arm(self.arm, self.cameras)
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
