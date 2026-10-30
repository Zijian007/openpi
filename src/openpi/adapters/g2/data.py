"""G2 dataset construction, episode-safe temporal sampling, and native config integration."""

import copy
import dataclasses
import pathlib

from g2_policy_protocol.data.dataset import TemporalDataset
from g2_policy_protocol.data.dataset import action_label_column
from g2_policy_protocol.data.dataset import references_from_sample
from g2_policy_protocol.data.encoding import action_dimensions
from typing_extensions import override

from openpi import transforms as native_transforms
from openpi.adapters.g2 import transforms as g2
from openpi.adapters.g2.config import POLICY_DEFAULTS
from openpi.adapters.g2.config import cameras_for_arm
from openpi.adapters.g2.contract import CAMERA_OBSERVATION_KEYS
from openpi.models import model as _model
from openpi.training.config import DataConfig
from openpi.training.config import DataConfigFactory
from openpi.training.config import ModelTransformFactory

__all__ = ["DatasetFactory", "TemporalDataset", "LeRobotG2DataConfig", "RepackG2", "action_label_column"]


@dataclasses.dataclass(frozen=True)
class RepackG2(native_transforms.DataTransformFn):
    policy: dict
    cameras: tuple[str, ...]

    def __call__(self, data):
        pose, joint = self.policy["state"]["pose"]["enabled"], self.policy["state"]["joint"]["enabled"]
        state_key = "state.joint" if joint and not pose else "state.pose"
        result = {
            "observation/state": data[state_key],
            "actions": data.get("g2.actions", data.get(action_label_column(self.policy))),
        }
        result.update(references_from_sample(data))
        for camera in self.cameras:
            result[CAMERA_OBSERVATION_KEYS[camera]] = data[f"observation.images.{camera}"]
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
        return TemporalDataset(dataset, self.policy, self.cameras, horizon)


@dataclasses.dataclass(frozen=True)
class LeRobotG2DataConfig(DataConfigFactory):
    arm: str = "both"
    cameras: tuple[str, ...] | None = None
    policy: dict = dataclasses.field(default_factory=lambda: copy.deepcopy(POLICY_DEFAULTS))

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        policy = copy.deepcopy(self.policy)
        selected = cameras_for_arm(self.arm, self.cameras)
        dim, _ = action_dimensions(policy, self.arm)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            dataset_factory=DatasetFactory(policy, selected),
            repack_transforms=native_transforms.Group(inputs=[RepackG2(policy, selected)]),
            data_transforms=native_transforms.Group(
                inputs=[g2.G2Inputs(policy=policy, arm=self.arm, cameras=selected)],
                outputs=[g2.G2Outputs(action_dim=dim)],
            ),
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=(action_label_column(policy),),
        )
