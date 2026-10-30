"""G2 dataset construction, episode-safe temporal sampling, and native config integration."""

import copy
import dataclasses
import logging
import pathlib
from pathlib import Path

from g2_policy_protocol.data.dataset import TemporalDataset
from g2_policy_protocol.data.dataset import action_label_column
from g2_policy_protocol.data.dataset import references_from_sample
from g2_policy_protocol.data.dataset import validate_dataset_contract
from g2_policy_protocol.data.encoding import action_dimensions
from g2_policy_protocol.data.sources import MixtureIndex
import numpy as np
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

__all__ = ["DatasetFactory", "LeRobotG2DataConfig", "RepackG2", "TemporalDataset", "action_label_column"]


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
    sources: list[dict] | None = None
    root: str | None = None
    mixture: dict | None = None
    seed: int = 0

    def __call__(self, data_config, horizon, model_config):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        if self.sources is None:
            dataset = (LeRobotDataset(data_config.repo_id) if self.root is None
                       else LeRobotDataset(data_config.repo_id, root=Path(self.root)))
            return G2Dataset([TemporalDataset(dataset, self.policy, self.cameras, horizon)], self.cameras)
        parts, lengths = [], []
        for source in self.sources:
            try:
                dataset = LeRobotDataset(source["repo_id"], root=Path(source["path"]))
                validate_dataset_contract(dataset.meta.features, dataset.meta.info, self.policy,
                                          tuple(f"observation.images.{camera}" for camera in self.cameras))
                parts.append(TemporalDataset(dataset, self.policy, self.cameras, horizon))
                lengths.append(temporal_lengths(dataset.episode_data_index, dataset.meta.fps,
                                                self.policy["timing"]["sample_hz"]))
            except (ValueError, KeyError) as error:
                raise ValueError(f"Source {source['name']}: {error}") from error
        if self.mixture is not None and lengths != self.mixture["lengths"]:
            raise ValueError("Dataset episode lengths changed after resolving the mixture")
        index = MixtureIndex(lengths, [source["weight"] for source in self.sources], self.seed,
                             plan=None if self.mixture is None else self.mixture["plan"])
        logging.info("G2 mixture: %s", ", ".join(
            f"{source['name']}={ratio:.2%}" for source, ratio in zip(self.sources, index.proportions, strict=True)))
        return G2Dataset(parts, self.cameras, index)


def temporal_lengths(bounds, fps, sample_hz):
    return [int(np.floor((int(stop) - int(start) - 1) * sample_hz / fps + 1e-7)) + 1
            for start, stop in zip(bounds["from"], bounds["to"], strict=True)]


class G2Dataset:
    """Keep all temporal windows inside their source, including virtual copies."""

    def __init__(self, parts, cameras, index=None):
        self.parts, self.cameras, self.index = parts, cameras, index

    def __len__(self):
        return len(self.parts[0]) if self.index is None else len(self.index)

    def sample(self, index, *, images=True):
        source, local, _ = (0, index, 0) if self.index is None else self.index.locate(index)
        return self.parts[source].sample(local, images=images)

    def __getitem__(self, index):
        return self.sample(index)

    def normalization_dataset(self):
        return G2NormalizationDataset(self)


class G2NormalizationDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.sample(index, images=False)
        # G2Inputs expects camera keys. Tiny placeholders are discarded before collate.
        for camera in self.dataset.cameras:
            sample[f"observation.images.{camera}"] = np.zeros((3, 1, 1), np.uint8)
        return sample


@dataclasses.dataclass(frozen=True)
class LeRobotG2DataConfig(DataConfigFactory):
    arm: str = "both"
    cameras: tuple[str, ...] | None = None
    policy: dict = dataclasses.field(default_factory=lambda: copy.deepcopy(POLICY_DEFAULTS))
    sources: list[dict] | None = None
    data_root: str | None = None
    mixture: dict | None = None
    seed: int = 0

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        policy = copy.deepcopy(self.policy)
        selected = cameras_for_arm(self.arm, self.cameras)
        dim, _ = action_dimensions(policy, self.arm)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            dataset_factory=DatasetFactory(policy, selected, self.sources, self.data_root, self.mixture, self.seed),
            repack_transforms=native_transforms.Group(inputs=[RepackG2(policy, selected)]),
            data_transforms=native_transforms.Group(
                inputs=[g2.G2Inputs(policy=policy, arm=self.arm, cameras=selected)],
                outputs=[g2.G2Outputs(action_dim=dim)],
            ),
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=(action_label_column(policy),),
        )
