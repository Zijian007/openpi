import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from openpi.adapters.g2.config import POLICY_DEFAULTS
from openpi.adapters.g2.data import G2Dataset
from openpi.adapters.g2.transforms import G2Inputs
from openpi.adapters.g2.transforms import matrix_pose
from openpi.models.model import ModelType


class RecordedDataset:
    def __init__(self, fps=4):
        self.hf_dataset = []
        for episode in range(2):
            for frame in range(4):
                t = np.eye(4)
                t[0, 3] = episode * 100 + frame
                t[:3, :3] = Rotation.from_rotvec([0, 0, frame * 0.2]).as_matrix()
                pose = np.tile(matrix_pose(t, 0.4), 2)
                self.hf_dataset.append(
                    {
                        "state.pose": pose,
                        "action.pose": pose + np.tile([10, 0, 0, 0, 0, 0, 0, 0, 0, 0], 2),
                        "state.joint": np.full(16, episode * 100 + frame, np.float32),
                        "action.joint": np.full(16, episode * 100 + frame + 10, np.float32),
                        "task": "tea",
                        "observation.images.head_color": np.full((3, 8, 8), frame, np.uint8),
                    }
                )
        self.meta = SimpleNamespace(fps=fps, features={key: {} for key in self.hf_dataset[0]})
        self.episode_data_index = {"from": [0, 4], "to": [4, 8]}

    def __getitem__(self, i):
        return copy.deepcopy(self.hf_dataset[i])


def test_resampling_offsets_stride_and_episode_boundaries():
    policy = copy.deepcopy(POLICY_DEFAULTS)
    policy["action"]["label_source"] = "measured"
    policy["timing"].update(sample_hz=2, target_offset_steps=1, stride_steps=2, history_steps=3)
    dataset = G2Dataset(RecordedDataset(), policy, ("head_color",), 3)
    assert len(dataset) == 4
    sample = dataset[0]
    np.testing.assert_allclose(sample["g2.actions"][:, 0], [2, 3, 3])
    np.testing.assert_array_equal(sample["g2.pose_history"][:, 0], [0, 0, 0])
    np.testing.assert_array_equal(sample["observation.images.head_color"][:, 0, 0, 0], [0, 0, 0])
    np.testing.assert_allclose(dataset[1]["g2.actions"][:, 0], [3, 3, 3])
    np.testing.assert_allclose(dataset[2]["g2.actions"][:, 0], [102, 103, 103])
    np.testing.assert_array_equal(dataset[2]["g2.pose_history"][:, 0], [100, 100, 100])


def test_upsampling_interpolates_rotation_on_so3():
    policy = copy.deepcopy(POLICY_DEFAULTS)
    policy["timing"]["sample_hz"] = 8
    sample = G2Dataset(RecordedDataset(), policy, ("head_color",), 2)[1]
    assert sample["state.pose"][0] == 0.5
    expected = Rotation.from_rotvec([0, 0, 0.1]).as_matrix()[:, 0]
    np.testing.assert_allclose(sample["state.pose"][3:6], expected, atol=1e-6)


@pytest.mark.parametrize(("quantity", "space", "width"), [("joint_velocity", "joint", 16), ("eef_twist", "eef", 14)])
def test_velocity_derivatives_preserve_absolute_grippers(quantity, space, width):
    policy = copy.deepcopy(POLICY_DEFAULTS)
    policy["action"].update(
        control_quantity=quantity, space=space, representation="absolute", relative_to=None, delta_scope=None
    )
    policy["timing"]["sample_hz"] = 4
    sample = G2Dataset(RecordedDataset(), policy, ("head_color",), 5)[0]["g2.actions"]
    assert sample.shape == (5, width)
    if quantity == "joint_velocity":
        np.testing.assert_allclose(sample[:, 0], 4)
        np.testing.assert_allclose(sample[:, 7], [10, 11, 12, 13, 13])
    else:
        np.testing.assert_allclose(sample[:, 0], 4)
        np.testing.assert_allclose(sample[:, 5], 0.8, atol=1e-6)
        np.testing.assert_allclose(sample[:, 6], 0.4)


def test_torque_requires_recorded_torque_not_position_differences():
    policy = copy.deepcopy(POLICY_DEFAULTS)
    policy["action"].update(
        control_quantity="joint_torque", space="joint", representation="absolute", relative_to=None, delta_scope=None
    )
    with pytest.raises(ValueError, match="recorded action.joint_torque"):
        G2Dataset(RecordedDataset(), policy, ("head_color",), 2)[0]
    raw = RecordedDataset()
    raw.meta.features["action.joint_torque"] = {}
    for row in raw.hf_dataset:
        row["action.joint_torque"] = np.arange(16, dtype=np.float32)
    np.testing.assert_array_equal(
        G2Dataset(raw, policy, ("head_color",), 2)[0]["g2.actions"], np.tile(np.arange(16), (2, 1))
    )


def test_combined_history_input_preserves_all_state_values_and_frames():
    policy = copy.deepcopy(POLICY_DEFAULTS)
    policy["state"].update(space="both", representation="relative", relative_to="episode_start_pose")
    policy["timing"].update(history_steps=2, sample_hz=4)
    from openpi.adapters.g2.data import RepackG2

    sample = G2Dataset(RecordedDataset(), policy, ("head_color",), 2)[2]
    result = G2Inputs(ModelType.PI05, policy=policy, arm="right", cameras=("head_color",))(
        RepackG2(policy, ("head_color",))(sample)
    )
    assert result["state"].shape == (36,)
    assert result["image"]["base_0_rgb"].shape == (224, 448, 3)
    assert result["image"]["base_0_rgb"][0, 0, 0] == 1
    assert result["image"]["base_0_rgb"][0, 224, 0] == 2
    assert not any(key.startswith("observation/") for key in result)


def test_native_dataset_hook_is_used_without_native_robot_imports():
    from openpi.training.config import DataConfig
    from openpi.training.data_loader import create_torch_dataset

    marker = object()
    calls = []

    def factory(config, horizon, model):
        calls.append((config.repo_id, horizon, model))
        return marker

    config = DataConfig(repo_id="test", dataset_factory=factory)
    with patch(
        "openpi.training.data_loader.lerobot_dataset.LeRobotDataset", side_effect=AssertionError("native default path")
    ):
        assert create_torch_dataset(config, 7, "model") is marker
    assert calls == [("test", 7, "model")]


@pytest.mark.parametrize("space", ["eef", "joint", "both"])
def test_relative_state_uses_episode_start_and_preserves_grippers(space):
    from openpi.adapters.g2.data import RepackG2

    policy = copy.deepcopy(POLICY_DEFAULTS)
    policy["state"].update(space=space, representation="relative", relative_to="episode_start_pose")
    policy["timing"]["sample_hz"] = 4
    sample = G2Dataset(RecordedDataset(), policy, ("head_color",), 2)[2]
    result = G2Inputs(ModelType.PI05, policy=policy, cameras=("head_color",))(RepackG2(policy, ("head_color",))(sample))
    if space in ("eef", "both"):
        np.testing.assert_allclose(result["state"][[0, 10]], [2, 2])
        np.testing.assert_allclose(result["state"][[9, 19]], [0.4, 0.4])
    if space in ("joint", "both"):
        joints = result["state"][-16:]
        np.testing.assert_array_equal(joints, np.full(16, 2))


def test_full_combined_history_reaches_native_tokenizer_without_changing_action_width(tmp_path, monkeypatch):
    from openpi import transforms
    from openpi.adapters.g2 import config as adapter_config
    from openpi.adapters.g2.data import LeRobotG2DataConfig
    from openpi.training import config as native_config

    captured = []

    class Tokenizer:
        def __init__(self, max_len):
            self.length = max_len

        def tokenize(self, prompt, state):
            captured.append(state.copy())
            return np.zeros(self.length, np.int32), np.ones(self.length, bool)

    monkeypatch.setattr(native_config._tokenizer, "PaligemmaTokenizer", Tokenizer)  # noqa: SLF001
    monkeypatch.setattr(native_config.DataConfigFactory, "_load_norm_stats", lambda *args: None)
    settings = adapter_config.load_settings()
    settings["policy"]["state"]["space"] = "both"
    settings["policy"]["timing"].update(history_steps=2, sample_hz=4)
    config = adapter_config.build_train_config(settings)
    assert config.model.action_dim == 32
    assert config.model.max_token_len > 200
    dc = LeRobotG2DataConfig(repo_id="test", policy=settings["policy"], arm="right", cameras=("head_color",)).create(
        tmp_path, config.model
    )
    sample = G2Dataset(RecordedDataset(), settings["policy"], ("head_color",), 2)[2]
    for transform in (*dc.repack_transforms.inputs, *dc.data_transforms.inputs):
        sample = transform(sample)
    assert sample["state"].shape == (36,)
    expected = sample["state"].copy()
    sample = transforms.Normalize({"state": transforms.NormStats(mean=np.zeros(36), std=np.ones(36))})(sample)
    for transform in dc.model_transforms.inputs:
        sample = transform(sample)
    np.testing.assert_allclose(captured[0], expected / (1 + 1e-6))
    assert sample["state"].shape == (36,)
    assert sample["actions"].shape == (2, 32)
    assert sample["image"]["base_0_rgb"].shape == (224, 224, 3)
