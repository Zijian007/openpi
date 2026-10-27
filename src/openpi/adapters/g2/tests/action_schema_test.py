"""Action taxonomy preserves LeRobot label selection, codec behavior and snapshots."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from openpi.adapters.g2 import config as g2_config
from openpi.adapters.g2 import settings_schema
from openpi.adapters.g2.data import G2Dataset
from openpi.adapters.g2.data import RepackG2
from openpi.adapters.g2.transforms import ActionCodec
from openpi.adapters.g2.transforms import matrix_pose


def resolve_action(tmp_path, action):
    path = tmp_path / "action.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 4, "policy": {"action": action}}))
    return g2_config.resolve_settings(g2_config.load_settings(path), {})


@pytest.mark.parametrize("source", ["command", "measured"])
@pytest.mark.parametrize("scope", ["chunk", "step"])
@pytest.mark.parametrize("method", list(settings_schema.ACTION_METHODS))
def test_new_and_legacy_yaml_produce_identical_lerobot_labels_and_decoded_poses(tmp_path, source, scope, method):
    settings = resolve_action(
        tmp_path,
        {
            "target": {"quantity": "eef_pose", "source": source},
            "spatial": {
                "mode": "relative",
                "method": method,
            },
            "temporal": {"anchor": "current_measured", "reference_scope": scope},
        },
    )
    old = g2_config.public_settings(settings)
    old["schema_version"] = 3
    old["policy"]["action"] = {
        "control_quantity": "pose_target",
        "label_source": source,
        "encoding": settings_schema.ACTION_METHODS[method],
        "reference": "current_measured",
        "delta_scope": scope,
    }
    previous = g2_config.upgrade_settings(old)
    assert previous == settings
    old_v4 = g2_config.public_settings(settings)
    old_v4["policy"]["action"].pop("spatial")
    old_v4["policy"]["action"].pop("temporal")
    strategy = "fixed" if scope == "chunk" else "previous_target"
    old_v4["policy"]["action"]["encoding"] = {
        "mode": "relative",
        "method": method,
        "reference": {"anchor": "current_measured", "strategy": strategy},
    }
    assert g2_config.upgrade_settings(old_v4) == settings

    rows = []
    for i in range(4):
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_rotvec([0.1 * i, 0.2, 0.3 * i]).as_matrix()
        transform[:3, 3] = [0.4 + 0.03 * i, 0.1, 0.2]
        measured = np.tile(matrix_pose(transform, 0.2 + 0.1 * i), 2)
        command = measured.copy()
        command[[0, 10]] += 0.02
        command[[9, 19]] += 0.1
        rows.append({"state.pose": measured, "action.pose": command})

    class RecordedDataset:
        def __init__(self):
            self.hf_dataset = rows
            self.meta = SimpleNamespace(fps=30, features={key: {} for key in rows[0]})
            self.episode_data_index = {"from": [0], "to": [4]}

        def __getitem__(self, index):
            return copy.deepcopy(self.hf_dataset[index])

    outputs = []
    for resolved in (settings, previous):
        policy = resolved["policy"]
        sample = G2Dataset(RecordedDataset(), policy, (), 3)[0]
        repacked = RepackG2(policy, ())(sample)
        codec = ActionCodec(policy, "right")
        encoded = codec.encode(repacked["actions"], repacked)
        decoded = codec.decode(encoded, repacked)
        column = "action.pose" if source == "command" else "state.pose"
        targets = np.stack([row[column] for row in rows[:3]])
        np.testing.assert_array_equal(encoded[:, -1], targets[:, 19])
        np.testing.assert_allclose(decoded[:, 10:], targets[:, 10:], atol=1e-6)
        if method == "component_difference":
            expected = targets[:, 10].copy()
            if scope == "chunk":
                expected -= rows[0]["state.pose"][10]
            else:
                expected -= np.r_[rows[0]["state.pose"][10], targets[:-1, 10]]
            np.testing.assert_allclose(encoded[:, 0], expected, atol=1e-6)
        outputs.append(encoded)
    np.testing.assert_array_equal(*outputs)


@pytest.mark.parametrize("source", ["command", "measured"])
@pytest.mark.parametrize("quantity", list(settings_schema.ACTION_QUANTITIES))
def test_absolute_quantities_select_the_correct_lerobot_column(tmp_path, source, quantity):
    settings = resolve_action(
        tmp_path, {"target": {"quantity": quantity, "source": source}, "spatial": {"mode": "absolute"}}
    )
    preview = g2_config.describe_policy(settings)
    suffix = {"eef_pose": "pose", "joint_position": "joint"}.get(quantity, quantity)
    assert preview["label_column"] == f"{'action' if source == 'command' else 'state'}.{suffix}"
    assert preview["action"]["spatial"] == {"mode": "absolute", "method": None}
    assert preview["action"]["temporal"] == {"anchor": None, "reference_scope": None}
    assert preview["gripper_encoding"] == "absolute_close_ratio"
    assert g2_config.upgrade_settings(g2_config.public_settings(settings)) == settings


def test_partial_target_spatial_and_temporal_overrides_are_independent(tmp_path):
    settings = resolve_action(
        tmp_path,
        {
            "target": {"source": "measured"},
            "spatial": {"method": "hybrid"},
            "temporal": {"reference_scope": "step"},
        },
    )
    assert g2_config.describe_policy(settings)["label_column"] == "state.pose"
    action = g2_config.public_settings(settings)["policy"]["action"]
    assert action["target"] == {"quantity": "eef_pose", "source": "measured"}
    assert action["spatial"] == {"mode": "relative", "method": "hybrid"}
    assert action["temporal"] == {"anchor": "current_measured", "reference_scope": "step"}


@pytest.mark.parametrize(
    ("new", "old"),
    [
        (
            {"spatial": "hybrid", "temporal": {"reference_scope": "step"}},
            {"method": "hybrid", "reference": {"strategy": "previous_target"}},
        ),
        ({"temporal": {"anchor": "episode_start"}}, {"reference": {"anchor": "episode_start"}}),
        (
            {"mode": "absolute", "spatial": None, "temporal": None},
            {"mode": "absolute", "method": None, "reference": None},
        ),
    ],
)
def test_legacy_partial_overrides_keep_the_same_effective_settings(tmp_path, new, old):
    assert resolve_action(tmp_path, {"encoding": new}) == resolve_action(tmp_path, {"encoding": old})


@pytest.mark.parametrize("anchor", list(settings_schema.REFERENCES))
def test_reference_anchor_compiles_without_changing_its_meaning(tmp_path, anchor):
    settings = resolve_action(tmp_path, {"encoding": {"temporal": {"anchor": anchor}}})
    assert settings["policy"]["action"]["relative_to"] == settings_schema.REFERENCES[anchor]


@pytest.mark.parametrize(
    ("action", "match"),
    [
        ({"target": "eef_pose"}, "target must be a YAML mapping"),
        ({"target": {"quantity": "pose_target"}}, "target.quantity must be one of"),
        ({"target": {"source": "action"}}, "target.source must be one of"),
        ({"target": {"quatity": "eef_pose"}}, "Unknown YAML field"),
        ({"target": {}, "label_source": "command"}, "Cannot mix"),
        ({"target": {}, "encoding": "absolute"}, "encoding must be a YAML mapping"),
        ({"encoding": {"mode": "delta"}}, "mode must be absolute or relative"),
        ({"encoding": {"mode": "relative"}}, "spatial is required"),
        ({"encoding": {"method": "component_delta"}}, "method must be one of"),
        ({"encoding": {"mode": "absolute", "method": "hybrid"}}, "must be null"),
        ({"encoding": {"mode": "absolute", "reference": {}}}, "must be null"),
        ({"encoding": {"reference": "current_measured"}}, "reference must be a YAML mapping"),
        ({"encoding": {"reference": {"strategy": "chunk"}}}, "strategy must be one of"),
        ({"encoding": {"reference": {"anchor": "typo"}}}, "anchor must be one of"),
        ({"encoding": {"reference": {"scope": "fixed"}}}, "Unknown YAML field"),
        ({"encoding": {"reference": None}}, "Relative actions require"),
        ({"target": {"quantity": "joint_velocity"}}, "spatial.mode: absolute"),
        ({"target": {"quantity": "joint_position"}, "encoding": {"method": "se3_body"}}, "component_difference"),
        (
            {"target": {"quantity": "joint_position"}, "encoding": {"reference": {"anchor": "object"}}},
            "temporal.anchor: current_measured",
        ),
        ({"control_quantity": "pose_target"}, "schema v4 action uses"),
        ({"encoding": {"spatial": "typo"}}, "spatial must be one of"),
        ({"encoding": {"spatial": []}}, "spatial must be one of"),
        ({"encoding": {"mode": "absolute", "spatial": "hybrid"}}, "must be null"),
        ({"encoding": {"mode": "absolute", "temporal": {}}}, "must be null"),
        ({"encoding": {"temporal": "current_measured"}}, "temporal must be a YAML mapping"),
        ({"encoding": {"temporal": {"anchor": "typo"}}}, "temporal.anchor must be one of"),
        ({"encoding": {"temporal": {"reference_scope": "fixed"}}}, "reference_scope must be one of"),
        ({"encoding": {"temporal": {"strategy": "fixed"}}}, "Unknown YAML field"),
        ({"encoding": {"spatial": "hybrid", "reference": {}}}, "Cannot mix"),
        ({"encoding": {"method": "hybrid", "temporal": {}}}, "Cannot mix"),
        ({"encoding": {"temporal": None}}, "Relative actions require"),
    ],
)
def test_invalid_and_mixed_action_taxonomies_are_rejected(tmp_path, action, match):
    with pytest.raises(ValueError, match=match):
        resolve_action(tmp_path, action)


def test_v4_snapshot_restores_without_reading_the_current_yaml(tmp_path):
    settings = resolve_action(
        tmp_path,
        {"target": {"source": "measured"}, "encoding": {"spatial": "se3_spatial", "temporal": {"anchor": "object"}}},
    )
    config = g2_config.build_train_config(settings)
    snapshot = yaml.safe_load(config.checkpoint_assets[g2_config.CHECKPOINT_CONFIG])["settings"]
    assert snapshot["schema_version"] == 4
    assert snapshot["policy"]["action"]["target"]["source"] == "measured"
    assert snapshot["policy"]["action"]["spatial"] == {"mode": "relative", "method": "se3_spatial"}
    assert snapshot["policy"]["action"]["temporal"] == {"anchor": "object", "reference_scope": "chunk"}
    with patch.object(g2_config, "load_settings", side_effect=AssertionError("must not read template")):
        restored = g2_config.build_train_config(snapshot)
    assert restored.model == config.model
    assert restored.data.policy == config.data.policy
    assert g2_config.policy_semantics(g2_config.upgrade_settings(snapshot)) == g2_config.policy_semantics(settings)
