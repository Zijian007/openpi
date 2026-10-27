"""Sibling spatial/temporal groups retain partial overrides and legacy snapshot semantics."""

import copy
from unittest.mock import patch

import pytest
import yaml

from openpi.adapters.g2 import config as g2_config


def resolve_policy(tmp_path, policy):
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 4, "policy": policy}))
    return g2_config.resolve_settings(g2_config.load_settings(path), {})


@pytest.mark.parametrize("group", ["state", "action"])
def test_sibling_groups_match_both_older_encoding_layouts(tmp_path, group):
    temporal = {"anchor": "episode_start"}
    if group == "action":
        temporal["reference_scope"] = "step"
    new = resolve_policy(tmp_path, {group: {"spatial": {"mode": "relative", "method": "hybrid"}, "temporal": temporal}})
    nested = resolve_policy(
        tmp_path, {group: {"encoding": {"mode": "relative", "spatial": "hybrid", "temporal": temporal}}}
    )
    assert new == nested
    if group == "action":
        old = resolve_policy(
            tmp_path,
            {
                group: {
                    "encoding": {
                        "mode": "relative",
                        "method": "hybrid",
                        "reference": {"anchor": "episode_start", "strategy": "previous_target"},
                    }
                }
            },
        )
        assert old == new
    public = g2_config.public_settings(new)["policy"][group]
    assert "encoding" not in public
    assert public["spatial"] == {"mode": "relative", "method": "hybrid"}
    assert public["temporal"] == temporal


@pytest.mark.parametrize("group", ["state", "action"])
def test_absolute_mode_clears_inherited_encoding_and_accepts_explicit_nulls(tmp_path, group):
    relative = resolve_policy(
        tmp_path, {group: {"spatial": {"method": "se3_body"}, "temporal": {"anchor": "episode_start"}}}
    )
    absolute = g2_config.resolve_settings(relative, {f"{group}_mode": "absolute"})
    temporal = {"anchor": None}
    if group == "action":
        temporal["reference_scope"] = None
    explicit = resolve_policy(
        tmp_path, {group: {"spatial": {"mode": "absolute", "method": None}, "temporal": temporal}}
    )
    assert absolute == explicit
    public = g2_config.public_settings(absolute)
    assert public["policy"][group]["temporal"] == temporal
    partial = copy.deepcopy(public)
    partial["policy"][group] = {"temporal": temporal}
    path = tmp_path / "nulls.yaml"
    path.write_text(yaml.safe_dump(partial))
    # A reference-only null override inherits the default absolute state;
    # relative action defaults correctly require a valid anchor and scope.
    if group == "state":
        assert g2_config.resolve_settings(g2_config.load_settings(path), {}) == absolute
    else:
        with pytest.raises(ValueError, match="Relative actions require"):
            g2_config.resolve_settings(g2_config.load_settings(path), {})


@pytest.mark.parametrize("group", ["state", "action"])
@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ({"spatial": None}, "spatial must be a YAML mapping"),
        ({"spatial": "hybrid"}, "spatial must be a YAML mapping"),
        ({"spatial": {"mode": "delta"}}, "spatial.mode must be absolute or relative"),
        ({"spatial": {"mode": "relative"}}, "spatial.method is required"),
        ({"spatial": {"method": "typo"}}, "spatial.method must be one of"),
        ({"spatial": {"methd": "hybrid"}}, "Unknown YAML field"),
        ({"spatial": {"mode": "absolute", "method": "hybrid"}}, "must be null"),
        ({"spatial": {"mode": "absolute"}, "temporal": {"anchor": "episode_start"}}, "must be null"),
        ({"temporal": "episode_start"}, "temporal must be a YAML mapping"),
        ({"temporal": {"anchor": "typo"}}, "temporal.anchor must be one of"),
        ({"spatial": {}, "encoding": {}}, "Cannot mix"),
        ({"temporal": {}, "relative_to": "episode_start_pose"}, "Cannot mix"),
    ],
)
def test_current_yaml_errors_use_public_paths(tmp_path, group, spec, match):
    with pytest.raises(ValueError, match=match):
        resolve_policy(tmp_path, {group: spec})


def test_previous_nested_state_and_action_snapshot_restores_without_template(tmp_path):
    settings = resolve_policy(
        tmp_path,
        {
            "state": {"spatial": {"method": "hybrid"}, "temporal": {"anchor": "episode_start"}},
            "action": {"spatial": {"method": "se3_body"}, "temporal": {"anchor": "object", "reference_scope": "step"}},
        },
    )
    old = g2_config.public_settings(settings)
    for group in ("state", "action"):
        spec = old["policy"][group]
        spatial, temporal = spec.pop("spatial"), spec.pop("temporal")
        spec["encoding"] = {"mode": spatial["mode"], "spatial": spatial["method"], "temporal": temporal}
    with patch.object(g2_config, "load_settings", side_effect=AssertionError("must not read current YAML")):
        restored = g2_config.build_train_config(old)
    assert restored.data.policy == settings["policy"]
    saved = yaml.safe_load(restored.checkpoint_assets[g2_config.CHECKPOINT_CONFIG])["settings"]
    assert saved == g2_config.public_settings(settings)
