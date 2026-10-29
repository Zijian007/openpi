"""Checkpoint restoration and robot output adaptation around the native Policy."""

import copy
import dataclasses
from pathlib import Path

from g2_policy_tools.geometry import ActionCodec
from openpi_client import base_policy

from openpi.adapters.g2 import config as g2_config
from openpi.adapters.g2.data import LeRobotG2DataConfig
from openpi.policies import policy_config as native_policy_config
from openpi.policies.policy import PolicyRecorder
from openpi.shared import download


class G2Policy(base_policy.BasePolicy):
    """Adapt decoded model actions to the full G2 wire layout, outside native inference."""

    def __init__(self, policy, settings):
        self._policy = policy
        self._settings = settings
        self._codec = ActionCodec(settings, self.metadata.get("arm", "both"))

    @property
    def metadata(self) -> dict:
        return self._policy.metadata

    def reset(self) -> None:
        self._policy.reset()

    def infer(self, obs: dict, *, noise=None) -> dict:
        context = self.prepare_observation(obs)
        result = self._policy.infer(context) if noise is None else self._policy.infer(context, noise=noise)
        result = dict(result)
        result["actions"] = self._codec.decode(result["actions"], context)
        return result

    def new_session(self):
        return G2Policy(self._policy, self._settings)

    def prepare_observation(self, obs):
        context = dict(obs)
        if self._settings["state"]["pose"]["enabled"]:
            context.setdefault("observation/pose", context["observation/state"])
        if self._settings["state"]["joint"]["enabled"]:
            context.setdefault("observation/joint", context.get("observation/joint", context["observation/state"]))
        return context


def create_trained_policy(checkpoint_dir, **kwargs) -> G2Policy:
    """Restore a schema v6 snapshot, then let native OpenPI load weights and transforms."""
    checkpoint_dir = Path(download.maybe_download(str(checkpoint_dir)))
    snapshot = checkpoint_dir / "assets" / g2_config.CHECKPOINT_CONFIG
    if not snapshot.is_file():
        raise ValueError("G2 serving requires a g2_train_config.yaml snapshot (schema_version: 6)")
    payload = g2_config.read_yaml(snapshot)
    train_config = g2_config.build_train_config(
        g2_config.normalize_settings(payload["settings"]), tuple(payload["extra_args"])
    )
    if not isinstance(train_config.data, LeRobotG2DataConfig):
        raise ValueError("The G2 policy loader requires a G2 data adapter")

    arm = train_config.data.arm
    settings = copy.deepcopy(train_config.data.policy)
    action_dim, wire_dim = ActionCodec(settings, arm).dimensions
    metadata = {
        **(train_config.policy_metadata or {}),
        "arm": arm,
        "action_dim": wire_dim,
        "model_action_dim": action_dim,
        "policy_contract": settings,
        "required_cameras": list(g2_config.cameras_for_arm(arm, train_config.data.cameras)),
        "requires_joint_state": g2_config.needs_joint_state(settings),
        "requires_prompt": True,
    }
    configured = dataclasses.replace(train_config, policy_metadata=metadata)
    return G2Policy(native_policy_config.create_trained_policy(configured, checkpoint_dir, **kwargs), settings)


class G2PolicyRecorder(PolicyRecorder):
    """Keep policy record paths separate for each connection."""

    def new_session(self):
        import uuid

        return G2PolicyRecorder(self._policy.new_session(), self._record_dir / uuid.uuid4().hex)
