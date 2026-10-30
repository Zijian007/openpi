"""Resolve a mixture from small v2.1 metadata before launching training workers."""

import json
from pathlib import Path

from g2_policy_protocol.data.dataset import validate_dataset_contract
from g2_policy_protocol.data.sources import MixtureIndex
from g2_policy_protocol.data.sources import dataset_identity
from g2_policy_protocol.data.sources import mixture_fingerprint
from g2_policy_protocol.data.sources import resolve_sources
import numpy as np


def prepare_mixture(settings, *, base_dir):
    sources = resolve_sources(settings["data"], base_dir=base_dir)
    lengths, identities = [], []
    for source in sources:
        root = Path(source["path"])
        try:
            info = json.loads((root / "meta/info.json").read_text())
            if info.get("codebase_version") != "v2.1":
                raise ValueError("OpenPI requires LeRobot v2.1")
            from openpi.adapters.g2.config import selected_cameras

            validate_dataset_contract(info["features"], info, settings["policy"],
                                      tuple(f"observation.images.{c}" for c in selected_cameras(settings)))
            episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()
                        if line.strip()]
            # LeRobot loads episodes in episode-index order, independently of JSONL order.
            episodes.sort(key=lambda row: row["episode_index"])
            hz = settings["policy"]["timing"]["sample_hz"]
            lengths.append([int(np.floor((episode["length"] - 1) * hz / info["fps"] + 1e-7)) + 1
                            for episode in episodes])
            identities.append(dataset_identity(root))
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(f"Source {source['name']}: {error}") from error
    seed = settings["training"]["seed"]
    index = MixtureIndex(lengths, [source["weight"] for source in sources], seed)
    fingerprint = mixture_fingerprint(sources, identities, settings["policy"], settings["experiment"]["arm"],
                                      settings["data"].get("cameras"), settings["data"].get("image_size"),
                                      seed, index.rows)
    settings["mixture"] = {
        "asset_id": f"mix-{fingerprint[:12]}", "fingerprint": fingerprint,
        "proportions": index.proportions, "lengths": lengths, "plan": index.plan,
    }
    return sources
