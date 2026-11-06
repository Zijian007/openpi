"""G2 W&B session: colocate logs with the experiment, resume softly, and backfill gaps.

Native ``scripts/train.py`` is left unchanged. The G2 train pipeline installs these
wrappers around ``wandb`` and checkpoint saves before calling OpenPI training.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
from pathlib import Path
from typing import Any

import wandb

from openpi.training import checkpoints as _checkpoints

_LOGGER = logging.getLogger(__name__)

_checkpoint_dir: Path | None = None
_project: str | None = None
_name: str | None = None
_installed = False
_original_init = wandb.init
_original_log = wandb.log
_original_save_state = _checkpoints.save_state


def experiment_wandb_dir(checkpoint_dir: Path) -> Path:
    """Local W&B files live next to ``checkpoints/`` under the experiment directory."""
    root = checkpoint_dir.parent if checkpoint_dir.name == "checkpoints" else checkpoint_dir
    root.mkdir(parents=True, exist_ok=True)
    return root


def _history_roots(checkpoint_dir: Path) -> list[Path]:
    roots: list[Path] = []
    repo_wandb = Path(__file__).resolve().parents[4] / "wandb"
    for raw in (
        experiment_wandb_dir(checkpoint_dir) / "wandb",
        checkpoint_dir / "wandb",
        Path.cwd() / "wandb",
        repo_wandb,
    ):
        if raw.is_dir() and raw not in roots:
            roots.append(raw)
    return roots


def _scalar_history_from_file(path: Path) -> dict[int, dict[str, float]]:
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal.datastore import DataStore

    store = DataStore()
    store.open_for_scan(os.fspath(path))
    history: dict[int, dict[str, float]] = {}
    while True:
        data = store.scan_data()
        if data is None:
            break
        record = pb.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "history":
            continue
        parsed: dict[str, Any] = {}
        for item in record.history.item:
            key = item.key or (".".join(item.nested_key) if item.nested_key else "")
            if not key or not item.value_json:
                continue
            try:
                parsed[key] = json.loads(item.value_json)
            except Exception:
                continue
        step = parsed.get("_step")
        if not isinstance(step, (int, float)):
            continue
        metrics = {
            key: float(value)
            for key, value in parsed.items()
            if key not in {"_step", "_runtime", "_timestamp"}
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }
        if metrics:
            history[int(step)] = metrics
    return history


def local_scalar_history(run_id: str, checkpoint_dir: Path) -> dict[int, dict[str, float]]:
    merged: dict[int, dict[str, float]] = {}
    files: list[Path] = []
    for root in _history_roots(checkpoint_dir):
        files.extend(sorted(root.glob(f"run-*-{run_id}/run-{run_id}.wandb")))
    files.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0)
    for path in files:
        try:
            merged.update(_scalar_history_from_file(path))
        except Exception as e:
            _LOGGER.warning("Failed to read local W&B history %s (%s)", path, e)
    return merged


def _wandb_kwargs() -> dict[str, str]:
    kwargs: dict[str, str] = {}
    if _checkpoint_dir is not None:
        kwargs["dir"] = str(experiment_wandb_dir(_checkpoint_dir))
    if _project:
        kwargs["project"] = _project
    if _name:
        kwargs["name"] = _name
    return kwargs


def _write_run_id() -> None:
    if _checkpoint_dir is None or wandb.run is None or not wandb.run.id:
        return
    (_checkpoint_dir / "wandb_id.txt").write_text(wandb.run.id)


def backfill_history(checkpoint_dir: Path | None = None) -> int:
    """Upload local scalar points that are ahead of the resumed cloud step."""
    run = wandb.run
    directory = checkpoint_dir or _checkpoint_dir
    if run is None or not run.id or directory is None:
        return 0
    current = int(getattr(run, "step", 0) or 0)
    pending = [(step, metrics) for step, metrics in sorted(local_scalar_history(run.id, directory).items()) if step > current]
    if not pending:
        return 0
    _LOGGER.info(
        "W&B backfill: uploading %d local points (%d -> %d); cloud step was %d",
        len(pending),
        pending[0][0],
        pending[-1][0],
        current,
    )
    for step, metrics in pending:
        _original_log(metrics, step=step)
    return len(pending)


def close() -> None:
    if wandb.run is None:
        return
    try:
        wandb.finish()
    except Exception as e:
        _LOGGER.warning("wandb.finish failed (%s)", e)


def reconnect(*, backfill: bool = True) -> None:
    if _checkpoint_dir is None:
        return
    wandb_id_path = _checkpoint_dir / "wandb_id.txt"
    run_id = wandb.run.id if wandb.run is not None else (wandb_id_path.read_text().strip() if wandb_id_path.exists() else "")
    close()
    if not run_id:
        return
    try:
        _original_init(id=run_id, resume="allow", **_wandb_kwargs())
        _write_run_id()
        if backfill:
            backfill_history(_checkpoint_dir)
    except Exception as e:
        _LOGGER.warning("W&B reconnect failed for %s (%s)", run_id, e)


def _patched_init(*args, **kwargs):
    if kwargs.get("mode") == "disabled" or _checkpoint_dir is None:
        return _original_init(*args, **kwargs)
    kwargs = {**kwargs, **_wandb_kwargs()}
    if kwargs.get("resume") == "must":
        kwargs["resume"] = "allow"
    try:
        run = _original_init(*args, **kwargs)
    except Exception as e:
        _LOGGER.warning("Failed to resume wandb run %s (%s); starting a new run", kwargs.get("id"), e)
        kwargs.pop("id", None)
        kwargs.pop("resume", None)
        run = _original_init(*args, **kwargs)
    _write_run_id()
    _LOGGER.info("W&B local dir: %s/wandb  (id=%s)", kwargs.get("dir"), getattr(wandb.run, "id", None))
    backfill_history(_checkpoint_dir)
    return run


def _patched_log(data: Any = None, step: int | None = None, **kwargs):
    run = wandb.run
    if run is not None and step == 0:
        current = int(getattr(run, "step", 0) or 0)
        if current > 0:
            step = current
    try:
        return _original_log(data, step=step, **kwargs)
    except Exception as e:
        _LOGGER.warning("wandb.log failed at step %s (%s); reconnecting", step, e)
        reconnect()
        return _original_log(data, step=step, **kwargs)


def _patched_save_state(*args, **kwargs):
    result = _original_save_state(*args, **kwargs)
    if _checkpoint_dir is not None:
        reconnect()
    return result


def _finalize() -> None:
    if _checkpoint_dir is None:
        return
    reconnect()
    close()


def install(config: Any) -> None:
    """Activate G2 W&B behavior for the current training process."""
    global _checkpoint_dir, _project, _name, _installed
    _checkpoint_dir = Path(config.checkpoint_dir)
    _project = config.project_name
    _name = config.exp_name
    if _installed:
        return
    wandb.init = _patched_init
    wandb.log = _patched_log
    _checkpoints.save_state = _patched_save_state
    atexit.register(_finalize)
    _installed = True


def uninstall() -> None:
    """Restore native wandb/checkpoint functions. Tests only."""
    global _checkpoint_dir, _project, _name, _installed
    wandb.init = _original_init
    wandb.log = _original_log
    _checkpoints.save_state = _original_save_state
    _checkpoint_dir = None
    _project = None
    _name = None
    _installed = False
