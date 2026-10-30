"""Check W&B connectivity and authentication before expensive dataset preparation."""

import os
import re
import subprocess
import sys

_TIMEOUT_SECONDS = 20
_LOCAL_MODES = ("offline", "dryrun", "disabled")


def check_wandb(*, enabled: bool, mode: str | None = None) -> None:
    """Use the training interpreter and environment; never create a W&B run."""
    if not enabled:
        return
    if (mode or os.environ.get("WANDB_MODE")) in _LOCAL_MODES:
        print("[preflight] W&B offline/disabled; skipping online check.", flush=True)
        return
    print(f"[preflight] Checking W&B connectivity and login (timeout {_TIMEOUT_SECONDS}s)...", flush=True)
    try:
        result = subprocess.run(
            [sys.executable, "-m", __name__, mode or ""],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=True,
            timeout=_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        reason = f"timed out after {_TIMEOUT_SECONDS}s"
    except subprocess.CalledProcessError as error:
        lines = (error.stderr or "").strip().splitlines()
        reason = lines[-1] if lines else f"check process exited with code {error.returncode}"
        reason = re.sub(r"(https?://)[^/@\s]+@", r"\1<redacted>@", reason)
        if key := os.environ.get("WANDB_API_KEY"):
            reason = reason.replace(key, "<redacted>")
    else:
        print(result.stdout.strip(), flush=True)
        return
    raise RuntimeError(
        f"W&B preflight failed: {reason}. Normalization and training have not started. "
        "Check HTTP_PROXY/HTTPS_PROXY/ALL_PROXY and their lowercase variants in this terminal "
        "(tmux may retain an old proxy), and run wandb login if credentials are missing or invalid. "
        "To train without W&B, set wandb.mode: off in the training YAML."
    )


def probe_wandb(mode: str | None) -> None:
    import wandb

    # Read SDK settings without starting the background training service.
    settings = wandb.sdk.wandb_setup._setup(start_service=False).settings  # noqa: SLF001
    if (mode or settings.mode) in _LOCAL_MODES:
        print("[preflight] W&B offline/disabled; skipping online check.")
        return
    api = wandb.Api(timeout=10, overrides={"base_url": settings.base_url})
    viewer = api.viewer
    if not getattr(viewer, "id", None):
        raise RuntimeError("W&B did not return an authenticated user")
    print(f"[preflight] W&B connected; login verified ({viewer.username}).")


if __name__ == "__main__":
    try:
        probe_wandb(sys.argv[1] or None)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from None
