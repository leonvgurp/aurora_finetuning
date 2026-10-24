"""Explicit transfer and retained-data limits; never prune user artifacts."""

from aurora_finetuning.config import Config
from aurora_finetuning.telemetry import Recorder


def retained_size(config: Config) -> int:
    """Count project data/model/output files, excluding environments and symlinks."""
    return sum(
        path.stat().st_size
        for name in ("data", "checkpoints", "outputs")
        for path in (config.root / name).rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def check_budget(config: Config, recorder: Recorder, estimated_bytes: int) -> None:
    """Reject a transfer before execution when its estimate exceeds limits."""
    if estimated_bytes < 0:
        raise ValueError("Negative acquisition estimate")
    if recorder.network_bytes + estimated_bytes > config.max_download_gb * 1024**3:
        raise ValueError("Acquisition estimate exceeds max_download_gb")
    if retained_size(config) + estimated_bytes > config.max_storage_gb * 1024**3:
        raise ValueError(
            "Acquisition estimate exceeds max_storage_gb; no files removed"
        )
