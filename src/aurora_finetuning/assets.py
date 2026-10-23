"""Pinned model assets with streaming checksum verification."""

import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from aurora_finetuning.acquisition import sha256
from aurora_finetuning.config import Config
from aurora_finetuning.telemetry import Recorder, write_json

CHECKPOINT = "aurora-0.25-v1.5.ckpt"
STATIC = "aurora-0.25-v1.5-static.pickle"
REPOSITORY = "microsoft/aurora"


def download_assets(config: Config, recorder: Recorder) -> Path:
    """Resolve a revision once, download assets and pin verified hashes."""
    root = config.root / "checkpoints/base"
    manifest = root / "assets.json"
    if manifest.exists():
        verify_assets(config)
        return manifest
    with recorder.measure("download.assets.metadata"):
        revision = config.checkpoint_revision or HfApi().model_info(REPOSITORY).sha
        if not revision:
            raise ValueError("Could not resolve model repository revision")
    entries: dict[str, str] = {}
    for name in (CHECKPOINT, STATIC):
        with recorder.measure(f"download.assets.{name}"):
            path = Path(
                hf_hub_download(REPOSITORY, name, revision=revision, local_dir=root)
            )
            entries[name] = sha256(path)
    write_json(
        manifest, {"repository": REPOSITORY, "revision": revision, "sha256": entries}
    )
    return manifest


def verify_assets(config: Config) -> dict[str, object]:
    """Verify local assets before loading pickle/checkpoint content."""
    root = config.root / "checkpoints/base"
    manifest = root / "assets.json"
    if not manifest.exists():
        raise ValueError("Run download-model first")
    info = json.loads(manifest.read_text())
    if config.checkpoint_revision and info["revision"] != config.checkpoint_revision:
        raise ValueError(
            "Configured checkpoint revision differs from downloaded assets"
        )
    for name in (CHECKPOINT, STATIC):
        if not (root / name).exists() or sha256(root / name) != info["sha256"][name]:
            raise ValueError(f"Missing or modified asset: {name}")
    return {"repository": str(info["repository"]), "revision": str(info["revision"])}
