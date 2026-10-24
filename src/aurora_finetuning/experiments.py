"""Resumable campaign execution with exclusive artifacts and isolated workers."""

import csv
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from aurora_finetuning.acquisition import sha256
from aurora_finetuning.artifacts import atomic_output
from aurora_finetuning.config import Config, load_config
from aurora_finetuning.experiment_config import (
    Campaign,
    Case,
    Method,
    load_campaign,
    timestamp,
)
from aurora_finetuning.experiment_scores import score_case, validate_dataset
from aurora_finetuning.telemetry import Recorder, write_json

VERSION = 1


def identity(campaign: Campaign) -> dict[str, object]:
    """Fingerprint executable source and all declared, existing campaign inputs."""
    paths = {campaign.path}
    for case in campaign.cases:
        paths.add(case.observations)
        if case.history:
            paths.add(case.history)
    for method in campaign.methods:
        paths.update(method.paths.values())
        if method.config:
            paths.add(method.config)
    for job in campaign.jobs:
        for value in job.arguments[1:]:
            candidate = (campaign.path.parent / value).resolve()
            if candidate.is_file():
                paths.add(candidate)
    paths.update(Path(__file__).parent.glob("*.py"))
    repository = Path(__file__).resolve().parents[2]
    paths.update(repository / name for name in ("pyproject.toml", "pixi.lock"))
    return {
        "schema_version": VERSION,
        "input_sha256": {
            str(p): sha256(p) if p.is_file() else "missing" for p in sorted(paths)
        },
    }


def plan_campaign(path: Path) -> dict[str, object]:
    """Inspect prerequisites without creating outputs or contacting providers."""
    c = load_campaign(path)
    return {
        "campaign": c.id,
        "track": c.track,
        "output": str(c.output),
        "budgets": {
            "seconds": c.timeout_seconds,
            "download_bytes": c.max_download_bytes,
            "storage_bytes": c.max_storage_bytes,
        },
        "cases": [
            {
                "id": x.id,
                "split": x.split,
                "availability": x.availability,
                "observations_present": x.observations.is_file(),
            }
            for x in c.cases
        ],
        "methods": [
            {
                "id": m.id,
                "kind": m.kind,
                "missing_cases": [
                    x.id
                    for x in c.cases
                    if m.kind == "cached"
                    and (x.id not in m.paths or not m.paths[x.id].is_file())
                ],
            }
            for m in c.methods
        ],
        "jobs": [asdict(j) for j in c.jobs],
        "estimate": ("Transfer sizes and model timings require pilot measurements."),
        **identity(c),
    }


def read_dataset(path: Path) -> xr.Dataset:
    """Load and close an input, keeping file handles bounded."""
    with xr.open_dataset(path) as source:
        result = source.load()
    validate_dataset(result)
    return result


def prospective_aurora_inputs(config: Config, case: Case) -> dict[str, str]:
    """Check availability of the exact prepared IFS files selected for both states.

    The preparation manifest's available_utc is the recorded availability of the
    complete prepared file, including its supplements, rather than a claim based
    only on initialization or the source observation timestamps.
    """
    from aurora_finetuning.grid import make_grid
    from aurora_finetuning.preparation import open_era5

    recorder = Recorder()
    initialization = case.initialization.replace(tzinfo=None)
    times = [initialization - timedelta(hours=6), initialization]
    for kind in ("single", "pressure"):
        open_era5(
            config.root / "data/raw/ifs", make_grid(config), times, kind, recorder
        )
    checked = dict(recorder.inputs)
    for name, digest in recorder.inputs.items():
        manifest = Path(name).with_suffix(".json")
        document = json.loads(manifest.read_text())
        if document.get("sha256") != digest:
            raise ValueError(f"Input checksum mismatch: {name}")
        if timestamp(document.get("available_utc")) > case.issue:
            raise ValueError(
                f"Prospective Aurora input unavailable at issuance: {name}"
            )
        checked[str(manifest)] = sha256(manifest)
    return checked


def prediction(c: Campaign, case: Case, method: Method) -> xr.Dataset:
    """Construct forecasts without reading verification targets for baselines."""
    if method.kind == "cached":
        if case.id not in method.paths:
            raise FileNotFoundError(f"No {method.id} prediction for {case.id}")
        result = read_dataset(method.paths[case.id])
        init = result.attrs.get("initialization_utc")
        if init is None:
            raise ValueError("Cached forecasts need initialization_utc")
        initialized = datetime.fromisoformat(str(init))
        if initialized.tzinfo is None:
            initialized = initialized.replace(tzinfo=UTC)
        if initialized > case.issue:
            raise ValueError("Forecast initialized after issuance")
        if c.track == "prospective":
            available = result.attrs.get("available_utc")
            if available is None or timestamp(available) > case.issue:
                raise ValueError("Prospective forecast was unavailable at issuance")
        return result
    if method.kind == "aurora":
        if method.config is None:
            raise ValueError("Missing weather configuration")
        from dataclasses import replace

        if c.track == "prospective" and method.source != "ifs":
            raise ValueError("Prospective Aurora requires IFS; ERA5 is retrospective")
        config = replace(load_config(method.config), seed=c.seed, precision="float32")
        checked = (
            prospective_aurora_inputs(config, case) if c.track == "prospective" else {}
        )
        from aurora_finetuning.engine import ForecastEngine

        age = int((case.issue - case.initialization).total_seconds() / 3600)
        recorder = Recorder()
        with ForecastEngine(config, recorder) as engine:
            result = engine.predict(
                case.initialization.replace(tzinfo=None), age + c.hours, method.source
            )
        if c.track == "prospective":
            consumed = {
                name: digest
                for name, digest in recorder.inputs.items()
                if Path(name).parent == config.root / "data/raw/ifs"
            }
            expected = {
                name: digest
                for name, digest in checked.items()
                if Path(name).suffix == ".nc"
            }
            if consumed != expected:
                raise ValueError("Prospective Aurora inputs changed during prediction")
            verify_recorded_inputs(checked)
            recorder.inputs.update(checked)
        result.precipitation_mm.attrs["interval_hours"] = 1
        result.attrs["execution_json"] = json.dumps(
            {"timings": recorder.report(), "inputs": recorder.inputs}
        )
        return result
    if case.history is None:
        raise FileNotFoundError("Persistence/dry need a separate past-history dataset")
    history = read_dataset(case.history)
    if c.track == "prospective":
        if timestamp(history.attrs.get("available_utc")) > case.issue:
            raise ValueError("History was unavailable at issuance")
    cutoff = np.datetime64(case.issue.replace(tzinfo=None), "ns")
    if not history.time.size or bool((history.time > cutoff).any()):
        raise ValueError("History must contain only observations at/before issuance")
    if cutoff - history.time.values[-1] > np.timedelta64(1, "h"):
        raise ValueError("Persistence history is more than one hour old")
    latest = history.isel(time=-1, drop=True)
    times = [
        np.datetime64((case.issue + timedelta(hours=h)).replace(tzinfo=None), "ns")
        for h in range(1, c.hours + 1)
    ]
    result = latest.expand_dims(time=times).copy(deep=True)
    if method.kind == "dry":
        result = result[["precipitation_mm"]]
        result["precipitation_mm"] = xr.zeros_like(result.precipitation_mm)
        result.precipitation_mm.attrs.update(units="mm", interval_hours=1)
    result.attrs.update(
        initialization_utc=case.issue.isoformat(), source=f"observation {method.kind}"
    )
    return result


def verify_recorded_inputs(values: object) -> None:
    """Reject changed indirect inputs recorded by model/training workers."""
    if not isinstance(values, dict):
        raise ValueError("Invalid recorded input checksums")
    for name, digest in values.items():
        if not isinstance(name, str) or not isinstance(digest, str):
            raise ValueError("Invalid recorded input checksum")
        source = Path(name)
        if not source.is_file() or sha256(source) != digest:
            raise ValueError(f"Recorded input changed: {name}; use a fresh output")


def score_inputs(campaign: Campaign, case: Case, folder: Path) -> dict[str, str]:
    """Fingerprint the complete prediction set and targets for immutable scoring."""
    inputs = {
        str(case.observations): sha256(case.observations)
        if case.observations.is_file()
        else "missing"
    }
    for method in campaign.methods:
        target = folder / f"{method.id}.nc"
        receipt = target.with_suffix(".nc.receipt.json")
        if target.is_file() and receipt.is_file():
            digest = sha256(target)
            if digest != json.loads(receipt.read_text())["sha256"]:
                raise ValueError(f"Corrupt completed output: {target}")
            inputs[str(target)] = digest
    return inputs


def worker(path: Path, operation: str, key: str, destination: Path) -> None:
    """Execute one isolated unit; the parent enforces its remaining wall time."""
    c = load_campaign(path)
    if operation == "score":
        case = next(x for x in c.cases if x.id == key)
        dependencies = score_inputs(c, case, destination.parent)
        predictions = {}
        for method in c.methods:
            target = destination.parent / f"{method.id}.nc"
            if target.exists() and target.with_suffix(".nc.receipt.json").exists():
                predictions[method.id] = read_dataset(target)
        scores = score_case(c, case, predictions, read_dataset(case.observations))
        if not scores:
            raise ValueError("No matched verification intervals or variables")
        verify_recorded_inputs(dependencies)
        write_json(destination, {"rows": scores, "input_sha256": dependencies})
        return
    if operation == "predict":
        case_id, method_id = key.split("/")
        case = next(x for x in c.cases if x.id == case_id)
        method = next(x for x in c.methods if x.id == method_id)
        dataset = prediction(c, case, method)
        validate_dataset(dataset)
        with atomic_output(destination) as temporary:
            dataset.to_netcdf(temporary)
        return
    from dataclasses import replace

    from aurora_finetuning.cli import dispatch, parser

    job = next(j for j in c.jobs if j.id == key)
    args = parser().parse_args(list(job.arguments))
    config = replace(
        load_config(args.config),
        max_download_gb=int(os.environ["AURORA_EXPERIMENT_DOWNLOAD_BYTES"]) / 1024**3,
        max_storage_gb=c.max_storage_bytes / 1024**3,
    )
    recorder = Recorder()
    # Preserve CLI overrides, while keeping job output inside its own report.
    if getattr(args, "steps", None) is not None:
        config = replace(config, max_steps=args.steps)
    if getattr(args, "device", None) is not None:
        config = replace(config, device=args.device)
    result: object = None
    succeeded = False
    try:
        with recorder.measure(f"experiment.{job.id}"):
            result = dispatch(args, config, recorder)
        succeeded = True
    finally:
        write_json(
            destination,
            {
                "result": result,
                "success": succeeded,
                "timings": recorder.report(),
                "network_bytes": recorder.network_bytes,
                "input_sha256": recorder.inputs,
            },
        )


def storage_size(root: Path) -> int:
    """Measure retained regular files without following directory symlinks."""
    return sum(
        p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink()
    )


def run_campaign(path: Path, phase: str = "run") -> dict[str, object]:
    """Run bounded jobs/predictions and score immutable checksummed artifacts."""
    c = load_campaign(path)
    if phase not in {"run", "fetch"}:
        raise ValueError("Unknown campaign phase")
    folder = c.output / phase
    folder.mkdir(parents=True, exist_ok=True)
    current = identity(c)
    manifest = folder / "manifest.json"
    if manifest.exists():
        prior = json.loads(manifest.read_text())
        if prior["identity"] != current:
            raise ValueError(
                "Inputs or source changed; choose a fresh output directory"
            )
    else:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        write_json(
            manifest,
            {
                "identity": current,
                "campaign": c.id,
                "track": c.track,
                "commit": commit,
                "python": platform.python_version(),
                "hardware": platform.platform(),
                "seed": c.seed,
                "created_utc": datetime.now(UTC).isoformat(),
                "config_toml": path.read_text(),
                "campaign_path": str(c.path),
            },
        )
    started = time.monotonic()
    transferred = 0
    states: list[dict[str, object]] = []

    def execute(
        operation: str,
        key: str,
        destination: Path,
        dependencies: dict[str, str] | None = None,
    ) -> None:
        nonlocal transferred
        receipt = destination.with_suffix(destination.suffix + ".receipt.json")
        if destination.exists() and receipt.exists():
            previous = json.loads(receipt.read_text())
            if sha256(destination) != previous["sha256"]:
                raise ValueError(f"Corrupt completed output: {destination}")
            if operation == "score":
                if previous.get("input_sha256") != dependencies:
                    raise ValueError(f"Score dependency mismatch: {destination}")
                verify_recorded_inputs(dependencies)
            elif operation == "job":
                verify_recorded_inputs(
                    json.loads(destination.read_text()).get("input_sha256", {})
                )
            elif operation == "predict":
                with xr.open_dataset(destination) as source:
                    execution = json.loads(
                        str(source.attrs.get("execution_json", "{}"))
                    )
                verify_recorded_inputs(execution.get("inputs", {}))
            states.append({"unit": key, "status": "completed", "reused": True})
            return
        if destination.exists():
            states.append(
                {
                    "unit": key,
                    "status": "blocked",
                    "reason": "Incomplete artifact; use a fresh output directory",
                }
            )
            return
        remaining = c.timeout_seconds - (time.monotonic() - started)
        if remaining <= 0 or storage_size(c.output) >= c.max_storage_bytes:
            states.append(
                {"unit": key, "status": "blocked", "reason": "Resource limit reached"}
            )
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment["AURORA_EXPERIMENT_DOWNLOAD_BYTES"] = str(
            max(1, c.max_download_bytes - transferred)
        )
        before = time.monotonic()
        try:
            log = destination.with_suffix(".log")
            with log.open("ab") as handle:
                command = [
                    sys.executable,
                    "-m",
                    "aurora_finetuning.experiments",
                    str(path.resolve()),
                    operation,
                    key,
                    str(destination),
                ]
                with subprocess.Popen(
                    command,
                    cwd=c.path.parent,
                    env=environment,
                    stdout=handle,
                    stderr=handle,
                    start_new_session=True,
                ) as process:
                    try:
                        status = process.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        raise
                    if status:
                        raise subprocess.CalledProcessError(status, command)

            if (
                operation == "score"
                and json.loads(destination.read_text()).get("input_sha256")
                != dependencies
            ):
                raise ValueError(
                    f"Score inputs changed during verification: {destination}"
                )
            write_json(
                receipt,
                {
                    "sha256": sha256(destination),
                    "seconds": time.monotonic() - before,
                    "input_sha256": dependencies,
                }
                if dependencies is not None
                else {
                    "sha256": sha256(destination),
                    "seconds": time.monotonic() - before,
                },
            )
            states.append(
                {
                    "unit": key,
                    "status": "completed",
                    "reused": False,
                    "seconds": time.monotonic() - before,
                }
            )
        except subprocess.TimeoutExpired:
            if phase == "fetch":
                # A killed downloader cannot publish final byte accounting.
                transferred = c.max_download_bytes
            states.append(
                {"unit": key, "status": "failed", "reason": "Wall-time budget exceeded"}
            )
        except subprocess.CalledProcessError:
            states.append({"unit": key, "status": "failed", "reason": f"See {log}"})
        finally:
            if operation == "job" and destination.exists():
                transferred += int(
                    json.loads(destination.read_text()).get("network_bytes", 0)
                )
            elif operation == "job" and phase == "fetch":
                transferred = c.max_download_bytes

    for job in c.jobs:
        if job.phase == phase:
            if phase == "fetch" and transferred >= c.max_download_bytes:
                states.append(
                    {
                        "unit": job.id,
                        "status": "blocked",
                        "reason": "Download budget exhausted",
                    }
                )
            else:
                execute("job", job.id, folder / "jobs" / f"{job.id}.json")
    all_scores: list[dict[str, object]] = []
    if phase == "run":
        for case in c.cases:
            for method in c.methods:
                target = folder / case.id / f"{method.id}.nc"
                execute("predict", f"{case.id}/{method.id}", target)
            dependencies = score_inputs(c, case, folder / case.id)
            fingerprint = hashlib.sha256(
                json.dumps(dependencies, sort_keys=True).encode()
            ).hexdigest()
            scores_path = folder / case.id / f"scores-{fingerprint}.json"
            execute("score", case.id, scores_path, dependencies)
            if (
                scores_path.exists()
                and scores_path.with_suffix(".json.receipt.json").exists()
            ):
                all_scores.extend(json.loads(scores_path.read_text())["rows"])
        # Each invocation publishes a fresh snapshot, including partial/failed runs.
        snapshot = folder / f"snapshot-{time.time_ns()}"
        snapshot.mkdir()
        if all_scores:
            with atomic_output(snapshot / "scores.csv") as temporary:
                with temporary.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(all_scores[0]))
                    writer.writeheader()
                    writer.writerows(all_scores)
    else:
        snapshot = folder / f"snapshot-{time.time_ns()}"
        snapshot.mkdir()
    summary = {
        "schema_version": VERSION,
        "campaign": c.id,
        "track": c.track,
        "status": "completed"
        if all(s["status"] == "completed" for s in states)
        else "incomplete",
        "states": states,
        "network_bytes": transferred,
        "seconds": time.monotonic() - started,
        "scores": len(all_scores),
        "snapshot": str(snapshot),
    }
    write_json(snapshot / "summary.json", summary)
    return summary


if __name__ == "__main__":
    worker(Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]))
