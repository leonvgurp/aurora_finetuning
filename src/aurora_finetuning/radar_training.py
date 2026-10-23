"""Bounded training and held-out evaluation of the radar-conditioned rain adapter."""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from aurora_finetuning.acquisition import sha256
from aurora_finetuning.config import Config
from aurora_finetuning.evaluation import metrics
from aurora_finetuning.grid import FloatArray
from aurora_finetuning.model import check_device, check_memory
from aurora_finetuning.radar_data import history_at, validate_native
from aurora_finetuning.radar_model import (
    ARCHITECTURE,
    RADIUS,
    RadarRainAdapter,
    baseline_at,
    feature_tensors,
    load_radar_adapter,
    radar_loss,
)
from aurora_finetuning.telemetry import Recorder, write_json


@dataclass(frozen=True)
class RadarCase:
    """Paths and the strict observation cutoff for one retrospective experiment."""

    initialization: datetime
    forecast: Path
    radar: Path


def read_radar_cases(path: Path) -> dict[str, list[RadarCase]]:
    """Resolve manifest paths and require separated chronological splits."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or set(raw) != {"train", "validation", "test"}:
        raise ValueError("Radar cases require train, validation and test arrays")
    cases: dict[str, list[RadarCase]] = {}
    for split in ("train", "validation", "test"):
        if not isinstance(raw[split], list) or not raw[split]:
            raise ValueError(f"Radar {split} cases must be a nonempty array")
        entries: list[RadarCase] = []
        for item in raw[split]:
            if (
                not isinstance(item, dict)
                or set(item) != {"initialization", "forecast", "radar"}
                or not all(isinstance(v, str) for v in item.values())
            ):
                raise ValueError(
                    "Each radar case needs initialization, forecast and radar strings"
                )
            time = datetime.fromisoformat(item["initialization"])
            if (
                time.tzinfo is not None
                or time.minute % 5
                or time.second
                or time.microsecond
            ):
                raise ValueError(
                    "Radar cutoffs must be naive UTC on five-minute boundaries"
                )
            entries.append(
                RadarCase(
                    time,
                    (path.parent / item["forecast"]).resolve(),
                    (path.parent / item["radar"]).resolve(),
                )
            )
        if len({entry.initialization for entry in entries}) != len(entries):
            raise ValueError("Duplicate radar case cutoffs")
        cases[split] = sorted(entries, key=lambda entry: entry.initialization)
    for before, after in (("train", "validation"), ("validation", "test")):
        if (
            cases[before][-1].initialization + timedelta(hours=30)
            >= cases[after][0].initialization
        ):
            raise ValueError(
                "Radar splits require >30h separation of history/target windows"
            )
    return cases


def validate_case(
    case: RadarCase, config: Config, base_adapter: Path | None, recorder: Recorder
) -> None:
    """Validate training/validation windows without reading held-out test data."""
    for path in (case.forecast, case.radar):
        recorder.inputs[str(path)] = sha256(path)
    with (
        xr.open_dataset(case.forecast) as forecast,
        xr.open_dataset(case.radar) as radar,
    ):
        validate_native(radar)
        if min(radar.sizes["y"], radar.sizes["x"]) < config.radar_patch_size:
            raise ValueError("Native radar dataset is smaller than radar_patch_size")
        initialization = datetime.fromisoformat(
            str(forecast.attrs["initialization_utc"])
        )
        if initialization.tzinfo is not None or not timedelta(
            0
        ) <= case.initialization - initialization <= timedelta(hours=12):
            raise ValueError(
                "Aurora initialization must be at or before the radar cutoff "
                "and no more than 12 hours earlier"
            )
        if (
            forecast.attrs.get("checkpoint_revision") != config.checkpoint_revision
            or forecast.attrs.get("resolution") != config.resolution
            or forecast.attrs.get("adapter_sha256", "")
            != (sha256(base_adapter) if base_adapter else "")
            or forecast.precipitation_mm.attrs.get("units") != "mm"
        ):
            raise ValueError(
                "Aurora forecast has incompatible model, adapter, resolution or units"
            )
        history_at(radar, case.initialization)
        wanted = [
            np.datetime64(case.initialization + timedelta(minutes=5 * i), "ns")
            for i in range(1, 73)
        ]
        if not all(time in radar.time.values for time in wanted):
            raise ValueError(
                "Radar case requires six hours of subsequent measured targets"
            )
        # Check spatial and temporal baseline coverage on a small native-grid slice.
        for minute in (5, 360):
            values = baseline_at(
                forecast,
                radar.isel(y=slice(None, None, 16), x=slice(None, None, 16)),
                case.initialization + timedelta(minutes=minute),
            )
            if not np.isfinite(values.values).all():
                raise ValueError(
                    "Aurora baseline does not cover the native radar patch"
                )


def example(
    case: RadarCase,
    minutes: int,
    config: Config,
    origin: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load one bounded spatial patch; targets are never included in input features."""
    with (
        xr.open_dataset(case.radar) as dataset,
        xr.open_dataset(case.forecast) as forecast,
    ):
        size = config.radar_patch_size
        row, col = origin or (
            (dataset.sizes["y"] - size) // 2,
            (dataset.sizes["x"] - size) // 2,
        )
        patch = dataset.isel(y=slice(row, row + size), x=slice(col, col + size))
        history = history_at(patch, case.initialization).load()
        valid = case.initialization + timedelta(minutes=minutes)
        try:
            target = patch.precipitation_mm.sel(time=np.datetime64(valid, "ns")).load()
        except KeyError as error:
            raise ValueError("Missing measured radar target") from error
        if np.isinf(target.values).any() or np.any(
            np.isfinite(target.values) & (target.values < 0)
        ):
            raise ValueError("Invalid measured radar target")
        baseline = baseline_at(forecast, patch, valid).load()
        initialization = datetime.fromisoformat(
            str(forecast.attrs["initialization_utc"])
        )
        features, base = feature_tensors(
            history,
            baseline,
            minutes,
            (case.initialization - initialization).total_seconds() / 3600,
            config,
        )
        observed = torch.tensor(
            target.values[None], dtype=torch.float32, device=config.device
        )
        persistence = torch.tensor(
            history.values[-1:], dtype=torch.float32, device=config.device
        )
    return features, base, observed, persistence


def train_radar(
    config: Config,
    manifest: Path,
    output: Path,
    recorder: Recorder,
    steps: int | None = None,
    base_adapter: Path | None = None,
) -> dict[str, object]:
    """Train a small rain adapter and select weights using validation observations."""
    cases = read_radar_cases(manifest)
    count = config.max_steps if steps is None else steps
    if count < 1:
        raise ValueError("Radar training steps must be positive")
    if output.exists():
        raise FileExistsError(output)
    check_device(config)
    for case in cases["train"] + cases["validation"]:
        validate_case(case, config, base_adapter, recorder)
    recorder.inputs[str(manifest)] = sha256(manifest)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = RadarRainAdapter().to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    output.mkdir(parents=True)
    history: list[dict[str, float | int]] = []
    best = float("inf")
    for step in range(count):
        case = cases["train"][step % len(cases["train"])]
        minutes = int(rng.integers(1, 73)) * 5
        with xr.open_dataset(case.radar) as ds:
            origin = (
                int(rng.integers(0, ds.sizes["y"] - config.radar_patch_size + 1)),
                int(rng.integers(0, ds.sizes["x"] - config.radar_patch_size + 1)),
            )
        model.train()
        with recorder.measure("radar.training_step"):
            features, baseline, target, _ = example(case, minutes, config, origin)
            optimizer.zero_grad(set_to_none=True)
            loss = radar_loss(model(features, baseline), target, config)
            torch.autograd.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with recorder.measure("radar.validation"), torch.no_grad():
            scores = []
            for validation in cases["validation"]:
                for lead in (5, 30, 60, 120, 240, 360):
                    features, baseline, target, _ = example(validation, lead, config)
                    scores.append(
                        float(
                            radar_loss(model(features, baseline), target, config).cpu()
                        )
                    )
        score = float(np.mean(scores))
        if not np.isfinite(score):
            raise ValueError("Nonfinite radar validation loss")
        history.append(
            {
                "step": step + 1,
                "training_loss": float(loss.detach().cpu()),
                "validation_loss": score,
            }
        )
        if score < best:
            best = score
            checkpoint = output / "best.pt"
            torch.save(
                {
                    "architecture": ARCHITECTURE,
                    "parameters": model.state_dict(),
                    "base_revision": config.checkpoint_revision,
                    "base_adapter_sha256": sha256(base_adapter)
                    if base_adapter
                    else None,
                    "base_resolution": config.resolution,
                    "precipitation_scale": config.precipitation_scale,
                    "trained_steps": step + 1,
                    "seed": config.seed,
                },
                checkpoint,
            )
        check_memory(config)
    checkpoint = output / "best.pt"
    write_json(
        checkpoint.with_suffix(".json"),
        {
            "sha256": sha256(checkpoint),
            "architecture": ARCHITECTURE,
            "best_validation_loss": best,
        },
    )
    report: dict[str, object] = {
        "mode": "frozen Aurora with trainable native-grid radar rain decoder",
        "radar_inputs": "24 measured five-minute frames at or before each cutoff",
        "targets": "subsequent measured five-minute rainfall; no KNMI forecasts",
        "history": history,
        "best_validation_loss": best,
        "test_evaluated": False,
        "validation_sampling": "central patch; 5, 30, 60, 120, 240, 360 minute leads",
        "cases": json.loads(manifest.read_text()),
        "input_sha256": recorder.inputs,
        "config": {
            "seed": config.seed,
            "patch_size": config.radar_patch_size,
            "learning_rate": config.learning_rate,
            "wet_weight": config.radar_wet_weight,
            "precipitation_scale": config.precipitation_scale,
        },
        "timings": recorder.report(),
    }
    write_json(output / "training.json", report)
    return report


def evaluate_adapter(
    config: Config,
    manifest: Path,
    adapter: Path,
    recorder: Recorder,
    base_adapter: Path | None = None,
) -> dict[str, object]:
    """Score held-out central patches at all 72 steps, including hourly accumulation."""
    cases = read_radar_cases(manifest)["test"]
    check_device(config)
    model = load_radar_adapter(adapter, config, base_adapter, recorder)
    output: list[dict[str, object]] = []
    for case in cases:
        validate_case(case, config, base_adapter, recorder)
        predictions: list[FloatArray] = []
        observations: list[FloatArray] = []
        baselines: list[FloatArray] = []
        persistence: list[FloatArray] = []
        for lead in range(5, 361, 5):
            with recorder.measure("radar.test_step"), torch.inference_mode():
                features, base, target, persist = example(case, lead, config)
                predicted = model(features, base)
            for records, values in (
                (predictions, predicted),
                (observations, target),
                (baselines, base),
                (persistence, persist),
            ):
                records.append(values[0, RADIUS:-RADIUS, RADIUS:-RADIUS].cpu().numpy())
        measured = (
            np.stack(observations).reshape(6, 12, *observations[0].shape).sum(axis=1)
        )
        row: dict[str, object] = {"cutoff": case.initialization.isoformat()}
        for name, records in (
            ("radar_conditioned", predictions),
            ("aurora", baselines),
            ("radar_persistence", persistence),
        ):
            hourly = np.stack(records).reshape(6, 12, *records[0].shape).sum(axis=1)
            row[name] = {
                "hourly": metrics(hourly, measured),
                "six_hour_total": metrics(hourly.sum(axis=0), measured.sum(axis=0)),
            }
        output.append(row)
        check_memory(config)
    return {
        "test": output,
        "spatial_sampling": "central native-grid patch interior only",
        "input_sha256": recorder.inputs,
        "scientific_skill_verified": False,
    }
