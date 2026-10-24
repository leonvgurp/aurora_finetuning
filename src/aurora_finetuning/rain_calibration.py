"""Checkpoint-free validation and test scoring for deterministic rain methods."""

import json
from datetime import datetime, timedelta
from importlib.metadata import version
from pathlib import Path

import numpy as np
import xarray as xr

from aurora_finetuning.acquisition import sha256
from aurora_finetuning.config import Config
from aurora_finetuning.grid import FloatArray, make_grid
from aurora_finetuning.radar_data import history_at, measured_targets
from aurora_finetuning.radar_nowcast import VERSION, motion_totals
from aurora_finetuning.radar_training import RadarCase, read_radar_cases
from aurora_finetuning.rainfall_metrics import (
    bootstrap_cases,
    coarse_grid_report,
    event_scores,
    hourly_report,
    motion_forecast,
)
from aurora_finetuning.telemetry import Recorder, write_json


def compatibility(config: Config) -> dict[str, object]:
    """Describe numerical/domain settings independently of storage location."""
    return {
        "revision": config.checkpoint_revision,
        "precision": config.precision,
        "resolution": config.resolution,
        "grid": make_grid(config).describe(),
        "regional": "continuous grid, canonical positions, no cyclic wrap",
        "pysteps": version("pysteps"),
        "motion_code_sha256": sha256(Path(__file__).with_name("radar_nowcast.py")),
        "rain_code_sha256": sha256(Path(__file__)),
    }


def load_weights(path: Path, config: Config) -> list[float]:
    """Require checksummed calibration for the exact base model and region."""
    digest = json.loads(path.with_suffix(".sha256.json").read_text())
    if not isinstance(digest, dict) or sha256(path) != digest.get("sha256"):
        raise ValueError("Rain calibration checksum mismatch")
    document = json.loads(path.read_text())
    if (
        not isinstance(document, dict)
        or document.get("version") != VERSION
        or document.get("compatibility") != compatibility(config)
    ):
        raise ValueError("Incompatible rainfall calibration")
    weights = document.get("weights")
    if (
        not isinstance(weights, list)
        or len(weights) != 8
        or any(isinstance(w, bool) or w not in (0, 0.25, 0.5, 0.75, 1) for w in weights)
    ):
        raise ValueError("Invalid rainfall blending weights")
    return [float(w) for w in weights]


def case_arrays(
    case: RadarCase, config: Config, recorder: Recorder
) -> tuple[dict[str, FloatArray], FloatArray, xr.Dataset]:
    """Read one complete case; measured targets never enter history or motion."""
    for p in (case.forecast, case.radar):
        recorder.inputs[str(p)] = sha256(p)
    with xr.open_dataset(case.forecast) as source:
        forecast = source.load()
    with xr.open_dataset(case.radar) as source:
        radar = source.load()
    init = datetime.fromisoformat(str(forecast.attrs["initialization_utc"]))
    if not timedelta(0) <= case.initialization - init <= timedelta(hours=12):
        raise ValueError("Case atmospheric initialization is future-dated or stale")
    if (
        forecast.attrs.get("checkpoint_revision") != config.checkpoint_revision
        or forecast.attrs.get("resolution") != config.resolution
        or forecast.attrs.get("precision") != config.precision
        or forecast.precipitation_mm.attrs.get("units") != "mm"
    ):
        raise ValueError("Incompatible Aurora rainfall inputs")
    grid = make_grid(config)
    if not np.array_equal(forecast.latitude, grid.latitude) or not np.array_equal(
        forecast.longitude, grid.longitude
    ):
        raise ValueError("Rain case grid differs from configured domain")
    history = np.asarray(history_at(radar, case.initialization).values, dtype=float)
    ends = [case.initialization + timedelta(hours=i) for i in range(1, 9)]
    totals = motion_totals(radar, case.initialization, ends)
    observed = np.stack(
        [measured_targets(radar, case.initialization, i).values for i in range(1, 9)]
    )
    baseline_maps: dict[datetime, FloatArray] = {}
    baseline = []
    fft = []
    for end in ends:
        base_hour = np.zeros_like(history[-1])
        fft_hour = np.zeros_like(history[-1])
        for i in range(12):
            valid = end - timedelta(minutes=5 * (11 - i))
            key = valid.replace(minute=0, second=0, microsecond=0)
            if valid != key:
                key += timedelta(hours=1)
            if key not in baseline_maps:
                # Match the public locality query: rainfall uses a nearest cell.
                field = forecast.precipitation_mm.sel(time=np.datetime64(key, "ns"))
                mapped = (
                    field.sel(
                        latitude=radar.latitude,
                        longitude=radar.longitude,
                        method="nearest",
                    )
                    / 12
                )
                inside = (
                    (radar.latitude >= forecast.latitude.min())
                    & (radar.latitude <= forecast.latitude.max())
                    & (radar.longitude >= forecast.longitude.min())
                    & (radar.longitude <= forecast.longitude.max())
                )
                baseline_maps[key] = np.asarray(
                    mapped.where(inside).values, dtype=float
                )
            base_hour += baseline_maps[key]
            fft_hour += motion_forecast(
                history, int((valid - case.initialization).total_seconds() / 60)
            )
        baseline.append(base_hour)
        fft.append(fft_hour)
    raw = np.where(totals.valid_bins.values == 12, totals.motion_mm.values, np.nan)
    methods = {
        "aurora": np.asarray(baseline, dtype=float),
        "radar_persistence": np.broadcast_to(history[-1] * 12, observed.shape).copy(),
        "fft_motion": np.asarray(fft, dtype=float),
        "pysteps_motion": np.asarray(raw, dtype=float),
        "motion_known": np.asarray(totals.motion_mm.values, dtype=float),
        "motion_count": np.asarray(totals.valid_bins.values, dtype=float),
    }
    return methods, np.asarray(observed, dtype=float), radar


def blend(
    base: FloatArray, known: FloatArray, count: FloatArray, weights: list[float]
) -> FloatArray:
    """Correct only supported bins, leaving unavailable future bins at Aurora."""
    return np.asarray(
        base + np.array(weights)[:, None, None] * (known - base * count / 12),
        dtype=float,
    )


def choose_weight(
    base: FloatArray, known: FloatArray, count: FloatArray, target: FloatArray
) -> tuple[float, list[dict[str, object]]]:
    """Select on identical support, requiring strict MAE gain without worse CSI."""
    raw = event_scores(base, target, 0.1)
    chosen = 0.0
    best = float(raw["mae"]) if raw["mae"] is not None else float("inf")
    trials = []
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        candidate = base + weight * (known - base * count / 12)
        score = event_scores(candidate, target, 0.1)
        trials.append({"weight": weight, "scores": score})
        csi_ok = (score["csi"] is None and raw["csi"] is None) or (
            score["csi"] is not None
            and raw["csi"] is not None
            and score["csi"] >= raw["csi"]
        )
        if score["mae"] is not None and score["mae"] < best and csi_ok:
            best = float(score["mae"])
            chosen = weight
    return chosen, trials


def calibrate_rain(
    config: Config, manifest: Path, output: Path, recorder: Recorder
) -> dict[str, object]:
    """Select lead weights using validation only, without opening held-out cases."""
    cases = read_radar_cases(manifest)
    accum: list[list[tuple[FloatArray, FloatArray, FloatArray, FloatArray]]] = [
        [] for _ in range(8)
    ]
    for case in cases["validation"]:
        methods, target, _ = case_arrays(case, config, recorder)
        for i in range(8):
            accum[i].append(
                tuple(
                    a[i].ravel()
                    for a in (
                        methods["aurora"],
                        methods["motion_known"],
                        methods["motion_count"],
                        target,
                    )
                )
            )
    weights = []
    trials = []
    for entries in accum:
        joined = [np.concatenate([entry[i] for entry in entries]) for i in range(4)]
        weight, scores = choose_weight(*joined)
        weights.append(weight)
        trials.append(scores)
    result = {
        "version": VERSION,
        "compatibility": compatibility(config),
        "weights": weights,
        "validation": trials,
        "input_sha256": recorder.inputs.copy(),
        "cases": [c.initialization.isoformat() for c in cases["validation"]],
        "manifest_sha256": sha256(manifest),
        "pilot": True,
        "test_used": False,
    }
    write_json(output, result)
    write_json(output.with_suffix(".sha256.json"), {"sha256": sha256(output)})
    return result


def evaluate_rain(
    config: Config, manifest: Path, calibration: Path | None, recorder: Recorder
) -> dict[str, object]:
    """Score test cases with paired baselines; raw unsupported motion stays missing."""
    cases = read_radar_cases(manifest)["test"]
    weights = load_weights(calibration, config) if calibration else [1.0] * 8
    if calibration is not None:
        validation = json.loads(calibration.read_text())["cases"]
        last_validation = max(datetime.fromisoformat(t) for t in validation)
        if cases[0].initialization - timedelta(hours=2) <= last_validation + timedelta(
            hours=38
        ):
            raise ValueError("Test windows overlap calibration data or its 30h embargo")
    reports = []
    errors: dict[str, list[list[float]]] = {}
    for case in cases:
        methods, observed, geometry = case_arrays(case, config, recorder)
        methods["combined"] = blend(
            methods["aurora"],
            methods.pop("motion_known"),
            methods.pop("motion_count"),
            weights,
        )
        scores = {}
        for name, values in methods.items():
            support = np.isfinite(values) & np.isfinite(observed)
            paired = np.where(support, methods["aurora"], np.nan)
            scores[name] = {
                "native": hourly_report(values, observed),
                "coarse": coarse_grid_report(
                    values,
                    observed,
                    geometry.latitude.values,
                    geometry.longitude.values,
                    make_grid(config),
                    config.resolution,
                ),
                "aurora_on_same_support": hourly_report(
                    paired, np.where(support, observed, np.nan)
                ),
                "aurora_coarse_on_same_support": coarse_grid_report(
                    paired,
                    np.where(support, observed, np.nan),
                    geometry.latitude.values,
                    geometry.longitude.values,
                    make_grid(config),
                    config.resolution,
                ),
            }
            errors.setdefault(name, []).append(
                [
                    float(
                        np.abs(values[i][support[i]] - observed[i][support[i]]).mean()
                    )
                    if support[i].any()
                    else float("nan")
                    for i in range(8)
                ]
            )
        reports.append({"cutoff": case.initialization.isoformat(), "methods": scores})
    return {
        "test": reports,
        "weights": weights,
        "pilot": True,
        "scientific_skill_verified": False,
        "mode": "retrospective unless publication availability reconstructed",
        "uncertainty": {
            name: bootstrap_cases(np.asarray(values, dtype=float))
            for name, values in errors.items()
        },
        "input_sha256": recorder.inputs.copy(),
    }
