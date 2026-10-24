"""Explicit bounded benchmarks; no downloads or training are implicit."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr

from aurora_finetuning.config import Config
from aurora_finetuning.engine import ForecastEngine
from aurora_finetuning.forecast import query_grid
from aurora_finetuning.model import save_forecast
from aurora_finetuning.telemetry import Recorder, write_json

TOLERANCES = {
    "temperature_c": 0.05,
    "pressure_hpa": 0.1,
    "wind_u_ms": 0.1,
    "wind_v_ms": 0.1,
    "precipitation_mm": 0.05,
}


def numerical_comparison(
    reference: xr.Dataset, candidate: xr.Dataset
) -> dict[str, object]:
    """Compare aligned physical outputs; never promote incompatible precision."""
    reference, candidate = xr.align(reference, candidate, join="exact")
    errors = {}
    for name, tolerance in TOLERANCES.items():
        if name not in reference or name not in candidate:
            raise ValueError(f"Precision comparison missing {name}")
        if (
            not np.isfinite(reference[name]).all()
            or not np.isfinite(candidate[name]).all()
        ):
            raise ValueError("Nonfinite precision benchmark")
        error = float(np.abs(reference[name].values - candidate[name].values).mean())
        errors[name] = {
            "mae_difference": error,
            "tolerance": tolerance,
            "passed": error <= tolerance,
        }
    return {
        "variables": errors,
        "passed": all(row["passed"] for row in errors.values()),
    }


def benchmark(
    config: Config,
    initialization: datetime,
    source: str,
    output: Path,
    recorder: Recorder,
    compare_precision: bool = False,
) -> dict[str, object]:
    """Measure cold engine, warm engine and cache query using existing inputs."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    config = replace(config, precision="float32")
    with ForecastEngine(config, recorder) as engine:
        with recorder.measure("benchmark.cold_engine"):
            reference = engine.predict(initialization, config.hours, source)
        with recorder.measure("benchmark.warm_engine"):
            repeat = engine.predict(initialization, config.hours, source)
    save_forecast(reference, output / "reference.nc")
    with (
        recorder.measure("benchmark.cached_query"),
        xr.open_dataset(output / "reference.nc") as cached,
    ):
        query_grid(cached, 52.1, 5.1, config.hours, initialization.replace(tzinfo=UTC))
    report: dict[str, object] = {
        "repeat": numerical_comparison(reference, repeat),
        "precision_promoted": False,
        "network_bytes": recorder.network_bytes,
        "timings": recorder.report(),
        "scope": (
            "existing inputs; cold/warm engine and cache query; "
            "observation refresh timed by weather reports"
        ),
    }
    if compare_precision:
        with ForecastEngine(replace(config, precision="float16"), recorder) as engine:
            with recorder.measure("benchmark.float16"):
                candidate = engine.predict(initialization, config.hours, source)
        report["float16"] = numerical_comparison(reference, candidate)
        save_forecast(candidate, output / "float16.nc")
    write_json(output / "benchmark.json", report)
    return report


def compare_reference(
    config: Config,
    initialization: datetime,
    source: str,
    reference_path: Path,
    output: Path,
    recorder: Recorder,
) -> dict[str, object]:
    """Compare one reduced-precision run with an existing float32 reference."""
    if output.exists():
        raise FileExistsError(output)
    with xr.open_dataset(reference_path) as dataset:
        reference = dataset.load()
    if (
        reference.attrs.get("checkpoint_revision") != config.checkpoint_revision
        or reference.attrs.get("precision") != "float32"
        or reference.attrs.get("initialization_utc") != initialization.isoformat()
    ):
        raise ValueError("Reference model/precision/initialization mismatch")
    from aurora_finetuning.acquisition import sha256

    recorder.inputs[str(reference_path)] = sha256(reference_path)
    with ForecastEngine(replace(config, precision="float16"), recorder) as engine:
        with recorder.measure("benchmark.float16"):
            candidate = engine.predict(
                initialization, int(reference.sizes["time"]), source
            )
    comparison = numerical_comparison(reference, candidate)
    output.mkdir(parents=True)
    save_forecast(candidate, output / "float16.nc")
    report = {
        "float16": comparison,
        "precision_promoted": False,
        "timings": recorder.report(),
        "reference": str(reference_path),
    }
    write_json(output / "benchmark.json", report)
    return report
