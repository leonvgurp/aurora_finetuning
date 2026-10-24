"""Explicit bounded case-study runs using frozen Aurora and cached step-zero data."""

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from aurora_finetuning.acquisition import sha256
from aurora_finetuning.config import Config
from aurora_finetuning.engine import ForecastEngine
from aurora_finetuning.forecast import query_grid
from aurora_finetuning.grid import make_grid
from aurora_finetuning.ifs import decode_grib, sea_ice_supplement
from aurora_finetuning.model import save_forecast
from aurora_finetuning.telemetry import Recorder, write_json
from aurora_finetuning.weather import pipeline_identity


def run_candidate(
    config: Config,
    initialization: datetime,
    start: datetime,
    latitude: float,
    longitude: float,
    output: Path,
) -> dict[str, object]:
    """Run the complete requested window; preserve predictions and resource costs."""
    if output.exists():
        raise FileExistsError(output)
    if initialization.tzinfo is not None or start.tzinfo is None:
        raise ValueError("Use naive UTC initialization and an aware query start")
    if config.precision != "float32":
        raise ValueError("Reference case experiments require float32")
    first = start.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    horizon = int((first.replace(tzinfo=None) - initialization).total_seconds() / 3600)
    horizon += config.hours
    if not 1 <= horizon <= 48:
        raise ValueError("Case exceeds the bounded 48-hour rollout")
    output.mkdir(parents=True)
    recorder = Recorder()
    try:
        with recorder.measure("case.full_request"):
            with ForecastEngine(config, recorder) as engine:
                grid = engine.predict(initialization, horizon, "ifs")
            result = query_grid(grid, latitude, longitude, config.hours, start)
            save_forecast(grid, output / "forecast.nc")
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "error_type": type(error).__name__,
                "timings": recorder.report(),
                "input_sha256": recorder.inputs,
            },
        )
        raise
    settings = asdict(config)
    settings["root"] = str(config.root)
    document = {
        "result": result,
        "config": settings,
        "code_sha256": pipeline_identity(),
        "input_sha256": recorder.inputs,
        "forecast_sha256": sha256(output / "forecast.nc"),
        "timings": recorder.report(),
        "network_bytes": recorder.network_bytes,
        "retained_bytes": (output / "forecast.nc").stat().st_size,
        "scope": "retrospective development replay; no future forecast inputs",
    }
    write_json(output / "prediction.json", document)
    return document


def prepare_larger_inputs(
    source: Config, larger: Config, initialization: datetime
) -> dict[str, str]:
    """Recrop retained global GRIB, without redownloading atmospheric fields."""
    from datetime import timedelta

    inputs = {}
    grid = make_grid(larger)
    for time in (initialization - timedelta(hours=6), initialization):
        ice_paths = sorted((larger.root / "data/raw/sea-ice").glob("*.nc"))
        if not ice_paths:
            raise ValueError("Acquire an explicit larger-domain sea-ice snapshot first")
        ice = sea_ice_supplement(ice_paths[-1], time, larger)
        inputs[str(ice_paths[-1])] = sha256(ice_paths[-1])
        for kind in ("single", "pressure"):
            stem = f"{time.date()}-{kind}-{time.hour:02d}"
            files = [source.root / "data/raw/ifs" / f"{stem}.grib2"]
            if kind == "single":
                files.append(source.root / "data/raw/ifs" / f"{stem}.soil.grib2")
            for path in files:
                inputs[str(path)] = sha256(path)
            target = larger.root / "data/raw/ifs" / f"{stem}.nc"
            decoded = decode_grib(
                files, grid, kind, time, ice if kind == "single" else None
            )
            save_forecast(decoded, target)
            write_json(
                target.with_suffix(".json"),
                {
                    "sha256": sha256(target),
                    "source": "retained global IFS step-zero GRIB",
                    "grid": grid.describe(),
                    "source_sha256": {str(p): sha256(p) for p in files},
                },
            )
    return inputs


def input_diagnostics(config: Config, initialization: datetime) -> dict[str, object]:
    """Record local initial temperature and input units without model inference."""
    from datetime import timedelta

    from aurora_finetuning.acquisition import LEVELS
    from aurora_finetuning.preparation import open_era5

    times = [initialization - timedelta(hours=6), initialization]
    surface = open_era5(
        config.root / "data/raw/ifs", make_grid(config), times, "single"
    )
    atmos = open_era5(
        config.root / "data/raw/ifs", make_grid(config), times, "pressure"
    )
    return {
        "input_times_utc": [t.isoformat() for t in times],
        "grid": make_grid(config).describe(),
        "surface_units": {
            k: v.attrs.get("units") for k, v in surface.data_vars.items()
        },
        "pressure_levels_hpa": list(LEVELS),
        "available_levels_hpa": atmos.level.values.tolist(),
        "initial_temperature_c": (
            surface.t2m.interp(latitude=52.1168945, longitude=5.40476582) - 273.15
        ).values.tolist(),
        "solar_context": (
            "prescribed insolation for both UTC input times; "
            "upstream updates at each valid time"
        ),
        "output_conversions": (
            "2t K minus 273.15; inverse-log-transformed scaled_tp_1h m times 1000; "
            "msl Pa divided by 100; wind m/s"
        ),
        "regional_caveat": (
            "global pretrained model on cropped domain "
            "with non-wrapping regional attention"
        ),
    }
