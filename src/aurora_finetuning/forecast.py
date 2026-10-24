"""Location-based Python interface over timed regional Aurora forecasts."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import xarray as xr

from aurora_finetuning.config import load_config
from aurora_finetuning.model import initialization_of
from aurora_finetuning.telemetry import Recorder


def validate_location(latitude: float, longitude: float) -> None:
    """Reject locations outside the European Netherlands bounding envelope.

    This envelope includes border/sea cells; an exact administrative polygon is
    deliberately not claimed. Coordinates are required in v1.
    """
    if not (50.7 <= latitude <= 53.7 and 3.2 <= longitude <= 7.3):
        raise ValueError("Location must be within the European Netherlands envelope")


def query_grid(
    dataset: xr.Dataset, latitude: float, longitude: float, hours: int, start: datetime
) -> dict[str, object]:
    """Extract future hourly points; interpolate temperature but not rainfall."""
    validate_location(latitude, longitude)
    if start.tzinfo is None:
        raise ValueError("Query start must include a timezone")
    if not 1 <= hours <= 24:
        raise ValueError("hours must be between 1 and 24")
    for name, value in (("latitude", latitude), ("longitude", longitude)):
        axis = dataset[name].values
        if not float(axis.min()) <= value <= float(axis.max()):
            raise ValueError(f"Requested {name} lies outside forecast grid")
    start_utc = start.astimezone(UTC).replace(tzinfo=None)
    first = start_utc.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    times = [first + timedelta(hours=i) for i in range(hours)]
    try:
        selected = dataset.sel(time=[np.datetime64(t, "ns") for t in times])
    except KeyError as error:
        raise ValueError(
            "Forecast does not cover the requested future hours"
        ) from error
    temp = selected.temperature_c.interp(latitude=latitude, longitude=longitude).values
    rain = selected.precipitation_mm.sel(
        latitude=latitude, longitude=longitude, method="nearest"
    ).values
    extra = {
        name: selected[name].interp(latitude=latitude, longitude=longitude).values
        for name in ("pressure_hpa", "wind_u_ms", "wind_v_ms")
        if name in selected
    }
    rows: list[dict[str, object]] = []
    for i, time in enumerate(times):
        if not np.isfinite(temp[i]) or not np.isfinite(rain[i]):
            raise ValueError("Missing predictions at query location")
        end = time.replace(tzinfo=UTC)
        rows.append(
            {
                "time": end.astimezone(ZoneInfo("Europe/Amsterdam")).isoformat(),
                "interval_start": (end - timedelta(hours=1))
                .astimezone(ZoneInfo("Europe/Amsterdam"))
                .isoformat(),
                "temperature_c": round(float(temp[i]), 2),
                "precipitation_mm": round(float(rain[i]), 3),
            }
        )
    for i, row in enumerate(rows):
        for name, values in extra.items():
            if not np.isfinite(values[i]):
                raise ValueError(f"Missing predictions for {name}")
            row[name] = float(values[i])
        if "wind_u_ms" in extra and "wind_v_ms" in extra:
            row["wind_speed_ms"] = float(
                np.hypot(extra["wind_u_ms"][i], extra["wind_v_ms"][i])
            )
        for name in ("temperature_c", "precipitation_mm", *extra):
            row[f"aurora_{name}"] = row[name]
            row[f"{name}_source"] = "Aurora"
    return {
        "latitude": latitude,
        "longitude": longitude,
        "initialization_utc": str(dataset.attrs["initialization_utc"]),
        "source": str(dataset.attrs.get("source", "unknown")),
        "experimental": True,
        "hourly": rows,
    }


def forecast(
    latitude: float,
    longitude: float,
    hours: int = 8,
    config_path: str | Path | None = None,
    *,
    initialization: datetime | None = None,
    start: datetime | None = None,
    cached_forecast: Path | None = None,
    adapter: Path | None = None,
    source: str = "era5",
    device: str | None = None,
    recorder: Recorder | None = None,
) -> dict[str, object]:
    """Query a cached forecast or run an explicit historical ERA5 experiment.

    Live input completeness must be established with the IFS adapter before live
    model initialization is supported. Historical mode requires explicit UTC
    initialization and start; it never relabels historical data as today's weather.
    """
    validate_location(latitude, longitude)
    if adapter is not None:
        raise ValueError("Legacy surface-head adapters are unsupported")
    if not 1 <= hours <= 24:
        raise ValueError("hours must be between 1 and 24")
    config = load_config(Path(config_path) if config_path is not None else None)
    if device is not None:
        from dataclasses import replace

        config = replace(config, device=device)
    recorder = recorder or Recorder()
    if cached_forecast is not None:
        with recorder.measure("forecast.read_cache"):
            with xr.open_dataset(cached_forecast) as cached:
                dataset = cached.load()
        start = start or datetime.now(UTC)
        age = (
            start.astimezone(UTC).replace(tzinfo=None) - initialization_of(dataset)
        ).total_seconds() / 3600
        if age < 0 or age > config.max_input_age_hours:
            raise ValueError("Cached forecast initialization is stale or in the future")
    else:
        if source == "ifs" and initialization is not None:
            start = start or datetime.now(UTC)
        if initialization is None or start is None:
            raise ValueError(
                "Provide cached_forecast or historical initialization and start; "
                "live IFS initialization is not yet verified"
            )
        if initialization.tzinfo is not None or start.tzinfo is None:
            raise ValueError(
                "initialization must be naive UTC; start must include a timezone"
            )
        age = (
            start.astimezone(UTC).replace(tzinfo=None) - initialization
        ).total_seconds() / 3600
        if age < 0 or age > config.max_input_age_hours:
            raise ValueError(
                "Query start must be within max_input_age_hours of initialization"
            )
        horizon = int(np.ceil(age)) + hours
        from aurora_finetuning.engine import ForecastEngine

        with ForecastEngine(config, recorder) as engine:
            dataset = engine.predict(initialization, horizon, source)
    with recorder.measure("forecast.extract_location"):
        result = query_grid(dataset, latitude, longitude, hours, start)
    result["timings"] = recorder.report()
    result["initialization_age_hours"] = age
    return result
