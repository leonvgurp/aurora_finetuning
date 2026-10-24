"""Bounded live-input discovery and reusable regional forecast caching."""

import hashlib
import json
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import requests
import xarray as xr

from aurora_finetuning.acquisition import sha256
from aurora_finetuning.assets import download_assets
from aurora_finetuning.config import Config
from aurora_finetuning.engine import ForecastEngine
from aurora_finetuning.forecast import query_grid, validate_location
from aurora_finetuning.grid import make_grid
from aurora_finetuning.ifs import audit_ifs, pull_ifs, pull_sea_ice, sea_ice_supplement
from aurora_finetuning.model import save_forecast
from aurora_finetuning.regional import regional_context
from aurora_finetuning.telemetry import Recorder, write_json

LOGGER = logging.getLogger(__name__)


def recent_cycles(now: datetime, max_age: int) -> list[datetime]:
    """List UTC six-hour cycles within the age limit, newest first."""
    if now.tzinfo is None:
        raise ValueError("Current time must include a timezone")
    utc = now.astimezone(UTC).replace(tzinfo=None)
    cycle = utc.replace(hour=utc.hour // 6 * 6, minute=0, second=0, microsecond=0)
    result = []
    while (utc - cycle).total_seconds() <= max_age * 3600:
        result.append(cycle)
        cycle -= timedelta(hours=6)
    return result


def select_cycle(config: Config, now: datetime, recorder: Recorder) -> datetime:
    """Select the newest available pair; only sea ice may be supplemented."""
    failures: list[str] = []
    for cycle in recent_cycles(now, config.max_input_age_hours):
        try:
            audit = audit_ifs(cycle, recorder)
        except requests.HTTPError as error:
            if error.response is None or error.response.status_code != 404:
                raise
            LOGGER.info(
                "IFS cycle %s UTC is not published; trying an older pair", cycle
            )
            failures.append(f"{cycle.isoformat()}: not published")
            continue
        reports = audit["cycles"]
        if (
            isinstance(reports, list)
            and len(reports) == 2
            and all(
                isinstance(r, dict)
                and r.get("missing_surface") in ([], ["siconc"])
                and r.get("missing_pressure_fields") == []
                for r in reports
            )
        ):
            return cycle
        failures.append(f"{cycle.isoformat()}: incomplete fields {reports}")
    raise ValueError("No complete fresh IFS input pair. " + "; ".join(failures))


def available_cycle(config: Config, now: datetime, recorder: Recorder) -> datetime:
    """Audit freshness before forecast reuse; cache selection for ten minutes."""
    candidates = recent_cycles(now, config.max_input_age_hours)
    bucket = int(now.timestamp()) // 600
    path = (
        config.root
        / "outputs/availability"
        / (f"ifs-{bucket}-{config.max_input_age_hours}.json")
    )
    if path.exists():
        cycle = datetime.fromisoformat(json.loads(path.read_text())["cycle"])
        if cycle in candidates:
            return cycle
    cycle = select_cycle(config, now, recorder)
    if not path.exists():
        write_json(path, {"cycle": cycle.isoformat(), "checked_utc": now.isoformat()})
    return cycle


def find_sea_ice(config: Config, cycle: datetime, recorder: Recorder) -> Path:
    """Find a checksum-verified supplement valid for both atmospheric states."""
    for path in sorted((config.root / "data/raw/sea-ice").glob("*.nc"), reverse=True):
        manifest = path.with_suffix(".json")
        if not manifest.exists():
            continue
        expected = json.loads(manifest.read_text())["sha256"]
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Modified sea-ice supplement: {path}")
        try:
            for time in (cycle - timedelta(hours=6), cycle):
                sea_ice_supplement(path, time, config)
        except ValueError:
            continue
        recorder.inputs[str(path)] = actual
        return path
    raise ValueError(
        "No verified sea-ice supplement valid for both input times; "
        "run pixi run weather-setup (CDS credentials and dataset terms required)"
    )


def pipeline_identity() -> str:
    """Invalidate numerical caches when forecast or regional implementation changes."""
    from importlib.metadata import distribution

    upstream = distribution("microsoft-aurora")
    dependency_identity = upstream.version + (
        upstream.read_text("direct_url.json") or ""
    )
    directory = Path(__file__).parent
    return hashlib.sha256(
        (
            dependency_identity
            + "".join(
                sha256(directory / name)
                for name in (
                    "model.py",
                    "leads.py",
                    "engine.py",
                    "regional.py",
                    "preparation.py",
                    "grid.py",
                    "ifs.py",
                )
            )
        ).encode()
    ).hexdigest()


def cached_grid(
    config: Config,
    cycle: datetime,
    horizon: int,
    recorder: Recorder,
    requested_leads: list[int] | None = None,
) -> xr.Dataset | None:
    """Validate a cached forecast before constructing tensors or touching weights."""
    try:
        key = cache_identity(config, cycle, None)
    except FileNotFoundError:
        return None
    directory = config.root / "outputs/weather" / key
    wanted = set(requested_leads or range(1, horizon + 1))
    for path in sorted(directory.glob("*.nc")):
        manifest = path.with_suffix(".json")
        if not manifest.exists():
            raise ValueError(f"Incomplete forecast cache: {path}")
        info = json.loads(manifest.read_text())
        coverage = info.get("lead_hours")
        if not isinstance(coverage, list) or info.get("schema") != 3:
            raise ValueError(f"Incompatible forecast cache coverage: {path}")
        if not wanted.issubset(coverage):
            continue
        expected = info["sha256"]
        if sha256(path) != expected:
            raise ValueError(f"Modified forecast cache: {path}")
        with recorder.measure("forecast.read_cache"), xr.open_dataset(path) as source:
            result = source.load()
        actual = [
            int(
                (
                    datetime.fromisoformat(str(t.astype("datetime64[us]"))) - cycle
                ).total_seconds()
                / 3600
            )
            for t in result.time.values
        ]
        if actual != coverage or len(actual) != len(set(actual)):
            raise ValueError(f"Forecast cache lead coverage mismatch: {path}")
        if any(
            np.datetime64(cycle + timedelta(hours=lead), "ns") != stamp
            for lead, stamp in zip(actual, result.time.values, strict=True)
        ):
            raise ValueError(f"Forecast cache has non-hourly valid times: {path}")
        recorder.inputs[str(path)] = expected
        return result
    return None


def cache_identity(config: Config, cycle: datetime, adapter: Path | None) -> str:
    """Identify model, region, numerical settings and immutable input manifests."""
    manifests = [
        config.root / "data/raw/ifs" / f"{time.date()}-{kind}-{time.hour:02d}.json"
        for time in (cycle - timedelta(hours=6), cycle)
        for kind in ("single", "pressure")
    ]
    identity = {
        "version": 3,
        "output_schema": 3,
        "code": pipeline_identity(),
        "precision": config.precision,
        "initialization": cycle.isoformat(),
        "model": {
            "revision": config.checkpoint_revision,
            "seed": config.seed,
            "device": config.device,
        },
        "grid": make_grid(config).describe(),
        "adapter": sha256(adapter) if adapter is not None else None,
        "inputs": [sha256(path) for path in manifests],
        "assets": sha256(config.root / "checkpoints/base/assets.json"),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def predict_weather(
    config: Config,
    latitude: float,
    longitude: float,
    recorder: Recorder,
    *,
    start: datetime | None = None,
    now: datetime | None = None,
    adapter: Path | None = None,
    radar_adapter: Path | None = None,
    station_calibrator: Path | None = None,
    engine: ForecastEngine | None = None,
    rain_method: str | None = None,
    rain_calibration: Path | None = None,
) -> dict[str, object]:
    """Retrieve fresh inputs, reuse or generate a regional forecast, then query it."""
    validate_location(latitude, longitude)
    if adapter is not None:
        raise ValueError(
            "Legacy surface-head adapters are unsupported; use external calibrators"
        )
    method = rain_method or ("cnn" if radar_adapter is not None else "aurora")
    if method not in ("aurora", "cnn", "radar-motion", "radar-blend"):
        raise ValueError("Unknown rainfall method")
    if (method == "cnn") != (radar_adapter is not None):
        raise ValueError(
            "CNN rainfall requires --radar-adapter; other methods reject it"
        )
    if (method == "radar-blend") != (rain_calibration is not None):
        raise ValueError("Only radar-blend requires --rain-calibration")
    if rain_calibration is not None:
        from aurora_finetuning.rain_calibration import load_weights

        load_weights(rain_calibration, config)
    now = now or datetime.now(UTC)
    start = start or now
    if start.tzinfo is None or now.tzinfo is None:
        raise ValueError("--start must include a timezone")
    if start.astimezone(UTC) < now.astimezone(UTC):
        raise ValueError(
            "Live --start must be now or in the future; use forecast for history"
        )
    if start.astimezone(UTC) - now.astimezone(UTC) > timedelta(hours=47):
        raise ValueError("Requested window exceeds the 48-hour internal rollout limit")
    first = start.astimezone(UTC).replace(
        minute=0, second=0, microsecond=0
    ) + timedelta(hours=1)
    end = (first + timedelta(hours=config.hours - 1)).replace(tzinfo=None)
    cycle = available_cycle(config, now, recorder)
    horizon = int((end - cycle).total_seconds() / 3600)
    if not 1 <= horizon <= 48:
        raise ValueError("Requested window exceeds the 48-hour internal rollout limit")
    requested = (
        list(range(horizon - config.hours + 1, horizon + 1))
        if config.sparse_forecasts
        else None
    )
    dataset = cached_grid(config, cycle, horizon, recorder, requested)
    if dataset is None:
        ice = find_sea_ice(config, cycle, recorder)
        pull_ifs(config, cycle, recorder, ice)
        dataset = cached_grid(config, cycle, horizon, recorder, requested)
        if dataset is None:
            if engine is None:
                with ForecastEngine(config, recorder) as session:
                    dataset = (
                        session.predict(
                            cycle, horizon, "ifs", requested_leads=requested
                        )
                        if requested
                        else session.predict(cycle, horizon, "ifs")
                    )
            else:
                if engine.config != config:
                    raise ValueError("Forecast engine configuration mismatch")
                dataset = (
                    engine.predict(cycle, horizon, "ifs", requested_leads=requested)
                    if requested
                    else engine.predict(cycle, horizon, "ifs")
                )
            query_grid(dataset, latitude, longitude, config.hours, start)
            key = cache_identity(config, cycle, None)
            suffix = f"-{requested[0]:02d}" if requested else ""
            path = config.root / "outputs/weather" / key / f"{horizon:02d}{suffix}.nc"
            dataset.attrs["available_utc"] = datetime.now(UTC).isoformat()
            save_forecast(dataset, path)
            write_json(
                path.with_suffix(".json"),
                {
                    "sha256": sha256(path),
                    "key": key,
                    "schema": 3,
                    "lead_hours": requested or list(range(1, horizon + 1)),
                },
            )
    result = query_grid(dataset, latitude, longitude, config.hours, start)
    if station_calibrator is not None:
        from aurora_finetuning.station_data import apply_live_stations

        result = apply_live_stations(result, station_calibrator, config, recorder, now)
    if radar_adapter is not None:
        from aurora_finetuning.radar_model import apply_live_radar

        result = apply_live_radar(
            result,
            dataset,
            radar_adapter,
            config,
            recorder,
            now,
            latitude,
            longitude,
            adapter,
        )
    if method in ("radar-motion", "radar-blend"):
        from aurora_finetuning.radar_nowcast import apply_motion

        result = apply_motion(result, config, recorder, now, rain_calibration)
    result["rain_method"] = method
    result["initialization_age_hours"] = (
        now.astimezone(UTC).replace(tzinfo=None) - cycle
    ).total_seconds() / 3600
    rows = result["hourly"]
    if not isinstance(rows, list):
        raise ValueError("Invalid forecast rows")
    leads = [
        (
            datetime.fromisoformat(row["time"]).astimezone(UTC).replace(tzinfo=None)
            - cycle
        ).total_seconds()
        / 3600
        for row in rows
    ]
    result["model_lead_hours"] = leads
    result["requested_components"] = {
        "aurora": True,
        "stations": station_calibrator is not None,
        "radar": method != "aurora",
    }
    result["variable_sources"] = {
        name: sorted({str(row.get(f"{name}_source", "unknown")) for row in rows})
        for name in (
            "temperature_c",
            "pressure_hpa",
            "wind_u_ms",
            "wind_v_ms",
            "precipitation_mm",
        )
    }
    LOGGER.info(
        "Components requested: Aurora, stations=%s, radar=%s; model leads %s–%s h; "
        "station cutoff=%s; radar cutoff=%s; actual sources=%s",
        station_calibrator is not None,
        method != "aurora",
        leads[0],
        leads[-1],
        result.get("station_cutoff_utc", "none"),
        result.get("radar_cutoff_utc", "none"),
        result["variable_sources"],
    )
    LOGGER.info(
        "Correction identities (SHA256): stations=%s; radar=%s",
        result.get("station_calibrator_sha256", "none"),
        result.get(
            "radar_adapter_sha256", result.get("rain_calibration_sha256", "none")
        ),
    )
    LOGGER.warning(
        "Experimental Aurora forecast at ~25 km resolution; initialized %s UTC; "
        "input age %.1f hours",
        cycle,
        result["initialization_age_hours"],
    )
    return result


def setup_weather(
    config: Config, recorder: Recorder, sea_ice_date: date | None = None
) -> dict[str, object]:
    """Install assets and validate fresh live inputs without claiming forecast skill."""
    download_assets(config, recorder)
    now = datetime.now(UTC)
    cycle = select_cycle(config, now, recorder)
    try:
        ice = find_sea_ice(config, cycle, recorder)
    except ValueError:
        # ERA5 is delayed; choose a conservative date within the seven-day limit.
        day = sea_ice_date or (now - timedelta(days=5)).date()
        pull_sea_ice(config, day, recorder)
        ice = find_sea_ice(config, cycle, recorder)
    pull_ifs(config, cycle, recorder, ice)
    from aurora_finetuning.preparation import prepare_batch

    with regional_context():
        prepare_batch(config, cycle, recorder, "ifs")
    return {
        "inputs_validated": True,
        "live_forecast_verified": False,
        "initialization_utc": cycle.isoformat(),
        "next": "pixi run weather Utrecht --hours 1",
    }
