"""Bounded live-input discovery and reusable regional forecast caching."""

import hashlib
import json
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import requests
import xarray as xr

from aurora_finetuning.acquisition import sha256
from aurora_finetuning.assets import download_assets, verify_assets
from aurora_finetuning.config import Config
from aurora_finetuning.forecast import query_grid, validate_location
from aurora_finetuning.grid import make_grid
from aurora_finetuning.ifs import audit_ifs, pull_ifs, pull_sea_ice, sea_ice_supplement
from aurora_finetuning.model import infer, load_model, save_forecast
from aurora_finetuning.preparation import prepare_batch
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


def cache_identity(config: Config, cycle: datetime, adapter: Path | None) -> str:
    """Identify model, region, numerical settings and immutable input manifests."""
    manifests = [
        config.root / "data/raw/ifs" / f"{time.date()}-{kind}-{time.hour:02d}.json"
        for time in (cycle - timedelta(hours=6), cycle)
        for kind in ("single", "pressure")
    ]
    identity = {
        "version": 1,
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
) -> dict[str, object]:
    """Retrieve fresh inputs, reuse or generate a regional forecast, then query it."""
    validate_location(latitude, longitude)
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
    cycle = select_cycle(config, now, recorder)
    first = start.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    first += timedelta(hours=1)
    horizon = int(
        (
            (first + timedelta(hours=config.hours - 1)).replace(tzinfo=None) - cycle
        ).total_seconds()
        / 3600
    )
    if not 1 <= horizon <= 48:
        raise ValueError("Requested window exceeds the 48-hour internal rollout limit")
    verify_assets(config)
    ice = find_sea_ice(config, cycle, recorder)
    pull_ifs(config, cycle, recorder, ice)
    # Preparing even on a forecast cache hit verifies checksums, units and fields.
    with regional_context():
        batch = prepare_batch(config, cycle, recorder, "ifs")
        key = cache_identity(config, cycle, adapter)
        directory = config.root / "outputs/weather" / key
        dataset: xr.Dataset | None = None
        for path in sorted(directory.glob("*.nc")):
            if int(path.stem) < horizon:
                continue
            expected = json.loads(path.with_suffix(".json").read_text())["sha256"]
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f"Modified forecast cache: {path}")
            with xr.open_dataset(path) as cached:
                dataset = cached.load()
            recorder.inputs[str(path)] = actual
            break
        if dataset is None:
            model = load_model(config, recorder, adapter)
            dataset = infer(model, batch, horizon, config, recorder)
            # Validate the whole requested table before publishing a cache artifact.
            query_grid(dataset, latitude, longitude, config.hours, start)
            path = directory / f"{horizon:02d}.nc"
            save_forecast(dataset, path)
            write_json(path.with_suffix(".json"), {"sha256": sha256(path), "key": key})
    result = query_grid(dataset, latitude, longitude, config.hours, start)
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
    result["initialization_age_hours"] = (
        now.astimezone(UTC).replace(tzinfo=None) - cycle
    ).total_seconds() / 3600
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
    with regional_context():
        prepare_batch(config, cycle, recorder, "ifs")
    return {
        "inputs_validated": True,
        "live_forecast_verified": False,
        "initialization_utc": cycle.isoformat(),
        "next": "pixi run weather Utrecht --hours 1",
    }
