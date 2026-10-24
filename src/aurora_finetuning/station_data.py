"""Bounded station acquisition and observed-history application to local forecasts."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from aurora_finetuning.acquisition import knmi_files, sha256
from aurora_finetuning.config import Config
from aurora_finetuning.observations import load_stations, station_geometry
from aurora_finetuning.stations import apply_station
from aurora_finetuning.telemetry import Recorder


def station_history(config: Config, recorder: Recorder, now: datetime) -> xr.Dataset:
    """Reuse nationwide hourly context plus the latest hour of 10-minute data.

    Hourly operational files are published overnight; gaps in today's context are
    explicit rather than filled with delayed validated observations.
    """
    utc = now.astimezone(UTC).replace(tzinfo=None)
    begin = utc - timedelta(hours=73)
    hourly = knmi_files(
        config,
        "hourly-in-situ-meteorological-observations",
        "1.0",
        f"hourly-observations-{begin:%Y%m%d-%H}.nc",
        f"hourly-observations-{utc:%Y%m%d-%H}.nc",
        recorder,
        74,
    )
    recent_start = utc - timedelta(minutes=70)
    recent = knmi_files(
        config,
        "10-minute-in-situ-meteorological-observations",
        "1.0",
        f"KMDS__OPER_P___10M_OBS_L2_{recent_start:%Y%m%d%H%M}.nc",
        f"KMDS__OPER_P___10M_OBS_L2_{utc:%Y%m%d%H%M}.nc",
        recorder,
        8,
    )
    groups = []
    for paths in (hourly, recent):
        ds = load_stations(paths)
        available = []
        for path in paths:
            manifest = path.with_name(path.name + ".json")
            stamp = json.loads(manifest.read_text()).get("acquired_utc")
            if stamp is None:
                raise ValueError(
                    "Station cache lacks acquisition time; reacquire explicitly"
                )
            acquired = (
                datetime.fromisoformat(stamp).astimezone(UTC).replace(tzinfo=None)
            )
            with xr.open_dataset(path) as raw:
                available.extend([(t, acquired) for t in raw.time.values])
            recorder.inputs[str(path)] = sha256(path)
        mapping = dict(available)
        ds = ds.assign_coords(
            available_time=(
                "time",
                [np.datetime64(mapping[t], "ns") for t in ds.time.values],
            )
        )
        groups.append(ds)
    merged = xr.concat(
        groups, dim="time", join="outer", coords="different", compat="equals"
    ).sortby("time")
    # Near-real-time records win at coincident hourly timestamps.
    _, reverse = np.unique(merged.time.values[::-1], return_index=True)
    positions = np.sort(len(merged.time) - 1 - reverse)
    return station_geometry(
        merged.isel(time=positions).sel(
            time=slice(np.datetime64(utc - timedelta(hours=72)), np.datetime64(utc))
        )
    )


def apply_live_stations(
    result: dict[str, object],
    path: Path,
    config: Config,
    recorder: Recorder,
    now: datetime,
) -> dict[str, object]:
    """Use only observations acquired by a fixed issuance cutoff."""
    with recorder.measure("stations.acquire_history"):
        history = station_history(config, recorder, now)
    archives = []
    # Only verified forecasts carrying an explicit availability time can supply
    # prospective residuals. Legacy initialization-only files are not sufficient.
    for forecast in sorted((config.root / "outputs/weather").glob("*/*.nc")):
        with xr.open_dataset(forecast) as ds:
            if "available_utc" not in ds.attrs:
                continue
            manifest = forecast.with_suffix(".json")
            if (
                not manifest.exists()
                or sha256(forecast) != json.loads(manifest.read_text())["sha256"]
            ):
                raise ValueError(f"Invalid station residual forecast: {forecast}")
            earliest = np.datetime64(
                now.astimezone(UTC).replace(tzinfo=None) - timedelta(hours=72)
            )
            if ds.time.max().values < earliest:
                continue
            archives.append(ds.load())
    # Issuance follows acquisition; observation timestamps remain bounded by the
    # original request cutoff, while newly received records become usable now.
    issuance = datetime.now(UTC)
    with recorder.measure("stations.adjust"):
        adjusted = apply_station(result, history, archives, path, config, issuance)
    adjusted["station_history_requested_through_utc"] = now.isoformat()
    adjusted["issued_utc"] = issuance.isoformat()
    return adjusted
