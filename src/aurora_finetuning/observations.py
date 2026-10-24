"""KNMI observation decoding, hourly radar aggregation and local summaries."""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import xarray as xr

from aurora_finetuning.grid import FloatArray, Grid

STATION_VARIABLES = {
    "T": "temperature_c",
    "RH": "precipitation_mm",
    "SQ": "sunshine_hours",
    "DR": "precipitation_hours",
    "P": "pressure_hpa",
}


def quality_mask(source: xr.Dataset, name: str, data: xr.DataArray) -> xr.DataArray:
    """Honor declared quality flags; reject undocumented ancillary flag semantics."""
    for flag in str(source[name].attrs.get("ancillary_variables", "")).split():
        if flag not in source:
            raise ValueError(f"Missing station quality flag {flag}")
        quality = source[flag]
        meanings = str(quality.attrs.get("flag_meanings", "")).split()
        codes = np.asarray(quality.attrs.get("flag_values", []))
        if len(meanings) != len(codes) or not meanings:
            raise ValueError(f"Undocumented station quality flag {flag}")
        accepted = [
            code
            for code, meaning in zip(codes, meanings, strict=True)
            if meaning.lower()
            in ("good", "valid", "passed", "good_data", "no_qc_performed")
        ]
        if not accepted:
            raise ValueError(f"No supported good-data flag for {flag}")
        data = data.where(quality.isin(accepted))
    return data


def station_geometry(dataset: xr.Dataset) -> xr.Dataset:
    """Use each station's last finite coordinates within the selected time window."""
    for name in ("latitude", "longitude"):
        if "time" not in dataset[name].dims:
            continue
        values = np.asarray(
            dataset[name].transpose("time", "station").values, dtype=float
        )
        valid = np.isfinite(values)
        if not len(values):
            continue
        positions = len(values) - 1 - np.argmax(valid[::-1], axis=0)
        latest = values[positions, np.arange(values.shape[1])]
        latest[~valid.any(axis=0)] = np.nan
        dataset = dataset.assign_coords({name: ("station", latest)})
    return dataset


def normalize_stations(source: xr.Dataset) -> xr.Dataset:
    """Decode KDP hourly NetCDF values, already in physical units (not tenths).

    This adapter accepts documented uppercase KDP parameter names and standard
    time/station/latitude/longitude coordinates. Unknown layouts fail explicitly.
    """
    aliases = {"ta": "T", "pp": "P", "ff": "FF", "dd": "DD"}
    source = source.rename(
        {
            old: new
            for old, new in aliases.items()
            if old in source and new not in source
        }
    )
    source = source.rename(
        {
            name: replacement
            for name, replacement in (
                ("lat", "latitude"),
                ("lon", "longitude"),
                ("wsi", "station"),
                ("station_id", "station"),
            )
            if name in source and replacement not in source
        }
    )
    if "wsi" in source and source.wsi.dims == ("station",):
        source = source.assign_coords(station=source.wsi.values.astype(str)).drop_vars(
            "wsi"
        )
    if (
        "station" in source.coords
        and len(np.unique(source.station.values)) != source.sizes["station"]
    ):
        raise ValueError("Duplicate KNMI station identifiers")
    required = {"time", "station", "latitude", "longitude"}
    if not required.issubset(source.variables):
        raise ValueError(
            "Unsupported station schema; missing "
            f"{sorted(required - set(source.variables))}"
        )
    fields: dict[str, xr.DataArray] = {}
    for name, target in STATION_VARIABLES.items():
        if name not in source:
            continue
        data = source[name].transpose("time", "station").astype(float)
        unit = str(data.attrs.get("units", ""))
        supported = {
            "T": {"degrees_Celsius", "degree_Celsius", "degC", "°C", "Celsius"},
            "RH": {"mm"},
            "SQ": {"h", "hours", "hour"},
            "DR": {"h", "hours", "hour"},
            "P": {"hPa"},
        }
        if unit not in supported[name]:
            raise ValueError(f"Unsupported KNMI units for {name}: {unit!r}")
        if (
            name == "P"
            and data.attrs.get("standard_name") != "air_pressure_at_mean_sea_level"
        ):
            raise ValueError(
                "Station pressure must be explicitly mean-sea-level pressure"
            )
        data = quality_mask(source, name, data)
        if name == "RH" and "-1 for <0.05" in str(data.attrs.get("description", "")):
            # This is a trace amount, not missingness. Use its lower bound for
            # amount scoring; the 0.1 mm wet-event threshold is unaffected.
            data = data.where(data != -1, 0)
            data.attrs["trace_handling"] = "KNMI -1 trace (<0.05 mm) mapped to 0 mm"
        if name != "T":
            data = data.where(data >= 0)
        fields[target] = data
    if "FF" in source and "DD" in source:
        speed = quality_mask(
            source, "FF", source.FF.transpose("time", "station").astype(float)
        )
        direction = quality_mask(
            source, "DD", source.DD.transpose("time", "station").astype(float)
        )
        for name in ("FF", "DD"):
            description = str(source[name].attrs.get("description", "")).lower()
            live_names = {
                "FF": "Wind Speed at 10 m Mean with MD",
                "DD": "Wind Direction Mean with MD",
            }
            documented_live = source[name].attrs.get("long_name") == live_names[name]
            if not documented_live and (
                "10 minute" not in description
                or (name == "FF" and "10 meters" not in description)
            ):
                raise ValueError(
                    "Station wind must represent past 10 minute wind at 10 meters"
                )
        if speed.attrs.get("units") not in ("m/s", "m s-1") or direction.attrs.get(
            "units"
        ) not in ("°", "degree", "degrees"):
            raise ValueError("Unsupported station wind units")
        angle = np.deg2rad(direction.where((direction > 0) & (direction <= 360)))
        fields["wind_u_ms"] = (
            (-speed * np.sin(angle)).where(speed != 0, 0).where(speed >= 0)
        )
        fields["wind_v_ms"] = (
            (-speed * np.cos(angle)).where(speed != 0, 0).where(speed >= 0)
        )
    if not fields:
        raise ValueError("No supported KNMI station variables")
    return xr.Dataset(
        fields,
        coords={
            "time": source.time,
            "station": source.station,
            "latitude": source.latitude,
            "longitude": source.longitude,
            **(
                {"available_time": source.available_time}
                if "available_time" in source
                else {}
            ),
        },
    )


def load_stations(paths: list[Path]) -> xr.Dataset:
    """Read downloaded hourly station files and reject duplicate observations."""
    if not paths:
        raise ValueError("No KNMI station files supplied")
    frames: list[xr.Dataset] = []
    for path in paths:
        with xr.open_dataset(path) as ds:
            frames.append(normalize_stations(ds).load())
    result = xr.concat(
        frames, dim="time", join="outer", coords="different", compat="equals"
    ).sortby("time")
    if len(np.unique(result.time.values)) != result.sizes["time"]:
        raise ValueError("Duplicate KNMI observation timestamps")
    return result


def aggregate_radar(
    values: FloatArray,
    latitude: FloatArray,
    longitude: FloatArray,
    grid: Grid,
    resolution: float,
) -> FloatArray:
    """Area-weight equal-area radar pixels into model cells; preserve missing cells."""
    if values.shape != latitude.shape or values.shape != longitude.shape:
        raise ValueError("Radar values and geolocation must have the same shape")
    if np.any(np.isfinite(values) & (values < 0)):
        raise ValueError("Radar amounts must be nonnegative or missing")
    row = np.floor((grid.latitude[0] + resolution / 2 - latitude) / resolution).astype(
        int
    )
    col = np.floor(
        (longitude - grid.longitude[0] + resolution / 2) / resolution
    ).astype(int)
    h, w = grid.shape
    valid = (
        np.isfinite(values)
        & np.isfinite(latitude)
        & np.isfinite(longitude)
        & (row >= 0)
        & (row < h)
        & (col >= 0)
        & (col < w)
    )
    total = np.zeros((h, w), dtype=float)
    count = np.zeros((h, w), dtype=float)
    np.add.at(total, (row[valid], col[valid]), values[valid])
    np.add.at(count, (row[valid], col[valid]), 1)
    result: FloatArray = np.full((h, w), np.nan)
    np.divide(total, count, out=result, where=count > 0)
    return result


def hourly_radar(frames: xr.DataArray) -> xr.DataArray:
    """Sum twelve consecutive five-minute amounts, never interpreting gaps as dry."""
    times = frames.time.values.astype("datetime64[m]")
    if len(np.unique(times)) != len(times):
        raise ValueError("Duplicate radar timestamps")
    if len(times) > 1 and not np.all(np.diff(times) == np.timedelta64(5, "m")):
        raise ValueError("Radar frames must be consecutive at five-minute intervals")
    if np.any(times.astype(np.int64) % 5):
        raise ValueError("Radar frames must lie on five-minute boundaries")
    result = frames.resample(time="1h", closed="right", label="right").sum(skipna=False)
    counts = frames.resample(time="1h", closed="right", label="right").count()
    return result.where(counts == 12)


def summarize_history(
    observations: xr.Dataset, station: str, today: date
) -> list[dict[str, object]]:
    """Summarize the preceding three local days with explicit hourly coverage.

    Min/max/mean are sampled hourly temperatures, not instrument daily extrema.
    Interval totals use the local day containing the interval start.
    """
    ds = observations.sel(station=station)
    zone = ZoneInfo("Europe/Amsterdam")
    result: list[dict[str, object]] = []
    for offset in (3, 2, 1):
        day = today - timedelta(days=offset)
        start = datetime.combine(day, datetime.min.time(), zone).astimezone(UTC)
        end = datetime.combine(
            day + timedelta(days=1), datetime.min.time(), zone
        ).astimezone(UTC)
        first = np.datetime64((start + timedelta(hours=1)).replace(tzinfo=None), "ns")
        last = np.datetime64(end.replace(tzinfo=None), "ns")
        selected = ds.sel(time=slice(first, last))
        expected = int((end - start).total_seconds() / 3600)
        row: dict[str, object] = {
            "date": day.isoformat(),
            "expected_hours": expected,
            "station": station,
            "temperature_basis": "hourly samples",
        }
        # Temperature is instantaneous; assign samples by their own local date.
        # Accumulation intervals instead belong to the day of their start.
        temperature_start = np.datetime64(start.replace(tzinfo=None), "ns")
        temperature_end = np.datetime64(
            end.replace(tzinfo=None), "ns"
        ) - np.timedelta64(1, "ns")
        temp = ds.temperature_c.sel(
            time=slice(temperature_start, temperature_end)
        ).values
        valid = temp[np.isfinite(temp)]
        row.update(
            temperature_min_c=float(valid.min()) if valid.size else None,
            temperature_max_c=float(valid.max()) if valid.size else None,
            temperature_mean_c=float(valid.mean()) if valid.size else None,
            temperature_observed_hours=int(valid.size),
        )
        for variable in ("sunshine_hours", "precipitation_hours", "precipitation_mm"):
            values = selected[variable].values if variable in selected else np.array([])
            count = int(np.isfinite(values).sum())
            row[variable] = float(values.sum()) if count == expected else None
            row[f"{variable}_observed_hours"] = count
        result.append(row)
    return result
