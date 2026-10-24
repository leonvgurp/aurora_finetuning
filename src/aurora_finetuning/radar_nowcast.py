"""Deterministic native-grid radar motion, streamed rainfall totals and live reuse."""

import hashlib
import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from requests import RequestException

from aurora_finetuning.acquisition import sha256
from aurora_finetuning.config import Config
from aurora_finetuning.grid import FloatArray
from aurora_finetuning.radar_data import RadarUnavailable, history_at, live_history
from aurora_finetuning.telemetry import Recorder, write_json

VERSION = "radar-motion-v1"


def estimate_motion(history: FloatArray) -> FloatArray:
    """Estimate pixels per five-minute interval from masked log rain rates."""
    from pysteps.motion.lucaskanade import dense_lucaskanade

    if history.ndim != 3 or history.shape[0] != 24:
        raise ValueError("Motion needs exactly 24 measured five-minute frames")
    if np.isinf(history).any() or np.any(history[np.isfinite(history)] < 0):
        raise ValueError("Invalid measured rainfall")
    # NumPy does not type this constructor; keep its boundary signature explicit.
    masked: Callable[[FloatArray, NDArray[np.bool_]], object] = np.ma.MaskedArray
    images = masked(10 * np.log10(np.maximum(history * 12, 0.1)), ~np.isfinite(history))
    if not np.isfinite(history).any():
        raise RadarUnavailable("Radar history has no valid observations")
    if not np.isfinite(history[-1]).any():
        raise RadarUnavailable("Latest measured radar frame has no valid pixels")
    try:
        velocity: FloatArray = np.asarray(
            dense_lucaskanade(images, verbose=False), dtype=float
        )
    except (ValueError, RuntimeError) as error:
        raise RadarUnavailable(
            "Optical flow could not estimate valid motion"
        ) from error
    if velocity.shape != (2, *history.shape[1:]) or not np.isfinite(velocity).all():
        raise RadarUnavailable("Radar motion is unavailable or nonfinite")
    return velocity


def extrapolated_bins(latest: FloatArray, velocity: FloatArray) -> Iterator[FloatArray]:
    """Advect the original measured field, carrying only its displacement forward."""
    from pysteps.extrapolation.semilagrangian import extrapolate

    displacement = None
    for _ in range(96):
        values, displacement = extrapolate(
            latest,
            velocity,
            1,
            displacement_prev=displacement,
            return_displacement=True,
            allow_nonfinite_values=True,
            outval=np.nan,
            interp_order=1,
        )
        yield np.asarray(values[0], dtype=float)


def motion_totals(
    dataset: xr.Dataset,
    cutoff: datetime,
    ends: list[datetime],
    velocity: FloatArray | None = None,
) -> xr.Dataset:
    """Accumulate known motion bins and measured prefixes, retaining support counts."""
    history = history_at(dataset, cutoff)
    raw = np.asarray(history.values, dtype=float)
    velocity = estimate_motion(raw) if velocity is None else velocity
    shape = (len(ends), *raw.shape[1:])
    totals = np.zeros(shape, dtype=np.float32)
    counts = np.zeros(shape, dtype=np.int16)
    prefix = np.zeros(shape, dtype=np.float32)
    prefix_counts = np.zeros(shape, dtype=np.int16)
    expected_prefix = np.zeros(len(ends), dtype=np.int16)
    for i, end in enumerate(ends):
        if end.tzinfo is not None or end.minute % 5 or end.second or end.microsecond:
            raise ValueError("Rainfall ends must be naive UTC five-minute boundaries")
        for j in range(12):
            valid = end - timedelta(minutes=5 * (11 - j))
            if valid <= cutoff:
                expected_prefix[i] += 1
                if np.datetime64(valid, "ns") in history.time.values:
                    field = np.asarray(
                        history.sel(time=np.datetime64(valid, "ns")).values
                    )
                    prefix[i] += np.where(np.isfinite(field), field, 0)
                    prefix_counts[i] += np.isfinite(field)
    for step, field in enumerate(extrapolated_bins(raw[-1], velocity), start=1):
        valid = cutoff + timedelta(minutes=5 * step)
        for i, end in enumerate(ends):
            if end - timedelta(hours=1) < valid <= end:
                good = np.isfinite(field)
                totals[i] += np.where(good, field, 0)
                counts[i] += good
    prefix = np.where(prefix_counts == expected_prefix[:, None, None], prefix, np.nan)
    result = dataset.isel(time=0, drop=True).drop_vars("precipitation_mm").copy()
    result = result.assign_coords(time=[np.datetime64(t, "ns") for t in ends])
    for name, values in [
        ("motion_mm", totals),
        ("valid_bins", counts),
        ("prefix_mm", prefix),
    ]:
        result[name] = (("time", "y", "x"), values)
    result["future_bins"] = ("time", 12 - expected_prefix)
    result.attrs.update(motion_version=VERSION, radar_cutoff_utc=cutoff.isoformat())
    return result


def checked_cache(path: Path) -> xr.Dataset | None:
    """Read a complete immutable NetCDF cache, rejecting corruption."""
    manifest = path.with_suffix(".json")
    if not path.exists() and not manifest.exists():
        return None
    if (
        not path.exists()
        or not manifest.exists()
        or sha256(path) != json.loads(manifest.read_text())["sha256"]
    ):
        raise ValueError(f"Incomplete or modified radar motion cache: {path}")
    with xr.open_dataset(path) as dataset:
        return dataset.load()


def cached_totals(
    history: xr.Dataset,
    ends: list[datetime],
    config: Config,
    recorder: Recorder,
) -> xr.Dataset:
    """Share motion and eight-hour grids across localities independently of Aurora."""
    from importlib.metadata import version

    from aurora_finetuning.model import save_forecast

    digest = hashlib.sha256(
        (
            VERSION
            + sha256(Path(__file__))
            + version("pysteps")
            + str(history.attrs.get("input_sha256"))
        ).encode()
    )
    digest.update(history.time.values.tobytes())
    digest.update(history.precipitation_mm.values.tobytes())
    digest.update(str(history.attrs.get("projection")).encode())
    for name in ("x", "y", "latitude", "longitude"):
        digest.update(history[name].values.tobytes())
    key = digest.hexdigest()
    directory = config.root / "data/cache/radar-motion"
    velocity_path = directory / f"{key}-velocity.nc"
    cached = checked_cache(velocity_path)
    if cached is None:
        with recorder.measure("radar.estimate_motion"):
            velocity = estimate_motion(
                np.asarray(history.precipitation_mm.values, dtype=float)
            )
        cached = xr.Dataset({"velocity": (("component", "y", "x"), velocity)})
        save_forecast(cached, velocity_path)
        write_json(
            velocity_path.with_suffix(".json"), {"sha256": sha256(velocity_path)}
        )
    ending_key = hashlib.sha256(
        "|".join(t.isoformat() for t in ends).encode()
    ).hexdigest()[:16]
    path = directory / f"{key}-{ending_key}.nc"
    totals = checked_cache(path)
    if totals is None:
        cutoff = datetime.fromisoformat(
            str(history.time.values[-1].astype("datetime64[us]"))
        )
        with recorder.measure("radar.extrapolate_eight_hours"):
            totals = motion_totals(
                history, cutoff, ends, np.asarray(cached.velocity.values, dtype=float)
            )
        save_forecast(totals, path)
        write_json(path.with_suffix(".json"), {"sha256": sha256(path)})
    recorder.inputs[str(path)] = sha256(path)
    return totals


def apply_motion(
    result: dict[str, object],
    config: Config,
    recorder: Recorder,
    now: datetime,
    calibration: Path | None = None,
) -> dict[str, object]:
    """Apply measured prefixes and supported radar bins; retain Aurora elsewhere."""
    from aurora_finetuning.rain_calibration import load_weights

    rows = result["hourly"]
    if not isinstance(rows, list):
        raise ValueError("Invalid forecast rows")
    weights = (
        load_weights(calibration, config) if calibration is not None else [1.0] * 8
    )
    ends = [
        datetime.fromisoformat(row["time"]).astimezone(UTC).replace(tzinfo=None)
        for row in rows
    ]
    latitude, longitude = (
        float(str(result["latitude"])),
        float(str(result["longitude"])),
    )
    try:
        history = live_history(
            config, latitude, longitude, recorder, now, full_domain=True
        )
        totals = cached_totals(history, ends, config, recorder)
    except (RadarUnavailable, RequestException) as error:
        reason = (
            str(error)
            if isinstance(error, RadarUnavailable)
            else "radar acquisition failed"
        )
        for row in rows:
            row["precipitation_mm_source"] = f"Aurora; radar unavailable: {reason}"
        result["radar_fallback"] = reason
        return result
    distance = (totals.latitude - latitude) ** 2 + (
        (totals.longitude - longitude) * np.cos(np.deg2rad(latitude))
    ) ** 2
    y, x = np.unravel_index(int(np.argmin(distance.values)), distance.shape)
    if float(np.sqrt(distance.values[y, x])) > 0.03:
        reason = "Requested locality lies outside measured radar coverage"
        for row in rows:
            row["precipitation_mm_source"] = f"Aurora; {reason}"
        result["radar_fallback"] = reason
        return result
    cutoff = datetime.fromisoformat(str(totals.attrs["radar_cutoff_utc"]))
    for i, row in enumerate(rows):
        base = float(row["precipitation_mm"])
        row["aurora_precipitation_mm"] = row.get("aurora_precipitation_mm", base)
        count = int(totals.valid_bins.values[i, y, x])
        future = int(totals.future_bins.values[i])
        prefix = float(totals.prefix_mm.values[i, y, x])
        known = float(totals.motion_mm.values[i, y, x])
        lead = max(1, int(np.ceil((ends[i] - cutoff).total_seconds() / 3600)))
        weight = weights[lead - 1] if lead <= 8 else 0.0
        row["radar_coverage_fraction"] = count / future if future else 1.0
        row["raw_radar_precipitation_mm"] = (
            prefix + known if count == future and np.isfinite(prefix) else None
        )
        if np.isfinite(prefix):
            row["precipitation_mm"] = round(
                prefix + base * future / 12 + weight * (known - base * count / 12), 3
            )
            row["precipitation_mm_source"] = (
                "Aurora + measured radar prefix + radar motion"
                if count and weight
                else "Aurora + measured prefix; zero blending weight"
                if weight == 0
                else "Aurora + measured prefix; no supported radar correction"
            )
            if count < future:
                row["precipitation_mm_source"] += (
                    "; unsupported radar bins retain Aurora"
                )
        else:
            row["precipitation_mm_source"] = (
                "Aurora; missing measured first-hour prefix"
            )
    result["radar_cutoff_utc"] = cutoff.isoformat()
    result["rain_method"] = "radar-blend" if calibration else "radar-motion"
    if calibration:
        result["rain_calibration_sha256"] = sha256(calibration)
    return result
