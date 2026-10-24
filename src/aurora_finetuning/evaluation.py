"""Independent observation scoring and simple forecast baselines."""

import numpy as np
import xarray as xr

from aurora_finetuning.grid import FloatArray


def metrics(
    prediction: FloatArray, observation: FloatArray, threshold: float = 0.1
) -> dict[str, float | int | None]:
    """Score finite matched pairs and report wet-event counts at a fixed threshold."""
    if prediction.shape != observation.shape:
        raise ValueError("Prediction and observation shapes must match")
    mask = np.isfinite(prediction) & np.isfinite(observation)
    p, o = prediction[mask], observation[mask]
    if not p.size:
        return {"count": 0, "mae": None, "rmse": None, "bias": None}
    error = p - o
    return {
        "count": int(p.size),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
        "hits": int(((p >= threshold) & (o >= threshold)).sum()),
        "misses": int(((p < threshold) & (o >= threshold)).sum()),
        "false_alarms": int(((p >= threshold) & (o < threshold)).sum()),
    }


def evaluate_stations(
    prediction: xr.Dataset, stations: xr.Dataset
) -> dict[str, object]:
    """Score in-domain stations by lead hour, with a persistence baseline."""
    init = np.datetime64(str(prediction.attrs["initialization_utc"]), "ns")
    selection = (
        (stations.latitude >= prediction.latitude.min())
        & (stations.latitude <= prediction.latitude.max())
        & (stations.longitude >= prediction.longitude.min())
        & (stations.longitude <= prediction.longitude.max())
    )
    stations = stations.isel(station=np.flatnonzero(selection.values))
    if stations.sizes["station"] == 0:
        raise ValueError("No stations lie inside the forecast domain")
    times = np.intersect1d(prediction.time.values, stations.time.values)
    if not times.size:
        raise ValueError("No matching forecast/observation times")
    p = prediction.sel(time=times)
    o = stations.sel(time=times)
    temp = p.temperature_c.interp(
        latitude=stations.latitude, longitude=stations.longitude
    )
    rain = p.precipitation_mm.sel(
        latitude=stations.latitude, longitude=stations.longitude, method="nearest"
    )
    output: list[dict[str, object]] = []
    for index, time in enumerate(times):
        row: dict[str, object] = {
            "lead_hour": int((time - init) / np.timedelta64(1, "h")),
            "temperature": metrics(
                np.asarray(temp[index].values),
                np.asarray(o.temperature_c[index].values),
            ),
            "precipitation": metrics(
                np.asarray(rain[index].values),
                np.asarray(o.precipitation_mm[index].values),
            ),
            "dry_baseline": metrics(
                np.zeros(o.sizes["station"]),
                np.asarray(o.precipitation_mm[index].values),
            ),
        }
        if init in stations.time.values:
            row["temperature_persistence"] = metrics(
                np.asarray(stations.temperature_c.sel(time=init).values),
                np.asarray(o.temperature_c[index].values),
            )
        for variable in ("pressure_hpa", "wind_u_ms", "wind_v_ms"):
            if variable not in p or variable not in o:
                continue
            sampled = p[variable].interp(
                latitude=stations.latitude, longitude=stations.longitude
            )
            row[variable] = metrics(
                np.asarray(sampled[index].values), np.asarray(o[variable][index].values)
            )
            if init in stations.time.values and variable in stations:
                row[f"{variable}_persistence"] = metrics(
                    np.asarray(stations[variable].sel(time=init).values),
                    np.asarray(o[variable][index].values),
                )
        output.append(row)
    return {
        "by_lead_hour": output,
        "matched_stations": stations.sizes["station"],
        "temperature_reference_height_m": 1.5,
        "model_temperature_height_m": 2,
        "precipitation_wet_threshold_mm": 0.1,
    }


def evaluate_radar(prediction: xr.Dataset, radar: xr.Dataset) -> dict[str, object]:
    """Score an hourly radar dataset already aggregated onto the model grid."""
    p, o = xr.align(prediction.precipitation_mm, radar.precipitation_mm, join="inner")
    if any(size == 0 for size in p.shape):
        raise ValueError("No matching model/radar grid and timestamps")
    return {
        "hourly": metrics(np.asarray(p.values), np.asarray(o.values)),
        "accumulation": metrics(
            np.asarray(p.sum("time", skipna=False).values),
            np.asarray(o.sum("time", skipna=False).values),
        ),
    }
