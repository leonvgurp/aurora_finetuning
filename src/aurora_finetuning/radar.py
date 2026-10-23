"""Decode calibrated KNMI HDF5 radar frames using an explicit geolocation grid."""

import re
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

from aurora_finetuning.grid import FloatArray, Grid
from aurora_finetuning.observations import aggregate_radar, hourly_radar


def text_attribute(value: object) -> str:
    """Decode a scalar HDF attribute without evaluating executable expressions."""
    if isinstance(value, bytes):
        return value.decode("ascii")
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray) and value.size == 1:
        return text_attribute(value.item())
    raise ValueError("Expected a scalar string attribute")


def decode_frame(path: Path) -> tuple[datetime, FloatArray]:
    """Read linear KNMI calibration and nodata; reject unfamiliar calibration forms."""
    with h5py.File(path, "r") as handle:
        calibration = handle["image1/calibration"].attrs
        formula = text_attribute(calibration["calibration_formulas"])
        number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
        expression = formula.replace(" ", "")
        match = re.fullmatch(
            rf"GEO=({number})\*PV([+-](?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)?",
            expression,
        )
        if match is None:
            raise ValueError(f"Unsupported radar calibration formula: {formula}")
        scale, offset = float(match[1]), float(match[2] or 0)
        raw = np.asarray(handle["image1/image_data"], dtype=float)
        values = raw * scale + offset
        nodata = float(np.asarray(calibration["calibration_out_of_image"]).item())
        values[raw == nodata] = np.nan
        values[values < 0] = np.nan
        timestamp = text_attribute(handle["overview"].attrs["product_datetime_end"])
        time = datetime.strptime(timestamp, "%d-%b-%Y;%H:%M:%S.000")
    return time, values


def prepare_radar(
    paths: list[Path], geolocation: Path, grid: Grid, resolution: float
) -> xr.Dataset:
    """Decode and aggregate frames with a matching authoritative radar geolocation.

    The geolocation NetCDF must contain 2D latitude/longitude arrays matching the
    KNMI image orientation. No geotransform is guessed from image dimensions.
    """
    with xr.open_dataset(geolocation) as geo:
        latitude = np.asarray(geo.latitude.values, dtype=float)
        longitude = np.asarray(geo.longitude.values, dtype=float)
    # Discard each full-resolution image immediately after spatial aggregation.
    records: list[tuple[datetime, FloatArray]] = []
    for path in paths:
        time, values = decode_frame(path)
        records.append(
            (time, aggregate_radar(values, latitude, longitude, grid, resolution))
        )
    records.sort(key=lambda pair: pair[0])
    if not records:
        raise ValueError("No radar frames supplied")
    amounts = [values for _, values in records]
    frames = xr.DataArray(
        np.stack(amounts),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": [np.datetime64(t, "ns") for t, _ in records],
            "latitude": grid.latitude,
            "longitude": grid.longitude,
        },
    )
    return hourly_radar(frames).rename("precipitation_mm").to_dataset()
