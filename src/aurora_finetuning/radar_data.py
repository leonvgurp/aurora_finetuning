"""Measured five-minute radar data on its native projected one-kilometre grid."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import xarray as xr
from pyproj import CRS, Transformer

from aurora_finetuning.acquisition import http_session, knmi_files, sha256
from aurora_finetuning.config import Config
from aurora_finetuning.credentials import knmi_api_key
from aurora_finetuning.grid import FloatArray
from aurora_finetuning.radar import decode_frame, text_attribute
from aurora_finetuning.telemetry import Recorder

HISTORY_FRAMES = 24
RADAR_DATASET = "nl_rdr_data_rtcor_5m"


def radar_geometry(path: Path) -> xr.Dataset:
    """Interpret KNMI's explicit projection, pixel offsets and upper-left convention."""
    with h5py.File(path) as handle:
        geo = handle["geographic"]
        attrs = geo.attrs
        if text_attribute(attrs["geo_pixel_def"]) != "LU":
            raise ValueError(
                "Radar requires the documented upper-left pixel convention"
            )
        if text_attribute(attrs["geo_dim_pixel"]) != "KM,KM":
            raise ValueError("Radar pixel dimensions must be kilometres")
        projection = text_attribute(
            geo["map_projection"].attrs["projection_proj4_params"]
        )
        # Known legacy KNMI metadata typo, also corrected by the pySTEPS importer.
        legacy = (
            "+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 "
            "+a=6378.14 +b=6356.75 +x_0=0 y_0=0"
        )
        if projection == legacy:
            projection = (
                "+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 "
                "+a=6378137 +b=6356752 +x_0=0 +y_0=0 +units=m"
            )
        crs = CRS.from_proj4(projection)
        factor = crs.axis_info[0].unit_conversion_factor
        dx = float(np.asarray(attrs["geo_pixel_size_x"]).item())
        dy = float(np.asarray(attrs["geo_pixel_size_y"]).item())
        if not np.isclose(dx, 1, atol=0.001) or not np.isclose(dy, -1, atol=0.001):
            raise ValueError("Radar adapter requires native 1 km pixels")
        columns = int(np.asarray(attrs["geo_number_columns"]).item())
        rows = int(np.asarray(attrs["geo_number_rows"]).item())
        col_offset = float(np.asarray(attrs["geo_column_offset"]).item())
        row_offset = float(np.asarray(attrs["geo_row_offset"]).item())
    x = (np.arange(columns) + col_offset + 0.5) * dx * 1000 / factor
    y = (np.arange(rows) + row_offset + 0.5) * dy * 1000 / factor
    xx, yy = np.meshgrid(x, y)
    longitude, latitude = Transformer.from_crs(
        crs, "EPSG:4326", always_xy=True
    ).transform(xx, yy)
    return xr.Dataset(
        coords={
            "x": x,
            "y": y,
            "latitude": (("y", "x"), np.asarray(latitude)),
            "longitude": (("y", "x"), np.asarray(longitude)),
        },
        attrs={"projection": projection, "pixel_size_km": 1.0},
    )


def radar_time(path: Path) -> datetime:
    """Read the end of a measured radar interval; reject multi-image forecast files."""
    with h5py.File(path) as handle:
        overview = handle["overview"].attrs
        # Observed files also contain quality and adjustment-factor images.
        # Forecast files contain multiple precipitation images at different times.
        count = sum(
            1
            for name in handle
            if name.startswith("image")
            and text_attribute(handle[name].attrs["image_geo_parameter"])
            == "PRECIP_[MM]"
        )
        if count != 1:
            raise ValueError(
                "Only measured single-frame radar is allowed, not radar forecasts"
            )
        if (
            text_attribute(handle["image1"].attrs["image_geo_parameter"])
            != "PRECIP_[MM]"
        ):
            raise ValueError(
                "Radar must contain measured five-minute rainfall amounts in mm"
            )
        stamp = text_attribute(overview["product_datetime_end"])
        time = datetime.strptime(stamp, "%d-%b-%Y;%H:%M:%S.000")
        if "product_datetime_start" in overview:
            start = datetime.strptime(
                text_attribute(overview["product_datetime_start"]),
                "%d-%b-%Y;%H:%M:%S.000",
            )
            if time - start != timedelta(minutes=5):
                raise ValueError("Measured radar intervals must be five minutes")
    if time.minute % 5 or time.second or time.microsecond:
        raise ValueError("Radar timestamps must be exact five-minute UTC boundaries")
    return time


def native_radar(
    paths: list[Path],
    recorder: Recorder,
    *,
    latitude: float,
    longitude: float,
    size: int,
) -> xr.Dataset:
    """Prepare a bounded native-grid patch without spatial averaging or upsampling."""
    if not paths or len(paths) > 128 or not 128 <= size <= 256:
        raise ValueError("Supply 1–128 observed frames and a 128–256 pixel patch")
    geometry = radar_geometry(paths[0])
    distance = (geometry.latitude - latitude) ** 2 + (
        (geometry.longitude - longitude) * np.cos(np.deg2rad(latitude))
    ) ** 2
    row, col = np.unravel_index(int(np.argmin(distance.values)), distance.shape)
    if (
        row < size // 2
        or col < size // 2
        or row + (size + 1) // 2 > geometry.sizes["y"]
        or col + (size + 1) // 2 > geometry.sizes["x"]
    ):
        raise ValueError("Radar patch extends beyond the native image")
    selection = {
        "y": slice(row - size // 2, row - size // 2 + size),
        "x": slice(col - size // 2, col - size // 2 + size),
    }
    cropped = geometry.isel(selection)
    records: list[tuple[datetime, FloatArray]] = []
    for path in paths:
        time = radar_time(path)
        if path.name != f"RAD_NL25_RAC_RT_{time:%Y%m%d%H%M}.h5":
            raise ValueError(
                "Radar conditioning requires real-time RTCOR measured frames; "
                "retrospectively gauge-adjusted products are not causal inputs"
            )
        other = radar_geometry(path)
        if not geometry.identical(other):
            raise ValueError("Radar frames have different projections or native grids")
        decoded_time, values = decode_frame(path)
        if decoded_time != time or values.shape != distance.shape:
            raise ValueError("Radar image and geolocation do not agree")
        records.append((time, values[selection["y"], selection["x"]]))
        recorder.inputs[str(path)] = sha256(path)
    records.sort(key=lambda r: r[0])
    times = np.array([np.datetime64(t, "ns") for t, _ in records])
    if len(times) > 1 and not np.all(np.diff(times) == np.timedelta64(5, "m")):
        raise ValueError("Measured radar frames must be consecutive and unique")
    cropped["precipitation_mm"] = (
        ("time", "y", "x"),
        np.stack([v for _, v in records]).astype(np.float32),
    )
    cropped = cropped.assign_coords(time=times)
    cropped.precipitation_mm.attrs.update(units="mm", interval_minutes=5)
    cropped.attrs.update(
        source="KNMI measured radar",
        radar_product="RTCOR-5m",
        radar_role="observations",
        interval="(time - 5 minutes, time]",
        input_sha256=json.dumps(recorder.inputs, sort_keys=True),
    )
    return cropped


def validate_native(dataset: xr.Dataset) -> None:
    """Require observed native-grid amounts and exact, unique five-minute timestamps."""
    if (
        dataset.attrs.get("radar_role") != "observations"
        or dataset.attrs.get("source") != "KNMI measured radar"
        or dataset.attrs.get("radar_product") != "RTCOR-5m"
    ):
        raise ValueError(
            "Radar input must be prepared measured observations, not a forecast"
        )
    if dataset.attrs.get("pixel_size_km") != 1.0:
        raise ValueError("Radar input must retain its native 1 km grid")
    if (
        dataset.precipitation_mm.dims != ("time", "y", "x")
        or dataset.precipitation_mm.attrs.get("units") != "mm"
        or dataset.precipitation_mm.attrs.get("interval_minutes") != 5
    ):
        raise ValueError("Expected five-minute radar amounts in mm on (time, y, x)")
    if any(dataset[name].dims != ("y", "x") for name in ("latitude", "longitude")):
        raise ValueError("Radar requires matching two-dimensional geolocation")
    times = dataset.time.values.astype("datetime64[ns]")
    if (
        times.size == 0
        or np.isnat(times).any()
        or np.any(times.astype(np.int64) % (300 * 10**9))
        or (len(times) > 1 and not np.all(np.diff(times) == np.timedelta64(5, "m")))
    ):
        raise ValueError(
            "Radar timestamps must be consecutive five-minute UTC boundaries"
        )


def history_at(dataset: xr.Dataset, cutoff: datetime) -> xr.DataArray:
    """Select exactly two hours of past observations, excluding all future targets."""
    validate_native(dataset)
    times = [
        np.datetime64(cutoff - timedelta(minutes=5 * i), "ns")
        for i in range(HISTORY_FRAMES - 1, -1, -1)
    ]
    try:
        result: xr.DataArray = dataset["precipitation_mm"].sel(time=times)
        return result
    except KeyError as error:
        raise ValueError(
            "Missing measured radar history before the forecast cutoff"
        ) from error


def measured_targets(dataset: xr.Dataset, cutoff: datetime, lead: int) -> xr.DataArray:
    """Sum twelve subsequent measured five-minute amounts for a forecast lead hour."""
    validate_native(dataset)
    if not 1 <= lead <= 6:
        raise ValueError("Radar training supports leads 1–6 hours")
    times = [
        np.datetime64(cutoff + timedelta(minutes=60 * (lead - 1) + 5 * i), "ns")
        for i in range(1, 13)
    ]
    try:
        result: xr.DataArray = (
            dataset["precipitation_mm"].sel(time=times).sum("time", skipna=False)
        )
        return result
    except KeyError as error:
        raise ValueError("Missing future measured radar training targets") from error


def live_history(
    config: Config,
    latitude: float,
    longitude: float,
    recorder: Recorder,
    now: datetime,
) -> xr.Dataset:
    """Fetch measured past frames through the secret-safe KNMI downloader."""
    base = f"https://api.dataplatform.knmi.nl/open-data/v1/datasets/{RADAR_DATASET}/versions/1.0/files"
    with http_session() as session:
        session.headers["Authorization"] = knmi_api_key(config.root)
        with recorder.measure("radar.latest_measured_frame"):
            # Model loading can cross an observation boundary. Keep the requested
            # cutoff fixed instead of ingesting subsequently measured rain.
            end = f"RAD_NL25_RAC_RT_{now.astimezone(UTC):%Y%m%d%H%M}.h5"
            response = session.get(
                base,
                params={"maxKeys": "1", "sorting": "desc", "end": end},
                timeout=(10, 30),
            )
            response.raise_for_status()
        files = response.json()["files"]
    if not files:
        raise ValueError("No measured KNMI radar files are available")
    name = files[0]["filename"]
    prefix = "RAD_NL25_RAC_RT_"
    if (
        not isinstance(name, str)
        or not name.startswith(prefix)
        or not name.endswith(".h5")
    ):
        raise ValueError("Unexpected measured radar filename")
    cutoff = datetime.strptime(name[len(prefix) : -3], "%Y%m%d%H%M")
    age = (now.astimezone(UTC).replace(tzinfo=None) - cutoff).total_seconds() / 60
    if not 0 <= age <= config.radar_max_age_minutes:
        raise ValueError(
            "Measured radar is future-dated or exceeds radar_max_age_minutes"
        )
    start = cutoff - timedelta(minutes=5 * (HISTORY_FRAMES - 1))
    paths = knmi_files(
        config,
        RADAR_DATASET,
        "1.0",
        f"{prefix}{start:%Y%m%d%H%M}.h5",
        name,
        recorder,
        HISTORY_FRAMES,
    )
    with recorder.measure("radar.prepare_native_history"):
        dataset = native_radar(
            paths, recorder, latitude=latitude, longitude=longitude, size=128
        )
    history_at(dataset, cutoff)
    return dataset
