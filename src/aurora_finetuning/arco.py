"""Explicit ERA5 pressure-field import from Google's public ARCO mirror."""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from aurora_finetuning.acquisition import LEVELS, sha256
from aurora_finetuning.config import Config
from aurora_finetuning.grid import Grid, make_grid
from aurora_finetuning.telemetry import Recorder, write_json

STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
VARIABLES = {
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
    "geopotential": "z",
}


def select_pressure_field(
    source: xr.DataArray, grid: Grid, times: list[datetime]
) -> xr.DataArray:
    """Select exact physical cells before loading, keeping only regional values."""
    result = source.sel(
        time=[np.datetime64(t, "ns") for t in times],
        level=list(LEVELS),
        latitude=grid.latitude,
        longitude=grid.longitude % 360,
    )
    return result.assign_coords(longitude=grid.longitude).transpose(
        "time", "level", "latitude", "longitude"
    )


def pull_arco_pressure(
    config: Config, initialization: datetime, recorder: Recorder
) -> list[Path]:
    """Import only two input states; label the source and regridding explicitly.

    The upstream storage uses global 154 MB chunks per hour/variable. Reading is
    serial and each result is cropped immediately, bounding working memory. This
    source is ERA5 regridded by ARCO, not promised bit-identical to a CDS export.
    """
    if initialization.tzinfo is not None or (
        initialization.minute or initialization.second or initialization.microsecond
    ):
        raise ValueError("Initialization must be an exact naive UTC hour")
    times = [initialization - timedelta(hours=6), initialization]
    paths: list[Path] = []
    with recorder.measure("download.arco.open_metadata"):
        store = xr.open_zarr(
            STORE, chunks=None, consolidated=True, storage_options={"token": "anon"}
        )
    try:
        grid = make_grid(config)
        for day in sorted({t.date() for t in times}):
            selected_times = [t for t in times if t.date() == day]
            request = {
                "store": STORE,
                "times": [t.isoformat() for t in selected_times],
                "levels": list(LEVELS),
                "grid": grid.describe(),
            }
            identifier = hashlib.sha256(
                json.dumps(request, sort_keys=True).encode()
            ).hexdigest()[:12]
            path = (
                config.root / "data/raw/era5" / f"{day}-pressure-arco-{identifier}.nc"
            )
            manifest = path.with_suffix(".json")
            if path.exists() and manifest.exists():
                if sha256(path) != json.loads(manifest.read_text())["sha256"]:
                    raise ValueError(f"ARCO cache checksum mismatch: {path}")
                paths.append(path)
                continue
            if path.exists() or manifest.exists():
                raise ValueError(f"Incomplete ARCO cache entry: {path}")
            fields: dict[str, xr.DataArray] = {}
            for source, target in VARIABLES.items():
                with recorder.measure(f"download.arco.{day}.{target}"):
                    # Load one hour at a time to bound global decompression memory.
                    regions: list[xr.DataArray] = []
                    for time in selected_times:
                        region = select_pressure_field(
                            store[source], grid, [time]
                        ).load()
                        if not np.isfinite(region.values).all():
                            raise ValueError(f"Nonfinite ARCO pressure field: {source}")
                        regions.append(region)
                    fields[target] = xr.concat(regions, dim="time")
            dataset = xr.Dataset(fields)
            dataset.attrs.update(
                source=STORE,
                source_kind="ARCO ERA5 regridded 0.25 degrees",
                source_last_updated=str(store.attrs.get("last_updated", "unknown")),
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with recorder.measure("download.arco.write_regional"):
                dataset.to_netcdf(path)
                write_json(
                    manifest,
                    {
                        "dataset": "ARCO ERA5 pressure fields",
                        "request": request,
                        "sha256": sha256(path),
                        "source_attrs": {k: str(v) for k, v in store.attrs.items()},
                    },
                )
            paths.append(path)
    finally:
        store.close()
    return paths
