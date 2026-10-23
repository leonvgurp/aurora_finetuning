"""Convert regional ERA5 fields into reproducible, bounded-memory model inputs."""

import json
import pickle
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from aurora import Batch, Metadata
from aurora.insolation import insolation

from aurora_finetuning.acquisition import ATMOSPHERE, LEVELS, SURFACE, sha256
from aurora_finetuning.assets import STATIC, verify_assets
from aurora_finetuning.config import Config
from aurora_finetuning.grid import Grid, make_grid
from aurora_finetuning.telemetry import Recorder


def canonical(ds: xr.Dataset) -> xr.Dataset:
    """Normalise CDS coordinate names and sort a Greenwich-crossing regional grid."""
    renames = {
        old: new
        for old, new in (
            ("valid_time", "time"),
            ("pressure_level", "level"),
            ("lat", "latitude"),
            ("lon", "longitude"),
        )
        if old in ds.dims
    }
    ds = ds.rename(renames)
    if "longitude" not in ds.coords or "latitude" not in ds.coords:
        raise ValueError("Dataset requires latitude and longitude coordinates")
    lon = (ds.longitude + 180) % 360 - 180
    return (
        ds.assign_coords(longitude=lon)
        .sortby("longitude")
        .sortby("latitude", ascending=False)
    )


def subset(ds: xr.Dataset, grid: Grid) -> xr.Dataset:
    """Select exact configured cells; fail rather than extrapolate."""
    ds = canonical(ds)
    try:
        return ds.sel(latitude=grid.latitude, longitude=grid.longitude)
    except KeyError as error:
        raise ValueError("Input grid does not match configured aligned grid") from error


def open_era5(
    directory: Path,
    grid: Grid,
    times: list[datetime],
    kind: str,
    recorder: Recorder | None = None,
) -> xr.Dataset:
    """Load only requested time slices; avoid loading full months into memory."""
    slices: list[xr.Dataset] = []
    for day in sorted({t.date() for t in times}):
        candidates = sorted(directory.glob(f"{day}-{kind}-*.nc"))
        selected: xr.Dataset | None = None
        for path in candidates:
            manifest = path.with_suffix(".json")
            if not manifest.is_file():
                raise ValueError(f"Missing input provenance manifest: {manifest}")
            expected = json.loads(manifest.read_text())["sha256"]
            actual = sha256(path)
            if expected != actual:
                raise ValueError(f"Input checksum mismatch: {path}")
            with xr.open_dataset(path) as source:
                try:
                    candidate = subset(source, grid)
                except ValueError:
                    continue
                day_times = [
                    np.datetime64(t.replace(tzinfo=None), "ns")
                    for t in times
                    if t.date() == day
                ]
                # Earlier compatible cache files take precedence at shared times.
                available = [
                    t
                    for t in day_times
                    if t in candidate.time.values
                    and (selected is None or t not in selected.time.values)
                ]
                if not available:
                    continue
                piece = candidate.sel(time=available).load()
                if recorder is not None:
                    recorder.inputs[str(path)] = actual
                selected = (
                    piece
                    if selected is None
                    else xr.concat([selected, piece], dim="time")
                )
            if selected is not None and all(
                t in selected.time.values for t in day_times
            ):
                break
        if selected is None:
            raise ValueError(f"No compatible {kind} ERA5 file for {day}; run pull-cds")
        wanted = [np.datetime64(t, "ns") for t in times if t.date() == day]
        if not all(t in selected.time.values for t in wanted):
            raise ValueError(f"Missing required times for {day}")
        slices.append(selected.sel(time=wanted))
    return xr.concat(slices, dim="time").sortby("time")


def finite_tensor(array: xr.DataArray) -> torch.Tensor:
    """Reject unexpected missing or infinite values before model execution."""
    values = np.asarray(array.values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(
            f"Missing/nonfinite data in {array.name}; explicit masks required"
        )
    return torch.from_numpy(values.copy())


def require_units(array: xr.DataArray, allowed: set[str]) -> None:
    """Reject incompatible physical units instead of interpreting them silently."""
    units = str(array.attrs.get("units", ""))
    if units not in allowed:
        raise ValueError(
            f"Unexpected units for {array.name}: {units!r}; expected {allowed}"
        )


@lru_cache(maxsize=2)
def static_fields(config: Config) -> dict[str, torch.Tensor]:
    """Read only verified Microsoft static assets and select the regional cells."""
    grid = make_grid(config)
    verify_assets(config)
    # Only load the verified official pickle: it is executable content.
    with (config.root / "checkpoints/base" / STATIC).open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Expected a dictionary in static asset")
    result: dict[str, torch.Tensor] = {}
    for name, value in raw.items():
        values = np.asarray(value, dtype=np.float32)
        if values.shape != (721, 1440):
            raise ValueError(
                f"Unexpected static field dimensions for {name}: {values.shape}"
            )
        ilat = np.rint((90 - grid.latitude) / 0.25).astype(int)
        ilon = np.rint((grid.longitude % 360) / 0.25).astype(int) % 1440
        region = values[np.ix_(ilat, ilon)]
        if not np.isfinite(region).all():
            raise ValueError(f"Nonfinite static field {name}")
        result[str(name)] = torch.from_numpy(region.copy())
    return result


def prepare_batch(
    config: Config, initialization: datetime, recorder: Recorder, source: str = "era5"
) -> Batch:
    """Build two six-hour-separated atmospheric states, in a regional context."""
    if initialization.tzinfo is not None:
        raise ValueError("Initialization must be naive UTC")
    if initialization.minute or initialization.second or initialization.microsecond:
        raise ValueError("Initialization must be on an exact hour")
    grid = make_grid(config)
    times = [initialization - timedelta(hours=6), initialization]
    if source not in ("era5", "ifs"):
        raise ValueError("source must be era5 or ifs")
    directory = config.root / "data/raw" / source
    with recorder.measure("prepare.static"):
        static = static_fields(config)
    land_mask = xr.DataArray(
        static["lsm"].numpy(),
        dims=("latitude", "longitude"),
        coords={"latitude": grid.latitude, "longitude": grid.longitude},
    )
    with recorder.measure("prepare.surface"):
        surface = open_era5(directory, grid, times, "single", recorder)
        missing = set(SURFACE) - set(surface.data_vars)
        if missing:
            raise ValueError(f"Missing ERA5 surface variables: {sorted(missing)}")
        variables: dict[str, torch.Tensor] = {}
        for source, target in SURFACE.items():
            field = surface[source].transpose("time", "latitude", "longitude")
            if source in ("t2m", "d2m", "skt", "stl1"):
                require_units(field, {"K"})
            elif source in ("msl", "sp"):
                require_units(field, {"Pa"})
            elif source in ("u10", "v10", "u100", "v100"):
                require_units(field, {"m s**-1", "m s-1", "m/s"})
            # ERA5 ocean/land masks cause structural missing values.
            # Soil is undefined at sea; sea-ice concentration is undefined on land.
            if source in ("stl1", "swvl1"):
                field = field.where(~(field.isnull() & (land_mask < 0.5)), 0)
            if source == "siconc":
                # ERA5's categorical land mask can also exclude mixed coastal
                # cells that have <50% land in the supplied static fraction.
                field = field.where(~(field.isnull() & (land_mask > 0)), 0)
            variables[target] = finite_tensor(field)[None]
        solar = np.stack(
            [
                insolation([t], grid.latitude, grid.longitude, enforce_2d=True)[0]
                for t in times
            ]
        )
        variables["insolation"] = torch.from_numpy(solar.astype(np.float32))[None]
    with recorder.measure("prepare.atmosphere"):
        atmos = open_era5(directory, grid, times, "pressure", recorder).sel(
            level=list(LEVELS)
        )
        require_units(atmos["t"], {"K"})
        for name in ("u", "v"):
            require_units(atmos[name], {"m s**-1", "m s-1", "m/s"})
        require_units(atmos["q"], {"kg kg**-1", "kg kg-1", "kg/kg"})
        require_units(atmos["z"], {"m**2 s**-2", "m2 s-2", "m^2 s^-2"})
        atmosphere = {
            name: finite_tensor(
                atmos[name].transpose("time", "level", "latitude", "longitude")
            )[None]
            for name in ATMOSPHERE
        }
    return Batch(
        variables,
        static,
        atmosphere,
        Metadata(
            lat=torch.tensor(grid.latitude, dtype=torch.float32),
            lon=torch.tensor(grid.longitude, dtype=torch.float32),
            time=(initialization,),
            atmos_levels=LEVELS,
        ),
    )


def target_fields(
    config: Config, time: datetime, recorder: Recorder | None = None
) -> xr.Dataset:
    """Return ERA5 temperature and hourly precipitation targets in physical units."""
    ds = open_era5(
        config.root / "data/raw/era5", make_grid(config), [time], "single", recorder
    )
    if "tp" not in ds:
        raise ValueError("ERA5 targets require total_precipitation (tp)")
    require_units(ds["tp"], {"m"})
    require_units(ds["t2m"], {"K"})
    return xr.Dataset(
        {"temperature_c": ds.t2m - 273.15, "precipitation_mm": ds.tp * 1000}
    )
