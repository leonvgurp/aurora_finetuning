"""Strict ECMWF IFS initialization retrieval, with no fabricated missing fields."""

from datetime import date, datetime, timedelta
from pathlib import Path

import cfgrib
import numpy as np
import xarray as xr
from ecmwf.opendata import Client

from aurora_finetuning.acquisition import LEVELS, SURFACE, http_session, sha256
from aurora_finetuning.config import Config
from aurora_finetuning.grid import make_grid
from aurora_finetuning.preparation import canonical, require_units, subset
from aurora_finetuning.telemetry import Recorder, write_json

IFS_SURFACE = {
    "2t": "t2m",
    "10u": "u10",
    "10v": "v10",
    "msl": "msl",
    "2d": "d2m",
    "tcwv": "tcwv",
    "tcc": "tcc",
    "100u": "u100",
    "100v": "v100",
    "sp": "sp",
    "lcc": "lcc",
    "mcc": "mcc",
    "hcc": "hcc",
    "skt": "skt",
    "sot": "stl1",
    "vsw": "swvl1",
    "siconc": "siconc",
    "sd": "sd",
}


def regional_grib_field(dataset: xr.Dataset) -> xr.Dataset:
    """Remove per-field scalar axes while retaining atmospheric level dimensions."""
    for name in ("depthBelowLandLayer", "soilLayer"):
        if name in dataset.dims:
            if dataset.sizes[name] != 1:
                raise ValueError("IFS soil request must contain only the first layer")
            dataset = dataset.isel({name: 0}, drop=True)
    if "isobaricInhPa" in dataset.dims:
        dataset = dataset.rename({"isobaricInhPa": "level"})
    dataset = canonical(dataset)
    return dataset.drop_vars(
        [name for name in dataset.coords if name not in dataset.dims and name != "time"]
    )


def validate_ifs_units(dataset: xr.Dataset, kind: str) -> None:
    """Require complete inputs in physical units compatible with Aurora/ERA5."""
    wind = {"m s**-1", "m s-1", "m/s"}
    fraction = {"1", "(0 - 1)"}
    if kind == "single":
        units = {
            **dict.fromkeys(("t2m", "d2m", "skt", "stl1"), {"K"}),
            **dict.fromkeys(("msl", "sp"), {"Pa"}),
            **dict.fromkeys(("u10", "v10", "u100", "v100"), wind),
            **dict.fromkeys(("tcc", "lcc", "mcc", "hcc", "siconc"), fraction),
            "tcwv": {"kg m**-2", "kg m-2", "kg/m2"},
            "swvl1": {"m**3 m**-3", "m3 m-3", "m3/m3"},
            "sd": {"m of water equivalent", "m"},
        }
    else:
        units = {
            "t": {"K"},
            "u": wind,
            "v": wind,
            "q": {"kg kg**-1", "kg kg-1", "kg/kg"},
            "z": {"m**2 s**-2", "m2 s-2", "m^2 s^-2"},
        }
        if "level" not in dataset.coords or not set(LEVELS) <= set(
            dataset.level.values
        ):
            raise ValueError("IFS input is missing required pressure levels")
    missing = set(units) - set(dataset.data_vars)
    if missing:
        raise ValueError(f"IFS input is missing required fields: {sorted(missing)}")
    for name, allowed in units.items():
        require_units(dataset[name], allowed)


def audit_ifs(initialization: datetime, recorder: Recorder) -> dict[str, object]:
    """Check tiny public index files before downloading any global GRIB fields."""
    import json

    if initialization.tzinfo is not None or initialization.hour % 6:
        raise ValueError("IFS initialization must be a naive UTC six-hour cycle")
    reports: list[dict[str, object]] = []
    with http_session() as session:
        for time in (initialization - timedelta(hours=6), initialization):
            stamp = time.strftime("%Y%m%d%H0000")
            url = (
                f"https://data.ecmwf.int/forecasts/{time:%Y%m%d}/{time:%H}z/"
                f"ifs/0p25/oper/{stamp}-0h-oper-fc.index"
            )
            with recorder.measure("download.ifs.audit_index"):
                response = session.get(url, timeout=(10, 60))
                response.raise_for_status()
                records = [
                    json.loads(line) for line in response.text.splitlines() if line
                ]
            surface = {
                str(r.get("param"))
                for r in records
                if r.get("levtype") in ("sfc", "sol")
            }
            missing_surface = sorted(set(IFS_SURFACE) - surface)
            required_pairs = {
                (name, str(level))
                for name in ("t", "u", "v", "q", "z")
                for level in LEVELS
            }
            available_pairs = {
                (str(r.get("param")), str(r.get("levelist")))
                for r in records
                if r.get("levtype") == "pl"
            }
            reports.append(
                {
                    "cycle": time.isoformat(),
                    "missing_surface": missing_surface,
                    "missing_pressure_fields": sorted(required_pairs - available_pairs),
                    "compatible": not missing_surface
                    and required_pairs <= available_pairs,
                }
            )
    return {
        "cycles": reports,
        "compatible": all(r["compatible"] for r in reports),
        "note": "Availability only; units and physical equivalence need validation",
    }


def pull_ifs(
    config: Config,
    initialization: datetime,
    recorder: Recorder,
    sea_ice: Path | None = None,
) -> list[Path]:
    """Retrieve two step-zero IFS cycles and require all Aurora 1.5 inputs.

    Public IFS may omit sea-ice or land fields for a cycle/product. Retrieval
    deliberately fails on that mismatch rather than inventing values. The large
    GRIB fields are global on the provider; local NetCDF artifacts are regional.
    """
    if (
        initialization.tzinfo is not None
        or initialization.hour % 6
        or (
            initialization.minute or initialization.second or initialization.microsecond
        )
    ):
        raise ValueError("IFS initialization must be naive UTC on a six-hour cycle")
    if config.resolution != 0.25:
        raise ValueError("IFS adapter initially supports only the 0.25-degree product")
    audit = audit_ifs(initialization, recorder)
    cycles = audit["cycles"]
    supplement_resolves = (
        sea_ice is not None
        and isinstance(cycles, list)
        and all(
            c["missing_surface"] in ([], ["siconc"])
            and not c["missing_pressure_fields"]
            for c in cycles
        )
    )
    if not audit["compatible"] and not supplement_resolves:
        raise ValueError(
            f"Public IFS fields are incomplete for Aurora 1.5: {audit['cycles']}. "
            "No GRIB fields were downloaded; do not fabricate missing fields."
        )
    client: Client | None = None
    grid = make_grid(config)
    results: list[Path] = []
    for time in (initialization - timedelta(hours=6), initialization):
        ice = sea_ice_supplement(sea_ice, time, config) if sea_ice else None
        for kind in ("single", "pressure"):
            root = config.root / "data/raw/ifs"
            path = root / f"{time.date()}-{kind}-{time.hour:02d}.nc"
            manifest = path.with_suffix(".json")
            if path.exists():
                import json

                if not manifest.exists() or json.loads(manifest.read_text())[
                    "sha256"
                ] != sha256(path):
                    raise ValueError(f"Incomplete or modified IFS cache: {path}")
                with xr.open_dataset(path) as existing:
                    subset(existing, grid)
                    validate_ifs_units(existing, kind)
                results.append(path)
                continue
            root.mkdir(parents=True, exist_ok=True)
            request: dict[str, object] = {
                "date": time.strftime("%Y%m%d"),
                "time": time.hour,
                "type": "fc",
                "stream": "oper",
                "step": 0,
                "levtype": "sfc" if kind == "single" else "pl",
            }
            request["param"] = (
                list(IFS_SURFACE) if kind == "single" else ["t", "u", "v", "q", "z"]
            )
            if ice is not None and kind == "single":
                request["param"] = [name for name in IFS_SURFACE if name != "siconc"]
            if kind == "pressure":
                request["levelist"] = list(LEVELS)
            grib = path.with_suffix(".grib2")
            grib_paths = [grib]
            soil_request: dict[str, object] | None = None
            if kind == "single":
                request["param"] = [
                    name
                    for name in IFS_SURFACE
                    if name not in ("sot", "vsw")
                    and not (name == "siconc" and ice is not None)
                ]
                soil_request = {
                    **request,
                    "param": ["sot", "vsw"],
                    "levtype": "sol",
                    "levelist": [1],
                }
                grib_paths.append(path.with_suffix(".soil.grib2"))
            with recorder.measure(f"download.ifs.{time.isoformat()}.{kind}"):
                if client is None:
                    client = Client(source="ecmwf", model="ifs", resol="0p25")
                client.retrieve(request, target=str(grib))
                if soil_request is not None:
                    client.retrieve(soil_request, target=str(grib_paths[1]))
            with recorder.measure("prepare.ifs.grib"):
                chunks: list[xr.Dataset] = []
                datasets = [
                    dataset
                    for file in grib_paths
                    for dataset in cfgrib.open_datasets(
                        str(file), backend_kwargs={"indexpath": ""}
                    )
                ]
                for ds in datasets:
                    try:
                        # Height/depth describe each variable, not a shared axis.
                        # Keeping scalar 2/10/100 m coordinates makes xr.merge
                        # reject otherwise compatible surface fields.
                        chunks.append(subset(regional_grib_field(ds), grid).load())
                    finally:
                        ds.close()
                merged = xr.merge(chunks, compat="no_conflicts")
                if kind == "single":
                    # cfgrib emits CDS-like short names for surface fields.
                    merged = merged.rename(
                        {
                            name: target
                            for name, target in IFS_SURFACE.items()
                            if name in merged and target not in merged
                        }
                    )
                    merged = merged.rename(
                        {
                            name: target
                            for name, target in (("st", "stl1"), ("swvl", "swvl1"))
                            if name in merged and target not in merged
                        }
                    )
                    if ice is not None:
                        merged["siconc"] = ice
                    missing = set(SURFACE) - set(merged.data_vars)
                    if missing:
                        raise ValueError(
                            f"Public IFS is missing required fields {sorted(missing)}; "
                            "live forecasting remains unavailable for this product"
                        )
                if "time" not in merged.dims:
                    merged = merged.expand_dims(time=[np.datetime64(time, "ns")])
                merged.attrs.update(
                    source="ECMWF IFS step-zero cycles", experimental="true"
                )
                validate_ifs_units(merged, kind)
                merged.to_netcdf(path)
                write_json(
                    manifest,
                    {
                        "source": "ECMWF IFS",
                        "request": request,
                        "soil_request": soil_request,
                        "grid": grid.describe(),
                        "sha256": sha256(path),
                        "sea_ice_supplement": str(sea_ice) if sea_ice else None,
                        "sea_ice_sha256": sha256(sea_ice) if sea_ice else None,
                    },
                )
            results.append(path)
    return results


def pull_sea_ice(config: Config, day: date, recorder: Recorder) -> Path:
    """Retrieve a small ERA5 sea-ice supplement, retaining its original valid time."""
    import cdsapi

    grid = make_grid(config)
    path = config.root / "data/raw/sea-ice" / f"{day}.nc"
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    request: dict[str, object] = {
        "product_type": "reanalysis",
        "variable": ["sea_ice_cover"],
        "year": f"{day.year:04d}",
        "month": f"{day.month:02d}",
        "day": f"{day.day:02d}",
        "time": ["00:00"],
        "area": [
            float(grid.latitude[0]),
            float(grid.longitude[0]),
            float(grid.latitude[-1]),
            float(grid.longitude[-1]),
        ],
        "grid": [config.resolution, config.resolution],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    with recorder.measure("download.cds.sea_ice"):
        client = cdsapi.Client(quiet=True, timeout=120, retry_max=3, sleep_max=10)
        partial = path.with_suffix(".partial.nc")
        client.retrieve("reanalysis-era5-single-levels", request, str(partial))
        partial.replace(path)
        write_json(
            path.with_suffix(".json"),
            {
                "source": "ERA5 sea ice supplement",
                "request": request,
                "sha256": sha256(path),
            },
        )
    return path


def sea_ice_supplement(path: Path, time: datetime, config: Config) -> xr.DataArray:
    """Select an explicitly supplied ice snapshot no older than seven days.

    Persisting an older ice analysis is an explicit experimental assumption, not
    a same-time analysis. The original time is retained as variable metadata.
    """
    with xr.open_dataset(path) as ds:
        region = subset(ds, make_grid(config))
        if "siconc" not in region:
            raise ValueError("Sea-ice supplement requires ERA5 siconc")
        valid = region.sel(time=slice(None, np.datetime64(time, "ns")))
        if valid.sizes.get("time", 0) == 0:
            raise ValueError("Sea-ice supplement has no snapshot before initialization")
        selected = valid.siconc.isel(time=-1).load()
        stamp = np.datetime64(selected.time.values, "ns")
        age = float((np.datetime64(time, "ns") - stamp) / np.timedelta64(1, "h"))
        if not 0 <= age <= 168:
            raise ValueError("Sea-ice supplement must be at most seven days old")
        values = selected.values
        if np.any(np.isfinite(values) & ((values < 0) | (values > 1))):
            raise ValueError(
                "Sea-ice concentration must be a fraction from zero to one"
            )
        selected.attrs.update(
            supplement_valid_time=str(stamp),
            supplement_age_hours=age,
            supplement_assumption="persist ERA5 sea-ice analysis",
        )
        result = selected.drop_vars("time")
        if not isinstance(result, xr.DataArray):
            raise TypeError("Sea-ice supplement did not yield a data array")
        return result
