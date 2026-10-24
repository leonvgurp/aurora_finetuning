"""Strict public forecast acquisition and normalization, separate from Aurora inputs."""

import argparse
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import requests
import xarray as xr

from aurora_finetuning.acquisition import download_file, http_session, sha256
from aurora_finetuning.artifacts import atomic_output
from aurora_finetuning.config import Config
from aurora_finetuning.experiment_scores import validate_dataset
from aurora_finetuning.telemetry import Recorder, write_json


def interval_rain(
    values: xr.DataArray, initialization: datetime, interval: int
) -> xr.DataArray:
    """Difference cumulative forecasts only across complete native intervals."""
    times = values.time.values.astype("datetime64[ns]")
    if initialization.tzinfo is None:
        raise ValueError("Initialization requires an explicit UTC offset")
    init = np.datetime64(initialization.astimezone(UTC).replace(tzinfo=None), "ns")
    previous = np.concatenate(([init], times[:-1]))
    if np.any(times - previous != np.timedelta64(interval, "h")):
        raise ValueError(
            "Cumulative rain requires consecutive native steps from initialization"
        )
    result = values - values.shift(time=1, fill_value=0)
    if bool((result < -1e-6).any()):
        raise ValueError(
            "Accumulated rainfall decreases; mixed cycles or unsupported encoding"
        )
    return result.clip(min=0)


def normalize_public(
    paths: list[Path],
    output: Path,
    provider: str,
    initialization: datetime,
    interval: int,
    cumulative: bool,
    available: datetime | None = None,
) -> None:
    """Normalize explicit GRIB/NetCDF runs; require documented fields and units.

    KNMI radar nowcasts use already decoded five-minute amounts, aggregated by
    the separate nowcast normalizer. Forecast products never enter measured data.
    """
    if initialization.tzinfo is None or interval not in (1, 2, 3, 6):
        raise ValueError("Require aware initialization and a supported native interval")
    fields: dict[str, list[xr.DataArray]] = {}
    for path in paths:
        if path.suffix == ".nc":
            with xr.open_dataset(path) as source:
                datasets = [source.load()]
        else:
            import cfgrib

            datasets = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
        try:
            for dataset in datasets:
                # cfgrib represents run time as scalar time and validity on step.
                if "valid_time" in dataset.coords:
                    ds = dataset.drop_vars("time", errors="ignore")
                    if ds.valid_time.ndim == 0:
                        ds = ds.expand_dims(valid_time=[ds.valid_time.values])
                    elif "step" in ds.dims:
                        ds = ds.swap_dims({"step": "valid_time"})
                    ds = ds.rename({"valid_time": "time"})
                else:
                    ds = dataset
                for source_name, target in (
                    ("t2m", "temperature_c"),
                    ("2t", "temperature_c"),
                    ("tp", "precipitation_mm"),
                ):
                    if source_name not in ds:
                        continue
                    array = ds[source_name]
                    # GRIB groups carry conflicting scalar heights (2 m vs surface).
                    # The supported variable names already encode that distinction.
                    scalar_axes = [
                        name
                        for name in array.coords
                        if name not in array.dims
                        and name not in {"time", "latitude", "longitude"}
                    ]
                    array = array.drop_vars(scalar_axes)
                    units = array.attrs.get("units")
                    if target == "temperature_c":
                        if units != "K":
                            raise ValueError("Public temperature requires Kelvin input")
                        array = array - 273.15
                        array.attrs["units"] = "degC"
                    else:
                        if units not in {"m", "kg m**-2", "kg m-2", "mm"}:
                            raise ValueError("Unsupported public rainfall units")
                        array = array * (1000 if units == "m" else 1)
                        array.attrs.update(units="mm", interval_hours=interval)
                    fields.setdefault(target, []).append(array)
        finally:
            for dataset in datasets:
                dataset.close()
    if not fields:
        raise ValueError(
            "No supported t2m/2t or tp fields; inspect the provider GRIB codes"
        )
    variables = {}
    for name, arrays in fields.items():
        array = xr.concat(arrays, dim="time").sortby("time")
        if len(np.unique(array.time.values)) != array.sizes["time"]:
            raise ValueError("Duplicate validity times; do not mix forecast runs")
        if name == "precipitation_mm" and cumulative:
            array = interval_rain(array, initialization, interval)
            array.attrs.update(units="mm", interval_hours=interval)
        variables[name] = array
    result = xr.Dataset(variables)
    result.attrs.update(
        initialization_utc=initialization.astimezone(UTC).isoformat(),
        source=provider,
        verification_role="public_forecast_comparator",
    )
    if available:
        result.attrs["available_utc"] = available.astimezone(UTC).isoformat()
    validate_dataset(result)
    with atomic_output(output) as temporary:
        result.to_netcdf(temporary)
    write_json(
        output.with_suffix(".provenance.json"),
        {
            "provider": provider,
            "initialization_utc": initialization.isoformat(),
            "available_utc": available.isoformat() if available else None,
            "normalized_sha256": sha256(output),
            "inputs": {str(p): sha256(p) for p in paths},
            "rainfall_mode": "cumulative-differenced"
            if cumulative
            else "native-interval",
            "interval_hours": interval,
        },
    )


def fetch_ifs(
    initialization: datetime, steps: list[int], output: Path, budget: int
) -> dict[str, object]:
    """Retrieve only public IFS temperature/rain messages using bounded byte ranges."""
    if initialization.tzinfo is None:
        raise ValueError("IFS requires an aware initialization")
    initialization = initialization.astimezone(UTC)
    if initialization.hour % 6 or any(
        (initialization.minute, initialization.second, initialization.microsecond)
    ):
        raise ValueError("IFS requires an aware six-hour UTC initialization")
    initialization = initialization.astimezone(UTC)
    if not steps or any(s < 3 or s > 24 or s % 3 for s in steps) or budget <= 0:
        raise ValueError(
            "Use three-hour IFS steps between 3 and 24 and a positive budget"
        )
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    recorder = Recorder()
    config = Config(root=output, max_download_gb=budget / 1024**3)
    with http_session() as session:
        for step in sorted(set(steps)):
            stamp = initialization.strftime("%Y%m%d%H0000")
            base = (
                f"https://data.ecmwf.int/forecasts/{initialization:%Y%m%d}/{initialization:%H}z/"
                f"ifs/0p25/oper/{stamp}-{step}h-oper-fc"
            )
            response = session.get(base + ".index", timeout=(10, 60))
            response.raise_for_status()
            records = [json.loads(line) for line in response.text.splitlines() if line]
            selected = [
                r
                for r in records
                if r.get("param") in {"2t", "tp"} and r.get("levtype") == "sfc"
            ]
            if len(selected) != 2:
                raise ValueError("IFS product is missing unique surface 2t/tp fields")
            for record in selected:
                offset, length = int(record["_offset"]), int(record["_length"])
                session.headers["Range"] = f"bytes={offset}-{offset + length - 1}"
                # Reject global-file fallbacks before writing.
                with session.get(
                    base + ".grib2", stream=True, timeout=(10, 120)
                ) as check:
                    check.raise_for_status()
                    if (
                        check.status_code != 206
                        or check.headers.get("Content-Range")
                        != f"bytes {offset}-{offset + length - 1}/"
                        + check.headers.get("Content-Range", "").split("/")[-1]
                    ):
                        raise ValueError(
                            "Provider did not honor the requested byte range"
                        )
                download_file(
                    session,
                    base + ".grib2",
                    output / f"{step}-{record['param']}.grib2",
                    config,
                    recorder,
                )
                session.headers.pop("Range", None)
    result = {
        "provider": "ECMWF IFS 0p25",
        "initialization_utc": initialization.isoformat(),
        "acquired_utc": datetime.now(UTC).isoformat(),
        "network_bytes": recorder.network_bytes,
        "files": {p.name: sha256(p) for p in output.glob("*.grib2")},
    }
    write_json(output / "manifest.json", result)
    return result


def normalize_nowcast(path: Path, output: Path) -> None:
    """Aggregate decoded KNMI forecast bins, preserving complete-hour support.

    The input boundary is explicit: precipitation_mm(time,y,x) in mm per five
    minutes, UTC end times, latitude/longitude, and initialization_utc. Raw HDF5
    products must be decoded using their documented version, never as RTCOR.
    """
    with xr.open_dataset(path) as source:
        dataset = source.load()
    if (
        dataset.precipitation_mm.attrs.get("units") != "mm"
        or dataset.precipitation_mm.attrs.get("interval_minutes") != 5
    ):
        raise ValueError("Nowcast requires explicit five-minute mm amounts")
    init = datetime.fromisoformat(str(dataset.attrs["initialization_utc"]))
    if init.tzinfo is None:
        raise ValueError("Nowcast initialization requires a UTC offset")
    start = np.datetime64(init.astimezone(UTC).replace(tzinfo=None), "ns")
    ends = start + np.arange(1, 25) * np.timedelta64(5, "m")
    if not np.array_equal(dataset.time.values, ends):
        raise ValueError("Require 24 future five-minute bins; exclude the +0 analysis")
    hourly: list[xr.DataArray] = [
        dataset.precipitation_mm.isel(time=slice(i, i + 12)).sum("time", skipna=False)
        for i in (0, 12)
    ]
    rain = xr.concat(hourly, dim="time").assign_coords(time=[ends[11], ends[23]])

    rain.attrs.update(units="mm", interval_hours=1)
    result = rain.to_dataset(name="precipitation_mm")
    result.attrs.update(
        dataset.attrs,
        source="KNMI radar forecast",
        verification_role="public_forecast_comparator",
    )
    validate_dataset(result)
    with atomic_output(output) as temporary:
        result.to_netcdf(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    """Fetch IFS or normalize explicitly acquired public forecast artifacts."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("fetch-ifs", "normalize", "normalize-nowcast"))
    p.add_argument("--input", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--provider", choices=("harmonie", "ifs"), default="harmonie")
    p.add_argument("--initialization", type=datetime.fromisoformat)
    p.add_argument("--available", type=datetime.fromisoformat)
    p.add_argument("--interval-hours", type=int, default=1)
    p.add_argument("--cumulative", action="store_true")
    p.add_argument("--steps", type=int, nargs="+", default=[3, 6, 9])
    p.add_argument("--max-download-bytes", type=int, default=250_000_000)
    args = p.parse_args(argv)
    try:
        if args.action == "fetch-ifs":
            if args.initialization is None:
                raise ValueError("--initialization is required")
            result = fetch_ifs(
                args.initialization, args.steps, args.output, args.max_download_bytes
            )
            print(json.dumps(result, indent=2))
        elif args.action == "normalize-nowcast":
            if not args.input or len(args.input) != 1:
                raise ValueError("Provide exactly one decoded --input")
            normalize_nowcast(args.input[0], args.output)
        else:
            if not args.input or args.initialization is None:
                raise ValueError("--input and --initialization are required")
            normalize_public(
                args.input,
                args.output,
                args.provider,
                args.initialization,
                args.interval_hours,
                args.cumulative,
                args.available,
            )
        return 0
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        logging.error("%s", error)
        return 2
    except requests.RequestException as error:
        status = error.response.status_code if error.response is not None else None
        logging.error("Public forecast retrieval failed (HTTP %s)", status)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
