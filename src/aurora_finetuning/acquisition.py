"""Cached provider access with bounded requests and input provenance."""

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol
from zipfile import ZipFile, is_zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from aurora_finetuning.config import Config
from aurora_finetuning.credentials import knmi_api_key
from aurora_finetuning.grid import make_grid
from aurora_finetuning.telemetry import Recorder, write_json

SURFACE = {
    "t2m": "2t",
    "u10": "10u",
    "v10": "10v",
    "msl": "msl",
    "d2m": "2d",
    "tcwv": "tcwv",
    "tcc": "tcc",
    "u100": "100u",
    "v100": "100v",
    "sp": "sp",
    "lcc": "lcc",
    "mcc": "mcc",
    "hcc": "hcc",
    "skt": "skt",
    "stl1": "stl1",
    "swvl1": "swvl1",
    "siconc": "ci",
    "sd": "scaled_sd",
}
CDS_SURFACE = (
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "2m_dewpoint_temperature",
    "total_column_water_vapour",
    "total_cloud_cover",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "surface_pressure",
    "low_cloud_cover",
    "medium_cloud_cover",
    "high_cloud_cover",
    "skin_temperature",
    "soil_temperature_level_1",
    "volumetric_soil_water_layer_1",
    "sea_ice_cover",
    "snow_depth",
    "total_precipitation",
)
LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
ATMOSPHERE = ("t", "u", "v", "q", "z")


class CDSClient(Protocol):
    """Minimal CDS client interface, also used by deterministic transport tests."""

    def retrieve(
        self, name: str, request: dict[str, object], target: str
    ) -> object: ...


def sha256(path: Path) -> str:
    """Hash a file using bounded memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def http_session() -> requests.Session:
    """Create a session with bounded retries for idempotent requests only."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def download_file(session: requests.Session, url: str, path: Path) -> None:
    """Stream an unauthenticated temporary download URL to an atomic cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with session.get(url, stream=True, timeout=(10, 120)) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)
    partial.replace(path)


def normalize_cds_file(path: Path, recorder: Recorder) -> None:
    """Merge CDS step-type NetCDF bundles, retaining the original ZIP and hash."""
    if not is_zipfile(path):
        return
    import xarray as xr

    manifest = path.with_suffix(".json")
    info = json.loads(manifest.read_text())
    source_hash = sha256(path)
    archive_path = path.with_suffix(".source.zip")
    if archive_path.exists():
        raise FileExistsError(archive_path)
    normalized = path.with_suffix(".normalized.nc")
    with (
        recorder.measure("prepare.cds.zip_bundle"),
        TemporaryDirectory(prefix=".cds-", dir=path.parent) as temporary,
        ZipFile(path) as archive,
    ):
        datasets: list[xr.Dataset] = []
        for index, member in enumerate(archive.infolist()):
            if not member.filename.endswith(".nc"):
                raise ValueError("CDS ZIP contains an unexpected non-NetCDF member")
            target = Path(temporary) / f"part-{index}.nc"
            with archive.open(member) as source, target.open("wb") as destination:
                import shutil

                shutil.copyfileobj(source, destination)
            with xr.open_dataset(target) as dataset:
                datasets.append(dataset.load())
        if not datasets:
            raise ValueError("CDS ZIP has no NetCDF datasets")
        xr.merge(datasets, compat="no_conflicts", join="exact").to_netcdf(normalized)
    path.rename(archive_path)
    normalized.replace(path)
    info.update(
        sha256=sha256(path),
        source_archive=archive_path.name,
        source_sha256=source_hash,
        conversion="merge CDS step-type NetCDF bundle",
    )
    replacement = manifest.with_suffix(".replacement.json")
    write_json(replacement, info)
    replacement.replace(manifest)


def pull_cds(
    config: Config,
    start: date,
    end: date,
    recorder: Recorder,
    client: CDSClient | None = None,
    surface_only: bool = False,
) -> list[Path]:
    """Retrieve daily regional ERA5 requests, reusing verified cache entries."""
    if end < start:
        raise ValueError("end must not precede start")
    if client is None:
        import cdsapi

        client = cdsapi.Client(timeout=120, retry_max=3, sleep_max=10, quiet=True)
    grid = make_grid(config)
    bounds = grid.describe()
    paths: list[Path] = []
    day = start
    while day <= end:
        for kind, variables in (
            ("single", list(CDS_SURFACE)),
            (
                "pressure",
                [
                    "temperature",
                    "u_component_of_wind",
                    "v_component_of_wind",
                    "specific_humidity",
                    "geopotential",
                ],
            ),
        ):
            if surface_only and kind == "pressure":
                continue
            dataset = (
                f"reanalysis-era5-{'single' if kind == 'single' else 'pressure'}-levels"
            )
            request: dict[str, object] = {
                "product_type": "reanalysis",
                "variable": variables,
                "year": f"{day.year:04d}",
                "month": f"{day.month:02d}",
                "day": f"{day.day:02d}",
                "time": [f"{hour:02d}:00" for hour in range(24)],
                "area": [bounds[k] for k in ("north", "west", "south", "east")],
                "grid": [config.resolution, config.resolution],
                "data_format": "netcdf",
                "download_format": "unarchived",
            }
            if kind == "pressure":
                request["pressure_level"] = [str(p) for p in LEVELS]
            request_id = hashlib.sha256(
                json.dumps(request, sort_keys=True).encode()
            ).hexdigest()[:12]
            path = config.root / "data/raw/era5" / f"{day}-{kind}-{request_id}.nc"
            manifest = path.with_suffix(".json")
            with recorder.measure(f"download.cds.{day}.{kind}"):
                if path.exists() and manifest.exists():
                    info = json.loads(manifest.read_text())
                    if info["sha256"] != sha256(path):
                        raise ValueError(f"Cached file checksum mismatch: {path}")
                elif path.exists() or manifest.exists():
                    raise ValueError(
                        f"Incomplete cache entry: {path}; inspect before removing"
                    )
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    partial = path.with_suffix(".partial.nc")
                    client.retrieve(dataset, request, str(partial))
                    partial.replace(path)
                    write_json(
                        manifest,
                        {
                            "dataset": dataset,
                            "request": request,
                            "sha256": sha256(path),
                        },
                    )
            normalize_cds_file(path, recorder)
            paths.append(path)
        day += timedelta(days=1)
    return paths


def knmi_files(
    config: Config,
    dataset: str,
    version: str,
    start: str,
    end: str,
    recorder: Recorder,
    max_files: int = 4,
) -> list[Path]:
    """Download a bounded KNMI filename range with pagination and separate auth."""
    if max_files < 1 or start > end:
        raise ValueError("Invalid filename range or max-files")
    if not all(c.isalnum() or c in "_-" for c in dataset + version.replace(".", "")):
        raise ValueError("Invalid KNMI dataset or version")
    key = knmi_api_key(config.root)
    base = f"https://api.dataplatform.knmi.nl/open-data/v1/datasets/{dataset}/versions/{version}/files"
    paths: list[Path] = []
    with http_session() as api, http_session() as downloads:
        api.headers["Authorization"] = key
        token: str | None = None
        seen: set[str] = set()
        while len(paths) < max_files:
            with recorder.measure("download.knmi.list"):
                params: dict[str, str | int] = {
                    # KDP lists after begin. Request a preceding prefix, then
                    # enforce the inclusive user range locally.
                    "begin": start[:-1],
                    "end": end,
                    "maxKeys": min(100, max_files - len(paths)),
                    "sorting": "asc",
                }
                if token is not None:
                    params["nextPageToken"] = token
                response = api.get(base, params=params, timeout=(10, 60))
                response.raise_for_status()
                payload = response.json()
            files = payload.get("files", [])
            past_end = False
            for item in files:
                if len(paths) >= max_files:
                    break
                name = item["filename"]
                if (
                    not isinstance(name, str)
                    or name in ("", ".", "..")
                    or Path(name).name != name
                ):
                    raise ValueError("KNMI returned an unsafe filename")
                if name < start:
                    continue
                if name > end:
                    past_end = True
                    break
                path = config.root / "data/raw/knmi" / dataset / version / name
                with recorder.measure("download.knmi.file"):
                    manifest = path.with_name(path.name + ".json")
                    if path.exists() and manifest.exists():
                        if json.loads(manifest.read_text())["sha256"] != sha256(path):
                            raise ValueError(f"Cached file checksum mismatch: {path}")
                    elif path.exists() or manifest.exists():
                        raise ValueError(f"Incomplete cache entry: {path}")
                    else:
                        link = api.get(f"{base}/{name}/url", timeout=(10, 60))
                        link.raise_for_status()
                        download_file(
                            downloads, link.json()["temporaryDownloadUrl"], path
                        )
                        write_json(
                            manifest,
                            {
                                "dataset": dataset,
                                "version": version,
                                "filename": name,
                                "sha256": sha256(path),
                            },
                        )
                paths.append(path)
            next_token = payload.get("nextPageToken")
            if past_end or not next_token:
                break
            if not isinstance(next_token, str) or next_token in seen:
                raise ValueError("KNMI pagination did not advance")
            seen.add(next_token)
            token = next_token
    if not paths:
        raise ValueError("No KNMI files matched the requested filename range")
    return paths
