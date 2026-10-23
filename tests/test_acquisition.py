"""Test provider request construction, cache integrity and auth isolation offline."""

import json
from datetime import date
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
from urllib3.response import HTTPResponse

from aurora_finetuning.acquisition import knmi_files, pull_cds
from aurora_finetuning.config import Config
from aurora_finetuning.telemetry import Recorder


class FakeCDS:
    """Record the external transport request without contacting CDS."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def retrieve(self, name: str, request: dict[str, object], target: str) -> object:
        self.calls.append((name, request))
        Path(target).write_bytes(b"transport fixture")
        return None


def test_cds_cache_and_provenance(tmp_path: Path) -> None:
    config = Config(root=tmp_path)
    client = FakeCDS()
    paths = pull_cds(config, date(2024, 1, 1), date(2024, 1, 1), Recorder(), client)
    assert len(client.calls) == 2
    assert client.calls[1][1]["pressure_level"] == [
        str(x)
        for x in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
    ]
    pull_cds(config, date(2024, 1, 1), date(2024, 1, 1), Recorder(), client)
    assert len(client.calls) == 2
    paths[0].write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        pull_cds(config, date(2024, 1, 1), date(2024, 1, 1), Recorder(), client)


def test_knmi_pagination_and_no_key_on_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KNMI_API_KEY", "test-secret")
    requests_seen: list[requests.PreparedRequest] = []

    def send(request: requests.PreparedRequest, **kwargs: object) -> requests.Response:
        requests_seen.append(request)
        assert request.url is not None
        response = requests.Response()
        response.status_code = 200
        response.request = request
        if request.url.startswith("https://downloads.example/"):
            assert "Authorization" not in request.headers
            content = b"fixture data"
        else:
            assert request.headers["Authorization"] == "test-secret"
            if "/url" in request.url:
                content = json.dumps(
                    {"temporaryDownloadUrl": "https://downloads.example/data"}
                ).encode()
            elif "nextPageToken" in request.url:
                content = json.dumps(
                    {"files": [{"filename": "b.nc"}], "isTruncated": False}
                ).encode()
            else:
                content = json.dumps(
                    {
                        "files": [{"filename": "a.nc"}],
                        "isTruncated": True,
                        "nextPageToken": "page-two",
                    }
                ).encode()
        response.raw = HTTPResponse(body=BytesIO(content), preload_content=False)
        return response

    with patch.object(requests.Session, "send", side_effect=send):
        paths = knmi_files(
            Config(root=tmp_path), "station-data", "1.0", "a", "z", Recorder(), 2
        )
    assert [p.name for p in paths] == ["a.nc", "b.nc"]
    assert len(requests_seen) == 6
    assert all("test-secret" not in p.read_text() for p in tmp_path.rglob("*.json"))


def test_cds_step_type_zip_is_merged_and_original_preserved(tmp_path: Path) -> None:
    from zipfile import ZipFile, is_zipfile

    import xarray as xr

    from aurora_finetuning.acquisition import normalize_cds_file, sha256
    from aurora_finetuning.telemetry import write_json

    instant = tmp_path / "instant.nc"
    accumulated = tmp_path / "accumulated.nc"
    xr.Dataset({"t2m": ("time", [280.0])}, coords={"time": [0]}).to_netcdf(instant)
    xr.Dataset({"tp": ("time", [0.001])}, coords={"time": [0]}).to_netcdf(accumulated)
    path = tmp_path / "download.nc"
    with ZipFile(path, "w") as archive:
        archive.write(instant, "instant.nc")
        archive.write(accumulated, "accumulated.nc")
    original_hash = sha256(path)
    write_json(path.with_suffix(".json"), {"sha256": original_hash})
    normalize_cds_file(path, Recorder())
    assert not is_zipfile(path)
    assert sha256(path.with_suffix(".source.zip")) == original_hash
    with xr.open_dataset(path) as merged:
        assert merged.t2m.item() == 280
        assert merged.tp.item() == 0.001
    manifest = json.loads(path.with_suffix(".json").read_text())
    assert manifest["sha256"] == sha256(path)
    assert manifest["source_sha256"] == original_hash
