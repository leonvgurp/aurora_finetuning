"""Offline behavior of locality lookup, live orchestration and table-only CLI."""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import requests
import torch
import xarray as xr
from aurora import AuroraV1p5, Batch

from aurora_finetuning import locations, weather
from aurora_finetuning.acquisition import ATMOSPHERE, LEVELS, SURFACE, sha256
from aurora_finetuning.cli import main
from aurora_finetuning.config import Config
from aurora_finetuning.forecast import query_grid
from aurora_finetuning.grid import make_grid
from aurora_finetuning.locations import resolve_location
from aurora_finetuning.table import format_table
from aurora_finetuning.telemetry import Recorder, write_json


def test_lookup_caches_and_coordinates_bypass_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    session = requests.Session()

    def get(url: str, **kwargs: object) -> requests.Response:
        calls.append(url)
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(
            {
                "response": {
                    "numFound": 1,
                    "docs": [
                        {
                            "woonplaatsnaam": "Utrecht",
                            "weergavenaam": "Utrecht, Utrecht",
                            "centroide_ll": "POINT(5.12 52.09)",
                        }
                    ],
                }
            }
        ).encode()
        return response

    monkeypatch.setattr(session, "get", get)
    monkeypatch.setattr(locations, "http_session", lambda: session)
    point = resolve_location("Utrecht", None, None, tmp_path)
    assert point.latitude == 52.09 and point.longitude == 5.12
    assert resolve_location("  UTRECHT ", None, None, tmp_path) == point
    assert resolve_location(None, 52.09, 5.12, tmp_path).latitude == 52.09
    assert len(calls) == 1


@pytest.mark.parametrize(
    "name,lat,lon",
    [
        (None, None, None),
        (None, 52.0, None),
        ("Utrecht", 52.0, 5.0),
        (None, float("nan"), 5.0),
        (None, 49.0, 5.0),
        (" ", None, None),
    ],
)
def test_invalid_locations(
    tmp_path: Path, name: str | None, lat: float | None, lon: float | None
) -> None:
    with pytest.raises(ValueError):
        resolve_location(name, lat, lon, tmp_path)


@pytest.mark.parametrize(
    "records,match",
    [
        ([], "No Dutch locality"),
        (
            [
                {"woonplaatsnaam": "Bergen", "weergavenaam": "Bergen NH"},
                {"woonplaatsnaam": "Bergen", "weergavenaam": "Bergen Limburg"},
            ],
            "Bergen NH; Bergen Limburg",
        ),
    ],
)
def test_lookup_no_match_and_ambiguity(
    tmp_path: Path, records: list[dict[str, str]], match: str
) -> None:
    key = hashlib.sha256(b"bergen").hexdigest()
    write_json(tmp_path / f"{key}.json", records)
    with pytest.raises(ValueError, match=match):
        resolve_location("Bergen", None, None, tmp_path)


def grid_forecast(cycle: datetime, hours: int) -> xr.Dataset:
    times = [
        np.datetime64(cycle + timedelta(hours=h), "ns") for h in range(1, hours + 1)
    ]
    shape = (hours, 2, 2)
    return xr.Dataset(
        {
            "temperature_c": (("time", "latitude", "longitude"), np.full(shape, 12.0)),
            "precipitation_mm": (
                ("time", "latitude", "longitude"),
                np.full(shape, 0.25),
            ),
        },
        coords={"time": times, "latitude": [53.0, 51.0], "longitude": [4.0, 6.0]},
        attrs={"initialization_utc": cycle.isoformat()},
    )


@pytest.mark.parametrize(
    "cycle,expected",
    [
        (datetime(2024, 10, 26, 23), ["02:00:00+02:00", "02:00:00+01:00"]),
        (datetime(2024, 3, 31), ["03:00:00+02:00", "04:00:00+02:00"]),
        (datetime(2024, 1, 1, 22), ["00:00:00+01:00", "01:00:00+01:00"]),
    ],
)
def test_table_dst_midnight_and_intervals(cycle: datetime, expected: list[str]) -> None:
    result = query_grid(grid_forecast(cycle, 2), 52, 5, 2, cycle.replace(tzinfo=UTC))
    rows = result["hourly"]
    assert isinstance(rows, list)
    for row, suffix in zip(rows, expected, strict=True):
        assert row["time"].endswith(suffix)
        end = datetime.fromisoformat(row["time"]).astimezone(UTC)
        begin = datetime.fromisoformat(row["interval_start"]).astimezone(UTC)
        assert end - begin == timedelta(hours=1)
    table = format_table(result)
    assert len(table.splitlines()) == 4
    assert "12.00 | 0.250" in table
    with pytest.raises(ValueError, match="cover"):
        query_grid(grid_forecast(cycle, 2), 52, 5, 3, cycle.replace(tzinfo=UTC))


def test_table_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="Invalid temperature"):
        format_table(
            {
                "hourly": [
                    {
                        "time": "2024-01-01T01:00+01:00",
                        "temperature_c": float("nan"),
                        "precipitation_mm": 0,
                    }
                ]
            }
        )


def audit_result(missing: list[str] | None = None) -> dict[str, object]:
    return {
        "cycles": [
            {
                "missing_surface": ["siconc"] if missing is None else missing,
                "missing_pressure_fields": [],
            }
        ]
        * 2
    }


def test_cycle_fallback_and_age(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2024, 1, 2, 13, tzinfo=UTC)
    assert weather.recent_cycles(now, 12) == [
        datetime(2024, 1, 2, 12),
        datetime(2024, 1, 2, 6),
    ]
    seen: list[datetime] = []

    def audit(cycle: datetime, recorder: Recorder) -> dict[str, object]:
        seen.append(cycle)
        if cycle.hour == 12:
            response = requests.Response()
            response.status_code = 404
            raise requests.HTTPError(response=response)
        return audit_result()

    monkeypatch.setattr(weather, "audit_ifs", audit)
    assert weather.select_cycle(Config(), now, Recorder()).hour == 6
    assert len(seen) == 2
    monkeypatch.setattr(weather, "audit_ifs", lambda *args: audit_result(["sp"]))
    with pytest.raises(ValueError, match="No complete fresh"):
        weather.select_cycle(Config(), now, Recorder())

    def offline(cycle: datetime, recorder: Recorder) -> dict[str, object]:
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(weather, "audit_ifs", offline)
    with pytest.raises(requests.ConnectionError):
        weather.select_cycle(Config(), now, Recorder())


def ice_file(config: Config, stamp: datetime) -> Path:
    grid = make_grid(config)
    path = config.root / "data/raw/sea-ice" / f"{stamp.date()}.nc"
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {"siconc": (("time", "latitude", "longitude"), np.zeros((1, *grid.shape)))},
        coords={
            "time": [np.datetime64(stamp, "ns")],
            "latitude": grid.latitude,
            "longitude": grid.longitude,
        },
    ).to_netcdf(path)
    write_json(path.with_suffix(".json"), {"sha256": sha256(path)})
    return path


def test_supplement_valid_for_both_states_and_checksum(tmp_path: Path) -> None:
    config = Config(root=tmp_path)
    stamp = datetime(2024, 1, 1)
    path = ice_file(config, stamp)
    assert weather.find_sea_ice(config, stamp + timedelta(days=7), Recorder()) == path
    with pytest.raises(ValueError, match="weather-setup"):
        weather.find_sea_ice(config, stamp + timedelta(hours=3), Recorder())
    with pytest.raises(ValueError, match="weather-setup"):
        weather.find_sea_ice(config, stamp + timedelta(days=8), Recorder())
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Modified sea-ice"):
        weather.find_sea_ice(config, stamp, Recorder())


def atmospheric_inputs(config: Config, cycle: datetime) -> None:
    """Write realistic regional files so orchestration exercises real preparation."""
    grid = make_grid(config)
    surface_units = {
        "t2m": "K",
        "d2m": "K",
        "skt": "K",
        "stl1": "K",
        "msl": "Pa",
        "sp": "Pa",
        "u10": "m/s",
        "v10": "m/s",
        "u100": "m/s",
        "v100": "m/s",
    }
    atmos_units = {"t": "K", "u": "m/s", "v": "m/s", "q": "kg/kg", "z": "m2 s-2"}
    for time in (cycle - timedelta(hours=6), cycle):
        for kind in ("single", "pressure"):
            coords: dict[str, object] = {
                "time": [np.datetime64(time, "ns")],
                "latitude": grid.latitude,
                "longitude": grid.longitude,
            }
            if kind == "pressure":
                coords["level"] = list(LEVELS)
            ds = xr.Dataset(coords=coords)
            dims = (
                ("time", "latitude", "longitude")
                if kind == "single"
                else ("time", "level", "latitude", "longitude")
            )
            shape = (
                (1, *grid.shape) if kind == "single" else (1, len(LEVELS), *grid.shape)
            )
            for name in SURFACE if kind == "single" else ATMOSPHERE:
                ds[name] = (dims, np.ones(shape))
                ds[name].attrs["units"] = (
                    surface_units if kind == "single" else atmos_units
                ).get(name, "1")
            path = (
                config.root
                / "data/raw/ifs"
                / f"{time.date()}-{kind}-{time.hour:02d}.nc"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(path)
            write_json(path.with_suffix(".json"), {"sha256": sha256(path)})


@pytest.fixture
def live_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Config, datetime, list[int]]:
    config = Config(root=tmp_path, hours=24, device="cpu")
    cycle = datetime(2024, 1, 2, 6)
    atmospheric_inputs(config, cycle)
    ice_file(config, datetime(2024, 1, 1))
    assets = tmp_path / "checkpoints/base/assets.json"
    write_json(assets, {"revision": config.checkpoint_revision})
    monkeypatch.setattr(weather, "audit_ifs", lambda *args: audit_result())
    monkeypatch.setattr(weather, "verify_assets", lambda *args: {})
    monkeypatch.setattr(weather, "pull_ifs", lambda *args: [])
    monkeypatch.setattr(
        "aurora_finetuning.preparation.static_fields",
        lambda c: {
            "lsm": torch.zeros(make_grid(c).shape),
        },
    )
    monkeypatch.setattr(weather, "load_model", lambda *args: None)
    calls: list[int] = []

    def infer(
        model: AuroraV1p5, batch: Batch, hours: int, config: Config, recorder: Recorder
    ) -> xr.Dataset:
        assert batch.atmos_vars["t"].shape[2] == len(LEVELS)
        calls.append(hours)
        return grid_forecast(cycle, hours)

    monkeypatch.setattr(weather, "infer", infer)
    return config, cycle, calls


def test_rollout_age_and_cache_across_locations_and_horizons(
    live_fixture: tuple[Config, datetime, list[int]],
) -> None:
    config, cycle, calls = live_fixture
    now = (cycle + timedelta(hours=1, minutes=15)).replace(tzinfo=UTC)
    result = weather.predict_weather(config, 52, 5, Recorder(), now=now)
    assert calls == [25]
    rows = result["hourly"]
    assert isinstance(rows, list) and len(rows) == 24
    assert result["initialization_age_hours"] == 1.25
    recorder = Recorder()
    weather.predict_weather(replace(config, hours=8), 52.1, 5.1, recorder, now=now)
    assert calls == [25]
    assert any("outputs/weather" in path for path in recorder.inputs)
    cached = next((config.root / "outputs/weather").glob("*/*.nc"))
    cached.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Modified forecast"):
        weather.predict_weather(config, 52, 5, Recorder(), now=now)


def test_cli_table_only_and_config_precedence(
    live_fixture: tuple[Config, datetime, list[int]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, cycle, calls = live_fixture
    config_path = config.root / "config.toml"
    config_path.write_text('root = "."\nhours = 3\ndevice = "cpu"\n')
    real_predict = weather.predict_weather

    def predict(
        c: Config,
        lat: float,
        lon: float,
        recorder: Recorder,
        *,
        start: datetime | None,
        adapter: Path | None,
    ) -> dict[str, object]:
        print("provider progress")
        return real_predict(
            c,
            lat,
            lon,
            recorder,
            now=cycle.replace(tzinfo=UTC),
            start=start,
            adapter=adapter,
        )

    monkeypatch.setattr(weather, "predict_weather", predict)
    base = [
        "weather",
        "--latitude",
        "52",
        "--longitude",
        "5",
        "--config",
        str(config_path),
    ]
    assert main(base) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("| Hour ending")
    assert len(captured.out.splitlines()) == 5
    assert "provider progress" not in captured.out
    assert "provider progress" in captured.err
    assert main([*base, "--hours", "1"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 3
    assert calls == [3]
    assert main([*base, "--hours", "25"]) == 2
    assert capsys.readouterr().out == ""
    reports = [
        json.loads(p.read_text())
        for p in (config.root / "outputs/timings").glob("*.json")
    ]
    assert any(not report["success"] for report in reports)


def test_window_validation_before_network(tmp_path: Path) -> None:
    now = datetime(2024, 1, 1, tzinfo=UTC)
    for start in (
        datetime(2024, 1, 1),
        now - timedelta(hours=1),
        now + timedelta(days=3),
    ):
        with pytest.raises(ValueError):
            weather.predict_weather(
                Config(root=tmp_path), 52, 5, Recorder(), now=now, start=start
            )


def test_cli_provider_failure_has_no_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unavailable(*args: object, **kwargs: object) -> dict[str, object]:
        print("provider progress")
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(weather, "predict_weather", unavailable)
    report = tmp_path / "failure.json"
    assert (
        main(
            ["weather", "--latitude", "52", "--longitude", "5", "--report", str(report)]
        )
        == 3
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "provider progress" in captured.err
    assert json.loads(report.read_text())["success"] is False
