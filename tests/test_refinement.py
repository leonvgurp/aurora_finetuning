"""Rainfall spatial skill, exclusive artifacts and pressure/wind semantics."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from aurora_finetuning.artifacts import atomic_output
from aurora_finetuning.config import Config
from aurora_finetuning.forecast import query_grid
from aurora_finetuning.observations import normalize_stations
from aurora_finetuning.rainfall_metrics import (
    bootstrap_cases,
    fractions_skill,
    hourly_report,
    motion_forecast,
)
from aurora_finetuning.resources import check_budget
from aurora_finetuning.telemetry import Recorder


def test_spatial_scores_and_eight_hour_accumulation() -> None:
    truth = np.zeros((8, 16, 16))
    truth[:, 5:8, 5:8] = 1
    assert fractions_skill(truth[0], truth[0], 5, 0.1) == pytest.approx(1)
    shifted = np.roll(truth, 1, axis=2)
    small = fractions_skill(shifted[0], truth[0], 1, 0.1)
    large = fractions_skill(shifted[0], truth[0], 5, 0.1)
    assert small is not None and large is not None and large > small
    report = hourly_report(truth, truth)
    total, hourly = report["eight_hour_total"], report["by_lead_hour"]
    assert isinstance(total, dict) and total["mae"] == 0
    assert isinstance(hourly, list) and len(hourly) == 8
    assert bootstrap_cases(np.zeros((1, 8)))["ci95"] is None
    assert (
        np.asarray(bootstrap_cases(np.ones((3, 8)))["ci95"]).tolist() == [[1] * 8] * 2
    )
    with pytest.raises(ValueError, match="eight"):
        hourly_report(truth[:6], truth[:6])


def test_motion_extrapolates_without_longitude_wrap() -> None:
    history = np.zeros((24, 32, 32))
    history[-7, 10:13, 10:13] = 1
    history[-1, 10:13, 12:15] = 1
    future = motion_forecast(history, 30)
    assert future[11, 15] == 1
    assert np.isnan(future[:, :2]).all()


def test_atomic_failure_and_competing_writer_preserve_existing(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    with pytest.raises(RuntimeError), atomic_output(path) as temporary:
        temporary.write_text("partial")
        raise RuntimeError("interrupted")
    assert not path.exists() and not list(tmp_path.iterdir())
    with pytest.raises(FileExistsError), atomic_output(path) as temporary:
        temporary.write_text("ours")
        path.write_text("other writer")
    assert path.read_text() == "other writer"


def test_station_pressure_mean_wind_and_quality_flags() -> None:
    source = xr.Dataset(
        {
            "P": (
                ("time", "station"),
                [[1010.0, 1020.0]],
                {"units": "hPa", "standard_name": "air_pressure_at_mean_sea_level"},
            ),
            "FF": (
                ("time", "station"),
                [[5.0, 10.0]],
                {
                    "units": "m/s",
                    "description": "Past 10 minute mean wind at 10 meters",
                    "ancillary_variables": "wind_quality",
                },
            ),
            "DD": (
                ("time", "station"),
                [[90.0, 990.0]],
                {
                    "units": "°",
                    "description": "Past 10 minute mean direction at 10 meters",
                },
            ),
            "FX": (("time", "station"), [[100.0, 100.0]]),
            "wind_quality": (
                ("time", "station"),
                [[0, 1]],
                {"flag_values": [0, 1], "flag_meanings": "good bad"},
            ),
        },
        coords={
            "time": [np.datetime64("2024-01-01")],
            "station": ["a", "b"],
            "latitude": ("station", [52.0, 52.1]),
            "longitude": ("station", [5.0, 5.1]),
        },
    )
    decoded = normalize_stations(source)
    assert float(decoded.wind_u_ms[0, 0]) == pytest.approx(-5)
    assert float(decoded.wind_v_ms[0, 0]) == pytest.approx(0, abs=1e-10)
    assert np.isnan(decoded.wind_u_ms[0, 1])
    assert float(decoded.pressure_hpa[0, 0]) == 1010
    source.P.attrs["standard_name"] = "air_pressure"
    with pytest.raises(ValueError, match="mean-sea-level"):
        normalize_stations(source)


def test_wind_speed_is_derived_after_component_interpolation() -> None:
    values = np.array([[[5.0, -5.0], [5.0, -5.0]]])
    ds = xr.Dataset(
        {
            "temperature_c": (("time", "latitude", "longitude"), values * 0 + 10),
            "precipitation_mm": (("time", "latitude", "longitude"), values * 0),
            "wind_u_ms": (("time", "latitude", "longitude"), values),
            "wind_v_ms": (("time", "latitude", "longitude"), values * 0),
            "pressure_hpa": (("time", "latitude", "longitude"), values * 0 + 1010),
        },
        coords={
            "time": [np.datetime64("2024-01-01T01")],
            "latitude": [53.0, 51.0],
            "longitude": [4.0, 6.0],
        },
        attrs={"initialization_utc": "2024-01-01T00:00:00"},
    )
    result = query_grid(ds, 52, 5, 1, datetime(2024, 1, 1, tzinfo=UTC))
    rows = result["hourly"]
    assert isinstance(rows, list)
    assert rows[0]["wind_speed_ms"] == 0
    assert rows[0]["pressure_hpa"] == 1010
    assert rows[0]["aurora_temperature_c"] == 10


def test_resource_budget_has_no_automatic_deletion(tmp_path: Path) -> None:
    data = tmp_path / "data/keep.bin"
    data.parent.mkdir()
    data.write_bytes(b"keep")
    with pytest.raises(ValueError, match="max_download"):
        check_budget(Config(root=tmp_path, max_download_gb=1e-9), Recorder(), 10)
    with pytest.raises(ValueError, match="max_storage"):
        check_budget(Config(root=tmp_path, max_storage_gb=1e-9), Recorder(), 0)
    assert data.read_bytes() == b"keep"


def test_near_realtime_wigos_and_documented_wind_schema() -> None:
    source = xr.Dataset(
        {
            "wsi": ("station", ["0-20000-0-06260"]),
            "ta": (("time", "station"), [[10.0]], {"units": "°C"}),
            "ff": (
                ("time", "station"),
                [[5.0]],
                {"units": "m/s", "long_name": "Wind Speed at 10 m Mean with MD"},
            ),
            "dd": (
                ("time", "station"),
                [[90.0]],
                {"units": "°", "long_name": "Wind Direction Mean with MD"},
            ),
        },
        coords={
            "station": ["06260"],
            "time": [np.datetime64("2024-01-01")],
            "latitude": ("station", [52.1]),
            "longitude": ("station", [5.2]),
        },
    )
    decoded = normalize_stations(source)
    assert decoded.station.values.tolist() == ["0-20000-0-06260"]
    assert float(decoded.wind_u_ms[0, 0]) == pytest.approx(-5)


def test_coordinate_changes_are_selected_within_observation_window() -> None:
    from aurora_finetuning.observations import station_geometry

    ds = xr.Dataset(
        coords={
            "time": [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")],
            "station": ["a"],
            "latitude": (("time", "station"), [[52.0], [53.0]]),
            "longitude": (("time", "station"), [[5.0], [6.0]]),
        }
    )
    assert float(station_geometry(ds.isel(time=slice(0, 1))).latitude[0]) == 52
    assert float(station_geometry(ds).latitude[0]) == 53


def test_station_decoding_preserves_record_availability() -> None:
    source = xr.Dataset(
        {"T": (("time", "station"), [[10.0]], {"units": "degC"})},
        coords={
            "time": [np.datetime64("2024-01-01")],
            "station": ["a"],
            "latitude": ("station", [52.0]),
            "longitude": ("station", [5.0]),
            "available_time": ("time", [np.datetime64("2024-01-02")]),
        },
    )
    decoded = normalize_stations(source)
    assert decoded.available_time.identical(source.available_time)
