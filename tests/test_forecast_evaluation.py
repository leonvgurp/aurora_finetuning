"""Location interpolation, time coverage and independent scoring."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from aurora_finetuning.cli import main
from aurora_finetuning.evaluation import metrics
from aurora_finetuning.forecast import forecast, query_grid


def prediction() -> xr.Dataset:
    ds = xr.Dataset(
        {
            "temperature_c": (
                ("time", "latitude", "longitude"),
                np.array([[[10.0, 12.0], [14.0, 16.0]]]),
            ),
            "precipitation_mm": (
                ("time", "latitude", "longitude"),
                np.array([[[0.0, 1.0], [2.0, 3.0]]]),
            ),
        },
        coords={
            "time": [np.datetime64("2024-01-01T01")],
            "latitude": [53.0, 51.0],
            "longitude": [4.0, 6.0],
        },
        attrs={"initialization_utc": "2024-01-01T00:00:00"},
    )
    return ds


def test_interpolation_and_precipitation_interval() -> None:
    result = query_grid(prediction(), 52, 5, 1, datetime(2024, 1, 1, tzinfo=UTC))
    rows = result["hourly"]
    assert isinstance(rows, list)
    assert rows[0]["temperature_c"] == 13
    assert rows[0]["time"] == "2024-01-01T02:00:00+01:00"
    assert rows[0]["interval_start"] == "2024-01-01T01:00:00+01:00"
    with pytest.raises(ValueError, match="cover"):
        query_grid(prediction(), 52, 5, 2, datetime(2024, 1, 1, tzinfo=UTC))


def test_cached_python_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "forecast.nc"
    prediction().to_netcdf(path)
    result = forecast(
        52, 5, 1, cached_forecast=path, start=datetime(2024, 1, 1, tzinfo=UTC)
    )
    assert result["initialization_age_hours"] == 0
    assert (
        main(
            [
                "forecast",
                "--latitude",
                "52",
                "--longitude",
                "5",
                "--hours",
                "1",
                "--cached-forecast",
                str(path),
                "--start",
                "2024-01-01T00:00:00+00:00",
                "--report",
                str(tmp_path / "timings.json"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["latitude"] == 52
    assert json.loads((tmp_path / "timings.json").read_text())["success"]
    with pytest.raises(ValueError, match="stale"):
        forecast(52, 5, cached_forecast=path, start=datetime(2024, 2, 1, tzinfo=UTC))


def test_metrics_ignore_missing_pairs() -> None:
    scores = metrics(np.array([1.0, 3.0, np.nan]), np.array([2.0, 1.0, 5.0]))
    assert scores["count"] == 2
    assert scores["mae"] == 1.5
    assert scores["bias"] == 0.5


def test_precipitation_floor_keeps_raw_diagnostics() -> None:
    from aurora_finetuning.model import precipitation_output

    values, count, minimum = precipitation_output(np.array([[-0.02, 0.0, 1.5]]))
    assert values.tolist() == [[0.0, 0.0, 1.5]]
    assert count == 1
    assert minimum == -0.02
    with pytest.raises(ValueError, match="nonfinite"):
        precipitation_output(np.array([[np.nan]]))
