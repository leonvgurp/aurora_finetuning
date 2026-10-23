"""Physical units, precipitation accumulation and local day boundaries."""

from datetime import date

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from aurora_finetuning.config import Config
from aurora_finetuning.grid import make_grid
from aurora_finetuning.observations import (
    aggregate_radar,
    hourly_radar,
    normalize_stations,
    summarize_history,
)


def test_station_units_are_not_legacy_tenths() -> None:
    ds = xr.Dataset(
        {
            "T": (("time", "station"), [[12.3]], {"units": "degC"}),
            "RH": (("time", "station"), [[1.2]], {"units": "mm"}),
        },
        coords={
            "time": [np.datetime64("2024-01-01T01")],
            "station": ["De Bilt"],
            "latitude": ("station", [52.1]),
            "longitude": ("station", [5.2]),
        },
    )
    normalized = normalize_stations(ds)
    assert normalized.temperature_c.item() == 12.3
    assert normalized.precipitation_mm.item() == 1.2
    ds.T.attrs["units"] = "tenths"
    with pytest.raises(ValueError, match="units"):
        normalize_stations(ds)


def test_radar_requires_complete_hour_and_preserves_nodata() -> None:
    times: NDArray[np.datetime64] = np.arange(
        np.datetime64("2024-01-01T00:05"),
        np.datetime64("2024-01-01T01:05"),
        np.timedelta64(5, "m"),
    )
    frames = xr.DataArray(
        np.ones((12, 2)), dims=("time", "cell"), coords={"time": times}
    )
    frames[0, 1] = np.nan
    total = hourly_radar(frames)
    assert total[0, 0] == 12
    assert np.isnan(total[0, 1])
    assert np.isnan(hourly_radar(frames[1:])[0, 0])
    with pytest.raises(ValueError, match="consecutive"):
        hourly_radar(frames.isel(time=[0, 2]))


def test_radar_aggregation_mean_not_sum() -> None:
    grid = make_grid(Config())
    lat = np.array([[grid.latitude[0], grid.latitude[0]]])
    lon = np.array([[grid.longitude[0] - 0.01, grid.longitude[0] + 0.01]])
    result = aggregate_radar(np.array([[2.0, 4.0]]), lat, lon, grid, 0.25)
    assert result[0, 0] == 3
    assert np.isnan(result[1, 1])


def test_history_dst_day_has_23_hours() -> None:
    times: NDArray[np.datetime64] = np.arange(
        np.datetime64("2024-03-28T00"),
        np.datetime64("2024-04-01T01"),
        np.timedelta64(1, "h"),
    )
    values = np.ones((len(times), 1))
    ds = xr.Dataset(
        {
            name: (("time", "station"), values.copy())
            for name in (
                "temperature_c",
                "sunshine_hours",
                "precipitation_hours",
                "precipitation_mm",
            )
        },
        coords={"time": times, "station": ["260"]},
    )
    rows = summarize_history(ds, "260", date(2024, 4, 1))
    assert rows[-1]["expected_hours"] == 23
    assert rows[-1]["precipitation_mm"] == 23


def test_documented_knmi_trace_is_not_missing() -> None:
    source = xr.Dataset(
        {
            "T": (("time", "station"), [[10.0]], {"units": "°C"}),
            "RH": (
                ("time", "station"),
                [[-1.0]],
                {
                    "units": "mm",
                    "description": "Hourly amount; -1 for <0.05 millimeters",
                },
            ),
        },
        coords={
            "time": [np.datetime64("2024-01-01T01")],
            "station": ["260"],
            "latitude": ("station", [52.1]),
            "longitude": ("station", [5.2]),
        },
    )
    normalized = normalize_stations(source)
    assert normalized.precipitation_mm.item() == 0
    assert "trace_handling" in normalized.precipitation_mm.attrs
