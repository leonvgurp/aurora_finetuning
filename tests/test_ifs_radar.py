"""Explicit sea-ice supplementation and calibrated radar decoding."""

from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr

from aurora_finetuning.config import Config
from aurora_finetuning.grid import make_grid
from aurora_finetuning.ifs import sea_ice_supplement
from aurora_finetuning.radar import decode_frame


def test_sea_ice_age_and_domain_checked(tmp_path: Path) -> None:
    config = Config()
    grid = make_grid(config)
    path = tmp_path / "ice.nc"
    xr.Dataset(
        {"siconc": (("time", "latitude", "longitude"), np.zeros((1, *grid.shape)))},
        coords={
            "time": [np.datetime64("2024-01-01")],
            "latitude": grid.latitude,
            "longitude": grid.longitude,
        },
    ).to_netcdf(path)
    result = sea_ice_supplement(path, datetime(2024, 1, 6), config)
    assert result.attrs["supplement_age_hours"] == 120
    assert "time" not in result.coords
    with pytest.raises(ValueError, match="seven days"):
        sea_ice_supplement(path, datetime(2024, 1, 9), config)
    with pytest.raises(ValueError, match="before initialization"):
        sea_ice_supplement(path, datetime(2023, 12, 31), config)


def test_radar_calibration_and_nodata(tmp_path: Path) -> None:
    path = tmp_path / "radar.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "image1/image_data", data=np.array([[100, 65535]], dtype="uint16")
        )
        calibration = handle.create_group("image1/calibration")
        calibration.attrs["calibration_formulas"] = "GEO = 0.01 * PV + 0.0"
        calibration.attrs["calibration_out_of_image"] = 65535
        overview = handle.create_group("overview")
        overview.attrs["product_datetime_end"] = "01-JAN-2024;01:00:00.000"
    time, values = decode_frame(path)
    assert time == datetime(2024, 1, 1, 1)
    assert values[0, 0] == 1
    assert np.isnan(values[0, 1])
