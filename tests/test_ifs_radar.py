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


def test_surface_height_coordinates_merge_without_conflicts() -> None:
    from aurora_finetuning.ifs import regional_grib_field

    fields = []
    for name, height in (("t2m", 2), ("u10", 10), ("u100", 100)):
        ds = xr.Dataset(
            {name: (("latitude", "longitude"), np.ones((2, 2)))},
            coords={
                "latitude": [53.0, 52.0],
                "longitude": [4.0, 5.0],
                "heightAboveGround": height,
                "time": np.datetime64("2024-01-01"),
            },
        )
        fields.append(regional_grib_field(ds))
    combined = xr.merge(fields, compat="no_conflicts")
    assert set(combined.data_vars) == {"t2m", "u10", "u100"}
    assert "heightAboveGround" not in combined.coords
    assert "time" in combined.coords


def test_soil_and_pressure_coordinate_normalization() -> None:
    from aurora_finetuning.ifs import regional_grib_field

    ds = xr.Dataset(
        {"sot": (("soilLayer", "latitude", "longitude"), np.ones((1, 2, 2)))},
        coords={"soilLayer": [1], "latitude": [53.0, 52.0], "longitude": [4.0, 5.0]},
    )
    assert regional_grib_field(ds).sot.dims == ("latitude", "longitude")
    with pytest.raises(ValueError, match="first layer"):
        regional_grib_field(xr.concat([ds, ds], dim="soilLayer"))
    atmos = ds.rename({"soilLayer": "isobaricInhPa"}).assign_coords(
        isobaricInhPa=[1000]
    )
    assert regional_grib_field(atmos).sot.dims == ("level", "latitude", "longitude")


def test_pressure_units_levels_and_completeness() -> None:
    from aurora_finetuning.acquisition import LEVELS
    from aurora_finetuning.ifs import validate_ifs_units

    ds = xr.Dataset(coords={"level": list(LEVELS)})
    units = {"t": "K", "u": "m/s", "v": "m/s", "q": "kg/kg", "z": "m2 s-2"}
    for name, unit in units.items():
        ds[name] = ("level", np.ones(len(LEVELS)), {"units": unit})
    validate_ifs_units(ds, "pressure")
    with pytest.raises(ValueError, match="pressure levels"):
        validate_ifs_units(ds.isel(level=slice(1, None)), "pressure")
    with pytest.raises(ValueError, match="missing required fields"):
        validate_ifs_units(ds.drop_vars("q"), "pressure")
    ds.z.attrs["units"] = "m"
    with pytest.raises(ValueError, match="Unexpected units"):
        validate_ifs_units(ds, "pressure")
