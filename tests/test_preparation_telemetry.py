"""Preprocessing failures and timing evidence are deterministic and offline."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from aurora_finetuning.config import Config
from aurora_finetuning.grid import make_grid
from aurora_finetuning.preparation import canonical, finite_tensor, open_era5, subset
from aurora_finetuning.telemetry import Recorder, write_json


def test_greenwich_sort_keeps_values_attached() -> None:
    ds = xr.Dataset(
        {"value": (("latitude", "longitude"), [[1.0, 2.0, 3.0]])},
        coords={"latitude": [52.0], "longitude": [0.0, 1.0, 359.0]},
    )
    result = canonical(ds)
    assert list(result.longitude.values) == [-1.0, 0.0, 1.0]
    assert list(result.value.values[0]) == [3.0, 1.0, 2.0]
    with pytest.raises(ValueError, match="match"):
        subset(ds, make_grid(Config()))


def test_nonfinite_atmosphere_fails() -> None:
    with pytest.raises(ValueError, match="nonfinite"):
        finite_tensor(xr.DataArray([1.0, np.nan], name="t"))


@pytest.mark.parametrize("kind", ["single", "pressure"])
def test_overlapping_cache_times_are_loaded_once(tmp_path: Path, kind: str) -> None:
    """Overlapping pulls preserve the first cached state and fill later times."""
    from aurora_finetuning.acquisition import sha256

    grid = make_grid(Config())
    recorder = Recorder()
    for suffix, hours, value in (("a", [0, 6], 1.0), ("b", [6, 12], 2.0)):
        path = tmp_path / f"2024-01-01-{kind}-{suffix}.nc"
        dataset = xr.Dataset(
            {
                "t": (
                    ("time", "latitude", "longitude"),
                    np.full((2, *grid.shape), value),
                )
            },
            coords={
                "time": [np.datetime64(datetime(2024, 1, 1, h), "ns") for h in hours],
                "latitude": grid.latitude,
                "longitude": grid.longitude,
            },
        )
        dataset.to_netcdf(path)
        write_json(path.with_suffix(".json"), {"sha256": sha256(path)})
    wanted = [datetime(2024, 1, 1, h) for h in (6, 12)]
    result = open_era5(tmp_path, grid, wanted, kind, recorder)
    assert result.sizes["time"] == 2
    assert result.indexes["time"].is_unique
    np.testing.assert_array_equal(
        result.time.values, np.array(wanted, dtype="datetime64[ns]")
    )
    assert np.all(result.t.values[0] == 1.0)
    assert np.all(result.t.values[1] == 2.0)
    assert len(recorder.inputs) == 2


def test_timing_failure_and_exclusive_artifact(tmp_path: Path) -> None:
    recorder = Recorder()
    with pytest.raises(ValueError), recorder.measure("failing.stage"):
        raise ValueError("expected")
    assert recorder.timings[0].stage == "failing.stage"
    assert not recorder.timings[0].success
    assert recorder.timings[0].seconds >= 0
    path = tmp_path / "report.json"
    write_json(path, recorder.report())
    with pytest.raises(FileExistsError):
        write_json(path, {})


def test_regional_netcdf_preparation_and_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import datetime

    import torch

    from aurora_finetuning import preparation
    from aurora_finetuning.acquisition import ATMOSPHERE, LEVELS, SURFACE, sha256
    from aurora_finetuning.regional import regional_context

    config = Config(root=tmp_path, device="cpu")
    grid = make_grid(config)
    directory = tmp_path / "data/raw/era5"
    directory.mkdir(parents=True)
    time = np.array(
        ["2024-01-01T00", "2024-01-01T06", "2024-01-01T07"], dtype="datetime64[ns]"
    )
    dimensions = ("valid_time", "latitude", "longitude")
    values = np.zeros((3, *grid.shape), dtype=np.float32)
    surface = xr.Dataset(
        {name: (dimensions, values.copy()) for name in (*SURFACE, "tp")},
        coords={
            "valid_time": time,
            "latitude": grid.latitude,
            "longitude": grid.longitude % 360,
        },
    )
    for name in ("t2m", "d2m", "skt", "stl1"):
        surface[name][:] = 280
        surface[name].attrs["units"] = "K"
    for name in ("msl", "sp"):
        surface[name].attrs["units"] = "Pa"
    for name in ("u10", "v10", "u100", "v100"):
        surface[name].attrs["units"] = "m s**-1"
    surface["tp"][:] = 0.001
    surface["siconc"][:, 0, 0] = np.nan
    surface["tp"].attrs["units"] = "m"
    atmosphere = xr.Dataset(
        {
            name: (
                ("valid_time", "pressure_level", "latitude", "longitude"),
                np.zeros((3, len(LEVELS), *grid.shape), dtype=np.float32),
            )
            for name in ATMOSPHERE
        },
        coords={
            "valid_time": time,
            "pressure_level": list(LEVELS),
            "latitude": grid.latitude,
            "longitude": grid.longitude % 360,
        },
    )
    for name, unit in (
        ("t", "K"),
        ("q", "kg kg**-1"),
        ("u", "m s**-1"),
        ("v", "m s**-1"),
        ("z", "m**2 s**-2"),
    ):
        atmosphere[name].attrs["units"] = unit
    for kind, ds in (("single", surface), ("pressure", atmosphere)):
        path = directory / f"2024-01-01-{kind}-fixture.nc"
        ds.to_netcdf(path)
        write_json(path.with_suffix(".json"), {"sha256": sha256(path)})

    def static_fields(settings: Config) -> dict[str, torch.Tensor]:
        land = torch.ones(grid.shape)
        land[0, 0] = 0.3
        return {"lsm": land}

    # Isolate official static-asset disk access; exercise actual NetCDF transforms.
    monkeypatch.setattr(preparation, "static_fields", static_fields)
    recorder = Recorder()
    with regional_context():
        batch = preparation.prepare_batch(config, datetime(2024, 1, 1, 6), recorder)
    assert batch.spatial_shape == grid.shape
    assert batch.surf_vars["2t"].shape == (1, 2, *grid.shape)
    assert torch.all(batch.surf_vars["2t"] == 280)
    assert torch.all(batch.surf_vars["ci"][:, :, 0, 0] == 0)
    assert len(recorder.inputs) == 2
    target = preparation.target_fields(config, datetime(2024, 1, 1, 7))
    assert np.allclose(target.precipitation_mm.values, 1)
    assert np.allclose(target.temperature_c.values, 6.85, atol=0.001)
