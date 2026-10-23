"""The explicit ERA5 mirror preserves physical cells and input times."""

from datetime import datetime

import numpy as np
import xarray as xr

from aurora_finetuning.acquisition import LEVELS
from aurora_finetuning.arco import select_pressure_field
from aurora_finetuning.config import Config
from aurora_finetuning.grid import make_grid


def test_pressure_selection_preserves_greenwich_order() -> None:
    grid = make_grid(Config())
    values = np.arange(
        2 * len(LEVELS) * grid.shape[0] * grid.shape[1], dtype=np.float32
    ).reshape(2, len(LEVELS), *grid.shape)
    source = xr.DataArray(
        values,
        dims=("time", "level", "latitude", "longitude"),
        coords={
            "time": np.array(
                ["2024-01-01T00", "2024-01-01T06"], dtype="datetime64[ns]"
            ),
            "level": list(LEVELS),
            "latitude": grid.latitude,
            "longitude": grid.longitude % 360,
        },
        attrs={"units": "K"},
    )
    selected = select_pressure_field(source, grid, [datetime(2024, 1, 1, 6)])
    assert np.array_equal(selected.values[0], values[1])
    assert np.array_equal(selected.longitude.values, grid.longitude)
    assert selected.attrs["units"] == "K"
