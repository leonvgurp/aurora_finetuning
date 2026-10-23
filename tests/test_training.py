"""Differentiable KNMI local supervision and physical loss masks."""

from datetime import datetime

import numpy as np
import pytest
import torch
import xarray as xr

from aurora_finetuning.config import Config
from aurora_finetuning.grid import make_grid
from aurora_finetuning.training import physical_loss, station_loss


def test_station_supervision_backpropagates_to_grid() -> None:
    config = Config(device="cpu")
    grid = make_grid(config)
    temperature = torch.full(grid.shape, 10.0, requires_grad=True)
    rainfall = torch.full(grid.shape, 1.0, requires_grad=True)
    stations = xr.Dataset(
        {
            "temperature_c": (("time", "station"), [[12.0]]),
            "precipitation_mm": (("time", "station"), [[2.0]]),
        },
        coords={
            "time": [np.datetime64("2024-01-01T01")],
            "station": ["260"],
            "latitude": ("station", [52.1]),
            "longitude": ("station", [5.2]),
        },
    )
    loss = station_loss(
        temperature, rainfall, stations, datetime(2024, 1, 1, 1), config
    )
    torch.autograd.backward(loss)
    assert temperature.grad is not None and temperature.grad.abs().sum() > 0
    assert rainfall.grad is not None and rainfall.grad.abs().sum() > 0


def test_loss_rejects_empty_supervision() -> None:
    with pytest.raises(ValueError, match="valid target"):
        physical_loss(
            torch.zeros(2),
            torch.zeros(2),
            torch.full((2,), float("nan")),
            torch.zeros(2),
            Config(),
        )
