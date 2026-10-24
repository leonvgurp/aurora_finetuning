"""Session reuse and cache hits avoid Aurora loads and input tensor construction."""

from datetime import datetime

import pytest
import xarray as xr

from aurora_finetuning.config import Config
from aurora_finetuning.engine import ForecastEngine
from aurora_finetuning.telemetry import Recorder


def test_engine_loads_once_and_rejects_use_after_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads = []

    class FrozenModel:
        def requires_grad_(self, enabled: bool) -> None:
            assert enabled is False

    def load(*args: object) -> FrozenModel:
        loads.append(1)
        return FrozenModel()

    monkeypatch.setattr("aurora_finetuning.engine.load_model", load)
    monkeypatch.setattr("aurora_finetuning.engine.prepare_batch", lambda *args: None)
    monkeypatch.setattr("aurora_finetuning.engine.infer", lambda *args: xr.Dataset())
    with ForecastEngine(Config(device="cpu"), Recorder()) as engine:
        engine.predict(datetime(2024, 1, 1), 8, "era5")
        engine.predict(datetime(2024, 1, 2), 8, "era5")
    assert loads == [1]
    with pytest.raises(ValueError, match="closed"):
        engine.predict(datetime(2024, 1, 1), 8, "era5")
