"""Explicit session lifetime for reusable, frozen regional Aurora inference."""

from datetime import datetime
from types import TracebackType

import torch
import xarray as xr
from aurora import AuroraV1p5

from aurora_finetuning.config import Config
from aurora_finetuning.model import infer, load_model
from aurora_finetuning.preparation import prepare_batch
from aurora_finetuning.regional import regional_context
from aurora_finetuning.telemetry import Recorder


class ForecastEngine:
    """Load Aurora lazily once; serial regional compatibility remains scoped."""

    def __init__(self, config: Config, recorder: Recorder) -> None:
        self.config = config
        self.recorder = recorder
        self._model: AuroraV1p5 | None = None
        self._closed = False

    def __enter__(self) -> "ForecastEngine":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def predict(self, initialization: datetime, hours: int, source: str) -> xr.Dataset:
        """Forecast from explicit current-state inputs, never external future fields."""
        if self._closed:
            raise ValueError("Forecast engine is closed")
        with regional_context():
            batch = prepare_batch(self.config, initialization, self.recorder, source)
            if self._model is None:
                self._model = load_model(self.config, self.recorder)
                self._model.requires_grad_(False)
            return infer(self._model, batch, hours, self.config, self.recorder)

    def close(self) -> None:
        """Release model references and accelerator caches at session end."""
        self._model = None
        self._closed = True
        self.recorder.synchronize = lambda: None
        self.recorder.accelerator_memory = lambda: (None, None)
        if self.config.device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
