"""Explicit opt-in tests exercising a real, tiny Aurora architecture."""

import pytest

from aurora_finetuning.smoke import architecture_smoke
from aurora_finetuning.telemetry import Recorder


@pytest.mark.integration
def test_tiny_aurora_forward_backward() -> None:
    result = architecture_smoke("cpu", Recorder())
    assert result["finite"] is True
