"""Real optical-flow/advection and strict blending/caching behavior."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from aurora_finetuning.acquisition import sha256
from aurora_finetuning.config import Config
from aurora_finetuning.radar_nowcast import (
    cached_totals,
    estimate_motion,
    extrapolated_bins,
    motion_totals,
)
from aurora_finetuning.rain_calibration import (
    blend,
    choose_weight,
    compatibility,
    load_weights,
)
from aurora_finetuning.telemetry import Recorder, write_json


def radar_dataset(cutoff: datetime, frames: int = 24, size: int = 40) -> xr.Dataset:
    times = [
        np.datetime64(cutoff + timedelta(minutes=5 * (i - 23)), "ns")
        for i in range(frames)
    ]
    y, x = np.mgrid[:size, :size]
    data = np.zeros((frames, size, size), dtype=float)
    data[:, 12:20, 12:20] = 0.1
    return xr.Dataset(
        {
            "precipitation_mm": (
                ("time", "y", "x"),
                data,
                {"units": "mm", "interval_minutes": 5},
            )
        },
        coords={
            "time": times,
            "y": np.arange(size),
            "x": np.arange(size),
            "latitude": (("y", "x"), 52 + y * 0.009),
            "longitude": (("y", "x"), 5 + x * 0.014),
        },
        attrs={
            "source": "KNMI measured radar",
            "radar_role": "observations",
            "radar_product": "RTCOR-5m",
            "pixel_size_km": 1.0,
        },
    )


def test_actual_optical_flow_tracks_eastward_rain() -> None:
    history = np.zeros((24, 80, 80), dtype=float)
    for i in range(24):
        history[i, 25:40, 10 + i : 25 + i] = 0.5
    velocity = estimate_motion(history)
    assert np.median(velocity[0, 25:40, 35:45]) == pytest.approx(1, abs=0.15)
    assert abs(np.median(velocity[1])) < 0.15


def test_advection_direction_missing_mask_and_domain_exit() -> None:
    latest = np.zeros((24, 24))
    latest[8:12, 8:12] = 1
    latest[2, 2] = np.nan
    velocity = np.zeros((2, 24, 24))
    velocity[0] = 1
    frames = extrapolated_bins(latest, velocity)
    first = next(frames)
    np.testing.assert_allclose(first[8:12, 9:13], 1)
    assert np.isnan(first[:, 0]).all() and np.isnan(first[2, 3])
    last = first
    for frame in frames:
        last = frame
    assert np.isnan(last).all()


def test_dry_valid_history_is_not_missing() -> None:
    raw = np.zeros((24, 24, 24))
    assert np.array_equal(estimate_motion(raw), np.zeros((2, 24, 24)))
    with pytest.raises(ValueError, match="no valid"):
        estimate_motion(np.full_like(raw, np.nan))


def test_hour_boundaries_prefix_and_future_exclusion() -> None:
    cutoff = datetime(2024, 1, 1, 12, 30)
    history = radar_dataset(cutoff)
    ends = [datetime(2024, 1, 1, 13) + timedelta(hours=i) for i in range(8)]
    velocity = np.zeros((2, 40, 40))
    a = motion_totals(history, cutoff, ends, velocity)
    assert a.prefix_mm.values[0, 15, 15] == pytest.approx(0.6)
    assert a.valid_bins.values[0, 15, 15] == 6
    assert a.future_bins.values.tolist() == [6, 12, 12, 12, 12, 12, 12, 12]
    assert a.valid_bins.values[-1, 15, 15] == 12
    late = motion_totals(history, cutoff, [datetime(2024, 1, 1, 21)], velocity)
    assert late.valid_bins.values[0, 15, 15] == 6
    b = radar_dataset(cutoff, 120)
    b.precipitation_mm.values[24:] = 999
    result = motion_totals(b, cutoff, ends, velocity)
    xr.testing.assert_equal(a, result)


def test_complete_eight_hour_totals() -> None:
    cutoff = datetime(2024, 1, 1, 12)
    history = radar_dataset(cutoff)
    ends = [cutoff + timedelta(hours=i) for i in range(1, 9)]
    result = motion_totals(history, cutoff, ends, np.zeros((2, 40, 40)))
    assert result.motion_mm.values[:, 15, 15].sum() == pytest.approx(9.6)
    assert (result.valid_bins.values[:, 15, 15] == 12).all()


def test_blend_weights_require_gain_and_event_skill() -> None:
    base = np.zeros(4)
    target = np.ones(4)
    weight, _ = choose_weight(base, target, np.full(4, 12), target)
    assert weight == 1
    weight, _ = choose_weight(target, np.zeros(4), np.full(4, 12), target)
    assert weight == 0
    assert np.array_equal(
        blend(np.ones((8, 2, 2)), np.zeros((8, 2, 2)), np.zeros((8, 2, 2)), [1] * 8),
        np.ones((8, 2, 2)),
    )


def test_motion_cache_reuse_and_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cutoff = datetime(2024, 1, 1, 12)
    history = radar_dataset(cutoff)
    ends = [cutoff + timedelta(hours=i) for i in range(1, 9)]
    c = Config(root=tmp_path, device="cpu")
    r = Recorder()
    a = cached_totals(history, ends, c, r)

    def forbidden(*args: object) -> None:
        raise AssertionError("cache recomputed motion")

    monkeypatch.setattr("aurora_finetuning.radar_nowcast.estimate_motion", forbidden)
    b = cached_totals(history, ends, c, Recorder())
    xr.testing.assert_equal(a, b)
    next((tmp_path / "data/cache/radar-motion").glob("*velocity.nc")).write_bytes(
        b"broken"
    )
    with pytest.raises(ValueError, match="cache"):
        cached_totals(history, ends, c, Recorder())


def test_calibration_checksum_and_compatibility(tmp_path: Path) -> None:
    c = Config(root=tmp_path)
    p = tmp_path / "blend.json"
    write_json(
        p,
        {
            "version": "radar-motion-v1",
            "compatibility": compatibility(c),
            "weights": [0] * 8,
        },
    )
    write_json(p.with_suffix(".sha256.json"), {"sha256": sha256(p)})
    assert load_weights(p, c) == [0] * 8
    p.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_weights(p, c)
