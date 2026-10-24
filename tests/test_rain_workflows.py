"""Public mode selection, split isolation, and sparse lead coverage regressions."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from aurora_finetuning import rain_calibration, weather
from aurora_finetuning.acquisition import sha256
from aurora_finetuning.config import Config
from aurora_finetuning.grid import FloatArray
from aurora_finetuning.leads import lead_schedule
from aurora_finetuning.radar_data import RadarUnavailable
from aurora_finetuning.radar_nowcast import apply_motion
from aurora_finetuning.radar_training import RadarCase
from aurora_finetuning.telemetry import Recorder, write_json


def test_sparse_schedule_requires_bridges_and_rejects_invalid_leads() -> None:
    assert lead_schedule(15, list(range(8, 16))) == [6, *range(8, 16)]
    assert lead_schedule(13, [5, 7, 13]) == [5, 6, 7, 12, 13]
    assert lead_schedule(6, [6]) == [6]
    for leads in ([], [0], [7], [1, 1], [True]):
        with pytest.raises(ValueError):
            lead_schedule(6, leads)


@pytest.mark.parametrize(
    "method,adapter,calibration",
    [
        ("cnn", None, None),
        ("aurora", Path("cnn.pt"), None),
        ("radar-motion", Path("cnn.pt"), None),
        ("radar-blend", None, None),
        ("aurora", None, Path("weights.json")),
    ],
)
def test_contradictory_modes_fail_before_acquisition(
    method: str,
    adapter: Path | None,
    calibration: Path | None,
) -> None:
    with pytest.raises(ValueError):
        weather.predict_weather(
            Config(),
            52,
            5,
            Recorder(),
            rain_method=method,
            radar_adapter=adapter,
            rain_calibration=calibration,
        )


def test_missing_radar_retains_aurora(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(*args: object, **kwargs: object) -> xr.Dataset:
        raise RadarUnavailable("missing history")

    monkeypatch.setattr("aurora_finetuning.radar_nowcast.live_history", missing)
    result: dict[str, object] = {
        "latitude": 52,
        "longitude": 5,
        "hourly": [{"time": "2024-01-01T13:00:00+00:00", "precipitation_mm": 2}],
    }
    out = apply_motion(result, Config(), Recorder(), datetime(2024, 1, 1, 12))
    assert out["hourly"] == [
        {
            "time": "2024-01-01T13:00:00+00:00",
            "precipitation_mm": 2,
            "precipitation_mm_source": "Aurora; radar unavailable: missing history",
        }
    ]


@pytest.mark.parametrize("count,known,expected", [(6, 3.0, 5.0), (0, 0.0, 8.0)])
def test_measured_prefix_and_unsupported_bins(
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    known: float,
    expected: float,
) -> None:
    totals = xr.Dataset(
        {
            "motion_mm": (("time", "y", "x"), [[[known]]]),
            "valid_bins": (("time", "y", "x"), [[[count]]]),
            "prefix_mm": (("time", "y", "x"), [[[2.0]]]),
            "future_bins": ("time", [6]),
        },
        coords={"latitude": (("y", "x"), [[52.0]]), "longitude": (("y", "x"), [[5.0]])},
        attrs={"radar_cutoff_utc": "2024-01-01T12:30:00"},
    )
    monkeypatch.setattr(
        "aurora_finetuning.radar_nowcast.live_history", lambda *a, **k: xr.Dataset()
    )
    monkeypatch.setattr(
        "aurora_finetuning.radar_nowcast.cached_totals", lambda *a: totals
    )
    row = {"time": "2024-01-01T13:00:00+00:00", "precipitation_mm": 12.0}
    result: dict[str, object] = {"latitude": 52, "longitude": 5, "hourly": [row]}
    apply_motion(result, Config(), Recorder(), datetime(2024, 1, 1, 12, 30))
    assert row["precipitation_mm"] == expected
    assert row["aurora_precipitation_mm"] == 12
    assert row["radar_coverage_fraction"] == count / 6


def test_calibration_only_opens_validation_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "cases.json"
    write_json(
        manifest,
        {
            split: [
                {
                    "initialization": day,
                    "forecast": "not-present.nc",
                    "radar": "not-present-radar.nc",
                }
            ]
            for split, day in [
                ("train", "2024-01-01T12:00:00"),
                ("validation", "2024-04-01T12:00:00"),
                ("test", "2024-07-01T12:00:00"),
            ]
        },
    )
    opened = []

    def arrays(
        case: RadarCase, config: Config, recorder: Recorder
    ) -> tuple[dict[str, FloatArray], FloatArray, xr.Dataset]:
        opened.append(case.initialization)
        target = np.ones((8, 2, 2))
        return (
            {
                "aurora": np.zeros_like(target),
                "motion_known": target,
                "motion_count": np.full_like(target, 12),
            },
            target,
            xr.Dataset(),
        )

    monkeypatch.setattr(rain_calibration, "case_arrays", arrays)
    output = tmp_path / "weights.json"
    result = rain_calibration.calibrate_rain(Config(), manifest, output, Recorder())
    assert opened == [datetime(2024, 4, 1, 12)]
    assert result["weights"] == [1] * 8
    assert rain_calibration.load_weights(output, Config()) == [1] * 8
    # A different evaluation manifest cannot relabel calibration cases as test.
    document = json.loads(output.read_text())
    document["cases"] = ["2024-07-01T12:00:00"]
    output.write_text(json.dumps(document))
    output.with_suffix(".sha256.json").write_text(
        json.dumps({"sha256": sha256(output)})
    )
    with pytest.raises(ValueError, match="overlap calibration"):
        rain_calibration.evaluate_rain(Config(), manifest, output, Recorder())


def test_sparse_cache_requires_all_requested_valid_times(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(weather, "cache_identity", lambda *args: "key")
    config = Config(root=tmp_path)
    cycle = datetime(2024, 1, 1)
    path = tmp_path / "outputs/weather/key/15-08.nc"
    path.parent.mkdir(parents=True)
    times = [np.datetime64(cycle + timedelta(hours=i), "ns") for i in range(8, 16)]
    xr.Dataset(coords={"time": times}).to_netcdf(path)
    write_json(
        path.with_suffix(".json"),
        {
            "sha256": sha256(path),
            "schema": 3,
            "lead_hours": list(range(8, 16)),
        },
    )
    assert weather.cached_grid(config, cycle, 15, Recorder()) is None
    assert (
        weather.cached_grid(config, cycle, 15, Recorder(), list(range(8, 16)))
        is not None
    )
    doc = json.loads(path.with_suffix(".json").read_text())
    doc["lead_hours"] = [7, *range(9, 16)]
    path.with_suffix(".json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="coverage mismatch"):
        weather.cached_grid(config, cycle, 15, Recorder(), [9])
