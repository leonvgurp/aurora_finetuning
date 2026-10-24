"""External station calibration, chronological features and geographic transfer."""

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from aurora_finetuning.config import Config
from aurora_finetuning.seasonal import calendar_features
from aurora_finetuning.stations import (
    FEATURES,
    apply_station,
    feature_row,
    fit_ridge,
    load_calibrator,
    read_cases,
    recent_features,
    train_station,
)
from aurora_finetuning.telemetry import Recorder


def observations(cutoff: datetime) -> xr.Dataset:
    times = [
        np.datetime64(cutoff - timedelta(hours=h), "ns") for h in range(72, -1, -1)
    ]
    return xr.Dataset(
        {
            "temperature_c": (
                ("time", "station"),
                np.tile(np.arange(73)[:, None] / 10, (1, 3)),
            )
        },
        coords={
            "time": times,
            "station": ["a", "b", "far"],
            "latitude": ("station", [52.0, 52.1, 54.0]),
            "longitude": ("station", [5.0, 5.1, 8.0]),
        },
    )


def archive(cutoff: datetime, availability: datetime) -> xr.Dataset:
    times = [
        np.datetime64(cutoff - timedelta(hours=h), "ns") for h in range(72, -1, -1)
    ]
    return xr.Dataset(
        {"temperature_c": (("time", "latitude", "longitude"), np.zeros((73, 2, 2)))},
        coords={"time": times, "latitude": [53.0, 51.0], "longitude": [4.0, 6.0]},
        attrs={
            "initialization_utc": (cutoff - timedelta(hours=80)).isoformat(),
            "available_utc": availability.isoformat(),
        },
    )


def test_future_records_and_future_forecasts_cannot_change_features() -> None:
    cutoff = datetime(2024, 1, 4)
    ds = observations(cutoff)
    old = archive(cutoff, cutoff - timedelta(hours=80))
    future = archive(cutoff, cutoff + timedelta(hours=1))
    future.temperature_c[:] = 999
    a, usable = recent_features(ds, [old], "temperature_c", cutoff, 52, 5, 60)
    b, _ = recent_features(ds, [old, future], "temperature_c", cutoff, 52, 5, 60)
    assert usable and np.allclose(a, b)
    assert a[0] == pytest.approx(7.2)
    later = observations(cutoff + timedelta(hours=1)).isel(time=[-1])
    later.temperature_c[:] = 999
    c, _ = recent_features(
        xr.concat([ds, later], dim="time"), [old], "temperature_c", cutoff, 52, 5, 60
    )
    assert np.allclose(a, c)
    ds = ds.assign_coords(
        available_time=(
            "time",
            np.full(73, np.datetime64(cutoff + timedelta(hours=1), "ns")),
        )
    )
    assert not recent_features(ds, [old], "temperature_c", cutoff, 52, 5, 60)[1]


def test_exact_station_distance_and_missing_history() -> None:
    cutoff = datetime(2024, 1, 4)
    ds = observations(cutoff)
    summary, usable = recent_features(ds, [], "temperature_c", cutoff, 52, 5, 60)
    assert usable and summary[6] == pytest.approx(0)
    assert summary[8] == 1  # No eligible previous forecast residual.
    assert not recent_features(ds, [], "pressure_hpa", cutoff, 52, 5, 60)[1]
    assert not recent_features(
        ds, [], "temperature_c", cutoff + timedelta(hours=2), 52, 5, 60
    )[1]
    assert not recent_features(ds, [], "temperature_c", cutoff, 50.7, 3.2, 60)[1]


def test_calendar_is_continuous_and_utc() -> None:
    a = np.array(calendar_features(datetime(2024, 12, 31, 23, 59, tzinfo=UTC)))
    b = np.array(calendar_features(datetime(2025, 1, 1, tzinfo=UTC)))
    assert np.linalg.norm(a - b) < 0.01
    assert calendar_features(
        datetime.fromisoformat("2024-10-27T02:00:00+02:00")
    ) == calendar_features(datetime(2024, 10, 27, tzinfo=UTC))


def test_ridge_learns_residual_and_training_never_imports_torch(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(40, len(FEATURES)))
    y = 2 + x[:, 0] * 0.5
    model = fit_ridge(x, y, 0.1, False)
    assert np.abs(model.predict(x) - y).mean() < 0.01
    result = subprocess.run(
        [
            "pixi",
            "run",
            "python",
            "-c",
            (
                "import sys; import aurora_finetuning.stations; "
                "assert 'torch' not in sys.modules; assert 'aurora' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_saved_calibrator_blending_and_checksum(tmp_path: Path) -> None:
    config = Config(root=tmp_path, device="cpu")
    directory = tmp_path / "prepared"
    directory.mkdir()
    cutoff = datetime(2024, 1, 4)
    history = observations(cutoff)
    summary, _ = recent_features(history, [], "temperature_c", cutoff, 52, 5, 60)
    rows = np.stack(
        [
            feature_row(
                float(i), i % 8 + 1, cutoff + timedelta(hours=i % 8 + 1), 52, 5, summary
            )
            for i in range(24)
        ]
    )
    for split in ("train", "validation"):
        xr.Dataset(
            {
                "features": (("sample", "feature"), rows),
                "target": ("sample", rows[:, 0] + 2),
                "variable": ("sample", ["temperature_c"] * 24),
                "lead": ("sample", np.arange(24) % 8 + 1),
            },
            coords={"feature": list(FEATURES)},
        ).to_netcdf(directory / f"{split}.nc")
    path = tmp_path / "station.json"
    train_station(config, directory, path, Recorder())
    assert load_calibrator(path, config)["temperature_c"][1].tolist() == [1] * 8
    result: dict[str, object] = {
        "latitude": 52.0,
        "longitude": 5.0,
        "hourly": [{"time": "2024-01-04T01:00:00+00:00", "temperature_c": 10.0}],
    }
    adjusted = apply_station(result, history, [], path, config, cutoff)
    rows_result = adjusted["hourly"]
    assert isinstance(rows_result, list)
    assert rows_result[0]["temperature_c"] == pytest.approx(12)
    assert rows_result[0]["aurora_temperature_c"] == 10
    path.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_calibrator(path, config)


def test_station_windows_include_72h_history_and_eight_hour_targets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cases.json"
    template = {
        "forecast": "f.nc",
        "observations": ["o.nc"],
        "targets": ["t.nc"],
        "history_forecasts": [],
    }
    path.write_text(
        json.dumps(
            {
                split: [{**template, "issuance": f"2024-01-{day:02d}T00:00:00+00:00"}]
                for split, day in (("train", 1), ("validation", 4), ("test", 10))
            }
        )
    )
    with pytest.raises(ValueError, match="30h"):
        read_cases(path)


def test_prepare_train_evaluate_with_geographically_heldout_station(
    tmp_path: Path,
) -> None:
    from aurora_finetuning.stations import evaluate_station, prepare_examples

    config = Config(root=tmp_path, device="cpu")
    cases = {}
    for split, day in (("train", 4), ("validation", 10), ("test", 16)):
        cutoff = datetime(2024, 1, day)
        ds = observations(cutoff)
        raw = ds.rename({"temperature_c": "T"})
        raw.T[:] = 10
        raw.T.attrs["units"] = "degC"
        history_path = tmp_path / f"{split}-observations.nc"
        raw.to_netcdf(history_path)
        future = raw.isel(time=slice(0, 8)).assign_coords(
            time=[np.datetime64(cutoff + timedelta(hours=h), "ns") for h in range(1, 9)]
        )
        future.T[:] = 11
        targets = tmp_path / f"{split}-targets.nc"
        future.to_netcdf(targets)
        forecast = archive(cutoff + timedelta(hours=8), cutoff - timedelta(hours=80))
        forecast.temperature_c[:] = 10
        forecast.attrs["checkpoint_revision"] = config.checkpoint_revision
        forecast.attrs["resolution"] = config.resolution
        forecast.attrs["initialization_utc"] = cutoff.isoformat()
        forecast.attrs["available_utc"] = cutoff.isoformat()
        forecast_path = tmp_path / f"{split}-forecast.nc"
        forecast.to_netcdf(forecast_path)
        cases[split] = [
            {
                "issuance": cutoff.replace(tzinfo=UTC).isoformat(),
                "forecast": forecast_path.name,
                "observations": [history_path.name],
                "targets": [targets.name],
                "history_forecasts": [],
            }
        ]
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps(cases))
    prepared = tmp_path / "prepared"
    recorder = Recorder()
    prepare_examples(config, manifest, prepared, recorder, frozenset({"a"}))
    assert not any(Path(p).name.startswith("test-") for p in recorder.inputs)
    with xr.open_dataset(prepared / "train.nc") as dataset:
        assert "a" not in dataset.station_id.values
    calibrator = tmp_path / "calibration.json"
    train_station(config, prepared, calibrator, recorder)
    report = evaluate_station(config, manifest, calibrator, recorder, frozenset({"a"}))
    assert report["heldout_stations"] == ["a"]
    test = report["test"]
    assert isinstance(test, list) and len(test[0]["by_lead_hour"]) == 8
    assert test[0]["by_lead_hour"][0]["combined"]["mae"] < 0.01
    with pytest.raises(ValueError, match="excluded during"):
        evaluate_station(config, manifest, calibrator, recorder, frozenset({"b"}))
