"""Measured radar conditioning, leakage guards and real small-model training."""

import json
import warnings
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import xarray as xr
from pyproj import Transformer

from aurora_finetuning.config import Config
from aurora_finetuning.forecast import query_grid
from aurora_finetuning.radar_data import (
    history_at,
    measured_targets,
    native_radar,
    radar_geometry,
    radar_time,
    validate_native,
)
from aurora_finetuning.radar_model import (
    RadarRainAdapter,
    baseline_at,
    feature_tensors,
    load_radar_adapter,
    radar_loss,
    refine_location,
)
from aurora_finetuning.radar_training import (
    evaluate_adapter,
    read_radar_cases,
    train_radar,
)
from aurora_finetuning.telemetry import Recorder

PROJECTION = (
    "+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 "
    "+a=6378137 +b=6356752 +x_0=0 +y_0=0 +units=km"
)


@pytest.fixture(autouse=True)
def single_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observed(cutoff: datetime, size: int = 128, future: bool = True) -> xr.Dataset:
    xx, yy = Transformer.from_crs("EPSG:4326", PROJECTION, always_xy=True).transform(
        5.4, 52.1
    )
    x = xx + np.arange(size) - size // 2
    y = yy - np.arange(size) + size // 2
    gx, gy = np.meshgrid(x, y)
    lon, lat = Transformer.from_crs(PROJECTION, "EPSG:4326", always_xy=True).transform(
        gx, gy
    )
    times = [
        np.datetime64(cutoff + timedelta(minutes=5 * i), "ns")
        for i in range(-23, 97 if future else 1)
    ]
    values = np.full((len(times), size, size), 0.1, dtype=np.float32)
    return xr.Dataset(
        {
            "precipitation_mm": (
                ("time", "y", "x"),
                values,
                {"units": "mm", "interval_minutes": 5},
            )
        },
        coords={
            "time": times,
            "y": y,
            "x": x,
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), lon),
        },
        attrs={
            "radar_role": "observations",
            "source": "KNMI measured radar",
            "radar_product": "RTCOR-5m",
            "pixel_size_km": 1.0,
            "projection": PROJECTION,
        },
    )


def baseline(cutoff: datetime, config: Config, hours: int = 8) -> xr.Dataset:
    init = cutoff.replace(minute=0, second=0, microsecond=0)
    shape = (hours, 2, 2)
    return xr.Dataset(
        {
            "precipitation_mm": (
                ("time", "latitude", "longitude"),
                np.full(shape, 1.2),
                {"units": "mm"},
            ),
            "temperature_c": (("time", "latitude", "longitude"), np.full(shape, 10.0)),
        },
        coords={
            "time": [
                np.datetime64(init + timedelta(hours=h), "ns")
                for h in range(1, hours + 1)
            ],
            "latitude": [53.5, 50.5],
            "longitude": [3.0, 8.0],
        },
        attrs={
            "initialization_utc": init.isoformat(),
            "checkpoint_revision": config.checkpoint_revision,
            "resolution": config.resolution,
        },
    )


def raw_file(path: Path, stamp: datetime) -> None:
    size = 132
    template = observed(stamp, size, False)
    with h5py.File(path, "w") as handle:
        geo = handle.create_group("geographic")
        geo.attrs.update(
            {
                "geo_pixel_def": "LU",
                "geo_dim_pixel": "KM,KM",
                "geo_pixel_size_x": [1.0],
                "geo_pixel_size_y": [-1.0],
                "geo_number_columns": [size],
                "geo_number_rows": [size],
                "geo_column_offset": [float(template.x[0]) - 0.5],
                "geo_row_offset": [-float(template.y[0]) - 0.5],
            }
        )
        geo.create_group("map_projection").attrs["projection_proj4_params"] = PROJECTION
        overview = handle.create_group("overview")
        overview.attrs["number_image_groups"] = [2]
        overview.attrs["product_datetime_end"] = stamp.strftime(
            "%d-%b-%Y;%H:%M:%S.000"
        ).upper()
        overview.attrs["product_datetime_start"] = (
            (stamp - timedelta(minutes=5)).strftime("%d-%b-%Y;%H:%M:%S.000").upper()
        )
        image = handle.create_group("image1")
        image.attrs["image_geo_parameter"] = "PRECIP_[MM]"
        image.create_dataset(
            "image_data", data=np.full((size, size), 10, dtype=np.uint16)
        )
        calibration = image.create_group("calibration")
        calibration.attrs.update(
            {
                "calibration_formulas": "GEO=0.01*PV+0.0",
                "calibration_missing_data": 65534,
                "calibration_out_of_image": 65535,
            }
        )
        handle.create_group("image2").attrs["image_geo_parameter"] = "QUALITY_[-]"


def test_native_projection_and_quality_images(tmp_path: Path) -> None:
    time = datetime(2024, 1, 1)
    one = tmp_path / "RAD_NL25_RAC_RT_202401010000.h5"
    two = tmp_path / "RAD_NL25_RAC_RT_202401010005.h5"
    raw_file(one, time)
    raw_file(two, time + timedelta(minutes=5))
    assert radar_time(one) == time
    geometry = radar_geometry(one)
    assert float(geometry.latitude[66, 66]) == pytest.approx(52.1)
    assert float(geometry.longitude[66, 66]) == pytest.approx(5.4)
    recorder = Recorder()
    dataset = native_radar([two, one], recorder, latitude=52.1, longitude=5.4, size=128)
    validate_native(dataset)
    assert dataset.precipitation_mm.shape == (2, 128, 128)
    assert np.allclose(dataset.precipitation_mm, 0.1)
    assert len(recorder.inputs) == 2
    with h5py.File(one, "a") as handle:
        handle["image2"].attrs["image_geo_parameter"] = "PRECIP_[MM]"
    with pytest.raises(ValueError, match="not radar forecasts"):
        radar_time(one)


def test_radar_missing_sentinel_preserved(tmp_path: Path) -> None:
    from aurora_finetuning.radar import decode_frame

    path = tmp_path / "observed.h5"
    raw_file(path, datetime(2024, 1, 1))
    with h5py.File(path, "a") as handle:
        handle["image1/image_data"][0, :2] = [65534, 65535]
    _, values = decode_frame(path)
    assert np.isnan(values[0, :2]).all()


def test_cutoff_and_future_targets_are_separate() -> None:
    cutoff = datetime(2024, 1, 1)
    ds = observed(cutoff)
    original = history_at(ds, cutoff).copy(deep=True)
    ds.precipitation_mm[24:] = 999
    assert history_at(ds, cutoff).identical(original)
    assert float(measured_targets(ds, cutoff, 1)[0, 0]) == pytest.approx(12 * 999)
    assert original.time.max() == np.datetime64(cutoff)
    with pytest.raises(ValueError, match="Missing measured radar history"):
        history_at(ds.isel(time=slice(1, None)), cutoff)
    with pytest.raises(ValueError, match="consecutive"):
        validate_native(ds.isel(time=[0, 2]))
    with pytest.raises(ValueError, match="not a forecast"):
        validate_native(ds.assign_attrs(radar_role="forecast"))


def test_model_uses_radar_input_and_backpropagates() -> None:
    torch.manual_seed(42)
    cutoff = datetime(2024, 1, 1)
    config = Config(device="cpu", radar_patch_size=128)
    ds = observed(cutoff)
    prior = baseline_at(baseline(cutoff, config), ds, cutoff + timedelta(minutes=5))
    features, base = feature_tensors(history_at(ds, cutoff), prior, 5, 0, config)
    model = RadarRainAdapter()
    assert torch.allclose(model(features, base), base, atol=1e-7)
    # Activating the learned decoder makes output depend on measured history.
    with torch.no_grad():
        model.output.weight.fill_(0.1)
    first = model(features, base)
    changed = features.clone()
    changed[:, :24] = 3
    assert not torch.allclose(first, model(changed, base))
    target = torch.full_like(base, 0.5)
    loss = radar_loss(first, target, config)
    torch.autograd.backward(loss)
    assert model.output.weight.grad is not None
    first_layer = model.encoder[0]
    assert isinstance(first_layer, torch.nn.Conv2d)
    assert first_layer.weight.grad is not None
    assert first_layer.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="No observed"):
        radar_loss(first, torch.full_like(target, float("nan")), config)


def case_files(
    root: Path, cutoff: datetime, config: Config, label: str
) -> dict[str, str]:
    radar, forecast = root / f"{label}-radar.nc", root / f"{label}-forecast.nc"
    measured = observed(cutoff)
    measured.precipitation_mm[24:] = 0.3
    measured.to_netcdf(radar)
    baseline(cutoff, config).to_netcdf(forecast)
    return {
        "initialization": cutoff.isoformat(),
        "forecast": forecast.name,
        "radar": radar.name,
    }


def test_training_checkpoint_and_heldout_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = Config(root=tmp_path, device="cpu", radar_patch_size=128)
    manifest = tmp_path / "cases.json"
    records = {
        "train": [case_files(tmp_path, datetime(2024, 1, 1), config, "train")],
        "validation": [
            case_files(tmp_path, datetime(2024, 1, 3), config, "validation")
        ],
        "test": [
            {
                "initialization": "2024-01-05T00:00:00",
                "forecast": "test-forecast.nc",
                "radar": "test-radar.nc",
            }
        ],
    }
    manifest.write_text(json.dumps(records))
    recorder = Recorder()
    report = train_radar(config, manifest, tmp_path / "run", recorder, steps=1)
    assert report["test_evaluated"] is False
    history = report["history"]
    assert isinstance(history, list) and len(history) == 1
    assert not any(Path(path).name.startswith("test-") for path in recorder.inputs)
    checkpoint = tmp_path / "run/best.pt"
    model = load_radar_adapter(checkpoint, config, None, Recorder())
    assert not model.training
    assert model.output.weight.detach().abs().sum() > 0
    with pytest.raises(ValueError, match="incompatible"):
        load_radar_adapter(
            checkpoint, replace(config, precipitation_scale=2.0), None, Recorder()
        )
    case_files(tmp_path, datetime(2024, 1, 5), config, "test")
    scores = evaluate_adapter(config, manifest, checkpoint, Recorder())
    result = scores["test"]
    assert isinstance(result, list) and len(result) == 1
    assert result[0]["radar_conditioned"]["hourly"]["count"] == 32
    assert "aurora" in result[0] and "radar_persistence" in result[0]
    monkeypatch.setattr(
        "aurora_finetuning.rainfall_metrics.motion_forecast",
        lambda history, minutes: np.full_like(history[-1], np.nan),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        unavailable = evaluate_adapter(config, manifest, checkpoint, Recorder())
    unavailable_cases = unavailable["test"]
    assert isinstance(unavailable_cases, list)
    motion = unavailable_cases[0]["radar_motion"]
    assert motion["hourly"]["count"] == 0
    assert motion["coverage_fraction"] == 0
    for comparison in (motion, motion["aurora_on_motion_support"]):
        assert comparison["aurora_grid"]["eight_hour_total"]["mae"] is None
        assert all(
            row["count"] == 0 and row["coverage_fraction"] == 0
            for row in comparison["aurora_grid"]["by_lead_hour"]
        )
    assert unavailable_cases[0]["aurora"]["hourly"] == result[0]["aurora"]["hourly"]
    json.dumps(unavailable, allow_nan=False)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_radar_adapter(checkpoint, config, None, Recorder())


def test_split_gap_enforced(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    records = {
        split: [
            {
                "initialization": f"2024-01-0{day}T00:00:00",
                "forecast": "f.nc",
                "radar": "r.nc",
            }
        ]
        for split, day in (("train", 1), ("validation", 2), ("test", 5))
    }
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match=">30h"):
        read_radar_cases(path)


def test_first_hour_uses_measured_prefix_and_later_hours_use_model() -> None:
    cutoff = datetime(2024, 1, 1, 0, 50)
    config = Config(device="cpu")
    ds = observed(cutoff, future=False)
    ds.precipitation_mm[:] = 1.0
    forecast = baseline(cutoff, config)
    result = query_grid(forecast, 52.1, 5.4, 8, cutoff.replace(tzinfo=UTC))
    refined = refine_location(result, forecast, ds, RadarRainAdapter().eval(), config)
    rows = refined["hourly"]
    assert isinstance(rows, list)
    assert rows[0]["precipitation_mm"] == pytest.approx(10.2)
    assert rows[1]["precipitation_mm"] == pytest.approx(1.2)
    assert rows[0]["temperature_c"] == 10
    assert "conditioning" in rows[-1]["precipitation_source"]


@pytest.mark.parametrize("offset", [-25, 5])
def test_live_radar_rejects_stale_or_future_measurements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offset: int,
) -> None:
    import requests

    from aurora_finetuning import radar_data

    now = datetime(2024, 1, 1, 12, tzinfo=UTC)
    stamp = now + timedelta(minutes=offset)
    session = requests.Session()

    def get(url: str, **kwargs: object) -> requests.Response:
        assert "/nl_rdr_data_rtcor_5m/" in url
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(
            {
                "files": [
                    {
                        "filename": f"RAD_NL25_RAC_RT_{stamp:%Y%m%d%H%M}.h5",
                    }
                ]
            }
        ).encode()
        return response

    monkeypatch.setattr(session, "get", get)
    monkeypatch.setattr(radar_data, "http_session", lambda: session)
    monkeypatch.setattr(radar_data, "knmi_api_key", lambda root: "test-token")
    with pytest.raises(ValueError, match="radar_max_age_minutes"):
        radar_data.live_history(Config(root=tmp_path), 52.1, 5.4, Recorder(), now)


def test_native_radar_cache_skips_repeated_hdf5_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aurora_finetuning import radar_data
    from aurora_finetuning.acquisition import sha256
    from aurora_finetuning.telemetry import write_json

    cutoff = datetime(2024, 1, 1, 2)
    paths = []
    for index in range(24):
        stamp = cutoff - timedelta(minutes=5 * (23 - index))
        path = tmp_path / f"RAD_NL25_RAC_RT_{stamp:%Y%m%d%H%M}.h5"
        raw_file(path, stamp)
        write_json(path.with_name(path.name + ".json"), {"sha256": sha256(path)})
        paths.append(path)
    import requests

    session = requests.Session()

    def get(url: str, **kwargs: object) -> requests.Response:
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(
            {"files": [{"filename": paths[-1].name}]}
        ).encode()
        return response

    monkeypatch.setattr(session, "get", get)
    monkeypatch.setattr(radar_data, "http_session", lambda: session)
    monkeypatch.setattr(radar_data, "knmi_api_key", lambda root: "test")
    monkeypatch.setattr(radar_data, "knmi_files", lambda *args: paths)
    config = Config(root=tmp_path, device="cpu")
    first = radar_data.live_history(
        config, 52.1, 5.4, Recorder(), cutoff.replace(tzinfo=UTC)
    )

    def forbidden(*args: object, **kwargs: object) -> xr.Dataset:
        raise AssertionError("Cache hit must not decode HDF5 again")

    monkeypatch.setattr(radar_data, "native_radar", forbidden)
    second = radar_data.live_history(
        config, 52.1, 5.4, Recorder(), cutoff.replace(tzinfo=UTC)
    )
    assert first.identical(second)


def test_radar_refinement_cache_skips_model_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aurora_finetuning import radar_model
    from aurora_finetuning.acquisition import sha256
    from aurora_finetuning.telemetry import write_json

    cutoff = datetime(2024, 1, 1)
    config = Config(root=tmp_path, device="cpu")
    checkpoint = tmp_path / "radar.pt"
    model = RadarRainAdapter()
    torch.save(
        {
            "architecture": radar_model.ARCHITECTURE,
            "parameters": model.state_dict(),
            "base_revision": config.checkpoint_revision,
            "base_adapter_sha256": None,
            "base_resolution": config.resolution,
            "precipitation_scale": config.precipitation_scale,
            "trained_steps": 1,
            "seasonal": True,
            "lead_weights": [0.0] * 8,
        },
        checkpoint,
    )
    write_json(checkpoint.with_suffix(".json"), {"sha256": sha256(checkpoint)})
    history = observed(cutoff, future=False)
    monkeypatch.setattr(radar_model, "live_history", lambda *args: history)
    forecast = baseline(cutoff, config)
    result = query_grid(forecast, 52.1, 5.4, 8, cutoff.replace(tzinfo=UTC))
    first = radar_model.apply_live_radar(
        result,
        forecast,
        checkpoint,
        config,
        Recorder(),
        cutoff.replace(tzinfo=UTC),
        52.1,
        5.4,
        None,
    )

    def forbidden(*args: object) -> dict[str, object]:
        raise AssertionError("Cache hit must not execute radar model")

    monkeypatch.setattr(radar_model, "refine_location", forbidden)
    second = radar_model.apply_live_radar(
        result,
        forecast,
        checkpoint,
        config,
        Recorder(),
        cutoff.replace(tzinfo=UTC),
        52.1,
        5.4,
        None,
    )
    assert first == second
