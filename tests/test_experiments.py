"""Campaign behavior, interval semantics and reproducible offline analysis."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aurora_finetuning.experiment_cli import main
from aurora_finetuning.experiment_config import load_campaign
from aurora_finetuning.experiment_demo import create_demo
from aurora_finetuning.experiment_reports import (
    collect_evidence,
    legacy_metrics,
    load_results,
    paired_summary,
    report_results,
)
from aurora_finetuning.experiment_scores import score_case, validate_dataset
from aurora_finetuning.experiments import (
    plan_campaign,
    prediction,
    read_dataset,
    run_campaign,
    verify_recorded_inputs,
)
from aurora_finetuning.public_forecasts import (
    interval_rain,
    normalize_nowcast,
    normalize_public,
)


@pytest.fixture
def campaign_path(tmp_path: Path) -> Path:
    return create_demo(tmp_path / "demo")


def test_plan_is_read_only(campaign_path: Path) -> None:
    campaign = load_campaign(campaign_path)
    result = plan_campaign(campaign_path)
    assert result["track"] == "synthetic"
    assert not campaign.output.exists()


@pytest.mark.parametrize(
    "replacement",
    [
        "\nunknown = true\n",
        "\nhours = 24\n",
        "\ntimeout_seconds = 0\n",
    ],
)
def test_invalid_config(campaign_path: Path, replacement: str) -> None:
    text = campaign_path.read_text()
    if "hours" in replacement:
        text = text.replace("hours = 8", "hours = 24")
    elif "timeout" in replacement:
        text = text.replace("timeout_seconds = 120", "timeout_seconds = 0")
    else:
        text = replacement + text
    campaign_path.write_text(text)
    with pytest.raises(ValueError):
        load_campaign(campaign_path)


def test_development_cannot_be_test(campaign_path: Path) -> None:
    campaign_path.write_text(
        campaign_path.read_text().replace('split = "development"', 'split = "test"')
    )
    with pytest.raises(ValueError, match="qualification"):
        load_campaign(campaign_path)


def test_paths_resolve_at_config(
    campaign_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(campaign_path.parent.parent)
    campaign = load_campaign(campaign_path)
    assert campaign.cases[0].observations.is_file()


def test_missing_history_is_not_future_target(campaign_path: Path) -> None:
    c = load_campaign(campaign_path)
    case = replace(c.cases[0], history=None)
    with pytest.raises(FileNotFoundError, match="history"):
        prediction(c, case, c.methods[1])


def test_future_history_rejected(campaign_path: Path) -> None:
    c = load_campaign(campaign_path)
    case = replace(c.cases[0], history=c.cases[0].observations)
    with pytest.raises(ValueError, match="issuance"):
        prediction(c, case, c.methods[1])


def test_unknown_rain_interval_rejected(campaign_path: Path) -> None:
    c = load_campaign(campaign_path)
    observations = read_dataset(c.cases[0].observations)
    observations.precipitation_mm.attrs.pop("interval_hours")
    with pytest.raises(ValueError, match="interval"):
        validate_dataset(observations)


def test_duplicate_times_rejected(campaign_path: Path) -> None:
    c = load_campaign(campaign_path)
    observations = read_dataset(c.cases[0].observations)
    observations = observations.assign_coords(time=[observations.time.values[0]] * 8)
    with pytest.raises(ValueError, match="unique"):
        validate_dataset(observations)


def test_paired_mask_and_model_lead(campaign_path: Path) -> None:
    c = load_campaign(campaign_path)
    case = c.cases[0]
    obs = read_dataset(case.observations)
    a = obs.copy(deep=True)
    b = obs.copy(deep=True)
    a.temperature_c.values[:] += 1
    b.temperature_c.values[:, 0] = np.nan
    a.attrs["initialization_utc"] = case.initialization.isoformat()
    b.attrs["initialization_utc"] = case.issue.isoformat()
    scores = pd.DataFrame(score_case(c, case, {"a": a, "b": b}, obs))
    paired = scores[
        (scores.method == "a")
        & (scores.baseline == "b")
        & (scores.variable == "temperature_c")
        & (scores.metric == "mae")
    ]
    assert paired["count"].eq(1).all()
    assert paired.value.eq(1).all()
    lead = scores[(scores.method == "b") & (scores.lead_hour == 1)]
    assert lead.model_lead_hour.eq(1).all()


def test_coarse_rain_aggregates_baseline(campaign_path: Path) -> None:
    c = load_campaign(campaign_path)
    case = c.cases[1]
    obs = read_dataset(case.observations)
    hourly = obs.copy(deep=True)
    coarse = hourly.rolling(time=3, min_periods=3).sum().isel(time=[2, 5])
    coarse.precipitation_mm.attrs.update(units="mm", interval_hours=3)
    coarse.temperature_c.attrs["units"] = "degC"
    scores = pd.DataFrame(
        score_case(c, case, {"coarse": coarse, "hourly": hourly}, obs)
    )
    paired = scores[
        (scores.method == "coarse")
        & (scores.baseline == "hourly")
        & (scores.variable == "precipitation_mm")
        & (scores.metric == "paired_delta_mae")
    ]
    assert set(paired.lead_hour) == {3, 6}
    assert np.allclose(paired.value, 0)
    # Eight-hour total is not invented from six hours of coarse data.
    assert not ((scores.method == "coarse") & (scores.lead_hour == 0)).any()


def test_cumulative_rain_native_intervals() -> None:
    rain = xr.DataArray(
        [1.0, 3.0, 6.0],
        dims="time",
        coords={
            "time": np.array(
                ["2026-01-01T03", "2026-01-01T06", "2026-01-01T09"],
                dtype="datetime64[ns]",
            )
        },
    )
    result = interval_rain(rain, datetime(2026, 1, 1, tzinfo=UTC), 3)
    assert result.values.tolist() == [1, 2, 3]
    with pytest.raises(ValueError, match="consecutive"):
        interval_rain(rain.isel(time=[0, 2]), datetime(2026, 1, 1, tzinfo=UTC), 3)
    with pytest.raises(ValueError, match="decreases"):
        interval_rain(
            rain.copy(data=[2.0, 1.0, 3.0]), datetime(2026, 1, 1, tzinfo=UTC), 3
        )


def test_nowcast_excludes_analysis(tmp_path: Path) -> None:
    init = datetime(2026, 1, 1, tzinfo=UTC)
    times = np.datetime64("2026-01-01", "ns") + np.arange(1, 25) * np.timedelta64(
        5, "m"
    )
    ds = xr.Dataset(
        {"precipitation_mm": ("time", np.ones(24))},
        coords={"time": times},
        attrs={"initialization_utc": init.isoformat()},
    )
    ds.precipitation_mm.attrs.update(units="mm", interval_minutes=5)
    source, output = tmp_path / "source.nc", tmp_path / "out.nc"
    ds.to_netcdf(source)
    normalize_nowcast(source, output)
    with xr.open_dataset(output) as result:
        assert result.precipitation_mm.values.tolist() == [12, 12]


def test_single_event_has_no_bootstrap(campaign_path: Path) -> None:
    c = load_campaign(campaign_path)
    case = c.cases[0]
    obs = read_dataset(case.observations)
    rows = pd.DataFrame(score_case(c, case, {"a": obs, "b": obs}, obs))
    summary = paired_summary(rows)
    assert summary.events.eq(1).all()
    assert summary.lower.isna().all()


def test_end_to_end_resume_report(campaign_path: Path, tmp_path: Path) -> None:
    # Keep the default suite fast while testing real subprocess/package execution.
    text = campaign_path.read_text()
    start = text.index("[[cases]]", text.index("[[cases]]") + 1)
    methods = text.index("[[methods]]")
    text = text[:start] + text[methods:]
    for i in range(1, 6):
        text = text.replace(f'case-{i} = "case-{i}-forecast.nc"\n', "")
    campaign_path.write_text(text)
    first = run_campaign(campaign_path)
    assert first["status"] == "completed"
    second = run_campaign(campaign_path)
    states = second["states"]
    assert isinstance(states, list)
    assert all(s["reused"] for s in states if isinstance(s, dict))
    results = load_results(campaign_path.parent)
    assert not results.empty
    assert len(results) == first["scores"]
    report = report_results(campaign_path.parent, tmp_path / "report")
    assert report["plots"]
    with pytest.raises(FileExistsError):
        report_results(campaign_path.parent, tmp_path / "report")
    output = load_campaign(campaign_path).output
    artifact = next((output / "run/case-0").glob("*.nc"))
    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        run_campaign(campaign_path)


def test_inventory_preserves_unknown_schema(tmp_path: Path) -> None:
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs/old.json").write_text('{"loss": 0.1}')
    result = collect_evidence(tmp_path, tmp_path / "inventory")
    assert result["entries"] == 1
    assert result["comparisons"] == 0


def test_cli_missing_config() -> None:
    assert main(["plan"]) == 2


def test_recorded_indirect_inputs_detect_changes(tmp_path: Path) -> None:
    from aurora_finetuning.acquisition import sha256

    source = tmp_path / "training-input.json"
    source.write_text("original")
    recorded = {str(source): sha256(source)}
    verify_recorded_inputs(recorded)
    source.write_text("changed")
    with pytest.raises(ValueError, match="Recorded input changed"):
        verify_recorded_inputs(recorded)


def test_legacy_loss_is_not_qualification(tmp_path: Path) -> None:
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs/pilot.json").write_text('{"loss": 0.1}')
    results = legacy_metrics(tmp_path)
    assert results.scope.tolist() == ["unclassified historical result"]
    assert results.metric_path.tolist() == ["/loss"]


def test_rain_initialization_normalizes_offset() -> None:
    rain = xr.DataArray(
        [1.0, 3.0],
        dims="time",
        coords={
            "time": np.array(["2026-01-01T03", "2026-01-01T06"], dtype="datetime64[ns]")
        },
    )
    result = interval_rain(rain, datetime.fromisoformat("2026-01-01T02:00:00+02:00"), 3)
    assert result.values.tolist() == [1, 2]


def test_public_normalization_discards_scalar_grib_heights(tmp_path: Path) -> None:
    times = np.array(["2026-01-01T03", "2026-01-01T06"], dtype="datetime64[ns]")
    inputs = []
    for name, values, unit, height in (
        ("t2m", [283.15, 284.15], "K", 2),
        ("tp", [0.001, 0.003], "m", 0),
    ):
        dataset = xr.Dataset(
            {name: ("time", values)},
            coords={"time": times, "heightAboveGround": height},
        )
        dataset[name].attrs["units"] = unit
        path = tmp_path / f"{name}.nc"
        dataset.to_netcdf(path)
        inputs.append(path)
    output = tmp_path / "normalized.nc"
    normalize_public(inputs, output, "ifs", datetime(2026, 1, 1, tzinfo=UTC), 3, True)
    normalized = read_dataset(output)
    assert np.allclose(normalized.temperature_c.values, [10, 11])
    assert np.allclose(normalized.precipitation_mm.values, [1, 2])
    assert "heightAboveGround" not in normalized.coords


def test_native_rain_intervals_need_not_align_with_issuance(
    campaign_path: Path,
) -> None:
    """01Z issuance retains the complete 03–06Z and 06–09Z intervals."""
    from datetime import timedelta

    c = load_campaign(campaign_path)
    case = replace(c.cases[0], issue=c.cases[0].issue + timedelta(hours=1))
    times = np.datetime64(case.issue.replace(tzinfo=None), "ns") + np.arange(
        1, 9
    ) * np.timedelta64(1, "h")
    obs = xr.Dataset({"precipitation_mm": ("time", np.ones(8))}, coords={"time": times})
    obs.precipitation_mm.attrs.update(units="mm", interval_hours=1)
    coarse = xr.Dataset(
        {"precipitation_mm": ("time", [3.0, 3.0, 3.0])},
        coords={
            "time": np.datetime64(case.initialization.replace(tzinfo=None), "ns")
            + np.array([9, 12, 15]) * np.timedelta64(1, "h")
        },
    )
    coarse.precipitation_mm.attrs.update(units="mm", interval_hours=3)
    rows = pd.DataFrame(score_case(c, case, {"ifs": coarse, "hourly": obs}, obs))
    rain = rows[(rows.method == "ifs") & (rows.metric == "mae")]
    assert set(rain.lead_hour) == {5, 8}
    assert rain.value.eq(0).all()
    assert set(rain.baseline) == {"own", "hourly"}
    assert not ((rows.method == "ifs") & (rows.lead_hour == 0)).any()


def test_missing_baseline_hour_invalidates_three_hour_pair(campaign_path: Path) -> None:
    c = load_campaign(campaign_path)
    case = c.cases[0]
    obs = read_dataset(case.observations)[["precipitation_mm"]]
    coarse = obs.rolling(time=3, min_periods=3).sum().isel(time=[2, 5])
    coarse.precipitation_mm.attrs.update(units="mm", interval_hours=3)
    missing_hour = obs.isel(time=[0, 1, 2, 3, 5, 6, 7])
    rows = pd.DataFrame(
        score_case(c, case, {"coarse": coarse, "hourly": missing_hour}, obs)
    )
    paired = rows[(rows.method == "coarse") & (rows.baseline == "hourly")]
    late = paired[paired.lead_hour == 6]
    assert late["count"].eq(0).all()
    assert late[late.metric == "coverage"].value.eq(0).all()
    assert late[late.metric == "mae"].value.isna().all()
    early = paired[(paired.lead_hour == 3) & (paired.metric == "mae")]
    assert early["count"].eq(2).all()
    assert np.allclose(early.value, 0)


def test_prospective_aurora_rejects_reanalysis(campaign_path: Path) -> None:
    from aurora_finetuning.experiment_config import Method

    c = replace(load_campaign(campaign_path), track="prospective")
    method = Method("aurora", "aurora", {}, Path("configs/default.toml"), "era5")
    with pytest.raises(ValueError, match="ERA5 is retrospective"):
        prediction(c, c.cases[0], method)


@pytest.mark.parametrize("available", ["timely", "late", "missing", "naive"])
def test_prospective_aurora_checks_every_selected_input(
    campaign_path: Path,
    tmp_path: Path,
    available: str,
) -> None:
    import json
    from datetime import timedelta

    from aurora_finetuning.acquisition import sha256
    from aurora_finetuning.config import Config
    from aurora_finetuning.experiments import prospective_aurora_inputs
    from aurora_finetuning.grid import make_grid

    c = load_campaign(campaign_path)
    case = c.cases[0]
    config = Config(root=tmp_path)
    grid = make_grid(config)
    files = []
    for kind in ("single", "pressure"):
        for offset in (6, 0):
            time = (case.initialization - timedelta(hours=offset)).replace(tzinfo=None)
            path = (
                tmp_path / "data/raw/ifs" / f"{time.date()}-{kind}-{time.hour:02d}.nc"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            dataset = xr.Dataset(
                {
                    "field": (
                        ("time", "latitude", "longitude"),
                        np.zeros((1, len(grid.latitude), len(grid.longitude))),
                    )
                },
                coords={
                    "time": [np.datetime64(time, "ns")],
                    "latitude": grid.latitude,
                    "longitude": grid.longitude,
                },
            )
            dataset.to_netcdf(path)
            manifest: dict[str, object] = {
                "sha256": sha256(path),
                "available_utc": case.issue.isoformat(),
            }
            # The prior pressure state must be checked, not just the latest surface.
            if kind == "pressure" and offset == 6:
                if available == "missing":
                    manifest.pop("available_utc")
                elif available == "late":
                    manifest["available_utc"] = (
                        case.issue + timedelta(minutes=1)
                    ).isoformat()
                elif available == "naive":
                    manifest["available_utc"] = case.issue.replace(
                        tzinfo=None
                    ).isoformat()
            path.with_suffix(".json").write_text(json.dumps(manifest))
            files.append(path)
    if available != "timely":
        with pytest.raises(ValueError):
            prospective_aurora_inputs(config, case)
    else:
        checked = prospective_aurora_inputs(config, case)
        assert len(checked) == 8
        assert all(checked[str(p)] == sha256(p) for p in files)
        assert all(str(p.with_suffix(".json")) in checked for p in files)


def test_prospective_aurora_rejected_at_configuration_boundary(
    campaign_path: Path,
) -> None:
    text = campaign_path.read_text().replace(
        'track = "synthetic"', 'track = "prospective"'
    )
    text = text.replace(
        'split = "development"', 'split = "test"\navailability = "recorded"'
    )
    text += (
        '\n[[methods]]\nid = "aurora"\nkind = "aurora"'
        '\nsource = "era5"\nconfig = "weather.toml"\n'
    )
    campaign_path.write_text(text)
    with pytest.raises(ValueError, match="ERA5 is retrospective"):
        load_campaign(campaign_path)


def test_resume_rescores_after_a_blocked_prediction_recovers(
    campaign_path: Path,
) -> None:
    """Recover a genuine publication failure without changing configured inputs."""
    import json

    text = campaign_path.read_text()
    start = text.index("[[cases]]", text.index("[[cases]]") + 1)
    methods = text.index("[[methods]]")
    text = text[:start] + text[methods:]
    for i in range(1, 6):
        text = text.replace(f'case-{i} = "case-{i}-forecast.nc"\n', "")
    campaign_path.write_text(text)
    c = load_campaign(campaign_path)
    case_folder = c.output / "run/case-0"
    blocker = case_folder / "persistence.nc"
    blocker.mkdir(parents=True)
    first = run_campaign(campaign_path)
    assert first["status"] == "incomplete"
    initial_scores = [
        p
        for p in case_folder.glob("scores-*.json")
        if not p.name.endswith(".receipt.json")
    ]
    assert len(initial_scores) == 1
    original = initial_scores[0].read_bytes()
    assert {row["method"] for row in json.loads(original)["rows"]} == {
        "dry",
        "synthetic-forecast",
    }
    blocker.rmdir()
    recovered = run_campaign(campaign_path)
    assert recovered["status"] == "completed"
    assert (
        len(
            [
                p
                for p in case_folder.glob("scores-*.json")
                if not p.name.endswith(".receipt.json")
            ]
        )
        == 2
    )
    assert initial_scores[0].read_bytes() == original
    latest = load_results(campaign_path.parent)
    assert set(latest.method) == {"dry", "persistence", "synthetic-forecast"}
    assert "persistence" in set(latest.baseline)
    assert len(latest) == recovered["scores"]
    again = run_campaign(campaign_path)
    states = again["states"]
    assert isinstance(states, list)
    assert again["status"] == "completed"
    assert all(s["reused"] for s in states if isinstance(s, dict))
