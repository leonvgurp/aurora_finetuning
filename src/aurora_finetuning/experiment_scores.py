"""Explicit interval matching and paired observational forecast verification."""

from datetime import UTC, datetime, timedelta

import numpy as np
import xarray as xr

from aurora_finetuning.evaluation import metrics
from aurora_finetuning.experiment_config import Campaign, Case
from aurora_finetuning.rainfall_metrics import event_scores

UNITS = {
    "temperature_c": "degC",
    "precipitation_mm": "mm",
    "pressure_hpa": "hPa",
    "wind_u_ms": "m/s",
    "wind_v_ms": "m/s",
}


def validate_dataset(dataset: xr.Dataset) -> None:
    """Reject ambiguous times, units and rainfall intervals before scoring."""
    if "time" not in dataset.coords or dataset.time.ndim != 1:
        raise ValueError("A one-dimensional UTC time coordinate is required")
    if not np.issubdtype(dataset.time.dtype, np.datetime64):
        raise ValueError("Time must be encoded as UTC datetime64")
    if not any(name in dataset for name in UNITS):
        raise ValueError("Dataset contains no supported physical fields")
    times = dataset.time.values.astype("datetime64[ns]")
    if (
        np.isnat(times).any()
        or len(np.unique(times)) != len(times)
        or np.any(np.diff(times) <= np.timedelta64(0, "ns"))
    ):
        raise ValueError("Times must be finite, unique and strictly increasing")
    for name in UNITS:
        if name not in dataset:
            continue
        if dataset[name].attrs.get("units") != UNITS[name]:
            raise ValueError(f"{name} requires explicit {UNITS[name]} units")
        if "time" not in dataset[name].dims:
            raise ValueError(f"{name} requires a time dimension")
    if "precipitation_mm" in dataset:
        interval = dataset.precipitation_mm.attrs.get("interval_hours")
        if type(interval) not in (int, np.int64, np.int32) or interval not in (
            1,
            2,
            3,
            6,
        ):
            raise ValueError("Rainfall requires interval_hours = 1, 2, 3 or 6")
        if bool((dataset.precipitation_mm < 0).any()):
            raise ValueError("Negative rainfall amounts are invalid")


def match_field(prediction: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    """Map regular atmospheric grids to stations; never interpolate rainfall."""
    if {"latitude", "longitude"} <= set(prediction.dims) and "station" in target.dims:
        inside = (
            (target.latitude >= prediction.latitude.min())
            & (target.latitude <= prediction.latitude.max())
            & (target.longitude >= prediction.longitude.min())
            & (target.longitude <= prediction.longitude.max())
        )
        if prediction.name == "precipitation_mm":
            prediction = prediction.sel(
                latitude=target.latitude, longitude=target.longitude, method="nearest"
            )
        else:
            prediction = prediction.interp(
                latitude=target.latitude, longitude=target.longitude
            )
        prediction = prediction.where(inside)
    if set(prediction.dims) != set(target.dims):
        raise ValueError("Spatial dimensions differ; prepare a documented common grid")
    for dim in target.dims:
        if dim != "time" and not np.array_equal(prediction[dim], target[dim]):
            raise ValueError(
                f"Spatial coordinate {dim} differs; prepare common support"
            )
    return prediction.transpose(*target.dims)


def aligned_values(
    dataset: xr.Dataset,
    observations: xr.Dataset,
    variable: str,
    case: Case,
    hours: int,
    interval: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Match actual forecast endpoints whose whole interval is after issuance."""
    issue = np.datetime64(case.issue.replace(tzinfo=None), "ns")
    one_hour = np.timedelta64(1, "h")
    horizon = issue + hours * one_hour
    target_fields, fields, used = [], [], []
    for end in dataset.time.values.astype("datetime64[ns]"):
        elapsed = end - issue
        if end > horizon or elapsed < interval * one_hour or elapsed % one_hour:
            continue
        lead = int(elapsed / one_hour)
        if variable == "precipitation_mm":
            ends = [end - i * one_hour for i in range(interval)]
            if not all(t in observations.time.values for t in ends):
                continue
            target = observations[variable].sel(time=ends).sum("time", skipna=False)
        else:
            if end not in observations.time.values:
                continue
            target = observations[variable].sel(time=end)
        field = match_field(dataset[variable].sel(time=end), target)
        fields.append(np.asarray(field.values, dtype=float))
        target_fields.append(np.asarray(target.values, dtype=float))
        used.append(lead)
    return np.asarray(fields), np.asarray(target_fields), used


def score_case(
    campaign: Campaign,
    case: Case,
    predictions: dict[str, xr.Dataset],
    observations: xr.Dataset,
) -> list[dict[str, object]]:
    """Return own-support and pairwise-support metrics for each lead and total."""
    validate_dataset(observations)
    if (
        "precipitation_mm" in observations
        and observations.precipitation_mm.attrs["interval_hours"] != 1
    ):
        raise ValueError("Observation rainfall must be hourly before verification")
    for prediction in predictions.values():
        validate_dataset(prediction)
    output: list[dict[str, object]] = []
    for method, dataset in predictions.items():
        initialized = datetime.fromisoformat(
            str(
                dataset.attrs.get("initialization_utc", case.initialization.isoformat())
            )
        )
        if initialized.tzinfo is None:
            initialized = initialized.replace(tzinfo=UTC)
        model_age = (case.issue - initialized).total_seconds() / 3600
        for variable in UNITS:
            if variable not in dataset or variable not in observations:
                continue
            interval = (
                int(dataset[variable].attrs["interval_hours"])
                if variable == "precipitation_mm"
                else 1
            )
            p, o, leads = aligned_values(
                dataset, observations, variable, case, campaign.hours, interval
            )
            if not leads:
                continue
            for baseline, other in {"own": None, **predictions}.items():
                if baseline == method:
                    continue
                base_values = None
                if other is not None:
                    if variable not in other:
                        continue
                    other_interval = (
                        int(other[variable].attrs["interval_hours"])
                        if variable == "precipitation_mm"
                        else 1
                    )
                    # Aggregate hourly baseline rain to the candidate's native interval.
                    if other_interval not in {1, interval}:
                        continue
                    other_for_score = other
                    if (
                        variable == "precipitation_mm"
                        and other_interval == 1
                        and interval > 1
                    ):
                        # Rolling counts records, so insert absent hourly endpoints
                        # before aggregating; a missing hour must invalidate its sum.
                        times = [
                            np.datetime64(
                                (case.issue + timedelta(hours=h)).replace(tzinfo=None),
                                "ns",
                            )
                            for h in range(1, campaign.hours + 1)
                        ]
                        other_for_score = other.reindex(time=times)
                        other_for_score[variable] = (
                            other_for_score[variable]
                            .rolling(time=interval, min_periods=interval)
                            .sum()
                        )
                        other_for_score[variable].attrs.update(
                            other[variable].attrs, interval_hours=interval
                        )
                    bp, _, b_leads = aligned_values(
                        other_for_score,
                        observations,
                        variable,
                        case,
                        campaign.hours,
                        interval,
                    )
                    mapping = {lead: i for i, lead in enumerate(b_leads)}
                    base_values = np.full(p.shape, np.nan)
                    for i, lead in enumerate(leads):
                        if lead in mapping:
                            base_values[i] = bp[mapping[lead]]
                mask = np.isfinite(p) & np.isfinite(o)
                if base_values is not None:
                    mask &= np.isfinite(base_values)
                for index, lead in enumerate([*leads, 0]):
                    if lead:
                        a, b, valid = p[index], o[index], mask[index]
                        base = base_values[index] if base_values is not None else None
                        possible = int(b.size)
                    else:
                        # Totals require the complete horizon, not a partial archive.
                        if (
                            variable != "precipitation_mm"
                            or campaign.hours % interval
                            or leads
                            != list(range(interval, campaign.hours + 1, interval))
                        ):
                            continue
                        a, b, valid = p.sum(0), o.sum(0), mask.all(0)
                        base = base_values.sum(0) if base_values is not None else None
                        possible = int(b.size)
                    values = metrics(
                        np.where(valid, a, np.nan), np.where(valid, b, np.nan)
                    )
                    if variable == "precipitation_mm":
                        for threshold in (0.1, 1.0, 5.0):
                            for key, value in event_scores(
                                np.where(valid, a, np.nan),
                                np.where(valid, b, np.nan),
                                threshold,
                            ).items():
                                if key == "csi":
                                    values[f"csi_{threshold}"] = value
                                elif key == "recall":
                                    values[f"pod_{threshold}"] = value
                                elif key == "precision":
                                    values[f"far_{threshold}"] = (
                                        1 - float(value) if value is not None else None
                                    )
                    if base is not None:
                        bm = metrics(
                            np.where(valid, base, np.nan), np.where(valid, b, np.nan)
                        )
                        for key in ("mae", "rmse", "bias"):
                            av, bv = values[key], bm[key]
                            values[f"paired_delta_{key}"] = (
                                float(av) - float(bv)
                                if av is not None and bv is not None
                                else None
                            )
                    values["coverage"] = (
                        float(valid.sum() / possible) if possible else 0.0
                    )
                    for metric, value in values.items():
                        output.append(
                            {
                                "campaign": campaign.id,
                                "track": campaign.track,
                                "case": case.id,
                                "split": case.split,
                                "regime": case.regime,
                                "availability": case.availability,
                                "method": method,
                                "baseline": baseline,
                                "variable": variable,
                                "lead_hour": lead,
                                "interval_hours": interval,
                                "model_lead_hour": lead + model_age,
                                "metric": metric,
                                "value": value,
                                "count": int(valid.sum()),
                                "possible_count": possible,
                            }
                        )
    return output
