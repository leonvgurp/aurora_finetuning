"""Leakage-aware recent station features and small external ridge calibrators.

No Aurora or torch imports: training consumes saved forecast/observation pairs.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from aurora_finetuning.acquisition import sha256
from aurora_finetuning.artifacts import atomic_output
from aurora_finetuning.config import Config
from aurora_finetuning.evaluation import metrics
from aurora_finetuning.grid import FloatArray
from aurora_finetuning.observations import load_stations, station_geometry
from aurora_finetuning.seasonal import calendar_features
from aurora_finetuning.telemetry import Recorder, write_json

VARIABLES = ("temperature_c", "pressure_hpa", "wind_u_ms", "wind_v_ms")
VERSION = "station-ridge-v1"
FEATURES = (
    "forecast",
    "lead",
    "latitude",
    "longitude",
    "annual_sin",
    "annual_cos",
    "daily_sin",
    "daily_cos",
    "latest_residual",
    "age_hours",
    "trend_per_hour",
    "residual_6h",
    "residual_24h",
    "residual_72h",
    "distance_km",
    "coverage",
    "missing_latest",
    "missing_age",
    "missing_trend",
    "missing_6h",
    "missing_24h",
    "missing_72h",
)


def utc_naive(time: datetime) -> datetime:
    """Use naive UTC only at the xarray boundary."""
    return time.astimezone(UTC).replace(tzinfo=None) if time.tzinfo else time


def distances(
    latitude: float, longitude: float, lat: FloatArray, lon: FloatArray
) -> FloatArray:
    """Great-circle distances in kilometres, including coastal stations."""
    a = np.sin(np.deg2rad(lat - latitude) / 2) ** 2
    a += (
        np.cos(np.deg2rad(latitude))
        * np.cos(np.deg2rad(lat))
        * np.sin(np.deg2rad(lon - longitude) / 2) ** 2
    )
    result: FloatArray = 12742 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return result


def forecast_available(dataset: xr.Dataset) -> datetime:
    """Respect publication time; initialization-only archives are retrospective."""
    initialization = datetime.fromisoformat(str(dataset.attrs["initialization_utc"]))
    available = datetime.fromisoformat(
        str(dataset.attrs.get("available_utc", initialization.isoformat()))
    )
    if utc_naive(available) < utc_naive(initialization):
        raise ValueError("Forecast availability precedes initialization")
    return utc_naive(available)


def recent_features(
    observations: xr.Dataset,
    archives: list[xr.Dataset],
    variable: str,
    cutoff: datetime,
    latitude: float,
    longitude: float,
    max_age_minutes: int,
    excluded_stations: frozenset[str] = frozenset(),
) -> tuple[FloatArray, bool]:
    """Interpolate summaries from up to four fresh stations within 100 km.

    A residual uses only a forecast available by its observation time. Availability
    coordinates, when supplied, are also enforced for station records.
    """
    cutoff = utc_naive(cutoff)
    summaries: list[FloatArray] = []
    offsets: list[float] = []
    if variable not in observations:
        return np.zeros(14), False
    window = observations.sel(
        time=slice(np.datetime64(cutoff - timedelta(hours=72)), np.datetime64(cutoff))
    )
    window = station_geometry(window)
    if not window.sizes.get("time", 0):
        return np.zeros(14), False
    distance = distances(
        latitude,
        longitude,
        np.asarray(window.latitude.values, dtype=float),
        np.asarray(window.longitude.values, dtype=float),
    )
    for index in np.argsort(distance):
        if (
            distance[index] > 100
            or str(window.station.values[index]) in excluded_stations
        ):
            continue
        record = window.isel(station=index)
        values = np.asarray(record[variable].values, dtype=float)
        valid = np.isfinite(values)
        if "available_time" in record:
            valid &= record.available_time.values <= np.datetime64(cutoff)
        positions = np.flatnonzero(valid)
        if not positions.size:
            continue
        latest = int(positions[-1])
        age = float(
            (np.datetime64(cutoff) - record.time.values[latest])
            / np.timedelta64(1, "m")
        )
        if not 0 <= age <= max_age_minutes:
            continue
        residual = np.full(values.shape, np.nan)
        # Build only the selected station's bounded hourly history.
        for position in positions:
            time = datetime.fromisoformat(
                str(record.time.values[position].astype("datetime64[us]"))
            )
            eligible = [
                ds
                for ds in archives
                if variable in ds
                and forecast_available(ds) <= time
                and np.datetime64(time, "ns") in ds.time.values
            ]
            if eligible:
                archive = max(eligible, key=forecast_available)
                field = archive[variable].sel(time=np.datetime64(time, "ns"))
                prediction = float(
                    field.interp(
                        latitude=float(record.latitude),
                        longitude=float(record.longitude),
                    )
                )
                residual[position] = values[position] - prediction
        trend = float("nan")
        recent = positions[
            record.time.values[positions] >= np.datetime64(cutoff - timedelta(hours=6))
        ]
        if recent.size > 1:
            first = int(recent[0])
            elapsed = float(
                (record.time.values[latest] - record.time.values[first])
                / np.timedelta64(1, "h")
            )
            if elapsed > 0:
                trend = float((values[latest] - values[first]) / elapsed)
        means = []
        for hours in (6, 24, 72):
            r = residual[
                record.time.values >= np.datetime64(cutoff - timedelta(hours=hours))
            ]
            means.append(float(np.nanmean(r)) if np.isfinite(r).any() else float("nan"))
        raw = np.array([residual[latest], age / 60, trend, *means], dtype=float)
        summary = np.concatenate(
            (
                np.nan_to_num(raw),
                [
                    float(distance[index]),
                    float(
                        len(
                            np.unique(
                                record.time.values[positions].astype("datetime64[h]")
                            )
                        )
                        / 73
                    ),
                ],
                (~np.isfinite(raw)).astype(float),
            )
        )
        summaries.append(summary)
        offsets.append(float(distance[index]))
        if distance[index] < 1e-6 or len(summaries) == 4:
            break
    if not summaries:
        return np.zeros(14), False
    weights = 1 / np.maximum(offsets, 1e-6)
    result: FloatArray = np.average(np.stack(summaries), axis=0, weights=weights)
    return result, True


def feature_row(
    value: float,
    lead: float,
    time: datetime,
    latitude: float,
    longitude: float,
    summary: FloatArray,
) -> FloatArray:
    """Join physical forecast, calendar, geography and observation features."""
    result: FloatArray = np.array(
        [value, lead, latitude, longitude, *calendar_features(time), *summary],
        dtype=float,
    )
    if result.shape != (len(FEATURES),) or not np.isfinite(result).all():
        raise ValueError("Invalid station feature row")
    return result


@dataclass(frozen=True)
class StationCase:
    """Explicit issuance and artifacts; all manifest paths are relative."""

    issuance: datetime
    forecast: Path
    observations: tuple[Path, ...]
    targets: tuple[Path, ...]
    history_forecasts: tuple[Path, ...]


def read_cases(path: Path) -> dict[str, list[StationCase]]:
    """Require ordered, non-overlapping 72h-history/eight-hour-target windows."""
    raw = json.loads(path.read_text())
    if set(raw) != {"train", "validation", "test"}:
        raise ValueError("Station cases require train, validation and test")
    result: dict[str, list[StationCase]] = {}
    for split in ("train", "validation", "test"):
        entries = []
        for item in raw[split]:
            if set(item) != {
                "issuance",
                "forecast",
                "observations",
                "targets",
                "history_forecasts",
            }:
                raise ValueError("Invalid station case fields")
            issuance = datetime.fromisoformat(item["issuance"])
            if (
                issuance.tzinfo is None
                or issuance.minute
                or issuance.second
                or issuance.microsecond
            ):
                raise ValueError(
                    "Station issuance must be timezone-aware on an exact hour"
                )
            entries.append(
                StationCase(
                    utc_naive(issuance),
                    (path.parent / item["forecast"]).resolve(),
                    *(
                        tuple((path.parent / p).resolve() for p in item[name])
                        for name in ("observations", "targets", "history_forecasts")
                    ),
                )
            )
        if not entries or len({case.issuance for case in entries}) != len(entries):
            raise ValueError(
                "Station splits must be nonempty with unique issuance times"
            )
        result[split] = sorted(entries, key=lambda case: case.issuance)
    for before, after in (("train", "validation"), ("validation", "test")):
        # End of target + 30h embargo must precede start of next history.
        if result[before][-1].issuance + timedelta(hours=8 + 30) >= result[after][
            0
        ].issuance - timedelta(hours=72):
            raise ValueError("Station split windows require >30h separation")
    return result


def case_examples(
    case: StationCase,
    config: Config,
    recorder: Recorder,
    heldout: frozenset[str] = frozenset(),
) -> xr.Dataset:
    """Prepare cached rows without training or consuming later input observations."""
    paths = (case.forecast, *case.observations, *case.targets, *case.history_forecasts)
    for path in paths:
        recorder.inputs[str(path)] = sha256(path)
    observations = load_stations(list(case.observations))
    targets = station_geometry(load_stations(list(case.targets)))
    archives: list[xr.Dataset] = []
    for path in case.history_forecasts:
        with xr.open_dataset(path) as source:
            archives.append(source.load())
    rows: list[FloatArray] = []
    labels: list[float] = []
    variable_names: list[str] = []
    stations: list[str] = []
    leads: list[int] = []
    persistence: list[float] = []
    with xr.open_dataset(case.forecast) as forecast:
        if (
            forecast.attrs.get("checkpoint_revision") != config.checkpoint_revision
            or forecast.attrs.get("resolution") != config.resolution
        ):
            raise ValueError("Station forecast checkpoint/resolution mismatch")
        if forecast_available(forecast) > case.issuance:
            raise ValueError("Forecast was unavailable at issuance")
        initialization = utc_naive(
            datetime.fromisoformat(str(forecast.attrs["initialization_utc"]))
        )
        if (
            not 0
            <= (case.issuance - initialization).total_seconds() / 3600
            <= config.max_input_age_hours
        ):
            raise ValueError(
                "Station case uses a stale or future forecast initialization"
            )
        for station in targets.station.values:
            point = targets.sel(station=station)
            lat, lon = float(point.latitude), float(point.longitude)
            if not (
                config.south <= lat <= config.north
                and config.west <= lon <= config.east
            ):
                continue
            for variable in VARIABLES:
                if variable not in targets or variable not in forecast:
                    continue
                summary, usable = recent_features(
                    observations,
                    archives,
                    variable,
                    case.issuance,
                    lat,
                    lon,
                    config.station_max_age_minutes,
                    heldout,
                )
                if not usable:
                    continue
                for lead in range(1, 9):
                    time = case.issuance + timedelta(hours=lead)
                    try:
                        target = float(
                            point[variable].sel(time=np.datetime64(time, "ns"))
                        )
                        prior = float(
                            forecast[variable]
                            .sel(time=np.datetime64(time, "ns"))
                            .interp(latitude=lat, longitude=lon)
                        )
                    except KeyError:
                        continue
                    if not np.isfinite(target) or not np.isfinite(prior):
                        continue
                    rows.append(feature_row(prior, lead, time, lat, lon, summary))
                    labels.append(target)
                    variable_names.append(variable)
                    stations.append(str(station))
                    leads.append(lead)
                    finite = np.array([], dtype=float)
                    if (
                        variable in observations
                        and station in observations.station.values
                    ):
                        previous = observations[variable].sel(
                            station=station,
                            time=slice(None, np.datetime64(case.issuance)),
                        )
                        if "available_time" in previous.coords:
                            previous = previous.where(
                                previous.available_time <= np.datetime64(case.issuance)
                            )
                        finite = np.asarray(previous.values)[
                            np.isfinite(previous.values)
                        ]
                    persistence.append(
                        float(finite[-1]) if finite.size else float("nan")
                    )
    if not rows:
        raise ValueError(
            "No station examples with fresh input observations and matched targets"
        )
    return xr.Dataset(
        {
            "features": (("sample", "feature"), np.stack(rows)),
            "target": ("sample", labels),
            "variable": ("sample", variable_names),
            "station_id": ("sample", stations),
            "lead": ("sample", leads),
            "persistence": ("sample", persistence),
        },
        coords={"feature": list(FEATURES)},
        attrs={
            "issuance": case.issuance.isoformat(),
            "mode": "retrospective unless source availability is recorded",
        },
    )


def prepare_examples(
    config: Config,
    manifest: Path,
    output: Path,
    recorder: Recorder,
    heldout: frozenset[str] = frozenset(),
) -> dict[str, object]:
    """Materialize train/validation only; never open test targets during preparation."""
    if output.exists():
        raise FileExistsError(output)
    cases = read_cases(manifest)
    output.mkdir(parents=True)
    for split in ("train", "validation"):
        records = [
            case_examples(case, config, recorder, heldout) for case in cases[split]
        ]
        combined = xr.concat(records, dim="sample", combine_attrs="drop_conflicts")
        if heldout:
            combined = combined.isel(
                sample=np.flatnonzero(
                    ~np.isin(combined.station_id.values, list(heldout))
                )
            )
        if not combined.sizes["sample"]:
            raise ValueError("No station training examples after held-out exclusion")
        with atomic_output(output / f"{split}.nc") as temporary:
            combined.to_netcdf(temporary)
    write_json(
        output / "provenance.json",
        {
            "cases": str(manifest.resolve()),
            "sha256": recorder.inputs,
            "test_evaluated": False,
            "heldout_stations": sorted(heldout),
        },
    )
    return {"directory": str(output), "test_evaluated": False}


@dataclass(frozen=True)
class Ridge:
    """Standardized residual ridge fit, with an unpenalized intercept."""

    mean: FloatArray
    scale: FloatArray
    coefficients: FloatArray
    seasonal: bool

    def predict(self, features: FloatArray) -> FloatArray:
        """Predict a residual using frozen training-set normalization."""
        x = features.copy()
        if not self.seasonal:
            x[:, 4:8] = 0
        result: FloatArray = (
            np.column_stack((np.ones(len(x)), (x - self.mean) / self.scale))
            @ self.coefficients
        )
        return result


def fit_ridge(
    features: FloatArray, residuals: FloatArray, alpha: float, seasonal: bool
) -> Ridge:
    """Fit small dense ridge systems using NumPy, with no system environment changes."""
    if (
        len(features) < 2
        or not np.isfinite(features).all()
        or not np.isfinite(residuals).all()
        or alpha <= 0
    ):
        raise ValueError("Ridge needs finite examples and positive regularization")
    x = features.copy()
    if not seasonal:
        x[:, 4:8] = 0
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-8, 1, scale)
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ residuals)
    return Ridge(mean, scale, coefficients, seasonal)


def train_station(
    config: Config, prepared: Path, output: Path, recorder: Recorder
) -> dict[str, object]:
    """Select regularization, seasonality and lead weights on validation only."""
    models: dict[str, object] = {}
    for name in ("train", "validation"):
        recorder.inputs[str(prepared / f"{name}.nc")] = sha256(prepared / f"{name}.nc")
    with (
        xr.open_dataset(prepared / "train.nc") as training,
        xr.open_dataset(prepared / "validation.nc") as validation,
    ):
        if training.feature.values.tolist() != list(
            FEATURES
        ) or validation.feature.values.tolist() != list(FEATURES):
            raise ValueError("Prepared station feature schema mismatch")
        for variable in VARIABLES:
            a = training.isel(
                sample=np.flatnonzero(training.variable.values == variable)
            )
            b = validation.isel(
                sample=np.flatnonzero(validation.variable.values == variable)
            )
            if a.sizes["sample"] < 2 or not b.sizes["sample"]:
                continue
            x, v = np.asarray(a.features.values), np.asarray(b.features.values)
            best: tuple[float, Ridge, float] | None = None
            for seasonal in (False, True):
                for alpha in (0.1, 1.0, 10.0, 100.0):
                    model = fit_ridge(
                        x, np.asarray(a.target.values) - x[:, 0], alpha, seasonal
                    )
                    score = float(
                        np.abs(v[:, 0] + model.predict(v) - b.target.values).mean()
                    )
                    if best is None or score < best[0]:
                        best = score, model, alpha
            if best is None:
                continue
            score, model, alpha = best
            residual = model.predict(v)
            weights = []
            for lead in range(1, 9):
                mask = b.lead.values == lead
                weight = 0.0
                if mask.any():
                    errors = [
                        (
                            float(
                                np.abs(
                                    v[mask, 0]
                                    + w * residual[mask]
                                    - b.target.values[mask]
                                ).mean()
                            ),
                            w,
                        )
                        for w in (0.0, 0.25, 0.5, 0.75, 1.0)
                    ]
                    weight = min(errors)[1]
                weights.append(weight)
            models[variable] = {
                "mean": model.mean.tolist(),
                "scale": model.scale.tolist(),
                "coefficients": model.coefficients.tolist(),
                "seasonal": model.seasonal,
                "alpha": alpha,
                "validation_mae": score,
                "weights": weights,
            }
    if not models:
        raise ValueError("No trainable station variables")
    provenance_path = prepared / "provenance.json"
    provenance = (
        json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    )
    artifact = {
        "heldout_stations": provenance.get("heldout_stations", []),
        "version": VERSION,
        "base_revision": config.checkpoint_revision,
        "resolution": config.resolution,
        "features": list(FEATURES),
        "models": models,
        "inputs": recorder.inputs,
        "test_evaluated": False,
    }
    write_json(output, artifact)
    write_json(
        output.with_suffix(output.suffix + ".sha256.json"), {"sha256": sha256(output)}
    )
    return {"model": str(output), "variables": list(models), "test_evaluated": False}


def load_calibrator(path: Path, config: Config) -> dict[str, tuple[Ridge, FloatArray]]:
    """Verify artifact provenance and reject incompatible numerical coefficients."""
    digest = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text())
    if digest["sha256"] != sha256(path):
        raise ValueError("Station calibrator checksum mismatch")
    raw = json.loads(path.read_text())
    if (
        raw["version"] != VERSION
        or raw["features"] != list(FEATURES)
        or raw["base_revision"] != config.checkpoint_revision
        or raw["resolution"] != config.resolution
    ):
        raise ValueError("Incompatible station calibrator")
    result = {}
    for variable, item in raw["models"].items():
        mean, scale, coefficients, weights = (
            np.asarray(item[n], dtype=float)
            for n in ("mean", "scale", "coefficients", "weights")
        )
        if (
            variable not in VARIABLES
            or mean.shape != (len(FEATURES),)
            or scale.shape != mean.shape
            or coefficients.shape != (len(FEATURES) + 1,)
            or weights.shape != (8,)
            or np.any(scale <= 0)
            or not all(
                np.isfinite(a).all() for a in (mean, scale, coefficients, weights)
            )
            or not np.isin(weights, [0, 0.25, 0.5, 0.75, 1]).all()
        ):
            raise ValueError("Invalid station calibrator coefficients")
        result[variable] = (
            Ridge(mean, scale, coefficients, bool(item["seasonal"])),
            weights,
        )
    return result


def apply_station(
    result: dict[str, object],
    observations: xr.Dataset,
    archives: list[xr.Dataset],
    path: Path,
    config: Config,
    cutoff: datetime,
) -> dict[str, object]:
    """Adjust any locality, falling back explicitly per variable without recent data."""
    models = load_calibrator(path, config)
    lat, lon = float(str(result["latitude"])), float(str(result["longitude"]))
    rows = result["hourly"]
    if not isinstance(rows, list):
        raise ValueError("Invalid hourly result")
    for variable in VARIABLES:
        summary, usable = recent_features(
            observations,
            archives,
            variable,
            cutoff,
            lat,
            lon,
            config.station_max_age_minutes,
        )
        for row in rows:
            if not isinstance(row, dict) or variable not in row:
                continue
            row[f"aurora_{variable}"] = row[variable]
            lead = int(
                np.ceil(
                    (
                        utc_naive(datetime.fromisoformat(str(row["time"])))
                        - utc_naive(cutoff)
                    ).total_seconds()
                    / 3600
                )
            )
            if not usable or variable not in models or not 1 <= lead <= 8:
                row[f"{variable}_source"] = (
                    "Aurora; no compatible calibrator or fresh nearby observations"
                )
                continue
            model, weights = models[variable]
            feature = feature_row(
                float(row[variable]),
                (
                    utc_naive(datetime.fromisoformat(str(row["time"])))
                    - utc_naive(cutoff)
                ).total_seconds()
                / 3600,
                datetime.fromisoformat(str(row["time"])),
                lat,
                lon,
                summary,
            )
            row[variable] = float(row[variable]) + float(
                weights[lead - 1] * model.predict(feature[None])[0]
            )
            row[f"{variable}_source"] = (
                "Aurora + KNMI station calibration"
                if weights[lead - 1]
                else "Aurora; validation selected zero correction"
            )
    for row in rows:
        if "wind_u_ms" in row and "wind_v_ms" in row:
            row["wind_speed_ms"] = float(np.hypot(row["wind_u_ms"], row["wind_v_ms"]))
    return {
        **result,
        "station_cutoff_utc": utc_naive(cutoff).isoformat(),
        "station_calibrator_sha256": sha256(path),
    }


def evaluate_station(
    config: Config,
    manifest: Path,
    adapter: Path,
    recorder: Recorder,
    heldout: frozenset[str] = frozenset(),
) -> dict[str, object]:
    """Score unseen dates/stations per hour, retaining raw and corrected errors."""
    from aurora_finetuning.rainfall_metrics import bootstrap_cases

    models = load_calibrator(adapter, config)
    recorded = frozenset(json.loads(adapter.read_text()).get("heldout_stations", []))
    if heldout and not heldout <= recorded:
        raise ValueError("Held-out stations must be excluded during prepare-stations")
    reports = []
    case_errors: dict[str, list[list[float]]] = {}
    cases = read_cases(manifest)
    for case in cases["test"]:
        examples = case_examples(case, config, recorder, heldout)
        if heldout:
            examples = examples.isel(
                sample=np.flatnonzero(
                    np.isin(examples.station_id.values, list(heldout))
                )
            )
        rows = []
        for variable, (model, weights) in models.items():
            hourly_errors = [float("nan")] * 8
            for lead in range(1, 9):
                mask = (examples.variable.values == variable) & (
                    examples.lead.values == lead
                )
                selected = examples.isel(sample=np.flatnonzero(mask))
                if not selected.sizes["sample"]:
                    continue
                x, target = (
                    np.asarray(selected.features.values),
                    np.asarray(selected.target.values),
                )
                raw = x[:, 0] + model.predict(x)
                hourly_errors[lead - 1] = float(
                    np.abs(
                        x[:, 0] + weights[lead - 1] * (raw - x[:, 0]) - target
                    ).mean()
                )
                rows.append(
                    {
                        "variable": variable,
                        "lead_hour": lead,
                        "aurora": metrics(x[:, 0], target),
                        "raw_calibrated": metrics(raw, target),
                        "combined": metrics(
                            x[:, 0] + weights[lead - 1] * (raw - x[:, 0]), target
                        ),
                        "persistence": metrics(
                            np.asarray(selected.persistence.values), target
                        ),
                    }
                )
            case_errors.setdefault(variable, []).append(hourly_errors)
        reports.append({"issuance": case.issuance.isoformat(), "by_lead_hour": rows})
    return {
        "test": reports,
        "heldout_stations": sorted(heldout),
        "uncertainty": {
            name: bootstrap_cases(np.asarray(values), config.seed)
            for name, values in case_errors.items()
        },
        "seasonal_coverage": {
            split: sorted({case.issuance.month for case in records})
            for split, records in cases.items()
        },
        "mode": "retrospective unless availability recorded",
        "scientific_skill_verified": False,
    }
