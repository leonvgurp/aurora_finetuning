"""A small trainable native-grid rain decoder conditioned on measured radar history."""

import hashlib
import json
import logging
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch import nn

from aurora_finetuning.acquisition import sha256
from aurora_finetuning.config import Config
from aurora_finetuning.radar_data import HISTORY_FRAMES, history_at, live_history
from aurora_finetuning.telemetry import Recorder

RADIUS = 63  # Six dilated 3x3 layers: receptive field 127 native (1 km) pixels.
MAX_LEAD_MINUTES = 480
ARCHITECTURE = "radar-rain-v2"


class RadarRainAdapter(nn.Module):
    """Refine Aurora's rainfall into five-minute amounts using two hours of radar."""

    def __init__(self, seasonal: bool = True, radar_only: bool = False) -> None:
        super().__init__()
        self.seasonal = seasonal
        self.radar_only = radar_only
        self.lead_weights = [1.0] * 8
        layers: list[nn.Module] = []
        channels = 2 * HISTORY_FRAMES + 7
        for dilation in (1, 2, 4, 8, 16, 32):
            layers.extend(
                [
                    nn.Conv2d(channels, 8, 3, padding=dilation, dilation=dilation),
                    nn.SiLU(),
                ]
            )
            channels = 8
        self.encoder = nn.Sequential(*layers)
        self.output = nn.Conv2d(8, 1, 1)
        nn.init.zeros_(self.output.weight)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        """Return nonnegative five-minute amounts; zero residual preserves Aurora."""
        features = features.clone()
        if not self.seasonal:
            features[:, -4:] = 0
        if self.radar_only:
            features[:, 2 * HISTORY_FRAMES] = 0
            baseline = torch.zeros_like(baseline)
        residual = self.output(self.encoder(features))[:, 0]
        return torch.expm1((torch.log1p(baseline * 12) + residual).clamp(min=0)) / 12


def baseline_at(
    forecast: xr.Dataset, geometry: xr.Dataset, valid: datetime
) -> xr.DataArray:
    """Map the containing Aurora hourly accumulation to native radar cell centres."""
    initialization = datetime.fromisoformat(str(forecast.attrs["initialization_utc"]))
    if initialization.tzinfo is not None or initialization >= valid:
        raise ValueError(
            "Aurora baseline must be initialized before its predicted interval"
        )
    end = valid.replace(minute=0, second=0, microsecond=0)
    if valid != end:
        end += timedelta(hours=1)
    try:
        baseline = forecast.precipitation_mm.sel(time=np.datetime64(end, "ns"))
    except KeyError as error:
        raise ValueError(
            "Aurora baseline does not cover the radar prediction window"
        ) from error
    result: xr.DataArray = (
        baseline.interp(latitude=geometry.latitude, longitude=geometry.longitude) / 12
    )
    return result


def feature_tensors(
    history: xr.DataArray,
    baseline: xr.DataArray,
    lead_minutes: int,
    input_age_hours: float,
    config: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode amounts and coverage masks without treating missing targets as dry."""
    if (
        history.sizes["time"] != HISTORY_FRAMES
        or not 5 <= lead_minutes <= MAX_LEAD_MINUTES
    ):
        raise ValueError(
            "Radar features require 24 past frames and a 5–480 minute lead"
        )
    raw = np.asarray(history.values, dtype=np.float32)
    base = np.asarray(baseline.values, dtype=np.float32)
    if np.isinf(raw).any() or np.any(np.isfinite(raw) & (raw < 0)):
        raise ValueError("Radar history contains invalid rainfall")
    if (
        not np.isfinite(base).all()
        or np.any(base < 0)
        or not 0 <= input_age_hours <= 12
    ):
        raise ValueError(
            "Aurora baseline is missing, negative, future-dated or over 12 hours old"
        )
    if not np.isfinite(raw).all(axis=0).any():
        raise ValueError("Radar history has no pixel with complete coverage")
    mask = np.isfinite(raw)
    rain = np.log1p(np.nan_to_num(raw, nan=0) * 12 / config.precipitation_scale)
    from aurora_finetuning.seasonal import calendar_features

    cutoff = datetime.fromisoformat(
        str(history.time.values[-1].astype("datetime64[us]"))
    )
    seasonal = calendar_features(cutoff + timedelta(minutes=lead_minutes))
    planes = np.stack(
        [
            np.log1p(base * 12 / config.precipitation_scale),
            np.full_like(base, lead_minutes / MAX_LEAD_MINUTES),
            np.full_like(base, input_age_hours / 12),
            *(np.full_like(base, value) for value in seasonal),
        ]
    )
    features = np.concatenate([rain, mask.astype(np.float32), planes])
    return (
        torch.tensor(features[None], device=config.device),
        torch.tensor(base[None], device=config.device),
    )


def radar_loss(
    prediction: torch.Tensor, target: torch.Tensor, config: Config
) -> torch.Tensor:
    """Wet-weighted log-intensity loss on observed patch-interior pixels."""
    p = prediction[..., RADIUS:-RADIUS, RADIUS:-RADIUS]
    t = target[..., RADIUS:-RADIUS, RADIUS:-RADIUS]
    mask = torch.isfinite(t) & (t >= 0)
    if not mask.any():
        raise ValueError("No observed radar targets in the patch interior")
    if not torch.isfinite(p).all():
        raise ValueError("Nonfinite radar-adapter predictions")
    error = (
        torch.log1p(p[mask] * 12 / config.precipitation_scale)
        - torch.log1p(t[mask] * 12 / config.precipitation_scale)
    ).square()
    weight = torch.where(t[mask] * 12 >= 0.1, config.radar_wet_weight, 1.0)
    return (weight * error).sum() / weight.sum()


def load_radar_adapter(
    path: Path,
    config: Config,
    base_adapter: Path | None,
    recorder: Recorder,
) -> RadarRainAdapter:
    """Load trained weights compatible with the selected Aurora baseline."""
    manifest = json.loads(path.with_suffix(".json").read_text())
    actual = sha256(path)
    if manifest["sha256"] != actual:
        raise ValueError("Radar adapter checksum mismatch")
    if base_adapter is not None:
        raise ValueError("Legacy surface-head adapters are unsupported")
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected_adapter = sha256(base_adapter) if base_adapter is not None else None
    if (
        state["architecture"] != ARCHITECTURE
        or state["base_revision"] != config.checkpoint_revision
        or state["base_adapter_sha256"] != expected_adapter
        or state["base_resolution"] != config.resolution
        or state["precipitation_scale"] != config.precipitation_scale
        or state["trained_steps"] < 1
    ):
        raise ValueError(
            "Radar adapter is incompatible with the selected Aurora baseline/config"
        )
    model = RadarRainAdapter(
        seasonal=bool(state["seasonal"]),
        radar_only=bool(state.get("radar_only", False)),
    ).to(config.device)
    weights = state.get("lead_weights", [1.0] * 8)
    if len(weights) != 8 or any(w not in (0, 0.25, 0.5, 0.75, 1) for w in weights):
        raise ValueError("Invalid radar lead weights")
    model.lead_weights = [float(w) for w in weights]
    model.load_state_dict(state["parameters"], strict=True)
    model.eval()
    recorder.inputs[str(path)] = actual
    return model


def refine_location(
    result: dict[str, object],
    forecast: xr.Dataset,
    history_dataset: xr.Dataset,
    model: RadarRainAdapter,
    config: Config,
) -> dict[str, object]:
    """Integrate measured/predicted five-minute bins into complete clock-hour totals."""
    cutoff = datetime.fromisoformat(str(history_dataset.time.values[-1]))
    history = history_at(history_dataset, cutoff)
    row = history_dataset.sizes["y"] // 2
    col = history_dataset.sizes["x"] // 2
    if row < RADIUS or col < RADIUS:
        raise ValueError("Native radar history patch is too small")
    # The centred 127x127 receptive field makes inference independent of crop edges.
    selection = {
        "y": slice(row - RADIUS, row + RADIUS + 1),
        "x": slice(col - RADIUS, col + RADIUS + 1),
    }
    geometry = history_dataset.isel(selection)
    history = history.isel(selection)
    center = RADIUS
    initialization = datetime.fromisoformat(str(forecast.attrs["initialization_utc"]))
    age = (cutoff - initialization).total_seconds() / 3600
    rows = result["hourly"]
    if not isinstance(rows, list):
        raise ValueError("Invalid hourly forecast result")
    revised: list[dict[str, object]] = []
    for source_row in rows:
        if not isinstance(source_row, dict):
            raise ValueError("Invalid hourly forecast row")
        output = dict(source_row)
        end = datetime.fromisoformat(str(output["time"]))
        end = end.astimezone(UTC).replace(tzinfo=None)
        if end > cutoff + timedelta(minutes=MAX_LEAD_MINUTES):
            output["precipitation_source"] = (
                "Aurora; beyond radar adapter's eight-hour range"
            )
            revised.append(output)
            continue
        amount = 0.0
        for index in range(12):
            valid = end - timedelta(minutes=5 * (11 - index))
            if valid <= cutoff:
                try:
                    value = float(
                        history.sel(time=np.datetime64(valid, "ns"))[center, center]
                    )
                except KeyError as error:
                    raise ValueError(
                        "Radar history does not cover the first hour's elapsed part"
                    ) from error
                if not np.isfinite(value) or value < 0:
                    raise ValueError("Missing observed radar at the query location")
            else:
                baseline = baseline_at(forecast, geometry, valid)
                features, base = feature_tensors(
                    history,
                    baseline,
                    int((valid - cutoff).total_seconds() / 60),
                    age,
                    config,
                )
                with torch.inference_mode():
                    prediction = model(features, base)
                value = float(prediction[0, center, center].cpu())
                if not np.isfinite(value) or value < 0:
                    raise ValueError("Invalid radar-adapter precipitation")
            amount += value
        output["aurora_precipitation_mm"] = output.get(
            "aurora_precipitation_mm", output["precipitation_mm"]
        )
        output["raw_radar_precipitation_mm"] = amount
        lead = min(8, max(1, math.ceil((end - cutoff).total_seconds() / 3600)))
        weight = model.lead_weights[lead - 1]
        # Preserve measured prefix identically in baseline and refined totals.
        prefix = 0.0
        base_future = 0.0
        for index in range(12):
            valid = end - timedelta(minutes=5 * (11 - index))
            if valid <= cutoff:
                prefix += float(
                    history.sel(time=np.datetime64(valid, "ns"))[center, center]
                )
            else:
                base_future += float(
                    baseline_at(forecast, geometry, valid)[center, center]
                )
        output["precipitation_mm"] = round(
            prefix + base_future + weight * (amount - prefix - base_future), 3
        )
        output["measured_prefix_mm"] = prefix
        output["radar_refinement_weight"] = weight
        output["precipitation_source"] = "Aurora with measured radar conditioning"
        revised.append(output)
    return {
        **result,
        "hourly": revised,
        "radar_cutoff_utc": cutoff.isoformat(),
        "radar_pixel_size_km": 1,
        "radar_role": "past input; no KNMI forecasts",
    }


def apply_live_radar(
    result: dict[str, object],
    forecast: xr.Dataset,
    path: Path,
    config: Config,
    recorder: Recorder,
    now: datetime,
    latitude: float,
    longitude: float,
    base_adapter: Path | None,
) -> dict[str, object]:
    """Condition a live Aurora forecast on fresh measured radar and trained weights."""
    dataset = live_history(config, latitude, longitude, recorder, now)
    adapter_hash = sha256(path)
    identity = {
        "result": result,
        "adapter": adapter_hash,
        "adapter_manifest": sha256(path.with_suffix(".json")),
        "config": {
            "revision": config.checkpoint_revision,
            "resolution": config.resolution,
            "precipitation_scale": config.precipitation_scale,
            "device": config.device,
        },
        "observations": hashlib.sha256(
            dataset.precipitation_mm.values.tobytes()
        ).hexdigest(),
        "cutoff": str(dataset.time.values[-1]),
        "forecast": hashlib.sha256(
            forecast.precipitation_mm.values.tobytes()
        ).hexdigest(),
        "code": [
            sha256(Path(__file__)),
            sha256(Path(__file__).with_name("seasonal.py")),
        ],
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cached = config.root / "outputs/refinements/radar" / f"{key}.json"
    checksum = cached.with_suffix(".sha256.json")
    if cached.exists() or checksum.exists():
        if (
            not cached.exists()
            or not checksum.exists()
            or sha256(cached) != json.loads(checksum.read_text())["sha256"]
        ):
            raise ValueError("Incomplete or modified radar refinement cache")
        value = json.loads(cached.read_text())
        if not isinstance(value, dict):
            raise ValueError("Invalid radar refinement cache")
        return value
    model = load_radar_adapter(path, config, base_adapter, recorder)
    with recorder.measure("radar.conditioned_prediction"):
        result = refine_location(result, forecast, dataset, model, config)
    result["radar_adapter_sha256"] = adapter_hash
    from aurora_finetuning.telemetry import write_json

    write_json(cached, result)
    write_json(checksum, {"sha256": sha256(cached)})
    logging.getLogger(__name__).warning(
        "Experimental radar-conditioned rainfall; measured cutoff %s UTC; "
        "rows beyond eight hours from that cutoff retain Aurora rainfall",
        result["radar_cutoff_utc"],
    )
    return result
