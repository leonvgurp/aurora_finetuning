"""A small trainable native-grid rain decoder conditioned on measured radar history."""

import json
import logging
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
MAX_LEAD_MINUTES = 360
ARCHITECTURE = "radar-rain-v1"


class RadarRainAdapter(nn.Module):
    """Refine Aurora's rainfall into five-minute amounts using two hours of radar."""

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = 2 * HISTORY_FRAMES + 3
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
            "Radar features require 24 past frames and a 5–360 minute lead"
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
    planes = np.stack(
        [
            np.log1p(base * 12 / config.precipitation_scale),
            np.full_like(base, lead_minutes / MAX_LEAD_MINUTES),
            np.full_like(base, input_age_hours / 12),
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
    model = RadarRainAdapter().to(config.device)
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
                "Aurora; beyond radar adapter's six-hour range"
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
        output["precipitation_mm"] = round(amount, 3)
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
    model = load_radar_adapter(path, config, base_adapter, recorder)
    dataset = live_history(config, latitude, longitude, recorder, now)
    with recorder.measure("radar.conditioned_prediction"):
        result = refine_location(result, forecast, dataset, model, config)
    result["radar_adapter_sha256"] = sha256(path)
    logging.getLogger(__name__).warning(
        "Experimental radar-conditioned rainfall; measured cutoff %s UTC; "
        "rows beyond six hours from that cutoff retain Aurora rainfall",
        result["radar_cutoff_utc"],
    )
    return result
