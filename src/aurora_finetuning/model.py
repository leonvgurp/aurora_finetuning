"""Aurora loading and timed hourly regional inference."""

import json
import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from aurora import AuroraV1p5, Batch, rollout

from aurora_finetuning.assets import CHECKPOINT, verify_assets
from aurora_finetuning.config import Config
from aurora_finetuning.grid import FloatArray
from aurora_finetuning.telemetry import Recorder, peak_rss_gb

LOGGER = logging.getLogger(__name__)


def precipitation_output(values_mm: FloatArray) -> tuple[FloatArray, int, float]:
    """Apply a physical zero floor and retain diagnostics for raw negative amounts."""
    if not np.isfinite(values_mm).all():
        raise ValueError("Model produced nonfinite precipitation")
    count = int((values_mm < 0).sum())
    minimum = float(values_mm.min())
    clipped: FloatArray = np.maximum(values_mm, 0)
    return clipped, count, minimum


def synchronize(device: str) -> None:
    """Wait for accelerator kernels before reading wall-clock times."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def accelerator_memory(device: str) -> tuple[float | None, float | None]:
    """Report current accelerator allocations, not an instantaneous total RSS."""
    if device == "mps":
        return (
            torch.mps.current_allocated_memory() / 1024**3,
            torch.mps.driver_allocated_memory() / 1024**3,
        )
    if device == "cuda":
        return (
            torch.cuda.memory_allocated() / 1024**3,
            torch.cuda.memory_reserved() / 1024**3,
        )
    return None, None


def check_device(config: Config) -> None:
    """Reject unavailable devices; never silently switch to CPU."""
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError(
            "MPS unavailable; explicitly select device='cpu' for a CPU run"
        )
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")


def check_memory(config: Config) -> None:
    """Enforce a post-stage process RSS budget (not an OS hard memory limit)."""
    if peak_rss_gb() > config.max_memory_gb:
        raise ValueError(
            "Process peak RSS exceeded max_memory_gb; stop before larger runs"
        )


def load_model(
    config: Config, recorder: Recorder, adapter: Path | None = None
) -> AuroraV1p5:
    """Load verified frozen Aurora weights; reject legacy surface-head adapters."""
    if adapter is not None:
        raise ValueError(
            "Legacy surface-head adapters are unsupported; train external models"
        )
    check_device(config)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    with recorder.measure("model.verify_assets"):
        verify_assets(config)
    with recorder.measure("model.load"):
        model = AuroraV1p5(
            autocast=config.precision == "float16" and config.device != "mps",
            use_fp16_safe_attention=config.precision == "float16",
        )
        model.load_checkpoint_local(str(config.root / "checkpoints/base" / CHECKPOINT))
        model.eval()
        model.requires_grad_(False)
    check_memory(config)
    recorder.accelerator_memory = lambda: accelerator_memory(config.device)
    with recorder.measure("model.device_transfer"):
        model.to(config.device)
        synchronize(config.device)
    recorder.synchronize = lambda: synchronize(config.device)
    check_memory(config)
    return model


def infer(
    model: AuroraV1p5, batch: Batch, hours: int, config: Config, recorder: Recorder
) -> xr.Dataset:
    """Run hourly substeps and retain requested physical surface fields on CPU."""
    if not 1 <= hours <= 48:
        raise ValueError("Internal rollout horizon must be between 1 and 48 hours")
    fields: list[xr.Dataset] = []
    negative_count = 0
    minimum_raw = 0.0
    generator = rollout(
        model,
        batch,
        steps=math.ceil(hours / 6),
        fine_lead_times=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    with torch.inference_mode():
        for hour in range(1, hours + 1):
            with recorder.measure(f"inference.hour_{hour}"):
                # Upstream chooses CPU autocast on MPS; select the actual device.
                with torch.autocast(
                    device_type=config.device,
                    dtype=torch.float16,
                    enabled=config.precision == "float16",
                ):
                    prediction = next(generator)
                temp = prediction.surf_vars["2t"][0, 0].detach().cpu().numpy() - 273.15
                # Upstream returns log-untransformed water depth in metres.
                precip = np.asarray(
                    prediction.surf_vars["scaled_tp_1h"][0, 0].detach().cpu().numpy()
                    * 1000,
                    dtype=np.float64,
                )
                if not np.isfinite(temp).all() or not np.isfinite(precip).all():
                    raise ValueError("Model produced nonfinite predictions")
                precip, count, minimum = precipitation_output(precip)
                negative_count += count
                minimum_raw = min(minimum_raw, minimum)
                if count:
                    LOGGER.warning(
                        "Hour %s: floored %s negative precipitation cells; "
                        "raw minimum %.6f mm",
                        hour,
                        count,
                        minimum,
                    )
                fields.append(
                    xr.Dataset(
                        {
                            "temperature_c": (
                                ("time", "latitude", "longitude"),
                                temp[None],
                            ),
                            **{
                                name: (
                                    ("time", "latitude", "longitude"),
                                    prediction.surf_vars[key][0, 0]
                                    .detach()
                                    .cpu()
                                    .numpy()[None]
                                    / scale,
                                )
                                for name, key, scale in (
                                    ("pressure_hpa", "msl", 100),
                                    ("wind_u_ms", "10u", 1),
                                    ("wind_v_ms", "10v", 1),
                                )
                            },
                            "precipitation_mm": (
                                ("time", "latitude", "longitude"),
                                np.maximum(precip, 0)[None],
                            ),
                        },
                        coords={
                            "time": [np.datetime64(prediction.metadata.time[0], "ns")],
                            "latitude": prediction.metadata.lat.cpu().numpy(),
                            "longitude": prediction.metadata.lon.cpu().numpy(),
                        },
                    )
                )
            check_memory(config)
    generator.close()
    result = xr.concat(fields, dim="time")
    result.attrs.update(
        initialization_utc=batch.metadata.time[0].isoformat(),
        source="Aurora 1.5 regional experiment",
        resolution=config.resolution,
        precipitation_interval="(valid_time - 1 hour, valid_time]",
        experimental="true",
        seed=config.seed,
        checkpoint_revision=config.checkpoint_revision,
        regional_convention="continuous grid, canonical positions, no cyclic wrap",
        input_sha256=json.dumps(recorder.inputs, sort_keys=True),
        negative_precipitation_cells_floored=negative_count,
        raw_minimum_precipitation_mm=minimum_raw,
        precipitation_postprocessing="zero floor; nonfinite values fail",
    )
    for name, unit in (
        ("pressure_hpa", "hPa"),
        ("wind_u_ms", "m/s"),
        ("wind_v_ms", "m/s"),
    ):
        if not np.isfinite(result[name]).all():
            raise ValueError(f"Nonfinite forecast {name}")
        result[name].attrs["units"] = unit
    result.attrs["precision"] = config.precision
    result.temperature_c.attrs["units"] = "degC"
    result.precipitation_mm.attrs["units"] = "mm"
    return result


def save_forecast(dataset: xr.Dataset, path: Path) -> None:
    """Persist gridded predictions without silently overwriting previous results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    from aurora_finetuning.artifacts import atomic_output

    with atomic_output(path) as temporary:
        dataset.to_netcdf(temporary, mode="w")


def initialization_of(dataset: xr.Dataset) -> datetime:
    """Read the UTC initialization from a prediction artifact."""
    return datetime.fromisoformat(str(dataset.attrs["initialization_utc"]))
