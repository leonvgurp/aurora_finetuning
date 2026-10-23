"""Bounded surface-head fine-tuning with chronological validation."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
import xarray as xr
from aurora import AuroraV1p5

from aurora_finetuning.config import Config
from aurora_finetuning.grid import make_grid
from aurora_finetuning.model import check_memory, load_model
from aurora_finetuning.observations import load_stations
from aurora_finetuning.preparation import prepare_batch, target_fields
from aurora_finetuning.regional import regional_context
from aurora_finetuning.telemetry import Recorder, write_json


def read_cases(path: Path) -> dict[str, list[datetime]]:
    """Require separate chronological train/validation/test windows with a 30h gap."""
    raw = json.loads(path.read_text())
    if set(raw) != {"train", "validation", "test"}:
        raise ValueError("Cases must contain train, validation and test arrays")
    cases: dict[str, list[datetime]] = {}
    for split in ("train", "validation", "test"):
        entries = raw[split]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{split} must be a nonempty array")
        times = [datetime.fromisoformat(t) for t in entries]
        if any(
            t.tzinfo is not None or t.minute or t.second or t.microsecond for t in times
        ):
            raise ValueError("Case timestamps must be naive exact-hour UTC")
        if len(set(times)) != len(times):
            raise ValueError(f"Duplicate {split} cases")
        cases[split] = sorted(times)
    for before, after in (("train", "validation"), ("validation", "test")):
        if cases[before][-1] + timedelta(hours=30) >= cases[after][0]:
            raise ValueError(
                "Splits require >30h separation to avoid overlapping windows"
            )
    return cases


def trainable_surface_heads(model: AuroraV1p5) -> list[torch.nn.Parameter]:
    """Freeze the complete backbone and train decoder surface reconstruction heads."""
    parameters: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith(
            ("decoder.surf_heads.2t.", "decoder.surf_heads.scaled_tp_1h.")
        )
        parameter.requires_grad_(enabled)
        if enabled:
            parameters.append(parameter)
    if not parameters:
        raise ValueError("Pinned Aurora model has no expected surface heads")
    return parameters


def physical_loss(
    temperature: torch.Tensor,
    precipitation: torch.Tensor,
    target_temperature: torch.Tensor,
    target_precipitation: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    """Scale temperature and log1p rainfall errors using fixed physical scales."""
    mask_t = torch.isfinite(target_temperature)
    mask_p = torch.isfinite(target_precipitation) & (target_precipitation >= 0)
    if not mask_t.any() or not mask_p.any():
        raise ValueError("No valid target cells for one or more training variables")
    t = (
        ((temperature[mask_t] - target_temperature[mask_t]) / config.temperature_scale)
        .square()
        .mean()
    )
    p = (
        (
            torch.log1p(precipitation[mask_p].clamp(min=0) / config.precipitation_scale)
            - torch.log1p(target_precipitation[mask_p] / config.precipitation_scale)
        )
        .square()
        .mean()
    )
    return t + p


def station_loss(
    temperature: torch.Tensor,
    precipitation: torch.Tensor,
    stations: xr.Dataset,
    time: datetime,
    config: Config,
) -> torch.Tensor:
    """Sample differentiably at in-domain KNMI stations for local supervision."""
    grid = make_grid(config)
    mask = (
        (stations.latitude <= grid.latitude[0])
        & (stations.latitude >= grid.latitude[-1])
        & (stations.longitude >= grid.longitude[0])
        & (stations.longitude <= grid.longitude[-1])
    )
    selected = stations.isel(station=np.flatnonzero(mask.values))
    if selected.sizes["station"] == 0:
        raise ValueError("No in-domain KNMI training stations")
    try:
        selected = selected.sel(time=np.datetime64(time, "ns"))
    except KeyError as error:
        raise ValueError(f"Missing KNMI training observations for {time}") from error
    x = (selected.longitude.values - grid.longitude[0]) / (
        grid.longitude[-1] - grid.longitude[0]
    ) * 2 - 1
    y = (grid.latitude[0] - selected.latitude.values) / (
        grid.latitude[0] - grid.latitude[-1]
    ) * 2 - 1
    coordinates = torch.tensor(
        np.stack((x, y), axis=-1)[None, None],
        dtype=temperature.dtype,
        device=temperature.device,
    )
    sampled_t = functional.grid_sample(
        temperature[None, None], coordinates, mode="bilinear", align_corners=True
    ).flatten()
    sampled_p = functional.grid_sample(
        precipitation[None, None], coordinates, mode="nearest", align_corners=True
    ).flatten()
    target_t = torch.tensor(
        selected.temperature_c.values,
        dtype=temperature.dtype,
        device=temperature.device,
    )
    target_p = torch.tensor(
        selected.precipitation_mm.values,
        dtype=temperature.dtype,
        device=temperature.device,
    )
    return physical_loss(sampled_t, sampled_p, target_t, target_p, config)


def case_loss(
    model: AuroraV1p5,
    config: Config,
    time: datetime,
    recorder: Recorder,
    backward: bool,
    stations: xr.Dataset | None = None,
) -> float:
    """Supervise each of the six lead hours from the same two input states."""
    batch = prepare_batch(config, time, recorder)
    total = 0.0
    for lead in range(1, 7):
        with recorder.measure(f"training.targets.lead_{lead}"):
            target = target_fields(config, time + timedelta(hours=lead), recorder)
        with recorder.measure(
            f"training.{'backward' if backward else 'validation'}.lead_{lead}"
        ):
            prediction = model.forward(batch, lead_times=torch.tensor([float(lead)]))
            temperature = prediction.surf_vars["2t"][0, 0] - 273.15
            rain = prediction.surf_vars["scaled_tp_1h"][0, 0] * 1000
            target_t = torch.tensor(
                target.temperature_c.values[0],
                dtype=torch.float32,
                device=config.device,
            )
            target_p = torch.tensor(
                target.precipitation_mm.values[0],
                dtype=torch.float32,
                device=config.device,
            )
            loss = physical_loss(temperature, rain, target_t, target_p, config)
            if stations is not None:
                loss = loss + station_loss(
                    temperature, rain, stations, time + timedelta(hours=lead), config
                )
            loss = loss / 6
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            if backward:
                torch.autograd.backward(loss)
            total += float(loss.detach().cpu())
        check_memory(config)
    return total


def train(
    config: Config,
    cases_path: Path,
    output: Path,
    recorder: Recorder,
    steps: int | None = None,
    station_paths: list[Path] | None = None,
) -> dict[str, object]:
    """Run a bounded pilot, select on validation and never consume test targets."""
    cases = read_cases(cases_path)
    stations = load_stations(station_paths) if station_paths else None
    count = config.max_steps if steps is None else steps
    if count < 1:
        raise ValueError("Training steps must be positive")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    with regional_context():
        model = load_model(config, recorder)
        parameters = trainable_surface_heads(model)
        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
        best = float("inf")
        history: list[dict[str, float | int]] = []
        # Keep stochastic layers disabled; gradients are still enabled for the heads.
        model.eval()
        for step in range(count):
            optimizer.zero_grad(set_to_none=True)
            time = cases["train"][step % len(cases["train"])]
            training_loss = case_loss(
                model, config, time, recorder, backward=True, stations=stations
            )
            with recorder.measure("training.optimizer"):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
            with torch.no_grad():
                scores = [
                    case_loss(
                        model, config, t, recorder, backward=False, stations=stations
                    )
                    for t in cases["validation"]
                ]
            validation_loss = float(np.mean(scores))
            history.append(
                {
                    "step": step + 1,
                    "training_loss": training_loss,
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best:
                best = validation_loss
                state = {
                    name: p.detach().cpu().clone()
                    for name, p in model.named_parameters()
                    if p.requires_grad
                }
                revision = json.loads(
                    (config.root / "checkpoints/base/assets.json").read_text()
                )["revision"]
                # Replace best.pt only within this newly created run directory.
                torch.save(
                    {
                        "parameters": state,
                        "optimizer": optimizer.state_dict(),
                        "base_revision": revision,
                        "resolution": config.resolution,
                        "grid": make_grid(config).describe(),
                        "step": step + 1,
                        "seed": config.seed,
                    },
                    output / "best.pt",
                )
        report: dict[str, object] = {
            "history": history,
            "best_validation_loss": best,
            "training_mode": "surface heads only; frozen atmospheric backbone",
            "supervision": "ERA5 gridded temperature and precipitation",
            "knmi_role": "local supervision"
            if stations is not None
            else "evaluation only",
            "station_inputs": [str(p) for p in station_paths or []],
            "test_evaluated": False,
            "cases": json.loads(cases_path.read_text()),
            "timings": recorder.report(),
            "input_sha256": recorder.inputs,
        }
        write_json(output / "training.json", report)
    return report
