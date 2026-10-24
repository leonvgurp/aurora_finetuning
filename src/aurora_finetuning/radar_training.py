"""Bounded training and held-out evaluation of the radar-conditioned rain adapter."""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from aurora_finetuning.acquisition import sha256
from aurora_finetuning.config import Config
from aurora_finetuning.grid import FloatArray
from aurora_finetuning.model import check_device, check_memory
from aurora_finetuning.radar_data import history_at, validate_native
from aurora_finetuning.radar_model import (
    ARCHITECTURE,
    RADIUS,
    RadarRainAdapter,
    baseline_at,
    feature_tensors,
    load_radar_adapter,
    radar_loss,
)
from aurora_finetuning.telemetry import Recorder, write_json


@dataclass(frozen=True)
class RadarCase:
    """Paths and the strict observation cutoff for one retrospective experiment."""

    initialization: datetime
    forecast: Path
    radar: Path


def read_radar_cases(path: Path) -> dict[str, list[RadarCase]]:
    """Resolve manifest paths and require separated chronological splits."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or set(raw) != {"train", "validation", "test"}:
        raise ValueError("Radar cases require train, validation and test arrays")
    cases: dict[str, list[RadarCase]] = {}
    for split in ("train", "validation", "test"):
        if not isinstance(raw[split], list) or not raw[split]:
            raise ValueError(f"Radar {split} cases must be a nonempty array")
        entries: list[RadarCase] = []
        for item in raw[split]:
            if (
                not isinstance(item, dict)
                or set(item) != {"initialization", "forecast", "radar"}
                or not all(isinstance(v, str) for v in item.values())
            ):
                raise ValueError(
                    "Each radar case needs initialization, forecast and radar strings"
                )
            time = datetime.fromisoformat(item["initialization"])
            if (
                time.tzinfo is not None
                or time.minute % 5
                or time.second
                or time.microsecond
            ):
                raise ValueError(
                    "Radar cutoffs must be naive UTC on five-minute boundaries"
                )
            entries.append(
                RadarCase(
                    time,
                    (path.parent / item["forecast"]).resolve(),
                    (path.parent / item["radar"]).resolve(),
                )
            )
        if len({entry.initialization for entry in entries}) != len(entries):
            raise ValueError("Duplicate radar case cutoffs")
        cases[split] = sorted(entries, key=lambda entry: entry.initialization)
        for previous, following in pairwise(cases[split]):
            if following.initialization - previous.initialization <= timedelta(
                hours=40
            ):
                raise ValueError(
                    "Radar cases require >30h separation of complete windows"
                )
    for before, after in (("train", "validation"), ("validation", "test")):
        if cases[before][-1].initialization + timedelta(hours=8 + 30) >= cases[after][
            0
        ].initialization - timedelta(hours=2):
            raise ValueError(
                "Radar splits require >30h separation of history/target windows"
            )
    return cases


def validate_case(
    case: RadarCase, config: Config, base_adapter: Path | None, recorder: Recorder
) -> None:
    """Validate training/validation windows without reading held-out test data."""
    for path in (case.forecast, case.radar):
        recorder.inputs[str(path)] = sha256(path)
    with (
        xr.open_dataset(case.forecast) as forecast,
        xr.open_dataset(case.radar) as radar,
    ):
        validate_native(radar)
        if min(radar.sizes["y"], radar.sizes["x"]) < config.radar_patch_size:
            raise ValueError("Native radar dataset is smaller than radar_patch_size")
        initialization = datetime.fromisoformat(
            str(forecast.attrs["initialization_utc"])
        )
        if initialization.tzinfo is not None or not timedelta(
            0
        ) <= case.initialization - initialization <= timedelta(hours=12):
            raise ValueError(
                "Aurora initialization must be at or before the radar cutoff "
                "and no more than 12 hours earlier"
            )
        if (
            forecast.attrs.get("checkpoint_revision") != config.checkpoint_revision
            or forecast.attrs.get("resolution") != config.resolution
            or forecast.attrs.get("adapter_sha256", "")
            != (sha256(base_adapter) if base_adapter else "")
            or forecast.precipitation_mm.attrs.get("units") != "mm"
        ):
            raise ValueError(
                "Aurora forecast has incompatible model, adapter, resolution or units"
            )
        history_at(radar, case.initialization)
        wanted = [
            np.datetime64(case.initialization + timedelta(minutes=5 * i), "ns")
            for i in range(1, 97)
        ]
        if not all(time in radar.time.values for time in wanted):
            raise ValueError(
                "Radar case requires eight hours of subsequent measured targets"
            )
        # Check spatial and temporal baseline coverage on a small native-grid slice.
        for minute in (5, 480):
            values = baseline_at(
                forecast,
                radar.isel(y=slice(None, None, 16), x=slice(None, None, 16)),
                case.initialization + timedelta(minutes=minute),
            )
            if not np.isfinite(values.values).all():
                raise ValueError(
                    "Aurora baseline does not cover the native radar patch"
                )


@dataclass
class PreparedCase:
    """One bounded patch with reusable radar history and Aurora interpolation."""

    history: xr.DataArray
    geometry: xr.Dataset
    targets: FloatArray
    baselines: FloatArray
    age: float

    def tensors(
        self, minute: int, config: Config
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct one lead from cached observations and hourly baseline maps."""
        baseline = self.geometry.latitude.copy(
            data=self.baselines[minute // 5 - 1] / 12
        )
        features, base = feature_tensors(
            self.history, baseline, minute, self.age, config
        )
        target = torch.tensor(
            self.targets[minute // 5 - 1][None],
            dtype=torch.float32,
            device=config.device,
        )
        return features, base, target


def prepare_case(
    case: RadarCase, config: Config, origin: tuple[int, int] | None = None
) -> PreparedCase:
    """Read a patch once; map eight hourly Aurora fields onto its radar geometry."""
    with (
        xr.open_dataset(case.radar) as radar,
        xr.open_dataset(case.forecast) as forecast,
    ):
        size = config.radar_patch_size
        row, col = origin or (
            (radar.sizes["y"] - size) // 2,
            (radar.sizes["x"] - size) // 2,
        )
        patch = radar.isel(y=slice(row, row + size), x=slice(col, col + size)).load()
        history = history_at(patch, case.initialization)
        target_times = [
            np.datetime64(case.initialization + timedelta(minutes=i), "ns")
            for i in range(5, 481, 5)
        ]
        targets = np.asarray(
            patch.precipitation_mm.sel(time=target_times).values, dtype=float
        )
        if np.isinf(targets).any() or np.any(np.isfinite(targets) & (targets < 0)):
            raise ValueError("Invalid measured radar targets")
        # A cutoff may fall between clock hours; compute all containing hours exactly.
        mapping: dict[datetime, FloatArray] = {}
        maps = []
        for minute in range(5, 481, 5):
            valid = case.initialization + timedelta(minutes=minute)
            end = valid.replace(minute=0, second=0, microsecond=0)
            if valid != end:
                end += timedelta(hours=1)
            if end not in mapping:
                mapping[end] = (
                    np.asarray(baseline_at(forecast, patch, valid).values, dtype=float)
                    * 12
                )
            maps.append(mapping[end])
        age = (
            case.initialization
            - datetime.fromisoformat(str(forecast.attrs["initialization_utc"]))
        ).total_seconds() / 3600
        return PreparedCase(history, patch, targets, np.stack(maps), age)


def tile_origins(case: RadarCase, config: Config) -> list[tuple[int, int]]:
    """Cover deterministic interiors without counting overlapping edge tiles twice."""
    size = config.radar_patch_size
    stride = size - 2 * RADIUS
    if stride < 1:
        raise ValueError("Radar patches must have a nonempty context interior")
    with xr.open_dataset(case.radar) as radar:
        axes = []
        for name in ("y", "x"):
            maximum = radar.sizes[name] - size
            if maximum < 0:
                raise ValueError("Radar dataset smaller than configured tile")
            origins = list(range(0, maximum + 1, stride))
            if origins[-1] != maximum:
                origins.append(maximum)
            axes.append(origins)
    return [(row, col) for row in axes[0] for col in axes[1]]


def validation_predictions(
    model: RadarRainAdapter, records: list[PreparedCase], config: Config
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Gather all eight hours without opening target files during forward passes."""
    predicted, baseline, observed = [], [], []
    with torch.inference_mode():
        for record in records:
            steps = []
            for minute in range(5, 481, 5):
                features, base, _ = record.tensors(minute, config)
                steps.append(
                    model(features, base)[0, RADIUS:-RADIUS, RADIUS:-RADIUS]
                    .cpu()
                    .numpy()
                )
            shape = steps[0].shape
            predicted.append(np.stack(steps).reshape(8, 12, *shape).sum(1))
            baseline.append(
                record.baselines[:, RADIUS:-RADIUS, RADIUS:-RADIUS]
                .reshape(8, 12, *shape)
                .sum(1)
                / 12
            )
            observed.append(
                record.targets[:, RADIUS:-RADIUS, RADIUS:-RADIUS]
                .reshape(8, 12, *shape)
                .sum(1)
            )
    return np.stack(predicted), np.stack(baseline), np.stack(observed)


def select_weights(
    predicted: FloatArray, baseline: FloatArray, observed: FloatArray
) -> list[float]:
    """Select per-hour weights on validation MAE with a CSI non-regression gate."""
    from aurora_finetuning.rainfall_metrics import event_scores

    weights = []
    for hour in range(8):
        base = baseline[:, hour]
        raw = predicted[:, hour]
        target = observed[:, hour]
        reference = event_scores(base, target, 0.1)
        best = (
            float(reference["mae"]) if reference["mae"] is not None else float("inf"),
            0.0,
        )
        for weight in (0.25, 0.5, 0.75, 1.0):
            scores = event_scores(base + weight * (raw - base), target, 0.1)
            csi, base_csi = scores["csi"], reference["csi"]
            if scores["mae"] is not None and (
                base_csi is None or (csi is not None and csi >= base_csi)
            ):
                candidate = (float(scores["mae"]), weight)
                best = min(best, candidate)
        weights.append(best[1])
    return weights


def train_radar(
    config: Config,
    manifest: Path,
    output: Path,
    recorder: Recorder,
    steps: int | None = None,
    base_adapter: Path | None = None,
) -> dict[str, object]:
    """Compare calendar ablations and radar-only control on validation, never test."""
    from aurora_finetuning.artifacts import atomic_output

    if base_adapter is not None:
        raise ValueError("Legacy surface-head adapters are unsupported")
    if output.exists():
        raise FileExistsError(output)
    cases = read_radar_cases(manifest)
    count = config.max_steps if steps is None else steps
    if count < 1:
        raise ValueError("Training steps must be positive")
    check_device(config)
    for case in cases["train"] + cases["validation"]:
        validate_case(case, config, None, recorder)
    validation = [prepare_case(case, config) for case in cases["validation"]]
    # Keep only one training case resident at a time.
    output.mkdir(parents=True)
    candidates = []
    for radar_only in (False, True):
        best_variant: tuple[float, Path] | None = None
        for seasonal in (False, True):
            torch.manual_seed(config.seed)
            rng = np.random.default_rng(config.seed)
            model = RadarRainAdapter(seasonal, radar_only).to(config.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            best_score = float("inf")
            selected_state: dict[str, torch.Tensor] = {}
            history = []
            for step in range(count):
                case = cases["train"][step % len(cases["train"])]
                origins = tile_origins(case, config)
                origin = origins[int(rng.integers(len(origins)))]
                record = prepare_case(case, config, origin)
                minute = int(rng.integers(1, 97)) * 5
                model.train()
                with recorder.measure("radar.training_step"):
                    features, baseline, target = record.tensors(minute, config)
                    optimizer.zero_grad(set_to_none=True)
                    loss = radar_loss(model(features, baseline), target, config)
                    torch.autograd.backward(loss)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                model.eval()
                # Validation cadence bounds repeated inference for larger pilots.
                if (step + 1) % max(1, count // 5) == 0 or step + 1 == count:
                    with recorder.measure("radar.validation"):
                        p, b, o = validation_predictions(model, validation, config)
                    score = float(np.nanmean(np.abs(p - o)))
                    if not np.isfinite(score):
                        raise ValueError("Nonfinite validation score")
                    if score < best_score:
                        best_score = score
                        selected_state = {
                            name: tensor.detach().cpu().clone()
                            for name, tensor in model.state_dict().items()
                        }
                    history.append(
                        {
                            "step": step + 1,
                            "training_loss": float(loss.detach().cpu()),
                            "validation_mae": score,
                        }
                    )
                check_memory(config)
            model.load_state_dict(selected_state)
            model.eval()
            p, b, o = validation_predictions(model, validation, config)
            model.lead_weights = select_weights(p, b, o)
            path = (
                output / f"{'radar-only' if radar_only else 'conditioned'}"
                f"-seasonal-{seasonal}.pt"
            )
            with atomic_output(path) as temporary:
                torch.save(
                    {
                        "architecture": ARCHITECTURE,
                        "parameters": selected_state,
                        "base_revision": config.checkpoint_revision,
                        "base_adapter_sha256": None,
                        "base_resolution": config.resolution,
                        "precipitation_scale": config.precipitation_scale,
                        "trained_steps": count,
                        "seed": config.seed,
                        "seasonal": seasonal,
                        "radar_only": radar_only,
                        "lead_weights": model.lead_weights,
                    },
                    temporary,
                )
            write_json(
                path.with_suffix(".json"),
                {"sha256": sha256(path), "architecture": ARCHITECTURE},
            )
            candidates.append(
                {
                    "file": path.name,
                    "radar_only": radar_only,
                    "seasonal": seasonal,
                    "validation_mae": best_score,
                    "weights": model.lead_weights,
                    "history": history,
                }
            )
            candidate = (best_score, path)
            if best_variant is None or candidate[0] < best_variant[0]:
                best_variant = candidate
        if best_variant is None:
            raise ValueError("No valid radar model")
        selected = output / ("radar-only.pt" if radar_only else "best.pt")
        with atomic_output(selected) as temporary:
            temporary.write_bytes(best_variant[1].read_bytes())
        write_json(
            selected.with_suffix(".json"),
            {"sha256": sha256(selected), "architecture": ARCHITECTURE},
        )
    report: dict[str, object] = {
        "mode": "frozen Aurora with external radar decoder",
        "candidates": candidates,
        "history": candidates[0]["history"],
        "test_evaluated": False,
        "input_sha256": recorder.inputs,
        "timings": recorder.report(),
        "validation_sampling": (
            "central patch; all 96 leads; evaluate uses tiled test interiors"
        ),
    }
    write_json(output / "training.json", report)
    return report


def evaluate_adapter(
    config: Config,
    manifest: Path,
    adapter: Path,
    recorder: Recorder,
    base_adapter: Path | None = None,
) -> dict[str, object]:
    """Score deterministic test tiles, all hours, raw and selected refinements."""
    from aurora_finetuning.grid import make_grid
    from aurora_finetuning.rainfall_metrics import (
        bootstrap_cases,
        coarse_grid_report,
        hourly_report,
        motion_forecast,
    )

    cases = read_radar_cases(manifest)["test"]
    model = load_radar_adapter(adapter, config, base_adapter, recorder)
    radar_only_path = adapter.parent / "radar-only.pt"
    control = (
        load_radar_adapter(radar_only_path, config, None, recorder)
        if radar_only_path.exists()
        else None
    )
    output = []
    case_errors: dict[str, list[list[float]]] = {}
    for case in cases:
        validate_case(case, config, base_adapter, recorder)
        tiles = []
        with xr.open_dataset(case.radar) as source:
            height, width = source.sizes["y"], source.sizes["x"]
        arrays = {
            name: np.full((8, height, width), np.nan)
            for name in (
                "raw_adapter",
                "radar_conditioned",
                "aurora",
                "radar_persistence",
                "radar_motion",
                "radar_only",
            )
        }
        observations = np.full((8, height, width), np.nan)
        for origin in tile_origins(case, config):
            record = prepare_case(case, config, origin)
            with recorder.measure("radar.test_tile"):
                predicted, baseline, observed = validation_predictions(
                    model, [record], config
                )
                only = (
                    validation_predictions(control, [record], config)[0][0]
                    if control
                    else None
                )
            p, b, o = predicted[0], baseline[0], observed[0]
            history = np.asarray(record.history.values, dtype=float)
            persistence = np.broadcast_to(
                history[-1, RADIUS:-RADIUS, RADIUS:-RADIUS] * 12, p.shape
            )
            motion = (
                np.stack(
                    [
                        motion_forecast(history, minute)[RADIUS:-RADIUS, RADIUS:-RADIUS]
                        for minute in range(5, 481, 5)
                    ]
                )
                .reshape(8, 12, *p.shape[1:])
                .sum(1)
            )
            row, col = origin
            selection = (
                slice(None),
                slice(row + RADIUS, row + config.radar_patch_size - RADIUS),
                slice(col + RADIUS, col + config.radar_patch_size - RADIUS),
            )
            candidates = {
                "raw_adapter": p,
                "radar_conditioned": b
                + np.array(model.lead_weights)[:, None, None] * (p - b),
                "aurora": b,
                "radar_persistence": persistence,
                "radar_motion": motion,
            }
            if only is not None:
                candidates["radar_only"] = only
            for name, values in candidates.items():
                arrays[name][selection] = values
            observations[selection] = o
            tiles.append(list(origin))
        if control is None:
            del arrays["radar_only"]
        # Keep motion's missing coverage separate from the other methods.
        mask = np.isfinite(observations)
        for name, values in arrays.items():
            if name != "radar_motion":
                mask &= np.isfinite(values)
        measured = np.where(mask, observations, np.nan)
        row_report: dict[str, object] = {
            "cutoff": case.initialization.isoformat(),
            "tiles": tiles,
        }
        with xr.open_dataset(case.radar) as geometry:
            lat = np.asarray(geometry.latitude.values, dtype=float)
            lon = np.asarray(geometry.longitude.values, dtype=float)
        grid = make_grid(config)
        for name, values in arrays.items():
            values = np.where(mask, values, np.nan)
            support = np.isfinite(values) & np.isfinite(measured)
            report = hourly_report(values, measured)
            if name == "radar_motion":
                paired_aurora = np.where(support, arrays["aurora"], np.nan)
                paired_observations = np.where(support, measured, np.nan)
                paired_report = hourly_report(paired_aurora, paired_observations)
                paired_report["aurora_grid"] = coarse_grid_report(
                    paired_aurora,
                    paired_observations,
                    lat,
                    lon,
                    grid,
                    config.resolution,
                )
                report["aurora_on_motion_support"] = paired_report
                report["support_note"] = (
                    "Motion has limited spatial support; "
                    "paired Aurora uses identical support"
                )
            report["aurora_grid"] = coarse_grid_report(
                values, measured, lat, lon, grid, config.resolution
            )
            row_report[name] = report
            case_errors.setdefault(name, []).append(
                [
                    float(
                        np.abs(values[i][support[i]] - measured[i][support[i]]).mean()
                    )
                    if support[i].any()
                    else float("nan")
                    for i in range(8)
                ]
            )
        output.append(row_report)
        check_memory(config)
    return {
        "test": output,
        "spatial_sampling": "tiled interiors; shared masks, motion paired separately",
        "uncertainty": {
            name: bootstrap_cases(np.asarray(values), config.seed)
            for name, values in case_errors.items()
        },
        "mode": "retrospective; historical publication latency not reconstructed",
        "scientific_skill_verified": False,
        "input_sha256": recorder.inputs,
    }
