"""Explicit, bounded operational commands for the regional Aurora pipeline."""

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import requests

from aurora_finetuning.config import Config, load_config
from aurora_finetuning.telemetry import Recorder, write_json


def parser() -> argparse.ArgumentParser:
    """Build the CLI without loading PyTorch or credentials."""
    root = argparse.ArgumentParser(description="Aurora forecasting with KNMI data")
    root.add_argument(
        "--version", action="version", version=version("aurora-finetuning")
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path)
    common.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    common.add_argument("--device", choices=("cpu", "mps", "cuda"))
    common.add_argument(
        "--report", type=Path, help="Exclusive timing/provenance JSON output"
    )
    commands = root.add_subparsers(dest="command")
    commands.add_parser(
        "download-model", parents=[common], help="Download pinned Aurora 1.5 assets"
    )
    smoke = commands.add_parser(
        "smoke", parents=[common], help="Run an explicit feasibility gate"
    )
    smoke.add_argument(
        "--kind",
        choices=(
            "environment",
            "providers",
            "architecture",
            "weights",
            "data",
            "model",
        ),
        default="environment",
    )
    smoke.add_argument("--initialization", type=datetime.fromisoformat)
    smoke.add_argument(
        "--backward", action="store_true", help="Model gate: one backward pass"
    )
    cds = commands.add_parser(
        "pull-cds", parents=[common], help="Retrieve daily regional ERA5"
    )
    cds.add_argument("--start", type=date.fromisoformat, required=True)
    cds.add_argument("--end", type=date.fromisoformat)
    cds.add_argument(
        "--surface-only",
        action="store_true",
        help="Retrieve surface fields only; use explicit ARCO pressure import",
    )
    cds.add_argument(
        "--allow-bulk", action="store_true", help="Permit more than two days"
    )
    knmi = commands.add_parser(
        "pull-knmi", parents=[common], help="Retrieve a KNMI filename range"
    )
    knmi.add_argument(
        "--dataset", default="hourly-in-situ-meteorological-observations-validated"
    )
    knmi.add_argument("--dataset-version", default="1.0")
    knmi.add_argument(
        "--begin", required=True, help="Inclusive filename boundary (not date)"
    )
    knmi.add_argument("--end", required=True, help="Inclusive filename boundary")
    knmi.add_argument("--max-files", type=int, default=4)
    live = commands.add_parser(
        "pull-ifs",
        parents=[common],
        help="Strict live IFS field compatibility/download",
    )
    live.add_argument("--initialization", type=datetime.fromisoformat, required=True)
    live.add_argument(
        "--sea-ice",
        type=Path,
        help="Explicit ERA5 sea-ice supplement, at most seven days old",
    )
    live.add_argument(
        "--audit-only",
        action="store_true",
        help="Check small index files without downloading GRIB fields",
    )
    ice = commands.add_parser("pull-sea-ice", parents=[common])
    ice.add_argument("--date", type=date.fromisoformat, required=True)
    arco = commands.add_parser(
        "pull-arco-pressure",
        parents=[common],
        help="Explicit public ERA5 mirror for two input states",
    )
    arco.add_argument("--initialization", type=datetime.fromisoformat, required=True)
    for name in ("prepare", "hindcast"):
        p = commands.add_parser(name, parents=[common])
        p.add_argument("--initialization", type=datetime.fromisoformat, required=True)
        p.add_argument("--source", choices=("era5", "ifs"), default="era5")
        if name == "hindcast":
            p.add_argument("--hours", type=int)
            p.add_argument("--adapter", type=Path)
            p.add_argument("--output", type=Path, required=True)
    query = commands.add_parser(
        "forecast", parents=[common], help="Query hourly forecasts for NL coordinates"
    )
    query.add_argument("--latitude", type=float, required=True)
    query.add_argument("--longitude", type=float, required=True)
    query.add_argument("--hours", type=int)
    query.add_argument("--initialization", type=datetime.fromisoformat)
    query.add_argument(
        "--start", type=datetime.fromisoformat, help="Timezone-aware query start"
    )
    query.add_argument("--cached-forecast", type=Path)
    query.add_argument("--adapter", type=Path)
    query.add_argument("--source", choices=("era5", "ifs"), default="era5")
    training = commands.add_parser("train", parents=[common])
    training.add_argument("--cases", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--steps", type=int)
    training.add_argument(
        "--stations",
        type=Path,
        nargs="+",
        help="KNMI files for additional local training supervision",
    )
    evaluate = commands.add_parser("evaluate", parents=[common])
    evaluate.add_argument("--prediction", type=Path, required=True)
    evaluate.add_argument("--stations", type=Path, nargs="+")
    evaluate.add_argument("--radar", type=Path)
    radar = commands.add_parser("prepare-radar", parents=[common])
    radar.add_argument("--frames", type=Path, nargs="+", required=True)
    radar.add_argument("--geolocation", type=Path, required=True)
    radar.add_argument("--output", type=Path, required=True)
    history = commands.add_parser("history", parents=[common])
    history.add_argument("--stations", type=Path, nargs="+", required=True)
    history.add_argument("--station", required=True)
    history.add_argument("--today", type=date.fromisoformat, required=True)
    return root


def dispatch(args: argparse.Namespace, config: Config, recorder: Recorder) -> object:
    """Dispatch only the selected command, keeping heavy imports lazy."""
    if args.command == "download-model":
        from aurora_finetuning.assets import download_assets

        return {"manifest": str(download_assets(config, recorder))}
    if args.command == "pull-cds":
        from aurora_finetuning.acquisition import pull_cds

        end = args.end or args.start
        if (end - args.start).days > 1 and not args.allow_bulk:
            raise ValueError(
                "Smoke downloads are limited to two days; "
                "use --allow-bulk for larger pulls"
            )
        return {
            "files": [
                str(p)
                for p in pull_cds(
                    config, args.start, end, recorder, surface_only=args.surface_only
                )
            ]
        }
    if args.command == "pull-knmi":
        from aurora_finetuning.acquisition import knmi_files

        return {
            "files": [
                str(p)
                for p in knmi_files(
                    config,
                    args.dataset,
                    args.dataset_version,
                    args.begin,
                    args.end,
                    recorder,
                    args.max_files,
                )
            ]
        }
    if args.command == "pull-ifs":
        from aurora_finetuning.ifs import audit_ifs, pull_ifs

        if args.audit_only:
            return audit_ifs(args.initialization, recorder)

        return {
            "files": [
                str(p)
                for p in pull_ifs(config, args.initialization, recorder, args.sea_ice)
            ]
        }
    if args.command == "pull-sea-ice":
        from aurora_finetuning.ifs import pull_sea_ice

        return {"sea_ice": str(pull_sea_ice(config, args.date, recorder))}
    if args.command == "pull-arco-pressure":
        from aurora_finetuning.arco import pull_arco_pressure

        return {
            "files": [
                str(p)
                for p in pull_arco_pressure(config, args.initialization, recorder)
            ]
        }
    if args.command == "smoke" and args.kind in (
        "environment",
        "providers",
        "architecture",
    ):
        from aurora_finetuning.smoke import (
            architecture_smoke,
            preflight,
            provider_smoke,
        )

        if args.kind == "providers":
            return provider_smoke(recorder, config)

        return (
            preflight(config, recorder)
            if args.kind == "environment"
            else architecture_smoke(config.device, recorder)
        )
    if args.command == "smoke" and args.kind == "weights":
        from aurora_finetuning.model import load_model

        model = load_model(config, recorder)
        return {
            "full_checkpoint_loaded": True,
            "device": config.device,
            "parameters": sum(p.numel() for p in model.parameters()),
            "scientific_skill_verified": False,
        }
    if args.command in ("prepare", "hindcast", "smoke"):
        from aurora_finetuning.model import infer, load_model, save_forecast
        from aurora_finetuning.preparation import prepare_batch
        from aurora_finetuning.regional import regional_context

        if args.initialization is None:
            raise ValueError("Data/model smoke requires --initialization")
        with regional_context():
            batch = prepare_batch(
                config, args.initialization, recorder, getattr(args, "source", "era5")
            )
            if args.command == "prepare" or (
                args.command == "smoke" and args.kind == "data"
            ):
                return {"shape": list(batch.spatial_shape), "finite_inputs": True}
            model = load_model(config, recorder, getattr(args, "adapter", None))
            if args.command == "smoke" and args.backward:
                import torch

                from aurora_finetuning.training import trainable_surface_heads

                trainable_surface_heads(model)
                with recorder.measure("smoke.pretrained_forward_backward"):
                    prediction = model.forward(batch, lead_times=torch.tensor([1.0]))
                    loss = prediction.surf_vars["2t"].mean()
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite model smoke loss")
                    loss.backward()
                return {
                    "pretrained_backward_completed": True,
                    "scientific_skill_verified": False,
                }
            ds = infer(
                model,
                batch,
                (args.hours or config.hours) if args.command == "hindcast" else 1,
                config,
                recorder,
            )
            if args.command == "hindcast":
                with recorder.measure("forecast.write_netcdf"):
                    save_forecast(ds, args.output)
                return {"forecast": str(args.output), "hours": ds.sizes["time"]}
            return {
                "pretrained_forward_completed": True,
                "scientific_skill_verified": False,
            }
    if args.command == "forecast":
        from aurora_finetuning.forecast import forecast

        return forecast(
            args.latitude,
            args.longitude,
            args.hours or config.hours,
            args.config,
            initialization=args.initialization,
            start=args.start,
            cached_forecast=args.cached_forecast,
            adapter=args.adapter,
            source=args.source,
            device=args.device,
            recorder=recorder,
        )
    if args.command == "train":
        from aurora_finetuning.training import train

        return train(
            config, args.cases, args.output, recorder, args.steps, args.stations
        )
    if args.command == "evaluate":
        import xarray as xr

        from aurora_finetuning.evaluation import evaluate_radar, evaluate_stations
        from aurora_finetuning.observations import load_stations

        if not args.stations and not args.radar:
            raise ValueError("Supply --stations and/or --radar")
        result: dict[str, object] = {}
        with (
            recorder.measure("validation.score"),
            xr.open_dataset(args.prediction) as prediction,
        ):
            if args.stations:
                result["stations"] = evaluate_stations(
                    prediction, load_stations(args.stations)
                )
            if args.radar:
                with xr.open_dataset(args.radar) as radar:
                    result["radar"] = evaluate_radar(prediction, radar)
        return result
    if args.command == "prepare-radar":
        from aurora_finetuning.grid import make_grid
        from aurora_finetuning.model import save_forecast
        from aurora_finetuning.radar import prepare_radar

        with recorder.measure("prepare.radar"):
            ds = prepare_radar(
                args.frames, args.geolocation, make_grid(config), config.resolution
            )
            save_forecast(ds, args.output)
        return {"radar": str(args.output)}
    if args.command == "history":
        from aurora_finetuning.observations import load_stations, summarize_history

        with recorder.measure("history.summarize"):
            return summarize_history(
                load_stations(args.stations), args.station, args.today
            )
    raise ValueError("Unknown command")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one operation and persist timing/provenance even on failure."""
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.command is None:
        argument_parser.print_help()
        return 0
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    # HTTP debug traces can expose signed temporary download URLs.
    for name in ("urllib3", "requests", "httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)
    recorder = Recorder()
    report: Path | None = args.report
    config: Config | None = None
    success = False
    result: object = None
    try:
        config = load_config(args.config)
        if args.device is not None:
            from dataclasses import replace

            config = replace(config, device=args.device)
        if getattr(args, "hours", None) is not None:
            from dataclasses import replace

            config = replace(config, hours=args.hours)
        report = (
            report
            or config.root / "outputs/timings" / f"{args.command}-{uuid4().hex}.json"
        )
        if report.exists():
            raise FileExistsError(report)
        with recorder.measure(f"command.{args.command}"):
            result = dispatch(args, config, recorder)
        success = True
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, PermissionError) as error:
        logging.error("%s", error)
        return 2
    except requests.RequestException as error:
        status = error.response.status_code if error.response is not None else None
        logging.error(
            "Provider request failed (%s, HTTP %s). Check connectivity, credentials, "
            "dataset terms and the requested dates; see the failed timing stage.",
            type(error).__name__,
            status,
        )
        return 3
    finally:
        if report is not None and not report.exists():
            values = asdict(config) if config is not None else {}
            if "root" in values:
                values["root"] = str(values["root"])
            write_json(
                report,
                {
                    "command": args.command,
                    "arguments": {k: str(v) for k, v in vars(args).items()},
                    "success": success,
                    "created_utc": datetime.now(UTC).isoformat(),
                    "config": values,
                    "result": result,
                    "timings": recorder.report(),
                    "input_sha256": recorder.inputs,
                },
            )
            logging.info("Timing report: %s", report)
