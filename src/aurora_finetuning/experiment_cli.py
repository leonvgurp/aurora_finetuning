"""Command boundary for bounded experiments; independent of live weather defaults."""

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect, execute or report a campaign through the installed package."""
    parser = argparse.ArgumentParser(
        description="Reproducible local forecast experiments"
    )
    parser.add_argument("action", choices=("plan", "fetch", "run", "collect", "report"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    try:
        if args.action in {"plan", "fetch", "run"}:
            if args.config is None:
                raise ValueError("--config is required for a campaign")
            from aurora_finetuning.experiments import plan_campaign, run_campaign

            if args.output is not None:
                raise ValueError("Campaign output belongs in its TOML configuration")
            result = (
                plan_campaign(args.config)
                if args.action == "plan"
                else run_campaign(args.config, args.action)
            )
        else:
            from aurora_finetuning.experiment_reports import (
                collect_evidence,
                report_results,
            )

            if args.output is None:
                raise ValueError("Choose a fresh --output directory")
            result = (
                collect_evidence(args.root.resolve(), args.output)
                if args.action == "collect"
                else report_results(args.input, args.output)
            )
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result.get("status") != "incomplete" else 1
    except (ValueError, FileNotFoundError, FileExistsError, PermissionError) as error:
        logging.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
