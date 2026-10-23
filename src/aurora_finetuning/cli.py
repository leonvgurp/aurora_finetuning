"""Command-line entry point for the research project."""

import argparse
from collections.abc import Sequence
from importlib.metadata import version


def main(argv: Sequence[str] | None = None) -> int:
    """Display project help or the installed package version."""
    parser = argparse.ArgumentParser(
        description="Aurora forecasting with KNMI data (initial environment setup)."
    )
    parser.add_argument(
        "--version", action="version", version=version("aurora-finetuning")
    )
    parser.parse_args(argv)
    parser.print_help()
    return 0
