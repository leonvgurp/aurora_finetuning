"""Check the installed command-line entry points."""

import subprocess
import sys


def test_module_help() -> None:
    """The installed package provides help without loading model weights."""
    result = subprocess.run(
        [sys.executable, "-m", "aurora_finetuning", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Aurora forecasting with KNMI data" in result.stdout


def test_console_version() -> None:
    """The installed console script reports its package version."""
    result = subprocess.run(
        ["aurora-finetuning", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "0.1.0"
