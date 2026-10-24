"""Small deterministic synthetic campaign for checking the pipeline, never skill."""

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from aurora_finetuning.artifacts import atomic_output


def create_demo(root: Path) -> Path:
    """Write six independent synthetic events and an immediately runnable TOML."""
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    lines = [
        'id = "synthetic-smoke"',
        'track = "synthetic"',
        'output = "runs/synthetic"',
        "hours = 8",
        "timeout_seconds = 120",
        "max_download_bytes = 1",
        "seed = 42",
    ]
    rng = np.random.default_rng(42)
    for i in range(6):
        issue = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i * 60)
        times = [
            np.datetime64((issue + timedelta(hours=j)).replace(tzinfo=None), "ns")
            for j in range(1, 9)
        ]
        rain = np.maximum(0, rng.normal(0.8 if i % 3 else -1, 0.7, (8, 2)))
        temp = 10 + np.sin(np.arange(8)[:, None] / 3) + rng.normal(0, 0.1, (8, 2))
        ds = xr.Dataset(
            {
                "temperature_c": (("time", "station"), temp),
                "precipitation_mm": (("time", "station"), rain),
            },
            coords={
                "time": times,
                "station": ["interior", "coastal"],
                "latitude": ("station", [52.1, 52.5]),
                "longitude": ("station", [5.1, 4.6]),
            },
        )
        ds.temperature_c.attrs["units"] = "degC"
        ds.precipitation_mm.attrs.update(units="mm", interval_hours=1)
        history = ds.isel(time=[0]).assign_coords(
            time=[np.datetime64(issue.replace(tzinfo=None), "ns")]
        )
        forecast = ds.copy(deep=True)
        forecast.temperature_c.values[:] += 0.5
        forecast.precipitation_mm.values[:] *= 0.6
        forecast.attrs.update(
            initialization_utc=(issue - timedelta(hours=6)).isoformat(),
            source="synthetic fixture, not Aurora",
        )
        for name, dataset in (
            ("observations", ds),
            ("history", history),
            ("forecast", forecast),
        ):
            with atomic_output(root / f"case-{i}-{name}.nc") as temporary:
                dataset.to_netcdf(temporary)
        lines += [
            "",
            "[[cases]]",
            f'id = "case-{i}"',
            f'issue = "{issue.isoformat()}"',
            f'initialization = "{(issue - timedelta(hours=6)).isoformat()}"',
            'split = "development"',
            f'regime = "{["dry", "widespread", "showery"][i % 3]}"',
            f'observations = "case-{i}-observations.nc"',
            f'history = "case-{i}-history.nc"',
        ]
    lines += [
        "",
        "[[methods]]",
        'id = "synthetic-forecast"',
        'kind = "cached"',
        "[methods.paths]",
    ]
    lines += [f'case-{i} = "case-{i}-forecast.nc"' for i in range(6)]
    lines += [
        "",
        "[[methods]]",
        'id = "persistence"',
        'kind = "persistence"',
        "",
        "[[methods]]",
        'id = "dry"',
        'kind = "dry"',
    ]
    config = root / "campaign.toml"
    config.write_text("\n".join(lines) + "\n")
    return config


def main() -> None:
    """Generate explicit test data without downloading anything."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(create_demo(args.output.resolve()))


if __name__ == "__main__":
    main()
