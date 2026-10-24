"""Compare local predictions with a frozen, manually captured reference forecast."""

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from aurora_finetuning.acquisition import sha256
from aurora_finetuning.telemetry import write_json


def hourly_rows(result: dict[str, object]) -> list[dict[str, object]]:
    """Validate complete, consecutive hourly forecasts with aware timestamps."""
    rows = result.get("hourly")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Missing hourly reference/prediction rows")
    previous = None
    checked = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid hourly row")
        time = datetime.fromisoformat(str(row["time"]))
        if time.tzinfo is None:
            raise ValueError("Reference/prediction times must include a timezone")
        if time.minute or time.second or time.microsecond:
            raise ValueError("Forecast valid times must be on exact hours")
        time = time.astimezone(UTC)
        if previous is not None and (time - previous).total_seconds() != 3600:
            raise ValueError("Rows must be unique, consecutive, and hourly")
        previous = time
        for name in ("temperature_c", "precipitation_mm"):
            value = float(str(row[name]))
            if not math.isfinite(value) or (name == "precipitation_mm" and value < 0):
                raise ValueError(f"Invalid {name}")
        start = datetime.fromisoformat(str(row["interval_start"]))
        if start.tzinfo is None or (time - start).total_seconds() != 3600:
            raise ValueError("Rainfall must cover the preceding hour")
        checked.append(row)
    return checked


def compare_prediction(
    reference: dict[str, object], prediction: dict[str, object]
) -> dict[str, object]:
    """Score exactly matched times and coordinates; never fill missing leads."""
    target, rows = hourly_rows(reference), hourly_rows(prediction)
    if len(rows) != len(target):
        raise ValueError("Prediction must cover the complete reference window")
    for coord in ("latitude", "longitude"):
        a, b = float(str(reference[coord])), float(str(prediction[coord]))
        if not math.isfinite(a + b) or abs(a - b) > 1e-5:
            raise ValueError(f"Reference/prediction {coord} mismatch")
    for expected, row in zip(target, rows, strict=True):
        if datetime.fromisoformat(str(expected["time"])) != datetime.fromisoformat(
            str(row["time"])
        ):
            raise ValueError("Reference/prediction valid times differ")
    metrics = {}
    for name in ("temperature_c", "precipitation_mm"):
        actual = np.array([float(str(row[name])) for row in rows])
        expected_values = np.array([float(str(row[name])) for row in target])
        delta = actual - expected_values
        metrics[name] = {
            "mae": float(np.abs(delta).mean()),
            "bias": float(delta.mean()),
            "rmse": float(np.sqrt(np.mean(delta**2))),
        }
        if name == "precipitation_mm":
            metrics[name].update(
                prediction_total=float(actual.sum()),
                reference_total=float(expected_values.sum()),
            )
    return {
        "metrics": metrics,
        "hourly": [
            {
                "time": row["time"],
                "temperature_c": row["temperature_c"],
                "reference_temperature_c": expected["temperature_c"],
                "precipitation_mm": row["precipitation_mm"],
                "reference_precipitation_mm": expected["precipitation_mm"],
                "cumulative_rainfall_difference_mm": float(
                    sum(
                        float(str(rows[j]["precipitation_mm"]))
                        - float(str(target[j]["precipitation_mm"]))
                        for j in range(i + 1)
                    )
                ),
            }
            for i, (row, expected) in enumerate(zip(rows, target, strict=True))
        ],
        "prediction": prediction,
    }


def compare_files(
    reference: Path, predictions: list[Path], output: Path
) -> dict[str, object]:
    """Save an exclusive comparison, retaining original forecast provenance."""
    target = json.loads(reference.read_text())
    candidates = []
    for path in predictions:
        document = json.loads(path.read_text())
        prediction = document.get("result", document)
        if not isinstance(prediction, dict):
            raise ValueError(f"Missing prediction in {path}")
        candidates.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "comparison": compare_prediction(target, prediction),
                "run_metadata": {k: v for k, v in document.items() if k != "result"}
                if "result" in document
                else {},
            }
        )
    report = {
        "reference": target,
        "reference_sha256": sha256(reference),
        "scope": "single-case development agreement; not observed accuracy",
        "candidates": candidates,
    }
    write_json(output, report)
    return report
