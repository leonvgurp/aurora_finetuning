"""Strict time/interval matching for manually captured forecast references."""

import copy
import json
from pathlib import Path

import pytest

from aurora_finetuning.cli import main
from aurora_finetuning.reference import compare_prediction, hourly_rows


def target() -> dict[str, object]:
    result = json.loads(
        (
            Path(__file__).parents[1] / "experiments/leusden-20261024/buienradar.json"
        ).read_text()
    )
    assert isinstance(result, dict)
    return result


def test_reference_metrics_and_cumulative_rain() -> None:
    reference = target()
    prediction = copy.deepcopy(reference)
    for row in hourly_rows(prediction):
        row["temperature_c"] = float(str(row["temperature_c"])) + 2
        row["precipitation_mm"] = 0
    result = json.loads(json.dumps(compare_prediction(reference, prediction)))
    assert result["metrics"]["temperature_c"]["mae"] == 2
    assert result["metrics"]["temperature_c"]["bias"] == 2
    assert result["metrics"]["precipitation_mm"]["reference_total"] == pytest.approx(
        7.6
    )
    assert result["hourly"][-1]["cumulative_rainfall_difference_mm"] == pytest.approx(
        -7.6
    )


@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "interval", "location", "nan"]
)
def test_reference_rejects_unmatched_cases(defect: str) -> None:
    reference = target()
    prediction = copy.deepcopy(reference)
    rows = hourly_rows(prediction)
    prediction["hourly"] = rows
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows[1]["time"] = rows[0]["time"]
    elif defect == "interval":
        rows[0]["interval_start"] = rows[0]["time"]
    elif defect == "location":
        prediction["longitude"] = 6
    else:
        rows[0]["temperature_c"] = float("nan")
    with pytest.raises(ValueError):
        compare_prediction(reference, prediction)


def test_reference_cli_preserves_original_and_rejects_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reference, prediction, output = (
        tmp_path / name for name in ("r.json", "p.json", "o.json")
    )
    reference.write_text(json.dumps(target()))
    prediction.write_text(json.dumps({"result": target(), "network_bytes": 0}))
    argv = [
        "compare-reference",
        "--reference",
        str(reference),
        "--predictions",
        str(prediction),
        "--output",
        str(output),
        "--report",
        str(tmp_path / "timing.json"),
    ]
    assert main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["reference"]["issued_utc"] is None
    assert result["candidates"][0]["comparison"]["metrics"]["temperature_c"]["mae"] == 0
    argv[-1] = str(tmp_path / "second-timing.json")
    assert main(argv) == 2


def test_case_failure_preserves_diagnostics(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from aurora_finetuning.case_study import run_candidate
    from aurora_finetuning.config import Config

    output = tmp_path / "failed"
    with pytest.raises(FileNotFoundError):
        run_candidate(
            Config(root=tmp_path, device="cpu"),
            datetime(2024, 1, 1),
            datetime(2024, 1, 1, tzinfo=UTC),
            52,
            5,
            output,
        )
    failure = json.loads((output / "failure.json").read_text())
    assert failure["error_type"] == "FileNotFoundError"
    assert not failure["timings"][-1]["success"]
    assert not (output / "prediction.json").exists()
