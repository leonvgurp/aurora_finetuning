"""Plain Markdown presentation of validated hourly location forecasts."""

import math
from datetime import datetime


def format_table(result: dict[str, object]) -> str:
    """Render a complete hourly result, rejecting missing or invalid values."""
    rows = result.get("hourly")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Forecast has no hourly rows")
    lines = [
        "| Hour ending (Europe/Amsterdam) | Temperature (°C) | Rainfall (mm/hour) |",
        "|---|---:|---:|",
    ]
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid hourly forecast row")
        time = datetime.fromisoformat(str(row["time"]))
        if time.tzinfo is None:
            raise ValueError("Hourly timestamps must include a UTC offset")
        temperature = float(row["temperature_c"])
        rainfall = float(row["precipitation_mm"])
        if (
            not math.isfinite(temperature)
            or not math.isfinite(rainfall)
            or rainfall < 0
        ):
            raise ValueError("Invalid temperature or rainfall in hourly forecast")
        lines.append(
            f"| {time.isoformat(sep=' ', timespec='minutes')} "
            f"| {temperature:.2f} | {rainfall:.3f} |"
        )
    return "\n".join(lines)
