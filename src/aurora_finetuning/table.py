"""Plain Markdown presentation of validated hourly location forecasts."""

import math
from datetime import datetime


def format_table(result: dict[str, object]) -> str:
    """Render a complete hourly result, rejecting missing or invalid values."""
    rows = result.get("hourly")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Forecast has no hourly rows")
    atmospheric = all(
        isinstance(row, dict)
        and all(name in row for name in ("pressure_hpa", "wind_speed_ms"))
        for row in rows
    )
    lines = [
        "| Hour ending (Europe/Amsterdam) | Temperature (°C) | Rainfall (mm/hour) |",
        "|---|---:|---:|",
    ]
    if atmospheric:
        lines[0] += " Pressure (hPa) | Wind (m/s) |"
        lines[1] += "---:|---:|"
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
        extra = ""
        if atmospheric:
            pressure, wind = float(row["pressure_hpa"]), float(row["wind_speed_ms"])
            if not math.isfinite(pressure) or not math.isfinite(wind):
                raise ValueError("Invalid pressure or wind in hourly forecast")
            extra = f" {pressure:.1f} | {wind:.2f} |"
        lines.append(
            f"| {time.isoformat(sep=' ', timespec='minutes')} "
            f"| {temperature:.2f} | {rainfall:.3f} |{extra}"
        )
    return "\n".join(lines)
