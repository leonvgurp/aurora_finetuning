"""Continuous UTC calendar features shared by observation-conditioned models."""

import calendar
import math
from datetime import UTC, datetime


def calendar_features(time: datetime) -> tuple[float, float, float, float]:
    """Encode annual and daily position without month or midnight discontinuities."""
    utc = time.astimezone(UTC).replace(tzinfo=None) if time.tzinfo else time
    seconds = utc.hour * 3600 + utc.minute * 60 + utc.second
    day = 2 * math.pi * seconds / 86400
    year = (
        2
        * math.pi
        * ((utc.timetuple().tm_yday - 1) + seconds / 86400)
        / (366 if calendar.isleap(utc.year) else 365)
    )
    return math.sin(year), math.cos(year), math.sin(day), math.cos(day)
