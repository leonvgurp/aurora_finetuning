"""Spatial rainfall verification, simple motion baselines and case uncertainty."""

import numpy as np

from aurora_finetuning.evaluation import metrics
from aurora_finetuning.grid import FloatArray


def event_scores(
    prediction: FloatArray, observation: FloatArray, threshold: float
) -> dict[str, float | int | None]:
    """Return event scores; undefined denominators remain explicit nulls."""
    report = metrics(prediction, observation, threshold)
    hit = int(report.get("hits", 0) or 0)
    miss = int(report.get("misses", 0) or 0)
    false = int(report.get("false_alarms", 0) or 0)
    report.update(
        precision=hit / (hit + false) if hit + false else None,
        recall=hit / (hit + miss) if hit + miss else None,
        csi=hit / (hit + miss + false) if hit + miss + false else None,
    )
    return report


def neighbourhood_mean(values: FloatArray, width: int) -> FloatArray:
    """Compute centred square averages with explicit zero padding and integral sums."""
    before = width // 2
    after = width - before - 1
    padded = np.pad(values, ((before, after), (before, after)))
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    result: FloatArray = (
        integral[width:, width:]
        - integral[:-width, width:]
        - integral[width:, :-width]
        + integral[:-width, :-width]
    ) / width**2
    return result


def fractions_skill(
    prediction: FloatArray, observation: FloatArray, width: int, threshold: float
) -> float | None:
    """Score only fully observed neighbourhoods, excluding domain-edge padding."""
    valid = np.isfinite(prediction) & np.isfinite(observation)
    complete = neighbourhood_mean(valid.astype(float), width) >= 1 - 1e-10
    if not complete.any():
        return None
    p = neighbourhood_mean(((prediction >= threshold) & valid).astype(float), width)[
        complete
    ]
    o = neighbourhood_mean(((observation >= threshold) & valid).astype(float), width)[
        complete
    ]
    denominator = float(np.mean(p * p + o * o))
    return float(1 - np.mean((p - o) ** 2) / denominator) if denominator else None


def hourly_report(prediction: FloatArray, observation: FloatArray) -> dict[str, object]:
    """Score each hour and cumulative accumulation on identical finite pairs."""
    if prediction.shape != observation.shape or prediction.shape[0] != 8:
        raise ValueError("Rainfall verification requires eight matched hourly grids")
    hours = []
    for index in range(8):
        p, o = prediction[index], observation[index]
        hours.append(
            {
                "lead_hour": index + 1,
                **metrics(p, o),
                "events": {str(t): event_scores(p, o, t) for t in (0.1, 1.0)},
                "fractions_skill": {
                    str(t): {str(k): fractions_skill(p, o, k, t) for k in (1, 5, 10)}
                    for t in (0.1, 1.0)
                },
                "cumulative": metrics(
                    prediction[: index + 1].sum(0), observation[: index + 1].sum(0)
                ),
            }
        )
    return {
        "hourly": metrics(prediction, observation),
        "by_lead_hour": hours,
        "eight_hour_total": metrics(prediction.sum(0), observation.sum(0)),
        "coverage_fraction": float(np.isfinite(observation).mean()),
    }


def motion_forecast(history: FloatArray, minutes: int) -> FloatArray:
    """Extrapolate a 30-minute FFT translation; outside-domain rain is unknown."""
    previous, latest = history[-7], history[-1]
    a, b = np.nan_to_num(previous), np.nan_to_num(latest)
    displacement = [0, 0]
    if a.std() > 0 and b.std() > 0:
        correlation = np.fft.ifft2(np.fft.fft2(b) * np.conj(np.fft.fft2(a))).real
        peak = np.unravel_index(np.argmax(correlation), correlation.shape)
        displacement = [
            int(p if p <= size // 2 else p - size)
            for p, size in zip(peak, correlation.shape, strict=True)
        ]
    shift = [round(d * minutes / 30) for d in displacement]
    result = np.roll(latest, shift, axis=(0, 1)).copy()
    for axis, amount in enumerate(shift):
        if amount:
            selection = [slice(None), slice(None)]
            selection[axis] = slice(0, amount) if amount > 0 else slice(amount, None)
            result[tuple(selection)] = np.nan
    return np.asarray(result, dtype=float)


def bootstrap_cases(values: FloatArray, seed: int = 42) -> dict[str, object]:
    """Resample independent weather cases, never individual correlated pixels."""
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError("Bootstrap expects case × eight-hour scores")
    if len(values) < 2:
        return {"case_count": len(values), "ci95": None, "pilot": True}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), (1000, len(values)))
    intervals: list[list[float | None]] = [[], []]
    for hour in range(8):
        if np.isfinite(values[:, hour]).sum() < 2:
            intervals[0].append(None)
            intervals[1].append(None)
            continue
        draws = values[indices, hour]
        counts = np.isfinite(draws).sum(axis=1)
        usable = counts > 0
        means = np.nansum(draws[usable], axis=1) / counts[usable]
        bounds = np.quantile(means, [0.025, 0.975])
        intervals[0].append(float(bounds[0]))
        intervals[1].append(float(bounds[1]))
    return {"case_count": len(values), "ci95": intervals, "pilot": len(values) < 20}
