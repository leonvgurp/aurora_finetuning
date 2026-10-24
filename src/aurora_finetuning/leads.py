"""Sparse deterministic Aurora substeps with mandatory six-hour bridge states."""

from collections.abc import Iterator, Sequence

import torch
from aurora import AuroraV1p5, Batch


def lead_schedule(hours: int, requested: Sequence[int]) -> list[int]:
    """Plan only requested leads and the bridges needed to reach them."""
    if not requested or any(
        not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= hours
        for v in requested
    ):
        raise ValueError("Requested leads must be unique integers within the horizon")
    leads = sorted(set(requested))
    if len(leads) != len(requested):
        raise ValueError("Requested leads must be unique integers within the horizon")
    bridges = list(range(6, max(leads), 6))
    return sorted(set(leads + bridges))


def scheduled_rollout(
    model: AuroraV1p5, batch: Batch, hours: int, requested: Sequence[int]
) -> Iterator[tuple[int, Batch]]:
    """Keep the pinned upstream history/clipping rules; skip unused substeps."""
    # Private upstream boundary is deliberately isolated and regression-tested.
    from aurora.rollout import _advance_batch

    schedule = lead_schedule(hours, requested)
    if not model.variable_lead_time or model.backbone.stochastic:
        raise ValueError("Sparse rollout requires deterministic variable-lead Aurora")
    batch = model.batch_transform_hook(batch)
    parameter = next(model.parameters())
    batch = batch.type(parameter.dtype).crop(model.patch_size).to(parameter.device)
    initial = batch.metadata.time[0]
    for block in range((max(schedule) + 5) // 6):
        prediction = None
        for lead in schedule:
            if not block * 6 < lead <= block * 6 + 6:
                continue
            sample = next(iter(batch.surf_vars.values()))
            lead_times = torch.full(
                (sample.shape[0],),
                float(lead - block * 6),
                device=sample.device,
                dtype=sample.dtype,
            )
            prediction = model.forward(batch, lead_times=lead_times)
            elapsed = (prediction.metadata.time[0] - initial).total_seconds() / 3600
            if elapsed != lead:
                raise ValueError(
                    "Aurora substep valid time differs from requested lead"
                )
            yield lead, prediction
        if block * 6 + 6 < max(schedule):
            if prediction is None:
                raise ValueError("Missing six-hour bridge")
            prediction = model.apply_rollout_input_clipping(prediction)
            batch = _advance_batch(batch, prediction)
