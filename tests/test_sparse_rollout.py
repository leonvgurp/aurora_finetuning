"""Sparse scheduling preserves pinned Aurora's actual autoregressive behavior."""

import torch
from aurora import AuroraV1p5, rollout

from aurora_finetuning.leads import scheduled_rollout
from aurora_finetuning.regional import regional_context
from aurora_finetuning.smoke import synthetic_batch


def test_sparse_matches_dense_across_six_hour_boundaries() -> None:
    torch.manual_seed(42)
    model = AuroraV1p5(
        embed_dim=32,
        num_heads=2,
        encoder_depths=(1, 1, 1),
        decoder_depths=(1, 1, 1),
        encoder_num_heads=(2, 4, 8),
        decoder_num_heads=(8, 4, 2),
        window_size=(2, 2, 2),
        autocast=False,
        use_fp16_safe_attention=False,
    ).eval()
    model.requires_grad_(False)
    with regional_context(), torch.inference_mode():
        batch = synthetic_batch()
        dense = list(rollout(model, batch, steps=3, fine_lead_times=[1, 2, 3, 4, 5, 6]))
        for requested in (list(range(8, 16)), [5, 6, 7, 12, 13, 18]):
            for lead, prediction in scheduled_rollout(
                model, batch, max(requested), requested
            ):
                reference = dense[lead - 1]
                assert prediction.metadata.time == reference.metadata.time
                for name, value in prediction.surf_vars.items():
                    torch.testing.assert_close(
                        value, reference.surf_vars[name], rtol=0, atol=0
                    )
                for name, value in prediction.atmos_vars.items():
                    torch.testing.assert_close(
                        value, reference.atmos_vars[name], rtol=0, atol=0
                    )
