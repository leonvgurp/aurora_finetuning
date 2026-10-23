"""Regression tests for global assumptions at the regional model boundary."""

from datetime import datetime

import pytest
import torch
from aurora import Metadata
from aurora.model import posencoding, swin3d

from aurora_finetuning.regional import regional_context


def test_scoped_negative_longitudes_and_area() -> None:
    original = Metadata.__post_init__
    with regional_context():
        metadata = Metadata(
            lat=torch.tensor([53.0, 52.0]),
            lon=torch.tensor([-1.0, 0.0]),
            time=(datetime(2024, 1, 1),),
            atmos_levels=(850,),
        )
        assert metadata.lon[0] == -1
        area = posencoding.patch_root_area(
            torch.tensor([52.0]),
            torch.tensor([-1.0]),
            torch.tensor([53.0]),
            torch.tensor([0.0]),
        )
        assert torch.isfinite(area).all() and (area > 0).all()
    assert Metadata.__post_init__ is original
    with pytest.raises(ValueError, match="Longitudes"):
        Metadata(
            lat=torch.tensor([53.0, 52.0]),
            lon=torch.tensor([-1.0, 0.0]),
            time=(datetime(2024, 1, 1),),
            atmos_levels=(850,),
        )


def test_regional_mask_differs_from_cyclic_global_mask() -> None:
    args = (4, 8, 8, (2, 4, 4), (1, 2, 2), torch.device("cpu"), torch.float32)
    global_mask, _ = swin3d.compute_3d_shifted_window_mask(*args, warped=True)
    expected, _ = swin3d.compute_3d_shifted_window_mask(*args, warped=False)
    with regional_context():
        actual, _ = swin3d.compute_3d_shifted_window_mask(*args)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, global_mask)


def test_western_longitudes_keep_canonical_position_embeddings() -> None:
    from aurora.model import fourier

    latitude = torch.tensor([53.0, 52.0])
    longitude = torch.tensor([-2.0, -1.0])
    arguments = {
        "patch_dims": 2,
        "pos_expansion": fourier.pos_expansion,
        "scale_expansion": fourier.scale_expansion,
    }
    expected, expected_scale = posencoding.pos_scale_enc(
        16, latitude, longitude + 360, **arguments
    )
    with regional_context():
        actual, actual_scale = posencoding.pos_scale_enc(
            16, latitude, longitude, **arguments
        )
    assert torch.allclose(actual, expected)
    assert torch.allclose(actual_scale, expected_scale)
