"""Scoped compatibility adapters for the pinned upstream Aurora implementation.

Aurora assumes global positive longitudes and cyclic east/west attention. These
patches are process-local, serialised and restored even after failure. No upstream
files are edited. Fourier work runs on CPU for MPS float64 compatibility.
"""

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch.nn.functional as functional
from aurora import Metadata
from aurora.model import fourier, posencoding, swin3d

_LOCK = threading.RLock()
# Explicit signatures isolate the untyped, pinned upstream API boundary.
_ORIGINAL_METADATA: Callable[[Metadata], None] = Metadata.__post_init__
_ORIGINAL_AREA: Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
] = posencoding.patch_root_area
_ORIGINAL_FOURIER: Callable[
    [fourier.FourierExpansion, torch.Tensor, int], torch.Tensor
] = fourier.FourierExpansion.forward
_ORIGINAL_MASK = swin3d.compute_3d_shifted_window_mask
_ORIGINAL_POSITION = posencoding.pos_scale_enc_grid


def _metadata(self: Metadata) -> None:
    original = self.lon
    if torch.any(original < 0):
        # Validate the continuous regional axis without moving the actual coordinates.
        if torch.any(original < -180) or torch.any(original > 180):
            raise ValueError("Regional longitudes must lie within [-180, 180]")
        self.lon = original + 180
    try:
        _ORIGINAL_METADATA(self)
    finally:
        self.lon = original


def _area(
    lat_min: torch.Tensor,
    lon_min: torch.Tensor,
    lat_max: torch.Tensor,
    lon_max: torch.Tensor,
) -> torch.Tensor:
    # Longitude differences determine area; a common translation is exact.
    return _ORIGINAL_AREA(lat_min, lon_min + 180, lat_max, lon_max + 180)


def _fourier(self: fourier.FourierExpansion, x: torch.Tensor, d: int) -> torch.Tensor:
    device = x.device
    if device.type == "mps":
        x = x.cpu()
    return _ORIGINAL_FOURIER(self, x, d).to(device)


def _mask(
    C: int,
    H: int,
    W: int,
    ws: tuple[int, int, int],
    ss: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    warped: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    result = _ORIGINAL_MASK(C, H, W, ws, ss, device, dtype, warped=False)
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("Unexpected upstream attention mask result")
    mask, groups = result
    if not isinstance(mask, torch.Tensor) or not isinstance(groups, torch.Tensor):
        raise TypeError("Upstream attention mask must contain tensors")
    return mask, groups


def _position(
    encode_dim: int,
    grid: torch.Tensor,
    patch_dims: tuple[int, int],
    pos_expansion: fourier.FourierExpansion,
    scale_expansion: fourier.FourierExpansion,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep continuous geometry but encode physical longitudes in [0, 360)."""
    position, scale = _ORIGINAL_POSITION(
        encode_dim, grid, patch_dims, pos_expansion, scale_expansion
    )
    if not isinstance(position, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise TypeError("Unexpected upstream positional encoding result")
    longitude = functional.avg_pool2d(grid[:, 1], patch_dims).reshape(grid.shape[0], -1)
    encoded_longitude = pos_expansion(longitude.remainder(360), encode_dim // 2)
    return torch.cat(
        (position[..., : encode_dim // 2], encoded_longitude), dim=-1
    ), scale


@contextmanager
def regional_context() -> Iterator[None]:
    """Apply regional geometry and device compatibility for a complete operation.

    Regional crops remain an experimental domain shift; validation and regional
    adaptation are required. Do not run unrelated Aurora
    work concurrently in the same process.
    """
    with (
        _LOCK,
        patch.object(Metadata, "__post_init__", _metadata),
        patch.object(posencoding, "patch_root_area", _area),
        patch.object(posencoding, "pos_scale_enc_grid", _position),
        patch.object(fourier.FourierExpansion, "forward", _fourier),
        patch.object(swin3d, "compute_3d_shifted_window_mask", _mask),
    ):
        yield
