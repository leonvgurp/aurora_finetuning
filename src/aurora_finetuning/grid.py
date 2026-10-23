"""Regular regional grids with continuous physical longitudes."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aurora_finetuning.config import Config

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Grid:
    """Descending latitudes and increasing signed longitudes."""

    latitude: FloatArray
    longitude: FloatArray

    @property
    def shape(self) -> tuple[int, int]:
        """Return spatial dimensions."""
        return len(self.latitude), len(self.longitude)

    def describe(self) -> dict[str, object]:
        """Describe the actual patch-aligned domain."""
        return {
            "north": float(self.latitude[0]),
            "south": float(self.latitude[-1]),
            "west": float(self.longitude[0]),
            "east": float(self.longitude[-1]),
            "shape": list(self.shape),
        }


def make_grid(config: Config) -> Grid:
    """Expand requested bounds to multiples of 16 cells for the model hierarchy."""
    res = config.resolution
    north = math.ceil(config.north / res) * res
    west = math.floor(config.west / res) * res
    height = math.ceil(((north - config.south) / res + 1) / 16) * 16
    width = math.ceil(((config.east - west) / res + 1) / 16) * 16
    centre_lat = (config.north + config.south) / 2
    centre_lon = (config.west + config.east) / 2
    north = math.ceil((centre_lat + (height - 1) * res / 2) / res) * res
    west = math.floor((centre_lon - (width - 1) * res / 2) / res) * res
    lat = north - np.arange(height, dtype=np.float64) * res
    lon = west + np.arange(width, dtype=np.float64) * res
    if lat[-1] <= -90 or lon[-1] > 180:
        raise ValueError("Patch-aligned grid exceeds supported geographic bounds")
    return Grid(lat, lon)


def signed_longitude(values: FloatArray) -> FloatArray:
    """Map geographic longitudes to [-180, 180)."""
    return (values + 180) % 360 - 180
