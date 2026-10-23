"""Validated, explicit runtime settings for regional experiments."""

import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Regional settings; paths are resolved relative to the TOML file."""

    root: Path = Path(".")
    north: float = 56.5
    south: float = 47.5
    west: float = -2.5
    east: float = 12.5
    resolution: float = 0.25
    hours: int = 8
    device: str = "mps"
    seed: int = 42
    max_steps: int = 20
    learning_rate: float = 0.0001
    temperature_scale: float = 5.0
    precipitation_scale: float = 1.0
    max_input_age_hours: int = 12
    max_memory_gb: float = 12.0
    checkpoint_revision: str = "a96afd7ee6d65e3bd2d476f3be798a25a56f2296"
    training_head: str = "surface"

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(v)
            for v in (
                self.north,
                self.south,
                self.west,
                self.east,
                self.resolution,
                self.learning_rate,
                self.temperature_scale,
                self.precipitation_scale,
                self.max_memory_gb,
            )
        ):
            raise ValueError("Configuration numbers must be finite")
        if not (-90 < self.south < self.north < 90):
            raise ValueError("Expected -90 < south < north < 90")
        if not (-180 <= self.west < self.east <= 180):
            raise ValueError("Expected -180 <= west < east <= 180")
        if self.resolution not in (0.25, 0.5, 1.0):
            raise ValueError("resolution must be 0.25, 0.5 or 1.0 degrees")
        if not 1 <= self.hours <= 24:
            raise ValueError("hours must be between 1 and 24")
        if self.device not in ("mps", "cpu", "cuda"):
            raise ValueError("device must be mps, cpu or cuda")
        if self.seed < 0 or self.max_steps < 1 or self.max_input_age_hours < 1:
            raise ValueError("seed must be nonnegative; steps and input age positive")
        if (
            min(
                self.learning_rate,
                self.temperature_scale,
                self.precipitation_scale,
                self.max_memory_gb,
            )
            <= 0
        ):
            raise ValueError("Learning rate, scales and memory limit must be positive")
        if self.training_head != "surface":
            raise ValueError("Only surface-head training is supported initially")
        if self.checkpoint_revision and (
            len(self.checkpoint_revision) != 40
            or any(c not in "0123456789abcdef" for c in self.checkpoint_revision)
        ):
            raise ValueError(
                "checkpoint_revision must be a full Hugging Face commit SHA"
            )


def load_config(path: Path | None = None) -> Config:
    """Load strict flat TOML, rejecting unknown keys and incorrect value types."""
    if path is None:
        return Config(root=Path.cwd().resolve())
    with path.open("rb") as handle:
        values = tomllib.load(handle)
    defaults = Config()
    allowed = {f.name for f in fields(Config)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
    for name, value in values.items():
        if name == "root":
            if not isinstance(value, str):
                raise ValueError("root must be a string path")
            continue
        expected = type(getattr(defaults, name))
        if expected is float and type(value) in (float, int):
            values[name] = float(value)
        elif type(value) is not expected:
            raise ValueError(f"{name} must be {expected.__name__}")
    values["root"] = (path.resolve().parent / values.get("root", ".")).resolve()
    return Config(**values)
