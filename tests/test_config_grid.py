"""Runtime settings and geographic alignment invariants."""

from pathlib import Path

import numpy as np
import pytest

from aurora_finetuning.config import Config, load_config
from aurora_finetuning.grid import make_grid


def test_config_paths_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text('root = "data"\nhours = 4\n')
    config = load_config(path)
    assert config.root == tmp_path / "data"
    assert config.hours == 4
    path.write_text("hour = 4\n")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)
    path.write_text("hours = true\n")
    with pytest.raises(ValueError, match="int"):
        load_config(path)
    with pytest.raises(ValueError, match="finite"):
        Config(learning_rate=float("nan"))


def test_grid_encloses_requested_domain() -> None:
    config = Config()
    grid = make_grid(config)
    assert all(size % 16 == 0 for size in grid.shape)
    assert grid.latitude[0] >= config.north
    assert grid.latitude[-1] <= config.south
    assert grid.longitude[0] <= config.west
    assert grid.longitude[-1] >= config.east
    assert np.allclose(np.diff(grid.longitude), config.resolution)
    assert np.allclose(np.diff(grid.latitude), -config.resolution)
