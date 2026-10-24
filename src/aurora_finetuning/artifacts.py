"""Atomic, exclusive publication of locally generated artifacts."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile


@contextmanager
def atomic_output(path: Path) -> Iterator[Path]:
    """Publish a complete same-filesystem file without replacing another writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as f:
        temporary = Path(f.name)
    try:
        yield temporary
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
