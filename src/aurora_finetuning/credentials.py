"""Read local credentials without placing them in configuration or diagnostics."""

import os
import stat
from pathlib import Path


def knmi_api_key(root: Path) -> str:
    """Prefer the environment, then an owner-only local secret file."""
    value = os.environ.get("KNMI_API_KEY")
    if value is None:
        path = root / ".secrets/knmi-api-key"
        if not path.is_file():
            raise ValueError(
                "Set KNMI_API_KEY or create .secrets/knmi-api-key with mode 600"
            )
        if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise ValueError("KNMI secret file must be a regular owner-only file (600)")
        value = path.read_text(encoding="utf-8").strip()
    if not value or any(c.isspace() for c in value):
        raise ValueError("KNMI credential is empty or contains whitespace")
    return value
