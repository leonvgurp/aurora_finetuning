"""Credential lookup never needs to expose or execute secret contents."""

from pathlib import Path

import pytest

from aurora_finetuning.credentials import knmi_api_key


def test_local_secret_and_environment_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KNMI_API_KEY", raising=False)
    directory = tmp_path / ".secrets"
    directory.mkdir(mode=0o700)
    path = directory / "knmi-api-key"
    path.write_text("file-test-key\n")
    path.chmod(0o600)
    assert knmi_api_key(tmp_path) == "file-test-key"
    monkeypatch.setenv("KNMI_API_KEY", "environment-test-key")
    assert knmi_api_key(tmp_path) == "environment-test-key"


def test_missing_or_insecure_secret_fails_without_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KNMI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="KNMI_API_KEY"):
        knmi_api_key(tmp_path)
    directory = tmp_path / ".secrets"
    directory.mkdir()
    path = directory / "knmi-api-key"
    path.write_text("file-test-key")
    path.chmod(0o644)
    with pytest.raises(ValueError, match="owner-only") as error:
        knmi_api_key(tmp_path)
    assert "file-test-key" not in str(error.value)
