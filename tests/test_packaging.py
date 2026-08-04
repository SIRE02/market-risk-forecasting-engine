from __future__ import annotations

import subprocess
import sys

import pytest

from market_risk_forecasting import __version__
from market_risk_forecasting.cli import main


def test_version_is_available() -> None:
    assert __version__ == "0.1.0"


def test_cli_help_works() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "market_risk_forecasting", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "validate-input" in completed.stdout
    assert "reproduce" in completed.stdout


def test_invalid_config_is_concise(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["validate-input", "--config", "does-not-exist.toml"])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "CONFIG_INVALID" in captured.err
    assert "Traceback" not in captured.err
