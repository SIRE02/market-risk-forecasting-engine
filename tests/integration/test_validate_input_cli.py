from __future__ import annotations

from pathlib import Path

import pytest

from market_risk_forecasting.cli import main
from market_risk_forecasting.upstream import sha256_file


def test_validate_input_cli_constructs_series_and_writes_nothing(
    project_root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = project_root / "data" / "fixtures" / "upstream_run"
    output_dir = tmp_path / "must-not-be-created"
    source = (project_root / "config.example.toml").read_text(encoding="utf-8")
    config_path = tmp_path / "validate.toml"
    config_path.write_text(
        source.replace(
            'input_run_dir = "data/fixtures/upstream_run"',
            f'input_run_dir = "{fixture.as_posix()}"',
        ).replace(
            'output_dir = "outputs/risk-v02-example"',
            f'output_dir = "{output_dir.as_posix()}"',
        ),
        encoding="utf-8",
    )
    before = {
        path.name: sha256_file(path) for path in fixture.iterdir() if path.is_file()
    }

    exit_code = main(["validate-input", "--config", str(config_path)])

    captured = capsys.readouterr()
    after = {
        path.name: sha256_file(path) for path in fixture.iterdir() if path.is_file()
    }
    assert exit_code == 0
    assert "Input validation: ok" in captured.out
    assert "SPY, IEF, GLD, FIXTURE_PROXY" in captured.out
    assert "Return observations: 4957" in captured.out
    assert captured.err == ""
    assert before == after
    assert not output_dir.exists()
