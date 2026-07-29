from __future__ import annotations

from pathlib import Path

import pytest

from market_risk_forecasting.config import load_config
from market_risk_forecasting.errors import ConfigInvalidError


def test_frozen_configuration_loads(project_root: Path) -> None:
    config = load_config(project_root / "configs" / "frozen_research.toml")

    assert config.experiment.experiment_id == "risk-v01-frozen"
    assert config.upstream.instruments == ("SPY", "IEF", "GLD")
    assert config.portfolio_proxy.weights == {
        "SPY": 0.60,
        "IEF": 0.30,
        "GLD": 0.10,
    }
    assert config.ewma.lambda_ == 0.94
    assert config.garch.estimation_window == 1250


def test_effective_configuration_uses_public_toml_names(project_root: Path) -> None:
    config = load_config(project_root / "config.example.toml")

    effective = config.to_dict()

    assert effective["ewma"]["lambda"] == 0.94
    assert "lambda_" not in effective["ewma"]
    assert effective["periods"]["test_end"] == "2025-12-31"


def test_unknown_key_fails(project_root: Path, tmp_path: Path) -> None:
    source = (project_root / "config.example.toml").read_text(encoding="utf-8")
    path = tmp_path / "unknown.toml"
    path.write_text(source + "\nunknown = true\n", encoding="utf-8")

    with pytest.raises(ConfigInvalidError, match="unknown key"):
        load_config(path)


def test_invalid_period_order_fails(project_root: Path, tmp_path: Path) -> None:
    source = (project_root / "config.example.toml").read_text(encoding="utf-8")
    path = tmp_path / "periods.toml"
    path.write_text(
        source.replace(
            'validation_start = "2015-01-01"',
            'validation_start = "2014-12-31"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigInvalidError, match="strictly increasing"):
        load_config(path)


def test_invalid_proxy_weight_sum_fails(project_root: Path, tmp_path: Path) -> None:
    source = (project_root / "config.example.toml").read_text(encoding="utf-8")
    path = tmp_path / "weights.toml"
    path.write_text(source.replace("GLD = 0.10", "GLD = 0.11"), encoding="utf-8")

    with pytest.raises(ConfigInvalidError, match="weights must sum to one"):
        load_config(path)


def test_incompatible_upstream_identity_fails(
    project_root: Path, tmp_path: Path
) -> None:
    source = (project_root / "config.example.toml").read_text(encoding="utf-8")
    path = tmp_path / "upstream.toml"
    path.write_text(
        source.replace('package_version = "0.1.0"', 'package_version = "0.2.0"'),
        encoding="utf-8",
    )

    with pytest.raises(ConfigInvalidError, match="v0.1 public contract"):
        load_config(path)


def test_nonfrozen_historical_windows_fail(project_root: Path, tmp_path: Path) -> None:
    source = (project_root / "config.example.toml").read_text(encoding="utf-8")
    path = tmp_path / "windows.toml"
    path.write_text(
        source.replace("variance_window = 252", "variance_window = 251"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigInvalidError, match="variance_window=252"):
        load_config(path)


def test_nonfrozen_ewma_parameters_fail(project_root: Path, tmp_path: Path) -> None:
    source = (project_root / "config.example.toml").read_text(encoding="utf-8")
    path = tmp_path / "ewma.toml"
    path.write_text(source.replace("lambda = 0.94", "lambda = 0.95"), encoding="utf-8")

    with pytest.raises(ConfigInvalidError, match=r"ewma\.lambda=0\.94"):
        load_config(path)
