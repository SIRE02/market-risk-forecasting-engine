from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from historical_asset_risk.config import (
    load_configuration as load_upstream_configuration,
)

from market_risk_forecasting.config import load_config
from market_risk_forecasting.errors import ConfigInvalidError


def test_real_data_configuration_pair_matches(project_root: Path) -> None:
    name = "four_assets"
    acquisition = load_upstream_configuration(
        project_root / "configs" / "upstream" / f"{name}.toml"
    )
    forecasting = load_config(project_root / "configs" / f"{name}.toml")

    assert acquisition.tickers == forecasting.upstream.instruments
    assert acquisition.output_dir == forecasting.experiment.input_run_dir
    assert date.fromisoformat(acquisition.start_date) <= (
        forecasting.periods.development_start
    )
    assert date.fromisoformat(acquisition.end_date) > forecasting.periods.test_end


def test_configuration_accepts_dynamic_universe(project_root: Path) -> None:
    config = load_config(project_root / "configs" / "four_assets.toml")

    assert config.experiment.protocol_version == "2.0"
    assert config.upstream.instruments == ("AAPL", "MSFT", "GLD", "TLT")
    assert config.portfolio_proxy.enabled is False
    assert config.portfolio_proxy.series_id is None
    assert config.portfolio_proxy.weights == {}
    assert config.to_dict()["portfolio_proxy"] == {"enabled": False}


def test_proxy_weights_must_match_dynamic_universe(
    project_root: Path,
    tmp_path: Path,
) -> None:
    source = (project_root / "configs" / "four_assets.toml").read_text(encoding="utf-8")
    path = tmp_path / "custom-proxy.toml"
    path.write_text(
        source.replace(
            "[portfolio_proxy]\nenabled = false",
            '[portfolio_proxy]\nenabled = true\nseries_id = "TECH_GOLD"\n'
            "weights = { AAPL = 0.40, MSFT = 0.30, GLD = 0.20, TLT = 0.10 }",
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.portfolio_proxy.enabled is True
    assert config.portfolio_proxy.series_id == "TECH_GOLD"
    assert config.portfolio_proxy.weights == {
        "AAPL": 0.40,
        "MSFT": 0.30,
        "GLD": 0.20,
        "TLT": 0.10,
    }


def test_effective_configuration_uses_public_toml_names(project_root: Path) -> None:
    config = load_config(project_root / "config.example.toml")

    effective = config.to_dict()

    assert effective["ewma"]["lambda"] == 0.94
    assert "lambda_" not in effective["ewma"]
    assert effective["periods"]["test_end"] == "2025-12-31"
    assert effective["experiment"]["protocol_version"] == "2.0"
    assert effective["portfolio_proxy"] == {
        "enabled": True,
        "series_id": "FIXTURE_PROXY",
        "weights": {"SPY": 0.60, "IEF": 0.30, "GLD": 0.10},
    }


def test_legacy_protocol_version_fails(project_root: Path, tmp_path: Path) -> None:
    source = (project_root / "config.example.toml").read_text(encoding="utf-8")
    path = tmp_path / "legacy.toml"
    path.write_text(
        source.replace('protocol_version = "2.0"', 'protocol_version = "1.0"'),
        encoding="utf-8",
    )

    with pytest.raises(ConfigInvalidError, match="must be '2.0'"):
        load_config(path)


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
        source.replace('package_version = "0.1.0"', 'package_version = "9.9.9"'),
        encoding="utf-8",
    )

    with pytest.raises(ConfigInvalidError, match="v0.1 public contract"):
        load_config(path)


def test_protocol_historical_windows_fail(project_root: Path, tmp_path: Path) -> None:
    source = (project_root / "config.example.toml").read_text(encoding="utf-8")
    path = tmp_path / "windows.toml"
    path.write_text(
        source.replace("variance_window = 252", "variance_window = 251"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigInvalidError, match="variance_window=252"):
        load_config(path)


def test_protocol_ewma_parameters_fail(project_root: Path, tmp_path: Path) -> None:
    source = (project_root / "config.example.toml").read_text(encoding="utf-8")
    path = tmp_path / "ewma.toml"
    path.write_text(source.replace("lambda = 0.94", "lambda = 0.95"), encoding="utf-8")

    with pytest.raises(ConfigInvalidError, match=r"ewma\.lambda=0\.94"):
        load_config(path)


def test_protocol_garch_parameters_fail(project_root: Path, tmp_path: Path) -> None:
    source = (project_root / "config.example.toml").read_text(encoding="utf-8")
    path = tmp_path / "garch.toml"
    path.write_text(
        source.replace("refit_every_origins = 20", "refit_every_origins = 10"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigInvalidError, match="refit_every_origins=20"):
        load_config(path)


def test_protocol_evaluation_parameters_fail(
    project_root: Path,
    tmp_path: Path,
) -> None:
    source = (project_root / "config.example.toml").read_text(encoding="utf-8")
    path = tmp_path / "evaluation.toml"
    path.write_text(
        source.replace("bootstrap_resamples = 2000", "bootstrap_resamples = 1000"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigInvalidError, match="2000 resamples"):
        load_config(path)
