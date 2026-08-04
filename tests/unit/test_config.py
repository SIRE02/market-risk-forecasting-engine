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
    assert effective["portfolio_proxy"] == {
        "enabled": True,
        "series_id": "FIXTURE_PROXY",
        "weights": {"SPY": 0.60, "IEF": 0.30, "GLD": 0.10},
    }


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


def test_model_and_evaluation_controls_are_configurable(
    project_root: Path,
    tmp_path: Path,
) -> None:
    source = (project_root / "config.example.toml").read_text(encoding="utf-8")
    path = tmp_path / "custom-controls.toml"
    path.write_text(
        source.replace("random_seed = 42", "random_seed = 7")
        .replace("variance_window = 252", "variance_window = 126")
        .replace("var_window = 500", "var_window = 300")
        .replace('quantile_method = "linear"', 'quantile_method = "nearest"')
        .replace("lambda = 0.94", "lambda = 0.97")
        .replace("initialization_window = 252", "initialization_window = 126")
        .replace("estimation_window = 1250", "estimation_window = 750")
        .replace("refit_every_origins = 20", "refit_every_origins = 10")
        .replace("input_scale = 100.0", "input_scale = 10.0")
        .replace("retry_count = 1", "retry_count = 0")
        .replace("stationarity_tolerance = 1e-8", "stationarity_tolerance = 1e-6")
        .replace("bootstrap_block_length = 20", "bootstrap_block_length = 10")
        .replace("bootstrap_resamples = 2000", "bootstrap_resamples = 500")
        .replace("bootstrap_confidence = 0.95", "bootstrap_confidence = 0.90"),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.experiment.random_seed == 7
    assert config.historical.variance_window == 126
    assert config.historical.var_window == 300
    assert config.historical.quantile_method == "nearest"
    assert config.ewma.lambda_ == 0.97
    assert config.ewma.initialization_window == 126
    assert config.garch.estimation_window == 750
    assert config.garch.refit_every_origins == 10
    assert config.garch.input_scale == 10.0
    assert config.garch.retry_count == 0
    assert config.garch.stationarity_tolerance == 1e-6
    assert config.evaluation.bootstrap_block_length == 10
    assert config.evaluation.bootstrap_resamples == 500
    assert config.evaluation.bootstrap_confidence == 0.90


def test_unsupported_quantile_method_fails(project_root: Path, tmp_path: Path) -> None:
    source = (project_root / "config.example.toml").read_text(encoding="utf-8")
    path = tmp_path / "quantile.toml"
    path.write_text(
        source.replace('quantile_method = "linear"', 'quantile_method = "invalid"'),
        encoding="utf-8",
    )

    with pytest.raises(ConfigInvalidError, match="quantile_method"):
        load_config(path)


def test_more_than_one_garch_retry_fails(
    project_root: Path,
    tmp_path: Path,
) -> None:
    source = (project_root / "config.example.toml").read_text(encoding="utf-8")
    path = tmp_path / "retries.toml"
    path.write_text(
        source.replace("retry_count = 1", "retry_count = 2"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigInvalidError, match="zero or one"):
        load_config(path)
