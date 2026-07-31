from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from market_risk_forecasting.config import PortfolioProxyConfig, load_config
from market_risk_forecasting.datasets import (
    build_research_dataset,
    construct_research_series,
    persist_dataset_manifest,
)
from market_risk_forecasting.errors import InputValueInvalidError
from market_risk_forecasting.upstream import load_upstream_run


def test_portfolio_proxy_matches_hand_calculated_values(
    project_root: Path,
) -> None:
    config = load_config(project_root / "config.example.toml")
    upstream = load_upstream_run(
        project_root / "data" / "fixtures" / "upstream_run",
        config.upstream,
    )
    expected = pd.read_csv(
        project_root / "data" / "fixtures" / "expected" / "research_series_head.csv",
        index_col="date",
        parse_dates=True,
    )

    actual = construct_research_series(upstream.returns, config.portfolio_proxy)

    pd.testing.assert_frame_equal(
        actual.head(10),
        expected,
        check_exact=False,
        atol=1e-10,
        rtol=0.0,
    )


def test_individual_series_are_not_transformed(project_root: Path) -> None:
    config = load_config(project_root / "config.example.toml")
    upstream = load_upstream_run(
        project_root / "data" / "fixtures" / "upstream_run",
        config.upstream,
    )

    dataset = build_research_dataset(upstream, config.portfolio_proxy)

    pd.testing.assert_frame_equal(
        dataset.returns.loc[:, ["SPY", "IEF", "GLD"]],
        upstream.returns,
    )
    assert dataset.series_order == ("SPY", "IEF", "GLD", "MIX_60_30_10")


def test_custom_universe_can_disable_portfolio_proxy() -> None:
    frame = pd.DataFrame(
        {
            "AAPL": [0.01, -0.02],
            "MSFT": [0.03, 0.01],
            "GLD": [-0.01, 0.02],
        },
        index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
    )
    disabled = PortfolioProxyConfig(False, None, ())

    actual = construct_research_series(frame, disabled)

    pd.testing.assert_frame_equal(actual, frame)
    assert tuple(actual.columns) == ("AAPL", "MSFT", "GLD")


def test_custom_universe_builds_dynamic_portfolio_proxy() -> None:
    frame = pd.DataFrame(
        {"AAPL": [0.01], "MSFT": [0.03], "GLD": [-0.01]},
        index=pd.to_datetime(["2025-01-02"]),
    )
    proxy = PortfolioProxyConfig(
        True,
        "TECH_GOLD",
        (("AAPL", 0.50), ("MSFT", 0.30), ("GLD", 0.20)),
    )

    actual = construct_research_series(frame, proxy)

    assert tuple(actual.columns) == ("AAPL", "MSFT", "GLD", "TECH_GOLD")
    assert actual.loc[pd.Timestamp("2025-01-02"), "TECH_GOLD"] == pytest.approx(0.012)


def test_dataset_manifest_preserves_lineage(project_root: Path) -> None:
    config = load_config(project_root / "config.example.toml")
    upstream = load_upstream_run(
        project_root / "data" / "fixtures" / "upstream_run",
        config.upstream,
    )

    dataset = build_research_dataset(upstream, config.portfolio_proxy)
    manifest = dataset.manifest

    assert manifest["dataset"]["observation_count"] == len(dataset.returns)
    assert manifest["upstream"]["installed_package_version"] == "0.1.0"
    assert manifest["upstream"]["instrument_order"] == ["SPY", "IEF", "GLD"]
    assert manifest["upstream"]["consumed_file_checksums"] == dict(upstream.checksums)
    assert manifest["upstream"]["quality_adjustment_counts"] == {
        "duplicate_date_instrument_rows_removed": 0,
        "invalid_price_values_removed": 0,
        "common_history_rows_removed": 0,
    }
    assert manifest["portfolio_proxy"]["weight_sum"] == pytest.approx(1.0)


def test_wrong_input_order_fails(project_root: Path) -> None:
    config = load_config(project_root / "config.example.toml")
    frame = pd.DataFrame(
        {"IEF": [0.1], "SPY": [0.2], "GLD": [0.3]},
        index=pd.to_datetime(["2020-01-02"]),
    )

    with pytest.raises(InputValueInvalidError, match="ordered"):
        construct_research_series(frame, config.portfolio_proxy)


def test_dataset_manifest_serialization_is_deterministic(
    project_root: Path,
    tmp_path: Path,
) -> None:
    config = load_config(project_root / "config.example.toml")
    upstream = load_upstream_run(
        project_root / "data" / "fixtures" / "upstream_run",
        config.upstream,
    )
    dataset = build_research_dataset(upstream, config.portfolio_proxy)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    persist_dataset_manifest(dataset, first)
    persist_dataset_manifest(dataset, second)

    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8")) == dataset.manifest
