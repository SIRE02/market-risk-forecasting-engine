from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from historical_asset_risk.artifacts import load_artifact
from historical_asset_risk.cli import run_analysis
from historical_asset_risk.config import AnalysisConfig

from market_risk_forecasting.config import load_config
from market_risk_forecasting.errors import UpstreamQualityGateFailedError
from market_risk_forecasting.upstream import (
    UpstreamCoverageRequirements,
    load_upstream_run,
)


def test_synthetic_fixture_satisfies_public_upstream_loader(
    project_root: Path,
) -> None:
    fixture = project_root / "data" / "fixtures" / "upstream_run"
    manifest = json.loads((fixture / "run_manifest.json").read_text(encoding="utf-8"))
    quality = json.loads(
        (fixture / "data_quality_report.json").read_text(encoding="utf-8")
    )

    frame = load_artifact(
        fixture / "simple_returns.csv",
        fixture / "run_manifest.json",
    )

    assert manifest["fixture"]["synthetic"] is True
    assert manifest["project"] == "historical-asset-risk-engine"
    assert manifest["project_version"] == "0.2.1"
    assert manifest["data_source"]["instruments"] == ["SPY", "IEF", "GLD"]
    assert list(frame.columns) == ["SPY", "IEF", "GLD"]
    assert quality["common_date_count_after_alignment"] == 4958
    assert quality["first_common_date"] == "2007-01-01"
    assert quality["last_common_date"] == "2025-12-31"
    assert not frame.isna().any(axis=None)
    assert frame.index.is_monotonic_increasing
    assert frame.index.is_unique


@pytest.mark.parametrize("missing_position", [None, 1, 4])
def test_real_hare_returns_only_run_and_declared_session_calendar(
    project_root: Path, tmp_path: Path, missing_position: int | None
) -> None:
    dates = pd.bdate_range("2024-01-02", periods=8)
    records = [
        {
            "date": day.date().isoformat(),
            "ticker": ticker,
            "adjusted_close": 100 + position + offset,
        }
        for offset, ticker in enumerate(("SPY", "IEF", "GLD"))
        for position, day in enumerate(dates)
        if position != missing_position
    ]
    source = tmp_path / "prices.csv"
    pd.DataFrame(records).to_csv(source, index=False)
    run_dir = tmp_path / "hare-run"
    run_analysis(
        AnalysisConfig(
            provider="csv",
            csv_path=source,
            tickers=("SPY", "IEF", "GLD"),
            start_date="2024-01-02",
            end_date="2024-01-12",
            rolling_window=3,
            output_dir=run_dir,
        )
    )
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["project_version"] == "0.2.1"
    assert manifest["data_source"]["price_content_hash"].startswith("sha256:")
    assert "git_worktree_dirty" in manifest
    assert manifest["estimation_conventions"]["missing_data"][
        "gap_spanning_return_policy"
    ] == (
        "fail_when_retained_complete_case_dates_are_nonconsecutive_within_the_provider_observation_union"
    )
    config = load_config(project_root / "config.example.toml")
    expected = replace(config.upstream, market_calendar_id="XNYS")
    coverage = UpstreamCoverageRequirements(6, date(2024, 1, 2), date(2024, 1, 10))
    if missing_position is not None:
        with pytest.raises(UpstreamQualityGateFailedError, match="skip XNYS session"):
            load_upstream_run(run_dir, expected, coverage)
    else:
        run = load_upstream_run(run_dir, expected, coverage)
        prices = load_artifact(
            run_dir / "adjusted_prices.csv", run_dir / "run_manifest.json"
        )
        pd.testing.assert_frame_equal(
            run.returns, prices.pct_change(fill_method=None).iloc[1:]
        )
