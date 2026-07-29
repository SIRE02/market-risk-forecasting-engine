from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from market_risk_forecasting.config import ForecastConfig, load_config
from market_risk_forecasting.datasets import ResearchDataset, build_research_dataset
from market_risk_forecasting.models.ewma import EWMA_MODEL_ID
from market_risk_forecasting.models.historical import (
    HISTORICAL_VARIANCE_MODEL_ID,
)
from market_risk_forecasting.orchestration import (
    EXPERIMENT_WINDOW_COLUMNS,
    FIT_DIAGNOSTIC_COLUMNS,
    FORECAST_COLUMNS,
    REALIZATION_COLUMNS,
    persist_available_model_artifacts,
    run_available_models,
    run_ewma_candidate,
)
from market_risk_forecasting.upstream import load_upstream_run


def _fixture_inputs(
    project_root: Path,
) -> tuple[ForecastConfig, ResearchDataset, str]:
    config = load_config(project_root / "config.example.toml")
    upstream = load_upstream_run(
        project_root / "data" / "fixtures" / "upstream_run",
        config.upstream,
    )
    dataset = build_research_dataset(upstream, config.portfolio_proxy)
    return config, dataset, upstream.checksums["simple_returns.csv"]


def test_ewma_runs_on_the_historical_variance_common_panel(
    project_root: Path,
) -> None:
    config, dataset, checksum = _fixture_inputs(project_root)

    artifacts = run_available_models(
        dataset=dataset,
        config=config,
        upstream_simple_return_checksum=checksum,
    )
    ewma = artifacts.forecasts.loc[
        artifacts.forecasts["model_id"] == EWMA_MODEL_ID
    ].reset_index(drop=True)
    historical = artifacts.forecasts.loc[
        artifacts.forecasts["model_id"] == HISTORICAL_VARIANCE_MODEL_ID
    ].reset_index(drop=True)
    keys = ["series_id", "forecast_origin", "target_date"]

    assert len(ewma) == 18_820
    assert len(historical) == 18_820
    pd.testing.assert_frame_equal(ewma[keys], historical[keys])
    assert set(ewma["status"]) == {"ok"}
    assert ewma["variance"].gt(0.0).all()
    assert ewma["var_0_95"].le(ewma["var_0_99"]).all()


def test_ewma_uses_one_fit_identity_per_series(project_root: Path) -> None:
    config, dataset, checksum = _fixture_inputs(project_root)

    artifacts = run_ewma_candidate(
        dataset=dataset,
        config=config,
        upstream_simple_return_checksum=checksum,
    )
    diagnostics = artifacts.fit_diagnostics

    assert tuple(diagnostics.columns) == FIT_DIAGNOSTIC_COLUMNS
    assert len(diagnostics) == 4
    assert diagnostics["fit_id"].is_unique
    assert artifacts.forecasts.groupby("series_id")["fit_id"].nunique().eq(1).all()
    assert diagnostics["parameter_persistence"].eq(0.94).all()
    assert diagnostics["scaling_factor"].eq(1.0).all()
    assert diagnostics["converged"].all()
    assert set(diagnostics["optimizer_status"]) == {"not_applicable_fixed_parameter"}


@pytest.mark.parametrize("target_date", ["2015-01-02", "2020-01-02"])
def test_ewma_state_does_not_reset_at_period_boundaries(
    project_root: Path,
    target_date: str,
) -> None:
    config, dataset, checksum = _fixture_inputs(project_root)
    artifacts = run_ewma_candidate(
        dataset=dataset,
        config=config,
        upstream_simple_return_checksum=checksum,
    )
    spy = artifacts.forecasts.loc[
        (artifacts.forecasts["model_id"] == EWMA_MODEL_ID)
        & (artifacts.forecasts["series_id"] == "SPY")
    ].reset_index(drop=True)
    current_position = spy.index[spy["target_date"].eq(pd.Timestamp(target_date))][0]
    current = spy.iloc[current_position]
    previous = spy.iloc[current_position - 1]
    origin_return = float(dataset.returns.loc[current["forecast_origin"], "SPY"])
    expected = 0.94 * float(previous["variance"]) + 0.06 * origin_return**2

    assert current["variance"] == pytest.approx(expected)
    assert current["fit_id"] == previous["fit_id"]


def test_available_model_artifacts_round_trip(
    project_root: Path,
    tmp_path: Path,
) -> None:
    config, dataset, checksum = _fixture_inputs(project_root)
    artifacts = run_available_models(
        dataset=dataset,
        config=config,
        upstream_simple_return_checksum=checksum,
    )
    output_dir = tmp_path / "available-models"

    persist_available_model_artifacts(artifacts, output_dir)

    windows = pd.read_csv(output_dir / "experiment_windows.csv")
    realizations = pd.read_parquet(output_dir / "realizations.parquet")
    forecasts = pd.read_parquet(output_dir / "forecasts.parquet")
    diagnostics = pd.read_parquet(output_dir / "fit_diagnostics.parquet")
    assert tuple(windows.columns) == EXPERIMENT_WINDOW_COLUMNS
    assert tuple(realizations.columns) == REALIZATION_COLUMNS
    assert tuple(forecasts.columns) == FORECAST_COLUMNS
    assert tuple(diagnostics.columns) == FIT_DIAGNOSTIC_COLUMNS
    assert len(windows) == 55_468
    assert len(realizations) == 18_820
    assert len(forecasts) == 55_468
    assert len(diagnostics) == 4
