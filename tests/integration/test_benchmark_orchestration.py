from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from market_risk_forecasting.config import load_config
from market_risk_forecasting.datasets import ResearchDataset, build_research_dataset
from market_risk_forecasting.errors import OutputCollisionError
from market_risk_forecasting.models.historical import (
    HISTORICAL_SIMULATION_MODEL_ID,
    HISTORICAL_VARIANCE_MODEL_ID,
)
from market_risk_forecasting.orchestration import (
    EXPERIMENT_WINDOW_COLUMNS,
    FORECAST_COLUMNS,
    REALIZATION_COLUMNS,
    persist_benchmark_artifacts,
    run_historical_benchmarks,
)
from market_risk_forecasting.upstream import load_upstream_run


def _fixture_inputs(project_root: Path) -> tuple[object, object, str]:
    config = load_config(project_root / "config.example.toml")
    upstream = load_upstream_run(
        project_root / "data" / "fixtures" / "upstream_run",
        config.upstream,
    )
    dataset = build_research_dataset(upstream, config.portfolio_proxy)
    return config, dataset, upstream.checksums["simple_returns.csv"]


def test_synthetic_benchmark_experiment_reconciles(
    project_root: Path,
) -> None:
    config, dataset, checksum = _fixture_inputs(project_root)

    artifacts = run_historical_benchmarks(
        dataset=dataset,
        config=config,
        upstream_simple_return_checksum=checksum,
    )

    assert tuple(artifacts.experiment_windows.columns) == EXPERIMENT_WINDOW_COLUMNS
    assert tuple(artifacts.realizations.columns) == REALIZATION_COLUMNS
    assert tuple(artifacts.forecasts.columns) == FORECAST_COLUMNS
    assert len(artifacts.experiment_windows) == 36_648
    assert len(artifacts.forecasts) == 36_648
    assert len(artifacts.realizations) == 18_820
    assert artifacts.forecasts["forecast_id"].is_unique
    assert artifacts.forecasts["fit_id"].str.startswith("fit_").all()
    assert artifacts.forecasts["forecast_id"].str.startswith("fcst_").all()
    assert set(artifacts.forecasts["status"]) == {"ok"}
    assert set(artifacts.experiment_windows["period"]) == {
        "development",
        "validation",
        "test",
    }
    assert (
        artifacts.realizations["squared_return"]
        == artifacts.realizations["simple_return"].pow(2)
    ).all()
    assert (
        artifacts.realizations["loss"] == -artifacts.realizations["simple_return"]
    ).all()


def test_benchmark_outputs_have_model_specific_nulls(project_root: Path) -> None:
    config, dataset, checksum = _fixture_inputs(project_root)
    artifacts = run_historical_benchmarks(
        dataset=dataset,
        config=config,
        upstream_simple_return_checksum=checksum,
    )

    variance = artifacts.forecasts.loc[
        artifacts.forecasts["model_id"] == HISTORICAL_VARIANCE_MODEL_ID
    ]
    simulation = artifacts.forecasts.loc[
        artifacts.forecasts["model_id"] == HISTORICAL_SIMULATION_MODEL_ID
    ]
    assert variance["variance"].gt(0.0).all()
    assert variance["volatility"].gt(0.0).all()
    assert variance["return_quantile_0_05"].isna().all()
    assert variance["var_0_95"].isna().all()
    assert simulation["variance"].isna().all()
    assert simulation["volatility"].isna().all()
    assert simulation["return_quantile_0_05"].notna().all()
    assert simulation["var_0_95"].le(simulation["var_0_99"]).all()
    assert set(artifacts.forecasts["warning_codes"]) == {"[]"}


def test_benchmark_ids_and_values_are_reproducible(project_root: Path) -> None:
    config, dataset, checksum = _fixture_inputs(project_root)

    first = run_historical_benchmarks(
        dataset=dataset,
        config=config,
        upstream_simple_return_checksum=checksum,
    )
    second = run_historical_benchmarks(
        dataset=dataset,
        config=config,
        upstream_simple_return_checksum=checksum,
    )

    pd.testing.assert_frame_equal(first.forecasts, second.forecasts)
    pd.testing.assert_frame_equal(
        first.experiment_windows,
        second.experiment_windows,
    )
    pd.testing.assert_frame_equal(first.realizations, second.realizations)


def test_future_append_does_not_change_prior_forecasts(
    project_root: Path,
) -> None:
    config, dataset, checksum = _fixture_inputs(project_root)
    short = ResearchDataset(
        returns=dataset.returns.iloc[:800].copy(),
        manifest=dataset.manifest,
    )
    extended = ResearchDataset(
        returns=dataset.returns.iloc[:820].copy(),
        manifest=dataset.manifest,
    )

    short_artifacts = run_historical_benchmarks(
        dataset=short,
        config=config,
        upstream_simple_return_checksum=checksum,
    )
    extended_artifacts = run_historical_benchmarks(
        dataset=extended,
        config=config,
        upstream_simple_return_checksum=checksum,
    )
    prior = extended_artifacts.forecasts.loc[
        extended_artifacts.forecasts["target_date"] <= short.returns.index[-1]
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(short_artifacts.forecasts, prior)


def test_failed_variance_is_preserved_without_fallback(
    project_root: Path,
) -> None:
    config, dataset, checksum = _fixture_inputs(project_root)
    constant = ResearchDataset(
        returns=dataset.returns.iloc[:600].copy() * 0.0,
        manifest=dataset.manifest,
    )

    artifacts = run_historical_benchmarks(
        dataset=constant,
        config=config,
        upstream_simple_return_checksum=checksum,
    )
    variance = artifacts.forecasts.loc[
        artifacts.forecasts["model_id"] == HISTORICAL_VARIANCE_MODEL_ID
    ]
    simulation = artifacts.forecasts.loc[
        artifacts.forecasts["model_id"] == HISTORICAL_SIMULATION_MODEL_ID
    ]

    assert set(variance["status"]) == {"failed"}
    assert set(variance["error_code"]) == {"NONPOSITIVE_VARIANCE"}
    assert variance["variance"].isna().all()
    assert set(simulation["status"]) == {"ok"}


def test_phase_two_artifacts_round_trip(
    project_root: Path,
    tmp_path: Path,
) -> None:
    config, dataset, checksum = _fixture_inputs(project_root)
    artifacts = run_historical_benchmarks(
        dataset=dataset,
        config=config,
        upstream_simple_return_checksum=checksum,
    )
    output_dir = tmp_path / "benchmarks"

    persist_benchmark_artifacts(artifacts, output_dir)

    windows = pd.read_csv(output_dir / "experiment_windows.csv")
    realizations = pd.read_parquet(output_dir / "realizations.parquet")
    forecasts = pd.read_parquet(output_dir / "forecasts.parquet")
    assert tuple(windows.columns) == EXPERIMENT_WINDOW_COLUMNS
    assert tuple(realizations.columns) == REALIZATION_COLUMNS
    assert tuple(forecasts.columns) == FORECAST_COLUMNS
    assert len(windows) == len(artifacts.experiment_windows)
    assert len(realizations) == len(artifacts.realizations)
    assert len(forecasts) == len(artifacts.forecasts)

    with pytest.raises(OutputCollisionError):
        persist_benchmark_artifacts(artifacts, output_dir)
