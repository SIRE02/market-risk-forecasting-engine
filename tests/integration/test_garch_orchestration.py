from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from market_risk_forecasting.config import ForecastConfig, load_config
from market_risk_forecasting.datasets import ResearchDataset, build_research_dataset
from market_risk_forecasting.models import garch
from market_risk_forecasting.models.garch import (
    GAUSSIAN_GARCH_MODEL_ID,
    STUDENT_T_GARCH_MODEL_ID,
)
from market_risk_forecasting.orchestration import (
    FIT_DIAGNOSTIC_COLUMNS,
    persist_available_model_artifacts,
    run_available_models,
    run_garch_candidates,
)
from market_risk_forecasting.upstream import coverage_requirements, load_upstream_run


def _fixture_inputs(
    project_root: Path,
    *,
    row_count: int,
) -> tuple[ForecastConfig, ResearchDataset, str]:
    config = load_config(project_root / "config.example.toml")
    upstream = load_upstream_run(
        project_root / "data" / "fixtures" / "upstream_run",
        config.upstream,
        coverage_requirements(config),
    )
    complete = build_research_dataset(upstream, config.portfolio_proxy)
    dataset = ResearchDataset(
        returns=complete.returns.iloc[:row_count].copy(),
        manifest=complete.manifest,
    )
    return config, dataset, upstream.checksums["simple_returns.csv"]


def _successful_arch_fit(
    scaled_returns: pd.Series,
    distribution_name: str,
    starting_values: np.ndarray[Any, np.dtype[np.float64]] | None,
) -> SimpleNamespace:
    del scaled_returns, starting_values
    parameters = {
        "omega": 0.05,
        "alpha[1]": 0.05,
        "beta[1]": 0.90,
    }
    if distribution_name == "studentst":
        parameters["nu"] = 8.0
    return SimpleNamespace(
        params=pd.Series(parameters),
        conditional_volatility=pd.Series([1.0]),
        convergence_flag=0,
        optimization_result=SimpleNamespace(message="synthetic success"),
    )


def test_every_scheduled_fit_has_diagnostics_and_exact_window(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(garch, "_fit_arch_model", _successful_arch_fit)
    config, dataset, checksum = _fixture_inputs(project_root, row_count=1315)

    artifacts = run_garch_candidates(
        dataset=dataset,
        config=config,
        upstream_simple_return_checksum=checksum,
    )
    scheduled = artifacts.experiment_windows.loc[
        artifacts.experiment_windows["scheduled_refit"]
    ]
    diagnostics = artifacts.fit_diagnostics

    assert tuple(diagnostics.columns) == FIT_DIAGNOSTIC_COLUMNS
    assert len(artifacts.forecasts) == 520
    assert len(scheduled) == 32
    assert len(diagnostics) == len(scheduled)
    assert diagnostics["fit_id"].is_unique
    assert diagnostics["observation_count"].eq(1250).all()
    assert diagnostics["scaling_factor"].eq(100.0).all()
    np.testing.assert_allclose(diagnostics["parameter_persistence"], 0.95)
    assert diagnostics["converged"].all()
    assert artifacts.forecasts["fit_id"].isin(diagnostics["fit_id"]).all()
    assert set(artifacts.forecasts["status"]) == {"ok"}


def test_failed_scheduled_refit_cannot_reuse_stale_parameters(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, dataset, checksum = _fixture_inputs(project_root, row_count=1315)
    failed_origin = dataset.returns.index[1269]
    next_refit_origin = dataset.returns.index[1289]

    def selectively_failing_fit(
        scaled_returns: pd.Series,
        distribution_name: str,
        starting_values: np.ndarray[Any, np.dtype[np.float64]] | None,
    ) -> SimpleNamespace:
        del starting_values
        if (
            scaled_returns.name == "SPY"
            and distribution_name == "normal"
            and scaled_returns.index[-1] == failed_origin
        ):
            raise RuntimeError("scheduled test failure")
        return _successful_arch_fit(scaled_returns, distribution_name, None)

    monkeypatch.setattr(garch, "_fit_arch_model", selectively_failing_fit)

    artifacts = run_garch_candidates(
        dataset=dataset,
        config=config,
        upstream_simple_return_checksum=checksum,
    )
    spy = artifacts.forecasts.loc[
        (artifacts.forecasts["series_id"] == "SPY")
        & (artifacts.forecasts["model_id"] == GAUSSIAN_GARCH_MODEL_ID)
    ]
    failed_block = spy.loc[
        (spy["forecast_origin"] >= failed_origin)
        & (spy["forecast_origin"] < next_refit_origin)
    ]
    recovered = spy.loc[spy["forecast_origin"] == next_refit_origin].iloc[0]
    failed_diagnostic = artifacts.fit_diagnostics.loc[
        artifacts.fit_diagnostics["fit_id"] == failed_block["fit_id"].iloc[0]
    ].iloc[0]

    assert len(failed_block) == 20
    assert failed_block["fit_id"].nunique() == 1
    assert set(failed_block["status"]) == {"failed"}
    assert set(failed_block["error_code"]) == {"MODEL_FIT_FAILED"}
    assert not bool(failed_diagnostic["converged"])
    assert bool(failed_diagnostic["retry_used"])
    assert recovered["status"] == "ok"
    assert recovered["fit_id"] != failed_block["fit_id"].iloc[0]


def test_all_models_reconcile_with_garch_diagnostics(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(garch, "_fit_arch_model", _successful_arch_fit)
    config, dataset, checksum = _fixture_inputs(project_root, row_count=1315)

    artifacts = run_available_models(
        dataset=dataset,
        config=config,
        upstream_simple_return_checksum=checksum,
    )

    assert len(artifacts.experiment_windows) == 12_284
    assert len(artifacts.forecasts) == 12_284
    assert len(artifacts.realizations) == 4_252
    assert len(artifacts.fit_diagnostics) == 36
    assert set(artifacts.forecasts["model_id"]) == {
        "historical_variance",
        "historical_simulation",
        "ewma",
        GAUSSIAN_GARCH_MODEL_ID,
        STUDENT_T_GARCH_MODEL_ID,
    }

    output_dir = tmp_path / "all-models"
    persist_available_model_artifacts(artifacts, output_dir)
    restored = pd.read_parquet(output_dir / "fit_diagnostics.parquet")
    assert tuple(restored.columns) == FIT_DIAGNOSTIC_COLUMNS
    assert len(restored) == 36
    gaussian = restored.loc[restored["model_id"] == GAUSSIAN_GARCH_MODEL_ID]
    student_t = restored.loc[restored["model_id"] == STUDENT_T_GARCH_MODEL_ID]
    assert gaussian["degrees_of_freedom"].isna().all()
    assert student_t["degrees_of_freedom"].eq(8.0).all()


def test_appending_future_rows_does_not_change_prior_garch_forecasts(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(garch, "_fit_arch_model", _successful_arch_fit)
    config, short_dataset, checksum = _fixture_inputs(
        project_root,
        row_count=1295,
    )
    _, extended_dataset, _ = _fixture_inputs(project_root, row_count=1315)

    short = run_garch_candidates(
        dataset=short_dataset,
        config=config,
        upstream_simple_return_checksum=checksum,
    )
    extended = run_garch_candidates(
        dataset=extended_dataset,
        config=config,
        upstream_simple_return_checksum=checksum,
    )
    prior = extended.forecasts.loc[
        extended.forecasts["target_date"] <= short_dataset.returns.index[-1]
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(short.forecasts, prior)


def test_real_arch_backend_produces_typed_results(project_root: Path) -> None:
    config, dataset, checksum = _fixture_inputs(project_root, row_count=1252)

    artifacts = run_garch_candidates(
        dataset=dataset,
        config=config,
        upstream_simple_return_checksum=checksum,
    )

    assert len(artifacts.forecasts) == 16
    assert len(artifacts.fit_diagnostics) == 8
    assert set(artifacts.forecasts["status"]).issubset({"ok", "failed"})
    assert artifacts.fit_diagnostics["fit_id"].is_unique
