from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from market_risk_forecasting.config import ForecastConfig, load_config
from market_risk_forecasting.datasets import ResearchDataset, build_research_dataset
from market_risk_forecasting.errors import OutputCollisionError
from market_risk_forecasting.evaluation import EvaluationArtifacts
from market_risk_forecasting.orchestration import (
    EvaluatedModelArtifacts,
    evaluate_model_artifacts,
    persist_evaluated_model_artifacts,
    run_benchmarks_and_ewma,
)
from market_risk_forecasting.upstream import coverage_requirements, load_upstream_run


def _fixture_inputs(
    project_root: Path,
) -> tuple[ForecastConfig, ResearchDataset, str]:
    config = load_config(project_root / "config.example.toml")
    upstream = load_upstream_run(
        project_root / "data" / "fixtures" / "upstream_run",
        config.upstream,
        coverage_requirements(config),
    )
    dataset = build_research_dataset(upstream, config.portfolio_proxy)
    return config, dataset, upstream.checksums["simple_returns.csv"]


def _frames(artifacts: EvaluationArtifacts) -> tuple[pd.DataFrame, ...]:
    return (
        artifacts.forecast_availability,
        artifacts.variance_scores,
        artifacts.quantile_scores,
        artifacts.coverage_tests,
        artifacts.bootstrap_comparisons,
        artifacts.period_breakdowns,
    )


def test_synthetic_evaluation_is_reproducible_and_period_separated(
    project_root: Path,
) -> None:
    config, dataset, checksum = _fixture_inputs(project_root)
    models = run_benchmarks_and_ewma(
        dataset=dataset,
        config=config,
        upstream_simple_return_checksum=checksum,
    )

    first = evaluate_model_artifacts(artifacts=models, config=config)
    second = evaluate_model_artifacts(artifacts=models, config=config)

    for first_frame, second_frame in zip(
        _frames(first),
        _frames(second),
        strict=True,
    ):
        pd.testing.assert_frame_equal(first_frame, second_frame)
    assert set(first.variance_scores["period"]) == {"validation", "test"}
    assert set(first.quantile_scores["period"]) == {"validation", "test"}
    assert set(first.bootstrap_comparisons["period"]) == {
        "validation",
        "test",
    }
    assert set(first.period_breakdowns["period"]) == {
        "test_2020",
        "test_2021",
        "test_2022",
        "test_2023",
        "test_2024",
        "test_2025",
    }


def test_all_numerical_artifacts_round_trip_without_overwrite(
    project_root: Path,
    tmp_path: Path,
) -> None:
    config, dataset, checksum = _fixture_inputs(project_root)
    models = run_benchmarks_and_ewma(
        dataset=dataset,
        config=config,
        upstream_simple_return_checksum=checksum,
    )
    evaluation = evaluate_model_artifacts(artifacts=models, config=config)
    output_dir = tmp_path / "evaluated"

    persist_evaluated_model_artifacts(
        EvaluatedModelArtifacts(models=models, evaluation=evaluation),
        output_dir,
    )

    expected = {
        "experiment_windows.csv",
        "realizations.parquet",
        "forecasts.parquet",
        "fit_diagnostics.parquet",
        "forecast_availability.csv",
        "variance_scores.csv",
        "quantile_scores.csv",
        "coverage_tests.csv",
        "bootstrap_comparisons.csv",
        "period_breakdowns.csv",
    }
    assert {path.name for path in output_dir.iterdir()} == expected
    restored = pd.read_csv(output_dir / "bootstrap_comparisons.csv")
    assert len(restored) == len(evaluation.bootstrap_comparisons)

    with pytest.raises(OutputCollisionError):
        persist_evaluated_model_artifacts(
            EvaluatedModelArtifacts(models=models, evaluation=evaluation),
            output_dir,
        )
