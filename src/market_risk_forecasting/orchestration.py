"""Deterministic experiment orchestration assembled by implementation phase."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from market_risk_forecasting import __version__
from market_risk_forecasting.config import ForecastConfig
from market_risk_forecasting.datasets import ResearchDataset
from market_risk_forecasting.errors import (
    MarketRiskForecastingError,
    NonfiniteVarianceError,
    NonpositiveVarianceError,
    OutputCollisionError,
    WindowAlignmentError,
)
from market_risk_forecasting.evaluation import (
    EvaluationArtifacts,
    evaluate_forecasts,
    persist_evaluation_artifacts,
)
from market_risk_forecasting.identifiers import make_fit_id, make_forecast_id
from market_risk_forecasting.models.ewma import EWMA_MODEL_ID, EwmaModel
from market_risk_forecasting.models.garch import (
    GAUSSIAN_GARCH_MODEL_ID,
    STUDENT_T_GARCH_MODEL_ID,
    GarchFitOutcome,
    GarchForecast,
    GarchModel,
    GarchState,
)
from market_risk_forecasting.models.historical import (
    HISTORICAL_SIMULATION_MODEL_ID,
    HISTORICAL_VARIANCE_MODEL_ID,
    HistoricalSimulationModel,
    HistoricalVarianceModel,
)
from market_risk_forecasting.windows import (
    ForecastWindow,
    classify_target_date,
    iter_expanding_forecast_windows,
    iter_forecast_windows,
    iter_refit_forecast_windows,
    validate_canonical_index,
)

EXPERIMENT_WINDOW_COLUMNS = (
    "series_id",
    "model_id",
    "forecast_origin",
    "target_date",
    "train_start",
    "train_end",
    "train_observation_count",
    "period",
    "scheduled_refit",
)
REALIZATION_COLUMNS = (
    "series_id",
    "forecast_origin",
    "target_date",
    "simple_return",
    "squared_return",
    "loss",
    "period",
)
FORECAST_COLUMNS = (
    "experiment_id",
    "forecast_id",
    "series_id",
    "model_id",
    "model_version",
    "fit_id",
    "forecast_origin",
    "target_date",
    "variance",
    "volatility",
    "return_quantile_0_05",
    "var_0_95",
    "return_quantile_0_01",
    "var_0_99",
    "status",
    "error_code",
    "warning_codes",
)
FIT_DIAGNOSTIC_COLUMNS = (
    "fit_id",
    "series_id",
    "model_id",
    "fit_origin",
    "train_start",
    "train_end",
    "observation_count",
    "omega",
    "alpha",
    "beta",
    "degrees_of_freedom",
    "parameter_persistence",
    "converged",
    "optimizer_status",
    "retry_used",
    "runtime_seconds",
    "scaling_factor",
    "warning_codes",
)


@dataclass(frozen=True)
class BenchmarkArtifacts:
    """In-memory forecast artifacts with stable public schemas."""

    experiment_windows: pd.DataFrame
    realizations: pd.DataFrame
    forecasts: pd.DataFrame
    fit_diagnostics: pd.DataFrame


@dataclass(frozen=True)
class EvaluatedModelArtifacts:
    """Forecast artifacts paired with deterministic evaluation tables."""

    models: BenchmarkArtifacts
    evaluation: EvaluationArtifacts


def _all_benchmark_windows(
    dataset: ResearchDataset,
    config: ForecastConfig,
) -> list[ForecastWindow]:
    index = pd.DatetimeIndex(dataset.returns.index)
    validate_canonical_index(index)
    specifications = (
        (HISTORICAL_VARIANCE_MODEL_ID, config.historical.variance_window),
        (HISTORICAL_SIMULATION_MODEL_ID, config.historical.var_window),
    )
    result: list[ForecastWindow] = []
    for series_id in dataset.series_order:
        for model_id, training_window in specifications:
            result.extend(
                iter_forecast_windows(
                    index=index,
                    series_id=series_id,
                    model_id=model_id,
                    training_window=training_window,
                    periods=config.periods,
                )
            )
    return result


def _all_ewma_windows(
    dataset: ResearchDataset,
    config: ForecastConfig,
) -> list[ForecastWindow]:
    index = pd.DatetimeIndex(dataset.returns.index)
    validate_canonical_index(index)
    result: list[ForecastWindow] = []
    for series_id in dataset.series_order:
        result.extend(
            iter_expanding_forecast_windows(
                index=index,
                series_id=series_id,
                model_id=EWMA_MODEL_ID,
                initial_observations=config.ewma.initialization_window,
                periods=config.periods,
            )
        )
    return result


def _garch_models(config: ForecastConfig) -> tuple[GarchModel, GarchModel]:
    return (
        GarchModel(
            distribution="gaussian",
            estimation_window=config.garch.estimation_window,
            input_scale=config.garch.input_scale,
            retry_count=config.garch.retry_count,
            stationarity_tolerance=config.garch.stationarity_tolerance,
        ),
        GarchModel(
            distribution="student_t",
            estimation_window=config.garch.estimation_window,
            input_scale=config.garch.input_scale,
            retry_count=config.garch.retry_count,
            stationarity_tolerance=config.garch.stationarity_tolerance,
        ),
    )


def _all_garch_windows(
    dataset: ResearchDataset,
    config: ForecastConfig,
) -> list[ForecastWindow]:
    index = pd.DatetimeIndex(dataset.returns.index)
    validate_canonical_index(index)
    result: list[ForecastWindow] = []
    for series_id in dataset.series_order:
        for model in _garch_models(config):
            result.extend(
                iter_refit_forecast_windows(
                    index=index,
                    series_id=series_id,
                    model_id=model.model_id,
                    training_window=config.garch.estimation_window,
                    refit_every_origins=config.garch.refit_every_origins,
                    periods=config.periods,
                )
            )
    return result


def _realization_frame(
    dataset: ResearchDataset,
    config: ForecastConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start_origin = (
        min(
            config.historical.variance_window,
            config.historical.var_window,
        )
        - 1
    )
    index = pd.DatetimeIndex(dataset.returns.index)
    for series_id in dataset.series_order:
        series = dataset.returns[series_id]
        for origin_position in range(start_origin, len(index) - 1):
            target_position = origin_position + 1
            forecast_origin = pd.Timestamp(index[origin_position])
            target_date = pd.Timestamp(index[target_position])
            period = classify_target_date(target_date, config.periods)
            if period is None:
                continue
            realization = float(series.iloc[target_position])
            rows.append(
                {
                    "series_id": series_id,
                    "forecast_origin": forecast_origin,
                    "target_date": target_date,
                    "simple_return": realization,
                    "squared_return": realization**2,
                    "loss": -realization,
                    "period": period,
                }
            )
    return pd.DataFrame(rows, columns=REALIZATION_COLUMNS)


def _failed_values(error: MarketRiskForecastingError) -> dict[str, Any]:
    return {
        "variance": None,
        "volatility": None,
        "return_quantile_0_05": None,
        "var_0_95": None,
        "return_quantile_0_01": None,
        "var_0_99": None,
        "status": "failed",
        "error_code": error.code.value,
    }


def _forecast_values(
    *,
    window: ForecastWindow,
    forecast_paths: dict[tuple[str, str], pd.DataFrame],
) -> dict[str, Any]:
    try:
        path = forecast_paths[(window.series_id, window.model_id)]
        values = path.iloc[window.origin_position]
        if window.model_id == HISTORICAL_VARIANCE_MODEL_ID:
            variance = float(values["variance"])
            volatility = float(values["volatility"])
            if not math.isfinite(variance) or not math.isfinite(volatility):
                raise NonfiniteVarianceError(
                    "Historical sample variance is non-finite."
                )
            if variance <= 0.0:
                raise NonpositiveVarianceError(
                    "Historical sample variance is not strictly positive."
                )
            return {
                "variance": variance,
                "volatility": volatility,
                "return_quantile_0_05": None,
                "var_0_95": None,
                "return_quantile_0_01": None,
                "var_0_99": None,
                "status": "ok",
                "error_code": None,
            }
        if window.model_id == HISTORICAL_SIMULATION_MODEL_ID:
            return {
                "variance": None,
                "volatility": None,
                "return_quantile_0_05": float(values["return_quantile_0_05"]),
                "var_0_95": float(values["var_0_95"]),
                "return_quantile_0_01": float(values["return_quantile_0_01"]),
                "var_0_99": float(values["var_0_99"]),
                "status": "ok",
                "error_code": None,
            }
        if window.model_id == EWMA_MODEL_ID:
            numeric = {
                name: float(values[name])
                for name in (
                    "variance",
                    "volatility",
                    "return_quantile_0_05",
                    "var_0_95",
                    "return_quantile_0_01",
                    "var_0_99",
                )
            }
            if not all(math.isfinite(value) for value in numeric.values()):
                raise NonfiniteVarianceError(
                    "EWMA forecast contains non-finite values."
                )
            if numeric["variance"] <= 0.0:
                raise NonpositiveVarianceError(
                    "EWMA variance is not strictly positive."
                )
            return {
                **numeric,
                "status": "ok",
                "error_code": None,
            }
        raise WindowAlignmentError(f"Unknown benchmark model {window.model_id!r}.")
    except MarketRiskForecastingError as exc:
        return _failed_values(exc)


def _forecast_frame(
    *,
    windows: list[ForecastWindow],
    dataset: ResearchDataset,
    config: ForecastConfig,
    upstream_simple_return_checksum: str,
) -> pd.DataFrame:
    variance_model = HistoricalVarianceModel(window=config.historical.variance_window)
    simulation_model = HistoricalSimulationModel(
        window=config.historical.var_window,
    )
    forecast_paths: dict[tuple[str, str], pd.DataFrame] = {}
    for series_id in dataset.series_order:
        returns = dataset.returns[series_id]
        forecast_paths[(series_id, variance_model.model_id)] = (
            variance_model.forecast_path(returns)
        )
        forecast_paths[(series_id, simulation_model.model_id)] = (
            simulation_model.forecast_path(returns)
        )
    rows: list[dict[str, Any]] = []
    for window in windows:
        fit_id = make_fit_id(
            experiment_id=config.experiment.experiment_id,
            series_id=window.series_id,
            model_id=window.model_id,
            fit_origin=window.forecast_origin,
            train_start=window.train_start,
            train_end=window.train_end,
            upstream_simple_return_checksum=upstream_simple_return_checksum,
            package_version=__version__,
        )
        forecast_id = make_forecast_id(
            experiment_id=config.experiment.experiment_id,
            fit_id=fit_id,
            series_id=window.series_id,
            model_id=window.model_id,
            forecast_origin=window.forecast_origin,
            target_date=window.target_date,
        )
        values = _forecast_values(
            window=window,
            forecast_paths=forecast_paths,
        )
        rows.append(
            {
                "experiment_id": config.experiment.experiment_id,
                "forecast_id": forecast_id,
                "series_id": window.series_id,
                "model_id": window.model_id,
                "model_version": __version__,
                "fit_id": fit_id,
                "forecast_origin": window.forecast_origin,
                "target_date": window.target_date,
                **values,
                "warning_codes": json.dumps([]),
            }
        )
    return pd.DataFrame(rows, columns=FORECAST_COLUMNS)


def _ewma_forecast_frame(
    *,
    windows: list[ForecastWindow],
    dataset: ResearchDataset,
    config: ForecastConfig,
    upstream_simple_return_checksum: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = EwmaModel(
        lambda_=config.ewma.lambda_,
        initialization_window=config.ewma.initialization_window,
    )
    forecast_paths = {
        (series_id, model.model_id): model.forecast_path(dataset.returns[series_id])
        for series_id in dataset.series_order
    }
    first_windows = {
        series_id: next(window for window in windows if window.series_id == series_id)
        for series_id in dataset.series_order
    }
    fit_ids = {
        series_id: make_fit_id(
            experiment_id=config.experiment.experiment_id,
            series_id=series_id,
            model_id=model.model_id,
            fit_origin=window.forecast_origin,
            train_start=window.train_start,
            train_end=window.train_end,
            upstream_simple_return_checksum=upstream_simple_return_checksum,
            package_version=__version__,
        )
        for series_id, window in first_windows.items()
    }
    forecast_rows: list[dict[str, Any]] = []
    for window in windows:
        fit_id = fit_ids[window.series_id]
        forecast_id = make_forecast_id(
            experiment_id=config.experiment.experiment_id,
            fit_id=fit_id,
            series_id=window.series_id,
            model_id=window.model_id,
            forecast_origin=window.forecast_origin,
            target_date=window.target_date,
        )
        forecast_rows.append(
            {
                "experiment_id": config.experiment.experiment_id,
                "forecast_id": forecast_id,
                "series_id": window.series_id,
                "model_id": window.model_id,
                "model_version": __version__,
                "fit_id": fit_id,
                "forecast_origin": window.forecast_origin,
                "target_date": window.target_date,
                **_forecast_values(
                    window=window,
                    forecast_paths=forecast_paths,
                ),
                "warning_codes": json.dumps([]),
            }
        )

    diagnostic_rows = [
        {
            "fit_id": fit_ids[series_id],
            "series_id": series_id,
            "model_id": model.model_id,
            "fit_origin": window.forecast_origin,
            "train_start": window.train_start,
            "train_end": window.train_end,
            "observation_count": config.ewma.initialization_window,
            "omega": None,
            "alpha": None,
            "beta": None,
            "degrees_of_freedom": None,
            "parameter_persistence": config.ewma.lambda_,
            "converged": True,
            "optimizer_status": "not_applicable_fixed_parameter",
            "retry_used": False,
            "runtime_seconds": 0.0,
            "scaling_factor": 1.0,
            "warning_codes": json.dumps([]),
        }
        for series_id, window in first_windows.items()
    ]
    return (
        pd.DataFrame(forecast_rows, columns=FORECAST_COLUMNS),
        pd.DataFrame(diagnostic_rows, columns=FIT_DIAGNOSTIC_COLUMNS),
    )


def _garch_forecast_values(forecast: GarchForecast) -> dict[str, Any]:
    return {
        "variance": forecast.variance,
        "volatility": forecast.volatility,
        "return_quantile_0_05": forecast.return_quantile_0_05,
        "var_0_95": forecast.var_0_95,
        "return_quantile_0_01": forecast.return_quantile_0_01,
        "var_0_99": forecast.var_0_99,
        "status": "ok",
        "error_code": None,
    }


def _failed_forecast_values(error_code: str) -> dict[str, Any]:
    return {
        "variance": None,
        "volatility": None,
        "return_quantile_0_05": None,
        "var_0_95": None,
        "return_quantile_0_01": None,
        "var_0_99": None,
        "status": "failed",
        "error_code": error_code,
    }


def _garch_diagnostic_record(
    *,
    fit_id: str,
    window: ForecastWindow,
    outcome: GarchFitOutcome,
) -> dict[str, Any]:
    parameters = outcome.parameters
    return {
        "fit_id": fit_id,
        "series_id": window.series_id,
        "model_id": window.model_id,
        "fit_origin": window.forecast_origin,
        "train_start": window.train_start,
        "train_end": window.train_end,
        "observation_count": window.train_observation_count,
        "omega": parameters.omega if parameters is not None else None,
        "alpha": parameters.alpha if parameters is not None else None,
        "beta": parameters.beta if parameters is not None else None,
        "degrees_of_freedom": (
            parameters.degrees_of_freedom if parameters is not None else None
        ),
        "parameter_persistence": (
            parameters.persistence if parameters is not None else None
        ),
        "converged": outcome.converged,
        "optimizer_status": outcome.optimizer_status,
        "retry_used": outcome.retry_used,
        "runtime_seconds": outcome.runtime_seconds,
        "scaling_factor": outcome.scaling_factor,
        "warning_codes": json.dumps(list(outcome.warning_codes)),
    }


def _garch_forecast_frames(
    *,
    windows: list[ForecastWindow],
    dataset: ResearchDataset,
    config: ForecastConfig,
    upstream_simple_return_checksum: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows_by_series_model = {
        (series_id, model.model_id): [
            window
            for window in windows
            if window.series_id == series_id and window.model_id == model.model_id
        ]
        for series_id in dataset.series_order
        for model in _garch_models(config)
    }
    forecast_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for series_id in dataset.series_order:
        series = dataset.returns[series_id]
        for model in _garch_models(config):
            active_state: GarchState | None = None
            active_fit_id: str | None = None
            active_error_code = "MODEL_FIT_FAILED"
            for window in windows_by_series_model[(series_id, model.model_id)]:
                if window.scheduled_refit:
                    active_fit_id = make_fit_id(
                        experiment_id=config.experiment.experiment_id,
                        series_id=series_id,
                        model_id=model.model_id,
                        fit_origin=window.forecast_origin,
                        train_start=window.train_start,
                        train_end=window.train_end,
                        upstream_simple_return_checksum=(
                            upstream_simple_return_checksum
                        ),
                        package_version=__version__,
                    )
                    training = series.iloc[
                        window.origin_position
                        - config.garch.estimation_window
                        + 1 : window.origin_position + 1
                    ]
                    outcome = model.fit(training)
                    diagnostic_rows.append(
                        _garch_diagnostic_record(
                            fit_id=active_fit_id,
                            window=window,
                            outcome=outcome,
                        )
                    )
                    active_state = outcome.state
                    active_error_code = (
                        outcome.error_code
                        if outcome.error_code is not None
                        else "MODEL_FIT_FAILED"
                    )

                if active_fit_id is None:
                    raise WindowAlignmentError(
                        "The first eligible GARCH origin was not a scheduled refit."
                    )

                if active_state is None:
                    values = _failed_forecast_values(active_error_code)
                else:
                    try:
                        active_state, forecast = model.advance(
                            active_state,
                            float(series.iloc[window.origin_position]),
                        )
                        values = _garch_forecast_values(forecast)
                    except MarketRiskForecastingError as exc:
                        active_state = None
                        active_error_code = exc.code.value
                        values = _failed_forecast_values(active_error_code)

                forecast_id = make_forecast_id(
                    experiment_id=config.experiment.experiment_id,
                    fit_id=active_fit_id,
                    series_id=series_id,
                    model_id=model.model_id,
                    forecast_origin=window.forecast_origin,
                    target_date=window.target_date,
                )
                forecast_rows.append(
                    {
                        "experiment_id": config.experiment.experiment_id,
                        "forecast_id": forecast_id,
                        "series_id": series_id,
                        "model_id": model.model_id,
                        "model_version": __version__,
                        "fit_id": active_fit_id,
                        "forecast_origin": window.forecast_origin,
                        "target_date": window.target_date,
                        **values,
                        "warning_codes": json.dumps([]),
                    }
                )

    return (
        pd.DataFrame(forecast_rows, columns=FORECAST_COLUMNS),
        pd.DataFrame(diagnostic_rows, columns=FIT_DIAGNOSTIC_COLUMNS),
    )


def _validate_benchmark_artifacts(artifacts: BenchmarkArtifacts) -> None:
    windows = artifacts.experiment_windows
    realizations = artifacts.realizations
    forecasts = artifacts.forecasts
    diagnostics = artifacts.fit_diagnostics
    if tuple(windows.columns) != EXPERIMENT_WINDOW_COLUMNS:
        raise WindowAlignmentError("Experiment-window schema is invalid.")
    if tuple(realizations.columns) != REALIZATION_COLUMNS:
        raise WindowAlignmentError("Realization schema is invalid.")
    if tuple(forecasts.columns) != FORECAST_COLUMNS:
        raise WindowAlignmentError("Forecast schema is invalid.")
    if tuple(diagnostics.columns) != FIT_DIAGNOSTIC_COLUMNS:
        raise WindowAlignmentError("Fit-diagnostic schema is invalid.")
    if not (windows["forecast_origin"] < windows["target_date"]).all():
        raise WindowAlignmentError("A forecast origin does not precede its target.")
    if not (windows["train_end"] == windows["forecast_origin"]).all():
        raise WindowAlignmentError("A training window ends after its origin.")
    if len(windows) != len(forecasts):
        raise WindowAlignmentError("Forecast and window counts do not reconcile.")
    if forecasts["forecast_id"].duplicated().any():
        raise WindowAlignmentError("Forecast identifiers are not unique.")
    if realizations.duplicated(
        subset=["series_id", "forecast_origin", "target_date"]
    ).any():
        raise WindowAlignmentError("Realization keys are not unique.")
    realization_keys = set(
        realizations.loc[:, ["series_id", "forecast_origin", "target_date"]].itertuples(
            index=False, name=None
        )
    )
    forecast_keys = set(
        forecasts.loc[:, ["series_id", "forecast_origin", "target_date"]].itertuples(
            index=False, name=None
        )
    )
    if not forecast_keys.issubset(realization_keys):
        raise WindowAlignmentError("A forecast has no matching realization.")
    if diagnostics["fit_id"].duplicated().any():
        raise WindowAlignmentError("Fit diagnostic identifiers are not unique.")
    candidate_model_ids = {
        EWMA_MODEL_ID,
        GAUSSIAN_GARCH_MODEL_ID,
        STUDENT_T_GARCH_MODEL_ID,
    }
    candidate_fit_ids = set(
        forecasts.loc[
            forecasts["model_id"].isin(candidate_model_ids),
            "fit_id",
        ]
    )
    diagnostic_fit_ids = set(diagnostics["fit_id"])
    if candidate_fit_ids != diagnostic_fit_ids:
        raise WindowAlignmentError(
            "Candidate forecast and fit-diagnostic identifiers do not reconcile."
        )


def run_historical_benchmarks(
    *,
    dataset: ResearchDataset,
    config: ForecastConfig,
    upstream_simple_return_checksum: str,
) -> BenchmarkArtifacts:
    """Generate all Phase 2 windows, realizations, and permanent benchmarks."""
    windows = _all_benchmark_windows(dataset, config)
    artifacts = BenchmarkArtifacts(
        experiment_windows=pd.DataFrame(
            [window.to_record() for window in windows],
            columns=EXPERIMENT_WINDOW_COLUMNS,
        ),
        realizations=_realization_frame(dataset, config),
        forecasts=_forecast_frame(
            windows=windows,
            dataset=dataset,
            config=config,
            upstream_simple_return_checksum=upstream_simple_return_checksum,
        ),
        fit_diagnostics=pd.DataFrame(columns=FIT_DIAGNOSTIC_COLUMNS),
    )
    _validate_benchmark_artifacts(artifacts)
    return artifacts


def run_ewma_candidate(
    *,
    dataset: ResearchDataset,
    config: ForecastConfig,
    upstream_simple_return_checksum: str,
) -> BenchmarkArtifacts:
    """Generate the continuous EWMA candidate and initialization diagnostics."""
    windows = _all_ewma_windows(dataset, config)
    forecasts, diagnostics = _ewma_forecast_frame(
        windows=windows,
        dataset=dataset,
        config=config,
        upstream_simple_return_checksum=upstream_simple_return_checksum,
    )
    artifacts = BenchmarkArtifacts(
        experiment_windows=pd.DataFrame(
            [window.to_record() for window in windows],
            columns=EXPERIMENT_WINDOW_COLUMNS,
        ),
        realizations=_realization_frame(dataset, config),
        forecasts=forecasts,
        fit_diagnostics=diagnostics,
    )
    _validate_benchmark_artifacts(artifacts)
    return artifacts


def run_benchmarks_and_ewma(
    *,
    dataset: ResearchDataset,
    config: ForecastConfig,
    upstream_simple_return_checksum: str,
) -> BenchmarkArtifacts:
    """Run the permanent benchmarks and continuous EWMA candidate."""
    benchmarks = run_historical_benchmarks(
        dataset=dataset,
        config=config,
        upstream_simple_return_checksum=upstream_simple_return_checksum,
    )
    ewma = run_ewma_candidate(
        dataset=dataset,
        config=config,
        upstream_simple_return_checksum=upstream_simple_return_checksum,
    )
    artifacts = BenchmarkArtifacts(
        experiment_windows=pd.concat(
            [benchmarks.experiment_windows, ewma.experiment_windows],
            ignore_index=True,
        ),
        realizations=benchmarks.realizations.copy(),
        forecasts=pd.concat(
            [benchmarks.forecasts, ewma.forecasts],
            ignore_index=True,
        ),
        fit_diagnostics=ewma.fit_diagnostics.copy(),
    )
    _validate_benchmark_artifacts(artifacts)
    return artifacts


def run_garch_candidates(
    *,
    dataset: ResearchDataset,
    config: ForecastConfig,
    upstream_simple_return_checksum: str,
) -> BenchmarkArtifacts:
    """Run both scheduled GARCH candidates with typed failure preservation."""
    windows = _all_garch_windows(dataset, config)
    forecasts, diagnostics = _garch_forecast_frames(
        windows=windows,
        dataset=dataset,
        config=config,
        upstream_simple_return_checksum=upstream_simple_return_checksum,
    )
    artifacts = BenchmarkArtifacts(
        experiment_windows=pd.DataFrame(
            [window.to_record() for window in windows],
            columns=EXPERIMENT_WINDOW_COLUMNS,
        ),
        realizations=_realization_frame(dataset, config),
        forecasts=forecasts,
        fit_diagnostics=diagnostics,
    )
    _validate_benchmark_artifacts(artifacts)
    return artifacts


def run_available_models(
    *,
    dataset: ResearchDataset,
    config: ForecastConfig,
    upstream_simple_return_checksum: str,
) -> BenchmarkArtifacts:
    """Run every benchmark and candidate implemented through this boundary."""
    benchmarks_and_ewma = run_benchmarks_and_ewma(
        dataset=dataset,
        config=config,
        upstream_simple_return_checksum=upstream_simple_return_checksum,
    )
    garch = run_garch_candidates(
        dataset=dataset,
        config=config,
        upstream_simple_return_checksum=upstream_simple_return_checksum,
    )
    artifacts = BenchmarkArtifacts(
        experiment_windows=pd.concat(
            [
                benchmarks_and_ewma.experiment_windows,
                garch.experiment_windows,
            ],
            ignore_index=True,
        ),
        realizations=benchmarks_and_ewma.realizations.copy(),
        forecasts=pd.concat(
            [benchmarks_and_ewma.forecasts, garch.forecasts],
            ignore_index=True,
        ),
        fit_diagnostics=pd.concat(
            [
                frame.dropna(axis="columns", how="all")
                for frame in (
                    benchmarks_and_ewma.fit_diagnostics,
                    garch.fit_diagnostics,
                )
            ],
            ignore_index=True,
        ).reindex(columns=FIT_DIAGNOSTIC_COLUMNS),
    )
    _validate_benchmark_artifacts(artifacts)
    return artifacts


def evaluate_model_artifacts(
    *,
    artifacts: BenchmarkArtifacts,
    config: ForecastConfig,
) -> EvaluationArtifacts:
    """Evaluate already-generated forecasts without fitting any model."""
    return evaluate_forecasts(
        forecasts=artifacts.forecasts,
        realizations=artifacts.realizations,
        config=config,
    )


def run_available_models_with_evaluation(
    *,
    dataset: ResearchDataset,
    config: ForecastConfig,
    upstream_simple_return_checksum: str,
) -> EvaluatedModelArtifacts:
    """Run every available model and build the frozen evaluation tables."""
    models = run_available_models(
        dataset=dataset,
        config=config,
        upstream_simple_return_checksum=upstream_simple_return_checksum,
    )
    return EvaluatedModelArtifacts(
        models=models,
        evaluation=evaluate_model_artifacts(
            artifacts=models,
            config=config,
        ),
    )


def persist_benchmark_artifacts(
    artifacts: BenchmarkArtifacts,
    output_dir: Path,
) -> None:
    """Persist the three deterministic Phase 2 numerical artifacts."""
    destination = Path(output_dir)
    targets = {
        "experiment_windows.csv": destination / "experiment_windows.csv",
        "realizations.parquet": destination / "realizations.parquet",
        "forecasts.parquet": destination / "forecasts.parquet",
    }
    collisions = [name for name, path in targets.items() if path.exists()]
    if collisions:
        raise OutputCollisionError(
            f"Refusing to overwrite benchmark artifact(s): {', '.join(collisions)}."
        )
    destination.mkdir(parents=True, exist_ok=True)
    artifacts.experiment_windows.to_csv(
        targets["experiment_windows.csv"],
        index=False,
        date_format="%Y-%m-%d",
    )
    artifacts.realizations.to_parquet(
        targets["realizations.parquet"],
        index=False,
        engine="pyarrow",
    )
    artifacts.forecasts.to_parquet(
        targets["forecasts.parquet"],
        index=False,
        engine="pyarrow",
    )


def persist_available_model_artifacts(
    artifacts: BenchmarkArtifacts,
    output_dir: Path,
) -> None:
    """Persist windows, realizations, forecasts, and fit diagnostics."""
    destination = Path(output_dir)
    targets = {
        "experiment_windows.csv": destination / "experiment_windows.csv",
        "realizations.parquet": destination / "realizations.parquet",
        "forecasts.parquet": destination / "forecasts.parquet",
        "fit_diagnostics.parquet": destination / "fit_diagnostics.parquet",
    }
    collisions = [name for name, path in targets.items() if path.exists()]
    if collisions:
        raise OutputCollisionError(
            f"Refusing to overwrite forecast artifact(s): {', '.join(collisions)}."
        )
    destination.mkdir(parents=True, exist_ok=True)
    artifacts.experiment_windows.to_csv(
        targets["experiment_windows.csv"],
        index=False,
        date_format="%Y-%m-%d",
    )
    artifacts.realizations.to_parquet(
        targets["realizations.parquet"],
        index=False,
        engine="pyarrow",
    )
    artifacts.forecasts.to_parquet(
        targets["forecasts.parquet"],
        index=False,
        engine="pyarrow",
    )
    artifacts.fit_diagnostics.to_parquet(
        targets["fit_diagnostics.parquet"],
        index=False,
        engine="pyarrow",
    )


def persist_evaluated_model_artifacts(
    artifacts: EvaluatedModelArtifacts,
    output_dir: Path,
) -> None:
    """Persist every numerical forecast and evaluation artifact."""
    destination = Path(output_dir)
    names = (
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
    )
    collisions = [name for name in names if (destination / name).exists()]
    if collisions:
        raise OutputCollisionError(
            "Refusing to overwrite numerical artifact(s): "
            + ", ".join(collisions)
            + "."
        )
    persist_available_model_artifacts(artifacts.models, destination)
    persist_evaluation_artifacts(artifacts.evaluation, destination)


__all__ = [
    "BenchmarkArtifacts",
    "EvaluatedModelArtifacts",
    "EXPERIMENT_WINDOW_COLUMNS",
    "FIT_DIAGNOSTIC_COLUMNS",
    "FORECAST_COLUMNS",
    "REALIZATION_COLUMNS",
    "evaluate_model_artifacts",
    "persist_benchmark_artifacts",
    "persist_available_model_artifacts",
    "persist_evaluated_model_artifacts",
    "run_available_models",
    "run_available_models_with_evaluation",
    "run_benchmarks_and_ewma",
    "run_ewma_candidate",
    "run_garch_candidates",
    "run_historical_benchmarks",
]
