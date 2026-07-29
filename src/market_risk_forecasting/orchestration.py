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
from market_risk_forecasting.identifiers import make_fit_id, make_forecast_id
from market_risk_forecasting.models.historical import (
    HISTORICAL_SIMULATION_MODEL_ID,
    HISTORICAL_VARIANCE_MODEL_ID,
    HistoricalSimulationModel,
    HistoricalVarianceModel,
)
from market_risk_forecasting.windows import (
    ForecastWindow,
    classify_target_date,
    iter_forecast_windows,
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


@dataclass(frozen=True)
class BenchmarkArtifacts:
    """Phase 2 in-memory artifacts with stable public schemas."""

    experiment_windows: pd.DataFrame
    realizations: pd.DataFrame
    forecasts: pd.DataFrame


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


def _validate_benchmark_artifacts(artifacts: BenchmarkArtifacts) -> None:
    windows = artifacts.experiment_windows
    realizations = artifacts.realizations
    forecasts = artifacts.forecasts
    if tuple(windows.columns) != EXPERIMENT_WINDOW_COLUMNS:
        raise WindowAlignmentError("Experiment-window schema is invalid.")
    if tuple(realizations.columns) != REALIZATION_COLUMNS:
        raise WindowAlignmentError("Realization schema is invalid.")
    if tuple(forecasts.columns) != FORECAST_COLUMNS:
        raise WindowAlignmentError("Forecast schema is invalid.")
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
    )
    _validate_benchmark_artifacts(artifacts)
    return artifacts


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


__all__ = [
    "BenchmarkArtifacts",
    "EXPERIMENT_WINDOW_COLUMNS",
    "FORECAST_COLUMNS",
    "REALIZATION_COLUMNS",
    "persist_benchmark_artifacts",
    "run_historical_benchmarks",
]
