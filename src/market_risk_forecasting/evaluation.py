"""Deterministic forecast scoring, coverage testing, and paired inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import chi2

from market_risk_forecasting.config import ForecastConfig
from market_risk_forecasting.errors import OutputCollisionError, WindowAlignmentError
from market_risk_forecasting.models.ewma import EWMA_MODEL_ID
from market_risk_forecasting.models.garch import (
    GAUSSIAN_GARCH_MODEL_ID,
    STUDENT_T_GARCH_MODEL_ID,
)
from market_risk_forecasting.models.historical import (
    HISTORICAL_SIMULATION_MODEL_ID,
    HISTORICAL_VARIANCE_MODEL_ID,
)

EVALUATION_PERIODS = ("validation", "test")
ALL_SERIES_ID = "ALL"

VARIANCE_MODEL_IDS = (
    HISTORICAL_VARIANCE_MODEL_ID,
    EWMA_MODEL_ID,
    GAUSSIAN_GARCH_MODEL_ID,
    STUDENT_T_GARCH_MODEL_ID,
)
QUANTILE_MODEL_IDS = (
    HISTORICAL_SIMULATION_MODEL_ID,
    EWMA_MODEL_ID,
    GAUSSIAN_GARCH_MODEL_ID,
    STUDENT_T_GARCH_MODEL_ID,
)
CANDIDATE_MODEL_IDS = (
    EWMA_MODEL_ID,
    GAUSSIAN_GARCH_MODEL_ID,
    STUDENT_T_GARCH_MODEL_ID,
)

RESULT_COLUMNS = (
    "experiment_id",
    "period",
    "series_id",
    "model_id",
    "metric",
    "observation_count",
    "value",
)
FORECAST_AVAILABILITY_COLUMNS = RESULT_COLUMNS
VARIANCE_SCORE_COLUMNS = RESULT_COLUMNS
QUANTILE_SCORE_COLUMNS = (
    *RESULT_COLUMNS,
    "confidence_level",
    "tail_probability",
)
COVERAGE_TEST_COLUMNS = (
    *RESULT_COLUMNS,
    "confidence_level",
    "tail_probability",
    "status",
)
BOOTSTRAP_COMPARISON_COLUMNS = (
    *RESULT_COLUMNS,
    "benchmark_model_id",
    "paired_count",
    "effect_direction",
    "interval_lower",
    "interval_upper",
    "median_difference",
    "fraction_below_zero",
    "bootstrap_resamples",
    "block_length",
    "random_seed",
    "confidence_level",
)
PERIOD_BREAKDOWN_COLUMNS = (
    *RESULT_COLUMNS,
    "period_start",
    "period_end",
)

_KEY_COLUMNS = ("series_id", "forecast_origin", "target_date")
_TAIL_SPECS = (
    (0.95, 0.05, "return_quantile_0_05", "var_0_95", "pinball_loss_0_05"),
    (0.99, 0.01, "return_quantile_0_01", "var_0_99", "pinball_loss_0_01"),
)


@dataclass(frozen=True)
class EvaluationArtifacts:
    """All deterministic numerical result tables delivered by this phase."""

    forecast_availability: pd.DataFrame
    variance_scores: pd.DataFrame
    quantile_scores: pd.DataFrame
    coverage_tests: pd.DataFrame
    bootstrap_comparisons: pd.DataFrame
    period_breakdowns: pd.DataFrame


def variance_score_values(
    variance: float,
    realized_return: float,
) -> dict[str, float]:
    """Return QLIKE, squared error, and absolute error for one observation."""
    if not math.isfinite(variance) or variance <= 0.0:
        raise WindowAlignmentError(
            "Variance scoring requires a finite, strictly positive forecast."
        )
    if not math.isfinite(realized_return):
        raise WindowAlignmentError("Variance scoring requires a finite realization.")
    realized_variance = realized_return**2
    error = realized_variance - variance
    return {
        "qlike": math.log(variance) + realized_variance / variance,
        "squared_error": error**2,
        "absolute_error": abs(error),
    }


def pinball_loss(
    *,
    realized_return: float,
    return_quantile: float,
    tail_probability: float,
) -> float:
    """Calculate lower-tail pinball loss without altering the quantile."""
    if not all(math.isfinite(value) for value in (realized_return, return_quantile)):
        raise WindowAlignmentError(
            "Quantile scoring requires finite forecasts and realizations."
        )
    if not 0.0 < tail_probability < 1.0:
        raise WindowAlignmentError("Tail probability must lie strictly in (0, 1).")
    residual = realized_return - return_quantile
    return (tail_probability - float(residual < 0.0)) * residual


def exception_indicator(*, realized_loss: float, value_at_risk: float) -> bool:
    """Return the strict positive-loss VaR exception indicator."""
    if not all(math.isfinite(value) for value in (realized_loss, value_at_risk)):
        raise WindowAlignmentError(
            "Exception testing requires finite loss and VaR values."
        )
    if value_at_risk < 0.0:
        raise WindowAlignmentError("Reported positive-loss VaR cannot be negative.")
    return realized_loss > value_at_risk


def transition_counts(
    exceptions: np.ndarray[Any, np.dtype[np.bool_]],
) -> dict[str, int]:
    """Count consecutive binary transitions for Christoffersen testing."""
    values = np.asarray(exceptions, dtype=bool)
    if values.ndim != 1:
        raise WindowAlignmentError("Exception indicators must be one-dimensional.")
    if values.size < 2:
        return {"n00": 0, "n01": 0, "n10": 0, "n11": 0}
    previous = values[:-1]
    current = values[1:]
    return {
        "n00": int((~previous & ~current).sum()),
        "n01": int((~previous & current).sum()),
        "n10": int((previous & ~current).sum()),
        "n11": int((previous & current).sum()),
    }


def longest_exception_cluster(
    exceptions: np.ndarray[Any, np.dtype[np.bool_]],
) -> int:
    """Return the longest consecutive run of strict VaR exceptions."""
    longest = 0
    current = 0
    for observed in np.asarray(exceptions, dtype=bool):
        if observed:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def coverage_statistics(
    exceptions: np.ndarray[Any, np.dtype[np.bool_]],
    *,
    expected_probability: float,
) -> dict[str, float | int | str | None]:
    """Compute Kupiec and Christoffersen statistics for one ordered sequence."""
    values = np.asarray(exceptions, dtype=bool)
    if values.ndim != 1 or values.size == 0:
        raise WindowAlignmentError(
            "Coverage testing requires a non-empty one-dimensional sequence."
        )
    if not 0.0 < expected_probability < 1.0:
        raise WindowAlignmentError(
            "Expected exception probability must lie strictly in (0, 1)."
        )
    observation_count = int(values.size)
    exception_count = int(values.sum())
    exception_rate = exception_count / observation_count
    null_log_likelihood = _bernoulli_log_likelihood(
        successes=exception_count,
        trials=observation_count,
        probability=expected_probability,
    )
    alternative_log_likelihood = _bernoulli_log_likelihood(
        successes=exception_count,
        trials=observation_count,
        probability=exception_rate,
    )
    kupiec_lr = max(
        0.0,
        -2.0 * (null_log_likelihood - alternative_log_likelihood),
    )
    kupiec_p_value = float(chi2.sf(kupiec_lr, df=1))

    counts = transition_counts(values)
    result: dict[str, float | int | str | None] = {
        "expected_exception_probability": expected_probability,
        "observation_count": observation_count,
        "exception_count": exception_count,
        "exception_rate": exception_rate,
        "longest_exception_cluster": longest_exception_cluster(values),
        **counts,
        "kupiec_lr": kupiec_lr,
        "kupiec_p_value": kupiec_p_value,
        "christoffersen_independence_lr": None,
        "christoffersen_independence_p_value": None,
        "conditional_coverage_lr": None,
        "conditional_coverage_p_value": None,
        "christoffersen_status": "insufficient_events",
    }
    if any(counts[name] == 0 for name in ("n00", "n01", "n10", "n11")):
        return result

    n00 = counts["n00"]
    n01 = counts["n01"]
    n10 = counts["n10"]
    n11 = counts["n11"]
    probability_01 = n01 / (n00 + n01)
    probability_11 = n11 / (n10 + n11)
    unconditional_probability = (n01 + n11) / (n00 + n01 + n10 + n11)
    independent_log_likelihood = _bernoulli_log_likelihood(
        successes=n01 + n11,
        trials=n00 + n01 + n10 + n11,
        probability=unconditional_probability,
    )
    markov_log_likelihood = _bernoulli_log_likelihood(
        successes=n01,
        trials=n00 + n01,
        probability=probability_01,
    ) + _bernoulli_log_likelihood(
        successes=n11,
        trials=n10 + n11,
        probability=probability_11,
    )
    independence_lr = max(
        0.0,
        -2.0 * (independent_log_likelihood - markov_log_likelihood),
    )
    conditional_lr = kupiec_lr + independence_lr
    result.update(
        {
            "christoffersen_independence_lr": independence_lr,
            "christoffersen_independence_p_value": float(
                chi2.sf(independence_lr, df=1)
            ),
            "conditional_coverage_lr": conditional_lr,
            "conditional_coverage_p_value": float(chi2.sf(conditional_lr, df=2)),
            "christoffersen_status": "ok",
        }
    )
    return result


def moving_block_bootstrap(
    groups: tuple[np.ndarray[Any, np.dtype[np.float64]], ...],
    *,
    block_length: int,
    resamples: int,
    random_seed: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Bootstrap the pooled mean while keeping blocks within each series."""
    if not groups or any(values.size == 0 for values in groups):
        raise WindowAlignmentError(
            "Moving-block bootstrap requires non-empty series groups."
        )
    if block_length < 1 or resamples < 1:
        raise WindowAlignmentError(
            "Moving-block bootstrap controls must be positive integers."
        )
    random = np.random.default_rng(random_seed)
    pooled_sums = np.zeros(resamples, dtype=float)
    pooled_count = 0
    for raw_values in groups:
        values = np.asarray(raw_values, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise WindowAlignmentError(
                "Moving-block bootstrap inputs must be finite vectors."
            )
        effective_block = min(block_length, values.size)
        block_count = math.ceil(values.size / effective_block)
        starts = random.integers(
            0,
            values.size - effective_block + 1,
            size=(resamples, block_count),
        )
        offsets = np.arange(effective_block)
        indices = starts[:, :, None] + offsets[None, None, :]
        sampled = values[indices].reshape(resamples, -1)[:, : values.size]
        pooled_sums += sampled.sum(axis=1)
        pooled_count += values.size
    return pooled_sums / pooled_count


def evaluate_forecasts(
    *,
    forecasts: pd.DataFrame,
    realizations: pd.DataFrame,
    config: ForecastConfig,
) -> EvaluationArtifacts:
    """Build all evaluation and inference artifacts from saved numerical records."""
    joined, experiment_id = _join_forecasts_and_realizations(
        forecasts,
        realizations,
    )
    availability = _availability_table(joined, experiment_id)
    variance_observations = _variance_observations(joined)
    quantile_observations = _quantile_observations(joined)
    variance_scores = _score_table(
        variance_observations,
        experiment_id=experiment_id,
        output_columns=VARIANCE_SCORE_COLUMNS,
    )
    quantile_scores = _score_table(
        quantile_observations,
        experiment_id=experiment_id,
        output_columns=QUANTILE_SCORE_COLUMNS,
        extra_columns=("confidence_level", "tail_probability"),
    )
    coverage_tests = _coverage_table(
        quantile_observations,
        experiment_id=experiment_id,
    )
    bootstrap_comparisons = _bootstrap_table(
        variance_observations=variance_observations,
        quantile_observations=quantile_observations,
        experiment_id=experiment_id,
        config=config,
    )
    test_years = tuple(
        range(config.periods.test_start.year, config.periods.test_end.year + 1)
    )
    period_breakdowns = _period_breakdown_table(
        joined=joined,
        variance_observations=variance_observations,
        quantile_observations=quantile_observations,
        experiment_id=experiment_id,
        test_years=test_years,
    )
    artifacts = EvaluationArtifacts(
        forecast_availability=availability,
        variance_scores=variance_scores,
        quantile_scores=quantile_scores,
        coverage_tests=coverage_tests,
        bootstrap_comparisons=bootstrap_comparisons,
        period_breakdowns=period_breakdowns,
    )
    _validate_evaluation_artifacts(artifacts, test_years=test_years)
    return artifacts


def persist_evaluation_artifacts(
    artifacts: EvaluationArtifacts,
    output_dir: Path,
) -> None:
    """Write all deterministic evaluation tables without overwriting."""
    destination = Path(output_dir)
    tables = {
        "forecast_availability.csv": artifacts.forecast_availability,
        "variance_scores.csv": artifacts.variance_scores,
        "quantile_scores.csv": artifacts.quantile_scores,
        "coverage_tests.csv": artifacts.coverage_tests,
        "bootstrap_comparisons.csv": artifacts.bootstrap_comparisons,
        "period_breakdowns.csv": artifacts.period_breakdowns,
    }
    collisions = [name for name in tables if (destination / name).exists()]
    if collisions:
        raise OutputCollisionError(
            f"Refusing to overwrite evaluation artifact(s): {', '.join(collisions)}."
        )
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(destination / name, index=False, date_format="%Y-%m-%d")


def _join_forecasts_and_realizations(
    forecasts: pd.DataFrame,
    realizations: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    missing_forecast = sorted(
        {
            "experiment_id",
            "model_id",
            "status",
            "variance",
            "return_quantile_0_05",
            "var_0_95",
            "return_quantile_0_01",
            "var_0_99",
            *_KEY_COLUMNS,
        }
        - set(forecasts.columns)
    )
    missing_realization = sorted(
        {"simple_return", "squared_return", "loss", "period", *_KEY_COLUMNS}
        - set(realizations.columns)
    )
    if missing_forecast or missing_realization:
        raise WindowAlignmentError(
            "Evaluation inputs are missing required columns: "
            + ", ".join(missing_forecast + missing_realization)
            + "."
        )
    if realizations.duplicated(subset=list(_KEY_COLUMNS)).any():
        raise WindowAlignmentError("Evaluation realization keys are not unique.")
    experiment_ids = forecasts["experiment_id"].drop_duplicates().tolist()
    if len(experiment_ids) != 1 or not isinstance(experiment_ids[0], str):
        raise WindowAlignmentError(
            "Evaluation requires exactly one string experiment identifier."
        )
    joined = forecasts.merge(
        realizations,
        on=list(_KEY_COLUMNS),
        how="left",
        validate="many_to_one",
    )
    if joined["period"].isna().any():
        raise WindowAlignmentError("A forecast has no matching realization.")
    joined = joined.loc[joined["period"].isin(EVALUATION_PERIODS)].copy()
    return joined, experiment_ids[0]


def _availability_table(joined: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = (
        ("total_eligible_target_dates", None),
        ("valid_forecast_count", "ok"),
        ("failed_forecast_count", "failed"),
        ("unavailable_forecast_count", "unavailable"),
    )
    for period, period_frame in joined.groupby("period", sort=True):
        for (series_id, model_id), group in period_frame.groupby(
            ["series_id", "model_id"],
            sort=True,
        ):
            for metric, status in metrics:
                value = (
                    len(group)
                    if status is None
                    else int(group["status"].eq(status).sum())
                )
                rows.append(
                    _result_record(
                        experiment_id=experiment_id,
                        period=str(period),
                        series_id=str(series_id),
                        model_id=str(model_id),
                        metric=metric,
                        observation_count=len(group),
                        value=float(value),
                    )
                )
        for model_id, group in period_frame.groupby("model_id", sort=True):
            for metric, status in metrics:
                value = (
                    len(group)
                    if status is None
                    else int(group["status"].eq(status).sum())
                )
                rows.append(
                    _result_record(
                        experiment_id=experiment_id,
                        period=str(period),
                        series_id=ALL_SERIES_ID,
                        model_id=str(model_id),
                        metric=metric,
                        observation_count=len(group),
                        value=float(value),
                    )
                )
    return _frame(rows, FORECAST_AVAILABILITY_COLUMNS)


def _variance_observations(joined: pd.DataFrame) -> pd.DataFrame:
    valid = joined.loc[
        joined["model_id"].isin(VARIANCE_MODEL_IDS) & joined["status"].eq("ok")
    ].copy()
    rows: list[dict[str, Any]] = []
    for record in valid.itertuples(index=False):
        scores = variance_score_values(
            float(cast(Any, record.variance)),
            float(cast(Any, record.simple_return)),
        )
        for metric, value in scores.items():
            rows.append(
                {
                    "period": record.period,
                    "series_id": record.series_id,
                    "model_id": record.model_id,
                    "forecast_origin": record.forecast_origin,
                    "target_date": record.target_date,
                    "metric": metric,
                    "score": value,
                }
            )
    return pd.DataFrame(
        rows,
        columns=(
            "period",
            "series_id",
            "model_id",
            "forecast_origin",
            "target_date",
            "metric",
            "score",
        ),
    )


def _quantile_observations(joined: pd.DataFrame) -> pd.DataFrame:
    valid = joined.loc[
        joined["model_id"].isin(QUANTILE_MODEL_IDS) & joined["status"].eq("ok")
    ].copy()
    rows: list[dict[str, Any]] = []
    for record in valid.itertuples(index=False):
        for confidence, tail, quantile_column, var_column, metric in _TAIL_SPECS:
            quantile = float(getattr(record, quantile_column))
            value_at_risk = float(getattr(record, var_column))
            rows.append(
                {
                    "period": record.period,
                    "series_id": record.series_id,
                    "model_id": record.model_id,
                    "forecast_origin": record.forecast_origin,
                    "target_date": record.target_date,
                    "metric": metric,
                    "score": pinball_loss(
                        realized_return=float(cast(Any, record.simple_return)),
                        return_quantile=quantile,
                        tail_probability=tail,
                    ),
                    "confidence_level": confidence,
                    "tail_probability": tail,
                    "exception": exception_indicator(
                        realized_loss=float(cast(Any, record.loss)),
                        value_at_risk=value_at_risk,
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=(
            "period",
            "series_id",
            "model_id",
            "forecast_origin",
            "target_date",
            "metric",
            "score",
            "confidence_level",
            "tail_probability",
            "exception",
        ),
    )


def _score_table(
    observations: pd.DataFrame,
    *,
    experiment_id: str,
    output_columns: tuple[str, ...],
    extra_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["period", "series_id", "model_id", "metric", *extra_columns]
    for key_values, group in observations.groupby(keys, sort=True):
        values = key_values if isinstance(key_values, tuple) else (key_values,)
        key = dict(zip(keys, values, strict=True))
        record = _result_record(
            experiment_id=experiment_id,
            period=str(key["period"]),
            series_id=str(key["series_id"]),
            model_id=str(key["model_id"]),
            metric=str(key["metric"]),
            observation_count=len(group),
            value=float(group["score"].mean()),
        )
        record.update({column: key[column] for column in extra_columns})
        rows.append(record)
    aggregate_keys = ["period", "model_id", "metric", *extra_columns]
    for key_values, group in observations.groupby(aggregate_keys, sort=True):
        values = key_values if isinstance(key_values, tuple) else (key_values,)
        key = dict(zip(aggregate_keys, values, strict=True))
        record = _result_record(
            experiment_id=experiment_id,
            period=str(key["period"]),
            series_id=ALL_SERIES_ID,
            model_id=str(key["model_id"]),
            metric=str(key["metric"]),
            observation_count=len(group),
            value=float(group["score"].mean()),
        )
        record.update({column: key[column] for column in extra_columns})
        rows.append(record)
    return _frame(rows, output_columns)


def _coverage_table(
    observations: pd.DataFrame,
    *,
    experiment_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = (
        "period",
        "series_id",
        "model_id",
        "confidence_level",
        "tail_probability",
    )
    for key_values, group in observations.groupby(list(keys), sort=True):
        key = dict(zip(keys, key_values, strict=True))
        ordered = group.sort_values("target_date", kind="stable")
        statistics = coverage_statistics(
            ordered["exception"].to_numpy(dtype=bool),
            expected_probability=float(cast(Any, key["tail_probability"])),
        )
        observation_count = int(cast(Any, statistics["observation_count"]))
        christoffersen_status = str(statistics["christoffersen_status"])
        for metric, value in statistics.items():
            if metric in {"observation_count", "christoffersen_status"}:
                continue
            status = (
                christoffersen_status
                if metric.startswith("christoffersen")
                or metric.startswith("conditional_coverage")
                else "ok"
            )
            rows.append(
                {
                    **_result_record(
                        experiment_id=experiment_id,
                        period=str(key["period"]),
                        series_id=str(key["series_id"]),
                        model_id=str(key["model_id"]),
                        metric=metric,
                        observation_count=observation_count,
                        value=float(value) if value is not None else None,
                    ),
                    "confidence_level": float(cast(Any, key["confidence_level"])),
                    "tail_probability": float(cast(Any, key["tail_probability"])),
                    "status": status,
                }
            )
    return _frame(rows, COVERAGE_TEST_COLUMNS)


def _bootstrap_table(
    *,
    variance_observations: pd.DataFrame,
    quantile_observations: pd.DataFrame,
    experiment_id: str,
    config: ForecastConfig,
) -> pd.DataFrame:
    specifications = (
        (
            variance_observations,
            HISTORICAL_VARIANCE_MODEL_ID,
            ("qlike", "squared_error", "absolute_error"),
        ),
        (
            quantile_observations,
            HISTORICAL_SIMULATION_MODEL_ID,
            ("pinball_loss_0_05", "pinball_loss_0_01"),
        ),
    )
    series_ids = sorted(
        set(variance_observations["series_id"])
        | set(quantile_observations["series_id"])
    )
    rows: list[dict[str, Any]] = []
    for observations, benchmark_model_id, metrics in specifications:
        for period in EVALUATION_PERIODS:
            for candidate_model_id in CANDIDATE_MODEL_IDS:
                for metric in metrics:
                    paired = _paired_differences(
                        observations,
                        period=period,
                        candidate_model_id=candidate_model_id,
                        benchmark_model_id=benchmark_model_id,
                        metric=metric,
                    )
                    for series_id in (*series_ids, ALL_SERIES_ID):
                        selected = (
                            paired
                            if series_id == ALL_SERIES_ID
                            else paired.loc[paired["series_id"] == series_id]
                        )
                        rows.append(
                            _bootstrap_record(
                                selected,
                                experiment_id=experiment_id,
                                period=period,
                                series_id=series_id,
                                candidate_model_id=candidate_model_id,
                                benchmark_model_id=benchmark_model_id,
                                metric=metric,
                                config=config,
                            )
                        )
    return _frame(rows, BOOTSTRAP_COMPARISON_COLUMNS)


def _paired_differences(
    observations: pd.DataFrame,
    *,
    period: str,
    candidate_model_id: str,
    benchmark_model_id: str,
    metric: str,
) -> pd.DataFrame:
    selected = observations.loc[
        observations["period"].eq(period) & observations["metric"].eq(metric)
    ]
    candidate = selected.loc[
        selected["model_id"].eq(candidate_model_id),
        [*_KEY_COLUMNS, "score"],
    ].rename(columns={"score": "candidate_score"})
    benchmark = selected.loc[
        selected["model_id"].eq(benchmark_model_id),
        [*_KEY_COLUMNS, "score"],
    ].rename(columns={"score": "benchmark_score"})
    paired = candidate.merge(
        benchmark,
        on=list(_KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    paired["difference"] = paired["candidate_score"] - paired["benchmark_score"]
    return paired.sort_values(
        ["series_id", "target_date"],
        kind="stable",
    ).reset_index(drop=True)


def _bootstrap_record(
    paired: pd.DataFrame,
    *,
    experiment_id: str,
    period: str,
    series_id: str,
    candidate_model_id: str,
    benchmark_model_id: str,
    metric: str,
    config: ForecastConfig,
) -> dict[str, Any]:
    paired_count = len(paired)
    common = {
        **_result_record(
            experiment_id=experiment_id,
            period=period,
            series_id=series_id,
            model_id=candidate_model_id,
            metric=metric,
            observation_count=paired_count,
            value=None,
        ),
        "benchmark_model_id": benchmark_model_id,
        "paired_count": paired_count,
        "effect_direction": "unavailable",
        "interval_lower": None,
        "interval_upper": None,
        "median_difference": None,
        "fraction_below_zero": None,
        "bootstrap_resamples": config.evaluation.bootstrap_resamples,
        "block_length": config.evaluation.bootstrap_block_length,
        "random_seed": config.experiment.random_seed,
        "confidence_level": config.evaluation.bootstrap_confidence,
    }
    if paired_count == 0:
        return common
    groups = tuple(
        group["difference"].to_numpy(dtype=float)
        for _, group in paired.groupby("series_id", sort=True)
    )
    means = moving_block_bootstrap(
        groups,
        block_length=config.evaluation.bootstrap_block_length,
        resamples=config.evaluation.bootstrap_resamples,
        random_seed=config.experiment.random_seed,
    )
    difference = paired["difference"].to_numpy(dtype=float)
    mean_difference = float(difference.mean())
    tail = (1.0 - config.evaluation.bootstrap_confidence) / 2.0
    common.update(
        {
            "value": mean_difference,
            "effect_direction": (
                "candidate_better"
                if mean_difference < 0.0
                else "benchmark_better"
                if mean_difference > 0.0
                else "tie"
            ),
            "interval_lower": float(np.quantile(means, tail)),
            "interval_upper": float(np.quantile(means, 1.0 - tail)),
            "median_difference": float(np.median(difference)),
            "fraction_below_zero": float(np.mean(means < 0.0)),
        }
    )
    return common


def _period_breakdown_table(
    *,
    joined: pd.DataFrame,
    variance_observations: pd.DataFrame,
    quantile_observations: pd.DataFrame,
    experiment_id: str,
    test_years: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in test_years:
        period_name = f"test_{year}"
        period_start = f"{year}-01-01"
        period_end = f"{year}-12-31"
        annual_joined = joined.loc[
            joined["period"].eq("test") & _target_year(joined).eq(year)
        ]
        for (series_id, model_id), group in annual_joined.groupby(
            ["series_id", "model_id"],
            sort=True,
        ):
            for metric, status in (
                ("valid_forecast_count", "ok"),
                ("failed_forecast_count", "failed"),
                ("unavailable_forecast_count", "unavailable"),
            ):
                rows.append(
                    _period_record(
                        experiment_id=experiment_id,
                        period=period_name,
                        series_id=str(series_id),
                        model_id=str(model_id),
                        metric=metric,
                        observation_count=len(group),
                        value=float(group["status"].eq(status).sum()),
                        period_start=period_start,
                        period_end=period_end,
                    )
                )
        for model_id, group in annual_joined.groupby("model_id", sort=True):
            for metric, status in (
                ("valid_forecast_count", "ok"),
                ("failed_forecast_count", "failed"),
                ("unavailable_forecast_count", "unavailable"),
            ):
                rows.append(
                    _period_record(
                        experiment_id=experiment_id,
                        period=period_name,
                        series_id=ALL_SERIES_ID,
                        model_id=str(model_id),
                        metric=metric,
                        observation_count=len(group),
                        value=float(group["status"].eq(status).sum()),
                        period_start=period_start,
                        period_end=period_end,
                    )
                )
        for observations in (variance_observations, quantile_observations):
            annual_scores = observations.loc[
                observations["period"].eq("test") & _target_year(observations).eq(year)
            ]
            for (
                score_series_id,
                score_model_id,
                score_metric,
            ), group in annual_scores.groupby(
                ["series_id", "model_id", "metric"],
                sort=True,
            ):
                rows.append(
                    _period_record(
                        experiment_id=experiment_id,
                        period=period_name,
                        series_id=str(score_series_id),
                        model_id=str(score_model_id),
                        metric=str(score_metric),
                        observation_count=len(group),
                        value=float(group["score"].mean()),
                        period_start=period_start,
                        period_end=period_end,
                    )
                )
            for (
                score_model_id,
                score_metric,
            ), group in annual_scores.groupby(
                ["model_id", "metric"],
                sort=True,
            ):
                rows.append(
                    _period_record(
                        experiment_id=experiment_id,
                        period=period_name,
                        series_id=ALL_SERIES_ID,
                        model_id=str(score_model_id),
                        metric=str(score_metric),
                        observation_count=len(group),
                        value=float(group["score"].mean()),
                        period_start=period_start,
                        period_end=period_end,
                    )
                )
        annual_quantiles = quantile_observations.loc[
            quantile_observations["period"].eq("test")
            & _target_year(quantile_observations).eq(year)
        ]
        for (
            quantile_series_id,
            quantile_model_id,
            quantile_metric,
        ), group in annual_quantiles.groupby(
            ["series_id", "model_id", "metric"],
            sort=True,
        ):
            rows.append(
                _period_record(
                    experiment_id=experiment_id,
                    period=period_name,
                    series_id=str(quantile_series_id),
                    model_id=str(quantile_model_id),
                    metric=str(quantile_metric).replace(
                        "pinball_loss",
                        "exception_rate",
                    ),
                    observation_count=len(group),
                    value=float(group["exception"].mean()),
                    period_start=period_start,
                    period_end=period_end,
                )
            )
        for (
            quantile_model_id,
            quantile_metric,
        ), group in annual_quantiles.groupby(
            ["model_id", "metric"],
            sort=True,
        ):
            rows.append(
                _period_record(
                    experiment_id=experiment_id,
                    period=period_name,
                    series_id=ALL_SERIES_ID,
                    model_id=str(quantile_model_id),
                    metric=str(quantile_metric).replace(
                        "pinball_loss",
                        "exception_rate",
                    ),
                    observation_count=len(group),
                    value=float(group["exception"].mean()),
                    period_start=period_start,
                    period_end=period_end,
                )
            )
    return _frame(rows, PERIOD_BREAKDOWN_COLUMNS)


def _result_record(
    *,
    experiment_id: str,
    period: str,
    series_id: str,
    model_id: str,
    metric: str,
    observation_count: int,
    value: float | None,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "period": period,
        "series_id": series_id,
        "model_id": model_id,
        "metric": metric,
        "observation_count": observation_count,
        "value": value,
    }


def _target_year(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        pd.DatetimeIndex(frame["target_date"]).year,
        index=frame.index,
        dtype=int,
    )


def _period_record(
    *,
    period_start: str,
    period_end: str,
    **values: Any,
) -> dict[str, Any]:
    return {
        **values,
        "period_start": period_start,
        "period_end": period_end,
    }


def _frame(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> pd.DataFrame:
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            [
                column
                for column in (
                    "period",
                    "series_id",
                    "model_id",
                    "metric",
                    "confidence_level",
                    "benchmark_model_id",
                )
                if column in columns
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _bernoulli_log_likelihood(
    *,
    successes: int,
    trials: int,
    probability: float,
) -> float:
    failures = trials - successes
    success_term = 0.0 if successes == 0 else successes * math.log(probability)
    failure_term = 0.0 if failures == 0 else failures * math.log1p(-probability)
    return success_term + failure_term


def _validate_evaluation_artifacts(
    artifacts: EvaluationArtifacts,
    *,
    test_years: tuple[int, ...],
) -> None:
    expected = (
        (artifacts.forecast_availability, FORECAST_AVAILABILITY_COLUMNS),
        (artifacts.variance_scores, VARIANCE_SCORE_COLUMNS),
        (artifacts.quantile_scores, QUANTILE_SCORE_COLUMNS),
        (artifacts.coverage_tests, COVERAGE_TEST_COLUMNS),
        (artifacts.bootstrap_comparisons, BOOTSTRAP_COMPARISON_COLUMNS),
        (artifacts.period_breakdowns, PERIOD_BREAKDOWN_COLUMNS),
    )
    for table, columns in expected:
        if tuple(table.columns) != columns:
            raise WindowAlignmentError("An evaluation artifact schema is invalid.")
    for table, _ in expected[:-1]:
        if not set(table["period"]).issubset(EVALUATION_PERIODS):
            raise WindowAlignmentError(
                "Validation and final-test evaluation outputs are not separated."
            )
    if not set(artifacts.period_breakdowns["period"]).issubset(
        {f"test_{year}" for year in test_years}
    ):
        raise WindowAlignmentError("A period breakdown was not predeclared.")
    availability = artifacts.forecast_availability
    total = availability.loc[availability["metric"].eq("total_eligible_target_dates")]
    component = availability.loc[
        availability["metric"].isin(
            (
                "valid_forecast_count",
                "failed_forecast_count",
                "unavailable_forecast_count",
            )
        )
    ]
    keys = ["experiment_id", "period", "series_id", "model_id"]
    component_sum = component.groupby(keys, as_index=False)["value"].sum()
    reconciled = total.merge(
        component_sum,
        on=keys,
        suffixes=("_total", "_components"),
        validate="one_to_one",
    )
    if not reconciled["value_total"].eq(reconciled["value_components"]).all():
        raise WindowAlignmentError("Forecast availability counts do not reconcile.")

    comparisons = artifacts.bootstrap_comparisons
    if not comparisons["observation_count"].eq(comparisons["paired_count"]).all():
        raise WindowAlignmentError(
            "Bootstrap observation and paired counts do not reconcile."
        )
    score_counts = pd.concat(
        [
            artifacts.variance_scores,
            artifacts.quantile_scores.loc[:, RESULT_COLUMNS],
        ],
        ignore_index=True,
    ).loc[:, [*keys, "metric", "observation_count"]]
    candidate_counts = score_counts.rename(
        columns={
            "model_id": "model_id",
            "observation_count": "candidate_score_count",
        }
    )
    benchmark_counts = score_counts.rename(
        columns={
            "model_id": "benchmark_model_id",
            "observation_count": "benchmark_score_count",
        }
    )
    comparison_keys = ["experiment_id", "period", "series_id", "model_id", "metric"]
    checked = comparisons.merge(
        candidate_counts,
        on=comparison_keys,
        how="left",
        validate="many_to_one",
    ).merge(
        benchmark_counts,
        on=[
            "experiment_id",
            "period",
            "series_id",
            "benchmark_model_id",
            "metric",
        ],
        how="left",
        validate="many_to_one",
    )
    candidate_limit = checked["candidate_score_count"].fillna(0)
    benchmark_limit = checked["benchmark_score_count"].fillna(0)
    if (
        checked["paired_count"].gt(candidate_limit).any()
        or checked["paired_count"].gt(benchmark_limit).any()
    ):
        raise WindowAlignmentError(
            "A pairwise common-date count exceeds an available score count."
        )

    quantile_counts = artifacts.quantile_scores.loc[
        :,
        [
            "experiment_id",
            "period",
            "series_id",
            "model_id",
            "metric",
            "confidence_level",
            "tail_probability",
            "observation_count",
        ],
    ].rename(
        columns={
            "metric": "score_metric",
            "observation_count": "quantile_score_count",
        }
    )
    coverage_source = artifacts.coverage_tests.copy()
    coverage_source["score_metric"] = np.where(
        coverage_source["tail_probability"].eq(0.05),
        "pinball_loss_0_05",
        "pinball_loss_0_01",
    )
    coverage_counts = coverage_source.merge(
        quantile_counts,
        on=[
            "experiment_id",
            "period",
            "series_id",
            "model_id",
            "score_metric",
            "confidence_level",
            "tail_probability",
        ],
        how="left",
        validate="many_to_one",
    )
    if (
        coverage_counts["quantile_score_count"].isna().any()
        or not coverage_counts["observation_count"]
        .eq(coverage_counts["quantile_score_count"])
        .all()
    ):
        raise WindowAlignmentError(
            "Coverage and quantile-score observation counts do not reconcile."
        )


__all__ = [
    "ALL_SERIES_ID",
    "BOOTSTRAP_COMPARISON_COLUMNS",
    "COVERAGE_TEST_COLUMNS",
    "EVALUATION_PERIODS",
    "EvaluationArtifacts",
    "FORECAST_AVAILABILITY_COLUMNS",
    "PERIOD_BREAKDOWN_COLUMNS",
    "QUANTILE_SCORE_COLUMNS",
    "RESULT_COLUMNS",
    "VARIANCE_SCORE_COLUMNS",
    "coverage_statistics",
    "evaluate_forecasts",
    "exception_indicator",
    "longest_exception_cluster",
    "moving_block_bootstrap",
    "persist_evaluation_artifacts",
    "pinball_loss",
    "transition_counts",
    "variance_score_values",
]
