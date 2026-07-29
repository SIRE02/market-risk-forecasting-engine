"""Leakage-resistant forecast origins, targets, and experiment periods."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Literal

import pandas as pd

from market_risk_forecasting.config import PeriodConfig
from market_risk_forecasting.errors import WindowAlignmentError

type ExperimentPeriod = Literal["development", "validation", "test"]


@dataclass(frozen=True)
class ForecastWindow:
    """One exact trailing training window and its next observed target."""

    series_id: str
    model_id: str
    forecast_origin: pd.Timestamp
    target_date: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    train_observation_count: int
    period: ExperimentPeriod
    scheduled_refit: bool
    origin_position: int
    target_position: int

    def to_record(self) -> dict[str, object]:
        """Return the required public experiment-window columns."""
        return {
            "series_id": self.series_id,
            "model_id": self.model_id,
            "forecast_origin": self.forecast_origin,
            "target_date": self.target_date,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "train_observation_count": self.train_observation_count,
            "period": self.period,
            "scheduled_refit": self.scheduled_refit,
        }


def classify_target_date(
    target_date: pd.Timestamp,
    periods: PeriodConfig,
) -> ExperimentPeriod | None:
    """Classify an observation by target date, never by forecast origin."""
    observed = pd.Timestamp(target_date).date()
    if periods.development_start <= observed <= periods.development_end:
        return "development"
    if periods.validation_start <= observed <= periods.validation_end:
        return "validation"
    if periods.test_start <= observed <= periods.test_end:
        return "test"
    return None


def validate_canonical_index(index: pd.DatetimeIndex) -> None:
    """Reject dates that cannot support deterministic one-step windows."""
    if len(index) == 0:
        raise WindowAlignmentError("A non-empty DatetimeIndex is required.")
    if index.has_duplicates:
        raise WindowAlignmentError("Forecast input dates contain duplicates.")
    if not index.is_monotonic_increasing:
        raise WindowAlignmentError("Forecast input dates are not sorted.")


def iter_forecast_windows(
    *,
    index: pd.DatetimeIndex,
    series_id: str,
    model_id: str,
    training_window: int,
    periods: PeriodConfig,
    scheduled_refit: bool = False,
) -> Iterator[ForecastWindow]:
    """Yield exact trailing windows with origin strictly before target."""
    validate_canonical_index(index)
    if training_window < 2:
        raise WindowAlignmentError("Training windows require at least 2 observations.")
    if len(index) <= training_window:
        return

    for origin_position in range(training_window - 1, len(index) - 1):
        target_position = origin_position + 1
        origin = pd.Timestamp(index[origin_position])
        target = pd.Timestamp(index[target_position])
        if origin >= target:
            raise WindowAlignmentError(
                "Every forecast origin must strictly precede its target."
            )
        period = classify_target_date(target, periods)
        if period is None:
            continue
        train_start_position = origin_position - training_window + 1
        yield ForecastWindow(
            series_id=series_id,
            model_id=model_id,
            forecast_origin=origin,
            target_date=target,
            train_start=pd.Timestamp(index[train_start_position]),
            train_end=origin,
            train_observation_count=training_window,
            period=period,
            scheduled_refit=scheduled_refit,
            origin_position=origin_position,
            target_position=target_position,
        )


def iter_expanding_forecast_windows(
    *,
    index: pd.DatetimeIndex,
    series_id: str,
    model_id: str,
    initial_observations: int,
    periods: PeriodConfig,
) -> Iterator[ForecastWindow]:
    """Yield continuous recursive windows initialized once from early history."""
    validate_canonical_index(index)
    if initial_observations < 2:
        raise WindowAlignmentError(
            "Recursive initialization requires at least 2 observations."
        )
    if len(index) <= initial_observations:
        return

    for origin_position in range(initial_observations - 1, len(index) - 1):
        target_position = origin_position + 1
        origin = pd.Timestamp(index[origin_position])
        target = pd.Timestamp(index[target_position])
        if origin >= target:
            raise WindowAlignmentError(
                "Every forecast origin must strictly precede its target."
            )
        period = classify_target_date(target, periods)
        if period is None:
            continue
        yield ForecastWindow(
            series_id=series_id,
            model_id=model_id,
            forecast_origin=origin,
            target_date=target,
            train_start=pd.Timestamp(index[0]),
            train_end=origin,
            train_observation_count=origin_position + 1,
            period=period,
            scheduled_refit=False,
            origin_position=origin_position,
            target_position=target_position,
        )


def iter_refit_forecast_windows(
    *,
    index: pd.DatetimeIndex,
    series_id: str,
    model_id: str,
    training_window: int,
    refit_every_origins: int,
    periods: PeriodConfig,
) -> Iterator[ForecastWindow]:
    """Yield rolling windows with a position-based deterministic refit schedule."""
    if refit_every_origins < 1:
        raise WindowAlignmentError("Refit frequency must be at least one origin.")
    for eligible_position, window in enumerate(
        iter_forecast_windows(
            index=index,
            series_id=series_id,
            model_id=model_id,
            training_window=training_window,
            periods=periods,
        )
    ):
        yield replace(
            window,
            scheduled_refit=eligible_position % refit_every_origins == 0,
        )


__all__ = [
    "ExperimentPeriod",
    "ForecastWindow",
    "classify_target_date",
    "iter_expanding_forecast_windows",
    "iter_forecast_windows",
    "iter_refit_forecast_windows",
    "validate_canonical_index",
]
