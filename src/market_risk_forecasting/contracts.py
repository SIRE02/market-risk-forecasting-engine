"""Typed, side-effect-free forecast contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import pandas as pd

type ModelState = Mapping[str, Any]


@dataclass(frozen=True)
class FitResult:
    """Immutable result of fitting one model to one research series."""

    fit_id: str
    model_id: str
    series_id: str
    fit_origin: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    observation_count: int
    parameters: Mapping[str, float]
    state: ModelState
    converged: bool
    optimizer_status: str
    warning_codes: tuple[str, ...]
    runtime_seconds: float
    scaling_factor: float


@dataclass(frozen=True)
class ForecastResult:
    """Immutable public one-session-ahead forecast record."""

    model_id: str
    fit_id: str
    forecast_origin: pd.Timestamp
    target_date: pd.Timestamp
    variance: float | None
    volatility: float | None
    return_quantile_0_05: float | None
    var_0_95: float | None
    return_quantile_0_01: float | None
    var_0_99: float | None
    status: Literal["ok", "failed", "unavailable"]
    error_code: str | None
    warning_codes: tuple[str, ...]


class VarianceForecaster(Protocol):
    """Public interface shared by all variance forecasters."""

    model_id: str

    def fit(
        self,
        returns: pd.Series,
        *,
        fit_origin: pd.Timestamp,
    ) -> FitResult:
        """Fit at an origin using only the supplied history."""
        ...

    def forecast_one(
        self,
        state: ModelState,
        *,
        forecast_origin: pd.Timestamp,
        latest_return: float,
    ) -> ForecastResult:
        """Produce one forecast after observing ``latest_return``."""
        ...


__all__ = [
    "FitResult",
    "ForecastResult",
    "ModelState",
    "VarianceForecaster",
]
