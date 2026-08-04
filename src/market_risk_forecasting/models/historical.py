"""Transparent historical benchmark models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from market_risk_forecasting.errors import (
    InputValueInvalidError,
    InsufficientHistoryError,
    NonfiniteVarianceError,
    NonpositiveVarianceError,
)

HISTORICAL_VARIANCE_MODEL_ID = "historical_variance"
HISTORICAL_SIMULATION_MODEL_ID = "historical_simulation"

type QuantileMethod = Literal["linear", "lower", "higher", "nearest", "midpoint"]


@dataclass(frozen=True)
class VarianceForecast:
    variance: float
    volatility: float


@dataclass(frozen=True)
class HistoricalSimulationForecast:
    return_quantile_0_05: float
    var_0_95: float
    return_quantile_0_01: float
    var_0_99: float


def _exact_finite_window(returns: pd.Series, expected: int) -> pd.Series:
    if len(returns) != expected:
        raise InsufficientHistoryError(
            f"Expected exactly {expected} returns; received {len(returns)}."
        )
    values = returns.to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all():
        raise InputValueInvalidError("Forecast window contains invalid returns.")
    return pd.Series(values, index=returns.index, name=returns.name, dtype=float)


@dataclass(frozen=True)
class HistoricalVarianceModel:
    """Rolling sample variance using the configured window and ``ddof=1``."""

    window: int = 252
    model_id: str = HISTORICAL_VARIANCE_MODEL_ID

    def __post_init__(self) -> None:
        if self.window < 2:
            raise InputValueInvalidError(
                "Historical variance window must be at least two."
            )

    def forecast(self, returns: pd.Series) -> VarianceForecast:
        window = _exact_finite_window(returns, self.window)
        variance = float(window.var(ddof=1))
        if not math.isfinite(variance):
            raise NonfiniteVarianceError("Historical sample variance is non-finite.")
        if variance <= 0.0:
            raise NonpositiveVarianceError(
                "Historical sample variance is not strictly positive."
            )
        return VarianceForecast(
            variance=variance,
            volatility=math.sqrt(variance),
        )

    def forecast_path(self, returns: pd.Series) -> pd.DataFrame:
        """Vectorize the same trailing estimator over every eligible origin."""
        values = returns.to_numpy(dtype=float, copy=True)
        if not np.isfinite(values).all():
            raise InputValueInvalidError("Forecast path contains invalid returns.")
        variance = (
            returns.astype(float)
            .rolling(self.window, min_periods=self.window)
            .var(ddof=1)
        )
        return pd.DataFrame(
            {
                "variance": variance,
                "volatility": variance.pow(0.5),
            },
            index=returns.index,
        )


@dataclass(frozen=True)
class HistoricalSimulationModel:
    """Historical lower-tail quantiles using the configured rolling window."""

    window: int = 500
    quantile_method: QuantileMethod = "linear"
    model_id: str = HISTORICAL_SIMULATION_MODEL_ID

    def __post_init__(self) -> None:
        if self.window < 2:
            raise InputValueInvalidError(
                "Historical simulation window must be at least two."
            )

    def forecast(self, returns: pd.Series) -> HistoricalSimulationForecast:
        window = _exact_finite_window(returns, self.window)
        q_0_05 = float(window.quantile(0.05, interpolation=self.quantile_method))
        q_0_01 = float(window.quantile(0.01, interpolation=self.quantile_method))
        if not math.isfinite(q_0_05) or not math.isfinite(q_0_01):
            raise InputValueInvalidError(
                "Historical-simulation quantiles are non-finite."
            )
        return HistoricalSimulationForecast(
            return_quantile_0_05=q_0_05,
            var_0_95=max(0.0, -q_0_05),
            return_quantile_0_01=q_0_01,
            var_0_99=max(0.0, -q_0_01),
        )

    def forecast_path(self, returns: pd.Series) -> pd.DataFrame:
        """Vectorize exact trailing linear quantiles over eligible origins."""
        values = returns.to_numpy(dtype=float, copy=True)
        if not np.isfinite(values).all():
            raise InputValueInvalidError("Forecast path contains invalid returns.")
        rolling = returns.astype(float).rolling(
            self.window,
            min_periods=self.window,
        )
        q_0_05 = rolling.quantile(0.05, interpolation=self.quantile_method)
        q_0_01 = rolling.quantile(0.01, interpolation=self.quantile_method)
        return pd.DataFrame(
            {
                "return_quantile_0_05": q_0_05,
                "var_0_95": (-q_0_05).clip(lower=0.0),
                "return_quantile_0_01": q_0_01,
                "var_0_99": (-q_0_01).clip(lower=0.0),
            },
            index=returns.index,
        )


__all__ = [
    "HISTORICAL_SIMULATION_MODEL_ID",
    "HISTORICAL_VARIANCE_MODEL_ID",
    "HistoricalSimulationForecast",
    "HistoricalSimulationModel",
    "HistoricalVarianceModel",
    "QuantileMethod",
    "VarianceForecast",
]
