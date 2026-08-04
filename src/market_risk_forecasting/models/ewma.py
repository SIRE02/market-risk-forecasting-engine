"""Continuous EWMA variance and Gaussian VaR forecasts."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from market_risk_forecasting.errors import (
    InputValueInvalidError,
    InsufficientHistoryError,
    NonfiniteVarianceError,
    NonpositiveVarianceError,
)

EWMA_MODEL_ID = "ewma"


@dataclass(frozen=True)
class EwmaState:
    """One recursive variance state in decimal-return-squared units."""

    variance: float


@dataclass(frozen=True)
class EwmaForecast:
    variance: float
    volatility: float
    return_quantile_0_05: float
    var_0_95: float
    return_quantile_0_01: float
    var_0_99: float


@dataclass(frozen=True)
class EwmaModel:
    """EWMA with one initialization and a continuous variance recursion."""

    lambda_: float = 0.94
    initialization_window: int = 252
    model_id: str = EWMA_MODEL_ID

    def __post_init__(self) -> None:
        if not math.isfinite(self.lambda_) or not 0.0 < self.lambda_ < 1.0:
            raise InputValueInvalidError("EWMA lambda must lie strictly in (0, 1).")
        if self.initialization_window < 2:
            raise InputValueInvalidError(
                "EWMA initialization_window must be at least two."
            )

    def initialize(self, returns: pd.Series) -> EwmaState:
        """Initialize once with the configured sample-variance window."""
        if len(returns) != self.initialization_window:
            raise InsufficientHistoryError(
                f"EWMA initialization requires exactly "
                f"{self.initialization_window} returns; received {len(returns)}."
            )
        values = returns.to_numpy(dtype=float, copy=True)
        if not np.isfinite(values).all():
            raise InputValueInvalidError(
                "EWMA initialization contains invalid returns."
            )
        variance = float(pd.Series(values).var(ddof=1))
        return self._state(variance)

    def update(self, state: EwmaState, latest_return: float) -> EwmaState:
        """Update after observing one new origin return."""
        if not math.isfinite(latest_return):
            raise InputValueInvalidError("EWMA update return must be finite.")
        variance = (
            self.lambda_ * state.variance + (1.0 - self.lambda_) * latest_return**2
        )
        return self._state(variance)

    def forecast(self, state: EwmaState) -> EwmaForecast:
        """Convert one variance state to public decimal variance and Gaussian VaR."""
        variance = self._state(state.variance).variance
        volatility = math.sqrt(variance)
        q_0_05 = volatility * float(norm.ppf(0.05))
        q_0_01 = volatility * float(norm.ppf(0.01))
        return EwmaForecast(
            variance=variance,
            volatility=volatility,
            return_quantile_0_05=q_0_05,
            var_0_95=max(0.0, -q_0_05),
            return_quantile_0_01=q_0_01,
            var_0_99=max(0.0, -q_0_01),
        )

    def forecast_path(self, returns: pd.Series) -> pd.DataFrame:
        """Run one recursion continuously across every experiment period."""
        values = returns.to_numpy(dtype=float, copy=True)
        if not np.isfinite(values).all():
            raise InputValueInvalidError("EWMA forecast path contains invalid returns.")
        if len(returns) < self.initialization_window:
            raise InsufficientHistoryError(
                f"EWMA path requires at least {self.initialization_window} returns."
            )

        variance = np.full(len(returns), np.nan, dtype=float)
        initial_position = self.initialization_window - 1
        state = self.initialize(returns.iloc[: self.initialization_window])
        variance[initial_position] = state.variance
        for position in range(self.initialization_window, len(returns)):
            state = self.update(state, float(values[position]))
            variance[position] = state.variance

        volatility = np.sqrt(variance)
        z_0_05 = float(norm.ppf(0.05))
        z_0_01 = float(norm.ppf(0.01))
        q_0_05 = volatility * z_0_05
        q_0_01 = volatility * z_0_01
        return pd.DataFrame(
            {
                "variance": variance,
                "volatility": volatility,
                "return_quantile_0_05": q_0_05,
                "var_0_95": np.maximum(0.0, -q_0_05),
                "return_quantile_0_01": q_0_01,
                "var_0_99": np.maximum(0.0, -q_0_01),
            },
            index=returns.index,
        )

    @staticmethod
    def _state(variance: float) -> EwmaState:
        if not math.isfinite(variance):
            raise NonfiniteVarianceError("EWMA variance is non-finite.")
        if variance <= 0.0:
            raise NonpositiveVarianceError("EWMA variance is not strictly positive.")
        return EwmaState(variance=variance)


__all__ = [
    "EWMA_MODEL_ID",
    "EwmaForecast",
    "EwmaModel",
    "EwmaState",
]
