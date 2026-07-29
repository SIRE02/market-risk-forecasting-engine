from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from market_risk_forecasting.errors import (
    InsufficientHistoryError,
    NonpositiveVarianceError,
)
from market_risk_forecasting.models.historical import (
    HistoricalSimulationModel,
    HistoricalVarianceModel,
)


def test_sample_variance_uses_ddof_one() -> None:
    values = pd.Series(np.linspace(-0.03, 0.04, 252))

    forecast = HistoricalVarianceModel().forecast(values)

    expected = float(values.var(ddof=1))
    assert forecast.variance == pytest.approx(expected)
    assert forecast.volatility == pytest.approx(math.sqrt(expected))


@pytest.mark.parametrize("count", [251, 253])
def test_variance_requires_exactly_252_observations(count: int) -> None:
    with pytest.raises(InsufficientHistoryError, match="exactly 252"):
        HistoricalVarianceModel().forecast(pd.Series(np.arange(count)))


def test_nonpositive_sample_variance_fails() -> None:
    with pytest.raises(NonpositiveVarianceError):
        HistoricalVarianceModel().forecast(pd.Series(np.zeros(252)))


def test_historical_simulation_uses_exact_linear_quantiles() -> None:
    values = pd.Series(np.linspace(-0.10, 0.08, 500))

    forecast = HistoricalSimulationModel().forecast(values)

    assert forecast.return_quantile_0_05 == pytest.approx(
        values.quantile(0.05, interpolation="linear")
    )
    assert forecast.return_quantile_0_01 == pytest.approx(
        values.quantile(0.01, interpolation="linear")
    )


@pytest.mark.parametrize("count", [499, 501])
def test_historical_simulation_requires_exactly_500_observations(
    count: int,
) -> None:
    with pytest.raises(InsufficientHistoryError, match="exactly 500"):
        HistoricalSimulationModel().forecast(pd.Series(np.arange(count)))


def test_positive_loss_var_sign_and_confidence_order() -> None:
    values = pd.Series(np.linspace(-0.10, 0.08, 500))

    forecast = HistoricalSimulationModel().forecast(values)

    assert forecast.var_0_95 == pytest.approx(-forecast.return_quantile_0_05)
    assert forecast.var_0_99 == pytest.approx(-forecast.return_quantile_0_01)
    assert forecast.var_0_95 <= forecast.var_0_99


def test_positive_lower_tail_is_floored_only_for_loss_var() -> None:
    values = pd.Series(np.linspace(0.001, 0.02, 500))

    forecast = HistoricalSimulationModel().forecast(values)

    assert forecast.return_quantile_0_05 > 0.0
    assert forecast.return_quantile_0_01 > 0.0
    assert forecast.var_0_95 == 0.0
    assert forecast.var_0_99 == 0.0


def test_vectorized_variance_path_matches_single_window() -> None:
    values = pd.Series(np.sin(np.arange(600) / 17.0) / 100)
    model = HistoricalVarianceModel()

    path = model.forecast_path(values)
    single = model.forecast(values.iloc[:252])

    assert path.loc[251, "variance"] == pytest.approx(single.variance)
    assert path.loc[251, "volatility"] == pytest.approx(single.volatility)


def test_vectorized_quantile_path_matches_single_window() -> None:
    values = pd.Series(np.sin(np.arange(600) / 17.0) / 100)
    model = HistoricalSimulationModel()

    path = model.forecast_path(values)
    single = model.forecast(values.iloc[:500])

    assert path.loc[499, "return_quantile_0_05"] == pytest.approx(
        single.return_quantile_0_05
    )
    assert path.loc[499, "var_0_95"] == pytest.approx(single.var_0_95)
    assert path.loc[499, "return_quantile_0_01"] == pytest.approx(
        single.return_quantile_0_01
    )
    assert path.loc[499, "var_0_99"] == pytest.approx(single.var_0_99)
