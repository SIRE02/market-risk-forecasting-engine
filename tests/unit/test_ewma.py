from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from market_risk_forecasting.errors import (
    InputValueInvalidError,
    InsufficientHistoryError,
    NonpositiveVarianceError,
)
from market_risk_forecasting.models.ewma import EwmaModel, EwmaState


def _returns(count: int = 600) -> pd.Series:
    return pd.Series(
        np.sin(np.arange(count) / 13.0) / 100,
        index=pd.date_range("2007-01-01", periods=count, freq="D"),
    )


def test_initialization_is_exact_252_sample_variance() -> None:
    returns = _returns()
    model = EwmaModel()

    state = model.initialize(returns.iloc[:252])

    assert state.variance == pytest.approx(float(returns.iloc[:252].var(ddof=1)))
    with pytest.raises(InsufficientHistoryError, match="exactly 252"):
        model.initialize(returns.iloc[:251])


def test_hand_calculated_recursion() -> None:
    model = EwmaModel()
    state = EwmaState(variance=0.0004)
    latest_return = -0.03

    updated = model.update(state, latest_return)

    expected = 0.94 * 0.0004 + 0.06 * latest_return**2
    assert updated.variance == pytest.approx(expected)


def test_custom_decay_and_initialization_window_are_supported() -> None:
    returns = _returns(300)
    model = EwmaModel(lambda_=0.97, initialization_window=126)

    path = model.forecast_path(returns)

    assert path.iloc[:125]["variance"].isna().all()
    assert path.iloc[125]["variance"] == pytest.approx(
        float(returns.iloc[:126].var(ddof=1))
    )


@pytest.mark.parametrize(
    ("lambda_", "window"),
    [(0.0, 252), (1.0, 252), (0.94, 1)],
)
def test_invalid_ewma_controls_fail(lambda_: float, window: int) -> None:
    with pytest.raises(InputValueInvalidError):
        EwmaModel(lambda_=lambda_, initialization_window=window)


def test_zero_returns_decay_geometrically() -> None:
    model = EwmaModel()
    state = EwmaState(variance=0.0004)

    for _ in range(10):
        state = model.update(state, 0.0)

    assert state.variance == pytest.approx(0.0004 * 0.94**10)


def test_one_shock_then_zero_returns_decays_geometrically() -> None:
    model = EwmaModel()
    initial = EwmaState(variance=0.0001)
    shocked = model.update(initial, 0.05)
    state = shocked

    for _ in range(8):
        state = model.update(state, 0.0)

    assert state.variance == pytest.approx(shocked.variance * 0.94**8)


def test_forecast_path_uses_initial_variance_then_new_origin_return() -> None:
    returns = _returns()
    model = EwmaModel()

    path = model.forecast_path(returns)
    initial = float(returns.iloc[:252].var(ddof=1))
    next_expected = 0.94 * initial + 0.06 * float(returns.iloc[252]) ** 2

    assert path.iloc[:251]["variance"].isna().all()
    assert path.iloc[251]["variance"] == pytest.approx(initial)
    assert path.iloc[252]["variance"] == pytest.approx(next_expected)


def test_scaling_returns_scales_variance_by_c_squared() -> None:
    returns = _returns()
    model = EwmaModel()
    scale = 3.5

    original = model.forecast_path(returns)
    scaled = model.forecast_path(returns * scale)

    np.testing.assert_allclose(
        scaled["variance"].iloc[251:].to_numpy(),
        original["variance"].iloc[251:].to_numpy() * scale**2,
        rtol=1e-12,
        atol=1e-18,
    )


def test_future_append_invariance() -> None:
    returns = _returns()
    model = EwmaModel()

    short = model.forecast_path(returns.iloc[:550])
    extended = model.forecast_path(returns)

    pd.testing.assert_frame_equal(short, extended.iloc[:550])


def test_gaussian_quantiles_and_positive_loss_var() -> None:
    model = EwmaModel()
    forecast = model.forecast(EwmaState(variance=0.0004))

    expected_volatility = 0.02
    expected_q05 = expected_volatility * float(norm.ppf(0.05))
    expected_q01 = expected_volatility * float(norm.ppf(0.01))
    assert forecast.volatility == pytest.approx(expected_volatility)
    assert forecast.return_quantile_0_05 == pytest.approx(expected_q05)
    assert forecast.return_quantile_0_01 == pytest.approx(expected_q01)
    assert forecast.var_0_95 == pytest.approx(-expected_q05)
    assert forecast.var_0_99 == pytest.approx(-expected_q01)
    assert forecast.var_0_95 <= forecast.var_0_99


def test_nonpositive_initialization_fails() -> None:
    with pytest.raises(NonpositiveVarianceError):
        EwmaModel().initialize(pd.Series(np.zeros(252)))


def test_state_units_remain_decimal_variance() -> None:
    forecast = EwmaModel().forecast(EwmaState(variance=0.01**2))

    assert math.isclose(forecast.volatility, 0.01)
