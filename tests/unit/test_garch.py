from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t

from market_risk_forecasting.errors import (
    InvalidStudentTDofError,
    NonstationaryParametersError,
)
from market_risk_forecasting.models import garch
from market_risk_forecasting.models.garch import (
    GarchModel,
    GarchParameters,
    GarchState,
)


def _returns() -> pd.Series:
    values = np.sin(np.arange(1250) / 13.0) / 100 + np.cos(np.arange(1250) / 29.0) / 500
    return pd.Series(
        values,
        index=pd.date_range("2007-01-01", periods=1250, freq="D"),
        name="SPY",
    )


def _fake_result(
    *,
    distribution_name: str,
    convergence_flag: int = 0,
) -> SimpleNamespace:
    parameters = {
        "omega": 0.05,
        "alpha[1]": 0.08,
        "beta[1]": 0.90,
    }
    if distribution_name == "studentst":
        parameters["nu"] = 8.0
    return SimpleNamespace(
        params=pd.Series(parameters),
        conditional_volatility=pd.Series([1.8, 2.0]),
        convergence_flag=convergence_flag,
        optimization_result=SimpleNamespace(message="test optimizer"),
    )


def test_independent_gaussian_recursion_and_decimal_scaling() -> None:
    model = GarchModel(distribution="gaussian")
    parameters = GarchParameters(omega=0.1, alpha=0.1, beta=0.8)
    state = GarchState(parameters=parameters, variance_scaled=4.0)

    next_state, forecast = model.advance(state, latest_return=0.02)

    expected_scaled_variance = 0.1 + 0.1 * 2.0**2 + 0.8 * 4.0
    expected_decimal_variance = expected_scaled_variance / 100.0**2
    assert next_state.variance_scaled == pytest.approx(expected_scaled_variance)
    assert forecast.variance == pytest.approx(expected_decimal_variance)
    assert forecast.volatility == pytest.approx(math.sqrt(expected_decimal_variance))


def test_gaussian_quantiles_use_standard_normal() -> None:
    model = GarchModel(distribution="gaussian")
    state = GarchState(
        parameters=GarchParameters(omega=0.05, alpha=0.05, beta=0.90),
        variance_scaled=4.0,
    )

    forecast = model.forecast(state)

    assert forecast.volatility == pytest.approx(0.02)
    assert forecast.return_quantile_0_05 == pytest.approx(0.02 * float(norm.ppf(0.05)))
    assert forecast.return_quantile_0_01 == pytest.approx(0.02 * float(norm.ppf(0.01)))
    assert forecast.var_0_95 <= forecast.var_0_99


def test_student_t_quantiles_are_standardized_to_unit_variance() -> None:
    degrees_of_freedom = 8.0
    model = GarchModel(distribution="student_t")
    state = GarchState(
        parameters=GarchParameters(
            omega=0.05,
            alpha=0.05,
            beta=0.90,
            degrees_of_freedom=degrees_of_freedom,
        ),
        variance_scaled=4.0,
    )

    forecast = model.forecast(state)
    standardized = math.sqrt((degrees_of_freedom - 2.0) / degrees_of_freedom)

    assert forecast.return_quantile_0_05 == pytest.approx(
        0.02 * float(t.ppf(0.05, degrees_of_freedom)) * standardized
    )
    assert forecast.return_quantile_0_01 == pytest.approx(
        0.02 * float(t.ppf(0.01, degrees_of_freedom)) * standardized
    )


def test_invalid_persistence_and_student_t_dof_fail() -> None:
    with pytest.raises(NonstationaryParametersError):
        GarchModel(distribution="gaussian").validate_parameters(
            GarchParameters(omega=0.05, alpha=0.05, beta=0.95)
        )
    with pytest.raises(NonstationaryParametersError):
        GarchModel(distribution="gaussian").validate_parameters(
            GarchParameters(
                omega=0.05,
                alpha=0.05,
                beta=0.95 - 0.5e-8,
            )
        )
    with pytest.raises(InvalidStudentTDofError):
        GarchModel(distribution="student_t").validate_parameters(
            GarchParameters(
                omega=0.05,
                alpha=0.05,
                beta=0.90,
                degrees_of_freedom=2.0,
            )
        )


def test_fit_retries_once_with_the_protocol_starting_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, np.ndarray[Any, np.dtype[np.float64]] | None]] = []

    def fake_fit(
        scaled_returns: pd.Series,
        distribution_name: str,
        starting_values: np.ndarray[Any, np.dtype[np.float64]] | None,
    ) -> SimpleNamespace:
        del scaled_returns
        calls.append((distribution_name, starting_values))
        if starting_values is None:
            raise RuntimeError("first attempt failed")
        return _fake_result(distribution_name=distribution_name)

    monkeypatch.setattr(garch, "_fit_arch_model", fake_fit)
    returns = _returns()

    outcome = GarchModel(distribution="student_t").fit(returns)

    expected_variance = float((returns * 100.0).var(ddof=1))
    assert outcome.converged
    assert outcome.retry_used
    assert outcome.error_code is None
    assert outcome.state is not None
    assert outcome.state.variance_scaled == pytest.approx(4.0)
    assert calls[0] == ("studentst", None)
    assert calls[1][0] == "studentst"
    np.testing.assert_allclose(
        calls[1][1],
        np.array([0.05 * expected_variance, 0.05, 0.90, 8.0]),
    )


def test_failed_retry_returns_typed_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def failing_fit(*args: object, **kwargs: object) -> None:
        del args, kwargs
        nonlocal attempts
        attempts += 1
        raise RuntimeError("optimizer unavailable")

    monkeypatch.setattr(garch, "_fit_arch_model", failing_fit)

    outcome = GarchModel(distribution="gaussian").fit(_returns())

    assert attempts == 2
    assert not outcome.converged
    assert outcome.retry_used
    assert outcome.state is None
    assert outcome.error_code == "MODEL_FIT_FAILED"


def test_converged_nonstationary_fit_retains_invalid_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def nonstationary_fit(
        scaled_returns: pd.Series,
        distribution_name: str,
        starting_values: np.ndarray[Any, np.dtype[np.float64]] | None,
    ) -> SimpleNamespace:
        del scaled_returns, distribution_name, starting_values
        result = _fake_result(distribution_name="normal")
        result.params["alpha[1]"] = 0.10
        result.params["beta[1]"] = 0.90
        return result

    monkeypatch.setattr(garch, "_fit_arch_model", nonstationary_fit)

    outcome = GarchModel(distribution="gaussian").fit(_returns())

    assert outcome.converged
    assert outcome.state is None
    assert outcome.parameters is not None
    assert outcome.parameters.persistence == pytest.approx(1.0)
    assert outcome.error_code == "NONSTATIONARY_PARAMETERS"


def test_state_serialization_round_trip_preserves_forecast() -> None:
    model = GarchModel(distribution="student_t")
    original = GarchState(
        parameters=GarchParameters(
            omega=0.05,
            alpha=0.05,
            beta=0.90,
            degrees_of_freedom=8.0,
        ),
        variance_scaled=2.5,
    )

    restored = GarchState.from_dict(original.to_dict())

    assert restored == original
    assert model.forecast(restored) == model.forecast(original)
