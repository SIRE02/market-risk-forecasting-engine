from __future__ import annotations

import numpy as np
import pandas as pd

from market_risk_forecasting.reporting import (
    _report_identity,
    _rolling_loss_differences,
)


def test_custom_report_identity_uses_configured_series() -> None:
    effective = {
        "upstream": {"instruments": ["AAPL", "MSFT", "GLD"]},
        "portfolio_proxy": {"enabled": False},
    }

    title, series = _report_identity(effective)

    assert title == "# Market Risk Forecasting Engine Research Report"
    assert series == "AAPL, MSFT, GLD"


def test_report_identity_includes_configured_proxy() -> None:
    effective = {
        "upstream": {"instruments": ["SPY", "IEF", "GLD"]},
        "portfolio_proxy": {"enabled": True, "series_id": "FIXTURE_PROXY"},
    }

    title, series = _report_identity(effective)

    assert title == "# Market Risk Forecasting Engine Research Report"
    assert series == "SPY, IEF, GLD, FIXTURE_PROXY"


def test_rolling_loss_differences_preserve_direction_and_initial_gap() -> None:
    dates = pd.date_range("2025-01-02", periods=3, freq="B")
    keys = [
        {
            "series_id": "TEST",
            "forecast_origin": target_date - pd.offsets.BDay(1),
            "target_date": target_date,
        }
        for target_date in dates
    ]
    forecasts = pd.DataFrame(
        [
            {
                **key,
                "model_id": model_id,
                "status": "ok",
                "variance": variance,
                "return_quantile_0_05": quantile,
            }
            for key in keys
            for model_id, variance, quantile in (
                ("historical_variance", 0.04, np.nan),
                ("historical_simulation", np.nan, -0.20),
                ("ewma", 0.01, -0.10),
                ("garch_1_1_gaussian", 0.01, -0.10),
                ("garch_1_1_student_t", 0.01, -0.10),
            )
        ]
    )
    realizations = pd.DataFrame([{**key, "simple_return": -0.10} for key in keys])

    result = _rolling_loss_differences(forecasts, realizations, window=2)

    ewma_qlike = result.loc[
        result["metric"].eq("qlike") & result["model_id"].eq("ewma")
    ]
    ewma_pinball = result.loc[
        result["metric"].eq("pinball_loss_0_05") & result["model_id"].eq("ewma")
    ]
    assert np.isnan(ewma_qlike.iloc[0]["rolling_difference"])
    assert ewma_qlike.iloc[1:]["rolling_difference"].lt(0.0).all()
    assert np.isnan(ewma_pinball.iloc[0]["rolling_difference"])
    assert np.allclose(
        ewma_pinball.iloc[1:]["rolling_difference"],
        -0.005,
    )
