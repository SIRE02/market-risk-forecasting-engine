from __future__ import annotations

import numpy as np
import pandas as pd

from market_risk_forecasting.reporting import (
    _calibration_summary,
    _coverage_rows,
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


def test_rolling_loss_differences_require_a_balanced_series_panel() -> None:
    dates = pd.date_range("2025-01-02", periods=4, freq="B")
    forecasts: list[dict[str, object]] = []
    realizations: list[dict[str, object]] = []
    for target_date in dates:
        for series_id in ("A", "B"):
            key = {
                "series_id": series_id,
                "forecast_origin": target_date - pd.offsets.BDay(1),
                "target_date": target_date,
            }
            realizations.append({**key, "simple_return": -0.10})
            for model_id, variance, quantile in (
                ("historical_variance", 0.04, np.nan),
                ("historical_simulation", np.nan, -0.20),
                ("ewma", 0.01, -0.10),
                ("garch_1_1_gaussian", 0.01, -0.10),
                ("garch_1_1_student_t", 0.01, -0.10),
            ):
                status = (
                    "failed"
                    if model_id == "ewma"
                    and series_id == "B"
                    and target_date == dates[1]
                    else "ok"
                )
                forecasts.append(
                    {
                        **key,
                        "model_id": model_id,
                        "status": status,
                        "variance": variance,
                        "return_quantile_0_05": quantile,
                    }
                )

    result = _rolling_loss_differences(
        pd.DataFrame(forecasts),
        pd.DataFrame(realizations),
        window=2,
    )

    ewma_qlike = result.loc[
        result["metric"].eq("qlike") & result["model_id"].eq("ewma")
    ]
    assert len(ewma_qlike) == len(dates)
    assert ewma_qlike["rolling_difference"].iloc[:3].isna().all()
    assert ewma_qlike["rolling_difference"].iloc[3] < 0.0


def test_coverage_rows_select_requested_confidence_level() -> None:
    rows: list[dict[str, object]] = []
    for confidence_level, exception_rate in ((0.95, 0.05), (0.99, 0.01)):
        for metric, value, status in (
            ("exception_count", 5.0, "ok"),
            ("exception_rate", exception_rate, "ok"),
            ("kupiec_p_value", 0.75, "ok"),
            ("christoffersen_independence_lr", 0.1, "ok"),
        ):
            rows.append(
                {
                    "period": "test",
                    "series_id": "TEST",
                    "model_id": "ewma",
                    "metric": metric,
                    "confidence_level": confidence_level,
                    "value": value,
                    "status": status,
                    "observation_count": 100,
                }
            )

    assert _coverage_rows(pd.DataFrame(rows), confidence_level=0.99)[0][3] == "0.01"


def test_calibration_summary_separates_rejections_from_non_rejections() -> None:
    rows: list[dict[str, object]] = []
    for model_id in (
        "historical_simulation",
        "ewma",
        "garch_1_1_gaussian",
        "garch_1_1_student_t",
    ):
        for series_id in ("A", "B"):
            for confidence_level in (0.95, 0.99):
                p_value = (
                    0.01
                    if model_id != "garch_1_1_student_t"
                    and series_id == "A"
                    and confidence_level == 0.99
                    else 0.20
                )
                rows.append(
                    {
                        "period": "test",
                        "series_id": series_id,
                        "model_id": model_id,
                        "metric": "kupiec_p_value",
                        "confidence_level": confidence_level,
                        "value": p_value,
                    }
                )

    summary = _calibration_summary(pd.DataFrame(rows))

    assert "Student-t GARCH(1,1) had no Kupiec coverage rejection" in summary
    assert (
        "Historical simulation, EWMA, and Gaussian GARCH(1,1) had at least one "
        "rejection" in summary
    )
