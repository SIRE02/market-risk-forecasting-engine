from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_risk_forecasting.config import POOLED_SERIES_ID
from market_risk_forecasting.reporting import (
    _CANDIDATES,
    _calibration_summary,
    _coverage_rows,
    _interval_error_bars,
    _plot_comparison,
    _plot_series_comparisons,
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
    realizations = pd.DataFrame(
        [{**key, "simple_return": -0.10, "period": "test"} for key in keys]
    )

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
            realizations.append({**key, "simple_return": -0.10, "period": "test"})
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


def test_interval_figures_accept_intervals_that_exclude_the_estimate(
    tmp_path: Path,
) -> None:
    # Percentile bootstrap intervals need not contain the point estimate.
    bootstrap = pd.DataFrame(
        [
            {
                "period": "test",
                "series_id": series_id,
                "metric": metric,
                "model_id": model_id,
                "value": 5.0,
                "interval_lower": 0.0,
                "interval_upper": 2.5,
            }
            for series_id in (POOLED_SERIES_ID, "SPY", "GLD")
            for metric in ("qlike", "pinball_loss_0_05")
            for model_id in _CANDIDATES
        ]
    )

    _plot_comparison(
        bootstrap,
        metric="qlike",
        title="QLIKE",
        scale=1.0,
        ylabel="Loss difference",
        path=tmp_path / "comparison.png",
    )
    _plot_series_comparisons(bootstrap, tmp_path / "series.png")

    assert (tmp_path / "comparison.png").stat().st_size > 0
    assert (tmp_path / "series.png").stat().st_size > 0
    centers, half_widths = _interval_error_bars(np.array([0.0]), np.array([2.5]))
    assert centers.tolist() == [1.25]
    assert half_widths.tolist() == [1.25]


def _rolling_inputs(
    dates: pd.DatetimeIndex,
    series_ids: tuple[str, ...],
    benchmark_failed: set[tuple[str, pd.Timestamp]],
    periods: dict[pd.Timestamp, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecasts: list[dict[str, object]] = []
    realizations: list[dict[str, object]] = []
    for target_date in dates:
        for series_id in series_ids:
            key = {
                "series_id": series_id,
                "forecast_origin": target_date - pd.offsets.BDay(1),
                "target_date": target_date,
            }
            realizations.append(
                {
                    **key,
                    "simple_return": -0.10,
                    "period": (periods or {}).get(target_date, "test"),
                }
            )
            for model_id, variance, quantile in (
                ("historical_variance", 0.04, np.nan),
                ("historical_simulation", np.nan, -0.20),
                ("ewma", 0.01, -0.10),
                ("garch_1_1_gaussian", 0.01, -0.10),
                ("garch_1_1_student_t", 0.01, -0.10),
            ):
                failed = (
                    model_id == "historical_variance"
                    and (series_id, target_date) in benchmark_failed
                )
                forecasts.append(
                    {
                        **key,
                        "model_id": model_id,
                        "status": "failed" if failed else "ok",
                        "variance": np.nan if failed else variance,
                        "return_quantile_0_05": quantile,
                    }
                )
    return pd.DataFrame(forecasts), pd.DataFrame(realizations)


def _ewma_rolling_qlike(forecasts: pd.DataFrame, realizations: pd.DataFrame) -> Any:
    result = _rolling_loss_differences(forecasts, realizations, window=2)
    return result.loc[
        result["metric"].eq("qlike") & result["model_id"].eq("ewma")
    ].reset_index(drop=True)


def test_rolling_loss_differences_do_not_bridge_failed_benchmark_dates() -> None:
    dates = pd.date_range("2025-01-02", periods=6, freq="B")
    failed = {(series_id, dates[i]) for series_id in ("A", "B") for i in (2, 3, 4)}

    ewma_qlike = _ewma_rolling_qlike(*_rolling_inputs(dates, ("A", "B"), failed))

    assert len(ewma_qlike) == len(dates)
    assert ewma_qlike["rolling_difference"].iloc[1] < 0.0
    assert ewma_qlike["rolling_difference"].iloc[2:].isna().all()


def test_rolling_loss_differences_keep_series_without_valid_benchmarks() -> None:
    dates = pd.date_range("2025-01-02", periods=4, freq="B")
    failed = {("B", target_date) for target_date in dates}

    ewma_qlike = _ewma_rolling_qlike(*_rolling_inputs(dates, ("A", "B"), failed))

    assert len(ewma_qlike) == len(dates)
    assert ewma_qlike["rolling_difference"].isna().all()


def test_rolling_loss_differences_restart_at_period_boundaries() -> None:
    dates = pd.date_range("2025-01-02", periods=6, freq="B")
    periods = {
        target_date: "validation" if position < 3 else "test"
        for position, target_date in enumerate(dates)
    }

    ewma_qlike = _ewma_rolling_qlike(
        *_rolling_inputs(dates, ("A", "B"), set(), periods)
    )

    # Window 2: the first test date must not average in the last validation date.
    assert ewma_qlike["rolling_difference"].iloc[[0, 3]].isna().all()
    assert ewma_qlike["rolling_difference"].iloc[[1, 2, 4, 5]].lt(0.0).all()
