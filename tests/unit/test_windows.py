from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from market_risk_forecasting.config import PeriodConfig
from market_risk_forecasting.errors import WindowAlignmentError
from market_risk_forecasting.windows import (
    classify_target_date,
    iter_expanding_forecast_windows,
    iter_forecast_windows,
    validate_canonical_index,
)


@pytest.fixture
def periods() -> PeriodConfig:
    return PeriodConfig(
        development_start=date(2007, 1, 1),
        development_end=date(2014, 12, 31),
        validation_start=date(2015, 1, 1),
        validation_end=date(2019, 12, 31),
        test_start=date(2020, 1, 1),
        test_end=date(2025, 12, 31),
    )


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("2007-01-01", "development"),
        ("2014-12-31", "development"),
        ("2015-01-01", "validation"),
        ("2019-12-31", "validation"),
        ("2020-01-01", "test"),
        ("2025-12-31", "test"),
        ("2026-01-01", None),
    ],
)
def test_period_classification_uses_target_date(
    periods: PeriodConfig,
    target: str,
    expected: str | None,
) -> None:
    assert classify_target_date(pd.Timestamp(target), periods) == expected


def test_exact_origin_target_and_training_window(periods: PeriodConfig) -> None:
    index = pd.date_range("2014-12-26", periods=7, freq="D")

    windows = list(
        iter_forecast_windows(
            index=index,
            series_id="SPY",
            model_id="test_model",
            training_window=3,
            periods=periods,
        )
    )

    first = windows[0]
    assert first.train_start == index[0]
    assert first.train_end == index[2]
    assert first.forecast_origin == index[2]
    assert first.target_date == index[3]
    assert first.train_observation_count == 3
    assert first.target_position == first.origin_position + 1
    assert first.forecast_origin < first.target_date
    assert windows[-1].period == "validation"


def test_target_is_never_in_training_slice(periods: PeriodConfig) -> None:
    index = pd.date_range("2010-01-01", periods=20, freq="D")

    windows = list(
        iter_forecast_windows(
            index=index,
            series_id="SPY",
            model_id="test_model",
            training_window=5,
            periods=periods,
        )
    )

    for window in windows:
        training_index = index[
            window.origin_position
            - window.train_observation_count
            + 1 : window.origin_position + 1
        ]
        assert window.target_date not in training_index
        assert training_index[-1] == window.forecast_origin


def test_appending_future_dates_leaves_prior_windows_unchanged(
    periods: PeriodConfig,
) -> None:
    original = pd.date_range("2010-01-01", periods=20, freq="D")
    extended = pd.date_range("2010-01-01", periods=25, freq="D")

    original_windows = list(
        iter_forecast_windows(
            index=original,
            series_id="SPY",
            model_id="test_model",
            training_window=5,
            periods=periods,
        )
    )
    extended_prior = [
        window
        for window in iter_forecast_windows(
            index=extended,
            series_id="SPY",
            model_id="test_model",
            training_window=5,
            periods=periods,
        )
        if window.target_date <= original[-1]
    ]

    assert extended_prior == original_windows


def test_expanding_windows_keep_the_initial_training_start(
    periods: PeriodConfig,
) -> None:
    index = pd.date_range("2014-12-20", periods=20, freq="D")

    windows = list(
        iter_expanding_forecast_windows(
            index=index,
            series_id="SPY",
            model_id="ewma_lambda_0_94",
            initial_observations=5,
            periods=periods,
        )
    )

    assert [window.train_observation_count for window in windows] == list(range(5, 20))
    assert all(window.train_start == index[0] for window in windows)
    assert all(not window.scheduled_refit for window in windows)
    first_validation = next(
        window for window in windows if window.period == "validation"
    )
    assert first_validation.target_date == pd.Timestamp("2015-01-01")
    assert first_validation.train_start == index[0]
    assert first_validation.train_observation_count == 12


def test_duplicate_and_unsorted_dates_fail() -> None:
    with pytest.raises(WindowAlignmentError, match="duplicates"):
        validate_canonical_index(pd.DatetimeIndex(["2020-01-01", "2020-01-01"]))
    with pytest.raises(WindowAlignmentError, match="not sorted"):
        validate_canonical_index(pd.DatetimeIndex(["2020-01-02", "2020-01-01"]))
