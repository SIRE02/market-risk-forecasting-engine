from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_risk_forecasting.config import load_config
from market_risk_forecasting.errors import InsufficientHistoryError
from market_risk_forecasting.evaluation import (
    BOOTSTRAP_COMPARISON_COLUMNS,
    COVERAGE_TEST_COLUMNS,
    FORECAST_AVAILABILITY_COLUMNS,
    PERIOD_BREAKDOWN_COLUMNS,
    QUANTILE_SCORE_COLUMNS,
    VARIANCE_SCORE_COLUMNS,
    coverage_statistics,
    evaluate_forecasts,
    exception_indicator,
    moving_block_bootstrap,
    pinball_loss,
    transition_counts,
    variance_score_values,
)


def test_known_variance_scores() -> None:
    scores = variance_score_values(
        variance=0.04,
        realized_return=0.10,
    )

    assert scores["qlike"] == pytest.approx(math.log(0.04) + 0.01 / 0.04)
    assert scores["squared_error"] == pytest.approx((0.01 - 0.04) ** 2)
    assert scores["absolute_error"] == pytest.approx(0.03)


def test_known_pinball_loss_on_both_sides_of_quantile() -> None:
    below = pinball_loss(
        realized_return=-0.01,
        return_quantile=0.0,
        tail_probability=0.05,
    )
    above = pinball_loss(
        realized_return=0.01,
        return_quantile=0.0,
        tail_probability=0.05,
    )

    assert below == pytest.approx(0.0095)
    assert above == pytest.approx(0.0005)


def test_var_exception_is_strict() -> None:
    assert not exception_indicator(realized_loss=0.02, value_at_risk=0.02)
    assert exception_indicator(realized_loss=0.020001, value_at_risk=0.02)


def test_known_coverage_statistics_and_transition_counts() -> None:
    exceptions = np.array(
        [False, False, True, True, False, True, False, False, True, False]
    )

    statistics = coverage_statistics(
        exceptions,
        expected_probability=0.20,
    )

    assert transition_counts(exceptions) == {
        "n00": 2,
        "n01": 3,
        "n10": 3,
        "n11": 1,
    }
    assert statistics["exception_count"] == 4
    assert statistics["longest_exception_cluster"] == 2
    assert statistics["kupiec_lr"] == pytest.approx(2.092992575058192)
    assert statistics["kupiec_p_value"] == pytest.approx(0.1479759593850231)
    assert statistics["christoffersen_independence_lr"] == pytest.approx(
        1.1365105517087883
    )
    # Joint test on the nine transitions: 4 exceptions against p = 0.20 versus
    # the Markov chain with P(0->1) = 3/5 and P(1->1) = 1/4.
    assert statistics["conditional_coverage_lr"] == pytest.approx(3.878140985571868)
    assert statistics["christoffersen_status"] == "ok"


def test_conditional_coverage_uses_only_linked_transitions() -> None:
    exceptions = np.array(
        [False, False, True, True, False, True, False, False, True, False]
    )
    adjacent = np.ones(exceptions.size - 1, dtype=bool)
    adjacent[4] = False  # a missing forecast breaks the 0 -> 1 link at 4 -> 5

    statistics = coverage_statistics(
        exceptions,
        expected_probability=0.20,
        adjacent=adjacent,
    )

    assert statistics["excluded_gap_transition_count"] == 1
    assert statistics["kupiec_lr"] == pytest.approx(2.092992575058192)
    # Eight linked transitions with 3 exceptions, P(0->1) = 2/4, P(1->1) = 1/4.
    assert statistics["conditional_coverage_lr"] == pytest.approx(1.8442043863166706)


def test_undefined_christoffersen_test_is_labelled() -> None:
    statistics = coverage_statistics(
        np.zeros(20, dtype=bool),
        expected_probability=0.05,
    )

    assert statistics["christoffersen_status"] == "insufficient_events"
    assert statistics["christoffersen_independence_lr"] is None
    assert math.isfinite(float(statistics["kupiec_lr"]))


def test_moving_block_bootstrap_is_reproducible_and_synchronized() -> None:
    groups = (
        np.array([-2.0, -1.0, 0.0, 1.0]),
        np.array([2.0, 3.0, 4.0, 5.0]),
    )

    first = moving_block_bootstrap(
        groups,
        block_length=2,
        resamples=100,
        random_seed=42,
    )
    second = moving_block_bootstrap(
        groups,
        block_length=2,
        resamples=100,
        random_seed=42,
    )

    np.testing.assert_array_equal(first, second)
    assert first.shape == (100,)
    assert np.isfinite(first).all()


def test_pooled_bootstrap_preserves_cross_asset_dependence() -> None:
    values = np.array([-3.0, 2.0, 1.0, 4.0, -2.0, 5.0])
    controls = dict(block_length=2, resamples=200, random_seed=42)
    single = moving_block_bootstrap((values,), **controls)
    duplicate = moving_block_bootstrap((values, values), **controls)
    hedged = moving_block_bootstrap((values, -values), **controls)

    np.testing.assert_allclose(duplicate, single)
    np.testing.assert_array_equal(hedged, np.zeros(200))


def test_bootstrap_retains_missing_dates_in_time_blocks() -> None:
    values = np.array([1.0, np.nan, np.nan, 3.0, 5.0])
    draws = moving_block_bootstrap(
        (values,), block_length=3, resamples=200, random_seed=42
    )

    assert np.isfinite(draws).all()
    assert draws.min() == 1.0
    # Removing the missing dates would leave one length-three block, producing
    # the constant mean 3 in every replicate instead of this uncertainty.
    assert np.ptp(draws) > 0.0


@pytest.mark.parametrize("block_length", [4, 20])
def test_single_possible_bootstrap_block_rejects_inference(block_length: int) -> None:
    with pytest.raises(InsufficientHistoryError, match="multiple possible date blocks"):
        moving_block_bootstrap(
            (np.array([1.0, 2.0, 3.0, 4.0]),),
            block_length=block_length,
            resamples=2000,
            random_seed=42,
        )


def test_short_period_keeps_effects_without_confidence_claims(
    project_root: Path,
) -> None:
    forecasts, realizations = _evaluation_inputs()
    artifacts = evaluate_forecasts(
        forecasts=forecasts,
        realizations=realizations,
        config=load_config(project_root / "config.example.toml"),
    )
    comparisons = artifacts.bootstrap_comparisons
    observed = comparisons.loc[comparisons.paired_count.gt(0)]
    assert observed.value.notna().all()
    assert observed.median_difference.notna().all()
    assert (
        observed[["interval_lower", "interval_upper", "fraction_below_zero"]]
        .isna()
        .all()
        .all()
    )


def test_coverage_does_not_bridge_gaps_or_join_exception_clusters() -> None:
    values = np.array(
        [False, False, True, True, False, True, False, False, True, False]
    )
    links = np.ones(len(values) - 1, dtype=bool)
    links[2] = False
    continuous = coverage_statistics(values, expected_probability=0.20)
    gapped = coverage_statistics(values, expected_probability=0.20, adjacent=links)

    assert gapped["n11"] == 0
    assert gapped["n00"] == continuous["n00"]
    assert gapped["n01"] == continuous["n01"]
    assert gapped["n10"] == continuous["n10"]
    assert gapped["longest_exception_cluster"] == 1
    assert gapped["excluded_gap_transition_count"] == 1
    assert gapped["kupiec_lr"] == continuous["kupiec_lr"]
    assert gapped["christoffersen_status"] == "insufficient_events"


def test_evaluation_detects_missing_forecasts_as_transition_gaps(
    project_root: Path,
) -> None:
    forecasts, realizations = _evaluation_inputs()
    realizations["period"] = "test"
    # Remove one middle EWMA forecast from a otherwise valid sequence. The
    # next successful forecast cannot treat the earlier target as its origin.
    missing = forecasts["model_id"].eq("ewma") & forecasts["target_date"].eq(
        realizations["target_date"].iloc[1]
    )
    forecasts.loc[missing, "status"] = "failed"
    artifacts = evaluate_forecasts(
        forecasts=forecasts,
        realizations=realizations,
        config=load_config(project_root / "config.example.toml"),
    )
    gaps = artifacts.coverage_tests.loc[
        artifacts.coverage_tests["model_id"].eq("ewma")
        & artifacts.coverage_tests["metric"].eq("excluded_gap_transition_count")
    ]
    assert gaps["value"].eq(1).all()


def test_evaluation_aligns_assets_before_pooled_bootstrap(project_root: Path) -> None:
    forecasts, realizations = _evaluation_inputs()
    realizations["simple_return"] = 0.0
    realizations["squared_return"] = 0.0
    realizations["loss"] = 0.0
    realizations["period"] = "test"
    other_forecasts = forecasts.assign(series_id="QQQ")
    shift = np.array([-0.01, 0.01, -0.02, 0.02])
    forecasts.loc[forecasts.model_id.eq("ewma"), "return_quantile_0_05"] += shift
    other_forecasts.loc[
        other_forecasts.model_id.eq("ewma"), "return_quantile_0_05"
    ] -= shift
    config = load_config(project_root / "config.example.toml")
    config = replace(
        config, evaluation=replace(config.evaluation, bootstrap_block_length=1)
    )
    artifacts = evaluate_forecasts(
        forecasts=pd.concat([other_forecasts.iloc[::-1], forecasts], ignore_index=True),
        realizations=pd.concat([realizations, realizations.assign(series_id="QQQ")]),
        config=config,
    )
    pooled = artifacts.bootstrap_comparisons.loc[
        artifacts.bootstrap_comparisons.series_id.eq("ALL")
        & artifacts.bootstrap_comparisons.model_id.eq("ewma")
        & artifacts.bootstrap_comparisons.metric.eq("pinball_loss_0_05")
        & artifacts.bootstrap_comparisons.period.eq("test")
    ].iloc[0]
    assert pooled["paired_count"] == 8
    assert pooled["interval_lower"] == pytest.approx(pooled["value"])
    assert pooled["interval_upper"] == pytest.approx(pooled["value"])


def _evaluation_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(["2019-12-27", "2019-12-30", "2020-01-02", "2020-01-03"])
    origins = dates - pd.offsets.BDay(1)
    returns = np.array([-0.01, 0.005, -0.02, 0.01])
    realizations = pd.DataFrame(
        {
            "series_id": "SPY",
            "forecast_origin": origins,
            "target_date": dates,
            "simple_return": returns,
            "squared_return": returns**2,
            "loss": -returns,
            "period": ["validation", "validation", "test", "test"],
        }
    )
    model_ids = (
        "historical_variance",
        "historical_simulation",
        "ewma",
        "garch_1_1_gaussian",
        "garch_1_1_student_t",
    )
    rows: list[dict[str, object]] = []
    for model_id in model_ids:
        for position, (origin, target) in enumerate(zip(origins, dates, strict=True)):
            failed = model_id == "garch_1_1_student_t" and position == 2
            has_variance = model_id != "historical_simulation"
            has_quantile = model_id != "historical_variance"
            rows.append(
                {
                    "experiment_id": "risk-test",
                    "series_id": "SPY",
                    "model_id": model_id,
                    "forecast_origin": origin,
                    "target_date": target,
                    "variance": (0.0004 if has_variance and not failed else None),
                    "return_quantile_0_05": (
                        -0.03 if has_quantile and not failed else None
                    ),
                    "var_0_95": 0.03 if has_quantile and not failed else None,
                    "return_quantile_0_01": (
                        -0.05 if has_quantile and not failed else None
                    ),
                    "var_0_99": 0.05 if has_quantile and not failed else None,
                    "status": "failed" if failed else "ok",
                }
            )
    return pd.DataFrame(rows), realizations


def test_evaluation_tables_reconcile_and_keep_periods_separate(
    project_root: Path,
) -> None:
    forecasts, realizations = _evaluation_inputs()
    config = load_config(project_root / "config.example.toml")

    artifacts = evaluate_forecasts(
        forecasts=forecasts,
        realizations=realizations,
        config=config,
    )

    assert tuple(artifacts.forecast_availability.columns) == (
        FORECAST_AVAILABILITY_COLUMNS
    )
    assert tuple(artifacts.variance_scores.columns) == VARIANCE_SCORE_COLUMNS
    assert tuple(artifacts.quantile_scores.columns) == QUANTILE_SCORE_COLUMNS
    assert tuple(artifacts.coverage_tests.columns) == COVERAGE_TEST_COLUMNS
    assert tuple(artifacts.bootstrap_comparisons.columns) == (
        BOOTSTRAP_COMPARISON_COLUMNS
    )
    assert tuple(artifacts.period_breakdowns.columns) == PERIOD_BREAKDOWN_COLUMNS
    assert set(artifacts.variance_scores["period"]) == {"validation", "test"}
    assert set(artifacts.quantile_scores["period"]) == {"validation", "test"}
    assert set(artifacts.period_breakdowns["period"]) == {"test_2020"}

    failed = artifacts.forecast_availability.loc[
        artifacts.forecast_availability["metric"].eq("failed_forecast_count")
        & artifacts.forecast_availability["period"].eq("test")
        & artifacts.forecast_availability["series_id"].eq("SPY")
        & artifacts.forecast_availability["model_id"].eq("garch_1_1_student_t")
    ]
    assert failed["value"].item() == 1.0

    comparison = artifacts.bootstrap_comparisons.loc[
        artifacts.bootstrap_comparisons["period"].eq("test")
        & artifacts.bootstrap_comparisons["series_id"].eq("SPY")
        & artifacts.bootstrap_comparisons["model_id"].eq("garch_1_1_student_t")
        & artifacts.bootstrap_comparisons["metric"].eq("qlike")
    ]
    assert comparison["paired_count"].item() == 1
