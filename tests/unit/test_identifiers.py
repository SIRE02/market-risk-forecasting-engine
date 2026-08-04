from __future__ import annotations

import math

import pandas as pd
import pytest

from market_risk_forecasting.errors import ConfigInvalidError
from market_risk_forecasting.identifiers import (
    canonical_json,
    deterministic_id,
    make_fit_id,
    make_forecast_id,
)


def test_canonical_json_is_order_independent_and_compact() -> None:
    left = canonical_json({"b": 2, "a": {"date": pd.Timestamp("2020-01-02")}})
    right = canonical_json({"a": {"date": pd.Timestamp("2020-01-02")}, "b": 2})

    assert left == right == '{"a":{"date":"2020-01-02T00:00:00"},"b":2}'


def test_identifier_has_declared_prefix_and_24_hex_characters() -> None:
    identifier = deterministic_id("fit_", {"series": "SPY", "origin": "2020-01-02"})

    assert identifier.startswith("fit_")
    assert len(identifier) == len("fit_") + 24
    assert all(character in "0123456789abcdef" for character in identifier[4:])


def test_identifier_rejects_nonfinite_material() -> None:
    with pytest.raises(ConfigInvalidError, match="non-finite"):
        deterministic_id("fit_", {"variance": math.inf})


def test_fit_identifier_ignores_undeclared_runtime_data() -> None:
    arguments = {
        "experiment_id": "risk-test",
        "series_id": "SPY",
        "model_id": "garch_1_1_gaussian",
        "fit_origin": pd.Timestamp("2020-01-02"),
        "train_start": pd.Timestamp("2015-01-05"),
        "train_end": pd.Timestamp("2020-01-02"),
        "upstream_simple_return_checksum": "a" * 64,
        "package_version": "0.1.0",
    }

    assert make_fit_id(**arguments) == make_fit_id(**dict(reversed(arguments.items())))


def test_forecast_identifier_changes_with_target() -> None:
    common = {
        "experiment_id": "risk-test",
        "fit_id": "fit_" + "a" * 24,
        "series_id": "SPY",
        "model_id": "ewma",
        "forecast_origin": pd.Timestamp("2020-01-02"),
    }

    first = make_forecast_id(**common, target_date=pd.Timestamp("2020-01-03"))
    second = make_forecast_id(**common, target_date=pd.Timestamp("2020-01-06"))

    assert first != second
