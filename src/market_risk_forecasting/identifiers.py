"""Canonical JSON and deterministic experiment identifiers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_risk_forecasting.errors import ConfigInvalidError


def _canonicalize(value: Any) -> Any:
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ConfigInvalidError("Identifier material contains a non-finite number.")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ConfigInvalidError(
        f"Unsupported identifier material type: {type(value).__name__}."
    )


def canonical_json(material: Mapping[str, Any]) -> str:
    """Serialize declared material with sorted keys and no insignificant space."""
    normalized = _canonicalize(material)
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def deterministic_id(prefix: str, material: Mapping[str, Any]) -> str:
    """Return ``prefix`` plus the first 24 hex characters of a SHA-256 hash."""
    if not prefix:
        raise ConfigInvalidError("Identifier prefix must be non-empty.")
    digest = hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:24]}"


def make_fit_id(
    *,
    experiment_id: str,
    series_id: str,
    model_id: str,
    fit_origin: pd.Timestamp,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    upstream_simple_return_checksum: str,
    package_version: str,
) -> str:
    """Create a fit identifier from exactly the declared material."""
    return deterministic_id(
        "fit_",
        {
            "experiment_id": experiment_id,
            "series_id": series_id,
            "model_id": model_id,
            "fit_origin": fit_origin,
            "train_start": train_start,
            "train_end": train_end,
            "upstream_simple_return_checksum": upstream_simple_return_checksum,
            "forecasting_package_version": package_version,
        },
    )


def make_forecast_id(
    *,
    experiment_id: str,
    fit_id: str,
    series_id: str,
    model_id: str,
    forecast_origin: pd.Timestamp,
    target_date: pd.Timestamp,
) -> str:
    """Create a forecast identifier from exactly the declared material."""
    return deterministic_id(
        "fcst_",
        {
            "experiment_id": experiment_id,
            "fit_id": fit_id,
            "series_id": series_id,
            "model_id": model_id,
            "forecast_origin": forecast_origin,
            "target_date": target_date,
        },
    )


__all__ = [
    "canonical_json",
    "deterministic_id",
    "make_fit_id",
    "make_forecast_id",
]
