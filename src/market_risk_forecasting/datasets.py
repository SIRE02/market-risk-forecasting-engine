"""Canonical v0.1 research-series construction and lineage manifest."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from market_risk_forecasting.config import PortfolioProxyConfig
from market_risk_forecasting.errors import InputValueInvalidError
from market_risk_forecasting.upstream import (
    UpstreamRun,
    quality_adjustment_counts,
)

_INDIVIDUAL_SERIES = ("SPY", "IEF", "GLD")
_DATASET_SCHEMA_ID = "market-risk-forecasting/research-series"
_DATASET_SCHEMA_VERSION = "1.experimental"
_DATASET_UNITS = "decimal_simple_return_per_observation"


@dataclass(frozen=True)
class ResearchDataset:
    """The four canonical series and their deterministic lineage manifest."""

    returns: pd.DataFrame
    manifest: Mapping[str, Any]

    @property
    def series_order(self) -> tuple[str, ...]:
        return tuple(str(column) for column in self.returns.columns)


def construct_research_series(
    upstream_returns: pd.DataFrame,
    proxy: PortfolioProxyConfig,
) -> pd.DataFrame:
    """Construct the three unchanged ETF series and the constant-weight proxy."""
    actual = tuple(str(column) for column in upstream_returns.columns)
    if actual != _INDIVIDUAL_SERIES:
        raise InputValueInvalidError(
            "Research-series input columns must be ordered SPY, IEF, GLD."
        )
    weights = proxy.weights
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12):
        raise InputValueInvalidError("Portfolio proxy weights must sum to one.")

    result = upstream_returns.loc[:, list(_INDIVIDUAL_SERIES)].astype(float, copy=True)
    result[proxy.series_id] = sum(
        result[ticker] * weight for ticker, weight in weights.items()
    )
    if result.isna().any(axis=None):
        raise InputValueInvalidError(
            "Constructed research series contain missing data."
        )
    return result


def build_dataset_manifest(
    upstream: UpstreamRun,
    proxy: PortfolioProxyConfig,
    research_returns: pd.DataFrame,
) -> Mapping[str, Any]:
    """Build deterministic dataset evidence without writing it."""
    data_source = upstream.manifest.get("data_source")
    if not isinstance(data_source, Mapping):
        data_source = {}
    git_commit = upstream.manifest.get("git_commit")
    if not isinstance(git_commit, str):
        git_commit = None

    return {
        "schema_id": "market-risk-forecasting/dataset-manifest",
        "schema_version": "1.experimental",
        "dataset": {
            "schema_id": _DATASET_SCHEMA_ID,
            "schema_version": _DATASET_SCHEMA_VERSION,
            "units": _DATASET_UNITS,
            "observation_count": len(research_returns),
            "first_date": pd.Timestamp(research_returns.index[0]).date().isoformat(),
            "last_date": pd.Timestamp(research_returns.index[-1]).date().isoformat(),
            "series_order": [str(column) for column in research_returns.columns],
        },
        "upstream": {
            "project": upstream.manifest.get("project"),
            "project_version": upstream.manifest.get("project_version"),
            "installed_package_version": upstream.installed_package_version,
            "git_commit": git_commit,
            "instrument_order": list(upstream.instrument_order),
            "actual_start_date": upstream.first_date.date().isoformat(),
            "actual_end_date": upstream.last_date.date().isoformat(),
            "provider": data_source.get("provider"),
            "consumed_file_checksums": dict(upstream.checksums),
            "quality_adjustment_counts": dict(
                quality_adjustment_counts(upstream.quality_report)
            ),
        },
        "portfolio_proxy": {
            "series_id": proxy.series_id,
            "weights": dict(proxy.weights),
            "weight_sum": sum(proxy.weights.values()),
            "interpretation": "daily_constant_weight_return_proxy",
            "rebalanced_each_observation": True,
            "transaction_costs": False,
            "holdings_ledger": False,
        },
    }


def build_research_dataset(
    upstream: UpstreamRun,
    proxy: PortfolioProxyConfig,
) -> ResearchDataset:
    """Construct all research series and their in-memory manifest."""
    returns = construct_research_series(upstream.returns, proxy)
    return ResearchDataset(
        returns=returns,
        manifest=build_dataset_manifest(upstream, proxy, returns),
    )


def persist_dataset_manifest(
    dataset: ResearchDataset,
    path: Path,
) -> None:
    """Persist deterministic dataset evidence for experiment orchestration."""
    Path(path).write_text(
        json.dumps(
            dataset.manifest,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ResearchDataset",
    "build_dataset_manifest",
    "build_research_dataset",
    "construct_research_series",
    "persist_dataset_manifest",
]
