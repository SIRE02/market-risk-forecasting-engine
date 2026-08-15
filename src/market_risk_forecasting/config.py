"""Strict TOML configuration contracts for forecasting experiments."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast

from market_risk_forecasting.errors import ConfigInvalidError

type QuantileMethod = Literal["linear", "lower", "higher", "nearest", "midpoint"]

_TOP_LEVEL_KEYS = {
    "experiment",
    "upstream",
    "periods",
    "portfolio_proxy",
    "historical",
    "ewma",
    "garch",
    "evaluation",
}


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    input_run_dir: Path
    output_dir: Path
    random_seed: int


@dataclass(frozen=True)
class UpstreamConfig:
    project: str
    package_version: str
    simple_returns_schema_id: str
    simple_returns_schema_version: str
    simple_returns_units: str
    instruments: tuple[str, ...]


@dataclass(frozen=True)
class PeriodConfig:
    development_start: date
    development_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date


@dataclass(frozen=True)
class PortfolioProxyConfig:
    enabled: bool
    series_id: str | None
    weight_items: tuple[tuple[str, float], ...]

    @property
    def weights(self) -> Mapping[str, float]:
        return dict(self.weight_items)


@dataclass(frozen=True)
class HistoricalConfig:
    variance_window: int
    var_window: int
    quantile_method: QuantileMethod


@dataclass(frozen=True)
class EwmaConfig:
    lambda_: float
    initialization_window: int


@dataclass(frozen=True)
class GarchConfig:
    estimation_window: int
    refit_every_origins: int
    input_scale: float
    retry_count: int
    stationarity_tolerance: float


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_block_length: int
    bootstrap_resamples: int
    bootstrap_confidence: float


@dataclass(frozen=True)
class ForecastConfig:
    experiment: ExperimentConfig
    upstream: UpstreamConfig
    periods: PeriodConfig
    portfolio_proxy: PortfolioProxyConfig
    historical: HistoricalConfig
    ewma: EwmaConfig
    garch: GarchConfig
    evaluation: EvaluationConfig

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible effective configuration."""
        raw = asdict(self)
        raw["experiment"]["input_run_dir"] = str(self.experiment.input_run_dir)
        raw["experiment"]["output_dir"] = str(self.experiment.output_dir)
        raw["periods"] = {
            key: value.isoformat() for key, value in asdict(self.periods).items()
        }
        raw["ewma"]["lambda"] = raw["ewma"].pop("lambda_")
        raw["upstream"]["instruments"] = list(self.upstream.instruments)
        raw["portfolio_proxy"] = {"enabled": self.portfolio_proxy.enabled}
        if self.portfolio_proxy.enabled:
            raw["portfolio_proxy"].update(
                {
                    "series_id": self.portfolio_proxy.series_id,
                    "weights": dict(self.portfolio_proxy.weights),
                }
            )
        return raw


def _fail(message: str) -> ConfigInvalidError:
    return ConfigInvalidError(message)


def _validate_keys(
    raw: Mapping[str, Any],
    *,
    context: str,
    expected: set[str],
) -> None:
    actual = set(raw)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise _fail(f"{context} is missing required key(s): {', '.join(missing)}.")
    if unknown:
        raise _fail(f"{context} contains unknown key(s): {', '.join(unknown)}.")


def _section(
    raw: Mapping[str, Any], name: str, expected: set[str]
) -> Mapping[str, Any]:
    value = raw[name]
    if not isinstance(value, Mapping):
        raise _fail(f"[{name}] must be a TOML table.")
    _validate_keys(value, context=f"[{name}]", expected=expected)
    return value


def _string(raw: Mapping[str, Any], key: str, context: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise _fail(f"{context}.{key} must be a non-empty string.")
    return value.strip()


def _integer(
    raw: Mapping[str, Any],
    key: str,
    context: str,
    *,
    minimum: int = 0,
) -> int:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _fail(f"{context}.{key} must be an integer >= {minimum}.")
    return value


def _number(
    raw: Mapping[str, Any],
    key: str,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
    strict_maximum: bool = False,
) -> float:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(f"{context}.{key} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise _fail(f"{context}.{key} must be finite.")
    if minimum is not None and (
        result < minimum or (strict_minimum and result == minimum)
    ):
        operator = ">" if strict_minimum else ">="
        raise _fail(f"{context}.{key} must be {operator} {minimum}.")
    if maximum is not None and (
        result > maximum or (strict_maximum and result == maximum)
    ):
        operator = "<" if strict_maximum else "<="
        raise _fail(f"{context}.{key} must be {operator} {maximum}.")
    return result


def _date(raw: Mapping[str, Any], key: str, context: str) -> date:
    value = _string(raw, key, context)
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise _fail(f"{context}.{key} must use YYYY-MM-DD format.") from exc


def _string_sequence(raw: Mapping[str, Any], key: str, context: str) -> tuple[str, ...]:
    value = raw[key]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _fail(f"{context}.{key} must be an array of strings.")
    if not value or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise _fail(f"{context}.{key} must be a non-empty array of strings.")
    result = tuple(item.strip().upper() for item in value)
    if len(set(result)) != len(result):
        raise _fail(f"{context}.{key} must not contain duplicate strings.")
    return result


def _boolean(raw: Mapping[str, Any], key: str, context: str) -> bool:
    value = raw[key]
    if not isinstance(value, bool):
        raise _fail(f"{context}.{key} must be boolean.")
    return value


def _portfolio_proxy(
    raw: Mapping[str, Any],
    *,
    instruments: tuple[str, ...],
) -> PortfolioProxyConfig:
    allowed = {"enabled", "series_id", "weights"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise _fail(
            "[portfolio_proxy] contains unknown key(s): " + ", ".join(unknown) + "."
        )
    if "enabled" not in raw:
        raise _fail("[portfolio_proxy] is missing required key(s): enabled.")
    enabled = _boolean(raw, "enabled", "portfolio_proxy")
    if not enabled:
        unexpected = sorted(set(raw) - {"enabled"})
        if unexpected:
            raise _fail(
                "Disabled [portfolio_proxy] must not define "
                + ", ".join(unexpected)
                + "."
            )
        return PortfolioProxyConfig(False, None, ())
    missing = sorted({"series_id", "weights"} - set(raw))
    if missing:
        raise _fail(
            "[portfolio_proxy] is missing required key(s): " + ", ".join(missing) + "."
        )
    series_id = _string(raw, "series_id", "portfolio_proxy")
    weights_raw = raw["weights"]
    if not isinstance(weights_raw, Mapping):
        raise _fail("portfolio_proxy.weights must be a TOML table.")
    if set(weights_raw) != set(instruments):
        raise _fail(
            "portfolio_proxy.weights must contain exactly the configured "
            "upstream instruments."
        )
    weights = {
        ticker: _number(
            weights_raw,
            ticker,
            "portfolio_proxy.weights",
            minimum=0.0,
        )
        for ticker in instruments
    }
    if series_id in instruments:
        raise _fail(
            "portfolio_proxy.series_id must not duplicate an upstream instrument."
        )

    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12):
        raise _fail("Portfolio proxy weights must sum to one.")
    return PortfolioProxyConfig(
        enabled=True,
        series_id=series_id,
        weight_items=tuple(weights.items()),
    )


def _build_config(raw: Mapping[str, Any]) -> ForecastConfig:
    _validate_keys(raw, context="configuration", expected=_TOP_LEVEL_KEYS)

    experiment_raw = _section(
        raw,
        "experiment",
        {
            "experiment_id",
            "input_run_dir",
            "output_dir",
            "random_seed",
        },
    )
    experiment = ExperimentConfig(
        experiment_id=_string(experiment_raw, "experiment_id", "experiment"),
        input_run_dir=Path(_string(experiment_raw, "input_run_dir", "experiment")),
        output_dir=Path(_string(experiment_raw, "output_dir", "experiment")),
        random_seed=_integer(experiment_raw, "random_seed", "experiment"),
    )
    if experiment.input_run_dir == experiment.output_dir:
        raise _fail("experiment.input_run_dir and output_dir must differ.")

    upstream_raw = _section(
        raw,
        "upstream",
        {
            "project",
            "package_version",
            "simple_returns_schema_id",
            "simple_returns_schema_version",
            "simple_returns_units",
            "instruments",
        },
    )
    upstream = UpstreamConfig(
        project=_string(upstream_raw, "project", "upstream"),
        package_version=_string(upstream_raw, "package_version", "upstream"),
        simple_returns_schema_id=_string(
            upstream_raw, "simple_returns_schema_id", "upstream"
        ),
        simple_returns_schema_version=_string(
            upstream_raw, "simple_returns_schema_version", "upstream"
        ),
        simple_returns_units=_string(upstream_raw, "simple_returns_units", "upstream"),
        instruments=_string_sequence(upstream_raw, "instruments", "upstream"),
    )
    expected_upstream_identity = (
        "historical-asset-risk-engine",
        "0.1.1",
        "historical-asset-risk/simple-returns",
        "1.experimental",
        "decimal_return_per_observation",
    )
    actual_upstream_identity = (
        upstream.project,
        upstream.package_version,
        upstream.simple_returns_schema_id,
        upstream.simple_returns_schema_version,
        upstream.simple_returns_units,
    )
    if actual_upstream_identity != expected_upstream_identity:
        raise _fail(
            "The [upstream] project and artifact identity must match the "
            "historical-asset-risk v0.1 public contract."
        )
    periods_raw = _section(
        raw,
        "periods",
        {
            "development_start",
            "development_end",
            "validation_start",
            "validation_end",
            "test_start",
            "test_end",
        },
    )
    periods = PeriodConfig(
        development_start=_date(periods_raw, "development_start", "periods"),
        development_end=_date(periods_raw, "development_end", "periods"),
        validation_start=_date(periods_raw, "validation_start", "periods"),
        validation_end=_date(periods_raw, "validation_end", "periods"),
        test_start=_date(periods_raw, "test_start", "periods"),
        test_end=_date(periods_raw, "test_end", "periods"),
    )
    boundaries = (
        periods.development_start,
        periods.development_end,
        periods.validation_start,
        periods.validation_end,
        periods.test_start,
        periods.test_end,
    )
    if any(
        left >= right for left, right in zip(boundaries, boundaries[1:], strict=False)
    ):
        raise _fail("Period boundaries must be strictly increasing.")

    portfolio_value = raw["portfolio_proxy"]
    if not isinstance(portfolio_value, Mapping):
        raise _fail("[portfolio_proxy] must be a TOML table.")
    portfolio = _portfolio_proxy(
        portfolio_value,
        instruments=upstream.instruments,
    )

    historical_raw = _section(
        raw, "historical", {"variance_window", "var_window", "quantile_method"}
    )
    historical = HistoricalConfig(
        variance_window=_integer(
            historical_raw, "variance_window", "historical", minimum=2
        ),
        var_window=_integer(historical_raw, "var_window", "historical", minimum=2),
        quantile_method=cast(
            QuantileMethod,
            _string(historical_raw, "quantile_method", "historical"),
        ),
    )
    if historical.quantile_method not in {
        "linear",
        "lower",
        "higher",
        "nearest",
        "midpoint",
    }:
        raise _fail(
            "historical.quantile_method must be one of linear, lower, higher, "
            "nearest, or midpoint."
        )

    ewma_raw = _section(raw, "ewma", {"lambda", "initialization_window"})
    ewma = EwmaConfig(
        lambda_=_number(
            ewma_raw,
            "lambda",
            "ewma",
            minimum=0.0,
            maximum=1.0,
            strict_minimum=True,
            strict_maximum=True,
        ),
        initialization_window=_integer(
            ewma_raw, "initialization_window", "ewma", minimum=2
        ),
    )
    garch_raw = _section(
        raw,
        "garch",
        {
            "estimation_window",
            "refit_every_origins",
            "input_scale",
            "retry_count",
            "stationarity_tolerance",
        },
    )
    garch = GarchConfig(
        estimation_window=_integer(garch_raw, "estimation_window", "garch", minimum=2),
        refit_every_origins=_integer(
            garch_raw, "refit_every_origins", "garch", minimum=1
        ),
        input_scale=_number(
            garch_raw,
            "input_scale",
            "garch",
            minimum=0.0,
            strict_minimum=True,
        ),
        retry_count=_integer(garch_raw, "retry_count", "garch"),
        stationarity_tolerance=_number(
            garch_raw, "stationarity_tolerance", "garch", minimum=0.0
        ),
    )
    if garch.retry_count > 1:
        raise _fail("garch.retry_count must be zero or one.")
    if garch.stationarity_tolerance >= 1.0:
        raise _fail("garch.stationarity_tolerance must be less than one.")

    evaluation_raw = _section(
        raw,
        "evaluation",
        {
            "bootstrap_block_length",
            "bootstrap_resamples",
            "bootstrap_confidence",
        },
    )
    evaluation = EvaluationConfig(
        bootstrap_block_length=_integer(
            evaluation_raw,
            "bootstrap_block_length",
            "evaluation",
            minimum=1,
        ),
        bootstrap_resamples=_integer(
            evaluation_raw, "bootstrap_resamples", "evaluation", minimum=1
        ),
        bootstrap_confidence=_number(
            evaluation_raw,
            "bootstrap_confidence",
            "evaluation",
            minimum=0.0,
            maximum=1.0,
            strict_minimum=True,
            strict_maximum=True,
        ),
    )
    return ForecastConfig(
        experiment=experiment,
        upstream=upstream,
        periods=periods,
        portfolio_proxy=portfolio,
        historical=historical,
        ewma=ewma,
        garch=garch,
        evaluation=evaluation,
    )


def load_config(path: str | Path) -> ForecastConfig:
    """Load and validate one strict TOML configuration file."""
    config_path = Path(path)
    try:
        with config_path.open("rb") as stream:
            raw = tomllib.load(stream)
    except FileNotFoundError as exc:
        raise _fail(f"Configuration file does not exist: {config_path}.") from exc
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise _fail(f"Could not parse configuration {config_path}: {exc}") from exc
    return _build_config(raw)


__all__ = [
    "EvaluationConfig",
    "ExperimentConfig",
    "ForecastConfig",
    "GarchConfig",
    "HistoricalConfig",
    "PeriodConfig",
    "PortfolioProxyConfig",
    "UpstreamConfig",
    "load_config",
]
