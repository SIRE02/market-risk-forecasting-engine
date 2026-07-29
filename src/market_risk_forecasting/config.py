"""Strict TOML configuration contract for v0.1 experiments."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from market_risk_forecasting.errors import ConfigInvalidError

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
_REQUIRED_INSTRUMENTS = ("SPY", "IEF", "GLD")


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
    series_id: str
    SPY: float
    IEF: float
    GLD: float

    @property
    def weights(self) -> Mapping[str, float]:
        return {"SPY": self.SPY, "IEF": self.IEF, "GLD": self.GLD}


@dataclass(frozen=True)
class HistoricalConfig:
    variance_window: int
    var_window: int
    quantile_method: str


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
    var_confidence_levels: tuple[float, ...]
    primary_var_confidence: float
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
        raw["evaluation"]["var_confidence_levels"] = list(
            self.evaluation.var_confidence_levels
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
    result = tuple(value)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise _fail(f"{context}.{key} must be a non-empty array of strings.")
    return result


def _number_sequence(
    raw: Mapping[str, Any], key: str, context: str
) -> tuple[float, ...]:
    value = raw[key]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _fail(f"{context}.{key} must be an array of numbers.")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise _fail(f"{context}.{key} must be an array of finite numbers.")
        numeric = float(item)
        if not math.isfinite(numeric):
            raise _fail(f"{context}.{key} must be an array of finite numbers.")
        result.append(numeric)
    if not result:
        raise _fail(f"{context}.{key} must not be empty.")
    return tuple(result)


def _build_config(raw: Mapping[str, Any]) -> ForecastConfig:
    _validate_keys(raw, context="configuration", expected=_TOP_LEVEL_KEYS)

    experiment_raw = _section(
        raw,
        "experiment",
        {"experiment_id", "input_run_dir", "output_dir", "random_seed"},
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
    expected_upstream = (
        "historical-asset-risk-engine",
        "0.1.0",
        "historical-asset-risk/simple-returns",
        "1.experimental",
        "decimal_return_per_observation",
        _REQUIRED_INSTRUMENTS,
    )
    actual_upstream = (
        upstream.project,
        upstream.package_version,
        upstream.simple_returns_schema_id,
        upstream.simple_returns_schema_version,
        upstream.simple_returns_units,
        upstream.instruments,
    )
    if actual_upstream != expected_upstream:
        raise _fail("The [upstream] identity must match the v0.1 public contract.")

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

    portfolio_raw = _section(raw, "portfolio_proxy", {"series_id", "SPY", "IEF", "GLD"})
    portfolio = PortfolioProxyConfig(
        series_id=_string(portfolio_raw, "series_id", "portfolio_proxy"),
        SPY=_number(portfolio_raw, "SPY", "portfolio_proxy", minimum=0.0),
        IEF=_number(portfolio_raw, "IEF", "portfolio_proxy", minimum=0.0),
        GLD=_number(portfolio_raw, "GLD", "portfolio_proxy", minimum=0.0),
    )
    if portfolio.series_id != "MIX_60_30_10":
        raise _fail("portfolio_proxy.series_id must be MIX_60_30_10.")
    if not math.isclose(sum(portfolio.weights.values()), 1.0, abs_tol=1e-12):
        raise _fail("Portfolio proxy weights must sum to one.")

    historical_raw = _section(
        raw, "historical", {"variance_window", "var_window", "quantile_method"}
    )
    historical = HistoricalConfig(
        variance_window=_integer(
            historical_raw, "variance_window", "historical", minimum=2
        ),
        var_window=_integer(historical_raw, "var_window", "historical", minimum=2),
        quantile_method=_string(historical_raw, "quantile_method", "historical"),
    )
    if historical.quantile_method != "linear":
        raise _fail("historical.quantile_method must be linear.")
    if historical.variance_window != 252 or historical.var_window != 500:
        raise _fail("v0.1 requires historical.variance_window=252 and var_window=500.")

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
    if ewma.lambda_ != 0.94 or ewma.initialization_window != 252:
        raise _fail("v0.1 requires ewma.lambda=0.94 and initialization_window=252.")

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
    if (
        garch.estimation_window != 1250
        or garch.refit_every_origins != 20
        or garch.input_scale != 100.0
        or garch.retry_count != 1
        or garch.stationarity_tolerance != 1e-8
    ):
        raise _fail(
            "v0.1 requires garch.estimation_window=1250, "
            "refit_every_origins=20, input_scale=100.0, retry_count=1, "
            "and stationarity_tolerance=1e-8."
        )

    evaluation_raw = _section(
        raw,
        "evaluation",
        {
            "var_confidence_levels",
            "primary_var_confidence",
            "bootstrap_block_length",
            "bootstrap_resamples",
            "bootstrap_confidence",
        },
    )
    confidence_levels = _number_sequence(
        evaluation_raw, "var_confidence_levels", "evaluation"
    )
    if any(level <= 0.0 or level >= 1.0 for level in confidence_levels):
        raise _fail("evaluation.var_confidence_levels must lie strictly in (0, 1).")
    primary_confidence = _number(
        evaluation_raw,
        "primary_var_confidence",
        "evaluation",
        minimum=0.0,
        maximum=1.0,
        strict_minimum=True,
        strict_maximum=True,
    )
    evaluation = EvaluationConfig(
        var_confidence_levels=confidence_levels,
        primary_var_confidence=primary_confidence,
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
    if primary_confidence not in confidence_levels:
        raise _fail(
            "evaluation.primary_var_confidence must be listed in var_confidence_levels."
        )
    if (
        confidence_levels != (0.95, 0.99)
        or primary_confidence != 0.95
        or evaluation.bootstrap_block_length != 20
        or evaluation.bootstrap_resamples != 2000
        or evaluation.bootstrap_confidence != 0.95
        or experiment.random_seed != 42
    ):
        raise _fail(
            "v0.1 requires evaluation levels [0.95, 0.99], primary level "
            "0.95, bootstrap block length 20, 2000 resamples, confidence "
            "0.95, and random seed 42."
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
