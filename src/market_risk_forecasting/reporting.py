"""Saved-artifact-only research report and deterministic figure generation."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from market_risk_forecasting.errors import (
    ArtifactReconciliationFailedError,
    OutputCollisionError,
)
from market_risk_forecasting.experiment import (
    NUMERICAL_ARTIFACT_NAMES,
    artifact_sha256,
    finalize_run_manifest,
    verify_experiment_directory,
)
from market_risk_forecasting.models.ewma import EWMA_MODEL_ID
from market_risk_forecasting.models.garch import (
    GAUSSIAN_GARCH_MODEL_ID,
    STUDENT_T_GARCH_MODEL_ID,
)
from market_risk_forecasting.models.historical import (
    HISTORICAL_SIMULATION_MODEL_ID,
    HISTORICAL_VARIANCE_MODEL_ID,
)

_MODEL_LABELS = {
    HISTORICAL_VARIANCE_MODEL_ID: "Historical variance",
    HISTORICAL_SIMULATION_MODEL_ID: "Historical simulation",
    EWMA_MODEL_ID: "EWMA",
    GAUSSIAN_GARCH_MODEL_ID: "Gaussian GARCH(1,1)",
    STUDENT_T_GARCH_MODEL_ID: "Student-t GARCH(1,1)",
}
_CANDIDATES = (
    EWMA_MODEL_ID,
    GAUSSIAN_GARCH_MODEL_ID,
    STUDENT_T_GARCH_MODEL_ID,
)
_VARIANCE_MODELS = (
    HISTORICAL_VARIANCE_MODEL_ID,
    *_CANDIDATES,
)
_QUANTILE_MODELS = (
    HISTORICAL_SIMULATION_MODEL_ID,
    *_CANDIDATES,
)
_MODEL_COLORS = {
    HISTORICAL_VARIANCE_MODEL_ID: "#7a7a7a",
    HISTORICAL_SIMULATION_MODEL_ID: "#8a5a9e",
    EWMA_MODEL_ID: "#0072b2",
    GAUSSIAN_GARCH_MODEL_ID: "#d55e00",
    STUDENT_T_GARCH_MODEL_ID: "#009e73",
}
_FORECAST_KEYS = ("series_id", "forecast_origin", "target_date")
_FIGURE_NAMES = (
    "forecast_volatility_history.png",
    "var_exception_history.png",
    "rolling_model_advantage.png",
    "variance_qlike_comparison.png",
    "var_pinball_comparison.png",
    "forecast_availability.png",
    "var_calibration.png",
    "series_comparisons.png",
)


@dataclass(frozen=True)
class ReportResult:
    experiment_dir: Path
    report_path: Path
    reused: bool


@dataclass(frozen=True)
class _ReportLineage:
    experiment_id: str
    execution_started_at: str
    numerical_git_commit: str
    numerical_source_state: str
    numerical_source_tree_sha256: str
    upstream_project_version: str
    upstream_git_commit: str
    upstream_provider: str
    upstream_acquired_at: str
    upstream_actual_start_date: str
    upstream_actual_end_date: str
    simple_returns_sha256: str


def generate_report(experiment_dir: Path) -> ReportResult:
    """Generate the report using only checksummed saved numerical artifacts."""
    directory = Path(experiment_dir).resolve()
    verify_experiment_directory(directory, require_complete=False)
    report_path = directory / "research_report.md"
    figure_dir = directory / "figures"
    report_outputs = [
        report_path,
        *[figure_dir / name for name in _FIGURE_NAMES],
    ]
    existing = [path for path in report_outputs if path.exists()]
    if existing:
        if len(existing) != len(report_outputs):
            raise OutputCollisionError(
                "Experiment contains a partial report output set."
            )
        verify_experiment_directory(directory, require_complete=True)
        return ReportResult(
            experiment_dir=directory,
            report_path=report_path,
            reused=True,
        )

    numerical_before = {
        name: artifact_sha256(directory / name) for name in NUMERICAL_ARTIFACT_NAMES
    }
    tables = _load_tables(directory)
    effective_configuration = _load_effective_configuration(directory)
    lineage = _load_report_lineage(directory)
    temporary = Path(tempfile.mkdtemp(prefix=".report-building-", dir=directory))
    try:
        temporary_figures = temporary / "figures"
        temporary_figures.mkdir()
        _plot_comparison(
            tables["bootstrap"],
            metric="qlike",
            title="Final-test QLIKE difference vs historical variance",
            scale=1.0,
            ylabel="Candidate loss minus benchmark loss",
            path=temporary_figures / "variance_qlike_comparison.png",
        )
        _plot_comparison(
            tables["bootstrap"],
            metric="pinball_loss_0_05",
            title="Final-test 95% VaR pinball difference vs historical simulation",
            scale=10_000.0,
            ylabel="Candidate minus benchmark loss (basis points)",
            path=temporary_figures / "var_pinball_comparison.png",
        )
        _plot_availability(
            tables["availability"],
            temporary_figures / "forecast_availability.png",
        )
        rolling_window = int(
            _configuration_number(
                _configuration_section(effective_configuration, "historical"),
                "variance_window",
            )
        )
        _plot_forecast_volatility_history(
            tables["forecasts"],
            tables["realizations"],
            temporary_figures / "forecast_volatility_history.png",
        )
        _plot_var_exception_history(
            tables["forecasts"],
            tables["realizations"],
            temporary_figures / "var_exception_history.png",
        )
        _plot_rolling_model_advantage(
            tables["forecasts"],
            tables["realizations"],
            window=rolling_window,
            path=temporary_figures / "rolling_model_advantage.png",
        )
        _plot_var_calibration(
            tables["coverage"],
            temporary_figures / "var_calibration.png",
        )
        _plot_series_comparisons(
            tables["bootstrap"],
            temporary_figures / "series_comparisons.png",
        )
        report = _render_report(tables, effective_configuration, lineage)
        (temporary / "research_report.md").write_text(
            report,
            encoding="utf-8",
        )
        numerical_after = {
            name: artifact_sha256(directory / name) for name in NUMERICAL_ARTIFACT_NAMES
        }
        if numerical_before != numerical_after:
            raise ArtifactReconciliationFailedError(
                "Report generation altered a numerical artifact."
            )
        figure_dir.mkdir()
        for name in _FIGURE_NAMES:
            shutil.move(str(temporary_figures / name), figure_dir / name)
        shutil.move(str(temporary / "research_report.md"), report_path)
        finalize_run_manifest(directory, state="complete")
        verify_experiment_directory(directory, require_complete=True)
        return ReportResult(
            experiment_dir=directory,
            report_path=report_path,
            reused=False,
        )
    finally:
        if temporary.exists() and temporary.parent == directory:
            shutil.rmtree(temporary)


def _load_tables(directory: Path) -> dict[str, pd.DataFrame]:
    tables = {
        "availability": pd.read_csv(directory / "forecast_availability.csv"),
        "variance_scores": pd.read_csv(directory / "variance_scores.csv"),
        "quantile_scores": pd.read_csv(directory / "quantile_scores.csv"),
        "coverage": pd.read_csv(directory / "coverage_tests.csv"),
        "bootstrap": pd.read_csv(directory / "bootstrap_comparisons.csv"),
        "periods": pd.read_csv(directory / "period_breakdowns.csv"),
        "diagnostics": pd.read_parquet(directory / "fit_diagnostics.parquet"),
        "forecasts": pd.read_parquet(directory / "forecasts.parquet"),
        "realizations": pd.read_parquet(directory / "realizations.parquet"),
    }
    required = {
        "availability": {"period", "series_id", "model_id", "metric", "value"},
        "coverage": {
            "period",
            "series_id",
            "model_id",
            "metric",
            "confidence_level",
            "value",
            "status",
        },
        "bootstrap": {
            "period",
            "series_id",
            "model_id",
            "metric",
            "value",
            "interval_lower",
            "interval_upper",
            "paired_count",
        },
        "diagnostics": {
            "series_id",
            "model_id",
            "converged",
            "retry_used",
        },
        "forecasts": {
            "model_id",
            "series_id",
            "forecast_origin",
            "target_date",
            "status",
            "error_code",
            "variance",
            "volatility",
            "return_quantile_0_05",
            "var_0_95",
        },
        "realizations": {
            "series_id",
            "forecast_origin",
            "target_date",
            "simple_return",
            "squared_return",
            "loss",
            "period",
        },
    }
    for name, columns in required.items():
        missing = columns - set(tables[name].columns)
        if missing:
            raise ArtifactReconciliationFailedError(
                f"Saved {name} table is missing columns: {', '.join(sorted(missing))}."
            )
    return tables


def _load_effective_configuration(directory: Path) -> dict[str, Any]:
    path = directory / "experiment_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactReconciliationFailedError(
            f"Could not load effective report configuration from {path}: {exc}"
        ) from exc
    effective = manifest.get("effective_configuration")
    if not isinstance(effective, dict):
        raise ArtifactReconciliationFailedError(
            "Experiment manifest is missing effective_configuration."
        )
    return effective


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactReconciliationFailedError(
            f"Could not load {label} from {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ArtifactReconciliationFailedError(f"Saved {label} must be an object.")
    return value


def _manifest_object(
    value: dict[str, Any],
    key: str,
    *,
    label: str,
) -> dict[str, Any]:
    selected = value.get(key)
    if not isinstance(selected, dict):
        raise ArtifactReconciliationFailedError(
            f"Saved {label} is missing the {key!r} object."
        )
    return selected


def _manifest_text(value: dict[str, Any], key: str) -> str:
    selected = value.get(key)
    return selected if isinstance(selected, str) and selected else "unavailable"


def _load_report_lineage(directory: Path) -> _ReportLineage:
    run_manifest = _load_json_object(
        directory / "run_manifest.json",
        label="run manifest",
    )
    upstream_manifest = _load_json_object(
        directory / "upstream_run_manifest.json",
        label="upstream run manifest",
    )
    source_identity = _manifest_object(
        run_manifest,
        "source_identity",
        label="run manifest",
    )
    upstream = _manifest_object(run_manifest, "upstream", label="run manifest")
    upstream_checksums = _manifest_object(
        upstream,
        "checksums",
        label="run manifest upstream declaration",
    )
    data_source = _manifest_object(
        upstream_manifest,
        "data_source",
        label="upstream run manifest",
    )
    dirty = source_identity.get("git_dirty")
    source_state = (
        "dirty (uncommitted changes present)"
        if dirty is True
        else "clean"
        if dirty is False
        else "unavailable"
    )
    return _ReportLineage(
        experiment_id=_manifest_text(run_manifest, "experiment_id"),
        execution_started_at=_manifest_text(run_manifest, "execution_started_at"),
        numerical_git_commit=_manifest_text(source_identity, "git_commit"),
        numerical_source_state=source_state,
        numerical_source_tree_sha256=_manifest_text(
            source_identity,
            "source_tree_sha256",
        ),
        upstream_project_version=_manifest_text(
            upstream_manifest,
            "project_version",
        ),
        upstream_git_commit=_manifest_text(upstream_manifest, "git_commit"),
        upstream_provider=_manifest_text(data_source, "provider"),
        upstream_acquired_at=_manifest_text(data_source, "acquired_at"),
        upstream_actual_start_date=_manifest_text(
            data_source,
            "actual_start_date",
        ),
        upstream_actual_end_date=_manifest_text(data_source, "actual_end_date"),
        simple_returns_sha256=_manifest_text(
            upstream_checksums,
            "simple_returns.csv",
        ),
    )


def _report_identity(effective: dict[str, Any]) -> tuple[str, str]:
    title = "# Market Risk Forecasting Engine Research Report"

    upstream = effective.get("upstream", {})
    instruments = upstream.get("instruments", []) if isinstance(upstream, dict) else []
    series = [str(value) for value in instruments]
    proxy = effective.get("portfolio_proxy", {})
    if isinstance(proxy, dict) and proxy.get("enabled", True):
        proxy_id = proxy.get("series_id")
        if isinstance(proxy_id, str) and proxy_id:
            series.append(proxy_id)
    return title, ", ".join(series)


def _render_report(
    tables: dict[str, pd.DataFrame],
    effective_configuration: dict[str, Any],
    lineage: _ReportLineage,
) -> str:
    bootstrap = tables["bootstrap"]
    availability = tables["availability"]
    coverage = tables["coverage"]
    diagnostics = tables["diagnostics"]
    forecasts = tables["forecasts"]
    qlike_test = _comparison_rows(bootstrap, "test", "qlike")
    pinball_test = _comparison_rows(
        bootstrap,
        "test",
        "pinball_loss_0_05",
    )
    qlike_validation = _comparison_rows(bootstrap, "validation", "qlike")
    pinball_validation = _comparison_rows(
        bootstrap,
        "validation",
        "pinball_loss_0_05",
    )
    historical = _configuration_section(effective_configuration, "historical")
    ewma = _configuration_section(effective_configuration, "ewma")
    garch = _configuration_section(effective_configuration, "garch")
    evaluation = _configuration_section(effective_configuration, "evaluation")
    experiment = _configuration_section(effective_configuration, "experiment")
    interval_label = _percentage_label(
        _configuration_number(evaluation, "bootstrap_confidence")
    )
    direct_answer = _direct_answer(
        qlike_test,
        pinball_test,
        coverage,
        interval_label,
    )
    availability_table = _availability_rows(availability)
    coverage_95_table = _coverage_rows(coverage, confidence_level=0.95)
    coverage_99_table = _coverage_rows(coverage, confidence_level=0.99)
    diagnostic_table = _diagnostic_rows(diagnostics, forecasts)
    title, series_text = _report_identity(effective_configuration)
    rolling_window = int(_configuration_number(historical, "variance_window"))
    portfolio_limitation = (
        "The portfolio series is a constant-weight return projection rather than a "
        "holdings ledger. "
        if isinstance(effective_configuration.get("portfolio_proxy"), dict)
        and effective_configuration["portfolio_proxy"].get("enabled", True)
        else ""
    )

    sections = [
        title,
        "",
        "## Direct answer",
        "",
        direct_answer,
        "",
        "This is historical pseudo-out-of-sample evidence, not live or "
        "prospective forecasting. Lower QLIKE and pinball loss are better.",
        "A lower VaR loss score and correct exception-rate calibration are "
        "different properties; neither result alone establishes regulatory or "
        "live-trading suitability.",
        "",
        "## Forecasts through time",
        "",
        "One-session variance cannot be observed directly. The gray points show "
        "the absolute next-session return as a noisy realized-volatility proxy; "
        "the black line is a 21-session rolling root-mean-square return shown only "
        "as smoother context. Colored lines are the one-session-ahead volatility "
        "forecasts, and gaps show unavailable or failed forecasts.",
        "Light background bands identify the development, validation, and "
        "final-test periods; the horizontal extent ends at the last observed "
        "target date.",
        "",
        "![Forecast volatility and realized-return history]"
        "(figures/forecast_volatility_history.png)",
        "",
        "## 95% VaR through time",
        "",
        "The gray line is the realized daily loss and the colored lines are the "
        "one-session-ahead 95% VaR forecasts. An outlined marker in a model's "
        "color identifies a strict exception for that model: realized loss was "
        "greater than its VaR forecast.",
        "The same period bands and exact observed date range are used here.",
        "",
        "![Realized loss, VaR forecasts, and exceptions]"
        "(figures/var_exception_history.png)",
        "",
        "## Rolling model advantage",
        "",
        f"Each line is the candidate's {rolling_window}-session rolling mean loss "
        "minus its historical benchmark loss, calculated on a balanced panel: "
        "every configured series must have valid candidate and benchmark "
        "forecasts on a target date. A missing series makes that date unavailable "
        "to the rolling window. "
        "Values below zero mean the candidate performed better over that window; "
        "values above zero mean the benchmark performed better.",
        "",
        "![Rolling candidate-minus-benchmark loss differences]"
        "(figures/rolling_model_advantage.png)",
        "",
        "## Final-test variance comparison",
        "",
        _comparison_markdown(qlike_test, interval_label),
        "",
        "![Final-test QLIKE comparison](figures/variance_qlike_comparison.png)",
        "",
        "## Final-test 95% VaR comparison",
        "",
        _comparison_markdown(pinball_test, interval_label),
        "",
        "![Final-test VaR pinball comparison](figures/var_pinball_comparison.png)",
        "",
        "Pinball differences are plotted in basis points of return for legibility; "
        "the table retains the unscaled loss units.",
        "",
        "## Final-test results by series",
        "",
        "The aggregate result can conceal heterogeneous asset-level effects. "
        "Points below zero favor the candidate; horizontal bars are the configured "
        f"{interval_label} moving-block bootstrap intervals. VaR pinball effects "
        "are shown in basis points.",
        "",
        "![Per-series final-test loss differences](figures/series_comparisons.png)",
        "",
        "## Validation results",
        "",
        "Variance QLIKE:",
        "",
        _comparison_markdown(qlike_validation, interval_label),
        "",
        "95% VaR pinball loss:",
        "",
        _comparison_markdown(pinball_validation, interval_label),
        "",
        "Validation and final-test aggregates are kept separate; validation "
        "results are not pooled into the final-test claims.",
        "",
        "## Forecast availability",
        "",
        _markdown_table(
            (
                "Model",
                "Eligible",
                "Valid",
                "Failed",
                "Availability",
            ),
            availability_table,
        ),
        "",
        "![Final-test forecast availability](figures/forecast_availability.png)",
        "",
        "Failed dates remain in availability counts and are excluded only from "
        "explicit pairwise common-date score comparisons.",
        "",
        "## VaR calibration by series",
        "",
        "Observed final-test exception rates are shown for both reported VaR "
        "levels. Each point is one series; dashed lines show the nominal 5% and "
        "1% exception rates.",
        "",
        "![Final-test VaR calibration](figures/var_calibration.png)",
        "",
        "### 95% VaR coverage",
        "",
        _markdown_table(
            (
                "Series",
                "Model",
                "N",
                "Exception rate",
                "Kupiec p",
                "Independence status",
            ),
            coverage_95_table,
        ),
        "",
        "### 99% VaR coverage",
        "",
        _markdown_table(
            (
                "Series",
                "Model",
                "N",
                "Exception rate",
                "Kupiec p",
                "Independence status",
            ),
            coverage_99_table,
        ),
        "",
        "Equality between realized loss and VaR is not an exception. "
        "Christoffersen results with zero required transition cells are labelled "
        "`insufficient_events`.",
        "A model without a coverage rejection is consistent with nominal "
        "unconditional coverage at the chosen significance level; that is not "
        "proof of perfect calibration. Coverage evidence should also be read "
        "alongside forecast availability and exception independence.",
        "",
        "## Fit diagnostics",
        "",
        _markdown_table(
            (
                "Model",
                "Scheduled fits",
                "Optimizer converged",
                "Retries",
                "Failed forecasts",
            ),
            diagnostic_table,
        ),
        "",
        "A converged optimizer result can still fail the parameter rules. "
        "In particular, nonstationary fits are retained as failed forecasts and "
        "stale parameters are not reused.",
        "",
        "## Run identity and data lineage",
        "",
        f"- Experiment: `{lineage.experiment_id}`.",
        f"- Forecasting execution started: `{lineage.execution_started_at}`.",
        f"- Numerical source commit: `{lineage.numerical_git_commit}` "
        f"({lineage.numerical_source_state}).",
        f"- Numerical source-tree SHA-256: `{lineage.numerical_source_tree_sha256}`.",
        "- Upstream engine: historical-asset-risk-engine "
        f"{lineage.upstream_project_version}, commit "
        f"`{lineage.upstream_git_commit}`.",
        f"- Upstream provider: `{lineage.upstream_provider}`; acquired at "
        f"`{lineage.upstream_acquired_at}`.",
        "- Upstream adjusted-price coverage: "
        f"`{lineage.upstream_actual_start_date}` through "
        f"`{lineage.upstream_actual_end_date}`.",
        f"- Upstream `simple_returns.csv` SHA-256: `{lineage.simple_returns_sha256}`.",
        "",
        "## Method and traceability",
        "",
        f"- Series: {series_text}.",
        "- Forecast horizon: one observed session.",
        f"- Variance benchmark: {historical['variance_window']}-return sample "
        "variance.",
        f"- VaR benchmark: {historical['var_window']}-return historical "
        f"simulation using {historical['quantile_method']} interpolation.",
        f"- EWMA: lambda {ewma['lambda']} with a "
        f"{ewma['initialization_window']}-return initialization.",
        "- GARCH candidates: zero-mean Gaussian and Student-t GARCH(1,1), "
        f"{garch['estimation_window']}-return estimation windows, refitted every "
        f"{garch['refit_every_origins']} eligible origins.",
        "- Primary variance score: QLIKE.",
        "- Primary VaR score: 5% lower-tail pinball loss.",
        "- Uncertainty: moving-block bootstrap, block length "
        f"{evaluation['bootstrap_block_length']}, "
        f"{evaluation['bootstrap_resamples']} resamples, "
        f"seed {experiment['random_seed']}, {interval_label} intervals.",
        "- Every reported aggregate is traceable through the saved forecasts, "
        "realizations, score tables, experiment manifest, and run manifest.",
        "",
        "## Limitations",
        "",
        "Squared one-session returns are noisy realized-variance proxies. "
        + portfolio_limitation
        + "The study excludes costs, taxes, financing, Expected "
        "Shortfall scoring, asymmetric or multivariate volatility models, and "
        "all trading or regulatory claims. Historical results do not guarantee "
        "future performance.",
        "",
    ]
    return "\n".join(sections)


def _configuration_section(
    effective: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    section = effective.get(name)
    if not isinstance(section, dict):
        raise ArtifactReconciliationFailedError(
            f"Effective configuration is missing the [{name}] section."
        )
    return section


def _configuration_number(section: dict[str, Any], key: str) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactReconciliationFailedError(
            f"Effective configuration value {key!r} must be numeric."
        )
    return float(value)


def _percentage_label(value: float) -> str:
    return f"{value * 100:g}%"


def _comparison_rows(
    bootstrap: pd.DataFrame,
    period: str,
    metric: str,
) -> pd.DataFrame:
    selected = bootstrap.loc[
        bootstrap["period"].eq(period)
        & bootstrap["series_id"].eq("ALL")
        & bootstrap["metric"].eq(metric)
        & bootstrap["model_id"].isin(_CANDIDATES)
    ].copy()
    return selected.sort_values("model_id", kind="stable").reset_index(drop=True)


def _direct_answer(
    qlike_test: pd.DataFrame,
    pinball_test: pd.DataFrame,
    coverage: pd.DataFrame,
    interval_label: str,
) -> str:
    return (
        _direction_sentence(
            qlike_test,
            "one-session variance QLIKE",
            interval_label,
        )
        + " "
        + _direction_sentence(
            pinball_test,
            "95% VaR pinball loss",
            interval_label,
        )
        + " "
        + _calibration_summary(coverage)
        + " The tables report effect sizes, paired counts, and bootstrap "
        "intervals; no result is described as live, regulatory, or a guarantee."
    )


def _calibration_summary(
    coverage: pd.DataFrame,
    *,
    significance_level: float = 0.05,
) -> str:
    selected = coverage.loc[
        coverage["period"].eq("test")
        & coverage["metric"].eq("kupiec_p_value")
        & coverage["model_id"].isin(_QUANTILE_MODELS)
        & coverage["confidence_level"].isin((0.95, 0.99))
    ].copy()
    selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
    series_count = int(selected["series_id"].nunique())
    expected_count = series_count * 2
    consistent: list[str] = []
    rejected: list[str] = []
    incomplete: list[str] = []
    for model_id in _QUANTILE_MODELS:
        model = selected.loc[selected["model_id"].eq(model_id)]
        valid = model.loc[np.isfinite(model["value"].to_numpy(dtype=float))]
        if valid["value"].lt(significance_level).any():
            rejected.append(model_id)
        elif expected_count > 0 and len(valid) == expected_count:
            consistent.append(model_id)
        else:
            incomplete.append(model_id)

    parts = [
        "VaR calibration varied by model and series.",
        f"At the {significance_level:.0%} test significance level,",
    ]
    if consistent:
        parts.append(
            f"{_model_list(consistent)} had no Kupiec coverage rejection across "
            "every reported series at both 95% and 99% VaR;"
        )
    if rejected:
        parts.append(f"{_model_list(rejected)} had at least one rejection;")
    if incomplete:
        parts.append(
            f"{_model_list(incomplete)} lacked a complete set of valid Kupiec tests;"
        )
    return " ".join(parts).rstrip(";") + "."


def _model_list(model_ids: Sequence[str]) -> str:
    labels = [_MODEL_LABELS.get(model_id, model_id) for model_id in model_ids]
    if len(labels) < 2:
        return labels[0] if labels else "no model"
    if len(labels) == 2:
        return " and ".join(labels)
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _direction_sentence(
    frame: pd.DataFrame,
    label: str,
    interval_label: str,
) -> str:
    available = frame.loc[frame["paired_count"].gt(0)]
    lower = [
        _MODEL_LABELS[str(row.model_id)]
        for row in available.itertuples(index=False)
        if float(cast(Any, row.value)) < 0.0
    ]
    interval_supported = [
        _MODEL_LABELS[str(row.model_id)]
        for row in available.itertuples(index=False)
        if float(cast(Any, row.interval_upper)) < 0.0
    ]
    lower_text = ", ".join(lower) if lower else "no candidate"
    interval_text = (
        ", ".join(interval_supported) if interval_supported else "no candidate"
    )
    return (
        f"On the aggregate final-test common panel, {lower_text} had lower "
        f"{label} than their historical benchmark; {interval_text} also had a "
        f"{interval_label} bootstrap interval entirely below zero."
    )


def _comparison_markdown(frame: pd.DataFrame, interval_label: str) -> str:
    rows = [
        (
            _MODEL_LABELS.get(str(row.model_id), str(row.model_id)),
            str(int(cast(Any, row.paired_count))),
            _format_number(row.value),
            f"[{_format_number(row.interval_lower)}, "
            f"{_format_number(row.interval_upper)}]",
            _format_number(row.median_difference),
            _format_number(row.fraction_below_zero, digits=3),
        )
        for row in frame.itertuples(index=False)
    ]
    return _markdown_table(
        (
            "Candidate",
            "Paired N",
            "Mean difference",
            f"{interval_label} interval",
            "Median difference",
            "Bootstrap fraction < 0",
        ),
        rows,
    )


def _availability_rows(availability: pd.DataFrame) -> list[tuple[str, ...]]:
    selected = availability.loc[
        availability["period"].eq("test") & availability["series_id"].eq("ALL")
    ]
    pivot = selected.pivot(
        index="model_id",
        columns="metric",
        values="value",
    )
    rows: list[tuple[str, ...]] = []
    for model_id, values in pivot.sort_index().iterrows():
        eligible = float(values["total_eligible_target_dates"])
        valid = float(values["valid_forecast_count"])
        failed = float(values["failed_forecast_count"])
        rows.append(
            (
                _MODEL_LABELS.get(str(model_id), str(model_id)),
                str(int(eligible)),
                str(int(valid)),
                str(int(failed)),
                f"{valid / eligible:.2%}" if eligible else "n/a",
            )
        )
    return rows


def _coverage_rows(
    coverage: pd.DataFrame,
    *,
    confidence_level: float,
) -> list[tuple[str, ...]]:
    selected = coverage.loc[
        coverage["period"].eq("test")
        & np.isclose(coverage["confidence_level"], confidence_level)
    ]
    pivot = selected.pivot_table(
        index=["series_id", "model_id"],
        columns="metric",
        values="value",
        aggfunc="first",
    )
    status = selected.loc[
        selected["metric"].eq("christoffersen_independence_lr"),
        ["series_id", "model_id", "status"],
    ].set_index(["series_id", "model_id"])
    observation_counts = selected.loc[
        selected["metric"].eq("exception_count"),
        ["series_id", "model_id", "observation_count"],
    ].set_index(["series_id", "model_id"])
    rows: list[tuple[str, ...]] = []
    for index_value, values in pivot.sort_index().iterrows():
        series_id, model_id = cast(tuple[Any, Any], index_value)
        rows.append(
            (
                str(series_id),
                _MODEL_LABELS.get(str(model_id), str(model_id)),
                str(
                    int(
                        cast(
                            Any,
                            observation_counts.loc[
                                (series_id, model_id),
                                "observation_count",
                            ],
                        )
                    )
                ),
                _format_number(values["exception_rate"], digits=4),
                _format_number(values["kupiec_p_value"], digits=4),
                str(status.loc[(series_id, model_id), "status"]),
            )
        )
    return rows


def _diagnostic_rows(
    diagnostics: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> list[tuple[str, ...]]:
    failed = (
        forecasts.loc[forecasts["status"].eq("failed")]
        .groupby("model_id")
        .size()
        .to_dict()
    )
    rows: list[tuple[str, ...]] = []
    for model_id, group in diagnostics.groupby("model_id", sort=True):
        rows.append(
            (
                _MODEL_LABELS.get(str(model_id), str(model_id)),
                str(len(group)),
                str(int(group["converged"].sum())),
                str(int(group["retry_used"].sum())),
                str(int(failed.get(model_id, 0))),
            )
        )
    return rows


def _plot_comparison(
    bootstrap: pd.DataFrame,
    *,
    metric: str,
    title: str,
    scale: float,
    ylabel: str,
    path: Path,
) -> None:
    frame = _comparison_rows(bootstrap, "test", metric)
    values = frame["value"].to_numpy(dtype=float) * scale
    lower = frame["interval_lower"].to_numpy(dtype=float) * scale
    upper = frame["interval_upper"].to_numpy(dtype=float) * scale
    labels = [_MODEL_LABELS[str(value)] for value in frame["model_id"]]
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    positions = range(len(frame))
    axis.bar(positions, values, color="#315b7d")
    axis.errorbar(
        list(positions),
        values,
        yerr=[values - lower, upper - values],
        fmt="none",
        ecolor="#171717",
        capsize=5,
        linewidth=1.2,
    )
    axis.axhline(0.0, color="#8c2d2d", linewidth=1.0)
    axis.set_xticks(list(positions), labels, rotation=12, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=150,
        metadata={"Software": "market-risk-forecasting-engine"},
    )
    plt.close(figure)


def _plot_availability(availability: pd.DataFrame, path: Path) -> None:
    selected = availability.loc[
        availability["period"].eq("test") & availability["series_id"].eq("ALL")
    ]
    pivot = selected.pivot(
        index="model_id",
        columns="metric",
        values="value",
    ).sort_index()
    rates = pivot["valid_forecast_count"] / pivot["total_eligible_target_dates"]
    labels = [_MODEL_LABELS.get(str(value), str(value)) for value in rates.index]
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.bar(range(len(rates)), rates.to_numpy(), color="#537d3d")
    axis.set_xticks(range(len(rates)), labels, rotation=12, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Valid forecast fraction")
    axis.set_title("Final-test forecast availability")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=150,
        metadata={"Software": "market-risk-forecasting-engine"},
    )
    plt.close(figure)


def _plot_forecast_volatility_history(
    forecasts: pd.DataFrame,
    realizations: pd.DataFrame,
    path: Path,
) -> None:
    actual = _realizations_for_forecasts(forecasts, realizations)
    actual["absolute_return_percent"] = actual["simple_return"].abs() * 100.0
    actual["rolling_rms_percent"] = (
        actual.groupby("series_id", sort=False)["squared_return"]
        .transform(lambda values: values.rolling(21, min_periods=5).mean())
        .pow(0.5)
        * 100.0
    )
    series_ids = _ordered_series_ids(actual)
    figure, axes = plt.subplots(
        len(series_ids),
        1,
        figsize=(12.0, max(4.5, 2.6 * len(series_ids))),
        sharex=True,
        squeeze=False,
    )
    for position, series_id in enumerate(series_ids):
        axis = axes[position, 0]
        series_actual = actual.loc[actual["series_id"].eq(series_id)]
        _apply_period_context(axis, series_actual, show_labels=position == 0)
        axis.scatter(
            series_actual["target_date"],
            series_actual["absolute_return_percent"],
            color="#9b9b9b",
            alpha=0.28,
            s=7,
            linewidths=0,
            label="Absolute next-session return" if position == 0 else None,
            zorder=1,
        )
        axis.plot(
            series_actual["target_date"],
            series_actual["rolling_rms_percent"],
            color="#171717",
            linewidth=1.1,
            label="21-session realized RMS" if position == 0 else None,
            zorder=2,
        )
        for model_id in _VARIANCE_MODELS:
            model = _model_history(forecasts, series_id, model_id)
            volatility = pd.to_numeric(model["volatility"], errors="coerce").where(
                model["status"].eq("ok")
            )
            axis.plot(
                model["target_date"],
                volatility * 100.0,
                color=_MODEL_COLORS[model_id],
                linewidth=1.0,
                label=_MODEL_LABELS[model_id] if position == 0 else None,
                zorder=3,
            )
        axis.set_ylabel(f"{series_id}\nPercent")
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Target date")
    figure.suptitle(
        "One-session volatility forecasts and realized-return context",
        y=0.995,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    _save_figure(figure, path)


def _plot_var_exception_history(
    forecasts: pd.DataFrame,
    realizations: pd.DataFrame,
    path: Path,
) -> None:
    actual = _realizations_for_forecasts(forecasts, realizations)
    actual["loss_percent"] = actual["loss"] * 100.0
    series_ids = _ordered_series_ids(actual)
    markers = {
        HISTORICAL_SIMULATION_MODEL_ID: "o",
        EWMA_MODEL_ID: "s",
        GAUSSIAN_GARCH_MODEL_ID: "^",
        STUDENT_T_GARCH_MODEL_ID: "D",
    }
    figure, axes = plt.subplots(
        len(series_ids),
        1,
        figsize=(12.0, max(4.5, 2.6 * len(series_ids))),
        sharex=True,
        squeeze=False,
    )
    for position, series_id in enumerate(series_ids):
        axis = axes[position, 0]
        series_actual = actual.loc[actual["series_id"].eq(series_id)]
        _apply_period_context(axis, series_actual, show_labels=position == 0)
        axis.plot(
            series_actual["target_date"],
            series_actual["loss_percent"],
            color="#8a8a8a",
            linewidth=0.65,
            alpha=0.7,
            label="Realized daily loss" if position == 0 else None,
            zorder=1,
        )
        realization_values = series_actual.loc[:, [*_FORECAST_KEYS, "loss_percent"]]
        for model_id in _QUANTILE_MODELS:
            model = _model_history(forecasts, series_id, model_id)
            value_at_risk = pd.to_numeric(model["var_0_95"], errors="coerce").where(
                model["status"].eq("ok")
            )
            axis.plot(
                model["target_date"],
                value_at_risk * 100.0,
                color=_MODEL_COLORS[model_id],
                linewidth=1.0,
                label=_MODEL_LABELS[model_id] if position == 0 else None,
                zorder=2,
            )
            joined = model.loc[:, list(_FORECAST_KEYS)].copy()
            joined["var_percent"] = value_at_risk * 100.0
            joined = joined.merge(
                realization_values,
                on=list(_FORECAST_KEYS),
                how="inner",
                validate="one_to_one",
            )
            exceptions = joined.loc[joined["loss_percent"].gt(joined["var_percent"])]
            axis.scatter(
                exceptions["target_date"],
                exceptions["loss_percent"],
                facecolors="none",
                edgecolors=_MODEL_COLORS[model_id],
                marker=markers[model_id],
                s=28,
                linewidths=0.9,
                zorder=3,
            )
        axis.axhline(0.0, color="#b5b5b5", linewidth=0.6)
        axis.set_ylabel(f"{series_id}\nPercent")
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Target date")
    figure.suptitle(
        "Realized daily loss, 95% VaR forecasts, and exceptions",
        y=0.995,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    _save_figure(figure, path)


def _plot_rolling_model_advantage(
    forecasts: pd.DataFrame,
    realizations: pd.DataFrame,
    *,
    window: int,
    path: Path,
) -> None:
    differences = _rolling_loss_differences(
        forecasts,
        realizations,
        window=window,
    )
    metrics = (
        ("qlike", "Variance QLIKE difference vs historical variance"),
        ("pinball_loss_0_05", "95% VaR pinball difference vs historical simulation"),
    )
    figure, axes = plt.subplots(2, 1, figsize=(12.0, 7.2), sharex=True)
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        _apply_period_context(
            axis,
            realizations,
            show_labels=axis is axes[0],
        )
        selected = differences.loc[differences["metric"].eq(metric)]
        for model_id in _CANDIDATES:
            model = selected.loc[selected["model_id"].eq(model_id)]
            axis.plot(
                model["target_date"],
                model["rolling_difference"],
                color=_MODEL_COLORS[model_id],
                linewidth=1.25,
                label=_MODEL_LABELS[model_id],
            )
        axis.axhline(0.0, color="#7a2525", linewidth=0.9)
        axis.set_title(title)
        axis.set_ylabel("Candidate minus benchmark")
        axis.grid(alpha=0.2)
    axes[0].legend(loc="best", frameon=False)
    axes[-1].set_xlabel("Target date (below zero means candidate better)")
    figure.suptitle(f"Rolling {window}-session model advantage")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    _save_figure(figure, path)


def _rolling_loss_differences(
    forecasts: pd.DataFrame,
    realizations: pd.DataFrame,
    *,
    window: int,
) -> pd.DataFrame:
    if window < 1:
        raise ArtifactReconciliationFailedError(
            "Rolling comparison window must be at least one session."
        )
    realization_values = realizations.loc[:, [*_FORECAST_KEYS, "simple_return"]].copy()
    rows: list[pd.DataFrame] = []
    specifications = (
        (
            "qlike",
            HISTORICAL_VARIANCE_MODEL_ID,
            "variance",
        ),
        (
            "pinball_loss_0_05",
            HISTORICAL_SIMULATION_MODEL_ID,
            "return_quantile_0_05",
        ),
    )
    for metric, benchmark_id, value_column in specifications:
        benchmark = _valid_model_values(forecasts, benchmark_id, value_column)
        expected_series = set(benchmark["series_id"].astype(str))
        canonical_dates = pd.DatetimeIndex(
            pd.to_datetime(benchmark["target_date"].drop_duplicates()).sort_values()
        )
        for model_id in _CANDIDATES:
            candidate = _valid_model_values(forecasts, model_id, value_column)
            paired = candidate.merge(
                benchmark,
                on=list(_FORECAST_KEYS),
                how="inner",
                suffixes=("_candidate", "_benchmark"),
                validate="one_to_one",
            ).merge(
                realization_values,
                on=list(_FORECAST_KEYS),
                how="inner",
                validate="one_to_one",
            )
            if metric == "qlike":
                candidate_variance = paired[f"{value_column}_candidate"]
                benchmark_variance = paired[f"{value_column}_benchmark"]
                realized_variance = paired["simple_return"].pow(2)
                paired["difference"] = (
                    np.log(candidate_variance)
                    + realized_variance / candidate_variance
                    - np.log(benchmark_variance)
                    - realized_variance / benchmark_variance
                )
            else:
                realized_return = paired["simple_return"]
                candidate_residual = (
                    realized_return - paired[f"{value_column}_candidate"]
                )
                benchmark_residual = (
                    realized_return - paired[f"{value_column}_benchmark"]
                )
                candidate_loss = candidate_residual * np.where(
                    candidate_residual.lt(0.0), -0.95, 0.05
                )
                benchmark_loss = benchmark_residual * np.where(
                    benchmark_residual.lt(0.0), -0.95, 0.05
                )
                paired["difference"] = candidate_loss - benchmark_loss
            finite = np.isfinite(paired["difference"].to_numpy(dtype=float))
            finite_pairs = paired.loc[
                finite, ["series_id", "target_date", "difference"]
            ].copy()
            finite_pairs["series_id"] = finite_pairs["series_id"].astype(str)
            complete_dates = finite_pairs.groupby("target_date", sort=True)[
                "series_id"
            ].agg(lambda values, expected=expected_series: set(values) == expected)
            complete_dates = complete_dates.astype(bool)
            daily = (
                finite_pairs.loc[
                    finite_pairs["target_date"].isin(
                        complete_dates.loc[complete_dates].index
                    ),
                    ["target_date", "difference"],
                ]
                .groupby("target_date", sort=True)["difference"]
                .mean()
                .reindex(canonical_dates)
                .rename_axis("target_date")
                .reset_index()
            )
            daily["rolling_difference"] = (
                daily["difference"]
                .rolling(
                    window,
                    min_periods=window,
                )
                .mean()
            )
            daily["metric"] = metric
            daily["model_id"] = model_id
            rows.append(
                daily.loc[
                    :, ["metric", "model_id", "target_date", "rolling_difference"]
                ]
            )
    if not rows:
        return pd.DataFrame(
            columns=("metric", "model_id", "target_date", "rolling_difference")
        )
    result = pd.concat(rows, ignore_index=True)
    result["target_date"] = pd.to_datetime(result["target_date"])
    return result.sort_values(
        ["metric", "model_id", "target_date"], kind="stable"
    ).reset_index(drop=True)


def _plot_var_calibration(coverage: pd.DataFrame, path: Path) -> None:
    selected = coverage.loc[
        coverage["period"].eq("test")
        & coverage["metric"].eq("exception_rate")
        & coverage["model_id"].isin(_QUANTILE_MODELS)
    ].copy()
    selected["series_id"] = selected["series_id"].astype(str)
    series_ids = sorted(str(value) for value in selected["series_id"].unique())
    markers = ("o", "s", "^", "D", "P", "X")
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=False)
    for axis, confidence_level in zip(axes, (0.95, 0.99), strict=True):
        level = selected.loc[np.isclose(selected["confidence_level"], confidence_level)]
        positions = np.arange(len(_QUANTILE_MODELS), dtype=float)
        offsets = np.linspace(-0.24, 0.24, max(len(series_ids), 1))
        for series_position, (offset, series_id) in enumerate(
            zip(offsets, series_ids, strict=True)
        ):
            marker = markers[series_position % len(markers)]
            series = level.loc[level["series_id"].eq(series_id)].set_index("model_id")
            values = [
                float(cast(Any, series.loc[model_id, "value"])) * 100.0
                if model_id in series.index
                else np.nan
                for model_id in _QUANTILE_MODELS
            ]
            axis.scatter(
                positions + offset,
                values,
                c=[_MODEL_COLORS[model_id] for model_id in _QUANTILE_MODELS],
                marker=marker,
                s=48,
                edgecolors="white",
                linewidths=0.6,
                label=series_id,
                zorder=3,
            )
        nominal = (1.0 - confidence_level) * 100.0
        axis.axhline(
            nominal,
            color="#7a2525",
            linestyle="--",
            linewidth=1.1,
            label="Nominal rate",
        )
        axis.set_xticks(
            positions,
            [_MODEL_LABELS[model_id] for model_id in _QUANTILE_MODELS],
            rotation=15,
            ha="right",
        )
        axis.set_ylabel("Observed exception rate (%)")
        axis.set_title(f"{confidence_level:.0%} VaR")
        axis.set_ylim(bottom=0.0)
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=min(len(labels), 5),
        frameon=False,
    )
    figure.suptitle("Final-test VaR calibration by series", y=0.98)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
    _save_figure(figure, path)


def _plot_series_comparisons(bootstrap: pd.DataFrame, path: Path) -> None:
    specifications = (
        ("qlike", "Variance QLIKE", 1.0, "Loss difference"),
        (
            "pinball_loss_0_05",
            "95% VaR pinball loss",
            10_000.0,
            "Loss difference (basis points)",
        ),
    )
    selected = bootstrap.loc[
        bootstrap["period"].eq("test")
        & bootstrap["series_id"].ne("ALL")
        & bootstrap["model_id"].isin(_CANDIDATES)
    ].copy()
    series_ids = sorted(str(value) for value in selected["series_id"].unique())
    base_positions = np.arange(len(series_ids), dtype=float)
    offsets = dict(zip(_CANDIDATES, (-0.22, 0.0, 0.22), strict=True))
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.8), sharey=True)
    for axis, (metric, title, scale, xlabel) in zip(axes, specifications, strict=True):
        metric_rows = selected.loc[selected["metric"].eq(metric)]
        for model_id in _CANDIDATES:
            model_rows = metric_rows.loc[
                metric_rows["model_id"].eq(model_id)
            ].set_index("series_id")
            values = np.array(
                [
                    float(cast(Any, model_rows.loc[series_id, "value"])) * scale
                    if series_id in model_rows.index
                    else np.nan
                    for series_id in series_ids
                ]
            )
            lower = np.array(
                [
                    float(cast(Any, model_rows.loc[series_id, "interval_lower"]))
                    * scale
                    if series_id in model_rows.index
                    else np.nan
                    for series_id in series_ids
                ]
            )
            upper = np.array(
                [
                    float(cast(Any, model_rows.loc[series_id, "interval_upper"]))
                    * scale
                    if series_id in model_rows.index
                    else np.nan
                    for series_id in series_ids
                ]
            )
            y_positions = base_positions + offsets[model_id]
            axis.errorbar(
                values,
                y_positions,
                xerr=[values - lower, upper - values],
                fmt="o",
                color=_MODEL_COLORS[model_id],
                capsize=3,
                markersize=5,
                linewidth=1.1,
                label=_MODEL_LABELS[model_id],
            )
        axis.axvline(0.0, color="#7a2525", linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel(f"{xlabel} (below zero favors candidate)")
        axis.set_yticks(base_positions, series_ids)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)
    axes[0].legend(loc="best", frameon=False)
    figure.suptitle("Final-test candidate effects by series")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    _save_figure(figure, path)


def _apply_period_context(
    axis: Any,
    realizations: pd.DataFrame,
    *,
    show_labels: bool,
) -> None:
    calendar = realizations.loc[:, ["target_date", "period"]].drop_duplicates()
    calendar["target_date"] = pd.to_datetime(calendar["target_date"])
    duplicated = calendar["target_date"].duplicated(keep=False)
    if duplicated.any():
        raise ArtifactReconciliationFailedError(
            "A target date is assigned to more than one experiment period."
        )
    if calendar.empty:
        raise ArtifactReconciliationFailedError(
            "Saved realizations contain no period dates to plot."
        )
    colors = {
        "development": "#eeeeee",
        "validation": "#e8f1f8",
        "test": "#f4f7ed",
    }
    labels = {
        "development": "Development",
        "validation": "Validation",
        "test": "Final test",
    }
    full_span = calendar["target_date"].max() - calendar["target_date"].min()
    for period in ("development", "validation", "test"):
        dates = calendar.loc[calendar["period"].eq(period), "target_date"]
        if dates.empty:
            continue
        start = pd.Timestamp(dates.min())
        end = pd.Timestamp(dates.max())
        axis.axvspan(start, end, color=colors[period], zorder=0)
        if period != "development":
            axis.axvline(start, color="#666666", linestyle=":", linewidth=0.8)
        if show_labels:
            midpoint = start + (end - start) / 2
            short_period = (end - start) < full_span * 0.12
            axis.text(
                midpoint,
                0.98,
                labels[period],
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                color="#555555",
                fontsize=7 if short_period else 8,
                rotation=90 if short_period else 0,
            )
    axis.set_xlim(calendar["target_date"].min(), calendar["target_date"].max())


def _valid_model_values(
    forecasts: pd.DataFrame,
    model_id: str,
    value_column: str,
) -> pd.DataFrame:
    selected = forecasts.loc[
        forecasts["model_id"].eq(model_id) & forecasts["status"].eq("ok"),
        [*_FORECAST_KEYS, value_column],
    ].copy()
    selected[value_column] = pd.to_numeric(selected[value_column], errors="coerce")
    valid = np.isfinite(selected[value_column].to_numpy(dtype=float))
    if value_column == "variance":
        valid &= selected[value_column].to_numpy(dtype=float) > 0.0
    return selected.loc[valid]


def _realizations_for_forecasts(
    forecasts: pd.DataFrame,
    realizations: pd.DataFrame,
) -> pd.DataFrame:
    keys = forecasts.loc[:, list(_FORECAST_KEYS)].drop_duplicates()
    actual = realizations.merge(
        keys,
        on=list(_FORECAST_KEYS),
        how="inner",
        validate="one_to_one",
    ).copy()
    actual["target_date"] = pd.to_datetime(actual["target_date"])
    actual = actual.sort_values(
        ["series_id", "target_date"], kind="stable"
    ).reset_index(drop=True)
    if actual.empty:
        raise ArtifactReconciliationFailedError(
            "Saved artifacts contain no realizations matching forecasts."
        )
    return actual


def _model_history(
    forecasts: pd.DataFrame,
    series_id: str,
    model_id: str,
) -> pd.DataFrame:
    model = forecasts.loc[
        forecasts["series_id"].eq(series_id) & forecasts["model_id"].eq(model_id)
    ].copy()
    model["target_date"] = pd.to_datetime(model["target_date"])
    return model.sort_values("target_date", kind="stable")


def _ordered_series_ids(frame: pd.DataFrame) -> list[str]:
    series_ids = [str(value) for value in frame["series_id"].drop_duplicates()]
    if not series_ids:
        raise ArtifactReconciliationFailedError(
            "Saved artifacts contain no series to plot."
        )
    return series_ids


def _save_figure(figure: Any, path: Path) -> None:
    figure.savefig(
        path,
        dpi=150,
        metadata={"Software": "market-risk-forecasting-engine"},
    )
    plt.close(figure)


def _markdown_table(
    headers: tuple[str, ...],
    rows: Sequence[Sequence[str]],
) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _format_number(value: Any, *, digits: int = 6) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{digits}g}"


__all__ = ["ReportResult", "generate_report"]
