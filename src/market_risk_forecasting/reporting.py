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

_MODEL_LABELS = {
    "historical_variance_252": "Historical variance (252)",
    "historical_simulation_500": "Historical simulation (500)",
    EWMA_MODEL_ID: "EWMA (lambda 0.94)",
    GAUSSIAN_GARCH_MODEL_ID: "Gaussian GARCH(1,1)",
    STUDENT_T_GARCH_MODEL_ID: "Student-t GARCH(1,1)",
}
_CANDIDATES = (
    EWMA_MODEL_ID,
    GAUSSIAN_GARCH_MODEL_ID,
    STUDENT_T_GARCH_MODEL_ID,
)
_FIGURE_NAMES = (
    "variance_qlike_comparison.png",
    "var_pinball_comparison.png",
    "forecast_availability.png",
)


@dataclass(frozen=True)
class ReportResult:
    experiment_dir: Path
    report_path: Path
    reused: bool


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
    temporary = Path(tempfile.mkdtemp(prefix=".report-building-", dir=directory))
    try:
        temporary_figures = temporary / "figures"
        temporary_figures.mkdir()
        _plot_comparison(
            tables["bootstrap"],
            metric="qlike",
            title="Final-test QLIKE difference vs historical variance",
            path=temporary_figures / "variance_qlike_comparison.png",
        )
        _plot_comparison(
            tables["bootstrap"],
            metric="pinball_loss_0_05",
            title="Final-test 95% VaR pinball difference vs historical simulation",
            path=temporary_figures / "var_pinball_comparison.png",
        )
        _plot_availability(
            tables["availability"],
            temporary_figures / "forecast_availability.png",
        )
        report = _render_report(tables, effective_configuration)
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
        "forecasts": {"model_id", "status", "error_code"},
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


def _report_identity(effective: dict[str, Any]) -> tuple[str, str, bool]:
    experiment = effective.get("experiment", {})
    protocol_version = (
        str(experiment.get("protocol_version", "1.0"))
        if isinstance(experiment, dict)
        else "1.0"
    )
    frozen = protocol_version == "1.0"
    title = (
        "# Market Risk Forecasting Engine - Frozen v0.1 Research Report"
        if frozen
        else f"# Market Risk Forecasting Engine - Protocol v{protocol_version} Report"
    )

    upstream = effective.get("upstream", {})
    instruments = upstream.get("instruments", []) if isinstance(upstream, dict) else []
    series = [str(value) for value in instruments]
    proxy = effective.get("portfolio_proxy", {})
    if isinstance(proxy, dict) and proxy.get("enabled", True):
        proxy_id = proxy.get("series_id")
        if isinstance(proxy_id, str) and proxy_id:
            series.append(proxy_id)
    return title, ", ".join(series), frozen


def _render_report(
    tables: dict[str, pd.DataFrame],
    effective_configuration: dict[str, Any],
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
    direct_answer = _direct_answer(qlike_test, pinball_test)
    availability_table = _availability_rows(availability)
    coverage_table = _coverage_rows(coverage)
    diagnostic_table = _diagnostic_rows(diagnostics, forecasts)
    title, series_text, _ = _report_identity(effective_configuration)
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
        "This is frozen historical pseudo-out-of-sample evidence, not live or "
        "prospective forecasting. Lower QLIKE and pinball loss are better.",
        "",
        "## Final-test variance comparison",
        "",
        _comparison_markdown(qlike_test),
        "",
        "![Final-test QLIKE comparison](figures/variance_qlike_comparison.png)",
        "",
        "## Final-test 95% VaR comparison",
        "",
        _comparison_markdown(pinball_test),
        "",
        "![Final-test VaR pinball comparison](figures/var_pinball_comparison.png)",
        "",
        "## Validation results",
        "",
        "Variance QLIKE:",
        "",
        _comparison_markdown(qlike_validation),
        "",
        "95% VaR pinball loss:",
        "",
        _comparison_markdown(pinball_validation),
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
        "## 95% VaR coverage by series",
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
            coverage_table,
        ),
        "",
        "Equality between realized loss and VaR is not an exception. "
        "Christoffersen results with zero required transition cells are labelled "
        "`insufficient_events`.",
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
        "A converged optimizer result can still fail the frozen parameter rules. "
        "In particular, nonstationary fits are retained as failed forecasts and "
        "stale parameters are not reused.",
        "",
        "## Method and traceability",
        "",
        f"- Series: {series_text}.",
        "- Forecast horizon: one observed session.",
        "- Variance benchmark: 252-return sample variance.",
        "- VaR benchmark: 500-return historical simulation.",
        "- Candidates: EWMA, Gaussian GARCH(1,1), Student-t GARCH(1,1).",
        "- Primary variance score: QLIKE.",
        "- Primary VaR score: 5% lower-tail pinball loss.",
        "- Uncertainty: moving-block bootstrap, block length 20, 2,000 "
        "resamples, seed 42.",
        "- Every reported aggregate is traceable through the saved forecasts, "
        "realizations, score tables, experiment manifest, and run manifest.",
        "",
        "## Limitations",
        "",
        "Squared one-session returns are noisy realized-variance proxies. The "
        + portfolio_limitation
        + "The study excludes costs, taxes, financing, Expected "
        "Shortfall scoring, asymmetric or multivariate volatility models, and "
        "all trading or regulatory claims. Historical results do not guarantee "
        "future performance.",
        "",
    ]
    return "\n".join(sections)


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
) -> str:
    return (
        _direction_sentence(
            qlike_test,
            "one-session variance QLIKE",
        )
        + " "
        + _direction_sentence(
            pinball_test,
            "95% VaR pinball loss",
        )
        + " The tables report effect sizes, paired counts, and bootstrap "
        "intervals; no result is described as live, regulatory, or a guarantee."
    )


def _direction_sentence(frame: pd.DataFrame, label: str) -> str:
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
        "95% bootstrap interval entirely below zero."
    )


def _comparison_markdown(frame: pd.DataFrame) -> str:
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
            "95% interval",
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


def _coverage_rows(coverage: pd.DataFrame) -> list[tuple[str, ...]]:
    selected = coverage.loc[
        coverage["period"].eq("test") & coverage["confidence_level"].eq(0.95)
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
    path: Path,
) -> None:
    frame = _comparison_rows(bootstrap, "test", metric)
    values = frame["value"].to_numpy(dtype=float)
    lower = frame["interval_lower"].to_numpy(dtype=float)
    upper = frame["interval_upper"].to_numpy(dtype=float)
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
    axis.set_ylabel("Candidate loss minus benchmark loss")
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
