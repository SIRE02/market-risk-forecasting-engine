"""Command-line surface for market-risk forecasting experiments."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from market_risk_forecasting import __version__
from market_risk_forecasting.config import ForecastConfig, load_config
from market_risk_forecasting.datasets import ResearchDataset, build_research_dataset
from market_risk_forecasting.errors import MarketRiskForecastingError
from market_risk_forecasting.experiment import (
    ExperimentRunResult,
    execute_experiment,
    verify_experiment_directory,
)
from market_risk_forecasting.reporting import ReportResult, generate_report
from market_risk_forecasting.upstream import (
    UpstreamRun,
    coverage_requirements,
    load_upstream_run,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market-risk-forecast",
        description="Run reproducible one-session-ahead market-risk experiments.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("validate-input", "run", "reproduce"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--config", type=Path, required=True)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--experiment-dir", type=Path, required=True)
    return parser


def validate_input(config: ForecastConfig) -> tuple[UpstreamRun, ResearchDataset]:
    """Validate one configured input run and construct all canonical series."""
    upstream = load_upstream_run(
        config.experiment.input_run_dir,
        config.upstream,
        coverage_requirements(config),
    )
    dataset = build_research_dataset(upstream, config.portfolio_proxy)
    return upstream, dataset


def _print_validation_summary(
    upstream: UpstreamRun,
    dataset: ResearchDataset,
) -> None:
    print("Input validation: ok")
    package_identity = (
        f"historical-asset-risk-engine {upstream.installed_package_version}"
    )
    print(f"Upstream package: {package_identity}")
    print(f"Input run: {upstream.input_run_dir}")
    print(f"Return observations: {len(dataset.returns)}")
    print(
        "Return date range: "
        f"{dataset.returns.index[0].date().isoformat()} through "
        f"{dataset.returns.index[-1].date().isoformat()}"
    )
    print(f"Series: {', '.join(dataset.series_order)}")
    print(f"simple_returns.csv SHA-256: {upstream.checksums['simple_returns.csv']}")


def run_experiment(config: ForecastConfig) -> ExperimentRunResult:
    """Validate inputs and execute or reconcile the numerical experiment."""
    upstream, dataset = validate_input(config)
    return execute_experiment(
        config=config,
        upstream=upstream,
        dataset=dataset,
    )


def _print_run_summary(result: ExperimentRunResult) -> None:
    action = "reconciled" if result.reused else "completed"
    print(f"Numerical experiment: {action}")
    print(f"Experiment directory: {result.output_dir}")
    print(f"State: {result.run_manifest['state']}")


def _print_report_summary(result: ReportResult) -> None:
    action = "reconciled" if result.reused else "generated"
    print(f"Research report: {action}")
    print(f"Report path: {result.report_path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and keep expected failures concise."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-input":
            config = load_config(args.config)
            upstream, dataset = validate_input(config)
            _print_validation_summary(upstream, dataset)
            return 0
        if args.command == "run":
            experiment_result = run_experiment(load_config(args.config))
            _print_run_summary(experiment_result)
            return 0
        if args.command == "report":
            generated_report = generate_report(args.experiment_dir)
            _print_report_summary(generated_report)
            return 0
        if args.command == "reproduce":
            config = load_config(args.config)
            upstream, dataset = validate_input(config)
            _print_validation_summary(upstream, dataset)
            run_result = execute_experiment(
                config=config,
                upstream=upstream,
                dataset=dataset,
            )
            _print_run_summary(run_result)
            report_result = generate_report(run_result.output_dir)
            _print_report_summary(report_result)
            verify_experiment_directory(
                run_result.output_dir,
                require_complete=True,
            )
            print("Reproduction verification: ok")
            return 0
        raise AssertionError(f"Unhandled CLI command: {args.command!r}")
    except MarketRiskForecastingError as exc:
        print(str(exc), file=sys.stderr)
        return 2


__all__ = ["main", "run_experiment", "validate_input"]
