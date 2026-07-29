"""Command-line surface for the phased v0.1 implementation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from market_risk_forecasting import __version__
from market_risk_forecasting.config import ForecastConfig, load_config
from market_risk_forecasting.datasets import ResearchDataset, build_research_dataset
from market_risk_forecasting.errors import MarketRiskForecastingError
from market_risk_forecasting.upstream import UpstreamRun, load_upstream_run


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


def _future_phase_message(command: str) -> str:
    return f"{command} is reserved but not executable in the Phase 1 implementation."


def validate_input(config: ForecastConfig) -> tuple[UpstreamRun, ResearchDataset]:
    """Validate one configured input run and construct all canonical series."""
    upstream = load_upstream_run(
        config.experiment.input_run_dir,
        config.upstream,
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and keep expected failures concise."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-input":
            config = load_config(args.config)
            upstream, dataset = validate_input(config)
            _print_validation_summary(upstream, dataset)
            return 0
        if args.command in {"run", "reproduce"}:
            load_config(args.config)
        print(_future_phase_message(str(args.command)), file=sys.stderr)
        return 2
    except MarketRiskForecastingError as exc:
        print(str(exc), file=sys.stderr)
        return 2


__all__ = ["main"]
