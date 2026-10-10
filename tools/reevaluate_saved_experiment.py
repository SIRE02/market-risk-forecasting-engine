"""Recompute evaluation and reporting without refitting verified forecasts."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from market_risk_forecasting.config import load_config
from market_risk_forecasting.experiment import reevaluate_experiment
from market_risk_forecasting.reporting import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    config = replace(
        config, experiment=replace(config.experiment, output_dir=args.output_dir)
    )
    result = reevaluate_experiment(source_dir=args.source_dir, config=config)
    report = generate_report(result.output_dir)
    print(f"Reevaluation complete: {report.report_path}")


if __name__ == "__main__":
    main()
