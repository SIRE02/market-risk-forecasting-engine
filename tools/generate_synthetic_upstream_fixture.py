"""Generate the deterministic, redistributable upstream-contract fixture."""

from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path

UPSTREAM_COMMIT = "5d189b528306b78b87970e4e83dc2b8dc7b279b3"
INSTRUMENTS = ("SPY", "IEF", "GLD")


def _business_dates(start: date, end: date) -> list[date]:
    result: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            result.append(current)
        current += timedelta(days=1)
    return result


def _returns(position: int) -> tuple[float, float, float]:
    spy = ((position % 17) - 8) / 1000
    ief = (((position * 3) % 19) - 9) / 2000
    gld = (((position * 5) % 23) - 11) / 1500
    return spy, ief, gld


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    fixture_dir = root / "data" / "fixtures" / "upstream_run"
    expected_dir = root / "data" / "fixtures" / "expected"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    expected_dir.mkdir(parents=True, exist_ok=True)

    price_dates = _business_dates(date(2007, 1, 1), date(2025, 12, 31))
    dates = price_dates[1:]
    returns_path = fixture_dir / "simple_returns.csv"
    with returns_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("date", *INSTRUMENTS))
        for position, observation_date in enumerate(dates):
            writer.writerow(
                (
                    observation_date.isoformat(),
                    *(f"{value:.10f}" for value in _returns(position)),
                )
            )

    quality = {
        "fixture": {
            "synthetic": True,
            "generator": "tools/generate_synthetic_upstream_fixture.py",
        },
        "requested_instruments": list(INSTRUMENTS),
        "returned_instruments": list(INSTRUMENTS),
        "source_row_count": len(price_dates) * len(INSTRUMENTS),
        "invalid_date_count": 0,
        "duplicate_date_instrument_rows_removed": 0,
        "source_missing_adjusted_close_values": 0,
        "missing_adjusted_close_values": 0,
        "invalid_price_values_removed": 0,
        "union_date_count_before_alignment": len(price_dates),
        "common_date_count_after_alignment": len(price_dates),
        "common_history_rows_removed": 0,
        "first_common_date": price_dates[0].isoformat(),
        "last_common_date": price_dates[-1].isoformat(),
        "instruments": [
            {
                "ticker": ticker,
                "rows_after_date_filter": len(price_dates),
                "valid_observations_before_alignment": len(price_dates),
                "missing_values_before_alignment": 0,
                "valid_observations_after_alignment": len(price_dates),
                "common_history_reduction": 0,
            }
            for ticker in INSTRUMENTS
        ],
    }
    (fixture_dir / "data_quality_report.json").write_text(
        json.dumps(quality, indent=2) + "\n", encoding="utf-8"
    )

    artifact_schemas = {
        "data_quality_report.json": {
            "schema_id": "historical-asset-risk/data-quality-report",
            "schema_version": "1.experimental",
            "units": "structured_counts_and_lineage",
        },
        "run_manifest.json": {
            "schema_id": "historical-asset-risk/run-manifest",
            "schema_version": "1.experimental",
            "units": "structured_lineage",
        },
        "simple_returns.csv": {
            "schema_id": "historical-asset-risk/simple-returns",
            "schema_version": "1.experimental",
            "units": "decimal_return_per_observation",
        },
    }
    manifest = {
        "project": "historical-asset-risk-engine",
        "project_version": "0.1.0",
        "git_commit": UPSTREAM_COMMIT,
        "execution_timestamp": "2026-01-01T00:00:00+00:00",
        "fixture": {
            "synthetic": True,
            "generator": "tools/generate_synthetic_upstream_fixture.py",
        },
        "configuration": {
            "provider": "synthetic-offline-fixture",
            "tickers": list(INSTRUMENTS),
            "start_date": "2007-01-01",
            "end_date": "2026-01-01",
            "rolling_window": 252,
        },
        "estimation_conventions": {
            "missing_data": {
                "alignment": "complete_case_adjusted_prices",
                "forward_fill": False,
                "gap_spanning_return_policy": "fail",
            }
        },
        "data_source": {
            "provider": "synthetic-offline-fixture",
            "source": "generated",
            "acquired_at": "2026-01-01T00:00:00+00:00",
            "actual_start_date": price_dates[0].isoformat(),
            "actual_end_date": price_dates[-1].isoformat(),
            "observation_count": len(price_dates),
            "instruments": list(INSTRUMENTS),
            "canonical_record_stage": (
                "normalized_requested_in_range_pre_complete_case_alignment"
            ),
        },
        "dependency_versions": {},
        "generated_artifacts": sorted(artifact_schemas),
        "artifact_schemas": artifact_schemas,
        "consumer_compatibility": {
            "package_version": "0.1.0",
            "artifact_schemas": artifact_schemas,
            "fixture_identity": "market-risk-forecasting-synthetic-v1",
            "contract_status": "experimental",
        },
    }
    (fixture_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    with (expected_dir / "research_series_head.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("date", *INSTRUMENTS, "MIX_60_30_10"))
        for position, observation_date in enumerate(dates[:10]):
            spy, ief, gld = _returns(position)
            proxy = 0.60 * spy + 0.30 * ief + 0.10 * gld
            writer.writerow(
                (
                    observation_date.isoformat(),
                    f"{spy:.10f}",
                    f"{ief:.10f}",
                    f"{gld:.10f}",
                    f"{proxy:.10f}",
                )
            )


if __name__ == "__main__":
    main()
