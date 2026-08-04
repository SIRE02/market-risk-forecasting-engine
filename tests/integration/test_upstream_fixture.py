from __future__ import annotations

import json
from pathlib import Path

from historical_asset_risk.artifacts import load_artifact


def test_synthetic_fixture_satisfies_public_upstream_loader(
    project_root: Path,
) -> None:
    fixture = project_root / "data" / "fixtures" / "upstream_run"
    manifest = json.loads((fixture / "run_manifest.json").read_text(encoding="utf-8"))
    quality = json.loads(
        (fixture / "data_quality_report.json").read_text(encoding="utf-8")
    )

    frame = load_artifact(
        fixture / "simple_returns.csv",
        fixture / "run_manifest.json",
    )

    assert manifest["fixture"]["synthetic"] is True
    assert manifest["project"] == "historical-asset-risk-engine"
    assert manifest["project_version"] == "0.1.0"
    assert manifest["data_source"]["instruments"] == ["SPY", "IEF", "GLD"]
    assert list(frame.columns) == ["SPY", "IEF", "GLD"]
    assert quality["common_date_count_after_alignment"] == 4958
    assert quality["first_common_date"] == "2007-01-01"
    assert quality["last_common_date"] == "2025-12-31"
    assert not frame.isna().any(axis=None)
    assert frame.index.is_monotonic_increasing
    assert frame.index.is_unique
