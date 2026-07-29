from __future__ import annotations

import json
import shutil
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_risk_forecasting.config import ForecastConfig, load_config
from market_risk_forecasting.errors import (
    InputDateDuplicateError,
    InputDateUnsortedError,
    InputValueInvalidError,
    UpstreamManifestInvalidError,
    UpstreamPackageMissingError,
    UpstreamQualityGateFailedError,
    UpstreamSchemaIncompatibleError,
    UpstreamVersionIncompatibleError,
)
from market_risk_forecasting.upstream import load_upstream_run, sha256_file


@pytest.fixture
def config(project_root: Path) -> ForecastConfig:
    return load_config(project_root / "config.example.toml")


@pytest.fixture
def copied_run(project_root: Path, tmp_path: Path) -> Path:
    source = project_root / "data" / "fixtures" / "upstream_run"
    destination = tmp_path / "upstream_run"
    shutil.copytree(source, destination)
    return destination


def _mutate_json(path: Path, mutation: Any) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_accepted_v010_run_loads(copied_run: Path, config: ForecastConfig) -> None:
    run = load_upstream_run(copied_run, config.upstream)

    assert run.installed_package_version == "0.1.0"
    assert run.instrument_order == ("SPY", "IEF", "GLD")
    assert len(run.returns) == 4957
    assert set(run.checksums) == {
        "data_quality_report.json",
        "run_manifest.json",
        "simple_returns.csv",
    }


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("project", "other-project", UpstreamManifestInvalidError),
        ("project_version", "0.2.0", UpstreamVersionIncompatibleError),
    ],
)
def test_wrong_manifest_identity_fails(
    copied_run: Path,
    config: ForecastConfig,
    field: str,
    value: str,
    error: type[Exception],
) -> None:
    _mutate_json(
        copied_run / "run_manifest.json",
        lambda manifest: manifest.__setitem__(field, value),
    )

    with pytest.raises(error):
        load_upstream_run(copied_run, config.upstream)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_id", "other/schema"),
        ("schema_version", "2"),
        ("units", "percent_return"),
    ],
)
def test_wrong_schema_declaration_fails(
    copied_run: Path,
    config: ForecastConfig,
    field: str,
    value: str,
) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["artifact_schemas"]["simple_returns.csv"][field] = value

    _mutate_json(copied_run / "run_manifest.json", mutate)

    with pytest.raises(UpstreamSchemaIncompatibleError):
        load_upstream_run(copied_run, config.upstream)


def test_manifest_ticker_order_mismatch_fails(
    copied_run: Path, config: ForecastConfig
) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["data_source"]["instruments"] = ["IEF", "SPY", "GLD"]

    _mutate_json(copied_run / "run_manifest.json", mutate)

    with pytest.raises(UpstreamManifestInvalidError, match="ordering"):
        load_upstream_run(copied_run, config.upstream)


def test_missing_required_file_fails(copied_run: Path, config: ForecastConfig) -> None:
    (copied_run / "data_quality_report.json").unlink()

    with pytest.raises(UpstreamManifestInvalidError, match="missing"):
        load_upstream_run(copied_run, config.upstream)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("common_date_count_after_alignment", 3999),
        ("first_common_date", "2008-01-01"),
        ("last_common_date", "2025-12-29"),
    ],
)
def test_quality_threshold_failure(
    copied_run: Path,
    config: ForecastConfig,
    field: str,
    value: int | str,
) -> None:
    _mutate_json(
        copied_run / "data_quality_report.json",
        lambda quality: quality.__setitem__(field, value),
    )

    with pytest.raises(UpstreamQualityGateFailedError):
        load_upstream_run(copied_run, config.upstream)


def test_missing_returned_instrument_fails(
    copied_run: Path, config: ForecastConfig
) -> None:
    _mutate_json(
        copied_run / "data_quality_report.json",
        lambda quality: quality.__setitem__("returned_instruments", ["SPY", "IEF"]),
    )

    with pytest.raises(UpstreamQualityGateFailedError, match="absent"):
        load_upstream_run(copied_run, config.upstream)


def test_provider_return_order_does_not_override_canonical_order(
    copied_run: Path, config: ForecastConfig
) -> None:
    _mutate_json(
        copied_run / "data_quality_report.json",
        lambda quality: quality.__setitem__(
            "returned_instruments", ["GLD", "IEF", "SPY"]
        ),
    )

    run = load_upstream_run(copied_run, config.upstream)

    assert run.instrument_order == ("SPY", "IEF", "GLD")


def test_forward_fill_declaration_fails(
    copied_run: Path, config: ForecastConfig
) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["estimation_conventions"]["missing_data"]["forward_fill"] = True

    _mutate_json(copied_run / "run_manifest.json", mutate)

    with pytest.raises(UpstreamQualityGateFailedError, match="forward_fill"):
        load_upstream_run(copied_run, config.upstream)


def test_duplicate_return_date_fails(copied_run: Path, config: ForecastConfig) -> None:
    path = copied_run / "simple_returns.csv"
    frame = pd.read_csv(path)
    frame.loc[1, "date"] = frame.loc[0, "date"]
    frame.to_csv(path, index=False)

    with pytest.raises(InputDateDuplicateError):
        load_upstream_run(copied_run, config.upstream)


def test_unsorted_return_date_fails(copied_run: Path, config: ForecastConfig) -> None:
    path = copied_run / "simple_returns.csv"
    frame = pd.read_csv(path)
    frame.loc[[0, 1], "date"] = frame.loc[[1, 0], "date"].to_numpy()
    frame.to_csv(path, index=False)

    with pytest.raises(InputDateUnsortedError):
        load_upstream_run(copied_run, config.upstream)


def test_nonnumeric_return_fails(copied_run: Path, config: ForecastConfig) -> None:
    path = copied_run / "simple_returns.csv"
    frame = pd.read_csv(path)
    frame["SPY"] = frame["SPY"].astype(object)
    frame.loc[0, "SPY"] = "not-a-number"
    frame.to_csv(path, index=False)

    with pytest.raises(InputValueInvalidError):
        load_upstream_run(copied_run, config.upstream)


def test_checksums_are_deterministic(copied_run: Path) -> None:
    path = copied_run / "simple_returns.csv"

    first = sha256_file(path)
    second = sha256_file(path)

    assert first == second
    assert len(first) == 64


def test_missing_installed_package_is_typed(
    copied_run: Path,
    config: ForecastConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_name: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr("market_risk_forecasting.upstream.version", missing)

    with pytest.raises(UpstreamPackageMissingError):
        load_upstream_run(copied_run, config.upstream)


def test_wrong_installed_package_version_is_typed(
    copied_run: Path,
    config: ForecastConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "market_risk_forecasting.upstream.version",
        lambda _name: "0.2.0",
    )

    with pytest.raises(UpstreamVersionIncompatibleError):
        load_upstream_run(copied_run, config.upstream)
