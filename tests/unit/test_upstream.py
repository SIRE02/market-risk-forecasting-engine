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
from market_risk_forecasting.upstream import (
    UpstreamRun,
    coverage_requirements,
    load_upstream_run,
    sha256_file,
)


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


def _load_run(path: Path, config: ForecastConfig) -> UpstreamRun:
    return load_upstream_run(path, config.upstream, coverage_requirements(config))


def test_accepted_v010_run_loads(copied_run: Path, config: ForecastConfig) -> None:
    run = _load_run(copied_run, config)

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
        ("project_version", "9.9.9", UpstreamVersionIncompatibleError),
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
        _load_run(copied_run, config)


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
        _load_run(copied_run, config)


def test_manifest_ticker_order_mismatch_fails(
    copied_run: Path, config: ForecastConfig
) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["data_source"]["instruments"] = ["IEF", "SPY", "GLD"]

    _mutate_json(copied_run / "run_manifest.json", mutate)

    with pytest.raises(UpstreamManifestInvalidError, match="ordering"):
        _load_run(copied_run, config)


def test_missing_required_file_fails(copied_run: Path, config: ForecastConfig) -> None:
    (copied_run / "data_quality_report.json").unlink()

    with pytest.raises(UpstreamManifestInvalidError, match="missing"):
        _load_run(copied_run, config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("common_date_count_after_alignment", 1251),
        ("first_common_date", "2015-01-01"),
        ("last_common_date", "2019-12-31"),
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
        _load_run(copied_run, config)


def test_missing_returned_instrument_fails(
    copied_run: Path, config: ForecastConfig
) -> None:
    _mutate_json(
        copied_run / "data_quality_report.json",
        lambda quality: quality.__setitem__("returned_instruments", ["SPY", "IEF"]),
    )

    with pytest.raises(UpstreamQualityGateFailedError, match="absent"):
        _load_run(copied_run, config)


def test_provider_return_order_does_not_override_canonical_order(
    copied_run: Path, config: ForecastConfig
) -> None:
    _mutate_json(
        copied_run / "data_quality_report.json",
        lambda quality: quality.__setitem__(
            "returned_instruments", ["GLD", "IEF", "SPY"]
        ),
    )

    run = _load_run(copied_run, config)

    assert run.instrument_order == ("SPY", "IEF", "GLD")


def test_configured_instrument_universe_is_accepted(
    copied_run: Path,
    project_root: Path,
    tmp_path: Path,
) -> None:
    source = (project_root / "config.example.toml").read_text(encoding="utf-8")
    custom_path = tmp_path / "custom.toml"
    custom_path.write_text(
        source.replace(
            'instruments = ["SPY", "IEF", "GLD"]',
            'instruments = ["AAPL", "MSFT", "GLD"]',
        ).replace(
            "weights = { SPY = 0.60, IEF = 0.30, GLD = 0.10 }",
            "weights = { AAPL = 0.60, MSFT = 0.30, GLD = 0.10 }",
        ),
        encoding="utf-8",
    )
    custom = load_config(custom_path)
    returns_path = copied_run / "simple_returns.csv"
    frame = pd.read_csv(returns_path).rename(columns={"SPY": "AAPL", "IEF": "MSFT"})
    frame.to_csv(returns_path, index=False)

    def update_manifest(manifest: dict[str, Any]) -> None:
        manifest["data_source"]["instruments"] = ["AAPL", "MSFT", "GLD"]

    def update_quality(quality: dict[str, Any]) -> None:
        quality["requested_instruments"] = ["AAPL", "MSFT", "GLD"]
        quality["returned_instruments"] = ["AAPL", "MSFT", "GLD"]

    _mutate_json(copied_run / "run_manifest.json", update_manifest)
    _mutate_json(copied_run / "data_quality_report.json", update_quality)

    run = load_upstream_run(
        copied_run,
        custom.upstream,
        coverage_requirements(custom),
    )

    assert run.instrument_order == ("AAPL", "MSFT", "GLD")


def test_forward_fill_declaration_fails(
    copied_run: Path, config: ForecastConfig
) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["estimation_conventions"]["missing_data"]["forward_fill"] = True

    _mutate_json(copied_run / "run_manifest.json", mutate)

    with pytest.raises(UpstreamQualityGateFailedError, match="forward_fill"):
        _load_run(copied_run, config)


def test_duplicate_return_date_fails(copied_run: Path, config: ForecastConfig) -> None:
    path = copied_run / "simple_returns.csv"
    frame = pd.read_csv(path)
    frame.loc[1, "date"] = frame.loc[0, "date"]
    frame.to_csv(path, index=False)

    with pytest.raises(InputDateDuplicateError):
        _load_run(copied_run, config)


def test_unsorted_return_date_fails(copied_run: Path, config: ForecastConfig) -> None:
    path = copied_run / "simple_returns.csv"
    frame = pd.read_csv(path)
    frame.loc[[0, 1], "date"] = frame.loc[[1, 0], "date"].to_numpy()
    frame.to_csv(path, index=False)

    with pytest.raises(InputDateUnsortedError):
        _load_run(copied_run, config)


def test_nonnumeric_return_fails(copied_run: Path, config: ForecastConfig) -> None:
    path = copied_run / "simple_returns.csv"
    frame = pd.read_csv(path)
    frame["SPY"] = frame["SPY"].astype(object)
    frame.loc[0, "SPY"] = "not-a-number"
    frame.to_csv(path, index=False)

    with pytest.raises(InputValueInvalidError):
        _load_run(copied_run, config)


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
        _load_run(copied_run, config)


def test_wrong_installed_package_version_is_typed(
    copied_run: Path,
    config: ForecastConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "market_risk_forecasting.upstream.version",
        lambda _name: "9.9.9",
    )

    with pytest.raises(UpstreamVersionIncompatibleError):
        _load_run(copied_run, config)
