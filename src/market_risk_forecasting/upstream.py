"""Validated adapter for historical-asset-risk-engine v0.1.0 artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_risk_forecasting.config import ForecastConfig, UpstreamConfig
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

_PACKAGE_NAME = "historical-asset-risk-engine"
_REQUIRED_FILES = (
    "simple_returns.csv",
    "data_quality_report.json",
    "run_manifest.json",
)
_EARLIEST_ACCEPTABLE_FIRST_DATE = date(2007, 12, 31)
_LATEST_ACCEPTABLE_LAST_DATE = date(2025, 12, 30)
_MINIMUM_COMMON_DATE_COUNT = 4000


@dataclass(frozen=True)
class UpstreamCoverageRequirements:
    """Data coverage required by one experiment protocol."""

    minimum_common_date_count: int
    first_date_not_after: date
    last_date_not_before: date
    forecast_window: int | None = None
    periods: tuple[tuple[str, date, date], ...] = ()


_FROZEN_V01_COVERAGE = UpstreamCoverageRequirements(
    minimum_common_date_count=_MINIMUM_COMMON_DATE_COUNT,
    first_date_not_after=_EARLIEST_ACCEPTABLE_FIRST_DATE,
    last_date_not_before=_LATEST_ACCEPTABLE_LAST_DATE,
)


@dataclass(frozen=True)
class UpstreamRun:
    """Canonical returns and immutable evidence from one accepted upstream run."""

    input_run_dir: Path
    returns: pd.DataFrame
    manifest: Mapping[str, Any]
    quality_report: Mapping[str, Any]
    checksums: Mapping[str, str]
    installed_package_version: str

    @property
    def instrument_order(self) -> tuple[str, ...]:
        return tuple(str(column) for column in self.returns.columns)

    @property
    def first_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.returns.index[0])

    @property
    def last_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.returns.index[-1])


def sha256_file(path: Path) -> str:
    """Calculate a deterministic SHA-256 checksum without loading the file at once."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise UpstreamManifestInvalidError(f"Could not read {path}: {exc}") from exc
    return digest.hexdigest()


def _installed_upstream_version(expected: str) -> str:
    try:
        installed = version(_PACKAGE_NAME)
    except PackageNotFoundError as exc:
        raise UpstreamPackageMissingError(f"{_PACKAGE_NAME} is not installed.") from exc
    if installed != expected:
        raise UpstreamVersionIncompatibleError(
            f"Installed {_PACKAGE_NAME} version {installed!r}; expected {expected!r}."
        )
    return installed


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise UpstreamManifestInvalidError(
            f"Required upstream file is missing: {path.name}."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise UpstreamManifestInvalidError(
            f"Could not parse upstream {label} {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise UpstreamManifestInvalidError(
            f"Upstream {label} must contain a JSON object."
        )
    return value


def _mapping(
    parent: Mapping[str, Any],
    key: str,
    *,
    error_type: type[UpstreamManifestInvalidError]
    | type[UpstreamQualityGateFailedError]
    | type[UpstreamSchemaIncompatibleError],
) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise error_type(f"Upstream evidence is missing object {key!r}.")
    return value


def _ordered_strings(
    value: Any,
    *,
    label: str,
    error_type: type[UpstreamManifestInvalidError]
    | type[UpstreamQualityGateFailedError],
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise error_type(f"{label} must be an ordered array of strings.")
    result = tuple(value)
    if any(not isinstance(item, str) for item in result):
        raise error_type(f"{label} must be an ordered array of strings.")
    return result


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise UpstreamQualityGateFailedError(f"{label} must be an integer.")
    return value


def _iso_date(value: Any, *, label: str) -> date:
    if not isinstance(value, str):
        raise UpstreamQualityGateFailedError(f"{label} must be an ISO date.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise UpstreamQualityGateFailedError(f"{label} must be an ISO date.") from exc


def _validate_manifest(
    manifest: Mapping[str, Any],
    expected: UpstreamConfig,
) -> tuple[str, ...]:
    if manifest.get("project") != expected.project:
        raise UpstreamManifestInvalidError(
            f"Upstream project must be {expected.project!r}."
        )
    if manifest.get("project_version") != expected.package_version:
        raise UpstreamVersionIncompatibleError(
            f"Upstream manifest version must be {expected.package_version!r}."
        )

    data_source = _mapping(
        manifest, "data_source", error_type=UpstreamManifestInvalidError
    )
    instruments = _ordered_strings(
        data_source.get("instruments"),
        label="manifest.data_source.instruments",
        error_type=UpstreamManifestInvalidError,
    )
    if instruments != expected.instruments:
        raise UpstreamManifestInvalidError(
            "Manifest instrument identities or ordering do not match the "
            "configured v0.1 universe."
        )

    schemas = _mapping(
        manifest, "artifact_schemas", error_type=UpstreamSchemaIncompatibleError
    )
    declaration = _mapping(
        schemas,
        "simple_returns.csv",
        error_type=UpstreamSchemaIncompatibleError,
    )
    actual_schema = (
        declaration.get("schema_id"),
        declaration.get("schema_version"),
        declaration.get("units"),
    )
    expected_schema = (
        expected.simple_returns_schema_id,
        expected.simple_returns_schema_version,
        expected.simple_returns_units,
    )
    if actual_schema != expected_schema:
        raise UpstreamSchemaIncompatibleError(
            "simple_returns.csv schema identity, version, or units are incompatible."
        )

    conventions = _mapping(
        manifest,
        "estimation_conventions",
        error_type=UpstreamManifestInvalidError,
    )
    missing_data = _mapping(
        conventions, "missing_data", error_type=UpstreamManifestInvalidError
    )
    if missing_data.get("forward_fill") is not False:
        raise UpstreamQualityGateFailedError(
            "The upstream run must explicitly declare forward_fill=false."
        )
    return instruments


def _validate_quality(
    quality: Mapping[str, Any],
    expected_instruments: tuple[str, ...],
    coverage: UpstreamCoverageRequirements,
) -> None:
    requested = _ordered_strings(
        quality.get("requested_instruments"),
        label="quality.requested_instruments",
        error_type=UpstreamQualityGateFailedError,
    )
    if requested != expected_instruments:
        raise UpstreamQualityGateFailedError(
            "Quality-report requested instruments differ from the v0.1 universe."
        )
    returned = _ordered_strings(
        quality.get("returned_instruments"),
        label="quality.returned_instruments",
        error_type=UpstreamQualityGateFailedError,
    )
    missing = [ticker for ticker in expected_instruments if ticker not in returned]
    if missing:
        raise UpstreamQualityGateFailedError(
            f"Requested instrument(s) absent upstream: {', '.join(missing)}."
        )

    common_count = _integer(
        quality.get("common_date_count_after_alignment"),
        label="quality.common_date_count_after_alignment",
    )
    if common_count < coverage.minimum_common_date_count:
        raise UpstreamQualityGateFailedError(
            f"Common aligned history has {common_count} observations; "
            f"at least {coverage.minimum_common_date_count} are required."
        )
    first_date = _iso_date(
        quality.get("first_common_date"), label="quality.first_common_date"
    )
    if first_date > coverage.first_date_not_after:
        raise UpstreamQualityGateFailedError(
            f"First common date {first_date.isoformat()} is too late; it must be "
            f"on or before {coverage.first_date_not_after.isoformat()}."
        )
    last_date = _iso_date(
        quality.get("last_common_date"), label="quality.last_common_date"
    )
    if last_date < coverage.last_date_not_before:
        raise UpstreamQualityGateFailedError(
            f"Last common date {last_date.isoformat()} is too early; it must be "
            f"on or after {coverage.last_date_not_before.isoformat()}."
        )


def _load_public_artifact(input_run_dir: Path) -> pd.DataFrame:
    try:
        from historical_asset_risk.artifacts import (  # type: ignore[import-untyped]
            load_artifact,
        )
    except ImportError as exc:
        raise UpstreamPackageMissingError(
            "Could not import historical_asset_risk.artifacts.load_artifact."
        ) from exc

    try:
        return load_artifact(
            input_run_dir / "simple_returns.csv",
            input_run_dir / "run_manifest.json",
        )
    except Exception as exc:
        # The upstream public loader reports expected artifact-contract failures
        # through its own exception type, which this package deliberately does not
        # import across the dependency boundary.
        message = str(exc)
        if "missing or nonnumeric values" in message or "non-finite values" in message:
            raise InputValueInvalidError(
                f"Upstream public artifact loader rejected return values: {message}"
            ) from exc
        raise UpstreamSchemaIncompatibleError(
            f"Upstream public artifact loader rejected simple_returns.csv: {message}"
        ) from exc


def _validate_frame(
    frame: pd.DataFrame,
    *,
    manifest_instruments: tuple[str, ...],
    quality: Mapping[str, Any],
    coverage: UpstreamCoverageRequirements,
) -> pd.DataFrame:
    actual_instruments = tuple(str(column) for column in frame.columns)
    if actual_instruments != manifest_instruments:
        raise UpstreamManifestInvalidError(
            "Loaded return columns do not match manifest instrument ordering."
        )
    if frame.index.has_duplicates:
        raise InputDateDuplicateError("Loaded return dates contain duplicates.")
    if not frame.index.is_monotonic_increasing:
        raise InputDateUnsortedError("Loaded return dates are not strictly increasing.")
    values = frame.to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all():
        raise InputValueInvalidError(
            "Loaded returns contain missing, nonnumeric, or non-finite values."
        )
    aligned_price_count = _integer(
        quality.get("common_date_count_after_alignment"),
        label="quality.common_date_count_after_alignment",
    )
    if len(frame) != aligned_price_count - 1:
        raise UpstreamQualityGateFailedError(
            "Loaded return count must be exactly one less than the upstream "
            "aligned-price count."
        )
    if len(frame) == 0:
        raise UpstreamQualityGateFailedError("Loaded returns contain no observations.")

    first_loaded = pd.Timestamp(frame.index[0]).date()
    last_loaded = pd.Timestamp(frame.index[-1]).date()
    first_reported = _iso_date(
        quality.get("first_common_date"), label="quality.first_common_date"
    )
    last_reported = _iso_date(
        quality.get("last_common_date"), label="quality.last_common_date"
    )
    if first_loaded <= first_reported or last_loaded != last_reported:
        raise UpstreamQualityGateFailedError(
            "Loaded return dates do not reconcile with the upstream aligned-price "
            "date range."
        )
    if coverage.forecast_window is not None:
        observed_dates = pd.DatetimeIndex(frame.index).date
        for label, start, end in coverage.periods:
            eligible_positions = [
                position
                for position, observed in enumerate(observed_dates)
                if start <= observed <= end and position >= coverage.forecast_window
            ]
            if not eligible_positions:
                raise UpstreamQualityGateFailedError(
                    f"Configured {label} period has no target observation with "
                    f"{coverage.forecast_window} prior returns."
                )
    return frame.astype(float, copy=True)


def coverage_requirements(config: ForecastConfig) -> UpstreamCoverageRequirements:
    """Derive input gates without changing the immutable frozen-v0.1 contract."""
    if config.is_frozen_v01:
        return _FROZEN_V01_COVERAGE
    maximum_window = max(
        config.historical.variance_window,
        config.historical.var_window,
        config.ewma.initialization_window,
        config.garch.estimation_window,
    )
    periods = config.periods
    return UpstreamCoverageRequirements(
        minimum_common_date_count=maximum_window + 2,
        first_date_not_after=periods.development_end,
        last_date_not_before=periods.test_start,
        forecast_window=maximum_window,
        periods=(
            ("development", periods.development_start, periods.development_end),
            ("validation", periods.validation_start, periods.validation_end),
            ("test", periods.test_start, periods.test_end),
        ),
    )


def load_upstream_run(
    input_run_dir: Path,
    expected: UpstreamConfig,
    coverage: UpstreamCoverageRequirements | None = None,
) -> UpstreamRun:
    """Validate and load one complete upstream v0.1.0 run without network access."""
    run_dir = Path(input_run_dir)
    required_coverage = coverage or _FROZEN_V01_COVERAGE
    missing = [name for name in _REQUIRED_FILES if not (run_dir / name).is_file()]
    if missing:
        raise UpstreamManifestInvalidError(
            f"Required upstream file(s) missing: {', '.join(missing)}."
        )

    installed = _installed_upstream_version(expected.package_version)
    manifest = _load_json(run_dir / "run_manifest.json", label="run manifest")
    quality = _load_json(run_dir / "data_quality_report.json", label="quality report")
    manifest_instruments = _validate_manifest(manifest, expected)
    _validate_quality(quality, expected.instruments, required_coverage)
    frame = _validate_frame(
        _load_public_artifact(run_dir),
        manifest_instruments=manifest_instruments,
        quality=quality,
        coverage=required_coverage,
    )
    checksums = {name: sha256_file(run_dir / name) for name in sorted(_REQUIRED_FILES)}
    if any(
        not isinstance(value, str) or len(value) != 64 for value in checksums.values()
    ):
        raise UpstreamManifestInvalidError("Upstream checksum calculation failed.")

    return UpstreamRun(
        input_run_dir=run_dir,
        returns=frame,
        manifest=manifest,
        quality_report=quality,
        checksums=checksums,
        installed_package_version=installed,
    )


def quality_adjustment_counts(quality: Mapping[str, Any]) -> Mapping[str, int]:
    """Extract the upstream adjustment counts that downstream must preserve."""
    names = (
        "duplicate_date_instrument_rows_removed",
        "invalid_price_values_removed",
        "common_history_rows_removed",
    )
    result: dict[str, int] = {}
    for name in names:
        value = quality.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise UpstreamQualityGateFailedError(
                f"quality.{name} must be a non-negative integer."
            )
        result[name] = value
    return result


__all__ = [
    "UpstreamCoverageRequirements",
    "UpstreamRun",
    "coverage_requirements",
    "load_upstream_run",
    "quality_adjustment_counts",
    "sha256_file",
]
