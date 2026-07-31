"""Transactional experiment execution, lineage manifests, and reconciliation."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from market_risk_forecasting import __version__
from market_risk_forecasting.config import ForecastConfig
from market_risk_forecasting.datasets import (
    ResearchDataset,
    persist_dataset_manifest,
)
from market_risk_forecasting.errors import (
    ArtifactReconciliationFailedError,
    OutputCollisionError,
)
from market_risk_forecasting.evaluation import (
    BOOTSTRAP_COMPARISON_COLUMNS,
    COVERAGE_TEST_COLUMNS,
    FORECAST_AVAILABILITY_COLUMNS,
    PERIOD_BREAKDOWN_COLUMNS,
    QUANTILE_SCORE_COLUMNS,
    VARIANCE_SCORE_COLUMNS,
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
from market_risk_forecasting.orchestration import (
    EXPERIMENT_WINDOW_COLUMNS,
    FIT_DIAGNOSTIC_COLUMNS,
    FORECAST_COLUMNS,
    REALIZATION_COLUMNS,
    persist_evaluated_model_artifacts,
    run_available_models_with_evaluation,
)
from market_risk_forecasting.upstream import UpstreamRun

PROJECT_NAME = "market-risk-forecasting-engine"
RUN_MANIFEST_NAME = "run_manifest.json"
EXPERIMENT_MANIFEST_NAME = "experiment_manifest.json"

NUMERICAL_ARTIFACT_NAMES = (
    "upstream_run_manifest.json",
    "upstream_data_quality_report.json",
    "dataset_manifest.json",
    EXPERIMENT_MANIFEST_NAME,
    "experiment_windows.csv",
    "realizations.parquet",
    "forecasts.parquet",
    "fit_diagnostics.parquet",
    "forecast_availability.csv",
    "variance_scores.csv",
    "quantile_scores.csv",
    "coverage_tests.csv",
    "bootstrap_comparisons.csv",
    "period_breakdowns.csv",
)
REPORT_ARTIFACT_NAMES = (
    "research_report.md",
    "figures/variance_qlike_comparison.png",
    "figures/var_pinball_comparison.png",
    "figures/forecast_availability.png",
)
REQUIRED_COMPLETE_ARTIFACT_NAMES = (
    *NUMERICAL_ARTIFACT_NAMES,
    *REPORT_ARTIFACT_NAMES,
)

_DEPENDENCIES = (
    "arch",
    "historical-asset-risk-engine",
    "matplotlib",
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
)
_MODEL_INVENTORY = (
    HISTORICAL_VARIANCE_MODEL_ID,
    HISTORICAL_SIMULATION_MODEL_ID,
    EWMA_MODEL_ID,
    GAUSSIAN_GARCH_MODEL_ID,
    STUDENT_T_GARCH_MODEL_ID,
)
_DATASET_SCHEMA = {
    "schema_id": "market-risk-forecasting/research-series",
    "schema_version": "1.experimental",
    "units": "decimal_simple_return_per_observation",
}
_ARTIFACT_SCHEMAS: Mapping[str, Mapping[str, Any]] = {
    "upstream_run_manifest.json": {
        "schema_id": "historical-asset-risk/run-manifest",
        "schema_version": "1.experimental",
    },
    "upstream_data_quality_report.json": {
        "schema_id": "historical-asset-risk/data-quality-report",
        "schema_version": "1.experimental",
    },
    "dataset_manifest.json": {
        "schema_id": "market-risk-forecasting/dataset-manifest",
        "schema_version": "1.experimental",
    },
    EXPERIMENT_MANIFEST_NAME: {
        "schema_id": "market-risk-forecasting/experiment-manifest",
        "schema_version": "1.experimental",
    },
    "experiment_windows.csv": {
        "schema_id": "market-risk-forecasting/experiment-windows",
        "schema_version": "1.experimental",
        "columns": list(EXPERIMENT_WINDOW_COLUMNS),
    },
    "realizations.parquet": {
        "schema_id": "market-risk-forecasting/realizations",
        "schema_version": "1.experimental",
        "columns": list(REALIZATION_COLUMNS),
    },
    "forecasts.parquet": {
        "schema_id": "market-risk-forecasting/forecasts",
        "schema_version": "1.experimental",
        "columns": list(FORECAST_COLUMNS),
    },
    "fit_diagnostics.parquet": {
        "schema_id": "market-risk-forecasting/fit-diagnostics",
        "schema_version": "1.experimental",
        "columns": list(FIT_DIAGNOSTIC_COLUMNS),
    },
    "forecast_availability.csv": {
        "schema_id": "market-risk-forecasting/forecast-availability",
        "schema_version": "1.experimental",
        "columns": list(FORECAST_AVAILABILITY_COLUMNS),
    },
    "variance_scores.csv": {
        "schema_id": "market-risk-forecasting/variance-scores",
        "schema_version": "1.experimental",
        "columns": list(VARIANCE_SCORE_COLUMNS),
    },
    "quantile_scores.csv": {
        "schema_id": "market-risk-forecasting/quantile-scores",
        "schema_version": "1.experimental",
        "columns": list(QUANTILE_SCORE_COLUMNS),
    },
    "coverage_tests.csv": {
        "schema_id": "market-risk-forecasting/coverage-tests",
        "schema_version": "1.experimental",
        "columns": list(COVERAGE_TEST_COLUMNS),
    },
    "bootstrap_comparisons.csv": {
        "schema_id": "market-risk-forecasting/bootstrap-comparisons",
        "schema_version": "1.experimental",
        "columns": list(BOOTSTRAP_COMPARISON_COLUMNS),
    },
    "period_breakdowns.csv": {
        "schema_id": "market-risk-forecasting/period-breakdowns",
        "schema_version": "1.experimental",
        "columns": list(PERIOD_BREAKDOWN_COLUMNS),
    },
}


@dataclass(frozen=True)
class ExperimentRunResult:
    output_dir: Path
    reused: bool
    run_manifest: Mapping[str, Any]


def execute_experiment(
    *,
    config: ForecastConfig,
    upstream: UpstreamRun,
    dataset: ResearchDataset,
) -> ExperimentRunResult:
    """Run or reconcile one immutable numerical experiment directory."""
    output_dir = Path(config.experiment.output_dir).resolve()
    experiment_manifest = build_experiment_manifest(
        config=config,
        upstream=upstream,
        dataset=dataset,
    )
    if output_dir.exists():
        return _reconcile_existing(output_dir, experiment_manifest)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.building-",
            dir=output_dir.parent,
        )
    ).resolve()
    if temporary.parent != output_dir.parent.resolve():
        raise OutputCollisionError(
            "Temporary experiment directory escaped the output parent."
        )
    try:
        _write_json(
            temporary / "upstream_run_manifest.json",
            upstream.manifest,
        )
        _write_json(
            temporary / "upstream_data_quality_report.json",
            upstream.quality_report,
        )
        persist_dataset_manifest(dataset, temporary / "dataset_manifest.json")
        _write_json(
            temporary / EXPERIMENT_MANIFEST_NAME,
            experiment_manifest,
        )

        artifacts = run_available_models_with_evaluation(
            dataset=dataset,
            config=config,
            upstream_simple_return_checksum=upstream.checksums["simple_returns.csv"],
        )
        persist_evaluated_model_artifacts(artifacts, temporary)
        run_manifest = build_run_manifest(
            config=config,
            upstream=upstream,
            experiment_manifest=experiment_manifest,
            experiment_dir=temporary,
            state="numerical_complete",
        )
        _write_json(temporary / RUN_MANIFEST_NAME, run_manifest)
        verify_experiment_directory(temporary, require_complete=False)
        temporary.replace(output_dir)
        return ExperimentRunResult(
            output_dir=output_dir,
            reused=False,
            run_manifest=run_manifest,
        )
    except Exception:
        if temporary.exists() and temporary.parent == output_dir.parent.resolve():
            shutil.rmtree(temporary)
        raise


def build_experiment_manifest(
    *,
    config: ForecastConfig,
    upstream: UpstreamRun,
    dataset: ResearchDataset,
) -> Mapping[str, Any]:
    """Build the deterministic identity used to prevent material overwrite."""
    source = source_identity()
    return {
        "schema_id": "market-risk-forecasting/experiment-manifest",
        "schema_version": "1.experimental",
        "project": PROJECT_NAME,
        "package_version": __version__,
        "experiment_id": config.experiment.experiment_id,
        "effective_configuration": config.to_dict(),
        "source_identity": source,
        "upstream": {
            "project": upstream.manifest.get("project"),
            "project_version": upstream.manifest.get("project_version"),
            "installed_package_version": upstream.installed_package_version,
            "git_commit": upstream.manifest.get("git_commit"),
            "checksums": dict(upstream.checksums),
        },
        "dataset": dict(dataset.manifest),
        "model_inventory": list(_MODEL_INVENTORY),
        "random_seeds": {
            "experiment": config.experiment.random_seed,
            "moving_block_bootstrap": config.experiment.random_seed,
        },
    }


def build_run_manifest(
    *,
    config: ForecastConfig,
    upstream: UpstreamRun,
    experiment_manifest: Mapping[str, Any],
    experiment_dir: Path,
    state: str,
    prior_manifest: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Build runtime lineage and checksums for every generated artifact."""
    generated = _generated_artifact_declarations(experiment_dir)
    started_at = (
        prior_manifest.get("execution_started_at")
        if prior_manifest is not None
        else None
    )
    if not isinstance(started_at, str):
        started_at = _utc_now()
    return {
        "schema_id": "market-risk-forecasting/run-manifest",
        "schema_version": "1.experimental",
        "project": PROJECT_NAME,
        "package_version": __version__,
        "experiment_id": config.experiment.experiment_id,
        "state": state,
        "execution_started_at": started_at,
        "last_updated_at": _utc_now(),
        "effective_configuration": config.to_dict(),
        "source_identity": experiment_manifest["source_identity"],
        "upstream": {
            "installed_package_version": upstream.installed_package_version,
            "checksums": dict(upstream.checksums),
        },
        "dependency_versions": dependency_versions(),
        "input_schema_identities": {
            "simple_returns": {
                "schema_id": config.upstream.simple_returns_schema_id,
                "schema_version": config.upstream.simple_returns_schema_version,
                "units": config.upstream.simple_returns_units,
            },
            "dataset": {
                "schema_id": _DATASET_SCHEMA["schema_id"],
                "schema_version": _DATASET_SCHEMA["schema_version"],
                "units": _DATASET_SCHEMA["units"],
            },
        },
        "output_schema_identities": {
            name: dict(schema) for name, schema in _ARTIFACT_SCHEMAS.items()
        },
        "model_inventory": list(_MODEL_INVENTORY),
        "random_seeds": {
            "experiment": config.experiment.random_seed,
            "moving_block_bootstrap": config.experiment.random_seed,
        },
        "generated_artifacts": generated,
        "success": state in {"numerical_complete", "complete"},
    }


def finalize_run_manifest(
    experiment_dir: Path,
    *,
    state: str = "complete",
) -> Mapping[str, Any]:
    """Refresh report checksums after report-only generation."""
    directory = Path(experiment_dir).resolve()
    prior = _load_json(directory / RUN_MANIFEST_NAME)
    refreshed = {
        **prior,
        "state": state,
        "last_updated_at": _utc_now(),
        "generated_artifacts": _generated_artifact_declarations(directory),
        "success": state in {"numerical_complete", "complete"},
    }
    _write_json(directory / RUN_MANIFEST_NAME, refreshed)
    return refreshed


def verify_experiment_directory(
    experiment_dir: Path,
    *,
    require_complete: bool,
) -> Mapping[str, Any]:
    """Verify declared artifacts, checksums, and completion state."""
    directory = Path(experiment_dir).resolve()
    manifest = _load_json(directory / RUN_MANIFEST_NAME)
    state = manifest.get("state")
    permitted = {"complete"} if require_complete else {"numerical_complete", "complete"}
    if state not in permitted:
        raise ArtifactReconciliationFailedError(
            f"Experiment state {state!r} is not one of {sorted(permitted)}."
        )
    generated = manifest.get("generated_artifacts")
    if not isinstance(generated, Mapping):
        raise ArtifactReconciliationFailedError(
            "Run manifest generated_artifacts must be an object."
        )
    required = (
        REQUIRED_COMPLETE_ARTIFACT_NAMES
        if require_complete
        else NUMERICAL_ARTIFACT_NAMES
    )
    missing = [name for name in required if name not in generated]
    if missing:
        raise ArtifactReconciliationFailedError(
            "Run manifest is missing required artifact declarations: "
            + ", ".join(missing)
            + "."
        )
    for name, declaration in generated.items():
        if not isinstance(name, str) or not isinstance(declaration, Mapping):
            raise ArtifactReconciliationFailedError(
                "Run manifest artifact declarations are invalid."
            )
        expected = declaration.get("sha256")
        path = directory / name
        if not path.is_file():
            raise ArtifactReconciliationFailedError(
                f"Declared artifact is missing: {name}."
            )
        actual = artifact_sha256(path)
        if expected != actual:
            raise ArtifactReconciliationFailedError(
                f"Artifact checksum mismatch: {name}."
            )
    figures = directory / "figures"
    if require_complete and not figures.is_dir():
        raise ArtifactReconciliationFailedError(
            "Completed experiment is missing figures/."
        )
    return manifest


def source_identity() -> Mapping[str, Any]:
    """Record commit identity plus a fingerprint that includes dirty source."""
    root = _source_root()
    commit = _git_output(root, "rev-parse", "HEAD")
    status = _git_output(root, "status", "--porcelain", "--untracked-files=normal")
    return {
        "git_commit": commit,
        "git_dirty": bool(status),
        "source_tree_sha256": _source_tree_sha256(root),
    }


def dependency_versions() -> Mapping[str, str]:
    result = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }
    for package in _DEPENDENCIES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactReconciliationFailedError(
            f"Could not checksum artifact {path}: {exc}"
        ) from exc
    return digest.hexdigest()


def _reconcile_existing(
    output_dir: Path,
    expected_experiment_manifest: Mapping[str, Any],
) -> ExperimentRunResult:
    existing_path = output_dir / EXPERIMENT_MANIFEST_NAME
    if not existing_path.is_file():
        raise OutputCollisionError(
            f"Output directory already exists without {EXPERIMENT_MANIFEST_NAME}: "
            f"{output_dir}."
        )
    existing = _load_json(existing_path)
    if existing != expected_experiment_manifest:
        raise OutputCollisionError(
            "Output directory belongs to a materially different experiment: "
            f"{output_dir}."
        )
    run_manifest = verify_experiment_directory(
        output_dir,
        require_complete=False,
    )
    return ExperimentRunResult(
        output_dir=output_dir,
        reused=True,
        run_manifest=run_manifest,
    )


def _source_root() -> Path:
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file():
        return candidate
    return Path(__file__).resolve().parent


def _source_tree_sha256(root: Path) -> str:
    if (root / "src").is_dir():
        paths = [
            *sorted((root / "src").rglob("*.py")),
            *[
                path
                for path in (
                    root / "pyproject.toml",
                    root / "requirements.lock",
                )
                if path.is_file()
            ],
        ]
    else:
        paths = sorted(root.rglob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _git_output(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactReconciliationFailedError(
            f"Could not read experiment JSON {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ArtifactReconciliationFailedError(
            f"Experiment JSON must contain an object: {path}."
        )
    return value


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _generated_artifact_declarations(
    experiment_dir: Path,
) -> Mapping[str, Mapping[str, Any]]:
    return {
        name: {
            "sha256": artifact_sha256(experiment_dir / name),
            "schema": dict(_ARTIFACT_SCHEMAS.get(name, {})),
        }
        for name in REQUIRED_COMPLETE_ARTIFACT_NAMES
        if (experiment_dir / name).is_file()
    }


__all__ = [
    "EXPERIMENT_MANIFEST_NAME",
    "ExperimentRunResult",
    "NUMERICAL_ARTIFACT_NAMES",
    "REPORT_ARTIFACT_NAMES",
    "REQUIRED_COMPLETE_ARTIFACT_NAMES",
    "RUN_MANIFEST_NAME",
    "artifact_sha256",
    "build_experiment_manifest",
    "build_run_manifest",
    "dependency_versions",
    "execute_experiment",
    "finalize_run_manifest",
    "source_identity",
    "verify_experiment_directory",
]
