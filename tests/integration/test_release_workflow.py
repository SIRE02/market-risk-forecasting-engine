from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_risk_forecasting import experiment
from market_risk_forecasting.cli import main, validate_input
from market_risk_forecasting.config import ForecastConfig, load_config
from market_risk_forecasting.errors import (
    ArtifactReconciliationFailedError,
    OutputCollisionError,
)
from market_risk_forecasting.experiment import (
    NUMERICAL_ARTIFACT_NAMES,
    REQUIRED_COMPLETE_ARTIFACT_NAMES,
    artifact_sha256,
    execute_experiment,
    reevaluate_experiment,
    verify_experiment_directory,
)
from market_risk_forecasting.orchestration import (
    EvaluatedModelArtifacts,
    evaluate_model_artifacts,
    run_benchmarks_and_ewma,
)
from market_risk_forecasting.reporting import generate_report


def _config_path(
    project_root: Path,
    tmp_path: Path,
    *,
    experiment_id: str = "risk-release-test",
) -> Path:
    fixture = project_root / "data" / "fixtures" / "upstream_run"
    output = tmp_path / "experiment"
    source = (project_root / "config.example.toml").read_text(encoding="utf-8")
    path = tmp_path / f"{experiment_id}.toml"
    path.write_text(
        source.replace(
            'experiment_id = "risk-example"',
            f'experiment_id = "{experiment_id}"',
        )
        .replace(
            'input_run_dir = "data/fixtures/upstream_run"',
            f'input_run_dir = "{fixture.as_posix()}"',
        )
        .replace(
            'output_dir = "outputs/risk-example"',
            f'output_dir = "{output.as_posix()}"',
        ),
        encoding="utf-8",
    )
    return path


def _evaluated_benchmarks(
    config: ForecastConfig,
) -> tuple[Any, Any, EvaluatedModelArtifacts]:
    upstream, dataset = validate_input(config)
    models = run_benchmarks_and_ewma(
        dataset=dataset,
        config=config,
        upstream_simple_return_checksum=upstream.checksums["simple_returns.csv"],
    )
    return (
        upstream,
        dataset,
        EvaluatedModelArtifacts(
            models=models,
            evaluation=evaluate_model_artifacts(
                artifacts=models,
                config=config,
            ),
        ),
    )


def test_transactional_run_report_and_reproduce_workflow(
    project_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    actual_source_identity = experiment.source_identity
    actual_dependencies = experiment.dependency_versions
    fitting_source = {
        "git_commit": "a" * 40,
        "git_dirty": True,
        "source_tree_sha256": "b" * 64,
    }
    fitting_dependencies = {
        "python": "3.12.0",
        "arch": "7.2.0",
        "fitting_backend": "original",
    }
    monkeypatch.setattr(experiment, "source_identity", lambda: fitting_source)
    monkeypatch.setattr(experiment, "dependency_versions", lambda: fitting_dependencies)
    config_path = _config_path(project_root, tmp_path)
    config = load_config(config_path)
    upstream, dataset, evaluated = _evaluated_benchmarks(config)

    def use_precomputed(**kwargs: object) -> EvaluatedModelArtifacts:
        del kwargs
        return evaluated

    monkeypatch.setattr(
        experiment,
        "run_available_models_with_evaluation",
        use_precomputed,
    )
    result = execute_experiment(
        config=config,
        upstream=upstream,
        dataset=dataset,
    )

    assert not result.reused
    assert result.run_manifest["state"] == "numerical_complete"
    assert all(
        (result.output_dir / name).is_file() for name in NUMERICAL_ARTIFACT_NAMES
    )
    numerical_hashes = {
        name: artifact_sha256(result.output_dir / name)
        for name in NUMERICAL_ARTIFACT_NAMES
    }

    def refit_forbidden(**kwargs: object) -> EvaluatedModelArtifacts:
        del kwargs
        raise AssertionError("an identical experiment must not refit")

    monkeypatch.setattr(
        experiment,
        "run_available_models_with_evaluation",
        refit_forbidden,
    )
    reconciled = execute_experiment(
        config=config,
        upstream=upstream,
        dataset=dataset,
    )
    assert reconciled.reused

    report = generate_report(result.output_dir)
    assert not report.reused
    complete = verify_experiment_directory(
        result.output_dir,
        require_complete=True,
    )
    assert complete["state"] == "complete"
    assert complete["report_source_identity"]["source_tree_sha256"]
    assert complete["report_dependency_versions"]["python"]
    assert all(
        (result.output_dir / name).is_file()
        for name in REQUIRED_COMPLETE_ARTIFACT_NAMES
    )
    assert numerical_hashes == {
        name: artifact_sha256(result.output_dir / name)
        for name in NUMERICAL_ARTIFACT_NAMES
    }
    report_text = report.report_path.read_text(encoding="utf-8")
    assert "## Direct answer" in report_text
    assert "## Forecasts through time" in report_text
    assert "## 95% VaR through time" in report_text
    assert "## Rolling model advantage" in report_text
    assert "## Final-test results by series" in report_text
    assert "## VaR calibration by series" in report_text
    assert "## Run identity and data lineage" in report_text
    assert "### 99% VaR coverage" in report_text
    assert "balanced panel" in report_text
    assert "Values below zero mean the candidate performed better" in report_text
    assert "historical pseudo-out-of-sample evidence" in report_text
    assert "Kupiec" in report_text
    assert "Upstream `simple_returns.csv` SHA-256" in report_text
    assert "Numerical source commit" in report_text
    assert "regulatory" in report_text

    assert main(["run", "--config", str(config_path)]) == 0
    assert (
        main(
            [
                "report",
                "--experiment-dir",
                str(result.output_dir),
            ]
        )
        == 0
    )
    assert main(["reproduce", "--config", str(config_path)]) == 0
    captured = capsys.readouterr()
    assert "Reproduction verification: ok" in captured.out
    assert captured.err == ""

    monkeypatch.setattr(experiment, "source_identity", actual_source_identity)
    monkeypatch.setattr(experiment, "dependency_versions", actual_dependencies)
    reevaluated_config = replace(
        config,
        experiment=replace(config.experiment, output_dir=tmp_path / "reevaluated"),
    )
    reevaluated = reevaluate_experiment(
        source_dir=result.output_dir, config=reevaluated_config
    )
    assert reevaluated.run_manifest["forecast_provenance"]["source_directory"] == str(
        result.output_dir
    )
    assert (
        artifact_sha256(reevaluated.output_dir / "forecasts.parquet")
        == numerical_hashes["forecasts.parquet"]
    )
    generate_report(reevaluated.output_dir)
    reevaluated_text = (reevaluated.output_dir / "research_report.md").read_text(
        encoding="utf-8"
    )
    assert "forecasts were reused unchanged" in reevaluated_text
    assert "Original forecast source-tree SHA-256" in reevaluated_text
    verify_experiment_directory(reevaluated.output_dir, require_complete=True)
    assert numerical_hashes == {
        name: artifact_sha256(result.output_dir / name)
        for name in NUMERICAL_ARTIFACT_NAMES
    }
    assert reevaluate_experiment(
        source_dir=result.output_dir, config=reevaluated_config
    ).reused
    reused = execute_experiment(
        config=reevaluated_config, upstream=upstream, dataset=dataset
    )
    assert reused.reused
    reused_config_path = tmp_path / "reevaluated.toml"
    reused_config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            f'output_dir = "{config.experiment.output_dir.as_posix()}"',
            f'output_dir = "{reevaluated_config.experiment.output_dir.as_posix()}"',
        ),
        encoding="utf-8",
    )
    assert main(["run", "--config", str(reused_config_path)]) == 0
    assert main(["reproduce", "--config", str(reused_config_path)]) == 0
    again_config = replace(
        config,
        experiment=replace(
            config.experiment, output_dir=tmp_path / "reevaluated-again"
        ),
    )
    again = reevaluate_experiment(
        source_dir=reevaluated.output_dir, config=again_config
    )
    provenance = again.run_manifest["forecast_provenance"]
    assert provenance["source_identity"] == fitting_source
    assert provenance["dependency_versions"] == fitting_dependencies
    assert provenance["effective_configuration"] == config.to_dict()
    assert provenance["original_source_directory"] == str(result.output_dir)
    assert provenance["source_directory"] == str(reevaluated.output_dir)
    assert (
        provenance["parent_evaluation"]["source_identity"] == actual_source_identity()
    )
    assert (
        provenance["parent_evaluation"]["dependency_versions"] == actual_dependencies()
    )
    assert (
        artifact_sha256(again.output_dir / "forecasts.parquet")
        == numerical_hashes["forecasts.parquet"]
    )
    verify_experiment_directory(again.output_dir, require_complete=False)
    assert again.run_manifest["forecast_policies"] == experiment.FORECAST_POLICIES
    generate_report(again.output_dir)
    again_text = (again.output_dir / "research_report.md").read_text(encoding="utf-8")
    assert f"forecasts were reused unchanged from `{result.output_dir}`" in again_text
    assert f"intermediate reevaluation `{reevaluated.output_dir}`" in again_text
    assert "intermediate reevaluation" not in reevaluated_text
    bad_provenance = dict(provenance)
    bad_provenance["preserved_artifacts"] = {
        **provenance["preserved_artifacts"],
        "forecasts.parquet": "0" * 64,
    }
    with pytest.raises(ArtifactReconciliationFailedError, match="provenance checksum"):
        experiment._verify_forecast_provenance(
            again.output_dir, bad_provenance, {"forecast_provenance": bad_provenance}
        )
    with pytest.raises(OutputCollisionError, match="different configuration"):
        reevaluate_experiment(
            source_dir=result.output_dir,
            config=replace(
                reevaluated_config,
                garch=replace(
                    config.garch,
                    refit_every_origins=config.garch.refit_every_origins + 1,
                ),
            ),
        )
    with pytest.raises(OutputCollisionError, match="separate output directory"):
        reevaluate_experiment(source_dir=result.output_dir, config=config)

    report.report_path.write_text(report_text + "\n", encoding="utf-8")
    with pytest.raises(
        ArtifactReconciliationFailedError,
        match="checksum mismatch",
    ):
        verify_experiment_directory(
            result.output_dir,
            require_complete=True,
        )


def test_materially_different_experiment_cannot_overwrite(
    project_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_config = load_config(_config_path(project_root, tmp_path))
    upstream, dataset, evaluated = _evaluated_benchmarks(first_config)
    monkeypatch.setattr(
        experiment,
        "run_available_models_with_evaluation",
        lambda **kwargs: evaluated,
    )
    execute_experiment(
        config=first_config,
        upstream=upstream,
        dataset=dataset,
    )
    second_config = load_config(
        _config_path(
            project_root,
            tmp_path,
            experiment_id="risk-materially-different",
        )
    )

    with pytest.raises(OutputCollisionError, match="materially different"):
        execute_experiment(
            config=second_config,
            upstream=upstream,
            dataset=dataset,
        )


@pytest.mark.parametrize(
    ("refit_after_gap", "accepted"),
    [(True, True), (False, False)],
)
def test_reevaluation_gap_guard_allows_only_unaffected_garch_windows(
    tmp_path: Path,
    refit_after_gap: bool,
    accepted: bool,
) -> None:
    origins = ["2020-01-01", "2020-01-02", "2020-01-06", "2020-01-07"]
    targets = ["2020-01-02", "2020-01-03", "2020-01-07", "2020-01-08"]
    path = tmp_path / "experiment_windows.csv"
    pd.DataFrame(
        {
            "series_id": "SPY",
            "model_id": experiment.GAUSSIAN_GARCH_MODEL_ID,
            "forecast_origin": origins,
            "target_date": targets,
            "scheduled_refit": [True, False, refit_after_gap, False],
        }
    ).to_csv(path, index=False)

    if accepted:
        experiment._require_gap_safe_garch_forecasts(path)
    else:
        with pytest.raises(ArtifactReconciliationFailedError, match="period-gap"):
            experiment._require_gap_safe_garch_forecasts(path)
