from __future__ import annotations

from pathlib import Path
from typing import Any

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
    experiment_id: str = "risk-v02-release-test",
) -> Path:
    fixture = project_root / "data" / "fixtures" / "upstream_run"
    output = tmp_path / "experiment"
    source = (project_root / "config.example.toml").read_text(encoding="utf-8")
    path = tmp_path / f"{experiment_id}.toml"
    path.write_text(
        source.replace(
            'experiment_id = "risk-v02-example"',
            f'experiment_id = "{experiment_id}"',
        )
        .replace(
            'input_run_dir = "data/fixtures/upstream_run"',
            f'input_run_dir = "{fixture.as_posix()}"',
        )
        .replace(
            'output_dir = "outputs/risk-v02-example"',
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
    assert "historical pseudo-out-of-sample evidence" in report_text
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
            experiment_id="risk-v02-materially-different",
        )
    )

    with pytest.raises(OutputCollisionError, match="materially different"):
        execute_experiment(
            config=second_config,
            upstream=upstream,
            dataset=dataset,
        )
