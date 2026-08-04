"""Stable exception taxonomy for expected package failures."""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    """Machine-readable error codes exposed by the forecasting engine."""

    CONFIG_INVALID = "CONFIG_INVALID"
    UPSTREAM_PACKAGE_MISSING = "UPSTREAM_PACKAGE_MISSING"
    UPSTREAM_VERSION_INCOMPATIBLE = "UPSTREAM_VERSION_INCOMPATIBLE"
    UPSTREAM_MANIFEST_INVALID = "UPSTREAM_MANIFEST_INVALID"
    UPSTREAM_SCHEMA_INCOMPATIBLE = "UPSTREAM_SCHEMA_INCOMPATIBLE"
    UPSTREAM_QUALITY_GATE_FAILED = "UPSTREAM_QUALITY_GATE_FAILED"
    INPUT_DATE_DUPLICATE = "INPUT_DATE_DUPLICATE"
    INPUT_DATE_UNSORTED = "INPUT_DATE_UNSORTED"
    INPUT_VALUE_INVALID = "INPUT_VALUE_INVALID"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    WINDOW_ALIGNMENT_ERROR = "WINDOW_ALIGNMENT_ERROR"
    MODEL_FIT_FAILED = "MODEL_FIT_FAILED"
    MODEL_FORECAST_FAILED = "MODEL_FORECAST_FAILED"
    NONSTATIONARY_PARAMETERS = "NONSTATIONARY_PARAMETERS"
    INVALID_STUDENT_T_DOF = "INVALID_STUDENT_T_DOF"
    NONFINITE_VARIANCE = "NONFINITE_VARIANCE"
    NONPOSITIVE_VARIANCE = "NONPOSITIVE_VARIANCE"
    COVERAGE_TEST_UNDEFINED = "COVERAGE_TEST_UNDEFINED"
    OUTPUT_COLLISION = "OUTPUT_COLLISION"
    ARTIFACT_RECONCILIATION_FAILED = "ARTIFACT_RECONCILIATION_FAILED"


class MarketRiskForecastingError(Exception):
    """Base class for expected, concise CLI failures."""

    code: ErrorCode

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"{self.code.value}: {self.message}"


class ConfigInvalidError(MarketRiskForecastingError):
    code = ErrorCode.CONFIG_INVALID


class UpstreamPackageMissingError(MarketRiskForecastingError):
    code = ErrorCode.UPSTREAM_PACKAGE_MISSING


class UpstreamVersionIncompatibleError(MarketRiskForecastingError):
    code = ErrorCode.UPSTREAM_VERSION_INCOMPATIBLE


class UpstreamManifestInvalidError(MarketRiskForecastingError):
    code = ErrorCode.UPSTREAM_MANIFEST_INVALID


class UpstreamSchemaIncompatibleError(MarketRiskForecastingError):
    code = ErrorCode.UPSTREAM_SCHEMA_INCOMPATIBLE


class UpstreamQualityGateFailedError(MarketRiskForecastingError):
    code = ErrorCode.UPSTREAM_QUALITY_GATE_FAILED


class InputDateDuplicateError(MarketRiskForecastingError):
    code = ErrorCode.INPUT_DATE_DUPLICATE


class InputDateUnsortedError(MarketRiskForecastingError):
    code = ErrorCode.INPUT_DATE_UNSORTED


class InputValueInvalidError(MarketRiskForecastingError):
    code = ErrorCode.INPUT_VALUE_INVALID


class InsufficientHistoryError(MarketRiskForecastingError):
    code = ErrorCode.INSUFFICIENT_HISTORY


class WindowAlignmentError(MarketRiskForecastingError):
    code = ErrorCode.WINDOW_ALIGNMENT_ERROR


class ModelFitFailedError(MarketRiskForecastingError):
    code = ErrorCode.MODEL_FIT_FAILED


class ModelForecastFailedError(MarketRiskForecastingError):
    code = ErrorCode.MODEL_FORECAST_FAILED


class NonstationaryParametersError(MarketRiskForecastingError):
    code = ErrorCode.NONSTATIONARY_PARAMETERS


class InvalidStudentTDofError(MarketRiskForecastingError):
    code = ErrorCode.INVALID_STUDENT_T_DOF


class NonfiniteVarianceError(MarketRiskForecastingError):
    code = ErrorCode.NONFINITE_VARIANCE


class NonpositiveVarianceError(MarketRiskForecastingError):
    code = ErrorCode.NONPOSITIVE_VARIANCE


class CoverageTestUndefinedError(MarketRiskForecastingError):
    code = ErrorCode.COVERAGE_TEST_UNDEFINED


class OutputCollisionError(MarketRiskForecastingError):
    code = ErrorCode.OUTPUT_COLLISION


class ArtifactReconciliationFailedError(MarketRiskForecastingError):
    code = ErrorCode.ARTIFACT_RECONCILIATION_FAILED


__all__ = [
    "ArtifactReconciliationFailedError",
    "ConfigInvalidError",
    "CoverageTestUndefinedError",
    "ErrorCode",
    "InputDateDuplicateError",
    "InputDateUnsortedError",
    "InputValueInvalidError",
    "InsufficientHistoryError",
    "InvalidStudentTDofError",
    "MarketRiskForecastingError",
    "ModelFitFailedError",
    "ModelForecastFailedError",
    "NonfiniteVarianceError",
    "NonpositiveVarianceError",
    "NonstationaryParametersError",
    "OutputCollisionError",
    "UpstreamManifestInvalidError",
    "UpstreamPackageMissingError",
    "UpstreamQualityGateFailedError",
    "UpstreamSchemaIncompatibleError",
    "UpstreamVersionIncompatibleError",
    "WindowAlignmentError",
]
