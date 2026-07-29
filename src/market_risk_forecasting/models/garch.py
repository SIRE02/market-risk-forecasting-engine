"""Scheduled zero-mean GARCH(1,1) variance and parametric VaR models."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm, t

from market_risk_forecasting.errors import (
    InputValueInvalidError,
    InsufficientHistoryError,
    InvalidStudentTDofError,
    MarketRiskForecastingError,
    ModelFitFailedError,
    NonfiniteVarianceError,
    NonpositiveVarianceError,
    NonstationaryParametersError,
)

GAUSSIAN_GARCH_MODEL_ID = "garch_1_1_gaussian"
STUDENT_T_GARCH_MODEL_ID = "garch_1_1_student_t"

type GarchDistribution = Literal["gaussian", "student_t"]
type ArchDistributionName = Literal["normal", "studentst"]


def _serialized_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputValueInvalidError(
            f"Serialized GARCH {field} must be a finite number."
        )
    result = float(value)
    if not math.isfinite(result):
        raise InputValueInvalidError(
            f"Serialized GARCH {field} must be a finite number."
        )
    return result


@dataclass(frozen=True)
class GarchParameters:
    """Fixed GARCH parameters in scaled-return units."""

    omega: float
    alpha: float
    beta: float
    degrees_of_freedom: float | None = None

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    def to_dict(self) -> dict[str, float | None]:
        return {
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "degrees_of_freedom": self.degrees_of_freedom,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> GarchParameters:
        degrees_of_freedom = values["degrees_of_freedom"]
        return cls(
            omega=_serialized_float(values["omega"], "omega"),
            alpha=_serialized_float(values["alpha"], "alpha"),
            beta=_serialized_float(values["beta"], "beta"),
            degrees_of_freedom=(
                _serialized_float(degrees_of_freedom, "degrees_of_freedom")
                if degrees_of_freedom is not None
                else None
            ),
        )


@dataclass(frozen=True)
class GarchState:
    """Conditional variance state in scaled-return-squared units."""

    parameters: GarchParameters
    variance_scaled: float

    def to_dict(self) -> dict[str, object]:
        return {
            "parameters": self.parameters.to_dict(),
            "variance_scaled": self.variance_scaled,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> GarchState:
        raw_parameters = values["parameters"]
        if not isinstance(raw_parameters, Mapping):
            raise InputValueInvalidError(
                "Serialized GARCH parameters must be a mapping."
            )
        parameters = GarchParameters.from_dict(raw_parameters)
        return cls(
            parameters=parameters,
            variance_scaled=_serialized_float(
                values["variance_scaled"],
                "variance_scaled",
            ),
        )


@dataclass(frozen=True)
class GarchForecast:
    variance: float
    volatility: float
    return_quantile_0_05: float
    var_0_95: float
    return_quantile_0_01: float
    var_0_99: float


@dataclass(frozen=True)
class GarchFitOutcome:
    """Final result of the deterministic initial attempt and optional retry."""

    state: GarchState | None
    parameters: GarchParameters | None
    converged: bool
    optimizer_status: str
    retry_used: bool
    runtime_seconds: float
    scaling_factor: float
    warning_codes: tuple[str, ...]
    error_code: str | None


def _fit_arch_model(
    scaled_returns: pd.Series,
    distribution_name: ArchDistributionName,
    starting_values: np.ndarray[Any, np.dtype[np.float64]] | None,
) -> Any:
    """Call arch with the exact frozen v0.1 fitting policy."""
    model = arch_model(
        scaled_returns,
        mean="Zero",
        vol="GARCH",
        p=1,
        o=0,
        q=1,
        dist=distribution_name,
        rescale=False,
    )
    return model.fit(
        update_freq=0,
        disp="off",
        show_warning=False,
        tol=1e-8,
        options={"maxiter": 2000},
        starting_values=starting_values,
    )


@dataclass(frozen=True)
class GarchModel:
    """Gaussian or variance-standardized Student-t GARCH(1,1)."""

    distribution: GarchDistribution
    estimation_window: int = 1250
    input_scale: float = 100.0
    retry_count: int = 1
    stationarity_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if self.distribution not in ("gaussian", "student_t"):
            raise InputValueInvalidError(
                f"Unsupported GARCH distribution {self.distribution!r}."
            )
        if self.estimation_window != 1250:
            raise InputValueInvalidError(
                "v0.1 GARCH requires a 1,250-observation estimation window."
            )
        if self.input_scale != 100.0:
            raise InputValueInvalidError("v0.1 GARCH requires input_scale=100.0.")
        if self.retry_count != 1:
            raise InputValueInvalidError("v0.1 GARCH permits exactly one retry.")
        if self.stationarity_tolerance != 1e-8:
            raise InputValueInvalidError(
                "v0.1 GARCH requires stationarity_tolerance=1e-8."
            )

    @property
    def model_id(self) -> str:
        if self.distribution == "gaussian":
            return GAUSSIAN_GARCH_MODEL_ID
        return STUDENT_T_GARCH_MODEL_ID

    @property
    def arch_distribution_name(self) -> ArchDistributionName:
        if self.distribution == "gaussian":
            return "normal"
        return "studentst"

    def fit(self, returns: pd.Series) -> GarchFitOutcome:
        """Fit exactly one rolling window using one deterministic retry."""
        if len(returns) != self.estimation_window:
            raise InsufficientHistoryError(
                "GARCH fitting requires exactly "
                f"{self.estimation_window} returns; received {len(returns)}."
            )
        values = returns.to_numpy(dtype=float, copy=True)
        if not np.isfinite(values).all():
            raise InputValueInvalidError("GARCH fitting returns must be finite.")
        scaled = pd.Series(
            values * self.input_scale,
            index=returns.index,
            name=returns.name,
        )
        started = time.perf_counter()
        last_status = "fit_not_attempted"
        last_optimizer_converged = False
        last_parameters: GarchParameters | None = None
        last_error: MarketRiskForecastingError = ModelFitFailedError(
            "GARCH fit was not attempted."
        )

        for attempt in range(self.retry_count + 1):
            try:
                starting_values = (
                    None if attempt == 0 else self._retry_starting_values(scaled)
                )
                result = _fit_arch_model(
                    scaled,
                    self.arch_distribution_name,
                    starting_values,
                )
                last_status = self._optimizer_status(result)
                last_optimizer_converged = int(result.convergence_flag) == 0
                parameters = self._extract_parameters(result)
                last_parameters = parameters
                if not last_optimizer_converged:
                    raise ModelFitFailedError(
                        "GARCH optimizer did not converge: " + last_status
                    )
                self.validate_parameters(parameters)
                volatility = np.asarray(
                    result.conditional_volatility,
                    dtype=float,
                )
                if volatility.size == 0:
                    raise ModelFitFailedError(
                        "GARCH fit returned no conditional variance state."
                    )
                final_variance = float(volatility[-1] ** 2)
                self._validate_variance(final_variance)
                return GarchFitOutcome(
                    state=GarchState(
                        parameters=parameters,
                        variance_scaled=final_variance,
                    ),
                    parameters=parameters,
                    converged=True,
                    optimizer_status=last_status,
                    retry_used=attempt == 1,
                    runtime_seconds=time.perf_counter() - started,
                    scaling_factor=self.input_scale,
                    warning_codes=(),
                    error_code=None,
                )
            except MarketRiskForecastingError as exc:
                last_error = exc
            except Exception as exc:
                last_error = ModelFitFailedError(
                    f"GARCH optimizer failed: {type(exc).__name__}: {exc}"
                )
                last_status = f"{type(exc).__name__}: {exc}"

        return GarchFitOutcome(
            state=None,
            parameters=last_parameters,
            converged=last_optimizer_converged,
            optimizer_status=last_status,
            retry_used=True,
            runtime_seconds=time.perf_counter() - started,
            scaling_factor=self.input_scale,
            warning_codes=(),
            error_code=last_error.code.value,
        )

    def advance(
        self,
        state: GarchState,
        latest_return: float,
    ) -> tuple[GarchState, GarchForecast]:
        """Update through one observed origin return and forecast its target."""
        self.validate_parameters(state.parameters)
        self._validate_variance(state.variance_scaled)
        if not math.isfinite(latest_return):
            raise InputValueInvalidError("GARCH update return must be finite.")
        scaled_return = latest_return * self.input_scale
        parameters = state.parameters
        next_variance = (
            parameters.omega
            + parameters.alpha * scaled_return**2
            + parameters.beta * state.variance_scaled
        )
        self._validate_variance(next_variance)
        next_state = GarchState(
            parameters=parameters,
            variance_scaled=next_variance,
        )
        return next_state, self.forecast(next_state)

    def forecast(self, state: GarchState) -> GarchForecast:
        """Convert a scaled conditional variance state to public decimal units."""
        self.validate_parameters(state.parameters)
        self._validate_variance(state.variance_scaled)
        variance = state.variance_scaled / self.input_scale**2
        volatility = math.sqrt(variance)
        z_0_05, z_0_01 = self._standardized_quantiles(state.parameters)
        q_0_05 = volatility * z_0_05
        q_0_01 = volatility * z_0_01
        return GarchForecast(
            variance=variance,
            volatility=volatility,
            return_quantile_0_05=q_0_05,
            var_0_95=max(0.0, -q_0_05),
            return_quantile_0_01=q_0_01,
            var_0_99=max(0.0, -q_0_01),
        )

    def validate_parameters(self, parameters: GarchParameters) -> None:
        """Enforce the frozen sign, stationarity, and distribution constraints."""
        numeric = (parameters.omega, parameters.alpha, parameters.beta)
        if not all(math.isfinite(value) for value in numeric):
            raise ModelFitFailedError("GARCH parameters must be finite.")
        if parameters.omega <= 0.0 or parameters.alpha < 0.0 or parameters.beta < 0.0:
            raise ModelFitFailedError("GARCH requires omega > 0 and alpha, beta >= 0.")
        if parameters.persistence >= 1.0 - self.stationarity_tolerance:
            raise NonstationaryParametersError(
                "GARCH alpha + beta must remain below one outside tolerance."
            )
        if self.distribution == "student_t":
            degrees_of_freedom = parameters.degrees_of_freedom
            if (
                degrees_of_freedom is None
                or not math.isfinite(degrees_of_freedom)
                or degrees_of_freedom <= 2.0
            ):
                raise InvalidStudentTDofError(
                    "Student-t GARCH degrees of freedom must be greater than two."
                )
        elif parameters.degrees_of_freedom is not None:
            raise ModelFitFailedError(
                "Gaussian GARCH must not contain Student-t degrees of freedom."
            )

    def _extract_parameters(self, result: Any) -> GarchParameters:
        raw = result.params
        return GarchParameters(
            omega=float(raw["omega"]),
            alpha=float(raw["alpha[1]"]),
            beta=float(raw["beta[1]"]),
            degrees_of_freedom=(
                float(raw["nu"]) if self.distribution == "student_t" else None
            ),
        )

    def _retry_starting_values(
        self,
        scaled_returns: pd.Series,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        sample_variance = float(scaled_returns.var(ddof=1))
        if not math.isfinite(sample_variance) or sample_variance <= 0.0:
            raise NonpositiveVarianceError(
                "GARCH retry initialization variance is not strictly positive."
            )
        values = [0.05 * sample_variance, 0.05, 0.90]
        if self.distribution == "student_t":
            values.append(8.0)
        return np.asarray(values, dtype=float)

    def _standardized_quantiles(
        self,
        parameters: GarchParameters,
    ) -> tuple[float, float]:
        if self.distribution == "gaussian":
            return float(norm.ppf(0.05)), float(norm.ppf(0.01))
        degrees_of_freedom = parameters.degrees_of_freedom
        if degrees_of_freedom is None:
            raise InvalidStudentTDofError(
                "Student-t GARCH degrees of freedom are unavailable."
            )
        scale = math.sqrt((degrees_of_freedom - 2.0) / degrees_of_freedom)
        return (
            float(t.ppf(0.05, degrees_of_freedom)) * scale,
            float(t.ppf(0.01, degrees_of_freedom)) * scale,
        )

    @staticmethod
    def _optimizer_status(result: Any) -> str:
        optimization_result = result.optimization_result
        return (
            f"code={int(result.convergence_flag)}; "
            f"message={str(optimization_result.message)}"
        )

    @staticmethod
    def _validate_variance(variance: float) -> None:
        if not math.isfinite(variance):
            raise NonfiniteVarianceError("GARCH variance is non-finite.")
        if variance <= 0.0:
            raise NonpositiveVarianceError("GARCH variance is not strictly positive.")


__all__ = [
    "GAUSSIAN_GARCH_MODEL_ID",
    "STUDENT_T_GARCH_MODEL_ID",
    "ArchDistributionName",
    "GarchDistribution",
    "GarchFitOutcome",
    "GarchForecast",
    "GarchModel",
    "GarchParameters",
    "GarchState",
]
