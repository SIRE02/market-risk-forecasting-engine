# Market Risk Forecasting Engine - Frozen v0.1 Research Report

## Direct answer

On the aggregate final-test common panel, EWMA (lambda 0.94), Gaussian
GARCH(1,1), and Student-t GARCH(1,1) each had lower one-session variance QLIKE
than their historical-variance benchmark. They also each had lower 95% VaR
pinball loss than the historical-simulation benchmark. Every corresponding 95%
moving-block bootstrap interval for the mean score difference was below zero.

This is frozen historical pseudo-out-of-sample evidence, not live or
prospective forecasting. It is not a trading, regulatory, or future-performance
claim. Lower QLIKE and pinball loss are better.

## Final-test variance comparison

| Candidate | Paired N | Mean difference | 95% interval | Median difference | Bootstrap fraction < 0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EWMA (lambda 0.94) | 6,032 | -0.296416 | [-0.472626, -0.163650] | -0.0706564 | 1 |
| Gaussian GARCH(1,1) | 6,032 | -0.300655 | [-0.481386, -0.166831] | -0.0790460 | 1 |
| Student-t GARCH(1,1) | 5,332 | -0.299128 | [-0.491548, -0.150439] | -0.0693377 | 1 |

Differences are candidate score minus benchmark score on explicit pairwise
common dates. Negative values favor the candidate.

## Final-test 95% VaR comparison

| Candidate | Paired N | Mean difference | 95% interval | Median difference | Bootstrap fraction < 0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EWMA (lambda 0.94) | 6,032 | -0.000102383 | [-0.000180999, -0.000038704] | -0.0000557025 | 1 |
| Gaussian GARCH(1,1) | 6,032 | -0.000130849 | [-0.000232644, -0.000058500] | -0.0000432671 | 1 |
| Student-t GARCH(1,1) | 5,332 | -0.000114126 | [-0.000218905, -0.000036863] | -0.0000512173 | 1 |

## Validation results

Variance QLIKE:

| Candidate | Paired N | Mean difference | 95% interval | Median difference | Bootstrap fraction < 0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EWMA (lambda 0.94) | 5,032 | -0.106580 | [-0.183042, -0.033913] | -0.0645289 | 1 |
| Gaussian GARCH(1,1) | 5,032 | -0.147147 | [-0.232676, -0.076887] | -0.0102766 | 1 |
| Student-t GARCH(1,1) | 5,032 | -0.150897 | [-0.237430, -0.080526] | -0.0162385 | 1 |

95% VaR pinball loss:

| Candidate | Paired N | Mean difference | 95% interval | Median difference | Bootstrap fraction < 0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EWMA (lambda 0.94) | 5,032 | -0.000074237 | [-0.000103996, -0.000045525] | -0.0000394859 | 1 |
| Gaussian GARCH(1,1) | 5,032 | -0.000078382 | [-0.000106317, -0.000050389] | -0.0000138612 | 1 |
| Student-t GARCH(1,1) | 5,032 | -0.000084245 | [-0.000112386, -0.000056000] | -0.0000241642 | 1 |

Validation and final-test aggregates remain separate; validation observations
are not pooled into the final-test claims.

## Forecast availability

| Model | Eligible | Valid | Failed | Availability |
| --- | ---: | ---: | ---: | ---: |
| EWMA (lambda 0.94) | 6,032 | 6,032 | 0 | 100.00% |
| Gaussian GARCH(1,1) | 6,032 | 6,032 | 0 | 100.00% |
| Student-t GARCH(1,1) | 6,032 | 5,332 | 700 | 88.40% |
| Historical simulation (500) | 6,032 | 6,032 | 0 | 100.00% |
| Historical variance (252) | 6,032 | 6,032 | 0 | 100.00% |

Student-t GARCH's lower availability prevents interpreting the aggregate table
as a full-panel contest. Its 700 failed forecasts came from 35 scheduled
refits that violated the frozen stationarity rule. Failed dates remain in the
availability counts and are excluded only from explicit pairwise common-date
score comparisons.

## Final-test 95% VaR coverage by series

| Series | Model | N | Exception rate | Kupiec p | Independence status |
| --- | --- | ---: | ---: | ---: | --- |
| GLD | EWMA (lambda 0.94) | 1,508 | 0.04973 | 0.9623 | ok |
| GLD | Gaussian GARCH(1,1) | 1,508 | 0.05703 | 0.2201 | ok |
| GLD | Student-t GARCH(1,1) | 1,508 | 0.06034 | 0.07382 | ok |
| GLD | Historical simulation (500) | 1,508 | 0.06167 | 0.04447 | ok |
| IEF | EWMA (lambda 0.94) | 1,508 | 0.05172 | 0.7599 | ok |
| IEF | Gaussian GARCH(1,1) | 1,508 | 0.05305 | 0.5903 | ok |
| IEF | Student-t GARCH(1,1) | 1,488 | 0.05578 | 0.3148 | ok |
| IEF | Historical simulation (500) | 1,508 | 0.05703 | 0.2201 | ok |
| MIX_60_30_10 | EWMA (lambda 0.94) | 1,508 | 0.04973 | 0.9623 | ok |
| MIX_60_30_10 | Gaussian GARCH(1,1) | 1,508 | 0.05504 | 0.3766 | ok |
| MIX_60_30_10 | Student-t GARCH(1,1) | 1,348 | 0.05267 | 0.6555 | ok |
| MIX_60_30_10 | Historical simulation (500) | 1,508 | 0.05637 | 0.2659 | ok |
| SPY | EWMA (lambda 0.94) | 1,508 | 0.05902 | 0.1177 | ok |
| SPY | Gaussian GARCH(1,1) | 1,508 | 0.05836 | 0.1465 | ok |
| SPY | Student-t GARCH(1,1) | 988 | 0.05972 | 0.1733 | ok |
| SPY | Historical simulation (500) | 1,508 | 0.04841 | 0.7756 | ok |

Equality between realized loss and VaR is not an exception. Christoffersen
results with zero required transition cells are labelled
`insufficient_events`.

## Fit diagnostics

| Model | Scheduled fits | Optimizer converged | Retries | Failed forecasts |
| --- | ---: | ---: | ---: | ---: |
| EWMA (lambda 0.94) | 4 | 4 | 0 | 0 |
| Gaussian GARCH(1,1) | 708 | 708 | 0 | 0 |
| Student-t GARCH(1,1) | 708 | 708 | 35 | 700 |

A converged optimizer result can still fail the frozen parameter rules.
Nonstationary fits are retained as failed forecasts and stale parameters are
not reused.

## Method and traceability

- Series: SPY, IEF, GLD, and MIX_60_30_10.
- Forecast horizon: one observed session.
- Variance benchmark: 252-return sample variance.
- VaR benchmark: 500-return historical simulation.
- Candidates: EWMA, Gaussian GARCH(1,1), Student-t GARCH(1,1).
- Primary variance score: QLIKE.
- Primary VaR score: 5% lower-tail pinball loss.
- Uncertainty: moving-block bootstrap, block length 20, 2,000 resamples,
  seed 42.
- Every aggregate is traceable through the saved forecasts, realizations,
  evaluation tables, experiment manifest, and checksum-bearing run manifest.

The experiment-local generated report also contains deterministic QLIKE,
pinball-loss, and availability figures. The tracked report records the same
saved-artifact results without committing generated experiment directories.

## Limitations

Squared one-session returns are noisy realized-variance proxies. The portfolio
series is a constant-weight return projection rather than a holdings ledger.
The study excludes costs, taxes, financing, Expected Shortfall scoring,
asymmetric or multivariate volatility models, and all trading or regulatory
claims. Historical results do not guarantee future performance.
