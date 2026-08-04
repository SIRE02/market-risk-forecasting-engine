# Methodology and limitations

## Research question

The engine tests whether conditional variance models improve one-session-ahead
variance and Value at Risk (VaR) forecasts over transparent rolling historical
benchmarks on identical target dates.

It consumes decimal simple returns from the public
`historical-asset-risk-engine` artifact contract. Research series consist of
the configured instruments and, when enabled, one constant-weight return
proxy.

## Data validation

The adapter checks the installed upstream package, project and schema
identities, units, instrument ordering, checksums, date order, finite values,
forward-fill declarations, and price/return observation reconciliation.
Required history is calculated from the configured model windows and periods;
the engine does not impose fixed calendar dates or a fixed observation count.

## Forecast timing

A forecast made at date `t` may use observations through `t`. Its realization
is the return at the next observed session, `t+1`. Period classification uses
the target date. Appending later observations cannot alter previously created
windows, identifiers, forecasts, or realizations.

Development, validation, and test results remain separate. Pairwise model
comparisons use only dates where both models have a valid forecast for the
same saved realization.

## Models

The historical variance benchmark calculates sample variance with `ddof=1`
over `historical.variance_window` trailing returns. It rejects non-finite or
non-positive results rather than clipping them.

Historical-simulation VaR uses `historical.var_window` trailing returns and
the configured quantile interpolation method. The engine saves 5% and 1%
lower-tail return quantiles and reports positive-loss VaR as
`max(0, -quantile)`.

EWMA initializes from the sample variance of
`ewma.initialization_window` returns and then applies

```text
next_variance = lambda * prior_variance + (1 - lambda) * origin_return^2
```

The state continues across experiment periods without resetting. Gaussian
5% and 1% quantiles convert its variance forecasts to VaR.

The Gaussian and Student-t candidates are zero-mean GARCH(1,1) models. They
use the configured estimation window, refit interval, temporary input scale,
retry policy, and stationarity tolerance. Fitted parameters remain fixed
between scheduled refits while each new return advances conditional variance.
Failed fits remain visible and stale parameters are not reused.

Valid GARCH parameters require positive omega, non-negative alpha and beta,
and persistence below one outside the configured tolerance. Student-t degrees
of freedom must exceed two, and its quantiles are standardized to unit
variance. Public outputs are converted back to decimal-return units.

## Evaluation

Variance forecasts are scored against squared one-session returns using QLIKE,
squared error, and absolute error. Lower-tail quantiles use pinball loss. VaR
coverage includes exception counts, Kupiec unconditional coverage,
Christoffersen independence, and combined conditional coverage.

An exception occurs only when realized loss is strictly greater than reported
VaR. Undefined independence cases are labelled `insufficient_events` rather
than receiving invented statistics.

Candidate-minus-benchmark loss differences use a deterministic moving-block
bootstrap. Block length, resample count, interval confidence, and random seed
come from the effective experiment configuration. Blocks are sampled within
each series before pooled all-series results are calculated.

## Reproducibility

Every run records the effective configuration, dependency versions, source
identity, input and output checksums, dataset lineage, forecast windows, fit
diagnostics, forecasts, realizations, evaluation tables, and report artifacts.
Reports are generated only from the saved and checksummed numerical outputs.

The generated report includes three historical views before its aggregate
tables: forecast volatility with realized-return context, realized loss with
95% VaR forecasts and strict exceptions, and rolling candidate-minus-benchmark
QLIKE and pinball differences. A negative rolling difference means the
candidate had the lower average loss over that window. The volatility view
uses absolute one-session returns as noisy points and a 21-session rolling
root-mean-square return only as smoother context; neither is an observation of
the latent one-session variance forecast target.

## Limitations

Squared one-session returns are noisy realized-variance proxies. The optional
portfolio series is a constant-weight return projection rather than a holdings
ledger and omits costs, taxes, financing, and cash management.

Results are historical pseudo-out-of-sample evidence, not live forecasts,
regulatory capital estimates, trading advice, or guarantees of future
performance. The engine does not model Expected Shortfall, expected returns,
multivariate or asymmetric volatility, intraday data, machine learning, or
portfolio optimization.
