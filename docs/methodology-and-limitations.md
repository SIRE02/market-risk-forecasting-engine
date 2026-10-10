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

The real-data study declares `upstream.market_calendar_id = "XNYS"`. The
adapter verifies every aligned price date, including the first date omitted
from returns, is a consecutive session on HARE's versioned calendar. Other
inputs without a declared calendar retain observation-based timing; the
synthetic weekday fixture is not an exchange-session dataset.

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

If experiment periods leave gaps, the GARCH filter still advances through
every intervening observed return before the next evaluated forecast.

Valid GARCH parameters require positive omega, non-negative alpha and beta,
and persistence below one outside the configured tolerance. Student-t degrees
of freedom must exceed two, and its quantiles are standardized to unit
variance. Public outputs are converted back to decimal-return units.

## Evaluation

Variance forecasts are scored against squared one-session returns using QLIKE,
squared error, and absolute error. Lower-tail quantiles use pinball loss. VaR
coverage includes exception counts, Kupiec unconditional coverage,
Christoffersen independence, and combined conditional coverage. Kupiec uses
every available forecast. Independence and conditional coverage use only
transitions between consecutive available forecasts, and conditional coverage
is a single likelihood ratio on those transitions rather than the sum of the
Kupiec and independence statistics, so both parts share one sample when
missing forecasts break the chain.

Loss scores and calibration answer different questions. Lower pinball loss
means a quantile forecast performed better under that scoring rule; it does
not guarantee that exceptions occurred at exactly the nominal rate. Likewise,
failure to reject a coverage test is evidence consistent with calibration at
the chosen significance level, not proof of perfect calibration. Coverage
results must also be read alongside exception independence and forecast
availability.

An exception occurs only when realized loss is strictly greater than reported
VaR. Undefined independence cases are labelled `insufficient_events` rather
than receiving invented statistics.

Independence transitions require the later forecast's origin to equal the
earlier target date. Failed forecasts break that chain and exception clusters;
the table records `excluded_gap_transition_count`. Kupiec coverage still uses
all available forecasts. These are availability-conditional results, and
missing forecasts can be informative about model performance.

Candidate-minus-benchmark loss differences use a deterministic moving-block
bootstrap. Block length, resample count, interval confidence, and random seed
come from the effective experiment configuration. Every series is aligned to
the period's full target-date grid. One set of contiguous date-block indices
is drawn for all assets in each replicate, preserving contemporaneous
dependence. Unavailable paired forecasts remain missing cells on that grid;
they are not compressed into adjacent observations. Replicates average the
available asset/date losses, using the same weighting as the point estimate.
Replicates with no available losses are redrawn. The intervals describe
performance conditional on available forecasts, not performance on failed
forecast dates.

Confidence intervals and bootstrap fractions are left unavailable when the
period has only one possible date block, fewer than two observed dates, or
fewer than two resamples. Descriptive mean/median effects and paired counts
remain available; a short period is not represented as a zero-width interval.

This engine models decimal simple returns. HARE's descriptive asset-volatility
tables use log returns, so their levels should not be treated as identical
quantities. Squared returns are a noisy second-moment proxy under the models'
zero-mean assumption; rolling sample variance is a centered benchmark.

## Reproducibility

Every run records the effective configuration, dependency versions, source
identity, input and output checksums, dataset lineage, forecast windows, fit
diagnostics, forecasts, realizations, evaluation tables, and report artifacts.
Reports are generated only from the saved and checksummed numerical outputs.

After an evaluation-only correction, use
`tools/reevaluate_saved_experiment.py --source-dir SOURCE --config CONFIG --output-dir NEW`
to create a separate run from verified saved forecasts. Input checksums and
fitting controls must match; GARCH period gaps require a fresh numerical run.
The new manifests and report preserve the original forecast source identity,
dependency versions, and copied-artifact hashes alongside the new evaluation
identity. The original experiment is not overwritten.
Repeated reevaluations retain the original fitting identity, dependencies,
and configuration, while separately recording the immediate parent
evaluation. Normal run/reproduce commands can reconcile a corrected run when
its material identity and preserved forecast checksums still match.
The human-readable report surfaces the acquisition timestamp, observed source
coverage, upstream return checksum, forecasting commit, source-tree state, and
source-tree fingerprint from those manifests.

The generated report includes forecast volatility with realized-return
context, realized loss with 95% VaR forecasts and strict exceptions, rolling
candidate-minus-benchmark QLIKE and pinball differences, per-series bootstrap
effects, and 95%/99% VaR calibration. Development, validation, and final-test
periods are shaded on every dated view, whose horizontal axis ends at the last
observed target date.

The rolling comparison uses a balanced panel. A target date contributes only
when every configured series has a valid candidate forecast, benchmark
forecast, and realization; the configured window is then applied on the
canonical eligible-date grid. This prevents changes in the set of available
assets from masquerading as changes in model performance. A negative rolling
difference means the candidate had the lower average loss over that window.

The volatility view uses absolute one-session returns as noisy points and a
21-session rolling root-mean-square return only as smoother context; neither is
an observation of the latent one-session variance forecast target. Aggregate
pinball charts use basis points for legibility while report tables retain the
unscaled loss units.

## Limitations

Squared one-session returns are noisy realized-variance proxies. The optional
portfolio series is a constant-weight return projection rather than a holdings
ledger and omits costs, taxes, financing, and cash management.

Results are historical pseudo-out-of-sample evidence, not live forecasts,
regulatory capital estimates, trading advice, or guarantees of future
performance. The engine does not model Expected Shortfall, expected returns,
multivariate or asymmetric volatility, intraday data, machine learning, or
portfolio optimization.

## References

- Bollerslev, T. (1986). *Generalized autoregressive conditional
  heteroskedasticity*. Journal of Econometrics, 31(3), 307–327.
- Christoffersen, P. F. (1998). *Evaluating interval forecasts*.
  International Economic Review, 39(4), 841–862.
- Engle, R. F. (1982). *Autoregressive conditional heteroscedasticity with
  estimates of the variance of United Kingdom inflation*. Econometrica, 50(4),
  987–1007.
- J.P. Morgan/Reuters (1996). *RiskMetrics Technical Document*, fourth edition.
- Kupiec, P. H. (1995). *Techniques for verifying the accuracy of risk
  measurement models*. The Journal of Derivatives, 3(2), 73–84.
- Patton, A. J. (2011). *Volatility forecast comparison using imperfect
  volatility proxies*. Journal of Econometrics, 160(1), 246–256.
