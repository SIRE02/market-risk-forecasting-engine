# Methodology

The v0.1 research question is whether simple conditional variance models
improve one-session-ahead variance or 95% VaR forecasts over transparent
rolling historical benchmarks on identical historical target dates.

All model inputs are decimal simple returns from the public
`historical-asset-risk-engine` v0.1.0 `simple_returns.csv` contract. The
research series are SPY, IEF, GLD, and the daily constant-weight proxy:

```text
MIX_60_30_10 = 0.60 * SPY + 0.30 * IEF + 0.10 * GLD
```

The adapter calls the released upstream `load_artifact` function and rejects
incompatible package versions, project or schema identities, instrument
universes, forward filling, insufficient common history, invalid dates or
values, and unreconciled price/return observation counts. It records SHA-256
checksums for the returns, upstream manifest, and quality report. Provider
return ordering is treated as quality evidence; canonical column ordering is
defined by agreement between the upstream manifest and loaded artifact.

A forecast at date `t` may use observations through `t`; its realization is
the return at `t+1`. Every evaluation table is built from saved forecast and
realization records, and pairwise comparisons use only common valid dates.

The permanent variance benchmark uses exactly 252 trailing returns, including
the origin return, and calculates sample variance with `ddof=1`. It rejects
non-finite or non-positive variance rather than clipping it. The permanent VaR
benchmark uses exactly 500 trailing returns and linear empirical 5% and 1%
quantiles. Reported positive-loss VaR is `max(0, -quantile)`; the underlying
return quantile remains unfloored.

The EWMA candidate is initialized once, at its first eligible origin, using
the sample variance (`ddof=1`) of the first 252 returns. Every later origin
updates the prior one-step variance forecast with
`0.94 * prior_variance + 0.06 * origin_return^2`. This state runs continuously
through development, validation, and test without resets. Assuming zero
conditional mean, standard-normal 5% and 1% quantiles convert the variance
forecast to return quantiles and positive-loss VaR. All public values remain
in decimal-return units.

The Gaussian and Student-t candidates use zero-mean GARCH(1,1) variance
recursions. They fit exactly 1,250 returns at the first eligible origin and
every twentieth subsequent origin, multiplying decimal returns by 100 only
during estimation. Between refits, fitted parameters remain fixed while each
new origin return advances the conditional variance state. Public variance,
volatility, quantiles, and VaR are converted back to decimal units.

Each scheduled fit gets the package-default optimizer attempt and at most one
retry using the frozen starting values. Fits must have positive omega,
nonnegative alpha and beta, and persistence below one outside the declared
tolerance. Student-t degrees of freedom must exceed two, and its quantiles are
standardized to unit variance. A failed scheduled fit invalidates the active
state: forecasts remain failed until the next scheduled attempt, and stale
parameters are never reused. Every scheduled attempt is retained in fit
diagnostics.

Evaluation joins forecasts to the identical saved realization keys and keeps
validation and final-test aggregates separate. Variance forecasts are scored
with QLIKE, squared error, and absolute error against squared one-session
returns. Lower-tail 5% and 1% return quantiles use pinball loss. Scores are
reported by series and across all four series; lower values are better.

VaR exceptions use the strict rule `realized_loss > reported_VaR`, so equality
is not an exception. Coverage tables retain exception counts and clusters,
Kupiec unconditional coverage, Christoffersen independence, and combined
conditional coverage. Undefined Christoffersen cases are labelled
`insufficient_events` instead of receiving invented statistics.

Candidate-versus-benchmark inference uses only dates with valid paired
forecasts and the same realization. Failed and unavailable dates remain
visible in availability tables. The paired loss difference is candidate minus
benchmark, and uncertainty uses a deterministic moving-block bootstrap with
block length 20, 2,000 resamples, seed 42, and 95% percentile intervals. For
the all-series result, blocks are sampled within each series before pooling.
The final-test descriptive breakdowns are fixed in advance to calendar years
2020 through 2025.

Window records are classified as development, validation, or test from
`target_date`, never from `forecast_origin`. Realization records begin at the
first 252-observation benchmark origin, while the 500-observation benchmark
joins the corresponding subset. Appending later observations cannot alter any
previous window, identifier, forecast, or realization.

The authoritative model equations, fitting policy, diagnostics, and scoring
rules are frozen in the implementation specification. This document will be
expanded alongside the corresponding implementation phases.
