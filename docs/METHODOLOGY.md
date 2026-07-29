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

Window records are classified as development, validation, or test from
`target_date`, never from `forecast_origin`. Realization records begin at the
first 252-observation benchmark origin, while the 500-observation benchmark
joins the corresponding subset. Appending later observations cannot alter any
previous window, identifier, forecast, or realization.

The authoritative model equations, fitting policy, diagnostics, and scoring
rules are frozen in the implementation specification. This document will be
expanded alongside the corresponding implementation phases.
