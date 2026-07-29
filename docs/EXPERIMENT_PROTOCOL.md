# Frozen v0.1 Experiment Protocol

Protocol ID: `risk-v01-frozen`

The following choices are frozen before final-test aggregates are inspected:

- series: SPY, IEF, GLD, and MIX_60_30_10;
- decimal simple returns and zero conditional mean;
- rolling historical variance (252 observations);
- historical-simulation VaR (500 observations);
- EWMA with lambda 0.94 and a 252-observation initialization;
- Gaussian and Student-t GARCH(1,1), using 1,250 observations and refitting
  every 20 eligible origins;
- 95% primary and 99% secondary VaR;
- QLIKE as the primary variance score and pinball loss as the primary quantile
  score;
- moving-block bootstrap length 20, 2,000 resamples, seed 42, and 95%
  confidence intervals;
- development through 2014, validation from 2015 through 2019, and frozen
  historical final test from 2020 through 2025.

The machine-readable version is
[`../configs/frozen_research.toml`](../configs/frozen_research.toml). Any
material protocol change requires a new experiment ID and configuration file.

