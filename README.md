# Market Risk Forecasting Engine

This repository is implementing a reproducible Python research package for
one-session-ahead variance and Value at Risk forecasts. The frozen v0.1 study
compares rolling historical benchmarks with EWMA, Gaussian GARCH(1,1), and
Student-t GARCH(1,1) on three ETFs and a 60/30/10 constant-weight return proxy.

The project forecasts risk, not returns, prices, trades, or profitable
strategies. Its final test is a frozen historical pseudo-out-of-sample study,
not live or prospective forecasting.

## Current implementation status

Phase 5 is the current implementation boundary. It provides:

- installable package and CLI entry point;
- strict TOML configuration parsing;
- typed error and forecast contracts;
- deterministic fit and forecast identifiers;
- an offline synthetic upstream fixture;
- immutable installation of `historical-asset-risk-engine` v0.1.0;
- upstream manifest, schema, quality, date, value, and package validation;
- SHA-256 inventory for every consumed upstream file;
- the three unchanged ETF series and MIX_60_30_10 proxy;
- deterministic dataset-manifest construction;
- a read-only `validate-input` command;
- exact trailing forecast-origin and next-session target windows;
- period classification using target dates;
- realization records with return, squared-return proxy, and loss;
- rolling 252-observation sample-variance forecasts with `ddof=1`;
- 500-observation linear historical-simulation quantiles and positive-loss VaR;
- continuous EWMA variance initialized once from the first 252 returns;
- fixed lambda 0.94 recursion without split or calendar resets;
- zero-mean Gaussian 95% and 99% EWMA VaR in decimal-return units;
- one deterministic EWMA fit identity and diagnostic record per series;
- zero-mean Gaussian and Student-t GARCH(1,1) candidates;
- exact rolling 1,250-return estimation with deterministic 20-origin refits;
- percent-scale fitting with all public forecasts converted to decimal units;
- fixed-parameter state updates between scheduled refits;
- one frozen retry policy and variance-standardized Student-t VaR;
- convergence, parameter, persistence, retry, scaling, and runtime diagnostics;
- typed fit failures that invalidate state until the next scheduled refit;
- combined benchmark and candidate artifacts on common forecast dates;
- QLIKE, squared-error, absolute-error, and lower-tail pinball scores;
- explicit valid, failed, unavailable, and pairwise common-date counts;
- strict VaR exceptions with Kupiec and Christoffersen coverage tests;
- deterministic moving-block bootstrap comparisons using the frozen seed;
- separate validation and final-test results plus predeclared 2020–2025 tables;
- stable CSV schemas for all evaluation and inference artifacts;
- deterministic benchmark forecast IDs and stable CSV/Parquet artifacts;
- typed benchmark failures without clipping or model substitution;
- frozen protocol documentation and CI.

Release orchestration, final reporting, and figures are intentionally deferred
to the final implementation phase. The public `run` command remains reserved
until it can satisfy the complete experiment-artifact contract.

## Development

Python 3.12 is required. Install the package and development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

The published dependency is pinned to upstream tag `v0.1.0` at full commit
`5d189b528306b78b87970e4e83dc2b8dc7b279b3`. The remote installation and
required public `load_artifact` API are verified.

Validate the committed offline fixture without writing output:

```powershell
market-risk-forecast validate-input --config config.example.toml
```

Run the quality gates:

```powershell
python -m ruff format --check .
python -m ruff check .
python -m mypy src
python -m pytest -q
python -m build
```

The controlling scope is
[`docs/market-risk-forecasting-engine-implementation-spec.md`](docs/market-risk-forecasting-engine-implementation-spec.md).
