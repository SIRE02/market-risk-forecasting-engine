# Market Risk Forecasting Engine

This repository is implementing a reproducible Python research package for
one-session-ahead variance and Value at Risk forecasts. The frozen v0.1 study
compares rolling historical benchmarks with EWMA, Gaussian GARCH(1,1), and
Student-t GARCH(1,1) on three ETFs and a 60/30/10 constant-weight return proxy.

The project forecasts risk, not returns, prices, trades, or profitable
strategies. Its final test is a frozen historical pseudo-out-of-sample study,
not live or prospective forecasting.

## Current implementation status

Phase 2 is the current implementation boundary. It provides:

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
- deterministic benchmark forecast IDs and stable CSV/Parquet artifacts;
- typed benchmark failures without clipping or model substitution;
- frozen protocol documentation and CI.

EWMA and GARCH candidate models are intentionally deferred to later phases.
The public `run` command remains reserved until it can satisfy the complete
experiment-artifact contract.

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
