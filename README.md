# Market Risk Forecasting Engine

A reproducible Python research package for one-session-ahead variance and
Value at Risk (VaR) forecasts. The frozen v0.1 study compares rolling
historical benchmarks with EWMA, Gaussian GARCH(1,1), and Student-t GARCH(1,1)
on SPY, IEF, GLD, and a 60/30/10 constant-weight return proxy.

In the frozen historical final test, all three candidate models had lower
aggregate variance QLIKE and 95% VaR pinball loss than their corresponding
historical benchmarks on their pairwise common panels. The 95% moving-block
bootstrap intervals for those mean differences were below zero. This is
historical pseudo-out-of-sample evidence, not a live forecast, a trading
strategy, a regulatory model, or a guarantee of future performance.

Student-t GARCH deserves an explicit caveat: its final-test availability was
88.40%, versus 100% for EWMA and Gaussian GARCH. Thirty-five scheduled
Student-t refits violated the frozen stationarity rule, producing 700 failed
forecasts that remain visible in the artifacts and report.

The complete empirical results are in
[`reports/research_report.md`](reports/research_report.md).

## Five-minute technical overview

The engine consumes the immutable output of
`historical-asset-risk-engine` v0.1.0, validates its schema, manifest, quality
status, hashes, dates, and installed package identity, then builds an
experiment-specific dataset manifest. It never downloads market data.

For each observed forecast origin, it creates the next-session target and:

- computes 252-return historical variance and 500-return historical-simulation
  VaR benchmarks;
- updates an EWMA variance process with fixed lambda 0.94;
- fits zero-mean Gaussian and Student-t GARCH(1,1) models to rolling
  1,250-return windows every 20 origins;
- carries fixed GARCH parameters forward between scheduled refits while
  updating state with newly observed returns;
- records forecasts, failures, fit diagnostics, realizations, and exact
  validation/final-test membership;
- evaluates variance with QLIKE and VaR with lower-tail pinball loss;
- reports strict VaR exceptions, Kupiec coverage, Christoffersen independence,
  and deterministic moving-block bootstrap comparisons.

`run` writes numerical artifacts transactionally. An existing experiment
directory is reused only when its experiment manifest matches exactly and all
declared checksums verify; materially different inputs are never overwritten.
`report` reads saved artifacts without refitting or changing numerical files.
`reproduce` performs validation, numerical execution, report generation, and a
final checksum verification in one command.

## Installation

Python 3.12 is required.

```powershell
conda env create -f environment.yml
conda activate market-risk-forecasting-engine
python -m pip install -e .
```

To update an existing environment:

```powershell
conda env update -n market-risk-forecasting-engine -f environment.yml --prune
conda activate market-risk-forecasting-engine
python -m pip install -e .
```

The upstream dependency is pinned immutably to tag `v0.1.0` and full commit
`f8a1b91b3f3e1e74040c232d8841397d0f032508`.

## Commands

Validate the committed offline fixture without writing an experiment:

```powershell
market-risk-forecast validate-input --config config.example.toml
```

Run numerical forecasting and evaluation:

```powershell
market-risk-forecast run --config config.example.toml
```

The destination is the `[experiment].output_dir` value in the configuration.

Generate a report strictly from that run's saved artifacts:

```powershell
market-risk-forecast report --experiment-dir outputs/risk-v01-example
```

Execute and verify the complete offline workflow:

```powershell
market-risk-forecast reproduce --config config.example.toml
```

Run `market-risk-forecast <command> --help` for command options.

### Custom instrument protocol

The frozen v0.1 configuration remains restricted to SPY, IEF, GLD, and its
60/30/10 proxy. New universes must opt into `experiment.protocol_version =
"2.0"`. Protocol v2 accepts any ordered upstream instrument universe and
supports either a dynamic constant-weight proxy or no proxy.

The included [`configs/aapl_msft_gld.toml`](configs/aapl_msft_gld.toml) profile
consumes the historical-engine output in
`../historical-asset-risk-engine/outputs/aapl-msft-gld` without constructing an
arbitrary combined portfolio:

```powershell
market-risk-forecast validate-input --config configs/aapl_msft_gld.toml
```

Protocol v2 derives minimum data coverage from the largest configured model
window and requires at least one eligible forecast target in each declared
development, validation, and test period. The current v2 milestone changes the
instrument and portfolio contract only; the vetted v0.1 model and evaluation
parameters remain fixed.

## Experiment artifacts

A complete experiment contains copies of the validated upstream manifest and
quality report, dataset and experiment manifests, forecast windows,
realizations, forecasts, fit diagnostics, six evaluation tables, three
figures, the Markdown research report, and `run_manifest.json`.

The run manifest records package and source identity, dependency versions,
configuration, upstream hashes, schema inventory, model settings, seed,
execution state, and the SHA-256 digest of every declared artifact. Failed
forecasts and retry diagnostics are retained rather than clipped, substituted,
or silently discarded.

## Development and release checks

```powershell
python -m ruff format --check .
python -m ruff check .
python -m mypy src
python -m pytest -q
python -m build
```

The frozen protocol and acceptance criteria are defined in the local
`docs/market-risk-forecasting-engine-implementation-spec.md`. That
implementation-planning file is intentionally excluded from Git; this README,
the empirical report, source, tests, configuration, and release metadata are
tracked.
