# Market Risk Forecasting Engine

A reproducible Python research tool for one-session-ahead variance and Value
at Risk (VaR) forecasting. It compares transparent historical benchmarks with
EWMA, Gaussian GARCH(1,1), and Student-t GARCH(1,1), evaluates the forecasts,
and produces a traceable research report.

This is historical research software. It is not a live trading system,
investment advice, a regulatory model, or a guarantee of future performance.

## Why are there two projects?

Reliable forecasting starts with reliable data, but downloading and cleaning
market data is a different responsibility from fitting and evaluating risk
models. The workflow deliberately separates them:

| Project | Responsibility | Network access |
| --- | --- | --- |
| `historical-asset-risk-engine` | Download adjusted prices, clean and align observations, calculate returns, and record data-quality evidence | Yes, when Yahoo is selected |
| `market-risk-forecasting-engine` | Validate the saved return artifacts, run forecasts, evaluate models, and generate the research report | No |

The separation provides a stable boundary:

```text
Yahoo Finance
    -> historical-asset-risk-engine
    -> simple_returns.csv + quality report + run manifest
    -> market-risk-forecasting-engine
    -> forecasts + evaluation tables + research report
```

You do not need to clone both repositories. Installing this repository also
installs the pinned historical package and its `historical-asset-risk` command.
The historical repository must remain publicly accessible because it is a Git
dependency.

## Quickstart from a clean computer

Run every command in this section from the root of this repository.

### 1. Install prerequisites

You need:

- Git;
- Miniconda or Anaconda;
- internet access for installation and the Yahoo download;
- a platform supported by Python 3.12 and the declared dependencies.

Clone the repository:

```powershell
git clone https://github.com/SIRE02/market-risk-forecasting-engine.git
Set-Location market-risk-forecasting-engine
```

Create and activate the environment:

```powershell
conda env create -f environment.yml
conda activate market-risk-forecasting-engine
python -m pip install -e .
```

Confirm that installation provided both commands:

```powershell
historical-asset-risk --help
market-risk-forecast --help
```

### 2. Download and prepare the example data

The checked-in historical configuration requests AAPL, MSFT, GLD, and TLT
from 2010 through 2025:

```powershell
historical-asset-risk --config configs/upstream/four_assets.toml
```

The command creates:

```text
data/upstream/aapl-msft-gld-tlt-2010-2025/
```

The three files required by forecasting are:

```text
simple_returns.csv
data_quality_report.json
run_manifest.json
```

The historical run also writes adjusted prices, descriptive statistics, and
charts. Its configured `end_date` is `2026-01-01` because provider end dates
are exclusive; the requested data therefore ends in 2025.

Downloaded provider data is intentionally ignored by Git. Every user acquires
their own copy and is responsible for complying with provider terms.

### 3. Validate the handoff

The matching forecasting configuration is
[`configs/four_assets.toml`](configs/four_assets.toml):

```powershell
market-risk-forecast validate-input --config configs/four_assets.toml
```

A successful validation reports:

```text
Input validation: ok
Series: AAPL, MSFT, GLD, TLT
```

Validation checks the historical package version, project identity, schema,
instrument identities and ordering, date coverage, data-quality evidence,
finite return values, observation counts, and SHA-256 checksums.

### 4. Run forecasting, evaluation, and reporting

Use `reproduce` for the complete workflow:

```powershell
market-risk-forecast reproduce --config configs/four_assets.toml
```

GARCH estimation is computationally heavier than input validation, so this
step can take several minutes. The completed experiment is written to:

```text
outputs/risk-aapl-msft-gld-tlt/
```

Start with the generated report:

```text
outputs/risk-aapl-msft-gld-tlt/research_report.md
```

The report starts with time-series plots of forecast volatility against
realized-return context, realized losses against 95% VaR with exceptions
marked, and rolling candidate-versus-benchmark performance. The output
directory also contains forecasts, realizations, fit diagnostics, aggregate
evaluation tables, figures, manifests, and checksums.

## How the paired configurations work

Each real-data example has two matching files:

| Data acquisition | Forecasting |
| --- | --- |
| [`configs/upstream/four_assets.toml`](configs/upstream/four_assets.toml) | [`configs/four_assets.toml`](configs/four_assets.toml) |

The acquisition configuration's `output_dir` must equal the forecasting
configuration's `input_run_dir`. Tickers must also match exactly, including
their ordering. Forecasting rejects accidental mismatches instead of silently
relabeling columns.

## Use your own assets

Copy one pair of example configurations and change both files together.

In the historical configuration, select the provider, tickers, requested
dates, and output directory:

```toml
[analysis]
provider = "yahoo"
tickers = ["NVDA", "AMZN", "GLD", "TLT"]
start_date = "2010-01-01"
end_date = "2026-01-01"
output_dir = "data/upstream/nvda-amzn-gld-tlt-2010-2025"
```

In the forecasting configuration, use the same ordered tickers and directory:

```toml
[experiment]
experiment_id = "risk-nvda-amzn-gld-tlt"
input_run_dir = "data/upstream/nvda-amzn-gld-tlt-2010-2025"
output_dir = "outputs/risk-nvda-amzn-gld-tlt"

[upstream]
instruments = ["NVDA", "AMZN", "GLD", "TLT"]
```

Keep the remaining schema identity and model sections from the example file.
Important rules:

- every asset needs sufficient shared history;
- every period needs targets with enough prior returns for the largest
  configured model window;
- development, validation, and test periods must each contain eligible target
  observations;
- choose experiment periods before inspecting comparative results;
- use a new experiment ID and output directory for a materially different run.

The example keeps only individual assets by setting
`portfolio_proxy.enabled = false`. The engine can also add a dynamic
constant-weight proxy. See the
[`configuration guide`](docs/configuration.md) for the complete contract.

## Documentation

- [`Configuration`](docs/configuration.md) explains the paired data workflow,
  experiment contract, portfolio proxies, and input validation.
- [`Methodology and limitations`](docs/methodology-and-limitations.md) documents the models,
  forecast timing, evaluation rules, statistical inference, and research
  boundaries.
- [`Data`](docs/data.md) distinguishes committed synthetic fixtures from
  ignored provider artifacts.
- [`Changelog`](docs/changelog.md) records published versions.

## Commands

| Command | Purpose |
| --- | --- |
| `historical-asset-risk --config FILE` | Acquire and prepare upstream returns |
| `market-risk-forecast validate-input --config FILE` | Validate without writing an experiment |
| `market-risk-forecast run --config FILE` | Run numerical forecasting and evaluation |
| `market-risk-forecast report --experiment-dir DIR` | Generate a report from saved numerical artifacts |
| `market-risk-forecast reproduce --config FILE` | Validate, run, report, and verify in one command |

`run` and `reproduce` never overwrite a materially different existing
experiment. If you change data or configuration, choose a new experiment ID
and output directory.

## What the forecasting engine does

For every configured series, the engine:

- computes configurable-window historical variance and historical-simulation
  VaR benchmarks;
- updates an EWMA variance process with configurable decay and initialization;
- fits zero-mean Gaussian and Student-t GARCH(1,1) models using configurable
  rolling windows, refit intervals, scaling, and retry behavior;
- creates strictly next-observation targets without future-data leakage;
- evaluates variance with QLIKE and VaR with lower-tail pinball loss;
- reports VaR exceptions, Kupiec coverage, Christoffersen independence, and
  deterministic moving-block bootstrap comparisons;
- plots forecast and realized histories, VaR exceptions, and rolling model
  advantage so changes through time are visible before the aggregate tables;
- retains failed fits and diagnostics instead of silently substituting another
  model.

Instruments, periods, the optional portfolio proxy, model windows, EWMA decay,
GARCH fitting controls, bootstrap settings, and the deterministic seed are
configurable. The current artifact schema reports 95% and 99% VaR.

## Reproducibility and artifacts

The forecasting engine consumes the public
`historical-asset-risk/simple-returns` artifact contract. It records:

- the effective experiment configuration;
- the installed package and source identities;
- upstream and generated-file checksums;
- dataset lineage and quality adjustments;
- exact training windows, forecast origins, and target dates;
- optimizer outcomes, retries, failures, and availability;
- separate validation and final-test results.

Downloaded data belongs under `data/upstream/`, and experiment results belong
under `outputs/`. Both directories are ignored by Git.

## Troubleshooting

### A command is not recognized

Activate the environment and reinstall the editable package:

```powershell
conda activate market-risk-forecasting-engine
python -m pip install -e .
```

### Installation cannot fetch the historical package

Confirm Git and internet access are available and that the pinned
`historical-asset-risk-engine` commit is public.

### Forecasting says required upstream files are missing

Run the matching `historical-asset-risk --config ...` command first and verify
that the acquisition `output_dir` matches the forecasting `input_run_dir`.

### Instruments or ordering do not match

Make the historical `tickers` list and forecasting `instruments` list
identical. Ordering is part of the artifact contract.

### History is insufficient

Choose assets with longer shared histories or request an earlier start date.
Newly listed assets can shorten the common aligned dataset for every series.

### An output directory already exists

Completed runs are treated as immutable evidence. Use a new experiment ID and
output directory rather than overwriting the old run.

## Development

Run the release checks from the repository root:

```powershell
python -m ruff format --check .
python -m ruff check .
python -m mypy src
python -m pytest -q
python -m build
```

Release history is recorded in [`docs/changelog.md`](docs/changelog.md).
