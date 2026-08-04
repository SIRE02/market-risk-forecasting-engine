# Configuration

The forecasting engine reads one TOML file for each experiment. A real-data
run normally uses a second TOML file for the upstream
`historical-asset-risk-engine` acquisition step.

## Paired data workflow

The repository includes a matching pair:

- `configs/upstream/four_assets.toml` acquires and prepares market data.
- `configs/four_assets.toml` validates that output and runs forecasting.

Run both commands from the repository root:

```powershell
historical-asset-risk --config configs/upstream/four_assets.toml
market-risk-forecast reproduce --config configs/four_assets.toml
```

The acquisition `output_dir` must equal the forecasting `input_run_dir`. The
instrument names and their order must also match.

## Experiment

`[experiment]` defines the run identity, paths, and deterministic random seed.

```toml
[experiment]
experiment_id = "risk-example"
input_run_dir = "data/fixtures/upstream_run"
output_dir = "outputs/risk-example"
random_seed = 42
```

Input and output directories must differ. Existing results are reused only
when their saved manifest matches the requested experiment exactly; a
different configuration cannot overwrite them.

## Upstream contract

`[upstream]` declares the public artifact contract and ordered instrument
universe expected from `historical-asset-risk-engine`.

```toml
[upstream]
project = "historical-asset-risk-engine"
package_version = "0.1.0"
simple_returns_schema_id = "historical-asset-risk/simple-returns"
simple_returns_schema_version = "1.experimental"
simple_returns_units = "decimal_return_per_observation"
instruments = ["SPY", "IEF", "GLD"]
```

The engine checks the installed dependency, manifest, schema, units, file
checksums, instrument order, dates, and values against this declaration.

## Experiment periods

`[periods]` contains three strictly ordered, non-overlapping date ranges:
development, validation, and test. There are no package-level calendar-date
limits. Coverage is derived from these dates and the configured model windows.

Each period must contain at least one forecast target with enough prior
observations for the largest configured window. Select the periods before
examining comparative results.

## Portfolio proxy

Set `enabled = false` to forecast only the upstream instruments:

```toml
[portfolio_proxy]
enabled = false
```

To add a daily constant-weight return proxy, provide a unique series ID and
one non-negative weight for every instrument:

```toml
[portfolio_proxy]
enabled = true
series_id = "TECH_GOLD"
weights = { AAPL = 0.50, MSFT = 0.30, GLD = 0.20 }
```

Weights must sum to one. The proxy is a return projection, not a rebalanced
holdings ledger.

## Model controls

The supplied configuration values are reproducible defaults, not enforced
study constants.

| Setting | Valid values | Purpose |
| --- | --- | --- |
| `historical.variance_window` | integer >= 2 | Sample-variance lookback |
| `historical.var_window` | integer >= 2 | Historical-simulation lookback |
| `historical.quantile_method` | `linear`, `lower`, `higher`, `nearest`, or `midpoint` | Quantile interpolation |
| `ewma.lambda` | number strictly between 0 and 1 | EWMA decay |
| `ewma.initialization_window` | integer >= 2 | Initial sample-variance window |
| `garch.estimation_window` | integer >= 2 | Rolling fit window |
| `garch.refit_every_origins` | integer >= 1 | Refit interval |
| `garch.input_scale` | positive number | Temporary estimation scale |
| `garch.retry_count` | `0` or `1` | Optional deterministic retry |
| `garch.stationarity_tolerance` | number in `[0, 1)` | Persistence boundary tolerance |

Changing a model control changes the effective experiment configuration and
therefore requires a new experiment ID and output directory.

## Evaluation controls

The moving-block bootstrap accepts a positive block length, a positive number
of resamples, and a confidence value strictly between zero and one. It uses
`experiment.random_seed` for deterministic resampling.

The current artifact schema deliberately reports 95% and 99% VaR. Those two
levels are supported model outputs, not configurable TOML fields.

## Input coverage

Minimum common history is derived from the largest historical, EWMA, or GARCH
window. The input must also cover all configured experiment periods. This
replaces the former fixed date and observation-count gates.

Run validation without producing forecasts:

```powershell
market-risk-forecast validate-input --config config.example.toml
```
