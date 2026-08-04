# Custom Protocol v2

Protocol v2 defines the configurable experiment contract used by the engine.

## Paired workflow

Real-data examples are supplied as configuration pairs:

- `configs/upstream/four_assets.toml` downloads and prepares the data;
- `configs/four_assets.toml` validates and forecasts that exact output.

Run both commands from the forecasting repository root:

```powershell
historical-asset-risk --config configs/upstream/four_assets.toml
market-risk-forecast reproduce --config configs/four_assets.toml
```

The historical configuration's `output_dir`, ticker identities, and ticker
ordering must match the forecasting configuration's `input_run_dir` and
`upstream.instruments`.

## Configuration contract

Set `protocol_version = "2.0"` in `[experiment]`, declare the exact ordered
instrument list in `[upstream]`, and configure `[portfolio_proxy]`.

To retain only the individual instruments:

```toml
[portfolio_proxy]
enabled = false
```

To add a daily constant-weight return proxy:

```toml
[portfolio_proxy]
enabled = true
series_id = "TECH_GOLD"
weights = { AAPL = 0.50, MSFT = 0.30, GLD = 0.20 }
```

The proxy weights must contain every configured upstream instrument exactly
once, be non-negative, and sum to one. The proxy series ID must not duplicate
an instrument ID.

## Input validation

The upstream manifest, quality report, and `simple_returns.csv` must agree with
the configured instrument identities and ordering. The adapter retains all
schema, package-version, checksum, finite-value, date-order, no-forward-fill,
and price/return reconciliation checks.

Protocol v2 derives minimum coverage from the largest configured model window.
Each development, validation, and test period must
contain at least one target observation with that many prior returns. This
prevents a syntactically valid but unusable short dataset from reaching model
orchestration.

## Current boundary

Protocol v2 currently generalizes instruments, periods, and the optional
portfolio proxy. Model definitions and evaluation parameters retain their
vetted values. Changing those numerical rules requires a separate protocol
revision.
