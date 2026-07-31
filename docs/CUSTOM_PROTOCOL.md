# Custom Protocol v2

Protocol v2 generalizes the experiment universe without changing the published
frozen v0.1 study or its saved evidence.

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

Unlike frozen v0.1, protocol v2 derives minimum coverage from the largest
configured model window. Each development, validation, and test period must
contain at least one target observation with that many prior returns. This
prevents a syntactically valid but unusable short dataset from reaching model
orchestration.

## Current boundary

Protocol v2 currently generalizes instruments, periods, and the optional
portfolio proxy. Model definitions and evaluation parameters retain the vetted
v0.1 values. Changing those numerical rules is a separate protocol revision.
