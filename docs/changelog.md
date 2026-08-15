# Changelog

This file records user-visible changes for each published package version.

## 0.1.0 - 2026-08-15

Initial public release of the market risk forecasting engine.

- Validate released `historical-asset-risk-engine` artifacts through their
  public manifest, schema, quality, and checksum contracts.
- Forecast arbitrary ordered instrument universes with an optional
  constant-weight return proxy.
- Support configurable experiment periods, model windows, EWMA decay, GARCH
  fitting controls, bootstrap settings, and deterministic seeds.
- Compare historical benchmarks, EWMA, Gaussian GARCH(1,1), and Student-t
  GARCH(1,1) using leakage-resistant one-session-ahead windows.
- Evaluate variance, 95% and 99% VaR, coverage, forecast availability, fit
  diagnostics, and moving-block bootstrap comparisons.
- Visualize forecast volatility with realized-return context, realized losses
  with 95% VaR exceptions, and rolling candidate-versus-benchmark performance.
- Produce transactional, checksummed experiment artifacts and a reproducible
  saved-artifact research report.
