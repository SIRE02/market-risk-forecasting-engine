# Changelog

This file records user-visible changes for each published package version.

## Unreleased

- Include shorter EWMA/GARCH windows in realization coverage, suppress
  bootstrap inference with insufficient date blocks, and preserve original
  forecast provenance through repeated reevaluations. Corrected runs support
  normal run/reproduce reconciliation with provenance integrity checks.
- Preserve cross-asset dependence and missing dates in moving-block bootstrap
  comparisons; exclude gaps from independence transitions and exception clusters.
- Advance GARCH states through observations between configured periods.
- Validate the declared XNYS session calendar for the real-data study and
  exercise HARE's actual offline output contract in integration tests.
- Pin `historical-asset-risk-engine` to the v0.2.1 release commit and require
  that version in forecasting configurations and synthetic fixture metadata.
- Resolve the acquisition output path relative to its configuration folder
  so it still matches the forecasting input directory.
- Update upstream compatibility tests and documentation for v0.2.1 while
  retaining the `1.experimental` handoff schemas and decimal-return units.

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
