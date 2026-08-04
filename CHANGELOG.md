# Changelog

## 0.1.0 - 2026-08-04

- Add a newcomer-first explanation of the two-project data and forecasting
  workflow.
- Add a matching historical acquisition configuration for the checked-in
  forecasting example.
- Verify the acquisition and forecasting configurations agree on tickers,
  ordering, paths, and date coverage.
- Use experiment protocol 2.0 with arbitrary ordered instrument universes.
- Support disabled or dynamically weighted constant-weight portfolio proxies.
- Derive input-history requirements from the largest model window and
  require eligible targets in every configured experiment period.
- Add protocol-aware report identity and series descriptions.
- Complete transactional experiment execution, saved-artifact reporting, and
  release-workflow verification.
