# Changelog

## 0.2.0 - 2026-07-31

- Add opt-in experiment protocol 2.0 with arbitrary ordered instrument
  universes.
- Support disabled or dynamically weighted constant-weight portfolio proxies.
- Derive custom-protocol history requirements from the largest model window and
  require eligible targets in every configured experiment period.
- Preserve the frozen protocol 1.0 configuration and its serialized evidence.
- Add protocol-aware report identity and series descriptions.
- Add AAPL/MSFT/GLD and four-asset example configurations.
- Complete transactional experiment execution, saved-artifact reporting, and
  release-workflow verification.
