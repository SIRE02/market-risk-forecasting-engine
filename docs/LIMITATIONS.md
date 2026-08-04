# Limitations

The engine uses daily asset returns and a squared one-session return as a noisy
realized-variance proxy. An optional portfolio series is a constant-weight
return projection, not a holdings ledger, and omits costs, taxes, financing,
and cash.

The final test is historical pseudo-out-of-sample evidence. It is not live,
prospective, regulatory, or a guarantee of future performance. VaR is evaluated
only at the declared confidence levels, and the current protocol does not score
Expected Shortfall as a primary claim.

The model set deliberately excludes expected-return models, multivariate
GARCH, asymmetric volatility models, machine learning, intraday data, and
portfolio optimization.
