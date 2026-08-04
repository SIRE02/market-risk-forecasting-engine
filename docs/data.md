# Data

`data/fixtures/upstream_run/` contains deterministic synthetic data committed
only for offline tests. It follows the public upstream v0.1.0 artifact contract
and is not provider market data.

Real provider artifacts belong under `data/upstream/`, which Git ignores.
Experiment runs preserve upstream manifests, quality evidence, and checksums,
but provider data must not be committed when redistribution is prohibited.
