# Data

`fixtures/upstream_run/` is deterministic synthetic data committed solely for
offline tests. It follows the public upstream v0.1.0 artifact identities and is
not provider market data.

Real provider artifacts belong under `data/upstream/`, which is gitignored.
Experiment runs preserve upstream manifests, quality evidence, and checksums
but must not commit provider data when redistribution is prohibited.
