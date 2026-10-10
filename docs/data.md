# Data

`data/fixtures/upstream_run/` contains deterministic synthetic data committed
only for offline tests. It follows the public upstream v0.2.1 artifact contract
and is not provider market data.

It is a handcrafted consumer fixture, not evidence that HARE acquired these
values. Offline integration tests also invoke HARE's real CSV pipeline and
validate its emitted manifest, quality report, return values, and calendar
rejection behavior.

Real provider artifacts belong under `data/upstream/`, which Git ignores.
Experiment runs preserve upstream manifests, quality evidence, and checksums,
but provider data must not be committed when redistribution is prohibited.
