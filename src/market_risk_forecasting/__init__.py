"""Market risk forecasting research package."""

from market_risk_forecasting.config import ForecastConfig, load_config
from market_risk_forecasting.datasets import (
    ResearchDataset,
    build_research_dataset,
    persist_dataset_manifest,
)
from market_risk_forecasting.upstream import (
    UpstreamCoverageRequirements,
    UpstreamRun,
    coverage_requirements,
    load_upstream_run,
)

__version__ = "0.1.0"

__all__ = [
    "ForecastConfig",
    "ResearchDataset",
    "UpstreamCoverageRequirements",
    "UpstreamRun",
    "__version__",
    "build_research_dataset",
    "coverage_requirements",
    "load_config",
    "load_upstream_run",
    "persist_dataset_manifest",
]
