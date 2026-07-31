"""Market risk forecasting research package."""

from market_risk_forecasting.config import ForecastConfig, load_config
from market_risk_forecasting.datasets import (
    ResearchDataset,
    build_research_dataset,
    persist_dataset_manifest,
)
from market_risk_forecasting.upstream import UpstreamRun, load_upstream_run

__version__ = "0.2.0"

__all__ = [
    "ForecastConfig",
    "ResearchDataset",
    "UpstreamRun",
    "__version__",
    "build_research_dataset",
    "load_config",
    "load_upstream_run",
    "persist_dataset_manifest",
]
