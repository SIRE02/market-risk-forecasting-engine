from __future__ import annotations

from market_risk_forecasting.reporting import _report_identity


def test_custom_report_identity_uses_configured_series() -> None:
    effective = {
        "experiment": {"protocol_version": "2.0"},
        "upstream": {"instruments": ["AAPL", "MSFT", "GLD"]},
        "portfolio_proxy": {"enabled": False},
    }

    title, series, frozen = _report_identity(effective)

    assert title == "# Market Risk Forecasting Engine - Protocol v2.0 Report"
    assert series == "AAPL, MSFT, GLD"
    assert frozen is False


def test_legacy_report_identity_preserves_frozen_title_and_proxy() -> None:
    effective = {
        "experiment": {"experiment_id": "risk-v01-frozen"},
        "upstream": {"instruments": ["SPY", "IEF", "GLD"]},
        "portfolio_proxy": {"series_id": "MIX_60_30_10"},
    }

    title, series, frozen = _report_identity(effective)

    assert title == "# Market Risk Forecasting Engine - Frozen v0.1 Research Report"
    assert series == "SPY, IEF, GLD, MIX_60_30_10"
    assert frozen is True
