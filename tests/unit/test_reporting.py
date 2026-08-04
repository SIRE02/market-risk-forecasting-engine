from __future__ import annotations

from market_risk_forecasting.reporting import _report_identity


def test_custom_report_identity_uses_configured_series() -> None:
    effective = {
        "experiment": {"protocol_version": "2.0"},
        "upstream": {"instruments": ["AAPL", "MSFT", "GLD"]},
        "portfolio_proxy": {"enabled": False},
    }

    title, series = _report_identity(effective)

    assert title == "# Market Risk Forecasting Engine - Protocol v2.0 Report"
    assert series == "AAPL, MSFT, GLD"


def test_report_identity_includes_configured_proxy() -> None:
    effective = {
        "experiment": {"protocol_version": "2.0"},
        "upstream": {"instruments": ["SPY", "IEF", "GLD"]},
        "portfolio_proxy": {"enabled": True, "series_id": "FIXTURE_PROXY"},
    }

    title, series = _report_identity(effective)

    assert title == "# Market Risk Forecasting Engine - Protocol v2.0 Report"
    assert series == "SPY, IEF, GLD, FIXTURE_PROXY"
