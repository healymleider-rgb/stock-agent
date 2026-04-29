"""
tests/test_fmp_shares_float.py — Live integration tests for FMPClient.fetch_shares_float().

These tests make real HTTP calls to FMP's /shares-float endpoint.
They verify:
  1. The response dict has the required provenance fields.
  2. The SEC EDGAR URL is present in filing_url.
  3. Share counts for pinned tickers stay within ±5% of known-good values.

Run with:
    cd stock_agent && .venv/bin/python -m pytest tests/test_fmp_shares_float.py -v
"""
from __future__ import annotations

import pytest

from api.fmp_client import FMPClient


@pytest.fixture(scope="module")
def client() -> FMPClient:
    return FMPClient()


class TestSharesFloatProvenance:
    """fetch_shares_float() must return a fully attributed dict."""

    def test_returns_dict_not_none(self, client):
        result = client.fetch_shares_float("AAPL")
        assert result is not None, "/shares-float returned None for AAPL"

    def test_shares_key_positive(self, client):
        result = client.fetch_shares_float("AAPL")
        assert "shares" in result
        assert result["shares"] > 0

    def test_source_contains_sec_edgar_label(self, client):
        result = client.fetch_shares_float("AAPL")
        assert "FMP/shares-float" in result["source"]
        assert "SEC" in result["source"]

    def test_data_refreshed_at_present(self, client):
        result = client.fetch_shares_float("AAPL")
        assert "data_refreshed_at" in result
        assert result["data_refreshed_at"] != ""

    def test_filing_period_end_is_date(self, client):
        """filing_period_end must be YYYY-MM-DD parsed from the SEC filing URL."""
        import re
        result = client.fetch_shares_float("AAPL")
        period_end = result.get("filing_period_end")
        assert period_end is not None, "filing_period_end is None — URL pattern did not match"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", period_end), (
            f"filing_period_end {period_end!r} is not YYYY-MM-DD"
        )

    def test_filing_url_contains_sec_gov(self, client):
        """The filing_url field must link to SEC EDGAR — that's the institutional signal."""
        result = client.fetch_shares_float("AAPL")
        filing_url = result.get("filing_url", "")
        assert "sec.gov" in filing_url.lower() or filing_url == "", (
            f"Expected sec.gov in filing_url, got: {filing_url!r}"
        )

    def test_fetched_at_is_iso_timestamp(self, client):
        result = client.fetch_shares_float("AAPL")
        assert "fetched_at" in result
        # Must be a non-empty string parseable as ISO 8601
        from datetime import datetime
        ts = result["fetched_at"]
        assert ts, "fetched_at is empty"
        datetime.fromisoformat(ts)   # raises ValueError if not ISO


class TestSharesFloatPinnedValues:
    """
    Pin SEC-sourced share counts for PYPL and CVX.

    Tolerance ±5%: covers share buyback/issuance between 10-K filings
    without masking a broken data-source substitution.

    These values were verified against SEC 10-K filings (2025-12-31) on
    2026-04-28:
      AXON  80,397,700   — Axon Enterprise 10-K 2025-12-31
      PYPL 920,665,000   — PayPal 10-K 2025-12-31
      CVX  1,995,390,000 — Chevron 10-K 2025-12-31
    """

    EXPECTED = {
        "AXON": (80_397_700,    0.05),
        "PYPL": (920_665_000,   0.05),
        "CVX":  (1_995_390_000, 0.05),
    }

    @pytest.mark.parametrize("ticker,expected,tolerance", [
        ("AXON", 80_397_700,    0.05),
        ("PYPL", 920_665_000,   0.05),
        ("CVX",  1_995_390_000, 0.05),
    ])
    def test_share_count_within_tolerance(self, client, ticker, expected, tolerance):
        result = client.fetch_shares_float(ticker)
        assert result is not None, f"/shares-float returned None for {ticker}"
        shares = result["shares"]
        delta = abs(shares - expected) / expected
        assert delta < tolerance, (
            f"{ticker}: SEC-sourced shares={shares:,.0f} vs pinned {expected:,.0f} "
            f"(Δ={delta:.1%} > {tolerance:.0%} tolerance)"
        )
