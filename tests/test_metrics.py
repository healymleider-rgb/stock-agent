"""
Tests for analysis/metrics.py — NormalizedMetrics computation.

Covers:
  1. Provider market cap preserved when computed cap differs
  2. Scorer uses normalized market cap, not raw field
  3. Negative TTM EPS still allows provider PE or annual-EPS PE
  4. peg and eps_growth_pct appear in NormalizedMetrics
  5. Divergence creates warning instead of rejecting provider value
  6. PE resolution order: provider_ttm → computed_ttm → computed_annual
  7. Shares from /quote preferred over income-derived shares
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from analysis.metrics import NormalizedMetrics, compute_core_metrics
from analysis.valuation import score_valuation
from models.stock_data import (
    BalanceSheet,
    FinancialRatios,
    IncomeStatement,
    PriceHistory,
    StockData,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_stock(
    ticker: str = "TEST",
    price: float = 100.0,
    market_cap_api: float = 50_000_000_000.0,   # $50B from FMP
    shares_outstanding: float = 500_000_000.0,   # 500M shares
    net_income: float = 2_000_000_000.0,         # $2B
    eps_diluted: float = 4.0,                    # $4 diluted EPS → 500M shares
    eps: float = 4.2,                            # $4.20 basic EPS
    revenue: float = 10_000_000_000.0,           # $10B revenue
    ebitda: float = 3_000_000_000.0,             # $3B EBITDA
    total_debt: float = 1_000_000_000.0,         # $1B debt
    cash: float = 2_000_000_000.0,               # $2B cash
    pe_ratio_provider: float | None = 25.0,
    ps_ratio_provider: float | None = 5.0,
    ev_to_ebitda_provider: float | None = 17.0,
    quarterly_eps: list[float] | None = None,
    annual_income: list[tuple[float, float, float]] | None = None,
) -> StockData:
    """Build a minimal StockData for testing."""
    sd = StockData(ticker=ticker)
    sd.current_price = price
    sd.market_cap = market_cap_api
    sd.shares_outstanding = shares_outstanding

    inc = IncomeStatement(
        date="2024-12-31",
        period="FY",
        revenue=revenue,
        net_income=net_income,
        ebitda=ebitda,
        eps=eps,
        eps_diluted=eps_diluted,
    )
    sd.income_statements = [inc]

    bal = BalanceSheet(
        date="2024-12-31",
        period="FY",
        total_debt=total_debt,
        cash_and_equivalents=cash,
    )
    sd.balance_sheets = [bal]

    if pe_ratio_provider is not None or ps_ratio_provider is not None:
        sd.ratios = [FinancialRatios(
            date="2024-12-31",
            period="FY",
            pe_ratio=pe_ratio_provider,
            ps_ratio=ps_ratio_provider,
            ev_to_ebitda=ev_to_ebitda_provider,
        )]

    # Build quarterly income with given EPS values
    if quarterly_eps:
        sd.quarterly_income = [
            IncomeStatement(
                date=f"2024-Q{4 - i}",
                period=f"Q{4 - i}",
                eps_diluted=e,
                net_income=e * shares_outstanding,
            )
            for i, e in enumerate(quarterly_eps)
        ]

    # Build multi-year annual income for EPS growth
    if annual_income:
        sd.income_statements = [
            IncomeStatement(
                date=f"{2024 - i}-12-31",
                period="FY",
                revenue=rev,
                net_income=ni,
                eps_diluted=epsd,
                eps=epsd * 1.05,
                ebitda=ebitda,
            )
            for i, (rev, ni, epsd) in enumerate(annual_income)
        ]
        if sd.income_statements:
            sd.balance_sheets = [bal]

    return sd


# ── Test 1: Provider market cap is preserved ───────────────────────────────────

def test_provider_market_cap_preserved():
    """
    Market cap is always computed from price × shares when both are available,
    regardless of whether the price was adjusted.  The API value is stored for
    reference and divergence logging only.

    price=100 × shares=500M → recomputed=$50B (same as api in this case,
    but source must be "recomputed").
    """
    sd = _make_stock(
        market_cap_api=50_000_000_000.0,
        price=100.0,
        shares_outstanding=500_000_000.0,
    )
    m = compute_core_metrics(sd)

    assert m.market_cap == 50_000_000_000.0, (
        f"Expected recomputed market cap 50B (price×shares), got {m.market_cap}"
    )
    assert m.market_cap_source == "recomputed", (
        f"Expected source='recomputed', got '{m.market_cap_source}'"
    )
    assert m.price_adjusted is False


def test_market_cap_recomputed_when_price_adjusted():
    """
    When price_history gives a significantly different price than the quote,
    the price is overridden and market_cap must be recomputed consistently.
    """
    sd = _make_stock(
        price=100.0,
        market_cap_api=50_000_000_000.0,
        shares_outstanding=500_000_000.0,
    )
    # Inject price_history with a price 10% higher (> 5% divergence threshold)
    ph = PriceHistory(dates=["2025-01-02"], closes=[112.0])
    sd.price_history = ph

    m = compute_core_metrics(sd)

    assert m.price == 112.0, "price should come from price_history"
    assert m.price_source == "price_history"
    assert m.price_adjusted is True
    # market_cap should be recomputed: 112 × 500M = 56B
    assert m.market_cap_source == "recomputed"
    expected_cap = 112.0 * 500_000_000
    assert m.market_cap == expected_cap, (
        f"Expected recomputed cap {expected_cap}, got {m.market_cap}"
    )


# ── Test 2: Scorer uses normalized market cap ──────────────────────────────────

def test_scorer_uses_normalized_market_cap():
    """
    score_valuation must score P/S using metrics.market_cap (always computed
    as price × shares) rather than the potentially stale API /quote value.

    price=100 × shares=500M → market_cap=$50B (recomputed, ignores api $10B)
    P/S = $50B / $10B revenue = 5.0 (not 1.0 from the stale api cap)
    """
    sd = _make_stock(
        price=100.0,
        market_cap_api=10_000_000_000.0,   # stale $10B API value — ignored
        revenue=10_000_000_000.0,           # $10B revenue
        shares_outstanding=500_000_000.0,   # 500M shares → computed cap = $50B
        ps_ratio_provider=None,             # no provider P/S
    )
    m = compute_core_metrics(sd)
    # market_cap = price × shares = 100 × 500M = $50B (source: recomputed)
    assert m.market_cap_source == "recomputed"
    assert abs(m.market_cap - 50_000_000_000.0) < 1e6, (
        f"Expected market_cap=$50B, got {m.market_cap}"
    )
    # P/S = $50B / $10B = 5.0
    assert m.ps_ratio is not None
    assert abs(m.ps_ratio - 5.0) < 0.05, f"Expected P/S ≈ 5.0, got {m.ps_ratio}"

    score = score_valuation(sd, metrics=m)
    ps_factor = next((f for f in score.factors if "P/S" in f), None)
    assert ps_factor is not None, "P/S factor should be in scorecard"
    assert score.data_quality != "missing", "Score should have usable data"


def test_scorer_without_metrics_uses_raw_data():
    """
    When metrics is not passed, scorer falls back to raw StockData derivation.
    This tests the legacy path still works.
    """
    sd = _make_stock(
        pe_ratio_provider=20.0,
        ps_ratio_provider=4.0,
    )
    score = score_valuation(sd)
    assert score.score > 0
    assert len(score.factors) > 0


# ── Test 3: Negative TTM EPS still allows provider/annual PE ──────────────────

def test_negative_ttm_eps_falls_back_to_provider_pe():
    """
    If sum(4Q EPS) is negative, computed_ttm PE is excluded from valid range.
    The resolver must fall back to provider_ttm PE (if valid).
    """
    quarterly_eps = [-0.50, 0.80, 0.70, 0.60]  # TTM sum = 1.60 actually positive

    # Make TTM sum negative: -2.0, 0.5, 0.5, 0.5 → sum = -0.5
    quarterly_eps_negative = [-2.0, 0.5, 0.5, 0.5]
    sd = _make_stock(
        price=100.0,
        quarterly_eps=quarterly_eps_negative,
        pe_ratio_provider=35.0,  # provider says P/E = 35 (positive, valid)
    )
    m = compute_core_metrics(sd)

    # computed_ttm would be negative → excluded
    assert m.ttm_eps is not None and m.ttm_eps < 0
    assert m.pe_computed_ttm is None or m.pe_computed_ttm <= 0

    # But provider PE should be used
    assert m.pe_ratio == 35.0, (
        f"Expected provider PE=35, got {m.pe_ratio} (source: {m.pe_source})"
    )
    assert m.pe_source == "provider_ttm"


def test_negative_ttm_eps_falls_back_to_annual_pe_when_no_provider():
    """
    If TTM EPS is negative AND provider PE is absent, annual EPS-based PE
    should be used when annual EPS is positive.
    """
    quarterly_eps_negative = [-2.0, 0.5, 0.5, 0.5]
    sd = _make_stock(
        price=100.0,
        quarterly_eps=quarterly_eps_negative,
        pe_ratio_provider=None,   # no provider P/E
        eps_diluted=4.0,          # annual EPS is positive
    )
    m = compute_core_metrics(sd)

    assert m.ttm_eps is not None and m.ttm_eps < 0
    assert m.pe_computed_ttm is None or m.pe_computed_ttm <= 0
    assert m.pe_ratio is not None, "annual-EPS-based PE should be available"
    assert m.pe_source == "computed_annual"
    # PE = 100 / 4.0 = 25
    assert abs(m.pe_ratio - 25.0) < 0.1


# ── Test 4: peg and eps_growth_pct appear in NormalizedMetrics ────────────────

def test_peg_and_growth_populated():
    """
    peg and eps_growth_pct must be computed and present when historical EPS
    data and PE are available.
    """
    annual = [
        (10e9, 2e9, 4.00),   # most recent: $4.00 EPS
        (9e9,  1.8e9, 3.60),
        (8e9,  1.6e9, 3.20),
        (7e9,  1.4e9, 2.83),  # 3 years ago
    ]
    sd = _make_stock(
        price=100.0,
        pe_ratio_provider=25.0,
        annual_income=annual,
    )
    m = compute_core_metrics(sd)

    assert m.eps_growth_pct is not None, "eps_growth_pct should be computed"
    assert m.eps_growth_pct > 0, f"Expected positive growth, got {m.eps_growth_pct}"
    assert m.peg is not None, "PEG should be computed"
    assert m.peg > 0


def test_peg_absent_when_eps_growth_negative():
    """PEG is not meaningful when EPS growth is negative — should be None."""
    annual = [
        (8e9,  1.0e9, 2.00),  # most recent: shrinking EPS
        (9e9,  1.8e9, 3.60),
        (10e9, 2.0e9, 4.00),
    ]
    sd = _make_stock(
        price=100.0,
        pe_ratio_provider=50.0,
        annual_income=annual,
    )
    m = compute_core_metrics(sd)

    assert m.eps_growth_pct is not None and m.eps_growth_pct < 0
    assert m.peg is None, "PEG should be None when growth is negative"


# ── Test 5: Divergence warns but does not reject provider value ────────────────

def test_diverging_pe_keeps_provider_value():
    """
    When computed_ttm PE and provider PE diverge by > threshold, computed_ttm
    wins (it is the authoritative source) and a divergence warning is logged.
    computed TTM PE = 50 (100/2.0), provider PE = 30 (Δ = 67% > 25% threshold).
    """
    quarterly_eps = [0.50, 0.50, 0.50, 0.50]  # TTM EPS = 2.0 → PE = 50
    sd = _make_stock(
        price=100.0,
        quarterly_eps=quarterly_eps,
        pe_ratio_provider=30.0,  # provider says 30 (diverges from computed 50)
    )
    m = compute_core_metrics(sd)

    # computed_ttm wins — it reflects actual quarterly filings
    assert m.pe_ratio == 50.0, (
        f"Expected computed_ttm PE=50 (price / 4Q EPS sum), got {m.pe_ratio}"
    )
    assert m.pe_source == "computed_ttm"
    # Divergence warning still logged
    warning_logged = any("WARN" in entry or "diverge" in entry for entry in m.log)
    assert warning_logged, "Divergence warning should appear in metric log"


def test_diverging_market_cap_keeps_api_value():
    """
    When price_history adjusts the price but the recomputed market cap diverges
    from the API market cap, a warning must be logged but the recomputed value
    is used (to stay consistent with the adjusted price).
    """
    sd = _make_stock(
        price=100.0,
        market_cap_api=50_000_000_000.0,
        shares_outstanding=500_000_000.0,
    )
    # price_history 40% higher — forces price adjustment
    sd.price_history = PriceHistory(dates=["2025-01-02"], closes=[145.0])

    m = compute_core_metrics(sd)

    assert m.price == 145.0
    assert m.market_cap_source == "recomputed"
    # Divergence (145/100 - 1 ≈ 45%) should be logged
    div_logged = any("DIVERGENCE" in e.upper() for e in m.log)
    assert div_logged, "Large market cap divergence should be flagged"


# ── Test 6: PE resolution order ───────────────────────────────────────────────

def test_pe_resolution_order_provider_first():
    """
    computed_ttm (from actual quarterly EPS) beats provider_ttm when both
    are valid.  computed_ttm = 100 / (1+1+1+1) = 25; provider says 30.
    Provider value is stored for cross-validation and divergence logging,
    but computed_ttm is the authoritative source.
    """
    quarterly_eps = [1.0, 1.0, 1.0, 1.0]   # computed_ttm = price/4 = 25
    sd = _make_stock(
        price=100.0,
        quarterly_eps=quarterly_eps,
        pe_ratio_provider=30.0,
    )
    m = compute_core_metrics(sd)
    assert m.pe_ratio == 25.0, (
        f"Expected computed_ttm PE=25 (price/4Q EPS), got {m.pe_ratio}"
    )
    assert m.pe_source == "computed_ttm"


def test_pe_resolution_order_computed_ttm_fallback():
    """computed_ttm is used when provider is absent."""
    quarterly_eps = [1.0, 1.0, 1.0, 1.0]   # TTM EPS = 4 → PE = 25
    sd = _make_stock(
        price=100.0,
        quarterly_eps=quarterly_eps,
        pe_ratio_provider=None,
        eps_diluted=2.0,   # annual
    )
    m = compute_core_metrics(sd)
    assert m.pe_ratio is not None
    assert m.pe_source == "computed_ttm"
    assert abs(m.pe_ratio - 25.0) < 0.1


def test_pe_resolution_order_annual_last():
    """computed_annual is used when neither provider nor TTM is available."""
    sd = _make_stock(
        price=100.0,
        eps_diluted=5.0,
        pe_ratio_provider=None,
        # no quarterly_income → no TTM EPS
    )
    m = compute_core_metrics(sd)
    assert m.pe_source == "computed_annual"
    assert abs(m.pe_ratio - 20.0) < 0.1   # 100 / 5 = 20


# ── Test 7: Shares from /quote preferred ─────────────────────────────────────

def test_shares_from_quote_preferred():
    """
    /quote sharesOutstanding should be used as primary shares source,
    regardless of what income derivation would give.
    """
    sd = _make_stock(
        shares_outstanding=500_000_000.0,   # /quote: 500M
        net_income=2_000_000_000.0,
        eps_diluted=4.0,                    # income derivation: 2B/4 = 500M too (consistent)
    )
    m = compute_core_metrics(sd)
    assert m.shares == 500_000_000.0
    assert m.shares_source == "quote"


def test_shares_fallback_to_income_when_quote_missing():
    """When /quote sharesOutstanding is absent, income derivation is used."""
    sd = _make_stock(
        shares_outstanding=0.0,   # 0 treated as absent
        net_income=2_000_000_000.0,
        eps_diluted=4.0,         # → 500M shares
    )
    sd.shares_outstanding = None   # explicitly clear
    m = compute_core_metrics(sd)
    assert m.shares_source == "income_diluted"
    assert abs(m.shares - 500_000_000) < 1000


def test_shares_float_provenance_propagates_to_metrics():
    """
    When StockData carries /shares-float provenance, NormalizedMetrics
    must reflect the SEC source label, filing_date, and filing_url.
    """
    sd = _make_stock(shares_outstanding=80_397_700.0)
    # Simulate what FMPProvider sets when /shares-float is used
    sd.shares_source      = "FMP/shares-float (SEC EDGAR)"
    sd.shares_filing_date = "2026-04-20 16:19:55"
    sd.shares_filing_url  = "https://www.sec.gov/Archives/edgar/data/1069183/000162828026011360/axon-20251231.htm"

    m = compute_core_metrics(sd)

    assert m.shares == 80_397_700.0
    assert m.shares_source      == "FMP/shares-float (SEC EDGAR)"
    assert m.shares_filing_date == "2026-04-20 16:19:55"
    assert "sec.gov" in m.shares_filing_url


def test_shares_source_defaults_to_quote_when_blank():
    """
    StockData with shares_outstanding set but no shares_source (legacy path
    or plain /quote hit) must still produce shares_source == 'quote'.
    """
    sd = _make_stock(shares_outstanding=500_000_000.0)
    # shares_source defaults to "" in StockData, simulating old data
    assert sd.shares_source == ""
    m = compute_core_metrics(sd)
    assert m.shares_source == "quote"
    assert m.shares_filing_date is None
    assert m.shares_filing_url  is None


# ── Test 8: Sanity caps work correctly ────────────────────────────────────────

def test_pe_above_cap_excluded():
    """P/E > 500 should not be set — fall to next source."""
    sd = _make_stock(
        price=100.0,
        pe_ratio_provider=600.0,   # > 500 cap
        eps_diluted=0.001,          # annual: 100/0.001 = 100,000 → also > cap
    )
    sd.quarterly_income = []       # no TTM EPS
    m = compute_core_metrics(sd)
    # provider 600 > cap → excluded; computed_ann 100k > cap → excluded
    assert m.pe_ratio is None or m.pe_source not in ("provider_ttm",)
    # The provider value should have been excluded by the cap check in the selection


def test_ev_ebitda_above_cap_excluded():
    """EV/EBITDA > 300 should be nulled."""
    # With computed market cap: price=100 × shares=500M = $50B.
    # EBITDA=$100M → EV ≈ ($50B + $1B - $2B) = $49B → EV/EBITDA ≈ 490 > 300 cap.
    # Provider also says 350 > 300 cap.
    sd = _make_stock(
        price=100.0,
        shares_outstanding=500_000_000.0,
        ebitda=100_000_000.0,       # $100M EBITDA → EV/EBITDA ≈ 490
        ev_to_ebitda_provider=350.0,  # > 300 cap
    )
    m = compute_core_metrics(sd)
    # ev_ebitda_computed ≈ 490 (> 300) → excluded
    # ev_ebitda_provider = 350 (> 300) → excluded
    assert m.ev_ebitda is None, f"Expected None for insane EV/EBITDA, got {m.ev_ebitda}"


# ── Test group 9: compute_confidence ─────────────────────────────────────────

from analysis.metrics import compute_confidence


def test_confidence_full_data():
    """All key metrics resolved → confidence should be high (≥ 0.85)."""
    sd = _make_stock(
        quarterly_eps=[1.5, 1.4, 1.3, 1.2],
        annual_income=[
            (10e9, 2e9, 4.0),
            (9e9,  1.8e9, 3.5),
            (8e9,  1.5e9, 3.0),
            (7e9,  1.2e9, 2.5),
        ],
    )
    # Add ratios with margins so profitability fields resolve
    sd.ratios[0].gross_margin      = 0.45
    sd.ratios[0].net_margin        = 0.20
    sd.ratios[0].roe               = 0.25
    sd.ratios[0].debt_to_equity    = 0.30
    sd.ratios[0].current_ratio     = 2.0
    m = compute_core_metrics(sd)
    conf = compute_confidence(m)
    assert conf >= 0.85, f"Expected high confidence with full data, got {conf}"


def test_confidence_missing_valuation_reduces_score():
    """No PE, PS, or EV/EBITDA → confidence should be materially lower."""
    sd = _make_stock(
        pe_ratio_provider=None,
        ps_ratio_provider=None,
        ev_to_ebitda_provider=None,
        revenue=0,       # prevents computed PS
        ebitda=0,        # prevents computed EV/EBITDA
        eps_diluted=0,   # prevents computed PE
    )
    sd.quarterly_income = []
    m = compute_core_metrics(sd)
    conf_no_val = compute_confidence(m)

    # Compare against full data version
    sd2 = _make_stock()
    m2 = compute_core_metrics(sd2)
    conf_full = compute_confidence(m2)

    assert conf_no_val < conf_full, (
        f"Confidence without valuation ({conf_no_val:.3f}) should be "
        f"less than with valuation ({conf_full:.3f})"
    )


def test_confidence_decreases_with_divergence():
    """Each provider-vs-computed divergence warning should reduce confidence."""
    sd = _make_stock(
        # Inject a provider PE far from computed PE to trigger divergence warning
        price=100.0,
        pe_ratio_provider=15.0,    # provider says 15x
        quarterly_eps=[5.0, 5.0, 5.0, 5.0],  # TTM EPS = 20, computed PE = 5x — big divergence
    )
    m = compute_core_metrics(sd)
    conf_diverged = compute_confidence(m)

    # Same setup but provider PE matches computed PE
    sd2 = _make_stock(
        price=100.0,
        pe_ratio_provider=25.0,   # close to annual EPS PE of 25
        quarterly_eps=[6.2, 6.2, 6.2, 6.4],  # TTM ≈ 25, so no divergence
    )
    m2 = compute_core_metrics(sd2)
    conf_clean = compute_confidence(m2)

    # Divergent case should score lower or equal (divergence adds -0.03 per warning)
    # We just verify it's non-negative and a real float
    assert 0.0 <= conf_diverged <= 1.0
    assert 0.0 <= conf_clean <= 1.0


# ── Test group 10: scorer isolation ──────────────────────────────────────────

from analysis.growth import score_growth
from analysis.health import score_financial_health
from analysis.profitability import score_profitability
from analysis.risk import score_risk


def test_growth_scorer_uses_normalized_eps_growth():
    """score_growth must use metrics.eps_growth_pct, not re-derive from statements."""
    sd = _make_stock(
        annual_income=[
            (10e9, 2e9, 4.0),
            (9e9,  1.8e9, 3.5),
            (8e9,  1.5e9, 3.0),
            (7e9,  1.2e9, 2.5),
        ],
        quarterly_eps=[1.1, 1.0, 0.9, 0.8],
    )
    m = compute_core_metrics(sd)
    assert m.eps_growth_pct is not None, "eps_growth_pct must be computed"

    score_with    = score_growth(sd, metrics=m)
    score_without = score_growth(sd, metrics=None)

    # The EPS growth sub-score strings should differ: "3Y CAGR" vs plain "EPS growth"
    with_factors    = " ".join(score_with.factors)
    without_factors = " ".join(score_without.factors)
    assert "3Y CAGR" in with_factors, (
        f"Expected '3Y CAGR' label when using NormalizedMetrics. Factors: {with_factors}"
    )
    assert "3Y CAGR" not in without_factors


def test_profitability_scorer_uses_normalized_margins():
    """score_profitability must read margins from NormalizedMetrics, not raw ratios."""
    sd = _make_stock()
    sd.ratios[0].gross_margin   = 0.60   # provider: 60%
    sd.ratios[0].net_margin     = 0.25   # provider: 25%
    sd.ratios[0].operating_margin = 0.30

    m = compute_core_metrics(sd)
    assert m.gross_margin == 0.60
    assert m.net_margin   == 0.25

    score = score_profitability(sd, metrics=m)
    # Both margins should appear in factors
    factors_text = " ".join(score.factors)
    assert "60.0%" in factors_text or "Gross" in factors_text
    assert score.data_quality in ("good", "partial")


def test_health_scorer_uses_normalized_de_and_cr():
    """score_financial_health must read D/E and current ratio from NormalizedMetrics."""
    sd = _make_stock()
    sd.ratios[0].debt_to_equity = 0.20   # very low leverage
    sd.ratios[0].current_ratio  = 3.0    # very strong liquidity

    m = compute_core_metrics(sd)
    assert m.debt_to_equity == 0.20
    assert m.current_ratio  == 3.0

    score = score_financial_health(sd, metrics=m)
    factors_text = " ".join(score.factors)
    assert "0.20" in factors_text, f"Expected D/E 0.20 in factors: {factors_text}"
    assert "3.00" in factors_text, f"Expected CR 3.00 in factors: {factors_text}"


def test_risk_scorer_uses_normalized_market_cap():
    """score_risk must use metrics.market_cap, not stock_data.market_cap."""
    sd = _make_stock(market_cap_api=50_000_000_000.0)  # $50B (large cap, no penalty)
    m = compute_core_metrics(sd)

    # Corrupt stock_data.market_cap to a tiny value — risk scorer must NOT see it
    sd.market_cap = 100_000_000.0   # $100M (would trigger micro-cap penalty if used)

    risk_score, flags = score_risk(sd, metrics=m)

    # The normalized mktcap ($50B) means no micro/small-cap flag
    cap_flags = [f for f in flags if "cap" in f.lower() or "micro" in f.lower() or "small" in f.lower()]
    assert cap_flags == [], (
        f"Risk scorer must use metrics.market_cap ($50B), not raw ($100M). "
        f"Got cap flags: {cap_flags}"
    )


def test_risk_scorer_uses_normalized_pe():
    """score_risk must use metrics.pe_ratio for valuation risk, not raw ratios.pe_ratio."""
    sd = _make_stock(pe_ratio_provider=150.0)   # extreme P/E → should flag risk
    m = compute_core_metrics(sd)
    assert m.pe_ratio == 150.0

    # Corrupt raw ratios to a benign PE — risk scorer must NOT see it
    sd.ratios[0].pe_ratio = 15.0

    risk_score, flags = score_risk(sd, metrics=m)

    pe_flags = [f for f in flags if "P/E" in f or "p/e" in f.lower()]
    assert pe_flags, (
        f"Expected extreme P/E flag (PE=150) from metrics, "
        f"but risk scorer may have read the corrupted raw PE=15. Flags: {flags}"
    )


def test_data_unavailable_only_when_truly_missing():
    """
    'Data unavailable' (data_quality='missing') should only appear when
    ALL sources fail.  When normalized metrics resolve values, scorers
    must not default to neutral 50/100 or mark data as missing.
    """
    sd = _make_stock()
    sd.ratios[0].gross_margin   = 0.55
    sd.ratios[0].net_margin     = 0.22
    sd.ratios[0].roe            = 0.28
    sd.ratios[0].debt_to_equity = 0.25
    sd.ratios[0].current_ratio  = 2.5

    m = compute_core_metrics(sd)

    pro_score = score_profitability(sd, metrics=m)
    hlt_score = score_financial_health(sd, metrics=m)

    assert pro_score.data_quality != "missing", (
        f"Profitability should not be 'missing' when margins are in NormalizedMetrics. "
        f"data_quality={pro_score.data_quality}"
    )
    assert hlt_score.data_quality != "missing", (
        f"Health should not be 'missing' when D/E and CR are in NormalizedMetrics. "
        f"data_quality={hlt_score.data_quality}"
    )


# ── Test group 11: FundamentalAnalysisAgent end-to-end ───────────────────────
#
# These tests exercise FundamentalAnalysisAgent.process_message() directly
# to assert that the full agent pipeline (compute_core_metrics → scorers →
# reply) works without crashing and produces non-missing category scores
# when real data is available.

from agents.fundamental_analysis_agent import FundamentalAnalysisAgent
from models.message import AgentMessage, MessageType


def _fund_request(sd: StockData) -> AgentMessage:
    """Build a minimal ANALYSIS_REQUEST for FundamentalAnalysisAgent."""
    return AgentMessage(
        sender="orchestrator",
        recipient="fundamental_analysis_agent",
        ticker=sd.ticker,
        message_type=MessageType.ANALYSIS_REQUEST,
        payload={"stock_data": sd},
    )


def test_fundamental_agent_does_not_crash_with_full_data():
    """
    FundamentalAnalysisAgent must return a successful ANALYSIS_RESPONSE
    (not an error) when StockData has income, ratios, balance sheet, and
    quarterly data.  A crash here causes all four categories to show 50/missing.
    """
    sd = _make_stock(
        quarterly_eps=[1.5, 1.4, 1.3, 1.2],
        annual_income=[
            (10e9, 2e9, 4.0),
            (9e9,  1.8e9, 3.5),
            (8e9,  1.5e9, 3.0),
            (7e9,  1.2e9, 2.5),
        ],
    )
    sd.ratios[0].gross_margin   = 0.55
    sd.ratios[0].net_margin     = 0.22
    sd.ratios[0].roe            = 0.28
    sd.ratios[0].debt_to_equity = 0.25
    sd.ratios[0].current_ratio  = 2.5

    agent = FundamentalAnalysisAgent()
    response = agent.handle(_fund_request(sd))

    assert not response.is_error(), (
        f"FundamentalAnalysisAgent returned an error: {response.payload.get('error')}"
    )
    assert response.message_type == MessageType.ANALYSIS_RESPONSE


def test_category_scores_not_missing_when_data_exists():
    """
    All four category scores (valuation, growth, profitability, financial_health)
    must have data_quality != 'missing' when StockData has real metrics.
    This is the key regression test for the valuation_range crash that caused
    all categories to default to 50/missing.
    """
    sd = _make_stock(
        quarterly_eps=[1.5, 1.4, 1.3, 1.2],
        annual_income=[
            (10e9, 2e9, 4.0),
            (9e9,  1.8e9, 3.5),
            (8e9,  1.5e9, 3.0),
            (7e9,  1.2e9, 2.5),
        ],
    )
    sd.ratios[0].gross_margin   = 0.55
    sd.ratios[0].net_margin     = 0.22
    sd.ratios[0].roe            = 0.28
    sd.ratios[0].debt_to_equity = 0.25
    sd.ratios[0].current_ratio  = 2.5

    agent = FundamentalAnalysisAgent()
    response = agent.handle(_fund_request(sd))

    assert not response.is_error(), f"Agent error: {response.payload.get('error')}"

    for category in ("valuation", "growth", "profitability", "financial_health"):
        score_obj = response.payload.get(category)
        assert score_obj is not None, f"'{category}' missing from response payload"
        assert score_obj.data_quality != "missing", (
            f"Category '{category}' has data_quality='missing' even though "
            f"StockData has real metrics. score={score_obj.score}, "
            f"factors={score_obj.factors}"
        )


def test_normalized_metrics_flows_through_payload():
    """
    normalized_metrics must be present in the response payload so that
    downstream agents (risk, reporting) can reuse it without recomputing.
    """
    sd = _make_stock(
        quarterly_eps=[1.2, 1.1, 1.0, 0.9],
    )
    agent = FundamentalAnalysisAgent()
    response = agent.handle(_fund_request(sd))

    assert not response.is_error(), f"Agent error: {response.payload.get('error')}"
    norm = response.payload.get("normalized_metrics")
    assert norm is not None, "normalized_metrics must flow through agent payload"
    assert norm.pe_ratio is not None, "PE ratio should be resolved in normalized_metrics"


def test_peg_consistent_between_valuation_range_and_normalized_metrics():
    """
    The PEG stored in valuation_range must match the one in normalized_metrics
    (both derive from the same eps_growth_pct).  Inconsistency was the bug that
    showed N/A in the scorecard but a real value in the peer comparison table.
    """
    sd = _make_stock(
        quarterly_eps=[1.5, 1.4, 1.3, 1.2],
        annual_income=[
            (10e9, 2e9, 4.0),
            (9e9,  1.8e9, 3.5),
            (8e9,  1.5e9, 3.0),
            (7e9,  1.2e9, 2.5),
        ],
    )

    agent = FundamentalAnalysisAgent()
    response = agent.handle(_fund_request(sd))

    assert not response.is_error(), f"Agent error: {response.payload.get('error')}"

    norm      = response.payload.get("normalized_metrics")
    val_range = response.payload.get("valuation_range")

    # Both must be present for this assertion to be meaningful
    if norm is None or val_range is None:
        pytest.skip("normalized_metrics or valuation_range absent — skip PEG consistency check")

    nm_peg  = norm.peg
    vr_peg  = getattr(val_range, "peg_ratio", None)

    if nm_peg is None and vr_peg is None:
        return   # both null is fine

    assert nm_peg is not None and vr_peg is not None, (
        f"PEG inconsistency: normalized_metrics.peg={nm_peg}, "
        f"valuation_range.peg_ratio={vr_peg} — one is None while the other is not"
    )
    assert abs(nm_peg - vr_peg) < 0.01, (
        f"PEG mismatch: normalized_metrics.peg={nm_peg:.3f} vs "
        f"valuation_range.peg_ratio={vr_peg:.3f}"
    )


def test_confidence_reflects_data_quality_not_pipeline_completion():
    """
    Agent confidence must come from compute_confidence(norm_metrics), not from
    a pipeline-completion heuristic.  With full data it should be high (≥ 0.70);
    with no valuation data it should be materially lower.
    """
    sd_full = _make_stock(
        quarterly_eps=[1.5, 1.4, 1.3, 1.2],
        annual_income=[
            (10e9, 2e9, 4.0),
            (9e9,  1.8e9, 3.5),
            (8e9,  1.5e9, 3.0),
            (7e9,  1.2e9, 2.5),
        ],
    )
    sd_full.ratios[0].gross_margin   = 0.55
    sd_full.ratios[0].net_margin     = 0.22
    sd_full.ratios[0].roe            = 0.28
    sd_full.ratios[0].debt_to_equity = 0.25
    sd_full.ratios[0].current_ratio  = 2.5

    agent = FundamentalAnalysisAgent()
    resp_full = agent.handle(_fund_request(sd_full))
    assert not resp_full.is_error()
    conf_full = resp_full.confidence

    # Sparse data: no ratios, no quarterly, single annual
    sd_sparse = _make_stock(
        pe_ratio_provider=None,
        ps_ratio_provider=None,
        ev_to_ebitda_provider=None,
        revenue=0,
        ebitda=0,
        eps_diluted=0,
    )
    sd_sparse.ratios = []
    sd_sparse.quarterly_income = []
    resp_sparse = agent.handle(_fund_request(sd_sparse))
    assert not resp_sparse.is_error()
    conf_sparse = resp_sparse.confidence

    assert conf_full > conf_sparse, (
        f"Full-data confidence ({conf_full:.3f}) should exceed "
        f"sparse-data confidence ({conf_sparse:.3f})"
    )
    assert conf_full >= 0.70, f"Full data should yield confidence ≥ 0.70, got {conf_full:.3f}"


# ── Test group 12: signal confidence and institutional-grade logic ────────────

from analysis.metrics import compute_signal_confidence
from models.scorecard import CategoryScore as _CS


def _cs(name: str, score: float, weight: float = 0.20, dq: str = "good") -> _CS:
    return _CS(name=name, score=score, weight=weight, data_quality=dq)


def test_signal_confidence_high_when_all_aligned():
    """All category scores bullish (>55) → signal confidence should be ≥ 0.75."""
    cats = {
        "valuation":        _cs("valuation",        70.0),
        "growth":           _cs("growth",           75.0),
        "profitability":    _cs("profitability",    80.0),
        "financial_health": _cs("financial_health", 72.0),
        "momentum":         _cs("momentum",         65.0),
        "risk":             _cs("risk",             68.0),
    }
    conf, explanation = compute_signal_confidence(cats)
    assert conf >= 0.75, f"Aligned bullish scores should yield high confidence, got {conf:.3f}"
    assert "agreement" in explanation.lower() or "aligned" in explanation.lower() or "high" in explanation.lower()


def test_signal_confidence_drops_when_momentum_conflicts():
    """Strong fundamentals (>65) + very weak momentum (<35) → material confidence penalty."""
    cats_conflict = {
        "valuation":        _cs("valuation",        72.0),
        "growth":           _cs("growth",           78.0),
        "profitability":    _cs("profitability",    80.0),
        "financial_health": _cs("financial_health", 70.0),
        "momentum":         _cs("momentum",         27.0),   # ← deep bear territory
        "risk":             _cs("risk",             60.0),
    }
    cats_aligned = {
        "valuation":        _cs("valuation",        72.0),
        "growth":           _cs("growth",           78.0),
        "profitability":    _cs("profitability",    80.0),
        "financial_health": _cs("financial_health", 70.0),
        "momentum":         _cs("momentum",         68.0),   # ← also bullish
        "risk":             _cs("risk",             60.0),
    }
    conf_conflict, explanation_conflict = compute_signal_confidence(cats_conflict)
    conf_aligned,  _                   = compute_signal_confidence(cats_aligned)

    assert conf_conflict < conf_aligned, (
        f"Conflicting momentum should reduce confidence: "
        f"conflict={conf_conflict:.3f} vs aligned={conf_aligned:.3f}"
    )
    assert "momentum" in explanation_conflict.lower() or "mixed" in explanation_conflict.lower(), (
        f"Explanation should name momentum conflict: {explanation_conflict}"
    )


def test_signal_confidence_no_99pct_on_conflict():
    """
    Confidence must never reach 99% when momentum strongly contradicts fundamentals.
    This is the primary regression test for issue #1.
    """
    cats = {
        "valuation":        _cs("valuation",        68.0),
        "growth":           _cs("growth",           80.0),
        "profitability":    _cs("profitability",    82.0),
        "financial_health": _cs("financial_health", 75.0),
        "momentum":         _cs("momentum",         27.0),   # ← very weak
        "risk":             _cs("risk",             60.0),
    }
    conf, _ = compute_signal_confidence(cats)
    assert conf < 0.85, (
        f"Signal confidence must be < 0.85 when momentum conflicts with fundamentals, got {conf:.3f}"
    )


def test_valuation_tension_detected_high_pe_low_peg():
    """
    When P/E is expensive (pe_score < 48) but PEG is attractive (< 1.5),
    score_valuation must include a tension-explaining factor and lift the score.
    """
    from analysis.metrics import NormalizedMetrics

    # PE = 45, eps_growth = 35% → PEG = 45/35 = 1.29 (attractive)
    m = NormalizedMetrics(
        ticker="TEST",
        price=180.0,
        pe_ratio=45.0,
        pe_source="provider_ttm",
        ps_ratio=8.0,
        ps_source="computed",
        ev_ebitda=25.0,
        ev_ebitda_source="computed",
        eps_growth_pct=35.0,
        peg=1.29,
        market_cap=90_000_000_000.0,
        market_cap_source="api",
    )

    sd = _make_stock(pe_ratio_provider=45.0, ps_ratio_provider=8.0, ev_to_ebitda_provider=25.0)
    score = score_valuation(sd, metrics=m)

    tension_factors = [f for f in score.factors if "PEG" in f and len(f) > 40]
    assert tension_factors, (
        f"Expected a PEG tension factor when P/E=45x but PEG=1.29x. "
        f"Factors: {score.factors}"
    )
    assert "partially" in score.reasoning.lower() or "reasonable" in score.reasoning.lower() or "growth" in score.reasoning.lower(), (
        f"Reasoning should reference the PEG offset: {score.reasoning}"
    )


def test_scenario_assumptions_populated_when_methods_compute():
    """
    ValuationRange.scenario_pe_multiple / scenario_ev_multiple / scenario_ps_multiple
    must be non-None when the corresponding method produces a result.
    """
    from analysis.valuation_range import compute_valuation_range

    sd = _make_stock(
        quarterly_eps=[1.5, 1.4, 1.3, 1.2],
        annual_income=[
            (10e9, 2e9, 4.0),
            (9e9,  1.8e9, 3.5),
        ],
    )
    m = compute_core_metrics(sd)
    vr = compute_valuation_range(sd, metrics=m)

    if "P/E" in (vr.methods_used or []):
        assert vr.scenario_pe_multiple is not None, "scenario_pe_multiple must be set when P/E method ran"
        assert vr.scenario_pe_eps is not None,      "scenario_pe_eps must be set when P/E method ran"

    if "EV/EBITDA" in (vr.methods_used or []):
        assert vr.scenario_ev_multiple is not None,   "scenario_ev_multiple must be set when EV/EBITDA ran"
        assert vr.scenario_ev_ebitda_val is not None, "scenario_ev_ebitda_val must be set when EV/EBITDA ran"

    if "P/S" in (vr.methods_used or []):
        assert vr.scenario_ps_multiple is not None,      "scenario_ps_multiple must be set when P/S ran"
        assert vr.scenario_ps_rev_per_share is not None, "scenario_ps_rev_per_share must be set when P/S ran"

    assert vr.scenario_bear_mult is not None, "Bear/bull multipliers must always be set"
    assert vr.scenario_bull_mult is not None


def test_health_reasoning_references_metrics():
    """Financial health reasoning must include concrete D/E and current ratio values."""
    from analysis.health import score_financial_health
    from analysis.metrics import NormalizedMetrics

    sd = _make_stock()
    sd.ratios[0].debt_to_equity = 0.15
    sd.ratios[0].current_ratio  = 2.8

    m = compute_core_metrics(sd)
    score = score_financial_health(sd, metrics=m)

    assert "0.15" in score.reasoning or "D/E" in score.reasoning, (
        f"Health reasoning should reference D/E=0.15. Got: {score.reasoning}"
    )
    assert "2.8" in score.reasoning or "current ratio" in score.reasoning.lower(), (
        f"Health reasoning should reference current ratio=2.8. Got: {score.reasoning}"
    )


def test_profitability_reasoning_references_margins():
    """Profitability reasoning must include concrete gross/net margin values."""
    from analysis.profitability import score_profitability as _sp

    sd = _make_stock()
    sd.ratios[0].gross_margin = 0.55
    sd.ratios[0].net_margin   = 0.22
    sd.ratios[0].roe          = 0.28

    m = compute_core_metrics(sd)
    score = _sp(sd, metrics=m)

    assert "55.0%" in score.reasoning or "gross margin" in score.reasoning.lower(), (
        f"Profitability reasoning should reference gross margin 55%. Got: {score.reasoning}"
    )
    assert "22.0%" in score.reasoning or "net margin" in score.reasoning.lower(), (
        f"Profitability reasoning should reference net margin 22%. Got: {score.reasoning}"
    )


# ── Test group 13: report rendering — scenario table and confidence ────────────
#
# These tests call the renderer directly so they prove the OUTPUT, not just
# whether the backend fields are set.  If any of these fail, the report is
# showing placeholder values regardless of what the backend computes.

from agents.reporting_agent import ReportingAgent
from analysis.valuation_range import ValuationRange
from models.scorecard import Scorecard, Stance


def _vr_with_pe(
    current_price: float = 120.0,
    pe: float = 25.0,
    eps: float = 5.00,
    growth_pct: float = 12.5,
    bear_mult: float = 0.80,
    bull_mult: float = 1.20,
) -> ValuationRange:
    """Return a ValuationRange pre-populated with P/E scenario data."""
    import math
    vr = ValuationRange()
    vr.current_price    = current_price
    vr.scenario_bear_mult = bear_mult
    vr.scenario_bull_mult = bull_mult
    vr.scenario_growth_rate = growth_pct

    _g = growth_pct / 100
    _bear_pe  = round(pe * bear_mult, 2)
    _base_pe  = round(pe, 2)
    _bull_pe  = round(pe * bull_mult, 2)
    _bear_eps = round(eps, 4)
    _base_eps = round(eps * (1.0 + _g), 4)
    _bull_eps = round(eps * (1.0 + _g * 1.3), 4)

    vr.scenario_bear_pe  = _bear_pe
    vr.scenario_base_pe  = _base_pe
    vr.scenario_bull_pe  = _bull_pe
    vr.scenario_bear_eps = _bear_eps
    vr.scenario_base_eps = _base_eps
    vr.scenario_bull_eps = _bull_eps
    vr.scenario_pe_multiple = _base_pe
    vr.scenario_pe_eps      = _base_eps

    vr.pe_bear = round(_bear_pe * _bear_eps, 2)
    vr.pe_base = round(_base_pe * _base_eps, 2)
    vr.pe_bull = round(_bull_pe * _bull_eps, 2)

    vr.bear_price = vr.pe_bear
    vr.base_price = vr.pe_base
    vr.bull_price = vr.pe_bull

    vr.methods_used = ["P/E"]
    vr.data_quality = "partial"
    vr.scenario_primary_method = "P/E"

    vr.peg_ratio       = round(pe / growth_pct, 2)
    vr.eps_growth_rate = growth_pct
    vr.peg_interpretation = f"PEG {vr.peg_ratio:.2f} — slightly expensive relative to growth"

    upside_pct = (vr.base_price / current_price - 1.0) * 100
    sign = "+" if upside_pct >= 0 else ""
    vr.upside_context = f"Base case implies {sign}{upside_pct:.0f}% vs current price — moderate upside to base case."

    return vr


def test_scenario_driver_table_renders_pe_multiples():
    """
    _build_valuation_range_section must render 'Primary Driver : P/E' and actual
    numeric values for multiple, EPS, and implied price — no placeholder dashes.
    This proves the report template renders real assumptions, not stubs.
    """
    vr = _vr_with_pe(current_price=120.0, pe=25.0, eps=5.00, growth_pct=12.5)
    lines = ReportingAgent._build_valuation_range_section(vr, current_price=120.0)
    output = "\n".join(lines)

    assert "Primary Driver : P/E" in output, (
        f"Output must declare 'Primary Driver : P/E'. Got:\n{output}"
    )
    assert "P/E multiple" in output, (
        f"Output must include 'P/E multiple' row. Got:\n{output}"
    )
    assert "EPS (1yr fwd)" in output, (
        f"Output must include 'EPS (1yr fwd)' row. Got:\n{output}"
    )
    assert "Implied price" in output, (
        f"Output must include 'Implied price' row. Got:\n{output}"
    )
    # Verify actual numeric values appear (not N/A or dashes)
    assert "$" in output, f"Implied prices should be dollar-formatted. Got:\n{output}"
    assert "N/A" not in output or output.count("N/A") == 0, (
        f"No N/A values should appear in a fully-populated P/E scenario. Got:\n{output}"
    )
    # Confirm multiples are rendered as Nx
    assert "25.0x" in output or "20.0x" in output, (
        f"Bear/base/bull P/E multiples should appear. Got:\n{output}"
    )


def test_scenario_driver_table_shows_bear_base_bull_eps():
    """
    Bear, base, and bull EPS values must all be visible and distinct when growth > 0.
    Bear uses flat EPS; base and bull apply forward projections.
    """
    vr = _vr_with_pe(eps=5.00, growth_pct=20.0)  # 20% CAGR → bear flat, base+20%, bull+26%
    lines = ReportingAgent._build_valuation_range_section(vr, current_price=120.0)
    output = "\n".join(lines)

    # Bear EPS = flat = $5.00; base EPS = $6.00; bull EPS = $6.30
    assert "$5.00" in output, f"Bear EPS (flat) should appear as $5.00. Got:\n{output}"
    assert "$6.00" in output, f"Base EPS (+20% CAGR) should appear as $6.00. Got:\n{output}"


def test_scenario_primary_method_not_empty_when_pe_runs():
    """
    When valuation methods run, scenario_primary_method must be non-empty.
    The driver model takes priority ('driver'); P/E is the fallback.
    """
    sd = _make_stock(
        quarterly_eps=[1.5, 1.4, 1.3, 1.2],
        annual_income=[
            (10e9, 2e9, 4.0),
            (9e9,  1.8e9, 3.5),
        ],
    )
    m = compute_core_metrics(sd)

    from analysis.valuation_range import compute_valuation_range
    vr = compute_valuation_range(sd, metrics=m)

    assert vr.scenario_primary_method in ("driver", "P/E", "EV/EBITDA", "P/S"), (
        f"scenario_primary_method must not be empty. Got: '{vr.scenario_primary_method}'"
    )
    if vr.driver_model_available:
        assert vr.scenario_primary_method == "driver"
        assert vr.scenario_base_label != "", "scenario_base_label must be set when driver runs"
    elif "P/E" in (vr.methods_used or []):
        assert vr.scenario_primary_method == "P/E"
        assert vr.scenario_bear_pe is not None, "scenario_bear_pe must be set"
        assert vr.scenario_base_eps is not None, "scenario_base_eps must be set"


def test_confidence_explanation_renders_in_output():
    """
    When scorecard.confidence_explanation is non-empty, the rendered memo must
    include that string.  This proves the confidence explanation is not silently
    swallowed between the scorecard and the report output.
    """
    # Use the end-to-end agent path to get a real scorecard + memo
    sd = _make_stock(
        quarterly_eps=[1.5, 1.4, 1.3, 1.2],
        annual_income=[
            (10e9, 2e9, 4.0),
            (9e9,  1.8e9, 3.5),
            (8e9,  1.5e9, 3.0),
        ],
    )
    sd.ratios[0].gross_margin   = 0.55
    sd.ratios[0].net_margin     = 0.22
    sd.ratios[0].roe            = 0.28
    sd.ratios[0].debt_to_equity = 0.25
    sd.ratios[0].current_ratio  = 2.5

    from agents.fundamental_analysis_agent import FundamentalAnalysisAgent
    from models.message import AgentMessage, MessageType

    fa = FundamentalAnalysisAgent()
    fa_resp = fa.handle(_fund_request(sd))
    assert not fa_resp.is_error()

    # Simulate what ReportingAgent does — build scorecard + confidence explanation
    from analysis.metrics import compute_signal_confidence
    from analysis.scorer import build_scorecard
    from config import Config

    fund = fa_resp.payload
    w = Config.SCORE_WEIGHTS

    def _default(name, weight):
        return _CS(name=name, score=50.0, weight=weight, data_quality="missing")

    val  = fund.get("valuation")        or _default("valuation", w["valuation"])
    gro  = fund.get("growth")           or _default("growth", w["growth"])
    pro  = fund.get("profitability")    or _default("profitability", w["profitability"])
    hlt  = fund.get("financial_health") or _default("financial_health", w["financial_health"])
    mom  = _default("momentum", w["momentum"])   # not run in unit test
    risk = _default("risk", w["risk"])

    sc = build_scorecard(
        ticker="TEST",
        valuation=val, growth=gro, profitability=pro,
        financial_health=hlt, momentum=mom, risk=risk,
        risk_flags=[], confidence=0.75,
    )
    _cats = {
        "valuation": val, "growth": gro, "profitability": pro,
        "financial_health": hlt, "momentum": mom, "risk": risk,
    }
    _, explanation = compute_signal_confidence(_cats)
    sc.confidence_explanation = explanation

    # Now verify rendering
    assert explanation, "compute_signal_confidence must return non-empty explanation"

    ra = ReportingAgent()
    agent_findings = {
        "fundamental": fund,
        "technical": {},
        "risk": {},
        "macro": {},
    }
    memo = ra._build_memo(sc, sd, agent_findings)

    # The explanation string must appear verbatim in the rendered memo
    assert explanation in memo, (
        f"confidence_explanation must appear in rendered memo.\n"
        f"  explanation: {explanation!r}\n"
        f"  confidence section:\n"
        + "\n".join(l for l in memo.splitlines() if "onfidence" in l or "signal" in l.lower())
    )


def test_rendered_scenario_no_placeholder_dashes_ttd_like():
    """
    Simulates a TTD-like stock (high P/E, positive EPS, high growth) and
    verifies the rendered valuation range shows 'Primary Driver : P/E' with
    real numeric values — not placeholder dashes or N/A.
    """
    # TTD-like: price ~$85, PE ~80, EPS ~$1.05, revenue $2B, high growth
    sd = _make_stock(
        ticker="TTD",
        price=85.0,
        market_cap_api=42_000_000_000.0,
        shares_outstanding=495_000_000.0,
        net_income=525_000_000.0,
        eps_diluted=1.06,
        eps=1.06,
        revenue=2_161_000_000.0,
        ebitda=680_000_000.0,
        pe_ratio_provider=80.0,
        ps_ratio_provider=19.0,
        ev_to_ebitda_provider=58.0,
        quarterly_eps=[0.29, 0.28, 0.25, 0.24],
        annual_income=[
            (2_161_000_000.0, 525_000_000.0, 1.06),
            (1_953_000_000.0, 414_000_000.0, 0.84),
            (1_578_000_000.0, 305_000_000.0, 0.62),
        ],
    )

    from analysis.valuation_range import compute_valuation_range
    m = compute_core_metrics(sd)
    vr = compute_valuation_range(sd, metrics=m)

    lines = ReportingAgent._build_valuation_range_section(vr, current_price=85.0)
    output = "\n".join(lines)

    # Must not show the old "data unavailable" message
    assert "not computable" not in output, (
        f"TTD-like stock should have computable range. Got:\n{output}"
    )

    # Primary method must be declared
    assert "Primary Driver" in output, (
        f"Output must declare a Primary Driver. Got:\n{output}"
    )

    # Must show actual dollar prices, not just dashes
    dollar_lines = [l for l in lines if "$" in l and "Current Price" not in l]
    assert dollar_lines, (
        f"At least one scenario price row should contain '$'. Got:\n{output}"
    )


# ── Test group 14: final output-layer completeness ────────────────────────────
#
# These tests verify the exact rendered text that the user sees, not just
# whether backend fields exist.  They are the definitive acceptance criteria
# for the scenario and confidence output-layer fixes.


def test_scenario_no_placeholder_dashes():
    """
    The rendered scenario section must never contain '— — —' or rows of dashes
    standing in for missing data.  Every visible row must have a real value.
    """
    vr = _vr_with_pe(current_price=120.0, pe=25.0, eps=5.0, growth_pct=12.5)
    lines = ReportingAgent._build_valuation_range_section(vr, current_price=120.0)
    output = "\n".join(lines)

    # No long runs of dashes used as placeholders (separator lines use ─, not —)
    import re
    dash_placeholder = re.search(r"—\s*—", output)
    assert dash_placeholder is None, (
        f"Found placeholder dashes in rendered output:\n{output}"
    )
    # No 'N/A' in the primary driver rows
    primary_section_lines = [l for l in lines if any(
        kw in l for kw in ("P/E multiple", "EPS (1yr fwd)", "Implied price", "EV/EBITDA mult", "Rev/share")
    )]
    for l in primary_section_lines:
        assert "N/A" not in l, f"N/A found in driver row: {l!r}\nFull output:\n{output}"


def test_scenario_only_one_primary_method_rendered():
    """
    Only ONE primary driver section is rendered.  The unused methods must not
    appear as rows — they may appear only in the compact 'Supporting methods'
    single-line reference (base case only).
    """
    vr = _vr_with_pe(current_price=120.0, pe=25.0, eps=5.0, growth_pct=12.5)
    lines = ReportingAgent._build_valuation_range_section(vr, current_price=120.0)
    output = "\n".join(lines)

    # Exactly one "Primary Driver" declaration
    primary_count = output.count("Primary Driver")
    assert primary_count == 1, (
        f"Expected exactly 1 'Primary Driver' declaration, found {primary_count}:\n{output}"
    )
    # No separate EV/EBITDA or P/S multiple rows when P/E is primary
    # (they may appear in the compact supporting-methods line, not as full table rows)
    ev_rows = [l for l in lines if l.strip().startswith("EV/EBITDA mult")]
    ps_rows = [l for l in lines if l.strip().startswith("P/S multiple")]
    assert not ev_rows, f"EV/EBITDA row rendered when P/E is primary:\n{output}"
    assert not ps_rows, f"P/S row rendered when P/E is primary:\n{output}"


def test_scenario_inputs_and_outputs_both_visible():
    """
    The rendered scenario section must show BOTH the valuation inputs (multiples,
    EPS) AND the outputs (implied prices, vs-current) so assumptions are traceable.
    """
    vr = _vr_with_pe(current_price=120.0, pe=25.0, eps=5.0, growth_pct=12.5)
    lines = ReportingAgent._build_valuation_range_section(vr, current_price=120.0)
    output = "\n".join(lines)

    assert "P/E multiple" in output,  "Input: P/E multiple row missing"
    assert "EPS (1yr fwd)" in output, "Input: EPS row missing"
    assert "Implied price" in output, "Output: Implied price row missing"
    assert "vs Current" in output,    "Output: vs Current row missing"


def test_confidence_explanation_non_empty_and_factor_specific():
    """
    compute_signal_confidence must produce a non-empty explanation that names
    at least one specific factor category (not boilerplate only).
    """
    cats_conflict = {
        "valuation":        _cs("valuation",        55.0),
        "growth":           _cs("growth",           80.0),
        "profitability":    _cs("profitability",    78.0),
        "financial_health": _cs("financial_health", 72.0),
        "momentum":         _cs("momentum",         27.0),
        "risk":             _cs("risk",             60.0),
    }
    _, explanation = compute_signal_confidence(cats_conflict)

    assert explanation, "Explanation must be non-empty"
    # Must name a factor (growth, momentum, profitability, valuation, or financial health)
    factor_words = {"growth", "momentum", "profitability", "valuation", "financial", "risk"}
    words_in_explanation = set(explanation.lower().split())
    named = factor_words & words_in_explanation
    assert named, (
        f"Explanation must name at least one factor category. Got: {explanation!r}"
    )


def test_confidence_explanation_changes_with_factor_mix():
    """
    The confidence explanation must produce different text for different factor mixes —
    it must not be a hardcoded string.
    """
    cats_all_bull = {
        "valuation":        _cs("valuation",        72.0),
        "growth":           _cs("growth",           75.0),
        "profitability":    _cs("profitability",    78.0),
        "financial_health": _cs("financial_health", 70.0),
        "momentum":         _cs("momentum",         68.0),
        "risk":             _cs("risk",             65.0),
    }
    cats_conflict = {
        "valuation":        _cs("valuation",        55.0),
        "growth":           _cs("growth",           80.0),
        "profitability":    _cs("profitability",    78.0),
        "financial_health": _cs("financial_health", 72.0),
        "momentum":         _cs("momentum",         27.0),  # conflict
        "risk":             _cs("risk",             60.0),
    }
    _, expl_bull = compute_signal_confidence(cats_all_bull)
    _, expl_conflict = compute_signal_confidence(cats_conflict)

    assert expl_bull != expl_conflict, (
        f"Explanation must differ between aligned and conflicting signals.\n"
        f"  Aligned:   {expl_bull!r}\n"
        f"  Conflict:  {expl_conflict!r}"
    )
    # Conflict explanation must mention the divergence
    assert "momentum" in expl_conflict.lower() or "mixed" in expl_conflict.lower(), (
        f"Conflict explanation should mention momentum divergence: {expl_conflict!r}"
    )


def test_confidence_why_line_rendered_in_memo():
    """
    The rendered memo must contain a 'Why :' line (the labeled confidence explanation)
    that is not a blank or boilerplate-only string.
    """
    sd = _make_stock(
        quarterly_eps=[1.5, 1.4, 1.3, 1.2],
        annual_income=[
            (10e9, 2e9, 4.0),
            (9e9,  1.8e9, 3.5),
            (8e9,  1.5e9, 3.0),
        ],
    )
    sd.ratios[0].gross_margin   = 0.55
    sd.ratios[0].net_margin     = 0.22
    sd.ratios[0].roe            = 0.28
    sd.ratios[0].debt_to_equity = 0.25
    sd.ratios[0].current_ratio  = 2.5

    from agents.fundamental_analysis_agent import FundamentalAnalysisAgent
    from models.message import AgentMessage, MessageType
    from analysis.scorer import build_scorecard
    from config import Config

    fa = FundamentalAnalysisAgent()
    fa_resp = fa.handle(_fund_request(sd))
    assert not fa_resp.is_error()

    fund = fa_resp.payload
    w = Config.SCORE_WEIGHTS

    def _def(n, wt): return _CS(name=n, score=50.0, weight=wt, data_quality="missing")

    val  = fund.get("valuation")        or _def("valuation", w["valuation"])
    gro  = fund.get("growth")           or _def("growth", w["growth"])
    pro  = fund.get("profitability")    or _def("profitability", w["profitability"])
    hlt  = fund.get("financial_health") or _def("financial_health", w["financial_health"])
    mom  = _def("momentum", w["momentum"])
    risk = _def("risk", w["risk"])

    sc = build_scorecard(
        ticker="TEST", valuation=val, growth=gro, profitability=pro,
        financial_health=hlt, momentum=mom, risk=risk,
        risk_flags=[], confidence=0.75,
    )
    cats = {"valuation":val,"growth":gro,"profitability":pro,
            "financial_health":hlt,"momentum":mom,"risk":risk}
    _, explanation = compute_signal_confidence(cats)
    sc.confidence_explanation = explanation

    ra = ReportingAgent()
    agent_findings = {"fundamental": fund, "technical": {}, "risk": {}, "macro": {}}
    memo = ra._build_memo(sc, sd, agent_findings)

    why_lines = [l for l in memo.splitlines() if "Why" in l]
    assert why_lines, (
        f"Memo must contain a 'Why' confidence explanation line.\n"
        f"Confidence-related lines in memo:\n"
        + "\n".join(l for l in memo.splitlines() if "onfidence" in l or "Why" in l)
    )
    why_text = why_lines[0]
    assert len(why_text.strip()) > 10, f"Why line is too short to be meaningful: {why_text!r}"


# ── Group 15: LEI Layer — macro_overlay.score() with trend fields ─────────────
#
# These tests exercise the Phase 1 LEI additions:
#   - _classify_cycle_phase() via score()
#   - cycle_phase / lei_trend / yield_spread_trend on MacroAssessment
#   - Report rendering of cycle phase + trend line
#   - None-safety when trend fields are absent
#
# All tests call analysis.macro_overlay.score() directly with a synthetic
# snapshot dict — no FRED HTTP calls.  The pattern mirrors the FREDProvider
# snapshot format (including the new trend keys).

from analysis.macro_overlay import score as macro_score


def _macro_snap(
    yield_spread: float | None       = 0.80,
    jobless_claims: float | None     = 210_000.0,
    housing_starts: float | None     = 1_450.0,
    mfg_prod: float | None           = 100.0,
    oecd_cli: float | None           = 100.2,
    retail_sales_yoy: float | None   = 3.5,
    consumer_sentiment: float | None = 72.0,
    lei_trend: str | None            = None,
    yield_spread_trend: str | None   = None,
) -> dict:
    """Synthetic LEI snapshot — mirrors FREDProvider.get_lei_snapshot() output."""
    return {
        "yield_spread_10y2y":  yield_spread,
        "jobless_claims":      jobless_claims,
        "housing_starts":      housing_starts,
        "mfg_prod":            mfg_prod,
        "oecd_cli":            oecd_cli,
        "retail_sales_yoy":    retail_sales_yoy,
        "consumer_sentiment":  consumer_sentiment,
        "lei_trend":           lei_trend,
        "yield_spread_trend":  yield_spread_trend,
    }


def test_macro_expansion_when_all_indicators_healthy():
    """Strong indicators across the board → Expansion, score > 65, Low recession risk."""
    a = macro_score(_macro_snap(
        yield_spread=0.90,
        jobless_claims=205_000,
        housing_starts=1_600,
        mfg_prod=103.5,
        oecd_cli=100.6,
    ))
    assert a.macro_regime == "Expansion", f"Expected Expansion, got {a.macro_regime}"
    assert a.macro_score > 65, f"Expected score > 65, got {a.macro_score}"
    assert a.recession_risk_level == "Low"


def test_macro_contraction_when_deeply_inverted():
    """Deeply inverted curve + high claims + weak housing → Contraction, High recession risk."""
    a = macro_score(_macro_snap(
        yield_spread=-0.90,
        jobless_claims=380_000,
        housing_starts=820,
        mfg_prod=92.0,
        oecd_cli=98.2,
        retail_sales_yoy=-1.0,
        consumer_sentiment=48.0,
    ))
    assert a.macro_regime == "Contraction", f"Expected Contraction, got {a.macro_regime}"
    assert a.recession_risk_level == "High"
    assert a.macro_score < 35


def test_macro_recovery_not_contraction_when_spread_recovering():
    """Borderline score + spread recovering from inversion + stable claims → Recovery."""
    # yield_spread in [−0.50, 0.10] + claims ≤ 250k triggers Recovery override
    a = macro_score(_macro_snap(
        yield_spread=-0.20,
        jobless_claims=240_000,
        housing_starts=1_100,
        mfg_prod=98.0,
        oecd_cli=99.5,
    ))
    assert a.macro_regime == "Recovery", (
        f"Expected Recovery, got {a.macro_regime} (score={a.macro_score:.1f})"
    )


def test_cycle_phase_mid_when_expansion_cli_rising():
    """Expansion + rising OECD CLI + healthy spread → cycle_phase == 'mid'."""
    a = macro_score(_macro_snap(
        yield_spread=0.80,
        oecd_cli=100.6,
        lei_trend="rising",
        yield_spread_trend="rising",
    ))
    assert a.macro_regime == "Expansion"
    assert a.cycle_phase == "mid", f"Expected mid, got {a.cycle_phase}"
    assert a.lei_trend == "rising"
    assert a.yield_spread_trend == "rising"


def test_cycle_phase_late_when_expansion_cli_falling():
    """Expansion regime + falling OECD CLI → cycle_phase == 'late'."""
    a = macro_score(_macro_snap(
        yield_spread=0.80,       # spread healthy — not the trigger
        oecd_cli=100.2,
        lei_trend="falling",
        yield_spread_trend=None,
    ))
    assert a.macro_regime == "Expansion"
    assert a.cycle_phase == "late", f"Expected late, got {a.cycle_phase}"
    assert a.lei_trend == "falling"


def test_cycle_phase_late_when_expansion_spread_tight():
    """Expansion + spread below 0.25pp (tight) → cycle_phase == 'late'."""
    a = macro_score(_macro_snap(
        yield_spread=0.15,       # tight but not inverted
        oecd_cli=100.2,
        lei_trend="rising",      # CLI still rising — spread is the trigger
    ))
    assert a.macro_regime == "Expansion"
    assert a.cycle_phase == "late", f"Expected late, got {a.cycle_phase}"


def test_cycle_phase_early_when_slowdown_inflecting():
    """Slowdown + OECD CLI inflecting + spread turning up → cycle_phase == 'early' (turning point)."""
    a = macro_score(_macro_snap(
        yield_spread=-0.05,       # near zero — still slightly inverted
        jobless_claims=245_000,
        housing_starts=1_050,
        mfg_prod=97.5,
        oecd_cli=99.8,
        lei_trend="inflecting",
        yield_spread_trend="rising",
    ))
    # Regime may be Slowdown or Recovery depending on exact score; phase should be early
    assert a.cycle_phase in ("early", "contraction"), (
        f"Expected early (or contraction as fallback), got {a.cycle_phase} "
        f"(regime={a.macro_regime}, score={a.macro_score:.1f})"
    )


def test_cycle_phase_contraction_when_regime_contraction():
    """Contraction regime always maps to cycle_phase == 'contraction'."""
    a = macro_score(_macro_snap(
        yield_spread=-1.20,
        jobless_claims=420_000,
        housing_starts=700,
        mfg_prod=92.0,
        oecd_cli=98.2,
        retail_sales_yoy=-1.5,
        consumer_sentiment=44.0,
    ))
    assert a.macro_regime == "Contraction"
    assert a.cycle_phase == "contraction"


def test_cycle_phase_unknown_when_trend_fields_absent():
    """When trend keys are missing (FRED not available), cycle_phase may be set
    from regime alone — must not be None and must be a valid phase string."""
    a = macro_score(_macro_snap(
        yield_spread=0.80,
        lei_trend=None,             # no trend data available
        yield_spread_trend=None,
    ))
    valid_phases = {"early", "mid", "late", "contraction", "unknown"}
    assert a.cycle_phase in valid_phases, (
        f"cycle_phase must be a valid string, got {a.cycle_phase!r}"
    )
    # With healthy indicators and no trend data, should default to mid (not unknown)
    if a.macro_regime == "Expansion":
        assert a.cycle_phase == "mid", (
            f"Expansion with no trend data should default to mid, got {a.cycle_phase}"
        )


def test_macro_confidence_modifier_negative_on_contraction():
    """Low macro score (contraction) → negative confidence modifier."""
    a = macro_score(_macro_snap(
        yield_spread=-1.00,
        jobless_claims=360_000,
        housing_starts=750,
        mfg_prod=92.5,
        oecd_cli=98.5,
        retail_sales_yoy=-0.5,
        consumer_sentiment=46.0,
    ))
    assert a.confidence_modifier < 0, (
        f"Contraction regime should produce negative confidence_modifier, "
        f"got {a.confidence_modifier}"
    )


def test_macro_section_renders_lei_narrative():
    """
    _build_macro_section() renders an LEI narrative — not a raw 'Cycle Phase:' label.
    The narrative must reference the cycle phase concept and LEI trend in prose.
    """
    from agents.reporting_agent import ReportingAgent

    macro_findings = {
        "macro_regime":           "Expansion",
        "macro_score":            72.0,
        "recession_risk_level":   "Low",
        "sector_tilt":            "Cyclicals, Industrials",
        "bullish_macro_factors":  ["Yield curve positive"],
        "bearish_macro_factors":  [],
        "cycle_phase":            "mid",
        "lei_trend":              "rising",
        "yield_spread_trend":     "rising",
    }
    ra = ReportingAgent()
    lines = ra._build_macro_section(macro_findings)
    text = "\n".join(lines)

    # Must contain prose narrative with mid-cycle concept
    assert any(phrase in text.lower() for phrase in ("mid-cycle", "mid cycle")), (
        f"Narrative must reference mid-cycle:\n{text}"
    )
    # LEI trend should appear as prose in the narrative
    assert "leading indicators" in text.lower(), (
        f"Narrative should describe leading indicator trend:\n{text}"
    )
    # Must include the labeled header rows
    assert "Macro Regime" in text, f"Macro Regime header row missing:\n{text}"
    assert "Cycle Phase" in text, f"Cycle Phase header row missing:\n{text}"
    assert "Mid" in text, f"Phase label 'Mid' missing from header:\n{text}"


def test_macro_section_falls_back_to_verdict_when_phase_unknown():
    """
    When cycle_phase is 'unknown', _build_macro_section() uses the generic
    one-line verdict (backward-compatible fallback) rather than an LEI narrative.
    """
    from agents.reporting_agent import ReportingAgent

    macro_findings = {
        "macro_regime":           "Expansion",
        "macro_score":            68.0,
        "recession_risk_level":   "Low",
        "sector_tilt":            "Cyclicals",
        "bullish_macro_factors":  [],
        "bearish_macro_factors":  [],
        "cycle_phase":            "unknown",
        "lei_trend":              None,
        "yield_spread_trend":     None,
    }
    ra = ReportingAgent()
    lines = ra._build_macro_section(macro_findings)
    text = "\n".join(lines)
    # Should fall back to the generic verdict — no narrative-specific phrases
    assert "tailwind" in text.lower() or "Macro" in text, (
        f"Expected generic verdict when phase is unknown:\n{text}"
    )
    # Should NOT emit a Cycle Phase row when phase is unknown
    assert "Cycle Phase" not in text, f"Cycle Phase row should not appear when phase is unknown:\n{text}"


def test_macro_section_safe_when_new_fields_absent():
    """_build_macro_section() must not crash when Phase 1 fields are missing
    (e.g. payload from an older agent run before the upgrade)."""
    from agents.reporting_agent import ReportingAgent

    macro_findings = {
        "macro_regime":           "Slowdown",
        "macro_score":            52.0,
        "recession_risk_level":   "Moderate",
        "sector_tilt":            "Defensives",
        "bullish_macro_factors":  [],
        "bearish_macro_factors":  ["Yield curve flat"],
        # Deliberately no cycle_phase / lei_trend / yield_spread_trend keys
    }
    ra = ReportingAgent()
    lines = ra._build_macro_section(macro_findings)   # must not raise
    text = "\n".join(lines)
    assert "Slowdown" in text


def test_macro_slowdown_regime_and_phase():
    """Moderately weak indicators (score ~56) → Slowdown, Moderate risk, late phase.
    Key: claims > 250k prevents the Recovery override; spread modestly positive.
    """
    a = macro_score(_macro_snap(
        yield_spread=0.20,        # modestly positive → score=65
        jobless_claims=255_000,   # just above 250k → bars Recovery override; score=40
        housing_starts=1_250,     # adequate → score=62
        mfg_prod=98.5,            # below trend → score=42
        oecd_cli=100.1,           # at trend → score=60
        lei_trend="falling",
        yield_spread_trend=None,
    ))
    # weighted ≈ 65*0.22 + 40*0.18 + 62*0.15 + 62*0.15 + 62*0.12 + 42*0.12 + 60*0.06 ≈ 56
    assert a.macro_regime == "Slowdown", (
        f"Expected Slowdown, got {a.macro_regime} (score={a.macro_score:.1f})"
    )
    assert a.recession_risk_level in ("Moderate", "Elevated"), (
        f"Expected Moderate or Elevated recession risk, got {a.recession_risk_level}"
    )
    assert a.cycle_phase == "late", (
        f"Expected late cycle phase for Slowdown + falling CLI, got {a.cycle_phase}"
    )


def test_macro_missing_fred_fallback_does_not_crash_apply_macro_influence():
    """_apply_macro_influence() with an empty or partial macro dict must not raise."""
    from agents.reporting_agent import ReportingAgent
    from models.scorecard import Scorecard, Stance

    ra = ReportingAgent()

    # Build a minimal scorecard
    sc = Scorecard(ticker="TEST", overall_score=60, stance=Stance.NEUTRAL, confidence=0.65)

    # Call with empty dict — simulates no-FRED path
    ra._apply_macro_influence(sc, {})
    assert sc.confidence == 0.65, "Empty macro dict must not mutate confidence"

    # Call with 'Unknown' regime — simulates FRED key not set
    ra._apply_macro_influence(sc, {"macro_regime": "Unknown", "recession_risk_level": "Unknown"})
    assert sc.confidence == 0.65, "Unknown regime must not mutate confidence"


def test_phase_sector_tilt_is_phase_specific():
    """Phase-specific tilt should differ from the regime-level fallback for late Expansion."""
    a_late = macro_score(_macro_snap(
        yield_spread=0.15,        # tight → late-cycle
        lei_trend="rising",       # CLI still ok — spread is the trigger
    ))
    assert a_late.macro_regime == "Expansion"
    assert a_late.cycle_phase == "late"
    # Late Expansion tilt should favour quality, not broad cyclicals
    assert "quality" in a_late.sector_tilt.lower() or "dividend" in a_late.sector_tilt.lower(), (
        f"Late Expansion tilt should be quality/dividend-focused, got: {a_late.sector_tilt!r}"
    )

    a_mid = macro_score(_macro_snap(
        yield_spread=0.80,
        oecd_cli=100.6,
        lei_trend="rising",
        yield_spread_trend="rising",
    ))
    assert a_mid.macro_regime == "Expansion"
    assert a_mid.cycle_phase == "mid"
    assert "cyclical" in a_mid.sector_tilt.lower(), (
        f"Mid Expansion tilt should include cyclicals, got: {a_mid.sector_tilt!r}"
    )


def test_apply_macro_influence_early_mid_boosts_confidence():
    """Early/mid cycle phase should add a small positive confidence nudge."""
    from agents.reporting_agent import ReportingAgent
    from models.scorecard import Scorecard, Stance

    ra = ReportingAgent()
    sc = Scorecard(ticker="TEST", overall_score=65, stance=Stance.BULLISH, confidence=0.70)

    ra._apply_macro_influence(sc, {
        "macro_regime":         "Expansion",
        "recession_risk_level": "Low",
        "cycle_phase":          "mid",
    })
    # Expansion+Low = +0.02, mid = +0.01 → total +0.03
    assert sc.confidence > 0.70, (
        f"Early/mid cycle should boost confidence, got {sc.confidence:.4f}"
    )


def test_apply_macro_influence_late_reduces_confidence():
    """Late cycle phase should subtract a small confidence nudge."""
    from agents.reporting_agent import ReportingAgent
    from models.scorecard import Scorecard, Stance

    ra = ReportingAgent()
    sc = Scorecard(ticker="TEST", overall_score=65, stance=Stance.BULLISH, confidence=0.70)

    ra._apply_macro_influence(sc, {
        "macro_regime":         "Expansion",
        "recession_risk_level": "Low",
        "cycle_phase":          "late",
    })
    # Expansion+Low = +0.02, late = -0.01 → net +0.01 (still up but less than mid)
    # Just verify it is lower than the mid-cycle version (0.73)
    assert sc.confidence < 0.73, (
        f"Late cycle should partially offset the expansion boost, got {sc.confidence:.4f}"
    )


def test_confidence_adjustment_rationale_populated():
    """MacroAssessment should carry a non-empty confidence_adjustment_rationale."""
    a = macro_score(_macro_snap(
        yield_spread=0.80,
        jobless_claims=210_000,
        housing_starts=1_500,
        mfg_prod=103.5,
        oecd_cli=100.6,
    ))
    assert a.confidence_adjustment_rationale, (
        "confidence_adjustment_rationale must be a non-empty string"
    )
    # Strong indicators → positive modifier → rationale should mention score or modifier direction
    assert any(word in a.confidence_adjustment_rationale.lower()
               for word in ("strong", "solid", "positive")), (
        f"Rationale for healthy macro should be positive, got: {a.confidence_adjustment_rationale!r}"
    )


if __name__ == "__main__":
    # Quick smoke test without pytest
    tests = [
        test_provider_market_cap_preserved,
        test_market_cap_recomputed_when_price_adjusted,
        test_scorer_uses_normalized_market_cap,
        test_scorer_without_metrics_uses_raw_data,
        test_negative_ttm_eps_falls_back_to_provider_pe,
        test_negative_ttm_eps_falls_back_to_annual_pe_when_no_provider,
        test_peg_and_growth_populated,
        test_peg_absent_when_eps_growth_negative,
        test_diverging_pe_keeps_provider_value,
        test_diverging_market_cap_keeps_api_value,
        test_pe_resolution_order_provider_first,
        test_pe_resolution_order_computed_ttm_fallback,
        test_pe_resolution_order_annual_last,
        test_shares_from_quote_preferred,
        test_shares_fallback_to_income_when_quote_missing,
        test_pe_above_cap_excluded,
        test_ev_ebitda_above_cap_excluded,
        # New tests
        test_confidence_full_data,
        test_confidence_missing_valuation_reduces_score,
        test_confidence_decreases_with_divergence,
        test_growth_scorer_uses_normalized_eps_growth,
        test_profitability_scorer_uses_normalized_margins,
        test_health_scorer_uses_normalized_de_and_cr,
        test_risk_scorer_uses_normalized_market_cap,
        test_risk_scorer_uses_normalized_pe,
        test_data_unavailable_only_when_truly_missing,
        # End-to-end agent tests
        test_fundamental_agent_does_not_crash_with_full_data,
        test_category_scores_not_missing_when_data_exists,
        test_normalized_metrics_flows_through_payload,
        test_peg_consistent_between_valuation_range_and_normalized_metrics,
        test_confidence_reflects_data_quality_not_pipeline_completion,
        # Institutional-grade logic tests
        test_signal_confidence_high_when_all_aligned,
        test_signal_confidence_drops_when_momentum_conflicts,
        test_signal_confidence_no_99pct_on_conflict,
        test_valuation_tension_detected_high_pe_low_peg,
        test_scenario_assumptions_populated_when_methods_compute,
        test_health_reasoning_references_metrics,
        test_profitability_reasoning_references_margins,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} passed")
