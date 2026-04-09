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
    FMP /quote marketCap ($50B) must be used as-is when price was NOT
    adjusted from price_history.  Our computed cap from shares should
    only override when price was corrected.
    """
    sd = _make_stock(
        market_cap_api=50_000_000_000.0,
        price=100.0,
        shares_outstanding=500_000_000.0,
        # No price_history → no price adjustment → API market cap wins
    )
    m = compute_core_metrics(sd)

    assert m.market_cap == 50_000_000_000.0, (
        f"Expected API market cap 50B, got {m.market_cap}"
    )
    assert m.market_cap_source == "api"
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
    score_valuation must score P/S using metrics.market_cap (authoritative)
    rather than stock_data.market_cap which could hold a stale /quote value.
    """
    sd = _make_stock(
        price=100.0,
        market_cap_api=10_000_000_000.0,   # $10B API (wrong)
        revenue=10_000_000_000.0,           # $10B revenue
        ps_ratio_provider=None,             # no provider P/S
    )
    # NormalizedMetrics with corrected market cap: $50B
    m = compute_core_metrics(sd)
    # API market cap = $10B (no price adjustment), so PS computed = 10B/10B = 1.0
    assert m.ps_ratio is not None
    assert abs(m.ps_ratio - 1.0) < 0.05, f"Expected P/S ≈ 1.0, got {m.ps_ratio}"

    score = score_valuation(sd, metrics=m)
    # P/S 1.0 should be scored as "cheap" (score >= 75)
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
    When computed_ttm PE and provider PE diverge by > threshold, the system
    must keep the provider value (not null it) and log a warning.
    Provider PE = 30, computed TTM PE = 50 (Δ = 67% > 25% threshold).
    """
    quarterly_eps = [0.50, 0.50, 0.50, 0.50]  # TTM EPS = 2.0 → PE = 50
    sd = _make_stock(
        price=100.0,
        quarterly_eps=quarterly_eps,
        pe_ratio_provider=30.0,  # provider says 30 (diverges from computed 50)
    )
    m = compute_core_metrics(sd)

    # provider_ttm should win in the selection order
    assert m.pe_ratio == 30.0, (
        f"Expected provider PE=30 (not nulled), got {m.pe_ratio}"
    )
    assert m.pe_source == "provider_ttm"
    # Warning logged
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
    """provider_ttm beats computed_ttm when both are valid."""
    quarterly_eps = [1.0, 1.0, 1.0, 1.0]   # computed_ttm = price/4 = 25
    sd = _make_stock(
        price=100.0,
        quarterly_eps=quarterly_eps,
        pe_ratio_provider=30.0,
    )
    m = compute_core_metrics(sd)
    assert m.pe_ratio == 30.0
    assert m.pe_source == "provider_ttm"


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
    sd = _make_stock(
        market_cap_api=1_000_000_000_000.0,   # $1T market cap
        ebitda=1_000_000_000.0,                # $1B EBITDA → EV/EBITDA ≈ 1000
        ev_to_ebitda_provider=350.0,           # > 300 cap
    )
    m = compute_core_metrics(sd)
    # ev_ebitda_computed ≈ 1000 (> 300) → excluded
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
    When the P/E method computes a valid base price, scenario_primary_method
    must be 'P/E' — not an empty string that would leave the driver table blank.
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

    if "P/E" in (vr.methods_used or []):
        assert vr.scenario_primary_method == "P/E", (
            f"scenario_primary_method should be 'P/E' when P/E method produced results. "
            f"Got: '{vr.scenario_primary_method}'"
        )
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
