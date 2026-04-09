"""
Risk scoring module.

Identifies specific red flags and computes a risk penalty score.
Risk score: 100 = minimal risk; 0 = extreme risk.
Higher score = safer stock.

When NormalizedMetrics is supplied:
  - market_cap  comes from metrics.market_cap  (authoritative API value)
  - pe_ratio    comes from metrics.pe_ratio    (consistent with report header)
  - debt_to_equity comes from metrics.debt_to_equity
This guarantees the risk flags reference the same values shown everywhere else.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from models.scorecard import CategoryScore
from models.stock_data import StockData
from utils.helpers import clamp, pct_change

if TYPE_CHECKING:
    from analysis.metrics import NormalizedMetrics


def score_risk(
    stock_data: StockData,
    weight: float = 0.10,
    metrics: "Optional[NormalizedMetrics]" = None,
) -> tuple[CategoryScore, list[str]]:
    """Returns (CategoryScore, risk_flags).

    When *metrics* is provided, market_cap, pe_ratio, and debt_to_equity
    are sourced from NormalizedMetrics so the risk flags are internally
    consistent with all other report sections.
    """
    flags: list[str] = []
    factors: list[str] = []
    penalty = 0.0

    ratios   = stock_data.latest_ratios
    balance  = stock_data.latest_balance
    income   = stock_data.latest_income
    cashflow = stock_data.latest_cashflow
    profile  = stock_data.profile

    # ── Resolve metrics-backed fields ─────────────────────────────────────────
    if metrics is not None:
        market_cap: Optional[float] = metrics.market_cap
        pe: Optional[float] = metrics.pe_ratio
        de: Optional[float] = metrics.debt_to_equity
        print(
            f"  [RISK] from NormalizedMetrics:"
            f" mktcap={market_cap} pe={pe}({metrics.pe_source}) de={de}"
        )
    else:
        market_cap = stock_data.market_cap
        pe = ratios.pe_ratio if ratios else None
        de = ratios.debt_to_equity if ratios else None

    # ── Negative free cash flow ────────────────────────────────────────────────
    if cashflow and cashflow.free_cash_flow is not None:
        if cashflow.free_cash_flow < 0:
            penalty += 18.0
            msg = f"Negative FCF (${cashflow.free_cash_flow/1e9:.2f}B) — company burns cash"
            flags.append(msg)
            factors.append(f"[RISK] {msg}")

    # ── High leverage ──────────────────────────────────────────────────────────
    if de is not None:
        if de > 3.0:
            penalty += 20.0
            msg = f"Very high D/E ratio ({de:.1f}x) — heavy debt burden"
            flags.append(msg)
            factors.append(f"[RISK] {msg}")
        elif de > 2.0:
            penalty += 12.0
            msg = f"High D/E ratio ({de:.1f}x)"
            flags.append(msg)
            factors.append(f"[RISK] {msg}")
        elif de > 1.0:
            penalty += 5.0
            factors.append(f"Moderate leverage (D/E {de:.1f}x)")

    # ── Margin compression ─────────────────────────────────────────────────────
    if len(stock_data.income_statements) >= 2:
        curr_gm = stock_data.income_statements[0].gross_profit_ratio
        prev_gm = stock_data.income_statements[1].gross_profit_ratio
        if curr_gm is not None and prev_gm is not None:
            delta = curr_gm - prev_gm
            if delta < -0.05:
                penalty += 14.0
                msg = f"Gross margin compressed {delta*100:.1f}pp YoY — pricing power concern"
                flags.append(msg)
                factors.append(f"[RISK] {msg}")
            elif delta < -0.02:
                penalty += 6.0
                factors.append(f"Gross margin slightly compressed ({delta*100:.1f}pp YoY)")

    # ── Earnings deterioration ─────────────────────────────────────────────────
    if len(stock_data.income_statements) >= 2:
        curr_eps = stock_data.income_statements[0].eps_diluted
        prev_eps = stock_data.income_statements[1].eps_diluted
        if curr_eps is not None and prev_eps is not None and prev_eps != 0:
            chg = pct_change(curr_eps, prev_eps)
            if chg is not None and chg < -0.20:
                penalty += 14.0
                msg = f"EPS declined {chg*100:.1f}% YoY — earnings deteriorating"
                flags.append(msg)
                factors.append(f"[RISK] {msg}")
            elif chg is not None and chg < -0.05:
                penalty += 6.0
                factors.append(f"EPS declining modestly ({chg*100:.1f}% YoY)")

    # ── Revenue contraction ────────────────────────────────────────────────────
    if len(stock_data.income_statements) >= 2:
        curr_rev = stock_data.income_statements[0].revenue
        prev_rev = stock_data.income_statements[1].revenue
        if curr_rev is not None and prev_rev is not None:
            chg = pct_change(curr_rev, prev_rev)
            if chg is not None and chg < -0.10:
                penalty += 10.0
                msg = f"Revenue contracted {chg*100:.1f}% YoY"
                flags.append(msg)
                factors.append(f"[RISK] {msg}")

    # ── Extreme valuation — use normalized PE, not raw ─────────────────────────
    if pe is not None:
        if pe > 100:
            penalty += 12.0
            msg = f"Extreme P/E ratio ({pe:.0f}x) — priced for perfection"
            flags.append(msg)
            factors.append(f"[RISK] {msg}")
        elif pe > 60:
            penalty += 6.0
            factors.append(f"High P/E ({pe:.0f}x) — little room for error")

    # ── High beta ─────────────────────────────────────────────────────────────
    if profile and profile.beta is not None:
        beta = profile.beta
        if beta > 2.5:
            penalty += 10.0
            msg = f"Very high beta ({beta:.1f}) — extremely volatile"
            flags.append(msg)
            factors.append(f"[RISK] {msg}")
        elif beta > 1.8:
            penalty += 5.0
            factors.append(f"Elevated beta ({beta:.1f}) — high market sensitivity")

    # ── Weak interest coverage ─────────────────────────────────────────────────
    if ratios and ratios.interest_coverage is not None:
        ic = ratios.interest_coverage
        if 0 < ic < 1.5:
            penalty += 16.0
            msg = f"Interest coverage {ic:.1f}x — barely covers debt service"
            flags.append(msg)
            factors.append(f"[RISK] {msg}")

    # ── Market cap / liquidity risk — use normalized market cap ───────────────
    if market_cap is not None:
        if market_cap < 300_000_000:
            penalty += 8.0
            factors.append(f"Micro-cap (${market_cap/1e6:.0f}M) — liquidity and concentration risk")
        elif market_cap < 2_000_000_000:
            penalty += 4.0
            factors.append(f"Small-cap (${market_cap/1e9:.2f}B) — elevated volatility risk")

    # ── Goodwill concentration ────────────────────────────────────────────────
    if balance and balance.goodwill and balance.total_assets:
        gw_pct = balance.goodwill / balance.total_assets
        if gw_pct > 0.40:
            penalty += 5.0
            factors.append(f"Goodwill = {gw_pct*100:.1f}% of assets — impairment risk")

    # ── Final score ────────────────────────────────────────────────────────────
    risk_score = clamp(100.0 - penalty)

    if not flags:
        factors.append("No major risk flags detected")
        reasoning = "No significant financial or structural red flags identified."
    elif len(flags) <= 2:
        reasoning = f"{len(flags)} risk flag(s) identified — manageable but warrant monitoring."
    else:
        reasoning = f"{len(flags)} risk flags — meaningful headwinds that need active monitoring."

    return (
        CategoryScore(
            name="risk",
            score=risk_score,
            weight=weight,
            factors=factors,
            reasoning=reasoning,
            data_quality="good" if ratios and balance else "partial",
        ),
        flags,
    )
