"""
Financial health scoring module.

Assesses balance sheet strength, liquidity, and debt serviceability.

When NormalizedMetrics is supplied, debt_to_equity and current_ratio are
sourced from the normalized object so the scorecard matches the report header.
Balance-sheet-level checks (cash position, FCF, interest coverage) still read
raw data because those fields are not carried in NormalizedMetrics.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from models.scorecard import CategoryScore
from models.stock_data import StockData
from utils.helpers import clamp, safe_divide

if TYPE_CHECKING:
    from analysis.metrics import NormalizedMetrics


def score_financial_health(
    stock_data: StockData,
    weight: float = 0.20,
    metrics: "Optional[NormalizedMetrics]" = None,
) -> CategoryScore:
    """
    Compute a 0-100 balance sheet health score.
    100 = fortress balance sheet: zero debt, ample liquidity, positive FCF.

    When *metrics* is provided, debt_to_equity and current_ratio are sourced
    from NormalizedMetrics — the same values shown in the report header.
    """
    ratios  = stock_data.latest_ratios
    balance = stock_data.latest_balance
    income  = stock_data.latest_income
    factors: list[str] = []
    sub_scores: list[tuple[float, float]] = []

    # ── Debt-to-Equity ─────────────────────────────────────────────────────────
    if metrics is not None and metrics.debt_to_equity is not None:
        de_ratio: Optional[float] = metrics.debt_to_equity
        print(f"  [HEALTH] D/E from NormalizedMetrics: {de_ratio}")
    else:
        de_ratio = ratios.debt_to_equity if ratios else None
        if de_ratio is None and balance:
            de_ratio = safe_divide(balance.total_debt, balance.total_equity)

    if de_ratio is None:
        de_s, de_f = 50.0, "D/E ratio: N/A"
    elif de_ratio < 0:
        de_s, de_f = 20.0, f"D/E negative ({de_ratio:.2f}) — equity deficit"
    elif de_ratio < 0.30:
        de_s, de_f = 95.0, f"D/E {de_ratio:.2f} — very low leverage"
    elif de_ratio < 0.60:
        de_s, de_f = 82.0, f"D/E {de_ratio:.2f} — conservative leverage"
    elif de_ratio < 1.0:
        de_s, de_f = 68.0, f"D/E {de_ratio:.2f} — moderate leverage"
    elif de_ratio < 1.5:
        de_s, de_f = 52.0, f"D/E {de_ratio:.2f} — elevated leverage"
    elif de_ratio < 2.5:
        de_s, de_f = 35.0, f"D/E {de_ratio:.2f} — high leverage, watch carefully"
    else:
        de_s, de_f = 18.0, f"D/E {de_ratio:.2f} — very high leverage, material risk"
    sub_scores.append((de_s, 0.30))
    factors.append(de_f)

    # ── Current Ratio ──────────────────────────────────────────────────────────
    if metrics is not None and metrics.current_ratio is not None:
        cr: Optional[float] = metrics.current_ratio
        print(f"  [HEALTH] Current ratio from NormalizedMetrics: {cr}")
    else:
        cr = ratios.current_ratio if ratios else None
        if cr is None and balance:
            cr = safe_divide(balance.total_current_assets, balance.total_current_liabilities)

    if cr is None:
        cr_s, cr_f = 50.0, "Current ratio: N/A"
    elif cr < 0.8:
        cr_s, cr_f = 15.0, f"Current ratio {cr:.2f} — liquidity risk"
    elif cr < 1.0:
        cr_s, cr_f = 35.0, f"Current ratio {cr:.2f} — below 1x, watch liquidity"
    elif cr < 1.3:
        cr_s, cr_f = 55.0, f"Current ratio {cr:.2f} — adequate"
    elif cr < 2.0:
        cr_s, cr_f = 78.0, f"Current ratio {cr:.2f} — healthy liquidity"
    else:
        cr_s, cr_f = 92.0, f"Current ratio {cr:.2f} — very strong liquidity"
    sub_scores.append((cr_s, 0.25))
    factors.append(cr_f)

    # ── Interest Coverage — always from raw (not in NormalizedMetrics) ─────────
    ic = ratios.interest_coverage if ratios else None
    if ic is None and income and income.operating_income and income.interest_expense:
        ie = abs(income.interest_expense)
        ic = safe_divide(income.operating_income, ie) if ie else None
    # Treat 0.0 as invalid — APIs sometimes return 0 for companies with no debt
    # or when interest expense is missing.  A genuine 0x coverage would be caught
    # by the negative-operating-income case already handled in profitability.
    if ic is not None and ic == 0.0:
        ic = None

    if ic is None:
        ic_s, ic_f = 50.0, "Interest coverage: N/A (likely no debt)"
    elif ic < 0:
        ic_s, ic_f = 10.0, f"Interest coverage {ic:.1f}x — cannot service debt from operations"
    elif ic < 1.5:
        ic_s, ic_f = 20.0, f"Interest coverage {ic:.1f}x — dangerously low"
    elif ic < 2.5:
        ic_s, ic_f = 38.0, f"Interest coverage {ic:.1f}x — tight"
    elif ic < 4.0:
        ic_s, ic_f = 58.0, f"Interest coverage {ic:.1f}x — adequate"
    elif ic < 8.0:
        ic_s, ic_f = 78.0, f"Interest coverage {ic:.1f}x — comfortable"
    else:
        ic_s, ic_f = 95.0, f"Interest coverage {ic:.1f}x — very strong"
    sub_scores.append((ic_s, 0.25))
    factors.append(ic_f)

    # ── Cash position ──────────────────────────────────────────────────────────
    if balance and balance.cash_and_equivalents and balance.total_assets:
        cash_pct = balance.cash_and_equivalents / balance.total_assets
        if cash_pct > 0.20:
            sub_scores.append((88.0, 0.10))
            factors.append(f"Cash = {cash_pct*100:.1f}% of assets — substantial war chest")
        elif cash_pct > 0.10:
            sub_scores.append((68.0, 0.10))
            factors.append(f"Cash = {cash_pct*100:.1f}% of assets — adequate")
        else:
            sub_scores.append((45.0, 0.10))
            factors.append(f"Cash = {cash_pct*100:.1f}% of assets — limited buffer")
    else:
        sub_scores.append((50.0, 0.10))

    # ── FCF positivity ─────────────────────────────────────────────────────────
    cf = stock_data.latest_cashflow
    if cf and cf.free_cash_flow is not None:
        if cf.free_cash_flow > 0:
            sub_scores.append((85.0, 0.10))
            factors.append(f"Positive FCF (${cf.free_cash_flow/1e9:.2f}B)")
        else:
            sub_scores.append((20.0, 0.10))
            factors.append(f"Negative FCF (${cf.free_cash_flow/1e9:.2f}B) — burns cash")
    else:
        sub_scores.append((50.0, 0.10))

    total_w   = sum(w for _, w in sub_scores)
    composite = sum(s * w for s, w in sub_scores) / total_w

    available    = sum(1 for s, _ in sub_scores if s != 50.0)
    data_quality = (
        "good"    if available >= 3 else
        "partial" if available >= 1 else
        "missing"
    )

    # ── Metric-referencing reasoning ──────────────────────────────────────────
    # Build a concrete summary from the actual values computed above so the
    # reasoning line is specific rather than generic.
    _parts: list[str] = []
    if de_ratio is not None:
        _parts.append(f"D/E {de_ratio:.2f}x")
    if cr is not None:
        _parts.append(f"current ratio {cr:.2f}x")
    if ic is not None:
        _parts.append(f"interest coverage {ic:.1f}x")
    _metric_str = ", ".join(_parts) if _parts else "balance sheet metrics"

    if composite >= 80:
        reasoning = f"Fortress balance sheet: {_metric_str} — low leverage with ample liquidity."
    elif composite >= 60:
        reasoning = f"Solid financial health: {_metric_str} — manageable leverage and adequate coverage."
    elif composite >= 40:
        reasoning = f"Mixed balance sheet: {_metric_str} — some concerns warrant monitoring."
    else:
        reasoning = f"Balance sheet under stress: {_metric_str} — high leverage or weak liquidity."

    return CategoryScore(
        name="financial_health",
        score=clamp(composite),
        weight=weight,
        factors=factors,
        reasoning=reasoning,
        data_quality=data_quality,
    )
