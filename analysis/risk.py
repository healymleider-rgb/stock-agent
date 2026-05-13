"""
Risk scoring module — multi-factor model.

Risk score: 100 = minimal risk; 0 = extreme risk.  Higher = safer.

Design
------
Five sub-components, each scored 0-100 with realistic defaults:
  1. Financial Risk      30%  — leverage, coverage, liquidity, FCF
  2. Business Model Risk 20%  — margin stability, revenue volatility
  3. Growth Risk         20%  — valuation/growth alignment, EPS consistency
  4. Market Risk         20%  — beta, drawdown history
  5. Qualitative Risk    10%  — sector, capex intensity, goodwill

  composite = weighted blend of sub-scores

Guardrails
----------
  Max:   93 (no equity perfectly risk-free; capped further when flags present)
  Floor: 15 (even distressed companies have residual value)
  Realistic well-run large-cap range: 72-87
"""
from __future__ import annotations

import math
from typing import Optional, TYPE_CHECKING

from models.scorecard import CategoryScore
from models.stock_data import StockData
from utils.helpers import clamp

if TYPE_CHECKING:
    from analysis.metrics import NormalizedMetrics
    from analysis.data_integrity import ValidationResult


_SECTOR_BASE_RISK: dict = {
    "utilities":               80,
    "consumer staples":        77,
    "healthcare":              73,
    "real estate":             67,
    "information technology":  65,
    "communication services":  63,
    "industrials":             62,
    "consumer discretionary":  60,
    "financials":              58,
    "materials":               56,
    "energy":                  50,
}
_SECTOR_BASE_DEFAULT = 63


def _safe_mean(vals: list) -> float:
    clean = [v for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else 0.0


def _safe_std(vals: list) -> float:
    clean = [v for v in vals if v is not None]
    if len(clean) < 2:
        return 0.0
    mu  = sum(clean) / len(clean)
    var = sum((v - mu) ** 2 for v in clean) / len(clean)
    return math.sqrt(var)


def _compute_max_drawdown(closes: list) -> Optional[float]:
    if not closes or len(closes) < 30:
        return None
    prices = list(reversed(closes))
    peak   = prices[0]
    worst  = 0.0
    for p in prices:
        if p > peak:
            peak = p
        dd = (p - peak) / peak
        if dd < worst:
            worst = dd
    return worst


# ── Sub-scorers ───────────────────────────────────────────────────────────────

def _score_financial_risk(de, ic, cr, fcf, rev):
    factors = []
    parts   = []

    if de is None:
        de_s = 68
        factors.append("Debt/Equity: not available (moderate risk assumed)")
    elif de <= 0.0:
        de_s = 93
        factors.append("Net cash / zero debt — fortress balance sheet")
    elif de <= 0.3:
        de_s = 87
        factors.append(f"Very low leverage (D/E {de:.2f}x)")
    elif de <= 0.8:
        de_s = 79
        factors.append(f"Low leverage (D/E {de:.2f}x)")
    elif de <= 1.5:
        de_s = 68
        factors.append(f"Moderate leverage (D/E {de:.2f}x)")
    elif de <= 2.5:
        de_s = 50
        factors.append(f"[RISK] High leverage (D/E {de:.2f}x) — elevated debt burden")
    elif de <= 4.0:
        de_s = 33
        factors.append(f"[RISK] Very high leverage (D/E {de:.2f}x) — significant debt risk")
    else:
        de_s = 16
        factors.append(f"[RISK] Extreme leverage (D/E {de:.2f}x) — financial distress risk")
    parts.append((de_s, 0.40))

    if ic is None:
        ic_s = 65
    elif ic <= 0:
        ic_s = 10
        factors.append(f"[RISK] Negative interest coverage — cannot service debt")
    elif ic < 1.5:
        ic_s = 22
        factors.append(f"[RISK] Interest coverage {ic:.1f}x — barely covers debt service")
    elif ic < 3.0:
        ic_s = 48
        factors.append(f"Interest coverage {ic:.1f}x — adequate but limited buffer")
    elif ic < 6.0:
        ic_s = 68
    elif ic < 12.0:
        ic_s = 83
        factors.append(f"Strong interest coverage ({ic:.1f}x)")
    else:
        ic_s = 93
        factors.append(f"Very strong interest coverage ({ic:.1f}x)")
    parts.append((ic_s, 0.30))

    if cr is None:
        cr_s = 63
    elif cr < 0.8:
        cr_s = 28
        factors.append(f"[RISK] Current ratio {cr:.2f}x — liquidity concern")
    elif cr < 1.0:
        cr_s = 45
        factors.append(f"Current ratio {cr:.2f}x — below 1")
    elif cr < 1.5:
        cr_s = 62
    elif cr < 2.5:
        cr_s = 80
    else:
        cr_s = 88
    parts.append((cr_s, 0.20))

    if fcf is None:
        fcf_s = 60
    elif fcf < 0:
        if rev and rev > 0:
            fcf_margin = fcf / rev
            if fcf_margin < -0.10:
                fcf_s = 20
                factors.append(f"[RISK] Negative FCF margin ({fcf_margin*100:.1f}%) — heavy cash burn")
            else:
                fcf_s = 38
                factors.append("Slightly negative FCF — monitor cash burn")
        else:
            fcf_s = 35
            factors.append("[RISK] Negative free cash flow")
    elif rev and rev > 0:
        fcf_margin = fcf / rev
        fcf_s = 93 if fcf_margin > 0.15 else 82 if fcf_margin > 0.08 else 68 if fcf_margin > 0.03 else 55
    else:
        fcf_s = 70
    parts.append((fcf_s, 0.10))

    total_w = sum(w for _, w in parts)
    return sum(s * w for s, w in parts) / total_w, factors


def _score_business_model_risk(stock_data: StockData):
    factors = []
    inc     = stock_data.income_statements

    gm_series = [s.gross_profit_ratio for s in inc[:5]
                 if getattr(s, "gross_profit_ratio", None) is not None]
    if len(gm_series) >= 3:
        gm_std  = _safe_std(gm_series)
        gm_mean = _safe_mean(gm_series)
        if gm_std < 0.02:
            gm_s = 90
            factors.append(f"Very stable gross margins ({gm_mean*100:.1f}% avg)")
        elif gm_std < 0.05:
            gm_s = 75
        elif gm_std < 0.10:
            gm_s = 55
            factors.append(f"Margin variability (σ={gm_std*100:.1f}pp YoY)")
        else:
            gm_s = 35
            factors.append(f"[RISK] High margin volatility (σ={gm_std*100:.1f}pp)")
    else:
        gm_s = 62

    revs = [s.revenue for s in inc[:5] if getattr(s, "revenue", None) and s.revenue > 0]
    if len(revs) >= 3:
        growths = [(revs[i] - revs[i+1]) / revs[i+1] for i in range(len(revs)-1)]
        rev_vol = _safe_std(growths)
        if rev_vol < 0.05:
            rev_s = 88
        elif rev_vol < 0.12:
            rev_s = 73
        elif rev_vol < 0.25:
            rev_s = 55
        else:
            rev_s = 33
            factors.append(f"[RISK] High revenue volatility (σ={rev_vol*100:.0f}pp YoY growth)")
    else:
        rev_s = 62

    om_series = [s.operating_income_ratio for s in inc[:3]
                 if getattr(s, "operating_income_ratio", None) is not None]
    if om_series:
        avg_om = _safe_mean(om_series)
        if avg_om > 0.25:
            op_s = 88
            factors.append(f"Strong operating margins ({avg_om*100:.1f}%)")
        elif avg_om > 0.15:
            op_s = 77
        elif avg_om > 0.05:
            op_s = 62
        elif avg_om > 0.0:
            op_s = 45
        else:
            op_s = 25
            factors.append(f"[RISK] Negative operating margin ({avg_om*100:.1f}%) — operating losses")
    else:
        op_s = 58

    return gm_s * 0.35 + rev_s * 0.35 + op_s * 0.30, factors


def _score_growth_risk(pe, eps_growth_pct, stock_data: StockData):
    factors = []
    inc     = stock_data.income_statements

    if pe is None:
        pg_s = 60
    elif pe < 0:
        pg_s = 35
        factors.append("[RISK] Negative P/E — company not yet profitable")
    elif pe > 100:
        pg_s = 28
        factors.append(f"[RISK] Extreme P/E ({pe:.0f}x) — priced for perfection")
    elif pe > 60:
        pg_s = 52 if (eps_growth_pct and eps_growth_pct > 25) else 33
        if pg_s == 33:
            factors.append(f"[RISK] High P/E ({pe:.0f}x) with limited growth support")
    elif pe > 35:
        pg_s = 65 if (eps_growth_pct and eps_growth_pct > 12) else 50
    else:
        pg_s = 80

    revs     = [s.revenue     for s in inc[:3] if getattr(s, "revenue",     None) and s.revenue > 0]
    eps_list = [s.eps_diluted for s in inc[:3] if getattr(s, "eps_diluted", None) is not None]

    rev_growth_yoy = None
    if len(revs) >= 2:
        rev_growth_yoy = (revs[0] - revs[1]) / abs(revs[1])

    div_s = 68
    if eps_growth_pct is not None and rev_growth_yoy is not None:
        divergence = abs(eps_growth_pct / 100 - rev_growth_yoy)
        if divergence < 0.10:
            div_s = 83
        elif divergence < 0.20:
            div_s = 68
        elif divergence < 0.35:
            div_s = 48
            factors.append(f"EPS growth ({eps_growth_pct:.0f}%) diverges from revenue ({rev_growth_yoy*100:.0f}%)")
        else:
            div_s = 30
            factors.append(f"[RISK] Large EPS/revenue divergence ({divergence*100:.0f}pp) — earnings quality concern")

    cons_s = 60
    if len(eps_list) >= 3:
        eps_growths = [
            (eps_list[i] - eps_list[i+1]) / abs(eps_list[i+1])
            for i in range(len(eps_list)-1)
            if eps_list[i+1] != 0
        ]
        if eps_growths:
            consist = sum(1 for g in eps_growths if g > 0) / len(eps_growths)
            cons_s = 88 if consist >= 0.85 else 70 if consist >= 0.65 else 48 if consist >= 0.40 else 28
            if cons_s == 28:
                factors.append("[RISK] Inconsistent earnings — EPS declining in majority of periods")

    return pg_s * 0.40 + div_s * 0.30 + cons_s * 0.30, factors


def _score_market_risk(beta, price_history, beta_reliable=True, beta_months=0):
    factors = []

    # Recompute months from live price_history (more accurate than the pre-computed
    # value, which is derived before price history is fully loaded).
    _live_closes  = getattr(price_history, "closes", []) if price_history else []
    _live_months  = int(len(_live_closes) / 21)
    _live_reliable = (
        _live_months >= 24
        and (beta is None or abs(beta) <= 5.0)
    )
    # Use whichever is more informative: if live data is available prefer it,
    # otherwise fall back to the pre-computed flag from NormalizedMetrics.
    _use_reliable = _live_reliable if _live_closes else beta_reliable
    _use_months   = _live_months   if _live_closes else beta_months

    if not _use_reliable:
        beta_s = 65  # neutral — don't penalise on unreliable figure
        _hist_note = f" ({_use_months} months of history)" if _use_months > 0 else ""
        factors.append(
            f"[RISK] Limited trading history — beta unreliable{_hist_note};"
            f" volatility metrics may understate risk"
        )
    elif beta is None:
        beta_s = 65
    elif beta <= 0.3:
        beta_s = 92
        factors.append(f"Very low beta ({beta:.2f})")
    elif beta <= 0.7:
        beta_s = 85
    elif beta <= 1.0:
        beta_s = 75
    elif beta <= 1.3:
        beta_s = 63
    elif beta <= 1.6:
        beta_s = 50
        factors.append(f"Above-average market sensitivity (β {beta:.2f})")
    elif beta <= 2.0:
        beta_s = 37
        factors.append(f"[RISK] High beta ({beta:.2f})")
    elif beta <= 2.5:
        beta_s = 24
        factors.append(f"[RISK] Very high beta ({beta:.2f}) — extremely volatile")
    else:
        beta_s = 13
        factors.append(f"[RISK] Extreme beta ({beta:.2f}) — speculative")

    closes = getattr(price_history, "closes", []) if price_history else []
    if len(closes) >= 60:
        dd = _compute_max_drawdown(closes)
        if dd is not None:
            dd_abs = abs(dd)
            if dd_abs < 0.12:
                dd_s = 90
            elif dd_abs < 0.22:
                dd_s = 75
            elif dd_abs < 0.35:
                dd_s = 58
            elif dd_abs < 0.50:
                dd_s = 38
                factors.append(f"[RISK] Max drawdown {dd_abs*100:.0f}%")
            else:
                dd_s = 20
                factors.append(f"[RISK] Severe max drawdown {dd_abs*100:.0f}%")
        else:
            dd_s = 65
        return beta_s * 0.55 + dd_s * 0.45, factors

    return float(beta_s), factors


def _score_qualitative_risk(stock_data: StockData):
    factors  = []
    profile  = stock_data.profile
    sector   = (profile.sector   or "").lower().strip() if profile else ""
    industry = (profile.industry or "").lower().strip() if profile else ""

    base = _SECTOR_BASE_RISK.get(sector, _SECTOR_BASE_DEFAULT)

    # ── Capex intensity (capital allocation risk) ─────────────────────────────
    cfs = stock_data.latest_cashflow
    inc = stock_data.latest_income
    capex_s = 72
    if (cfs and inc and getattr(cfs, "capital_expenditure", None) and
            getattr(inc, "revenue", None) and inc.revenue > 0):
        capex_ratio = abs(cfs.capital_expenditure) / inc.revenue
        if capex_ratio > 0.20:
            capex_s = 35
            factors.append(f"[RISK] High capex intensity ({capex_ratio*100:.0f}% of revenue) — capital allocation burden")
        elif capex_ratio > 0.12:
            capex_s = 52
            factors.append(f"Above-average capex intensity ({capex_ratio*100:.0f}% of revenue)")
        elif capex_ratio > 0.06:
            capex_s = 75
        else:
            capex_s = 90

    # ── Goodwill / acquisition risk ───────────────────────────────────────────
    bal  = stock_data.latest_balance
    gw_s = 72
    if (bal and getattr(bal, "goodwill", None) and
            getattr(bal, "total_assets", None) and bal.total_assets > 0):
        gw_pct = bal.goodwill / bal.total_assets
        if gw_pct > 0.55:
            gw_s = 35
            factors.append(f"[RISK] Goodwill = {gw_pct*100:.0f}% of assets — impairment risk")
        elif gw_pct > 0.35:
            gw_s = 52
            factors.append(f"Elevated goodwill ({gw_pct*100:.0f}% of assets)")
        elif gw_pct > 0.15:
            gw_s = 78
        else:
            gw_s = 90

    # ── Structural business model risk penalties ──────────────────────────────
    # These are MANDATORY risk penalties for known structurally risky archetypes.
    # Applied as additive deductions to the composite before blending — they
    # represent qualitative risk that financial statements alone cannot capture.
    structural_deduction = 0.0

    # Content / streaming spend risk
    # Companies with large, recurring content obligations have lumpy, hard-to-cut
    # fixed costs that compress margins during subscriber deceleration.
    _content_keywords = (
        "streaming", "media", "entertainment", "content", "film", "television",
        "broadcasting", "publishing", "music", "studio"
    )
    _is_content_heavy = any(kw in industry for kw in _content_keywords) or (
        sector in ("communication services",) and
        any(kw in industry for kw in ("internet", "interactive", "broadcast"))
    )
    if _is_content_heavy:
        structural_deduction += 6.0
        factors.append(
            "[RISK] Content-heavy business: recurring content/licensing obligations "
            "create fixed-cost leverage; margin compression risk during slowdowns"
        )

    # Competitive intensity / commoditisation risk
    # Industries with low switching costs, price competition, or no structural moat
    # are more susceptible to margin erosion from new entrants.
    _high_competition_keywords = (
        "restaurants", "retail", "apparel", "airlines", "trucking",
        "grocery", "discount", "e-commerce", "marketplace", "food delivery",
        "department stores", "specialty retail"
    )
    _is_highly_competitive = any(kw in industry for kw in _high_competition_keywords)
    if _is_highly_competitive:
        structural_deduction += 5.0
        factors.append(
            "[RISK] Highly competitive industry: low switching costs and pricing pressure "
            "limit structural moat; margin defence requires continuous investment"
        )

    # Legacy / dual transformation risk
    # Companies simultaneously facing secular decline in core AND funding a new
    # growth segment — capital allocation is inherently split.
    _legacy_keywords = (
        "cable", "satellite", "print", "newspaper", "traditional media",
        "landline", "fixed-line telecom", "directory"
    )
    _is_legacy_transform = any(kw in industry for kw in _legacy_keywords)
    if _is_legacy_transform:
        structural_deduction += 7.0
        factors.append(
            "[RISK] Legacy transformation risk: secular core decline while funding new "
            "growth segment creates capital allocation tension and margin pressure"
        )

    # ── Single-product clinical-stage biotech ─────────────────────────────────
    # Small healthcare companies with one approved/pipeline product carry
    # concentrated FDA, promotional compliance, and securities litigation risk
    # that financial statements cannot surface.  Flag when net margin < -100%
    # and revenue < $200M in Healthcare/Biotech or Pharmaceuticals.
    _biotech_industries = ("biotechnology", "pharmaceuticals", "drug manufacturers")
    _is_small_biotech = (
        sector == "healthcare"
        and any(kw in industry for kw in _biotech_industries)
    )
    if _is_small_biotech and inc:
        rev = getattr(inc, "revenue", None)
        nm  = getattr(inc, "net_income_ratio", None)
        if nm is None and rev and rev > 0 and getattr(inc, "net_income", None) is not None:
            nm = inc.net_income / rev
        if nm is not None and nm < -1.0 and rev is not None and rev < 200_000_000:
            factors.append(
                "[RISK] Single-product clinical-stage company — review FDA filings, "
                "promotional compliance issues, and pending litigation before "
                "initiating position. StockEval cannot detect these from financial "
                "data alone."
            )

    # ── Composite — max 95, realistic range 60-85 ────────────────────────────
    raw_qual = base * 0.45 + capex_s * 0.30 + gw_s * 0.25 - structural_deduction
    return raw_qual, factors


# ── Public entry point ────────────────────────────────────────────────────────

def score_risk(
    stock_data:  StockData,
    weight:      float = 0.10,
    metrics:     "Optional[NormalizedMetrics]" = None,
    validation:  "Optional[ValidationResult]"  = None,
) -> "tuple[CategoryScore, list[str]]":
    """
    Multi-factor risk score. Returns (CategoryScore, risk_flags).
    """
    ratios   = stock_data.latest_ratios
    balance  = stock_data.latest_balance
    income   = stock_data.latest_income
    cashflow = stock_data.latest_cashflow
    profile  = stock_data.profile

    if metrics is not None:
        market_cap: Optional[float] = metrics.market_cap
        pe:         Optional[float] = metrics.pe_ratio
        de:         Optional[float] = metrics.debt_to_equity
        print(f"  [RISK] from NormalizedMetrics: mktcap={market_cap} pe={pe}({metrics.pe_source}) de={de}")
    else:
        market_cap = stock_data.market_cap
        pe         = ratios.pe_ratio        if ratios else None
        de         = ratios.debt_to_equity  if ratios else None

    ic  = ratios.interest_coverage if ratios else None
    cr  = ratios.current_ratio     if ratios else None
    fcf = getattr(cashflow, "free_cash_flow", None) if cashflow else None
    rev = getattr(income,   "revenue",        None) if income   else None
    eps_growth_pct = getattr(metrics, "eps_growth_pct", None) if metrics else None
    beta           = profile.beta if profile else None
    _beta_reliable = getattr(metrics, "beta_reliable", True)  if metrics else True
    _beta_months   = getattr(metrics, "beta_months",   0)     if metrics else 0

    fin_score,  fin_factors  = _score_financial_risk(de, ic, cr, fcf, rev)
    biz_score,  biz_factors  = _score_business_model_risk(stock_data)
    grw_score,  grw_factors  = _score_growth_risk(pe, eps_growth_pct, stock_data)
    mkt_score,  mkt_factors  = _score_market_risk(
        beta, stock_data.price_history,
        beta_reliable=_beta_reliable, beta_months=_beta_months,
    )
    qual_score, qual_factors = _score_qualitative_risk(stock_data)

    composite = (
        0.30 * fin_score +
        0.20 * biz_score +
        0.20 * grw_score +
        0.20 * mkt_score +
        0.10 * qual_score
    )

    # Size overlay
    size_factors: list = []
    if market_cap is not None:
        if market_cap < 300_000_000:
            composite -= 8.0
            size_factors.append(f"[RISK] Micro-cap (${market_cap/1e6:.0f}M) — liquidity risk")
        elif market_cap < 2_000_000_000:
            composite -= 3.0
            size_factors.append(f"Small-cap (${market_cap/1e9:.2f}B)")

    all_factors = fin_factors + biz_factors + grw_factors + mkt_factors + qual_factors + size_factors
    flags       = [f for f in all_factors if f.startswith("[RISK]")]
    has_major   = any(
        kw in " ".join(flags).lower()
        for kw in ["cannot service", "negative interest", "extreme", "severe", "distress"]
    )

    composite = min(composite, 85 if has_major else 90 if flags else 93)

    # Validation penalty
    if validation is not None and validation.conviction_penalty < 1.0:
        composite -= (1.0 - validation.conviction_penalty) * 8.0

    risk_score = clamp(composite, 15.0, 97.0)

    flags_clean = [f.replace("[RISK] ", "") for f in flags]

    if not flags:
        reasoning = (
            f"No major risk flags — composite {risk_score:.0f}/100 "
            f"(fin:{fin_score:.0f} biz:{biz_score:.0f} grw:{grw_score:.0f} "
            f"mkt:{mkt_score:.0f} qual:{qual_score:.0f})."
        )
    elif len(flags) <= 2:
        reasoning = f"{len(flags)} risk flag(s) — composite {risk_score:.0f}/100."
    else:
        reasoning = f"{len(flags)} risk flags — meaningful headwinds. Composite: {risk_score:.0f}/100."

    print(
        f"  [RISK] fin={fin_score:.1f} biz={biz_score:.1f}"
        f" grw={grw_score:.1f} mkt={mkt_score:.1f}"
        f" qual={qual_score:.1f} → {risk_score:.1f} flags={len(flags)}"
    )

    full_factors = [
        f"Financial risk:      {fin_score:.0f}/100",
        f"Business model risk: {biz_score:.0f}/100",
        f"Growth risk:         {grw_score:.0f}/100",
        f"Market risk:         {mkt_score:.0f}/100",
        f"Qualitative risk:    {qual_score:.0f}/100",
    ] + [f for f in all_factors if not f.startswith("[RISK]")]

    data_quality = (
        "good"    if (ratios and balance and income) else
        "partial" if (ratios or balance) else
        "missing"
    )

    return (
        CategoryScore(
            name=        "risk",
            score=       risk_score,
            weight=      weight,
            factors=     full_factors,
            reasoning=   reasoning,
            data_quality=data_quality,
        ),
        flags_clean,
    )
