"""
Valuation scoring module.

Inputs : NormalizedMetrics (preferred) + StockData fallback
Output : CategoryScore (0–100, higher = more attractive valuation)

When NormalizedMetrics is provided, it is used exclusively for pe_ratio,
ps_ratio, ev_to_ebitda, and market_cap.  This ensures the scorecard and
the displayed report numbers are always consistent — both come from the
same validated source.

Thresholds are intentionally visible so analysts can tune them.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from models.scorecard import CategoryScore
from models.stock_data import StockData
from utils.helpers import clamp

if TYPE_CHECKING:
    from analysis.metrics import NormalizedMetrics


# ── Per-metric thresholds (upper_bound, score) ─────────────────────────────────

_PE_THRESHOLDS    = [(8, 98), (12, 90), (18, 80), (25, 68), (35, 52), (50, 38), (75, 25), (999, 15)]
_EV_THRESHOLDS    = [(6, 95), (10, 82), (15, 68), (20, 55), (30, 38), (50, 20), (999, 12)]
_PS_THRESHOLDS    = [(0.5, 95), (1.5, 82), (3, 68), (5, 55), (8, 40), (15, 25), (999, 12)]
_FCF_THRESHOLDS   = [(0.01, 35), (0.03, 50), (0.05, 65), (0.08, 80), (0.12, 92), (0.99, 95)]
_PB_THRESHOLDS    = [(1, 90), (2, 78), (4, 62), (8, 45), (15, 28), (999, 15)]

# PEG thresholds — higher score = cheaper relative to growth
# PEG < 1.0 typically considered attractive; > 2.0 expensive
_PEG_THRESHOLDS   = [(0.5, 92), (0.75, 82), (1.0, 70), (1.25, 60), (1.5, 50), (2.0, 36), (2.5, 24), (999, 12)]

# PEG sub-score weight caps
_PEG_WEIGHT_NORMAL  = 0.15   # max 15% of valuation score
_PEG_WEIGHT_FLAGGED = 0.05   # capped at 5% when LOW_GROWTH_QUALITY detected


def _score_from_thresholds(value: float, thresholds: list[tuple]) -> float:
    sorted_t = sorted(thresholds, key=lambda x: x[0])
    for bound, score in sorted_t:
        if value <= bound:
            return float(score)
    return float(sorted_t[-1][1])


def _pe_score(pe: Optional[float]) -> tuple[float, str]:
    if pe is None:
        return 50.0, "P/E not available"
    if pe < 0:
        return 40.0, f"Negative P/E ({pe:.1f}) — company unprofitable"
    score = _score_from_thresholds(pe, _PE_THRESHOLDS)
    label = "cheap" if score >= 75 else "fair" if score >= 55 else "expensive"
    return score, f"P/E {pe:.1f}x — {label}"


def _ev_ebitda_score(ev_ebitda: Optional[float]) -> tuple[float, str]:
    if ev_ebitda is None:
        return 50.0, "EV/EBITDA not available"
    if ev_ebitda < 0:
        return 35.0, f"Negative EV/EBITDA ({ev_ebitda:.1f})"
    score = _score_from_thresholds(ev_ebitda, _EV_THRESHOLDS)
    label = "cheap" if score >= 75 else "fair" if score >= 55 else "expensive"
    return score, f"EV/EBITDA {ev_ebitda:.1f}x — {label}"


def _ps_score(ps: Optional[float]) -> tuple[float, str]:
    if ps is None:
        return 50.0, "P/S not available"
    score = _score_from_thresholds(ps, _PS_THRESHOLDS)
    label = "cheap" if score >= 75 else "fair" if score >= 55 else "expensive"
    return score, f"P/S {ps:.1f}x — {label}"


def _fcf_yield_score(fcf_yield: Optional[float]) -> tuple[float, str]:
    if fcf_yield is None:
        return 50.0, "FCF yield not available"
    pct = fcf_yield * 100
    if fcf_yield < 0:
        return 20.0, f"Negative FCF yield ({pct:.1f}%) — no free cash generation"
    score = _score_from_thresholds(fcf_yield, _FCF_THRESHOLDS)
    return score, f"FCF yield {pct:.1f}%"


def _pb_score(pb: Optional[float]) -> tuple[float, str]:
    if pb is None:
        return 50.0, "P/B not available"
    score = _score_from_thresholds(pb, _PB_THRESHOLDS)
    return score, f"P/B {pb:.1f}x"


def _peg_score(peg: Optional[float]) -> tuple[float, str]:
    """Score PEG ratio. Returns (score 0-100, factor string)."""
    if peg is None:
        return 50.0, "PEG not available"
    if peg < 0:
        return 30.0, f"PEG {peg:.2f} — negative (unprofitable or declining earnings)"
    score = _score_from_thresholds(peg, _PEG_THRESHOLDS)
    label = "attractive" if score >= 70 else "fair" if score >= 50 else "expensive"
    return score, f"PEG {peg:.2f}x — {label} relative to growth"


def _earnings_leverage_check(
    metrics: "Optional[NormalizedMetrics]",
    stock_data: StockData,
) -> tuple:
    """
    Check whether EPS growth is earnings-leverage-driven rather than organic.

    Condition: EPS CAGR > 30% AND 3Y revenue CAGR < 20%.

    Returns (is_leverage_driven: bool, eps_cagr_pct: float|None,
             rev_cagr_pct: float|None, detail: str).

    Uses 3Y CAGRs for both (apples-to-apples multi-year comparison).
    Returns (False, None, None, "") when data is insufficient to classify.
    """
    # EPS CAGR — requires NormalizedMetrics 3Y CAGR
    eps_cagr_pct: Optional[float] = None
    if metrics is not None:
        eps_cagr_pct = metrics.eps_growth_pct   # annualized %, e.g. 35.0

    # 3Y Revenue CAGR from income statements
    rev_cagr_pct: Optional[float] = None
    inc = stock_data.income_statements
    if len(inc) >= 4 and inc[0].revenue and inc[3].revenue and inc[3].revenue > 0:
        rev_cagr_pct = ((inc[0].revenue / inc[3].revenue) ** (1 / 3) - 1) * 100

    if eps_cagr_pct is None or rev_cagr_pct is None:
        return False, eps_cagr_pct, rev_cagr_pct, ""

    is_leverage = eps_cagr_pct > 30.0 and rev_cagr_pct < 20.0
    if is_leverage:
        gap    = eps_cagr_pct - rev_cagr_pct
        detail = (
            f"[LOW_GROWTH_QUALITY] EPS CAGR ({eps_cagr_pct:.1f}%) far exceeds "
            f"revenue CAGR ({rev_cagr_pct:.1f}%) by {gap:.1f}pp — "
            f"PEG driven by margin expansion, not pure growth. "
            f"PEG weight reduced to {_PEG_WEIGHT_FLAGGED*100:.0f}% of valuation score."
        )
    else:
        detail = ""
    return is_leverage, eps_cagr_pct, rev_cagr_pct, detail


def score_valuation(
    stock_data: StockData,
    weight: float = 0.20,
    metrics: "Optional[NormalizedMetrics]" = None,
) -> CategoryScore:
    """
    Compute a 0–100 valuation attractiveness score.
    100 = very cheap; 0 = extreme bubble valuation.

    When *metrics* is provided its validated pe_ratio, ps_ratio, ev_ebitda,
    and market_cap are used instead of raw StockData fields.  This guarantees
    the scorecard is computed from the same values shown in the report header.
    """
    factors: list[str] = []
    sub_scores: list[tuple[float, float]] = []

    # ── Resolve valuation multiples ───────────────────────────────────────────
    if metrics is not None:
        # Use the single validated source of truth
        pe        = metrics.pe_ratio
        ps        = metrics.ps_ratio
        ev_ebitda = metrics.ev_ebitda
        mkt_cap   = metrics.market_cap
        price     = metrics.price
        print(
            f"  [FUND VAL] from NormalizedMetrics:"
            f" pe={pe}({metrics.pe_source})"
            f" ps={ps}({metrics.ps_source})"
            f" ev_ebitda={ev_ebitda}({metrics.ev_ebitda_source})"
        )
    else:
        # Legacy path: read from raw StockData (used when metrics is not passed)
        ratios    = stock_data.latest_ratios
        income    = stock_data.latest_income
        balance   = stock_data.latest_balance
        price     = stock_data.current_price
        mkt_cap   = stock_data.market_cap
        pe        = ratios.pe_ratio if ratios else None
        ps        = ratios.ps_ratio if ratios else None
        ev_ebitda = ratios.ev_to_ebitda if ratios else None

        # Derive missing multiples from raw statements
        shares = (mkt_cap / price) if (mkt_cap and price and price > 0) else None
        if pe is None and price and income:
            _eps = income.eps_diluted or income.eps
            if _eps is None and income.net_income and shares and shares > 0:
                _eps = income.net_income / shares
            if _eps and _eps > 0:
                pe = round(price / _eps, 2)
                print(f"  [FUND VAL] derived P/E={pe:.2f} (price={price}, eps={_eps:.2f})")
        if ps is None and mkt_cap and income and income.revenue and income.revenue > 0:
            ps = round(mkt_cap / income.revenue, 2)
            print(f"  [FUND VAL] derived P/S={ps:.2f}")
        if ev_ebitda is None and mkt_cap and income and income.ebitda and income.ebitda > 0:
            _debt = (balance.total_debt or 0.0) if balance else 0.0
            _cash = (balance.cash_and_equivalents or 0.0) if balance else 0.0
            _ev = mkt_cap + _debt - _cash
            if _ev > 0:
                ev_ebitda = round(_ev / income.ebitda, 2)
                print(f"  [FUND VAL] derived EV/EBITDA={ev_ebitda:.2f}")

    print(
        f"  [FUND VAL] scoring: PE={pe} PS={ps} EV/EBITDA={ev_ebitda}"
    )

    # FCF yield and P/B come from ratios only (no computed alternative)
    ratios_obj = stock_data.latest_ratios
    fcf_yield  = ratios_obj.fcf_yield if ratios_obj else None
    pb         = ratios_obj.pb_ratio  if ratios_obj else None
    # Derive P/B from metrics if P/B is missing from ratios
    if pb is None and metrics and metrics.pb_ratio is not None:
        pb = metrics.pb_ratio

    pe_s, pe_f = _pe_score(pe)
    ev_s, ev_f = _ev_ebitda_score(ev_ebitda)
    ps_s, ps_f = _ps_score(ps)
    fcf_s, fcf_f = _fcf_yield_score(fcf_yield)
    pb_s, pb_f = _pb_score(pb)

    # ── PEG: earnings leverage check + dynamic weight ─────────────────────────
    # PEG is scored as a proper sub-component (not a post-hoc composite nudge).
    # Weight: up to 15% normally; capped at 5% when growth is leverage-driven.
    # When PEG is included, the five base weights scale down proportionally so
    # total weight always sums to 1.0.
    peg = metrics.peg if metrics is not None else None

    _leverage_driven, _eps_cagr, _rev_cagr, _leverage_detail = _earnings_leverage_check(
        metrics, stock_data
    )
    peg_s, peg_f = _peg_score(peg)

    if peg is not None and peg >= 0:
        peg_w  = _PEG_WEIGHT_FLAGGED if _leverage_driven else _PEG_WEIGHT_NORMAL
        base_w = 1.0 - peg_w          # remaining weight distributed to base five
        # Base five raw weights (PE=30, EV=25, PS=20, FCF=20, PB=5 → sum=100)
        _base_raw = [0.30, 0.25, 0.20, 0.20, 0.05]
        _scaled   = [w * base_w for w in _base_raw]
        sub_scores = [
            (pe_s,  _scaled[0]),
            (ev_s,  _scaled[1]),
            (ps_s,  _scaled[2]),
            (fcf_s, _scaled[3]),
            (pb_s,  _scaled[4]),
            (peg_s, peg_w),
        ]
        print(
            f"  [FUND VAL] PEG included: peg={peg:.2f} score={peg_s:.0f}"
            f" weight={peg_w:.0%} leverage_driven={_leverage_driven}"
        )
    else:
        sub_scores = [
            (pe_s,  0.30),
            (ev_s,  0.25),
            (ps_s,  0.20),
            (fcf_s, 0.20),
            (pb_s,  0.05),
        ]
        peg_w = 0.0

    factors.append(pe_f)
    factors.append(ev_f)
    factors.append(ps_f)
    factors.append(fcf_f)
    factors.append(pb_f)
    if peg_w > 0:
        factors.append(peg_f)

    # Growth quality output line (always emitted when classification is available)
    if _eps_cagr is not None and _rev_cagr is not None:
        _gq_label = "earnings-leverage-driven" if _leverage_driven else (
            "revenue-driven" if _rev_cagr >= 20.0 else "mixed"
        )
        factors.append(
            f"Growth quality: {_gq_label} — "
            f"EPS CAGR {_eps_cagr:.1f}% vs revenue CAGR {_rev_cagr:.1f}%"
        )
    if _leverage_detail:
        factors.append(_leverage_detail)

    total_w   = sum(w for _, w in sub_scores)
    composite = sum(s * w for s, w in sub_scores) / total_w

    available = [s for s, _ in sub_scores if s != 50.0]
    data_quality = "good" if len(available) >= 3 else "partial" if available else "missing"

    # ── PEG vs P/E tension narrative (informational only — no composite tweak) ─
    # The composite already incorporates PEG through the weighted blend above.
    # This block adds a human-readable note when the two metrics point in
    # opposite directions — useful for the analyst but doesn't double-count.
    peg_tension = ""
    if peg is not None and peg >= 0 and pe is not None and pe > 0:
        if pe_s < 48 and peg < 1.5:
            peg_tension = (
                f"Headline P/E ({pe:.1f}x) looks elevated, but PEG {peg:.2f}x"
                f" suggests valuation is reasonable given the growth rate"
                + (" — note: PEG weight reduced (earnings leverage detected)" if _leverage_driven else ".")
            )
        elif pe_s > 65 and peg > 2.5:
            peg_tension = (
                f"P/E ({pe:.1f}x) appears inexpensive in isolation, but PEG {peg:.2f}x"
                " indicates the stock is expensive relative to its growth rate"
                " — headline cheapness may be misleading."
            )
    if peg_tension:
        factors.append(peg_tension)
        print(f"  [FUND VAL] PEG tension noted: {peg_tension[:80]}...")

    # ── Reasoning narrative (metric-referencing) ─────────────────────────────
    if _leverage_driven and peg_tension:
        reasoning = (
            f"Valuation reflects earnings leverage: {peg_tension.split(' —')[0]}. "
            f"PEG weight reduced — growth is margin-driven, not organic."
        )
    elif peg_tension:
        reasoning = peg_tension
    elif composite >= 75:
        parts: list[str] = []
        if pe is not None:
            parts.append(f"P/E {pe:.1f}x")
        if ev_ebitda is not None:
            parts.append(f"EV/EBITDA {ev_ebitda:.1f}x")
        if ps is not None:
            parts.append(f"P/S {ps:.1f}x")
        metric_list = " / ".join(parts) if parts else "multiple metrics"
        reasoning = f"Attractive valuation: {metric_list} all screen cheaply."
    elif composite >= 55:
        parts = []
        if pe is not None:
            parts.append(f"P/E {pe:.1f}x")
        if ps is not None:
            parts.append(f"P/S {ps:.1f}x")
        metric_list = " and ".join(parts) if parts else "current multiples"
        reasoning = f"Fair valuation at {metric_list} — not cheap, not stretched."
    else:
        parts = []
        if pe is not None and pe_s < 50:
            parts.append(f"P/E {pe:.1f}x")
        if ev_ebitda is not None and ev_s < 50:
            parts.append(f"EV/EBITDA {ev_ebitda:.1f}x")
        if ps is not None and ps_s < 50:
            parts.append(f"P/S {ps:.1f}x")
        stretched = " / ".join(parts) if parts else "multiples"
        reasoning = f"Valuation is stretched: {stretched} screen expensive relative to peers and history."

    return CategoryScore(
        name="valuation",
        score=clamp(composite),
        weight=weight,
        factors=factors,
        reasoning=reasoning,
        data_quality=data_quality,
    )
