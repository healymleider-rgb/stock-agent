"""
Growth scoring module.

Measures revenue growth, EPS growth, and FCF growth
using year-over-year changes from annual statements.

When NormalizedMetrics is supplied, the EPS growth sub-score is sourced
from metrics.eps_growth_pct (the same annualized CAGR used everywhere else
in the system).  Revenue and FCF growth remain from raw statements because
those series are not carried in NormalizedMetrics.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from models.scorecard import CategoryScore
from models.stock_data import StockData
from utils.helpers import clamp, pct_change

if TYPE_CHECKING:
    from analysis.metrics import NormalizedMetrics


def _growth_score(rate: float | None, label: str) -> tuple[float, str]:
    """Convert a YoY growth rate to a 0-100 score."""
    if rate is None:
        return 50.0, f"{label}: N/A"
    pct = rate * 100
    if rate > 0.30:
        score = 95.0
    elif rate > 0.20:
        score = 85.0
    elif rate > 0.10:
        score = 73.0
    elif rate > 0.05:
        score = 60.0
    elif rate > 0.0:
        score = 48.0
    elif rate > -0.05:
        score = 35.0
    elif rate > -0.15:
        score = 22.0
    else:
        score = 10.0
    trend = (
        "strong"   if score >= 80 else
        "solid"    if score >= 60 else
        "weak"     if score >= 35 else
        "declining"
    )
    return score, f"{label}: {pct:+.1f}% ({trend})"


def detect_eps_one_time_inflation(
    stock_data: "StockData",
    metrics: "NormalizedMetrics",
) -> tuple:
    """
    Detect likely one-time EPS inflation from 4 signals; fires when 2+ trigger.

    Returns (fired: bool, signal_ids: list[str]).

    One-time items (asset sales, tax windfalls, reserve releases, litigation
    settlements) inflate EPS without corresponding revenue or operating income
    growth.  When 2+ signals fire simultaneously, the eps_growth_pct is capped
    at min(eps_pct, rev_cagr_3y * 1.5) before growth scoring.
    """
    inc = stock_data.income_statements
    signals: list[str] = []

    eps_cagr_pct: Optional[float] = metrics.eps_growth_pct   # 3Y CAGR, annualised %

    # Signal 1 — single-year EPS jump >50% with revenue <20% and gap >30pp
    if len(inc) >= 2:
        rev_0, rev_1 = inc[0].revenue, inc[1].revenue
        eps_0, eps_1 = inc[0].eps_diluted, inc[1].eps_diluted
        if (rev_0 and rev_1 and rev_1 > 0 and eps_0 and eps_1 and eps_1 > 0):
            rev_yoy = (rev_0 - rev_1) / abs(rev_1) * 100
            eps_yoy = (eps_0 - eps_1) / abs(eps_1) * 100
            if eps_yoy > 50 and rev_yoy < 20 and (eps_yoy - rev_yoy) > 30:
                signals.append(
                    f"S1(eps_yoy={eps_yoy:.0f}%,rev_yoy={rev_yoy:.0f}%,"
                    f"gap={eps_yoy-rev_yoy:.0f}pp)"
                )

    # Signal 2 — 3Y EPS CAGR >30% with 3Y rev CAGR <15% and gap >15pp
    rev_cagr_3y_pct: Optional[float] = None
    if len(inc) >= 4 and inc[0].revenue and inc[3].revenue and inc[3].revenue > 0:
        rev_cagr_3y_pct = ((inc[0].revenue / inc[3].revenue) ** (1 / 3) - 1) * 100
    if (eps_cagr_pct is not None and rev_cagr_3y_pct is not None
            and eps_cagr_pct > 30 and rev_cagr_3y_pct < 15
            and (eps_cagr_pct - rev_cagr_3y_pct) > 15):
        signals.append(
            f"S2(eps_cagr={eps_cagr_pct:.0f}%,rev_cagr={rev_cagr_3y_pct:.0f}%,"
            f"gap={eps_cagr_pct-rev_cagr_3y_pct:.0f}pp)"
        )

    # Signal 3 — 3Y net income growth >50% but operating income growth <25%
    if len(inc) >= 4:
        ni_0  = inc[0].net_income
        ni_3  = inc[3].net_income
        op_0  = inc[0].operating_income
        op_3  = inc[3].operating_income
        if (ni_0 and ni_3 and ni_3 > 0 and op_0 and op_3 and op_3 > 0):
            ni_3yr_pct = (ni_0 - ni_3) / abs(ni_3) * 100
            op_3yr_pct = (op_0 - op_3) / abs(op_3) * 100
            if ni_3yr_pct > 50 and op_3yr_pct < 25:
                signals.append(
                    f"S3(ni_3yr={ni_3yr_pct:.0f}%,op_3yr={op_3yr_pct:.0f}%)"
                )

    # Signal 4 — net margin barely changed (<200 bps) but net income grew >500%
    # (massive NI jump with stable margins implies a large one-time item)
    if len(inc) >= 4:
        nm_0  = inc[0].net_income_ratio
        nm_3  = inc[3].net_income_ratio
        ni_0  = inc[0].net_income
        ni_3  = inc[3].net_income
        if (nm_0 is not None and nm_3 is not None
                and ni_0 is not None and ni_3 is not None and ni_3 > 0):
            nm_change_bps = abs(nm_0 - nm_3) * 10_000
            ni_change_pct = (ni_0 - ni_3) / abs(ni_3) * 100
            if nm_change_bps < 200 and ni_change_pct > 500:
                signals.append(
                    f"S4(nm_change={nm_change_bps:.0f}bps,ni_change={ni_change_pct:.0f}%)"
                )

    return len(signals) >= 2, signals


def _classify_growth_quality(
    eps_cagr_pct: float,
    rev_cagr_rate: float,
) -> tuple:
    """
    Classify growth quality based on EPS CAGR vs revenue CAGR alignment.

    Returns (is_leverage_driven: bool, label: str, detail: str).

    Earnings-leverage-driven growth (EPS CAGR > 30% while revenue CAGR < 20%)
    is less durable than organic revenue-led growth — it relies on margin
    expansion, cost cuts, buybacks, or tax benefits that can reverse.
    """
    is_leverage = eps_cagr_pct > 30.0 and (rev_cagr_rate * 100) < 20.0
    if is_leverage:
        gap = eps_cagr_pct - (rev_cagr_rate * 100)
        label  = "earnings-leverage-driven"
        detail = (
            f"EPS growth ({eps_cagr_pct:.1f}% CAGR) significantly outpaces "
            f"revenue growth ({rev_cagr_rate*100:.1f}% CAGR) by {gap:.1f}pp — "
            f"driven by margin expansion, cost reduction, or financial engineering "
            f"rather than organic volume growth. Less durable; PEG weight reduced."
        )
    else:
        label  = "revenue-driven" if rev_cagr_rate >= 0.10 else "mixed"
        detail = (
            f"EPS growth ({eps_cagr_pct:.1f}% CAGR) is broadly supported by "
            f"revenue growth ({rev_cagr_rate*100:.1f}% CAGR) — organic and durable."
        ) if not is_leverage else ""
    return is_leverage, label, detail


def score_growth(
    stock_data: StockData,
    weight: float = 0.20,
    metrics: "Optional[NormalizedMetrics]" = None,
) -> CategoryScore:
    """
    Compute a 0-100 growth quality score.
    100 = exceptional growth across all dimensions.

    When *metrics* is provided, EPS growth uses metrics.eps_growth_pct
    (the system-wide annualized CAGR) so the scorecard and all other
    report sections are consistent.
    """
    inc = stock_data.income_statements
    cfs = stock_data.cash_flows
    factors: list[str] = []
    sub_scores: list[tuple[float, float]] = []

    # ── Revenue growth (most recent YoY) ──────────────────────────────────────
    rev_rate = None
    if len(inc) >= 2 and inc[0].revenue and inc[1].revenue:
        rev_rate = pct_change(inc[0].revenue, inc[1].revenue)
    rev_s, rev_f = _growth_score(rev_rate, "Revenue growth")
    sub_scores.append((rev_s, 0.35))
    factors.append(rev_f)

    # ── EPS growth — use NormalizedMetrics CAGR when available ────────────────
    # metrics.eps_growth_pct is an annualized % (e.g. 12.5 means 12.5%).
    # Convert to a rate (0.125) for _growth_score.
    _eps_cagr_pct: Optional[float] = None    # preserved for growth quality check
    if metrics is not None and metrics.eps_growth_pct is not None:
        eps_rate      = metrics.eps_growth_pct / 100.0
        _eps_cagr_pct = metrics.eps_growth_pct
        eps_s, eps_f  = _growth_score(eps_rate, "EPS growth (3Y CAGR)")
        print(
            f"  [GROWTH] EPS growth from NormalizedMetrics:"
            f" {metrics.eps_growth_pct:.1f}% → score={eps_s:.0f}"
        )
    else:
        # Fallback: YoY change from most recent two annual statements
        eps_rate = None
        if len(inc) >= 2 and inc[0].eps_diluted and inc[1].eps_diluted:
            if inc[1].eps_diluted != 0:
                eps_rate = pct_change(inc[0].eps_diluted, inc[1].eps_diluted)
        eps_s, eps_f = _growth_score(eps_rate, "EPS growth")
        if metrics is not None:
            print(
                f"  [GROWTH] EPS growth: NormalizedMetrics.eps_growth_pct is None"
                f" — falling back to raw YoY EPS"
            )

    # ── FCF growth ─────────────────────────────────────────────────────────────
    fcf_rate = None
    if len(cfs) >= 2 and cfs[0].free_cash_flow and cfs[1].free_cash_flow:
        if cfs[1].free_cash_flow > 0:
            fcf_rate = pct_change(cfs[0].free_cash_flow, cfs[1].free_cash_flow)
    fcf_s, fcf_f = _growth_score(fcf_rate, "FCF growth")

    # ── 3-year revenue CAGR ────────────────────────────────────────────────────
    _rev_cagr_3y: Optional[float] = None     # preserved for growth quality check
    cagr_s, cagr_f = 50.0, "3Y revenue CAGR: N/A"
    if len(inc) >= 4 and inc[0].revenue and inc[3].revenue and inc[3].revenue > 0:
        _rev_cagr_3y = (inc[0].revenue / inc[3].revenue) ** (1 / 3) - 1
        cagr_s, cagr_f = _growth_score(_rev_cagr_3y, "3Y revenue CAGR")

    # ── EPS one-time inflation detection ─────────────────────────────────────
    # Must run BEFORE growth-quality classification so dampened value propagates.
    _one_time_fired = False
    _gq_detail      = ""
    if _eps_cagr_pct is not None and _rev_cagr_3y is not None and metrics is not None:
        _one_time_fired, _ot_signals = detect_eps_one_time_inflation(stock_data, metrics)
        if _one_time_fired:
            _rev_cagr_pct   = _rev_cagr_3y * 100
            _damped_eps_pct = min(_eps_cagr_pct, _rev_cagr_pct * 1.5)
            print(
                f"  [GROWTH] ONE_TIME_DAMPENED: eps_cagr={_eps_cagr_pct:.1f}%"
                f" → damped={_damped_eps_pct:.1f}%"
                f" (min({_eps_cagr_pct:.1f}%, {_rev_cagr_pct*1.5:.1f}%))"
                f" signals={_ot_signals}"
            )
            metrics.eps_one_time_dampened      = True
            metrics.eps_one_time_raw_pct       = _eps_cagr_pct
            metrics.eps_one_time_effective_pct = _damped_eps_pct
            metrics.eps_one_time_reason        = "; ".join(_ot_signals)
            _eps_cagr_pct = _damped_eps_pct
            eps_rate      = _damped_eps_pct / 100.0
            eps_s, eps_f  = _growth_score(eps_rate, "EPS growth (3Y CAGR)")
            eps_f         = eps_f + " [one-time dampened]"
            _gq_detail    = (
                f"EPS CAGR dampened from {metrics.eps_one_time_raw_pct:.1f}% to "
                f"{_damped_eps_pct:.1f}% — one-time item(s) detected "
                f"({len(_ot_signals)} signal(s)); capped at rev_cagr×1.5."
            )

    # ── Growth quality classification ─────────────────────────────────────────
    # Requires the 3Y CAGR path for both EPS and revenue so the comparison is
    # apples-to-apples (multi-year trends, not noisy single-year moves).
    # Skipped when one-time dampening already applied (stronger adjustment).
    _is_leverage_driven = False
    _gq_label           = "N/A"
    if _eps_cagr_pct is not None and _rev_cagr_3y is not None and not _one_time_fired:
        _is_leverage_driven, _gq_label, _gq_detail = _classify_growth_quality(
            _eps_cagr_pct, _rev_cagr_3y
        )
        if _is_leverage_driven:
            # Reduce eps_growth sub-score: leverage-driven earnings are less durable.
            # 15% haircut — meaningful but not punitive; real EPS growth still counts.
            eps_s = max(eps_s * 0.85, 20.0)
            eps_f = eps_f + " [quality-adjusted]"
            print(
                f"  [GROWTH] LOW_GROWTH_QUALITY: EPS CAGR={_eps_cagr_pct:.1f}%"
                f" vs rev CAGR={_rev_cagr_3y*100:.1f}% → eps_s haircut 15%"
            )
    elif _one_time_fired:
        _gq_label = "one-time dampened"

    # ── Assemble sub-scores ───────────────────────────────────────────────────
    sub_scores.append((eps_s,   0.35))
    factors.append(eps_f)
    sub_scores.append((fcf_s,   0.20))
    factors.append(fcf_f)
    sub_scores.append((cagr_s,  0.10))
    factors.append(cagr_f)

    # Growth quality output line (always emitted when data allows classification)
    if _eps_cagr_pct is not None and _rev_cagr_3y is not None:
        factors.append(
            f"Growth quality: {_gq_label} — "
            + (_gq_detail if _gq_detail else
               f"EPS CAGR {_eps_cagr_pct:.1f}% / Revenue CAGR {_rev_cagr_3y*100:.1f}%")
        )

    total_w   = sum(w for _, w in sub_scores)
    composite = sum(s * w for s, w in sub_scores) / total_w

    available_data = sum(1 for s, _ in sub_scores if s != 50.0)
    data_quality   = (
        "good"    if available_data >= 3 else
        "partial" if available_data >= 1 else
        "missing"
    )

    # Reasoning — name the quality classification when it's material
    if _one_time_fired and metrics is not None:
        reasoning = (
            f"EPS CAGR ({metrics.eps_one_time_raw_pct:.1f}%) dampened to "
            f"{metrics.eps_one_time_effective_pct:.1f}% — one-time item(s) detected; "
            f"capped at rev_cagr×1.5 for scoring."
        )
    elif _is_leverage_driven:
        reasoning = (
            f"EPS growth ({_eps_cagr_pct:.1f}% CAGR) overstates growth quality — "
            f"revenue CAGR ({_rev_cagr_3y*100:.1f}%) reveals earnings leverage, "
            f"not organic expansion. Score adjusted."
        )
    elif composite >= 80:
        reasoning = "Company is growing rapidly across revenue, earnings, and cash flow."
    elif composite >= 60:
        reasoning = "Moderate but consistent growth — solid execution."
    elif composite >= 40:
        reasoning = "Growth is slowing or mixed — worth monitoring."
    else:
        reasoning = "Revenue or earnings in decline — headwinds are material."

    return CategoryScore(
        name="growth",
        score=clamp(composite),
        weight=weight,
        factors=factors,
        reasoning=reasoning,
        data_quality=data_quality,
    )
