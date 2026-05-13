"""
Profitability scoring module.

Evaluates margin quality and capital efficiency.

When NormalizedMetrics is supplied, all margin and return-on-capital fields
(gross_margin, operating_margin, net_margin, roe, roa) are sourced from the
normalized object.  This guarantees the scorecard uses the same values as
the report header.  ROIC still comes from the raw ratios object because it
is not carried in NormalizedMetrics.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from models.scorecard import CategoryScore
from models.stock_data import StockData
from utils.helpers import clamp

if TYPE_CHECKING:
    from analysis.metrics import NormalizedMetrics


def _margin_score(value: float | None, thresholds: list[tuple], label: str) -> tuple[float, str]:
    if value is None:
        return 50.0, f"{label}: N/A"
    pct = value * 100
    for bound, score in sorted(thresholds, key=lambda x: x[0]):
        if value <= bound:
            return float(score), f"{label}: {pct:.1f}%"
    return float(thresholds[-1][1]), f"{label}: {pct:.1f}%"


def _return_score(value: float | None, label: str) -> tuple[float, str]:
    """Generic return metric (ROE, ROA, ROIC)."""
    if value is None:
        return 50.0, f"{label}: N/A"
    pct = value * 100
    if value >= 0.30:
        score = 95.0
    elif value >= 0.20:
        score = 85.0
    elif value >= 0.12:
        score = 72.0
    elif value >= 0.06:
        score = 58.0
    elif value >= 0.0:
        score = 40.0
    else:
        score = 20.0   # negative returns
    return score, f"{label}: {pct:.1f}%"


def score_profitability(
    stock_data: StockData,
    weight: float = 0.20,
    metrics: "Optional[NormalizedMetrics]" = None,
) -> CategoryScore:
    """
    Compute a 0–100 profitability quality score.
    100 = excellent margins + strong capital efficiency.

    When *metrics* is provided, margins and ROE/ROA are sourced from
    NormalizedMetrics (same values shown in the report header).
    """
    ratios  = stock_data.latest_ratios
    income  = stock_data.latest_income
    balance = stock_data.latest_balance
    factors: list[str] = []
    sub_scores: list[tuple[float, float]] = []

    # ── Resolve each metric: NormalizedMetrics first, raw fallback ─────────────

    # Gross margin
    if metrics is not None and metrics.gross_margin is not None:
        gross_margin: Optional[float] = metrics.gross_margin
    else:
        gross_margin = (ratios.gross_margin if ratios else None) or (
            income.gross_profit_ratio if income else None
        )
        if gross_margin is None and income and income.gross_profit and income.revenue and income.revenue > 0:
            gross_margin = income.gross_profit / income.revenue

    # Operating margin
    if metrics is not None and metrics.operating_margin is not None:
        op_margin: Optional[float] = metrics.operating_margin
    else:
        op_margin = (ratios.operating_margin if ratios else None) or (
            income.operating_income_ratio if income else None
        )
        if op_margin is None and income and income.operating_income and income.revenue and income.revenue > 0:
            op_margin = income.operating_income / income.revenue

    # Net margin
    if metrics is not None and metrics.net_margin is not None:
        net_margin: Optional[float] = metrics.net_margin
    else:
        net_margin = (ratios.net_margin if ratios else None) or (
            income.net_income_ratio if income else None
        )
        if net_margin is None and income and income.net_income and income.revenue and income.revenue > 0:
            net_margin = income.net_income / income.revenue

    # ROE — skip entirely when equity is negative/tiny (metrics flag set in compute_core_metrics)
    roe_nm = getattr(metrics, "roe_not_meaningful", False)
    if roe_nm:
        roe: Optional[float] = None
    elif metrics is not None and metrics.roe is not None:
        roe = metrics.roe
    else:
        roe = ratios.roe if ratios else None
        # Guard: only derive if equity is meaningfully positive (same rule as metrics.py)
        if roe is None and income and income.net_income and balance:
            eq = balance.total_equity
            ta = balance.total_assets
            eq_ok = eq is not None and eq > 0 and (ta is None or ta <= 0 or eq >= 0.05 * ta)
            if eq_ok:
                roe = income.net_income / balance.total_equity

    # ROA — sourced from NormalizedMetrics; no raw derivation (not critical enough)
    roa: Optional[float] = (metrics.roa if metrics is not None else None) or (ratios.roa if ratios else None)

    # ROIC — not in NormalizedMetrics; always from raw ratios
    roic: Optional[float] = ratios.roic if ratios else None

    print(
        f"  [PROFITABILITY] gross={gross_margin} op={op_margin}"
        f" net={net_margin} roe={roe} roa={roa} roic={roic}"
        f" source={'metrics' if metrics else 'raw'}"
    )

    # ── Score each metric ──────────────────────────────────────────────────────

    gm_thresholds = [(0.10, 25), (0.20, 40), (0.30, 58), (0.40, 72), (0.50, 84), (0.65, 92), (1.0, 97)]
    gm_s, gm_f = _margin_score(gross_margin, gm_thresholds, "Gross margin")
    sub_scores.append((gm_s, 0.20))
    factors.append(gm_f)

    om_thresholds = [(0.0, 20), (0.05, 38), (0.10, 55), (0.15, 70), (0.20, 82), (0.30, 91), (1.0, 97)]
    om_s, om_f = _margin_score(op_margin, om_thresholds, "Operating margin")
    sub_scores.append((om_s, 0.20))
    factors.append(om_f)

    nm_thresholds = [(0.0, 20), (0.02, 35), (0.05, 52), (0.10, 68), (0.15, 80), (0.25, 90), (1.0, 96)]
    nm_s, nm_f = _margin_score(net_margin, nm_thresholds, "Net margin")
    sub_scores.append((nm_s, 0.20))
    factors.append(nm_f)

    if roe_nm:
        roe_s, roe_f = 50.0, "ROE: — (negative/tiny equity base; metric not meaningful)"
    else:
        roe_s, roe_f = _return_score(roe, "ROE")
    sub_scores.append((roe_s, 0.20))
    factors.append(roe_f)

    roic_s, roic_f = _return_score(roic, "ROIC/ROCE")
    sub_scores.append((roic_s, 0.20))
    factors.append(roic_f)

    # ── Margin trend (last 2 years) ────────────────────────────────────────────
    if len(stock_data.income_statements) >= 2:
        curr = stock_data.income_statements[0].net_income_ratio
        prev = stock_data.income_statements[1].net_income_ratio
        if curr is not None and prev is not None:
            delta = curr - prev
            if delta > 0.02:
                factors.append("Net margins expanding YoY — positive trend")
            elif delta < -0.02:
                factors.append("Net margins compressing YoY — watch carefully")

    total_w = sum(w for _, w in sub_scores)
    composite = sum(s * w for s, w in sub_scores) / total_w

    available = sum(1 for s, _ in sub_scores if s != 50.0)
    data_quality = "good" if available >= 4 else "partial" if available >= 2 else "missing"

    # ── Metric-referencing reasoning ──────────────────────────────────────────
    _parts: list[str] = []
    if gross_margin is not None:
        _parts.append(f"gross margin {gross_margin*100:.1f}%")
    if net_margin is not None:
        _parts.append(f"net margin {net_margin*100:.1f}%")
    if roe is not None:
        _parts.append(f"ROE {roe*100:.1f}%")
    _metric_str = ", ".join(_parts) if _parts else "profitability metrics"

    if composite >= 80:
        reasoning = f"High-quality business: {_metric_str} — strong margins and capital returns."
    elif composite >= 60:
        reasoning = f"Decent profitability: {_metric_str} — adequate margins with acceptable returns."
    elif composite >= 40:
        reasoning = f"Below-average profitability: {_metric_str} — thin margins limit earnings power."
    else:
        reasoning = f"Weak profitability: {_metric_str} — poor returns on capital employed."

    return CategoryScore(
        name="profitability",
        score=clamp(composite),
        weight=weight,
        factors=factors,
        reasoning=reasoning,
        data_quality=data_quality,
    )
