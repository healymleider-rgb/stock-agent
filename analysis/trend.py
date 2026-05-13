"""
Trend detection for key financial metrics.

Uses 5-year historical income statement and ratio data already in memory.
No new API calls; no statistical libraries required.

classify_trend() uses slope direction + relative variance:
  Expanding     → ≥60% of year-over-year steps move in the improving direction
  Deteriorating → ≥60% of steps move in the deteriorating direction
  Volatile      → mixed direction AND relative step size > 15% of mean value
  Stable        → everything else
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.stock_data import StockData

_SIGNAL = {
    "Expanding":     "↑",
    "Deteriorating": "↓",
    "Stable":        "→",
    "Volatile":      "⚠",
}


def classify_trend(values_newest_first: list[float]) -> tuple[str, str]:
    """
    Classify a metric time-series into one of four trend labels.

    Parameters
    ----------
    values_newest_first : list of floats, newest observation first.
        Caller should already strip None values before passing.

    Returns (label, signal_char).
    """
    vals = [v for v in values_newest_first if v is not None]
    if len(vals) < 2:
        return "Stable", "→"

    # Oldest-first for directional analysis
    vals = list(reversed(vals))
    changes = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    n = len(changes)

    up = sum(1 for c in changes if c > 0)
    dn = sum(1 for c in changes if c < 0)

    mean_abs = sum(abs(v) for v in vals) / len(vals)
    avg_step = sum(abs(c) for c in changes) / n
    rel_vol  = avg_step / mean_abs if mean_abs > 1e-9 else 0.0

    threshold = 0.60  # 60% of steps must align for a directional call

    # Directional check FIRST — a clear trend overrides volatility.
    # "Volatile" only applies when direction is genuinely mixed (sign changes
    # or high relative swing) and neither directional threshold is met.
    if up >= n * threshold:
        label = "Expanding"
    elif dn >= n * threshold:
        label = "Deteriorating"
    elif up > 0 and dn > 0:
        # Mixed direction: volatile only on sign changes or large relative swings
        has_sign_change = any(vals[i] * vals[i + 1] < 0 for i in range(len(vals) - 1))
        label = "Volatile" if (has_sign_change or rel_vol > 0.20) else "Stable"
    else:
        label = "Stable"

    return label, _SIGNAL[label]


@dataclass
class TrendResult:
    # Per-metric trend labels
    revenue_growth:     str = "Stable"
    op_margin:          str = "Stable"
    net_margin:         str = "Stable"
    roe:                str = "Stable"
    roic:               str = "Stable"

    # Visual signals
    revenue_growth_sig: str = "→"
    op_margin_sig:      str = "→"
    net_margin_sig:     str = "→"
    roe_sig:            str = "→"
    roic_sig:           str = "→"

    # Score adjustments (applied externally by the scoring agent)
    growth_adj:         float = 0.0
    profitability_adj:  float = 0.0

    # Valuation driver adjustments — applied to base-case assumptions in driver model
    # valuation_margin_adj: pp shift to base operating margin (decimal, e.g. +0.01 = +1pp)
    # valuation_rev_adj:    rate shift to base revenue growth  (decimal, e.g. -0.03 = −3pp)
    valuation_margin_adj: float = 0.0
    valuation_rev_adj:    float = 0.0

    # Confidence penalty — subtracted from NormalizedMetrics confidence (0–1 scale)
    confidence_penalty:   float = 0.0

    # ── Trend window disagreement metadata ────────────────────────────────────
    # Set when the full-window op-margin trend and the recent-3Y trend disagree.
    # When they disagree, the recent 3Y window is used for valuation_margin_adj.
    op_margin_window_disagree: bool = False
    op_margin_full_trend:      str  = ""
    op_margin_recent_trend:    str  = ""


def _margin_series(stmts, numerator_attr: str) -> list[float]:
    """Extract margin values (as decimals) from income statements, newest first."""
    out = []
    for s in stmts:
        num = getattr(s, numerator_attr, None)
        rev = getattr(s, "revenue", None)
        if num is not None and rev and rev > 0:
            out.append(num / rev)
    return out


def _revenue_growth_series(stmts) -> list[float]:
    """YoY revenue growth rates, newest first (e.g. 0.10 = +10%)."""
    revs = [s.revenue for s in stmts if getattr(s, "revenue", None) and s.revenue > 0]
    if len(revs) < 2:
        return []
    return [(revs[i] - revs[i + 1]) / revs[i + 1] for i in range(len(revs) - 1)]


def _ratio_series(ratios, attr: str) -> list[float]:
    """Extract a ratio field across annual ratio objects, newest first."""
    return [v for r in ratios if (v := getattr(r, attr, None)) is not None]


def detect_trends(stock_data: "StockData") -> TrendResult:
    """
    Detect trends for the five key metrics using available historical data.

    Income statements and ratios must be annual, newest first (standard pipeline
    ordering). Uses up to 5 years; degrades gracefully with fewer periods.
    """
    inc   = stock_data.income_statements or []
    ratios = stock_data.ratios            or []

    # ── Series extraction ─────────────────────────────────────────────────────
    rev_g_series   = _revenue_growth_series(inc)
    op_mg_series   = _margin_series(inc, "operating_income")
    net_mg_series  = _margin_series(inc, "net_income")
    roe_series     = _ratio_series(ratios, "roe")
    roic_series    = _ratio_series(ratios, "roic")

    # ── Classify each metric ──────────────────────────────────────────────────
    rev_g_label,  rev_g_sig  = classify_trend(rev_g_series)
    net_mg_label, net_mg_sig = classify_trend(net_mg_series)
    roe_label,    roe_sig    = classify_trend(roe_series)
    roic_label,   roic_sig   = classify_trend(roic_series)

    # Op margin: compute full-window AND recent-3Y window.
    # When they disagree (e.g. long-run down but recent recovering), prefer the
    # recent window for the valuation adjustment — it reflects current direction.
    op_mg_full_label,   op_mg_sig   = classify_trend(op_mg_series)
    op_mg_recent_label, _           = classify_trend(op_mg_series[:3])  # newest 3 years
    _op_mg_window_disagree = (
        len(op_mg_series) >= 4          # only relevant when full window is longer
        and op_mg_full_label != op_mg_recent_label
    )
    op_mg_label = op_mg_recent_label if _op_mg_window_disagree else op_mg_full_label
    op_mg_sig   = _SIGNAL[op_mg_label]
    if _op_mg_window_disagree:
        print(
            f"  [TREND:window_disagree] op_margin full={op_mg_full_label} "
            f"recent={op_mg_recent_label} → using recent"
        )

    # ── Score adjustments ─────────────────────────────────────────────────────
    # Growth score adjustment from revenue growth trend
    _growth_adj_map = {
        "Expanding":     +3.0,
        "Deteriorating": -5.0,
        "Stable":         0.0,
        "Volatile":       -3.0,
    }
    # Profitability score adjustment from operating margin trend
    _prof_adj_map = {
        "Expanding":     +5.0,
        "Deteriorating": -5.0,
        "Stable":         0.0,
        "Volatile":       -3.0,
    }

    growth_adj        = _growth_adj_map[rev_g_label]
    profitability_adj = _prof_adj_map[op_mg_label]

    # ── Valuation driver adjustments ──────────────────────────────────────────
    # Expanding margins get credit in the base case (+1pp to base op_margin).
    # Deteriorating margins are penalised (−1pp): trend works against current level.
    # Volatile margins: no adjustment — too uncertain to shift the base.
    _val_margin_map = {
        "Expanding":     +0.01,
        "Deteriorating": -0.01,
        "Stable":         0.0,
        "Volatile":       0.0,
    }
    # Deteriorating revenue growth → cut base rev assumption by 3pp vs raw CAGR.
    # Volatile revenue → cut by 2pp: conservative when direction is unclear.
    # Expanding: CAGR already reflects the trend; no double-count.
    _val_rev_map = {
        "Expanding":      0.0,
        "Deteriorating": -0.03,
        "Stable":         0.0,
        "Volatile":      -0.02,
    }
    valuation_margin_adj = _val_margin_map[op_mg_label]
    valuation_rev_adj    = _val_rev_map[rev_g_label]

    # ── Confidence penalty for volatile metrics ────────────────────────────────
    all_labels = [rev_g_label, op_mg_label, net_mg_label, roe_label, roic_label]
    volatile_count = sum(1 for lbl in all_labels if lbl == "Volatile")
    confidence_penalty = 0.12 if volatile_count >= 2 else (0.05 if volatile_count == 1 else 0.0)

    return TrendResult(
        revenue_growth       = rev_g_label,
        op_margin            = op_mg_label,
        net_margin           = net_mg_label,
        roe                  = roe_label,
        roic                 = roic_label,
        revenue_growth_sig   = rev_g_sig,
        op_margin_sig        = op_mg_sig,
        net_margin_sig       = net_mg_sig,
        roe_sig              = roe_sig,
        roic_sig             = roic_sig,
        growth_adj           = growth_adj,
        profitability_adj    = profitability_adj,
        valuation_margin_adj = valuation_margin_adj,
        valuation_rev_adj    = valuation_rev_adj,
        confidence_penalty   = confidence_penalty,
        op_margin_window_disagree = _op_mg_window_disagree,
        op_margin_full_trend      = op_mg_full_label,
        op_margin_recent_trend    = op_mg_recent_label,
    )
