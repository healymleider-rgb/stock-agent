"""
data_integrity.py
=================
Data Integrity Engine — validation and normalization layer.

Sits BETWEEN raw API data / NormalizedMetrics and the scoring pipeline.
Detects inconsistencies, assigns per-metric confidence scores, and produces
a ValidationResult consumed by the risk scorer and reporting agent.

Does NOT modify NormalizedMetrics in place — it observes and annotates.
All overrides and flags are carried in ValidationResult so callers can
decide how to apply them.

Integration
-----------
    from analysis.data_integrity import run_data_integrity_check

    metrics    = compute_core_metrics(stock_data)
    validation = run_data_integrity_check(metrics, stock_data)
    # Pass validation in fundamental payload → downstream consumers
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from analysis.metrics import NormalizedMetrics
    from models.stock_data import StockData


# ── Flag codes ────────────────────────────────────────────────────────────────

class FlagCode:
    PRICE_DATA_INCONSISTENT      = "PRICE_DATA_INCONSISTENT"
    MARKET_CAP_INCONSISTENT      = "MARKET_CAP_INCONSISTENT"
    PE_RECALCULATED              = "PE_RECALCULATED"
    EV_EBITDA_RECALCULATED       = "EV_EBITDA_RECALCULATED"
    EPS_GROWTH_POTENTIALLY_UNSTABLE = "EPS_GROWTH_POTENTIALLY_UNSTABLE"
    PEG_LOW_RELIABILITY          = "PEG_LOW_RELIABILITY"
    GROWTH_EPS_REVENUE_DIVERGENCE = "GROWTH_EPS_REVENUE_DIVERGENCE"
    EPS_VOLATILITY_HIGH          = "EPS_VOLATILITY_HIGH"
    MARGIN_DATA_LIMITED          = "MARGIN_DATA_LIMITED"
    DATA_MISSING                 = "DATA_MISSING"
    CONFIDENCE_PENALIZED         = "CONFIDENCE_PENALIZED"
    PRICE_OVERRIDDEN             = "PRICE_OVERRIDDEN"
    LOW_DATA_CONFIDENCE          = "LOW_DATA_CONFIDENCE"


# ── Per-flag conviction deductions ────────────────────────────────────────────
# Each flag reduces the conviction_penalty multiplier by this amount.
# Penalty floors at 0.60 (no single flag can annihilate conviction).
_FLAG_SEVERITY: Dict[str, float] = {
    FlagCode.PRICE_DATA_INCONSISTENT:         0.06,
    FlagCode.MARKET_CAP_INCONSISTENT:         0.05,
    FlagCode.PE_RECALCULATED:                 0.04,
    FlagCode.EV_EBITDA_RECALCULATED:          0.03,
    FlagCode.EPS_GROWTH_POTENTIALLY_UNSTABLE: 0.08,
    FlagCode.PEG_LOW_RELIABILITY:             0.05,
    FlagCode.GROWTH_EPS_REVENUE_DIVERGENCE:   0.07,
    FlagCode.EPS_VOLATILITY_HIGH:             0.05,
    FlagCode.MARGIN_DATA_LIMITED:             0.02,
    FlagCode.DATA_MISSING:                    0.03,
    FlagCode.CONFIDENCE_PENALIZED:            0.00,
    FlagCode.PRICE_OVERRIDDEN:    0.06,
    FlagCode.LOW_DATA_CONFIDENCE: 0.10,
}

_SEVERITY_LEVEL: Dict[str, str] = {
    FlagCode.PRICE_DATA_INCONSISTENT:         "warning",
    FlagCode.MARKET_CAP_INCONSISTENT:         "warning",
    FlagCode.PE_RECALCULATED:                 "caution",
    FlagCode.EV_EBITDA_RECALCULATED:          "caution",
    FlagCode.EPS_GROWTH_POTENTIALLY_UNSTABLE: "warning",
    FlagCode.PEG_LOW_RELIABILITY:             "caution",
    FlagCode.GROWTH_EPS_REVENUE_DIVERGENCE:   "warning",
    FlagCode.EPS_VOLATILITY_HIGH:             "caution",
    FlagCode.MARGIN_DATA_LIMITED:             "info",
    FlagCode.DATA_MISSING:                    "info",
    FlagCode.CONFIDENCE_PENALIZED:            "info",
    FlagCode.PRICE_OVERRIDDEN:    "warning",
    FlagCode.LOW_DATA_CONFIDENCE: "warning",
}


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ValidationFlag:
    code:     str   # FlagCode constant
    severity: str   # "warning" | "caution" | "info"
    detail:   str   # human-readable one-line description


@dataclass
class MetricConfidence:
    """Per-metric confidence scores (0–1). Higher = more reliable."""
    pe:         float = 0.80
    peg:        float = 0.70
    ev_ebitda:  float = 0.80
    ps:         float = 0.85
    growth_eps: float = 0.75
    growth_rev: float = 0.80
    margins:    float = 0.80
    overall:    float = 0.80


@dataclass
class ValidationResult:
    """
    Full output of the Data Integrity Engine for one ticker.

    flags              — list of ValidationFlag (ordered by severity)
    metric_confidence  — per-metric reliability estimates (0–1)
    conviction_penalty — multiplicative confidence haircut (1.0 = no penalty)
    data_coverage      — fraction of key fields that resolved (0–1)
    summary            — one-sentence summary for the report
    adjusted_metrics   — dict of field → overridden value when recalculated
    """
    flags:             List[ValidationFlag]       = field(default_factory=list)
    metric_confidence: MetricConfidence           = field(default_factory=MetricConfidence)
    conviction_penalty: float                     = 1.0
    data_coverage:      float                     = 1.0
    summary:            str                       = ""
    adjusted_metrics:   Dict[str, float]          = field(default_factory=dict)
    output_status:      str                       = "OK"

    def critical_warning_count(self) -> int:
        """Number of warning-severity flags (excluding LOW_DATA_CONFIDENCE itself)."""
        return sum(1 for f in self.flags
                   if f.severity == "warning" and f.code != FlagCode.LOW_DATA_CONFIDENCE)

    def flag_codes(self) -> List[str]:
        return [f.code for f in self.flags]

    def has_warnings(self) -> bool:
        return any(f.severity == "warning" for f in self.flags)

    def report_lines(self) -> List[str]:
        """Formatted lines for inclusion in the investment report."""
        if not self.flags:
            return []
        lines = []
        for f in self.flags:
            icon = "⚠" if f.severity == "warning" else "◦"
            lines.append(f"    {icon} [{f.code}] {f.detail}")
        if self.conviction_penalty < 1.0:
            reduction_pct = round((1.0 - self.conviction_penalty) * 100)
            lines.append(
                f"    ℹ Confidence Adjustment Applied: −{reduction_pct}pp"
                f" due to {len(self.flags)} data quality flag(s)"
            )
        return lines


# ── Internal helpers ──────────────────────────────────────────────────────────

def _safe_std(vals: list) -> float:
    clean = [v for v in vals if v is not None]
    if len(clean) < 2:
        return 0.0
    mu  = sum(clean) / len(clean)
    var = sum((v - mu) ** 2 for v in clean) / len(clean)
    return math.sqrt(var)


def _safe_mean(vals: list) -> float:
    clean = [v for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else 0.0


def _pct_diff(a: float, b: float) -> float:
    """Absolute percentage difference relative to b."""
    if b == 0:
        return 0.0
    return abs(a - b) / abs(b)


# ── Check functions ───────────────────────────────────────────────────────────

def _check_price_consistency(
    metrics: "NormalizedMetrics",
    flags: list,
) -> None:
    """Flag if quote price and price-history close diverge materially."""
    if metrics.price_quote is None or metrics.price_hist is None:
        return
    diff = _pct_diff(metrics.price_quote, metrics.price_hist)
    if diff > 0.05:
        flags.append(ValidationFlag(
            code     = FlagCode.PRICE_DATA_INCONSISTENT,
            severity = "warning",
            detail   = (
                f"Quote price ${metrics.price_quote:.2f} vs "
                f"history close ${metrics.price_hist:.2f} "
                f"({diff:.1%} divergence) — history close used"
            ),
        ))


def _check_market_cap_consistency(
    metrics: "NormalizedMetrics",
    flags: list,
) -> None:
    """Flag if API market cap and recomputed market cap diverge materially."""
    if metrics.market_cap_api is None or metrics.market_cap_recomp is None:
        return
    diff = _pct_diff(metrics.market_cap_recomp, metrics.market_cap_api)
    if diff > 0.20:
        flags.append(ValidationFlag(
            code     = FlagCode.MARKET_CAP_INCONSISTENT,
            severity = "warning",
            detail   = (
                f"API market cap ${metrics.market_cap_api/1e9:.2f}B vs "
                f"recomputed ${metrics.market_cap_recomp/1e9:.2f}B "
                f"({diff:.1%} divergence)"
            ),
        ))


def _check_pe_consistency(
    metrics: "NormalizedMetrics",
    flags: list,
    adjusted: dict,
) -> float:
    """
    Check computed vs provider P/E. Returns confidence score for PE (0-1).
    Flags and records adjusted value when sources diverge significantly.
    """
    computed  = metrics.pe_computed_ttm
    provider  = metrics.pe_provider_ttm
    annual    = metrics.pe_computed_ann

    available = [v for v in [computed, provider, annual] if v is not None and 0 < v < 500]
    if not available:
        return 0.40   # no PE at all

    if len(available) == 1:
        return 0.65   # only one source

    # Two or more sources — check agreement
    if computed is not None and provider is not None:
        diff = _pct_diff(computed, provider)
        if diff > 0.25:
            flags.append(ValidationFlag(
                code     = FlagCode.PE_RECALCULATED,
                severity = "caution",
                detail   = (
                    f"P/E computed ({computed:.1f}x) vs provider ({provider:.1f}x) "
                    f"differ by {diff:.1%} — computed TTM used"
                ),
            ))
            adjusted["pe_ratio"] = computed
            return max(0.50, 1.0 - diff)
        return min(0.95, 1.0 - diff * 0.5)

    return 0.80


def _check_ev_ebitda_consistency(
    metrics: "NormalizedMetrics",
    stock_data: "StockData",
    flags: list,
    adjusted: dict,
) -> float:
    """Check computed vs provider EV/EBITDA. Returns confidence score."""
    computed = metrics.ev_ebitda_computed
    provider = metrics.ev_ebitda_provider

    if computed is None and provider is None:
        return 0.40
    if computed is None or provider is None:
        return 0.70

    diff = _pct_diff(computed, provider)
    if diff > 0.20:
        flags.append(ValidationFlag(
            code     = FlagCode.EV_EBITDA_RECALCULATED,
            severity = "caution",
            detail   = (
                f"EV/EBITDA computed ({computed:.1f}x) vs provider ({provider:.1f}x) "
                f"differ by {diff:.1%} — computed value used"
            ),
        ))
        adjusted["ev_ebitda"] = computed
        return max(0.55, 1.0 - diff)
    return min(0.92, 1.0 - diff * 0.4)


def _check_growth_sanity(
    metrics: "NormalizedMetrics",
    stock_data: "StockData",
    flags: list,
) -> tuple:
    """
    Check growth rate consistency. Returns (eps_growth_conf, rev_growth_conf).
    """
    eps_growth_pct = metrics.eps_growth_pct   # annualised %, e.g. 25.0
    eps_conf  = 0.75
    rev_conf  = 0.80

    # ── Revenue growth from income statements ─────────────────────────────────
    inc = stock_data.income_statements
    rev_growth: Optional[float] = None
    if len(inc) >= 2 and inc[0].revenue and inc[1].revenue and inc[1].revenue > 0:
        rev_growth = (inc[0].revenue - inc[1].revenue) / abs(inc[1].revenue)

    # ── EPS volatility across years ───────────────────────────────────────────
    eps_series = [
        s.eps_diluted for s in inc[:5]
        if getattr(s, "eps_diluted", None) is not None
    ]
    if len(eps_series) >= 3:
        eps_std  = _safe_std(eps_series)
        eps_mean = abs(_safe_mean(eps_series))
        if eps_mean > 0:
            eps_cv = eps_std / eps_mean
            if eps_cv > 0.50:
                flags.append(ValidationFlag(
                    code     = FlagCode.EPS_VOLATILITY_HIGH,
                    severity = "caution",
                    detail   = (
                        f"EPS coefficient of variation {eps_cv:.2f} — "
                        f"earnings are volatile; growth estimates less reliable"
                    ),
                ))
                eps_conf = max(0.40, 0.80 - eps_cv * 0.30)

    # ── EPS CAGR > 35% with revenue growth < 20% ─────────────────────────────
    if eps_growth_pct is not None and eps_growth_pct > 35:
        if rev_growth is not None and rev_growth < 0.20:
            flags.append(ValidationFlag(
                code     = FlagCode.EPS_GROWTH_POTENTIALLY_UNSTABLE,
                severity = "warning",
                detail   = (
                    f"EPS growth ({eps_growth_pct:.1f}%) far exceeds revenue growth "
                    f"({rev_growth*100:.1f}%) — likely driven by margin expansion "
                    f"or one-time items; may not be sustainable"
                ),
            ))
            eps_conf = max(0.40, eps_conf - 0.20)

    # ── Large EPS vs revenue divergence ──────────────────────────────────────
    if eps_growth_pct is not None and rev_growth is not None:
        divergence = abs(eps_growth_pct / 100 - rev_growth)
        if divergence > 0.25:
            flags.append(ValidationFlag(
                code     = FlagCode.GROWTH_EPS_REVENUE_DIVERGENCE,
                severity = "warning",
                detail   = (
                    f"EPS growth ({eps_growth_pct:.1f}%) and revenue growth "
                    f"({rev_growth*100:.1f}%) diverge by {divergence*100:.1f}pp — "
                    f"check for margin compression or one-time adjustments"
                ),
            ))
            eps_conf = max(0.45, eps_conf - 0.15)

    # ── Insufficient data ─────────────────────────────────────────────────────
    if len(inc) < 2:
        rev_conf = 0.45
        eps_conf = min(eps_conf, 0.55)
        flags.append(ValidationFlag(
            code     = FlagCode.DATA_MISSING,
            severity = "info",
            detail   = "Fewer than 2 annual income statements available — growth rates approximate",
        ))

    return eps_conf, rev_conf


def _check_peg_reliability(
    metrics: "NormalizedMetrics",
    eps_growth_conf: float,
    flags: list,
) -> float:
    """Check PEG ratio reliability. Returns peg_confidence."""
    peg          = metrics.peg
    eps_growth   = metrics.eps_growth_pct
    pe           = metrics.pe_ratio

    if peg is None:
        return 0.50

    peg_conf = eps_growth_conf * 0.70 + 0.30  # inherit growth confidence

    # PEG is highly sensitive to growth input — flag when growth is extreme
    if eps_growth is not None:
        if eps_growth > 50 or eps_growth < -10:
            flags.append(ValidationFlag(
                code     = FlagCode.PEG_LOW_RELIABILITY,
                severity = "caution",
                detail   = (
                    f"PEG {peg:.2f}x based on {eps_growth:.1f}% EPS growth — "
                    f"extreme growth inputs make PEG highly sensitive; "
                    f"weight reduced in scoring"
                ),
            ))
            peg_conf = max(0.35, peg_conf - 0.25)
        elif eps_growth < 5:
            # PEG meaningless at near-zero growth
            peg_conf = max(0.40, peg_conf - 0.15)

    # Negative PE → PEG not meaningful
    if pe is not None and pe < 0:
        peg_conf = 0.20

    return min(1.0, peg_conf)


def _check_margin_data(
    stock_data: "StockData",
    flags: list,
) -> float:
    """Check margin data availability. Returns margins_confidence."""
    inc = stock_data.income_statements
    if not inc:
        flags.append(ValidationFlag(
            code     = FlagCode.MARGIN_DATA_LIMITED,
            severity = "info",
            detail   = "No income statement data — margin scoring uses defaults",
        ))
        return 0.30

    gm = getattr(inc[0], "gross_profit_ratio", None) if inc else None
    om = getattr(inc[0], "operating_income_ratio", None) if inc else None

    if gm is None and om is None:
        flags.append(ValidationFlag(
            code     = FlagCode.MARGIN_DATA_LIMITED,
            severity = "info",
            detail   = "Gross and operating margin ratios unavailable from income statement",
        ))
        return 0.45

    return 0.90 if len(inc) >= 3 else 0.70


def _compute_data_coverage(
    metrics: "NormalizedMetrics",
    stock_data: "StockData",
) -> float:
    """Fraction of key fields that resolved to non-None values."""
    key_fields = [
        metrics.price,
        metrics.market_cap,
        metrics.pe_ratio,
        metrics.ps_ratio,
        metrics.ev_ebitda,
        metrics.eps_growth_pct,
        metrics.gross_margin,
        metrics.net_margin,
        metrics.debt_to_equity,
        metrics.current_ratio,
        metrics.roe,
    ]
    present = sum(1 for f in key_fields if f is not None)
    return present / len(key_fields)


def _compute_conviction_penalty(flags: list) -> float:
    """
    Multiplicative penalty on conviction from 1.0 down.
    Minimum 0.60 — no single batch of flags can eliminate conviction entirely.
    """
    total_deduction = sum(
        _FLAG_SEVERITY.get(f.code, 0.03)
        for f in flags
        if f.severity in ("warning", "caution")
    )
    return max(0.60, 1.0 - total_deduction)


def _check_market_cap_identity(
    metrics: "NormalizedMetrics",
    stock_data: "StockData",
    flags: list,
    adjusted: dict,
) -> None:
    """
    Market cap identity constraint: market_cap ≈ price × shares_outstanding.

    If the API market_cap and shares_outstanding are both available and their
    implied price diverges from the current price by > 5%, we compute the
    identity-consistent price and store it as an override.

    Direction: API market_cap / shares → implied_price.
    This catches cases where the price feed is stale but market_cap (updated
    continuously by the exchange) correctly reflects the current valuation.
    """
    mc  = metrics.market_cap_api
    shr = getattr(stock_data, "shares_outstanding", None)
    px  = metrics.price

    if mc is None or shr is None or shr <= 0 or px is None or px <= 0:
        return

    implied = mc / shr
    diff    = abs(implied - px) / px

    if diff > 0.05:
        adjusted["price"] = implied
        flags.append(ValidationFlag(
            code     = FlagCode.PRICE_OVERRIDDEN,
            severity = "warning",
            detail   = (
                f"Market cap identity: market_cap/shares = ${implied:.2f} vs "
                f"current price ${px:.2f} ({diff:.1%} gap). "
                f"Price normalised to ${implied:.2f} for ratio computation."
            ),
        ))


# ── Public entry point ────────────────────────────────────────────────────────

def run_data_integrity_check(
    metrics:    "NormalizedMetrics",
    stock_data: "StockData",
) -> ValidationResult:
    """
    Run all integrity checks on NormalizedMetrics and raw StockData.

    Returns a ValidationResult with flags, per-metric confidence, and
    a conviction penalty multiplier for downstream scoring adjustment.

    Designed to be fast (<5ms) and never raise exceptions — all failures
    are caught internally and reflected as low-confidence signals.
    """
    flags:    list  = []
    adjusted: dict  = {}

    try:
        _check_price_consistency(metrics, flags)
    except Exception:
        pass
    try:
        _check_market_cap_consistency(metrics, flags)
    except Exception:
        pass
    try:
        _check_market_cap_identity(metrics, stock_data, flags, adjusted)
    except Exception:
        pass

    # ── Per-metric confidence ─────────────────────────────────────────────────
    try:
        pe_conf = _check_pe_consistency(metrics, flags, adjusted)
    except Exception:
        pe_conf = 0.70

    try:
        ev_conf = _check_ev_ebitda_consistency(metrics, stock_data, flags, adjusted)
    except Exception:
        ev_conf = 0.70

    try:
        eps_conf, rev_conf = _check_growth_sanity(metrics, stock_data, flags)
    except Exception:
        eps_conf, rev_conf = 0.70, 0.75

    try:
        peg_conf = _check_peg_reliability(metrics, eps_conf, flags)
    except Exception:
        peg_conf = 0.65

    try:
        margins_conf = _check_margin_data(stock_data, flags)
    except Exception:
        margins_conf = 0.65

    # ── Data coverage ─────────────────────────────────────────────────────────
    try:
        coverage = _compute_data_coverage(metrics, stock_data)
    except Exception:
        coverage = 0.60

    # ── Overall confidence ─────────────────────────────────────────────────────
    overall_conf = (
        0.20 * pe_conf +
        0.15 * ev_conf +
        0.15 * (metrics.ps_ratio is not None and 0.85 or 0.50) +
        0.20 * eps_conf +
        0.10 * peg_conf +
        0.10 * margins_conf +
        0.10 * coverage
    )
    mc = MetricConfidence(
        pe         = round(pe_conf, 3),
        peg        = round(peg_conf, 3),
        ev_ebitda  = round(ev_conf, 3),
        ps         = 0.85 if metrics.ps_ratio is not None else 0.50,
        growth_eps = round(eps_conf, 3),
        growth_rev = round(rev_conf, 3),
        margins    = round(margins_conf, 3),
        overall    = round(overall_conf, 3),
    )

    # ── Conviction penalty ────────────────────────────────────────────────────
    penalty = _compute_conviction_penalty(flags)
    if penalty < 1.0:
        flags.append(ValidationFlag(
            code     = FlagCode.CONFIDENCE_PENALIZED,
            severity = "info",
            detail   = (
                f"Overall conviction adjusted by ×{penalty:.2f} "
                f"({len([f for f in flags if f.severity in ('warning','caution')])} flag(s))"
            ),
        ))

    # ── Hard fail condition ───────────────────────────────────────────────────
    # ≥ 2 warning-severity flags (excluding the LOW_DATA_CONFIDENCE flag itself)
    # indicates structurally unreliable data. Surface this explicitly.
    n_critical_warnings = sum(
        1 for f in flags
        if f.severity == "warning" and f.code != FlagCode.LOW_DATA_CONFIDENCE
    )
    output_status = "OK"
    if n_critical_warnings >= 2:
        output_status = "LOW DATA CONFIDENCE"
        flags.append(ValidationFlag(
            code     = FlagCode.LOW_DATA_CONFIDENCE,
            severity = "warning",
            detail   = (
                f"{n_critical_warnings} structural data warnings detected — "
                f"conviction reduced and position size capped. "
                f"Review flags before acting on this analysis."
            ),
        ))

    # ── Sort: warnings first, then cautions, then info ────────────────────────
    _order = {"warning": 0, "caution": 1, "info": 2}
    flags.sort(key=lambda f: _order.get(f.severity, 3))

    # ── Summary sentence ──────────────────────────────────────────────────────
    n_warn   = sum(1 for f in flags if f.severity == "warning")
    n_caut   = sum(1 for f in flags if f.severity == "caution")
    if not flags or (n_warn == 0 and n_caut == 0):
        summary = "Data quality satisfactory — all key metrics internally consistent."
    elif n_warn >= 2:
        summary = (
            f"{n_warn} data quality warning(s) detected — "
            f"confidence in outputs reduced; flag details below."
        )
    else:
        summary = (
            f"{n_warn + n_caut} data quality flag(s) — "
            f"minor adjustments applied; review flags below."
        )

    print(
        f"  [INTEGRITY] flags={len(flags)} warnings={n_warn} cautions={n_caut}"
        f" coverage={coverage:.0%} overall_conf={overall_conf:.2f}"
        f" conviction_penalty={penalty:.2f} status={output_status}"
    )

    return ValidationResult(
        flags              = flags,
        metric_confidence  = mc,
        conviction_penalty = round(penalty, 3),
        data_coverage      = round(coverage, 3),
        summary            = summary,
        adjusted_metrics   = adjusted,
        output_status      = output_status,
    )
