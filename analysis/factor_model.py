"""
factor_model.py
===============
Cross-sectional seven-factor scoring for the alpha decision engine.

Computes z-scores for seven factors relative to the peer universe, applies
regime-conditional weighting and archetype multipliers, and collapses the
result into a composite factor score (0-100) with PM-readable interpretation.

Seven factors
─────────────
Quality         — ROE, ROIC, earnings consistency, interest coverage, inv leverage
Profitability   — gross margin, operating margin, net margin, margin trend, op leverage
Growth          — revenue CAGR, EPS growth, FCF growth, revenue acceleration
Value           — inverse P/E, inverse EV/EBITDA, FCF yield, inverse PEG
Momentum        — 12M return, 6M return, 3M return, SMA-50/-200 spread
Low Volatility  — inverse beta, earnings CV, max drawdown, FCF consistency
Macro/Cyclical  — sector defensiveness, rev volatility, yield sensitivity, earn vol
                  (INVERTED CONVENTION: high score = more defensive/non-cyclical)

Normalisation pipeline
──────────────────────
1. Winsorise raw scores at 1st/99th percentile across the universe
2. Cross-sectional z-score (clamped ±3) relative to peer universe
3. CDF mapping: score_i = Φ(z_i) × 100  where Φ is the standard normal CDF
   When peer universe < 3, solo z-score is computed against absolute anchors.

Composite scoring
─────────────────
1. Apply additive regime weight shifts to base factor weights
2. Clip to ≥ 0; renormalise to sum to 1.0
3. Apply archetype multipliers (multiplicative, based on style_label)
4. Renormalise to sum to 1.0
5. Composite = Σ(weight_i × score_i) via safe_weighted_composite()

Integration
-----------
    from analysis.factor_model import build_factor_profile
    from analysis.monte_carlo import mc_from_valuation_range

    profile = build_factor_profile(stock_data, peer_rows)
    mc      = mc_from_valuation_range(vr, factor_profile=profile, ...)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.stock_data import StockData


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class FactorProfile:
    """
    Cross-sectional seven-factor exposure vector for a single stock.

    All z-scores are relative to the peer universe supplied to
    build_factor_profile().  Positive z-scores indicate the stock
    ranks above the peer median on that factor.

    quality_factor   — 0-100 CDF score for quality factor alone; used to
                       override the MC quality_tier when peers ≥ 3.
    composite_score  — 0-100 weighted composite of all seven factors
                       using regime-adjusted and archetype-multiplied weights.
    style_label      — qualitative archetype driving weight multipliers.
    n_peers          — number of peer rows used; lower → lower confidence.
    confidence       — "high" (≥6), "medium" (≥3), "low" (<3)
    """
    # ── Cross-sectional z-scores ───────────────────────────────────────────────
    quality_z:      float  # composite quality (capital efficiency + earnings)
    value_z:        float  # composite value   (positive = cheap vs peers)
    momentum_z:     float  # price momentum    (positive = outperforming)
    growth_z:       float  # revenue + EPS growth
    stability_z:    float  # balance sheet + low-vol (positive = more stable)
    profitability_z: float # margin quality (GM, OP, NM, trends)
    macro_z:        float  # macro defensiveness (INVERTED: positive = defensive)

    # ── CDF scores (0-100) ─────────────────────────────────────────────────────
    quality_factor:      float  # Φ(quality_z) × 100   — MC quality_tier override
    profitability_score: float  # Φ(profitability_z) × 100
    macro_score:         float  # Φ(macro_z) × 100  (< 35 = highly cyclical)
    composite_score:     float  # regime + archetype weighted composite

    # ── Derived ────────────────────────────────────────────────────────────────
    style_label:      str    # "Compounder" | "Quality Value" | "Value" | …
    strongest_factor: str    # factor name with highest effective weight × score
    weakest_factor:   str    # factor name with lowest effective weight × score
    effective_weights: Dict[str, float]   # post-regime post-archetype weights
    pm_interpretation: str   # one-sentence PM takeaway

    n_peers:    int
    confidence: str   # "high" | "medium" | "low"


# ── Factor name constants ─────────────────────────────────────────────────────

_FACTOR_NAMES = (
    "quality", "profitability", "growth", "value", "momentum", "lowvol", "macro"
)

# ── Base factor weights ───────────────────────────────────────────────────────
# Sum = 1.0
_BASE_WEIGHTS: Dict[str, float] = {
    "quality":       0.20,
    "profitability": 0.15,
    "growth":        0.18,
    "value":         0.12,
    "momentum":      0.15,
    "lowvol":        0.12,
    "macro":         0.08,
}

# ── Regime weight shifts (additive deltas; sum to ~0 within each regime) ──────
# Applied before renormalisation.
_REGIME_WEIGHT_SHIFTS: Dict[str, Dict[str, float]] = {
    "early_cycle": {   # growth just beginning — favor growth/value/momentum
        "quality":       -0.02,
        "profitability": -0.02,
        "growth":        +0.05,
        "value":         +0.03,
        "momentum":      +0.03,
        "lowvol":        -0.04,
        "macro":         -0.03,
    },
    "mid_cycle": {     # stable expansion — quality/profitability lead
        "quality":       +0.02,
        "profitability": +0.02,
        "growth":        -0.02,
        "value":         -0.02,
        "momentum":      +0.01,
        "lowvol":         0.00,
        "macro":         -0.01,
    },
    "late_cycle": {    # slowdown approaching — rotate defensive
        "quality":       -0.02,
        "profitability": -0.03,
        "growth":        -0.05,
        "value":         +0.03,
        "momentum":      -0.04,
        "lowvol":        +0.06,
        "macro":         +0.05,
    },
    "slowdown": {      # growth faltering — quality + defensives
        "quality":       +0.03,
        "profitability": -0.08,
        "growth":        -0.05,
        "value":         +0.02,
        "momentum":      -0.05,
        "lowvol":        +0.07,
        "macro":         +0.06,
    },
    "recession": {     # contraction — defensive quality + LowVol dominate
        "quality":       +0.04,
        "profitability": +0.03,
        "growth":        -0.08,
        "value":         -0.12,
        "momentum":      -0.06,
        "lowvol":        +0.10,
        "macro":         +0.09,
    },
    "recovery": {      # rebound — risk-on, growth + momentum
        "quality":       -0.02,
        "profitability": -0.02,
        "growth":        +0.06,
        "value":         +0.04,
        "momentum":      +0.05,
        "lowvol":        -0.06,
        "macro":         -0.05,
    },
}

# ── Archetype multipliers (applied multiplicatively, then renormalised) ───────
_ARCHETYPE_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "Compounder": {
        "quality": 1.40, "profitability": 1.30, "growth": 1.10,
        "value": 0.90,   "momentum": 1.00,      "lowvol": 1.20, "macro": 0.90,
    },
    "Quality Value": {
        "quality": 1.20, "profitability": 1.10, "growth": 0.90,
        "value": 1.50,   "momentum": 0.80,      "lowvol": 1.10, "macro": 1.00,
    },
    "High Growth": {
        "quality": 0.90, "profitability": 0.80, "growth": 1.60,
        "value": 0.70,   "momentum": 1.30,      "lowvol": 0.70, "macro": 0.90,
    },
    "Value": {
        "quality": 1.10, "profitability": 1.00, "growth": 0.80,
        "value": 1.60,   "momentum": 0.80,      "lowvol": 1.10, "macro": 1.00,
    },
    "Momentum": {
        "quality": 0.90, "profitability": 0.90, "growth": 1.20,
        "value": 0.70,   "momentum": 1.70,      "lowvol": 0.80, "macro": 0.90,
    },
    "Cyclical": {
        "quality": 0.90, "profitability": 0.90, "growth": 1.20,
        "value": 1.20,   "momentum": 1.00,      "lowvol": 0.70, "macro": 1.50,
    },
    "Speculative": {
        "quality": 0.70, "profitability": 0.60, "growth": 1.50,
        "value": 0.80,   "momentum": 1.40,      "lowvol": 0.60, "macro": 0.80,
    },
    "Blend": {
        "quality": 1.00, "profitability": 1.00, "growth": 1.00,
        "value": 1.00,   "momentum": 1.00,      "lowvol": 1.00, "macro": 1.00,
    },
}

# ── Sector cyclicality lookup ─────────────────────────────────────────────────
# 0 = highly cyclical (Energy, Mining), 1 = fully defensive (Utilities)
# Inverted in _macro_raw() → high macro_raw = defensive = high score
_SECTOR_CYCLICALITY: Dict[str, float] = {
    # Cyclical sectors
    "energy":                   0.10,
    "materials":                0.20,
    "industrials":              0.25,
    "consumer discretionary":   0.20,
    "financials":               0.30,
    "real estate":              0.40,
    # Mixed
    "information technology":   0.45,
    "communication services":   0.50,
    # Defensive
    "consumer staples":         0.75,
    "healthcare":               0.80,
    "utilities":                0.95,
}

_DEFAULT_SECTOR_CYCLICALITY = 0.45   # unknown sector → treat as mixed


# ── Internal helpers ──────────────────────────────────────────────────────────

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


def _norm(val: Optional[float], lo: float, hi: float) -> Optional[float]:
    """Clamp-normalise val ∈ [lo, hi] → [0, 1].  Returns None when val is None."""
    if val is None:
        return None
    return max(0.0, min(1.0, (val - lo) / (hi - lo))) if hi > lo else None


def z_to_score(z: float) -> float:
    """
    Map a z-score to [0, 100] via the standard normal CDF.
    Φ(z) = (1 + erf(z / √2)) / 2
    """
    cdf = (1.0 + math.erf(z / math.sqrt(2.0))) / 2.0
    return cdf * 100.0


def winsorize(vals: list, lo_pct: float = 0.01, hi_pct: float = 0.99) -> list:
    """
    Clip values to [lo_pct, hi_pct] quantiles (in-place copy).
    None entries are kept as None; percentiles computed on non-None values.
    """
    clean = sorted(v for v in vals if v is not None)
    if len(clean) < 4:
        return list(vals)
    lo_idx = max(0, int(lo_pct * len(clean)))
    hi_idx = min(len(clean) - 1, int(hi_pct * len(clean)))
    lo_val, hi_val = clean[lo_idx], clean[hi_idx]
    return [
        None if v is None else max(lo_val, min(hi_val, v))
        for v in vals
    ]


def safe_weighted_composite(
    scores:  Dict[str, Optional[float]],
    weights: Dict[str, float],
) -> Optional[float]:
    """
    Weighted composite of factor scores, redistributing weight from missing
    factors proportionally to the present ones.

    Returns None if the total available weight is < 50% of the total weight
    (i.e. more than half the weight is missing).
    """
    total_w     = sum(weights.values())
    present_w   = sum(w for k, w in weights.items() if scores.get(k) is not None)
    if total_w == 0 or present_w < total_w * 0.50:
        return None
    scale = total_w / present_w   # redistribute missing weight
    return sum(
        scores[k] * weights[k] * scale
        for k in weights
        if scores.get(k) is not None
    )


def _z_score(val: Optional[float], universe: list) -> float:
    """
    Standardise val against a list that may contain None entries.
    Returns 0.0 when std is degenerate or val is missing.
    Clamped to [−3, +3].
    """
    if val is None:
        return 0.0
    std = _safe_std(universe)
    if std < 1e-9:
        return 0.0
    mu = _safe_mean(universe)
    return max(-3.0, min(3.0, (val - mu) / std))


# ── Per-entity raw factor scores ──────────────────────────────────────────────
# Each function returns a float in [0, 1] or None if all inputs are missing.

def _quality_raw(
    roe:               Optional[float],
    roic:              Optional[float],
    eps_consistency:   Optional[float],   # stddev of eps/revenue ratio YoY; lower = better
    interest_coverage: Optional[float],
    debt_equity:       Optional[float],
) -> Optional[float]:
    """
    Composite quality [0, 1].
    Weights: ROE 30%, ROIC 25%, earnings consistency 20%, coverage 15%, inv_leverage 10%.
    """
    parts: list = []

    if roe is not None and roe > -1.0:
        parts.append((_norm(max(0.0, roe), 0.0, 0.40), 0.30))   # 0-40% ROE

    if roic is not None and roic > -1.0:
        parts.append((_norm(max(0.0, roic), 0.0, 0.35), 0.25))  # 0-35% ROIC

    if eps_consistency is not None:
        # lower eps_consistency (less volatile) = higher score
        ec = 1.0 - _norm(eps_consistency, 0.0, 0.50)            # 0-50% CV
        if ec is not None:
            parts.append((ec, 0.20))

    if interest_coverage is not None and interest_coverage > 0:
        ic = _norm(min(interest_coverage, 25.0), 0.0, 25.0)
        if ic is not None:
            parts.append((ic, 0.15))

    if debt_equity is not None and debt_equity >= 0:
        de_inv = 1.0 - _norm(debt_equity, 0.0, 3.0)
        if de_inv is not None:
            parts.append((de_inv, 0.10))

    if not parts:
        return None
    total_w = sum(w for _, w in parts)
    return sum(s * w for s, w in parts) / total_w


def _profitability_raw(
    gross_margin:   Optional[float],
    op_margin:      Optional[float],
    net_margin:     Optional[float],
    gm_trend:       Optional[float],   # GM change YoY (positive = expanding)
    op_leverage:    Optional[float],   # op income growth / rev growth (> 1 = scaling)
    ebitda_margin:  Optional[float],
) -> Optional[float]:
    """
    Composite profitability [0, 1].
    Weights: GM 25%, OP 25%, NM 15%, GM_trend 15%, op_leverage 10%, EBITDA_margin 10%.
    """
    parts: list = []

    if gross_margin is not None:
        gm = _norm(gross_margin, 0.0, 1.0)
        if gm is not None:
            parts.append((gm, 0.25))

    if op_margin is not None:
        om = _norm(op_margin, -0.30, 0.50)
        if om is not None:
            parts.append((om, 0.25))

    if net_margin is not None:
        nm = _norm(net_margin, -0.20, 0.40)
        if nm is not None:
            parts.append((nm, 0.15))

    if gm_trend is not None:
        # Positive trend = expanding margins
        gt = _norm(gm_trend, -0.10, 0.10)
        if gt is not None:
            parts.append((gt, 0.15))

    if op_leverage is not None:
        # > 1 means operating income growing faster than revenue (good)
        ol = _norm(op_leverage, 0.0, 2.0)
        if ol is not None:
            parts.append((ol, 0.10))

    if ebitda_margin is not None:
        em = _norm(ebitda_margin, -0.10, 0.50)
        if em is not None:
            parts.append((em, 0.10))

    if not parts:
        return None
    total_w = sum(w for _, w in parts)
    return sum(s * w for s, w in parts) / total_w


def _value_raw(
    pe:        Optional[float],
    ev_ebitda: Optional[float],
    fcf_yield: Optional[float],
    peg:       Optional[float],
    ps:        Optional[float],   # fallback when pe/ev unavailable
) -> Optional[float]:
    """
    Composite value [0, 1].  Higher = cheaper vs peers.
    Weights: inv_PE 25%, inv_EV/EBITDA 25%, FCF yield 20%, inv_PEG 15%, inv_PS 15%.
    """
    parts: list = []

    if pe is not None and 0 < pe < 200:
        inv = _norm(pe, 5.0, 100.0)
        if inv is not None:
            parts.append((1.0 - inv, 0.25))

    if ev_ebitda is not None and 0 < ev_ebitda < 100:
        inv = _norm(ev_ebitda, 5.0, 40.0)
        if inv is not None:
            parts.append((1.0 - inv, 0.25))

    if fcf_yield is not None and -0.10 <= fcf_yield <= 0.30:
        fy = _norm(fcf_yield, -0.05, 0.15)
        if fy is not None:
            parts.append((fy, 0.20))

    if peg is not None and 0 < peg < 10:
        inv = _norm(peg, 0.0, 5.0)
        if inv is not None:
            parts.append((1.0 - inv, 0.15))

    if ps is not None and 0 < ps < 50:
        inv = _norm(ps, 0.5, 20.0)
        if inv is not None:
            parts.append((1.0 - inv, 0.15))

    if not parts:
        return None
    total_w = sum(w for _, w in parts)
    return sum(s * w for s, w in parts) / total_w


def _momentum_raw(
    ret_12m:       Optional[float],
    ret_6m:        Optional[float],
    ret_3m:        Optional[float],
    pct_vs_sma50:  Optional[float],
    pct_vs_200sma: Optional[float],
) -> Optional[float]:
    """
    Composite momentum [0, 1].  Higher = stronger upward momentum.
    Weights: 12M 30%, 6M 20%, 3M 15%, SMA50 15%, SMA200 15%.
    Note: 12M return is 12M-1M to avoid short-term reversal contamination.
    """
    parts: list = []

    if ret_12m is not None:
        m = _norm(ret_12m, -0.60, 0.80)
        if m is not None:
            parts.append((m, 0.30))

    if ret_6m is not None:
        m = _norm(ret_6m, -0.40, 0.50)
        if m is not None:
            parts.append((m, 0.20))

    if ret_3m is not None:
        m = _norm(ret_3m, -0.30, 0.40)
        if m is not None:
            parts.append((m, 0.15))

    if pct_vs_sma50 is not None:
        m = _norm(pct_vs_sma50, -0.20, 0.20)
        if m is not None:
            parts.append((m, 0.15))

    if pct_vs_200sma is not None:
        m = _norm(pct_vs_200sma, -0.30, 0.30)
        if m is not None:
            parts.append((m, 0.15))

    if not parts:
        return None
    total_w = sum(w for _, w in parts)
    return sum(s * w for s, w in parts) / total_w


def _growth_raw(
    rev_growth: Optional[float],
    eps_growth: Optional[float],
    fcf_growth: Optional[float],
    rev_accel:  Optional[float],   # rev growth this year minus last year
) -> Optional[float]:
    """
    Composite growth [0, 1].  Higher = faster / improving growth.
    Weights: rev_growth 30%, eps_growth 30%, FCF growth 20%, rev acceleration 20%.
    """
    parts: list = []

    if rev_growth is not None:
        g = _norm(rev_growth, -0.20, 0.50)
        if g is not None:
            parts.append((g, 0.30))

    if eps_growth is not None:
        g = _norm(eps_growth, -0.30, 0.80)
        if g is not None:
            parts.append((g, 0.30))

    if fcf_growth is not None:
        g = _norm(fcf_growth, -0.50, 1.00)
        if g is not None:
            parts.append((g, 0.20))

    if rev_accel is not None:
        g = _norm(rev_accel, -0.20, 0.20)
        if g is not None:
            parts.append((g, 0.20))

    if not parts:
        return None
    total_w = sum(w for _, w in parts)
    return sum(s * w for s, w in parts) / total_w


def _stability_raw(
    beta:          Optional[float],
    earnings_cv:   Optional[float],   # coefficient of variation of EPS
    max_drawdown:  Optional[float],   # worst peak-to-trough drawdown (negative)
    fcf_consistency: Optional[float], # fraction of years with positive FCF
    vol_30d:       Optional[float],   # 30-day annualised return vol
) -> Optional[float]:
    """
    Composite low-volatility / stability [0, 1].  Higher = more stable.
    Weights: inv_beta 25%, inv_earnings_CV 25%, inv_max_drawdown 20%,
             FCF_consistency 20%, inv_vol_30 10%.
    """
    parts: list = []

    if beta is not None and beta > 0:
        b = _norm(beta, 0.20, 2.50)
        if b is not None:
            parts.append((1.0 - b, 0.25))

    if earnings_cv is not None:
        cv = _norm(max(0.0, earnings_cv), 0.0, 1.0)
        if cv is not None:
            parts.append((1.0 - cv, 0.25))

    if max_drawdown is not None:
        # max_drawdown is negative, e.g. -0.35 = 35% peak-to-trough loss
        dd = _norm(abs(max_drawdown), 0.0, 0.80)
        if dd is not None:
            parts.append((1.0 - dd, 0.20))

    if fcf_consistency is not None:
        fc = _norm(max(0.0, min(1.0, fcf_consistency)), 0.0, 1.0)
        if fc is not None:
            parts.append((fc, 0.20))

    if vol_30d is not None and vol_30d > 0:
        v = _norm(vol_30d, 0.10, 0.80)
        if v is not None:
            parts.append((1.0 - v, 0.10))

    if not parts:
        return None
    total_w = sum(w for _, w in parts)
    return sum(s * w for s, w in parts) / total_w


def _macro_raw(
    sector:             Optional[str],
    rev_growth_vol:     Optional[float],  # std dev of revenue growth rates
    debt_equity:        Optional[float],  # proxy for yield sensitivity
    interest_coverage:  Optional[float],  # proxy for rate sensitivity
    eps_vol:            Optional[float],  # std dev of annual EPS
    fcf_positive_rate:  Optional[float],  # fraction of years with positive FCF
) -> Optional[float]:
    """
    Composite macro defensiveness [0, 1].  INVERTED: higher = more defensive.

    Weights:
      inv_sector_cyclicality 30%  — sector-based cyclicality lookup
      inv_rev_vol 25%             — low revenue growth volatility = defensive
      inv_yield_sensitivity 20%  — low D/E + high coverage = rate-insensitive
      inv_earn_vol 15%            — low EPS volatility = defensive
      capital_stability 10%       — FCF generation consistency
    """
    parts: list = []

    # Sector cyclicality — lookup
    if sector is not None:
        sec_key = sector.lower().strip()
        cyc = _SECTOR_CYCLICALITY.get(sec_key, _DEFAULT_SECTOR_CYCLICALITY)
        # high cyclicality (low cyc score) → low defensiveness score
        parts.append((cyc, 0.30))

    # Inverse revenue volatility — low rev_vol = stable = defensive
    if rev_growth_vol is not None:
        inv_rv = 1.0 - _norm(rev_growth_vol, 0.0, 0.30)
        if inv_rv is not None:
            parts.append((inv_rv, 0.25))

    # Yield / rate sensitivity — proxy: low D/E + high interest coverage
    if debt_equity is not None and interest_coverage is not None:
        de_score  = 1.0 - _norm(min(debt_equity, 5.0), 0.0, 5.0)
        ic_score  = _norm(min(interest_coverage, 20.0), 0.0, 20.0)
        if de_score is not None and ic_score is not None:
            rate_score = 0.5 * de_score + 0.5 * ic_score
            parts.append((rate_score, 0.20))
    elif debt_equity is not None:
        de_score = 1.0 - _norm(min(debt_equity, 5.0), 0.0, 5.0)
        if de_score is not None:
            parts.append((de_score, 0.20))
    elif interest_coverage is not None:
        ic_score = _norm(min(interest_coverage, 20.0), 0.0, 20.0)
        if ic_score is not None:
            parts.append((ic_score, 0.20))

    # Inverse earnings volatility
    if eps_vol is not None:
        inv_ev = 1.0 - _norm(eps_vol, 0.0, 5.0)
        if inv_ev is not None:
            parts.append((inv_ev, 0.15))

    # FCF capital stability
    if fcf_positive_rate is not None:
        fc = _norm(max(0.0, min(1.0, fcf_positive_rate)), 0.0, 1.0)
        if fc is not None:
            parts.append((fc, 0.10))

    if not parts:
        return None
    total_w = sum(w for _, w in parts)
    return sum(s * w for s, w in parts) / total_w


# ── Stock data extraction ─────────────────────────────────────────────────────

def _eps_coefficient_of_variation(income_statements: list) -> Optional[float]:
    """Compute CV (std/mean) of EPS diluted across available years."""
    eps_vals = [
        s.eps_diluted for s in income_statements
        if getattr(s, "eps_diluted", None) is not None
    ]
    if len(eps_vals) < 2:
        return None
    mu = sum(eps_vals) / len(eps_vals)
    if abs(mu) < 1e-9:
        return None
    std = math.sqrt(sum((e - mu) ** 2 for e in eps_vals) / len(eps_vals))
    return std / abs(mu)


def _compute_max_drawdown(closes: list) -> Optional[float]:
    """Compute max peak-to-trough drawdown from price series (newest first)."""
    if not closes or len(closes) < 10:
        return None
    prices = list(reversed(closes))   # oldest first for drawdown calculation
    peak  = prices[0]
    worst = 0.0
    for p in prices:
        if p > peak:
            peak = p
        dd = (p - peak) / peak
        if dd < worst:
            worst = dd
    return worst


def _compute_vol_30d(closes: list) -> Optional[float]:
    """30-day annualised daily return volatility."""
    if len(closes) < 31:
        return None
    rets = [
        (closes[i] - closes[i + 1]) / closes[i + 1]
        for i in range(30)
        if closes[i + 1] > 0
    ]
    if len(rets) < 10:
        return None
    mu  = sum(rets) / len(rets)
    var = sum((r - mu) ** 2 for r in rets) / len(rets)
    return math.sqrt(var) * math.sqrt(252)


def _compute_rev_growth_vol(income_statements: list) -> Optional[float]:
    """Standard deviation of YoY revenue growth rates."""
    revs = [
        s.revenue for s in income_statements
        if getattr(s, "revenue", None) is not None and s.revenue > 0
    ]
    if len(revs) < 3:
        return None
    growths = [
        (revs[i] - revs[i + 1]) / revs[i + 1]
        for i in range(len(revs) - 1)
    ]
    return _safe_std(growths)


def _compute_gm_trend(income_statements: list) -> Optional[float]:
    """YoY change in gross margin ratio."""
    gms = [
        getattr(s, "gross_profit_ratio", None) for s in income_statements[:2]
    ]
    if any(g is None for g in gms) or len(gms) < 2:
        return None
    return gms[0] - gms[1]


def _compute_op_leverage(income_statements: list) -> Optional[float]:
    """Operating income growth / revenue growth (≥1 = improving margins at scale)."""
    if len(income_statements) < 2:
        return None
    inc = income_statements
    rev_base = getattr(inc[1], "revenue", None)
    rev_curr = getattr(inc[0], "revenue", None)
    op_base  = getattr(inc[1], "operating_income", None)
    op_curr  = getattr(inc[0], "operating_income", None)
    if not all(v is not None for v in [rev_base, rev_curr, op_base, op_curr]):
        return None
    if rev_base <= 0 or op_base is None or op_base == 0:
        return None
    rev_g = (rev_curr - rev_base) / abs(rev_base)
    op_g  = (op_curr  - op_base)  / abs(op_base)
    if abs(rev_g) < 0.005:
        return None
    return op_g / rev_g


def _compute_fcf_growth(cash_flows: list) -> Optional[float]:
    """YoY FCF growth rate."""
    fcfs = [
        getattr(c, "free_cash_flow", None) for c in cash_flows[:2]
    ]
    if len(fcfs) < 2 or any(f is None for f in fcfs):
        return None
    if fcfs[1] == 0:
        return None
    return (fcfs[0] - fcfs[1]) / abs(fcfs[1])


def _compute_fcf_consistency(cash_flows: list) -> Optional[float]:
    """Fraction of years with positive FCF."""
    fcfs = [
        getattr(c, "free_cash_flow", None) for c in cash_flows
        if getattr(c, "free_cash_flow", None) is not None
    ]
    if not fcfs:
        return None
    return sum(1 for f in fcfs if f > 0) / len(fcfs)


def _extract_own_factors(stock_data: "StockData") -> dict:
    """
    Extract raw factor inputs from a StockData container.
    Returns a dict with keys for all seven factors.
    All values are floats, strings, or None.
    """
    r   = stock_data.latest_ratios
    ph  = stock_data.price_history
    inc = stock_data.income_statements
    cfs = getattr(stock_data, "cash_flows", []) or []

    # ── Quality inputs ────────────────────────────────────────────────────────
    roe  = getattr(r, "roe",              None) if r else None
    roic = getattr(r, "roic",             None) if r else None
    ic   = getattr(r, "interest_coverage", None) if r else None
    de   = getattr(r, "debt_to_equity",   None) if r else None
    eps_cv = _eps_coefficient_of_variation(inc)

    # ── Profitability inputs ──────────────────────────────────────────────────
    gm       = getattr(r, "gross_margin",    None) if r else None
    opm      = getattr(r, "operating_margin", None) if r else None
    nm       = getattr(r, "net_margin",      None) if r else None
    gm_trend = _compute_gm_trend(inc)
    op_lev   = _compute_op_leverage(inc)
    ebitda_margin: Optional[float] = None
    if len(inc) >= 1 and getattr(inc[0], "ebitda", None) and getattr(inc[0], "revenue", None):
        if inc[0].revenue > 0:
            ebitda_margin = inc[0].ebitda / inc[0].revenue

    # ── Value inputs ──────────────────────────────────────────────────────────
    pe       = getattr(r, "pe_ratio",    None) if r else None
    ps       = getattr(r, "ps_ratio",    None) if r else None
    ev       = getattr(r, "ev_to_ebitda", None) if r else None
    fcf_yld  = getattr(r, "fcf_yield",   None) if r else None
    peg      = None   # will be looked up from val_range if available

    # ── Momentum inputs ───────────────────────────────────────────────────────
    ret_12m = None
    ret_6m  = None
    ret_3m  = None
    pct_50  = None
    pct_200 = None
    if ph and ph.latest_price and ph.latest_price > 0:
        lp = ph.latest_price
        if ph.price_12m_ago and ph.price_12m_ago > 0:
            ret_12m = (lp - ph.price_12m_ago) / ph.price_12m_ago
        if ph.price_6m_ago and ph.price_6m_ago > 0:
            ret_6m = (lp - ph.price_6m_ago) / ph.price_6m_ago
        closes = ph.closes
        if closes and len(closes) >= 63:
            p3m = closes[min(63, len(closes) - 1)]
            if p3m > 0:
                ret_3m = (lp - p3m) / p3m
        if closes and len(closes) >= 50:
            sma50 = sum(closes[:50]) / 50.0
            if sma50 > 0:
                pct_50 = (closes[0] - sma50) / sma50
        if closes and len(closes) >= 200:
            sma200 = sum(closes[:200]) / 200.0
            if sma200 > 0:
                pct_200 = (closes[0] - sma200) / sma200

    # ── Growth inputs ─────────────────────────────────────────────────────────
    rev_growth = None
    eps_growth = None
    rev_accel  = None
    fcf_growth = _compute_fcf_growth(cfs)

    if len(inc) >= 2:
        if inc[0].revenue and inc[1].revenue and inc[1].revenue > 0:
            rev_growth = (inc[0].revenue - inc[1].revenue) / abs(inc[1].revenue)
        if inc[0].eps_diluted and inc[1].eps_diluted and inc[1].eps_diluted > 0:
            eps_growth = (inc[0].eps_diluted - inc[1].eps_diluted) / abs(inc[1].eps_diluted)
    if len(inc) >= 3:
        if inc[1].revenue and inc[2].revenue and inc[2].revenue > 0:
            rev_prior = (inc[1].revenue - inc[2].revenue) / abs(inc[2].revenue)
            if rev_growth is not None:
                rev_accel = rev_growth - rev_prior

    # ── Stability inputs ──────────────────────────────────────────────────────
    beta  = getattr(stock_data.profile, "beta", None) if stock_data.profile else None
    cr    = getattr(r, "current_ratio", None) if r else None
    closes_list = ph.closes if ph else []
    dd     = _compute_max_drawdown(closes_list)
    vol30  = _compute_vol_30d(closes_list)
    fcf_cons = _compute_fcf_consistency(cfs)

    # ── Macro inputs ──────────────────────────────────────────────────────────
    sector       = getattr(stock_data.profile, "sector", None) if stock_data.profile else None
    rev_vol      = _compute_rev_growth_vol(inc)
    eps_vol_abs: Optional[float] = None
    eps_vals = [
        s.eps_diluted for s in inc
        if getattr(s, "eps_diluted", None) is not None
    ]
    if len(eps_vals) >= 2:
        eps_vol_abs = _safe_std(eps_vals)

    return dict(
        # quality
        roe=roe, roic=roic, interest_coverage=ic, debt_equity=de, eps_cv=eps_cv,
        # profitability
        gross_margin=gm, op_margin=opm, net_margin=nm,
        gm_trend=gm_trend, op_leverage=op_lev, ebitda_margin=ebitda_margin,
        # value
        pe=pe, ps=ps, ev_ebitda=ev, fcf_yield=fcf_yld, peg=peg,
        # momentum
        ret_12m=ret_12m, ret_6m=ret_6m, ret_3m=ret_3m,
        pct_vs_sma50=pct_50, pct_vs_200sma=pct_200,
        # growth
        rev_growth=rev_growth, eps_growth=eps_growth,
        fcf_growth=fcf_growth, rev_accel=rev_accel,
        # stability/lowvol
        beta=beta, eps_cv_stab=eps_cv, max_drawdown=dd,
        fcf_consistency=fcf_cons, vol_30d=vol30,
        # macro
        sector=sector, rev_growth_vol=rev_vol,
        eps_vol=eps_vol_abs, fcf_positive_rate=fcf_cons,
        # backward-compat aliases (used by regression_calibration)
        roe_alias=roe, current_ratio=cr,
    )


def _extract_peer_factors(peer) -> dict:
    """
    Extract raw factor inputs from a peer row.
    Accepts either a dict (API response) or an object with attributes.
    """
    def _g(key):
        if isinstance(peer, dict):
            return peer.get(key)
        return getattr(peer, key, None)

    gm  = _g("gross_margin")
    opm = _g("operating_margin")
    nm  = _g("net_margin")
    roe = _g("roe")
    roic = _g("roic")
    ic  = _g("interest_coverage")

    pe  = _g("pe")
    ps  = _g("ps")
    ev  = _g("ev_ebitda")

    eps_growth = _g("eps_growth")
    if eps_growth is not None and abs(eps_growth) > 10:
        eps_growth = eps_growth / 100.0   # stored as percent, convert

    rev_growth = _g("revenue_growth")
    if rev_growth is not None and rev_growth > 1.5:
        rev_growth = rev_growth / 100.0

    ebitda_growth = _g("ebitda_growth")
    if ebitda_growth is not None and abs(ebitda_growth) > 2.0:
        ebitda_growth = ebitda_growth / 100.0

    beta = _g("beta")
    de   = _g("debt_equity")
    cr   = _g("current_ratio")

    # Sector not available on peer rows — macro raw will use None
    return dict(
        # quality
        roe=roe, roic=roic, interest_coverage=ic, debt_equity=de, eps_cv=None,
        # profitability
        gross_margin=gm, op_margin=opm, net_margin=nm,
        gm_trend=None, op_leverage=None, ebitda_margin=None,
        # value
        pe=pe, ps=ps, ev_ebitda=ev, fcf_yield=None, peg=_g("peg"),
        # momentum — peers don't have price history; EPS growth as proxy
        ret_12m=eps_growth, ret_6m=None, ret_3m=None,
        pct_vs_sma50=None, pct_vs_200sma=None,
        # growth
        rev_growth=rev_growth, eps_growth=eps_growth,
        fcf_growth=None, rev_accel=None,
        # stability
        beta=beta, eps_cv_stab=None, max_drawdown=None,
        fcf_consistency=None, vol_30d=None,
        # macro
        sector=None, rev_growth_vol=None,
        eps_vol=None, fcf_positive_rate=None,
        # compat
        roe_alias=roe, current_ratio=cr,
    )


# ── Composite and style scoring ───────────────────────────────────────────────

def _normalize_regime_key(regime: str) -> str:
    """Map free-form macro regime string to a _REGIME_WEIGHT_SHIFTS key."""
    r = regime.lower().replace(" ", "_").replace("-", "_")
    for k in _REGIME_WEIGHT_SHIFTS:
        if k in r or r in k:
            return k
    return "mid_cycle"


def _classify_style(
    quality_z:      float,
    value_z:        float,
    momentum_z:     float,
    growth_z:       float,
    stability_z:    float,
    profitability_z: float,
    macro_z:        float,
) -> str:
    """
    Assign a qualitative style archetype from the seven z-score quadrant.
    Priority order prevents double-assignment.
    """
    # Compounder: high quality + high profitability + decent growth
    if quality_z >= 1.0 and profitability_z >= 0.5 and growth_z >= 0.3:
        return "Compounder"
    # Quality Value: high quality + cheap
    if quality_z >= 0.7 and value_z >= 0.5:
        return "Quality Value"
    # High Growth: very strong growth, less stable
    if growth_z >= 1.2 and stability_z < 0.0:
        return "High Growth"
    # Value: cheap but lower quality
    if value_z >= 1.0 and quality_z < 0.3:
        return "Value"
    # Momentum: strong price momentum
    if momentum_z >= 1.2:
        return "Momentum"
    # Cyclical: low macro_z (cyclical) with below-avg quality
    if macro_z <= -0.8 and quality_z < 0.3:
        return "Cyclical"
    if quality_z < -0.3 and growth_z < -0.3:
        return "Cyclical"
    # Speculative: low quality, low stability
    if quality_z < -0.5 and stability_z < -0.5:
        return "Speculative"
    return "Blend"


def compute_composite_factor_score(
    scores:     Dict[str, Optional[float]],
    style:      str,
    macro_regime: str = "mid_cycle",
) -> tuple:
    """
    Apply regime weight shifts + archetype multipliers, then compute
    a safe weighted composite score.

    Returns (composite_score, effective_weights, strongest_factor, weakest_factor).
    composite_score is None if insufficient data.
    """
    rk = _normalize_regime_key(macro_regime)
    shifts      = _REGIME_WEIGHT_SHIFTS.get(rk, {})
    multipliers = _ARCHETYPE_MULTIPLIERS.get(style, _ARCHETYPE_MULTIPLIERS["Blend"])

    # 1. Apply additive regime shifts
    adj_weights: Dict[str, float] = {}
    for f in _FACTOR_NAMES:
        adj_weights[f] = max(0.0, _BASE_WEIGHTS[f] + shifts.get(f, 0.0))

    # 2. Normalise after clipping to ≥ 0
    total = sum(adj_weights.values())
    if total > 0:
        adj_weights = {f: w / total for f, w in adj_weights.items()}

    # 3. Apply archetype multipliers
    for f in _FACTOR_NAMES:
        adj_weights[f] *= multipliers.get(f, 1.0)

    # 4. Renormalise
    total = sum(adj_weights.values())
    if total > 0:
        eff_weights: Dict[str, float] = {f: w / total for f, w in adj_weights.items()}
    else:
        eff_weights = dict(_BASE_WEIGHTS)

    # 5. Compute composite via safe_weighted_composite
    composite = safe_weighted_composite(scores, eff_weights)

    # 6. Find strongest and weakest by weighted contribution
    contribs: Dict[str, float] = {}
    for f, w in eff_weights.items():
        s = scores.get(f)
        if s is not None:
            contribs[f] = w * s
    if contribs:
        strongest = max(contribs, key=lambda k: contribs[k])
        weakest   = min(contribs, key=lambda k: contribs[k])
    else:
        strongest = "quality"
        weakest   = "value"

    return composite, eff_weights, strongest, weakest


def _generate_pm_interpretation(
    composite:      Optional[float],
    scores:         Dict[str, Optional[float]],
    style:          str,
    macro_regime:   str,
) -> str:
    """
    One-sentence PM interpretation based on composite score and factor pattern.
    Pattern-matched in priority order.
    """
    if composite is None:
        return "Insufficient factor data — score based on scorecard signals alone."

    q  = scores.get("quality") or 50.0
    p  = scores.get("profitability") or 50.0
    g  = scores.get("growth") or 50.0
    v  = scores.get("value") or 50.0
    m  = scores.get("momentum") or 50.0
    lv = scores.get("lowvol") or 50.0
    c  = scores.get("macro") or 50.0

    rk = _normalize_regime_key(macro_regime)
    in_downturn = rk in ("recession", "slowdown")

    # Cyclical / macro alerts first (most actionable)
    if c < 30 and in_downturn:
        return (
            "High cyclical exposure in a contractionary regime — "
            "limit position size; macro headwind likely dominant."
        )
    if m >= 75 and p < 45:
        return (
            "Momentum run outpacing underlying profitability — "
            "tactical caution warranted; reduce on further momentum extension."
        )
    # Composite extremes
    if composite >= 78:
        return (
            f"Strong cross-factor tailwind ({style}) — "
            "multiple signals aligned; full allocation appropriate."
        )
    if composite <= 30:
        return (
            "Broad factor weakness — multiple headwinds present; "
            "undersize or avoid until signals recover."
        )
    # Quality/value tension
    if q >= 70 and v < 35:
        return (
            "High-quality business at elevated valuation — "
            "hold existing; new buyers should await a better entry."
        )
    if q >= 65 and v >= 60:
        return "Quality at a discount — favorable entry; risk/reward supports initiating."
    # Growth/profitability tension
    if g >= 70 and p < 35:
        return (
            "Strong growth with unproven profitability — "
            "size as a staged position; monitor burn rate."
        )
    if g >= 65 and p >= 65:
        return "Profitable growth — durable compounder profile; full allocation supported."
    # Momentum/fundamentals divergence
    if m < 30 and q >= 60:
        return (
            "Fundamental strength not yet reflected in price — "
            "patient accumulation; await momentum inflection."
        )
    if m >= 65 and q < 40:
        return "Price momentum unsupported by quality fundamentals — risk of reversal."
    # Defensive stance
    if lv >= 70 and composite >= 55:
        return "Defensive compounder profile — stable through cycles; suitable anchor position."
    # Generic
    if composite >= 60:
        return f"{style} — above-average composite with no dominant risk flag."
    return f"{style} — mixed factor signals; balanced allocation with active monitoring."


# ── Public entry point ────────────────────────────────────────────────────────

def build_factor_profile(
    stock_data:   "StockData",
    peer_rows:    list,
    macro_regime: str = "Unknown",
) -> FactorProfile:
    """
    Build a cross-sectional FactorProfile for stock_data vs the peer universe.

    Parameters
    ----------
    stock_data   : StockData — the stock being evaluated
    peer_rows    : list of PeerRow objects or dicts from the peer comparison
    macro_regime : str — current macro regime for composite weight shifts

    When fewer than 3 peers are available all z-scores are 0.0 (no relative
    signal) and confidence is "low".  The raw scores are still computed and
    used to derive factor scores via solo z-scoring against absolute anchors.
    """
    # ── Extract raw scores ────────────────────────────────────────────────────
    own = _extract_own_factors(stock_data)

    peers_raw: list = []
    for p in (peer_rows or []):
        try:
            peers_raw.append(_extract_peer_factors(p))
        except Exception:
            continue

    n_peers = len(peers_raw)

    # ── Compute normalised raw score for each entity ──────────────────────────
    def _raw_scores(f: dict) -> dict:
        return {
            "quality": _quality_raw(
                f["roe"], f.get("roic"), f.get("eps_cv"),
                f["interest_coverage"], f["debt_equity"],
            ),
            "profitability": _profitability_raw(
                f["gross_margin"], f["op_margin"], f["net_margin"],
                f.get("gm_trend"), f.get("op_leverage"), f.get("ebitda_margin"),
            ),
            "value": _value_raw(
                f["pe"], f["ev_ebitda"], f.get("fcf_yield"),
                f.get("peg"), f["ps"],
            ),
            "momentum": _momentum_raw(
                f["ret_12m"], f["ret_6m"], f.get("ret_3m"),
                f.get("pct_vs_sma50"), f.get("pct_vs_200sma"),
            ),
            "growth": _growth_raw(
                f["rev_growth"], f["eps_growth"],
                f.get("fcf_growth"), f.get("rev_accel"),
            ),
            "stability": _stability_raw(
                f["beta"], f.get("eps_cv_stab"), f.get("max_drawdown"),
                f.get("fcf_consistency"), f.get("vol_30d"),
            ),
            "macro": _macro_raw(
                f.get("sector"), f.get("rev_growth_vol"),
                f.get("debt_equity"), f.get("interest_coverage"),
                f.get("eps_vol"), f.get("fcf_positive_rate"),
            ),
        }

    own_scores  = _raw_scores(own)
    peer_scores = [_raw_scores(p) for p in peers_raw]

    # ── Cross-sectional z-scores with winsorisation ───────────────────────────
    if n_peers >= 3:
        universe_scores = [own_scores] + peer_scores
        confidence = "high" if n_peers >= 6 else "medium"

        def _uz(factor):
            return [s[factor] for s in universe_scores]

        def _wz(factor) -> float:
            """Winsorise universe values, then z-score own vs peers."""
            raw_universe = _uz(factor)
            wins = winsorize(raw_universe)
            own_wins = wins[0]  # own is always first element
            return _z_score(own_wins, wins)

        quality_z      = _wz("quality")
        profitability_z = _wz("profitability")
        value_z        = _wz("value")
        momentum_z     = _wz("momentum")
        growth_z       = _wz("growth")
        stability_z    = _wz("stability")
        macro_z        = _wz("macro")

    else:
        # Insufficient peers — solo z-score centred on 0.5 raw value
        def _solo_z(raw: Optional[float], lo: float = 0.20, hi: float = 0.80) -> float:
            if raw is None:
                return 0.0
            mid  = (lo + hi) / 2.0
            span = (hi - lo) / 4.0   # ±2σ covers lo–hi
            return max(-2.0, min(2.0, (raw - mid) / max(span, 1e-9)))

        quality_z       = _solo_z(own_scores["quality"])
        profitability_z = _solo_z(own_scores["profitability"])
        value_z         = _solo_z(own_scores["value"])
        momentum_z      = _solo_z(own_scores["momentum"])
        growth_z        = _solo_z(own_scores["growth"])
        stability_z     = _solo_z(own_scores["stability"])
        macro_z         = _solo_z(own_scores["macro"])
        confidence      = "low"

    # ── CDF scores (0-100) ────────────────────────────────────────────────────
    quality_factor      = z_to_score(quality_z)
    profitability_score = z_to_score(profitability_z)
    macro_score         = z_to_score(macro_z)

    # Factor score dict for composite (all mapped to 0-100)
    factor_scores: Dict[str, Optional[float]] = {
        "quality":       quality_factor,
        "profitability": profitability_score,
        "value":         z_to_score(value_z),
        "momentum":      z_to_score(momentum_z),
        "growth":        z_to_score(growth_z),
        "lowvol":        z_to_score(stability_z),
        "macro":         macro_score,
    }
    # Null out factors with no raw data (raw score was None)
    _raw_null = {k for k, v in own_scores.items() if v is None}
    _factor_map = {
        "quality": "quality", "profitability": "profitability",
        "value": "value",     "momentum": "momentum",
        "growth": "growth",   "stability": "lowvol",
        "macro": "macro",
    }
    for raw_k, factor_k in _factor_map.items():
        if raw_k in _raw_null:
            factor_scores[factor_k] = None

    # ── Style label ───────────────────────────────────────────────────────────
    style_label = _classify_style(
        quality_z, value_z, momentum_z, growth_z,
        stability_z, profitability_z, macro_z,
    )

    # ── Composite with regime + archetype weighting ───────────────────────────
    composite_score, eff_weights, strongest, weakest = compute_composite_factor_score(
        factor_scores, style_label, macro_regime,
    )
    composite_score = composite_score if composite_score is not None else 50.0

    # ── PM interpretation ─────────────────────────────────────────────────────
    pm_interp = _generate_pm_interpretation(
        composite_score, factor_scores, style_label, macro_regime,
    )

    print(
        f"  [FACTOR] style={style_label!r} n_peers={n_peers} regime={macro_regime!r}"
        f" composite={composite_score:.0f}"
        f" Q_z={quality_z:+.2f} P_z={profitability_z:+.2f}"
        f" V_z={value_z:+.2f} G_z={growth_z:+.2f}"
        f" M_z={momentum_z:+.2f} S_z={stability_z:+.2f}"
        f" C_z={macro_z:+.2f}"
        f" qfactor={quality_factor:.0f}"
        f" macro_score={macro_score:.0f}"
    )

    return FactorProfile(
        quality_z        = quality_z,
        value_z          = value_z,
        momentum_z       = momentum_z,
        growth_z         = growth_z,
        stability_z      = stability_z,
        profitability_z  = profitability_z,
        macro_z          = macro_z,
        quality_factor   = quality_factor,
        profitability_score = profitability_score,
        macro_score      = macro_score,
        composite_score  = composite_score,
        style_label      = style_label,
        strongest_factor = strongest,
        weakest_factor   = weakest,
        effective_weights = eff_weights,
        pm_interpretation = pm_interp,
        n_peers          = n_peers,
        confidence       = confidence,
    )
