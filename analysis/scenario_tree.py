"""
scenario_tree.py
================
Probabilistic narrative scenario tree for the alpha decision engine.

Architecture
------------
Three-level tree with four macro outcomes × three execution outcomes
× three multiple outcomes = up to 36 terminal leaves (pruned for
internal consistency; typically 18-24 material leaves remain).

  Level 1 — Macro outcome   (4 branches: re_acceleration / base / slowdown / recession)
  Level 2 — Execution       (3 branches: beat / inline / miss)
  Level 3 — Multiple        (3 branches: expansion / stable / compression)

Probability assignment
----------------------
  Macro prior:     Markov transition from current regime to forward scenario
  Execution:       Conditional P(exec | macro), adjusted for quality_z,
                   profitability_z, AR(1) rho, and margin trend slope
  Multiple rxn:    Conditional P(mult | macro, exec), adjusted for
                   value_z, current P/E, and HRL valuation MR speed

Price targets
-------------
  eps_growth_adj = macro_base × execution_mult, blended 70/30 with
                   HRL AR(1) estimate when available; earnings_trend applies ±5 pp
  multiple_adj   = absolute P/E turns from regime × multiple reaction table;
                   valuation leverage amplifies compression for expensive stocks

Outputs
-------
  ScenarioTree carries:
    · legacy bear/base/bull_price + shock_prob/shock_mean_growth (MC integration)
    · weighted_return, best_case, worst_case (PM narrative)
    · scenario_std, downside_mass, upside_mass, concentration_3 (position sizing)
    · var_95 (risk management)

Integration
-----------
    from analysis.scenario_tree import build_scenario_tree, infer_earnings_trend

    trend = infer_earnings_trend(stock_data)
    tree  = build_scenario_tree(
        macro_regime   = "Late_Cycle",
        earnings_trend = trend,
        current_pe     = 28.5,
        base_eps       = 4.20,
        current_price  = 112.0,
        factor_profile = fp,
        hrl_result     = hrl,
    )
    # tree.shock_prob, tree.shock_mean_growth  → mc_from_valuation_range()
    # tree.weighted_return, tree.scenario_std  → position sizing
    # tree.best_case, tree.worst_case          → PM commentary
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from models.stock_data import StockData
    from analysis.factor_model import FactorProfile
    from analysis.historical_regression import HRLResult


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioLeaf:
    """One terminal node in the scenario tree."""
    # Descriptive
    label:            str     # "Recession / Miss / Compression"
    path:             str     # "recession→miss→compression"  (legacy compat)
    macro_regime:     str     # "re_acceleration" | "base" | "slowdown" | "recession"
    execution:        str     # "beat" | "inline" | "miss"
    multiple_rxn:     str     # "expansion" | "stable" | "compression"

    # Probability
    probability:      float   # joint probability; all leaves sum to ≈ 1.0

    # Price / return
    eps_growth_adj:   float   # decimal (e.g. 0.12 = +12%)
    multiple_adj:     float   # absolute P/E turns added to current multiple
    target_price:     float   # terminal stock price
    expected_return:  float   # target_price / current_price − 1

    # Legacy aliases (kept for backward compat with MC layer)
    terminal_price:   float   # == target_price
    growth_rate:      float   # == eps_growth_adj
    earnings_change:  float   # == eps_growth_adj  (fractional EPS change)
    multiple_change:  float   # fractional multiple change (for compat)


@dataclass
class ScenarioTree:
    """
    Full scenario tree output.

    Legacy MC integration fields
    ----------------------------
    shock_prob        — P(recession ∩ miss); feeds GrowthDistParams.shock_prob
    shock_mean_growth — mean EPS growth across recession×miss leaves
    bear/base/bull_price / bear/base/bull_multiple — P10/mean/P90 aggregates

    PM narrative fields
    -------------------
    weighted_return   — Σ p_i × r_i  (probability-weighted expected return)
    best_case         — highest-return leaf
    worst_case        — lowest-return leaf

    Position sizing signals
    -----------------------
    scenario_std      — std dev of leaf returns (dispersion)
    downside_mass     — Σ p_i where return < −20%
    upside_mass       — Σ p_i where return > +20%
    concentration_3   — probability mass in top-3 highest-probability leaves

    Risk management
    ---------------
    var_95            — 5th-percentile leaf return (probability-weighted)
    """
    leaves:            List[ScenarioLeaf]

    # Legacy fields
    bear_price:        float
    base_price:        float
    bull_price:        float
    bear_multiple:     float
    base_multiple:     float
    bull_multiple:     float
    shock_prob:        float
    shock_mean_growth: float
    macro_regime:      str
    method:            str

    # PM / sizing / risk fields
    weighted_return:   float
    best_case:         Optional[ScenarioLeaf]
    worst_case:        Optional[ScenarioLeaf]
    var_95:            float
    scenario_std:      float
    concentration_3:   float
    upside_mass:       float
    downside_mass:     float


# ─────────────────────────────────────────────────────────────────────────────
# Probability tables
# ─────────────────────────────────────────────────────────────────────────────

# One-period Markov transition matrix: current regime → forward scenario probs
# Rows = current regime (normalised key); Cols = scenario probabilities
_MACRO_TRANSITION: Dict[str, Dict[str, float]] = {
    "re_acceleration": {
        "re_acceleration": 0.30, "base": 0.45, "slowdown": 0.20, "recession": 0.05,
    },
    "base": {
        "re_acceleration": 0.15, "base": 0.50, "slowdown": 0.28, "recession": 0.07,
    },
    "slowdown": {
        "re_acceleration": 0.08, "base": 0.25, "slowdown": 0.40, "recession": 0.27,
    },
    "recession": {
        "re_acceleration": 0.25, "base": 0.35, "slowdown": 0.25, "recession": 0.15,
    },
}

# Conditional P(execution | macro) — base rates before factor adjustments
_EXEC_BASE: Dict[str, Dict[str, float]] = {
    "re_acceleration": {"beat": 0.40, "inline": 0.42, "miss": 0.18},
    "base":            {"beat": 0.28, "inline": 0.48, "miss": 0.24},
    "slowdown":        {"beat": 0.16, "inline": 0.42, "miss": 0.42},
    "recession":       {"beat": 0.08, "inline": 0.28, "miss": 0.64},
}

# Conditional P(multiple reaction | macro, execution) — base rates
_MULT_BASE: Dict[Tuple[str, str], Dict[str, float]] = {
    ("re_acceleration", "beat"):   {"expansion": 0.60, "stable": 0.30, "compression": 0.10},
    ("re_acceleration", "inline"): {"expansion": 0.30, "stable": 0.55, "compression": 0.15},
    ("re_acceleration", "miss"):   {"expansion": 0.10, "stable": 0.35, "compression": 0.55},
    ("base",            "beat"):   {"expansion": 0.40, "stable": 0.45, "compression": 0.15},
    ("base",            "inline"): {"expansion": 0.18, "stable": 0.58, "compression": 0.24},
    ("base",            "miss"):   {"expansion": 0.08, "stable": 0.32, "compression": 0.60},
    ("slowdown",        "beat"):   {"expansion": 0.20, "stable": 0.40, "compression": 0.40},
    ("slowdown",        "inline"): {"expansion": 0.08, "stable": 0.35, "compression": 0.57},
    ("slowdown",        "miss"):   {"expansion": 0.03, "stable": 0.17, "compression": 0.80},
    ("recession",       "beat"):   {"expansion": 0.10, "stable": 0.35, "compression": 0.55},
    ("recession",       "inline"): {"expansion": 0.03, "stable": 0.22, "compression": 0.75},
    ("recession",       "miss"):   {"expansion": 0.01, "stable": 0.09, "compression": 0.90},
}

# EPS growth base assumptions by macro regime (decimal)
_MACRO_EPS_BASE: Dict[str, float] = {
    "re_acceleration": 0.15,
    "base":            0.07,
    "slowdown":        0.01,
    "recession":      -0.12,
}

# Execution multiplier on macro EPS base
_EXEC_MULT: Dict[str, float] = {"beat": 1.40, "inline": 1.00, "miss": 0.55}

# Absolute P/E turns added to current multiple by (macro, multiple_rxn)
_MULT_ADJ_PE: Dict[Tuple[str, str], float] = {
    ("re_acceleration", "expansion"):   +3.5,
    ("re_acceleration", "stable"):      +0.5,
    ("re_acceleration", "compression"): -2.0,
    ("base",            "expansion"):   +2.0,
    ("base",            "stable"):       0.0,
    ("base",            "compression"): -2.5,
    ("slowdown",        "expansion"):   +1.0,
    ("slowdown",        "stable"):      -1.0,
    ("slowdown",        "compression"): -3.5,
    ("recession",       "expansion"):   +0.5,
    ("recession",       "stable"):      -2.0,
    ("recession",       "compression"): -5.5,
}

# P/S equivalent adjustments (×0.18 scale relative to P/E turns)
_MULT_ADJ_PS: Dict[Tuple[str, str], float] = {
    ("re_acceleration", "expansion"):   +0.60,
    ("re_acceleration", "stable"):      +0.10,
    ("re_acceleration", "compression"): -0.35,
    ("base",            "expansion"):   +0.35,
    ("base",            "stable"):       0.00,
    ("base",            "compression"): -0.45,
    ("slowdown",        "expansion"):   +0.15,
    ("slowdown",        "stable"):      -0.20,
    ("slowdown",        "compression"): -0.65,
    ("recession",       "expansion"):   +0.10,
    ("recession",       "stable"):      -0.35,
    ("recession",       "compression"): -1.00,
}


# ─────────────────────────────────────────────────────────────────────────────
# Regime normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _regime_key(regime: str) -> str:
    """
    Map a free-form macro regime string to one of the four scenario keys
    or to a _MACRO_TRANSITION row key.
    """
    r = regime.lower().replace(" ", "_").replace("-", "_")
    # Exact / partial match to transition table keys first
    for k in ("re_acceleration", "slowdown", "recession", "base"):
        if k in r:
            return k
    # Map legacy regime names from factor_model / monte_carlo
    if r in ("expansion", "recovery", "early_cycle", "mid_cycle"):
        return "base"
    if r in ("late_cycle", "contraction"):
        return "slowdown"
    return "base"


# ─────────────────────────────────────────────────────────────────────────────
# Probability helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(d: Dict[str, float]) -> Dict[str, float]:
    total = sum(d.values())
    if total < 1e-9:
        return d
    return {k: v / total for k, v in d.items()}


def _exec_probs(
    macro:          str,
    factor_profile: Optional[object],
    hrl_result:     Optional[object],
    earnings_trend: str,
) -> Dict[str, float]:
    """
    Compute conditional P(execution | macro), adjusted for company quality
    and earnings momentum signals.
    """
    probs = dict(_EXEC_BASE.get(macro, _EXEC_BASE["base"]))

    beat_mult = 1.0
    miss_mult = 1.0

    # ── Factor profile adjustments ────────────────────────────────────────────
    if factor_profile is not None:
        qz = getattr(factor_profile, "quality_z",      0.0) or 0.0
        pz = getattr(factor_profile, "profitability_z", 0.0) or 0.0

        # Quality: each σ lifts beat by ~10%, suppresses miss by ~8%
        beat_mult *= 1.0 + 0.10 * max(-1.5, min(1.5, qz))
        miss_mult *= 1.0 - 0.08 * max(-1.5, min(1.5, qz))

        # Profitability: thin-margin companies miss more in stress
        if pz < -1.5:
            beat_mult *= 0.75
            miss_mult *= 1.40
        elif pz > 1.0:
            beat_mult *= 1.10
            miss_mult *= 0.90

    # ── HRL adjustments ───────────────────────────────────────────────────────
    if hrl_result is not None:
        rho   = getattr(hrl_result, "ar1_eps_persistence", 0.65) or 0.65
        slope = getattr(hrl_result, "margin_trend_slope",  0.0)  or 0.0

        # Persistent earners (high rho) beat more consistently
        if rho > 0.80:
            beat_mult *= 1.15; miss_mult *= 0.85
        elif rho < 0.40:
            beat_mult *= 0.90; miss_mult *= 1.15

        # Rising margins → structural beat tendency
        if slope > 0.01:
            beat_mult *= 1.15; miss_mult *= 0.85
        elif slope < -0.01:
            beat_mult *= 0.80; miss_mult *= 1.30

    # ── Earnings trend adjustment ─────────────────────────────────────────────
    if earnings_trend == "accelerating":
        beat_mult *= 1.12; miss_mult *= 0.88
    elif earnings_trend == "decelerating":
        beat_mult *= 0.88; miss_mult *= 1.15

    probs["beat"] *= beat_mult
    probs["miss"] *= miss_mult
    return _normalise(probs)


def _mult_probs(
    macro:          str,
    execution:      str,
    current_pe:     Optional[float],
    factor_profile: Optional[object],
    hrl_result:     Optional[object],
) -> Dict[str, float]:
    """
    Compute conditional P(multiple_rxn | macro, execution), adjusted for
    current valuation and company-specific signals.
    """
    base = dict(_MULT_BASE.get((macro, execution), {"expansion": 0.15, "stable": 0.55, "compression": 0.30}))

    exp_mult  = 1.0
    comp_mult = 1.0

    # ── Valuation level ───────────────────────────────────────────────────────
    if current_pe is not None:
        if current_pe > 35:
            exp_mult *= 0.70; comp_mult *= 1.40
        elif current_pe > 25:
            exp_mult *= 0.85; comp_mult *= 1.20
        elif current_pe < 12:
            exp_mult *= 1.30; comp_mult *= 0.70
        elif current_pe < 18:
            exp_mult *= 1.15; comp_mult *= 0.85

    # ── Factor profile ────────────────────────────────────────────────────────
    if factor_profile is not None:
        vz  = getattr(factor_profile, "value_z",    0.0) or 0.0
        mz  = getattr(factor_profile, "momentum_z", 0.0) or 0.0

        # Cheap stocks re-rate more easily; expensive stocks compress harder
        if vz < -1.5:   # expensive
            exp_mult  *= 0.60; comp_mult *= 1.50
        elif vz < -0.8:
            exp_mult  *= 0.80; comp_mult *= 1.20
        elif vz > 1.5:  # cheap / undervalued
            exp_mult  *= 1.40; comp_mult *= 0.65
        elif vz > 0.8:
            exp_mult  *= 1.20; comp_mult *= 0.85

        # Strong momentum reduces compression probability on beat
        if mz > 1.5 and execution == "beat":
            exp_mult  *= 1.15; comp_mult *= 0.90

    # ── HRL: valuation mean-reversion speed ───────────────────────────────────
    if hrl_result is not None:
        kappa = getattr(hrl_result, "valuation_mr_speed", 0.15) or 0.15
        if kappa > 0.40:   # historically fast-reverting multiple
            exp_mult  *= 0.85; comp_mult *= 1.30
        elif kappa < 0.08: # historically sticky multiple
            exp_mult  *= 1.10; comp_mult *= 0.90

    probs = {
        "expansion":   base["expansion"]   * exp_mult,
        "stable":      base["stable"],
        "compression": base["compression"] * comp_mult,
    }
    return _normalise(probs)


# ─────────────────────────────────────────────────────────────────────────────
# EPS growth and price target per leaf
# ─────────────────────────────────────────────────────────────────────────────

def _leaf_eps_growth(
    macro:          str,
    execution:      str,
    earnings_trend: str,
    hrl_result:     Optional[object],
) -> float:
    """
    Compute eps_growth_adj for a leaf.
    Base = macro_eps_base × execution_mult.
    Blended 70% with HRL AR(1) estimate when available (30%).
    Earnings trend applies ±5 pp to base before blend.

    Sign-aware execution multipliers
    ---------------------------------
    When macro_eps_base is negative (slowdown/recession), a "beat" means
    earnings declined LESS than feared (multiplier < 1.0) and a "miss"
    means earnings declined MORE than feared (multiplier > 1.0).
    Multipliers are therefore flipped relative to the positive-base case.
    """
    macro_base = _MACRO_EPS_BASE.get(macro, 0.05)

    if macro_base < 0:
        # Negative base: beat → less negative (0.55×), miss → more negative (1.40×)
        _exec_mult_effective = {
            "beat":   _EXEC_MULT["miss"],    # 0.55 → smaller decline
            "inline": _EXEC_MULT["inline"],  # 1.00
            "miss":   _EXEC_MULT["beat"],    # 1.40 → larger decline
        }
    else:
        _exec_mult_effective = _EXEC_MULT

    base_g = macro_base * _exec_mult_effective.get(execution, 1.0)

    # Earnings trend nudge on base (before HRL blend)
    if earnings_trend == "accelerating":
        base_g += 0.05
    elif earnings_trend == "decelerating":
        base_g -= 0.06

    # HRL blend
    if hrl_result is not None:
        ar1_g = getattr(hrl_result, "ar1_growth_estimate", None)
        slope = getattr(hrl_result, "margin_trend_slope", 0.0) or 0.0
        if ar1_g is not None:
            base_g = 0.70 * base_g + 0.30 * ar1_g
        # Margin slope adds operating-leverage-adjusted EPS impact
        base_g += slope * 8.0

    return max(-0.70, min(1.50, base_g))


def _leaf_target_price(
    macro:         str,
    multiple_rxn:  str,
    current_price: float,
    current_pe:    Optional[float],
    base_eps:      Optional[float],
    eps_growth:    float,
    method:        str,
    current_ps:    Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Compute (target_price, multiple_adj, multiple_change_frac) for a leaf.

    For P/E: target = forward_eps × (current_pe + pe_turn_adj)
    For P/S: target = current_price × (1 + eps_growth) × ps_adj_factor
    Fallback: target = current_price × (1 + eps_growth × 0.80)

    Returns (target_price, multiple_adj_absolute, multiple_change_fractional).
    """
    if method == "P/E" and base_eps is not None and base_eps > 0 and current_pe is not None:
        pe_adj = _MULT_ADJ_PE.get((macro, multiple_rxn), 0.0)

        # Valuation leverage: expensive stocks compress harder
        if multiple_rxn == "compression" and current_pe > 20:
            lev    = min(2.0, current_pe / 20.0)
            pe_adj = pe_adj * lev

        fwd_eps   = base_eps * (1.0 + eps_growth)
        target_pe = max(4.0, current_pe + pe_adj)
        price     = max(fwd_eps * target_pe, current_price * 0.05)
        frac      = pe_adj / current_pe if current_pe > 0 else 0.0
        return price, pe_adj, frac

    elif method == "P/S" and current_ps is not None and current_ps > 0:
        ps_adj = _MULT_ADJ_PS.get((macro, multiple_rxn), 0.0)
        if multiple_rxn == "compression" and current_ps > 5:
            lev    = min(2.0, current_ps / 5.0)
            ps_adj = ps_adj * lev
        target_ps = max(0.3, current_ps + ps_adj)
        price     = current_price * (1.0 + eps_growth) * (target_ps / current_ps)
        price     = max(price, current_price * 0.05)
        frac      = ps_adj / current_ps if current_ps > 0 else 0.0
        return price, ps_adj, frac

    else:
        # Fallback: apply blended growth + partial multiple signal to price
        pe_adj = _MULT_ADJ_PE.get((macro, multiple_rxn), 0.0)
        frac   = pe_adj / (current_pe or 20.0)
        price  = current_price * (1.0 + eps_growth * 0.80 + frac * 0.05)
        price  = max(price, current_price * 0.05)
        return price, pe_adj, frac


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _price_percentile(leaves: List[ScenarioLeaf], p: float) -> float:
    """Probability-weighted p-th percentile of terminal prices."""
    if not leaves:
        return 0.0
    srt = sorted(leaves, key=lambda l: l.terminal_price)
    cum = 0.0
    for leaf in srt:
        cum += leaf.probability
        if cum >= p:
            return leaf.terminal_price
    return srt[-1].terminal_price


def _return_percentile(leaves: List[ScenarioLeaf], p: float) -> float:
    """Probability-weighted p-th percentile of expected returns."""
    if not leaves:
        return 0.0
    srt = sorted(leaves, key=lambda l: l.expected_return)
    cum = 0.0
    for leaf in srt:
        cum += leaf.probability
        if cum >= p:
            return leaf.expected_return
    return srt[-1].expected_return


def _weighted_std(
    leaves: List[ScenarioLeaf],
    mean:   float,
) -> float:
    """Probability-weighted standard deviation of expected returns."""
    if len(leaves) < 2:
        return 0.0
    variance = sum(l.probability * (l.expected_return - mean) ** 2 for l in leaves)
    return math.sqrt(max(0.0, variance))


# ─────────────────────────────────────────────────────────────────────────────
# Earnings trend inference (unchanged from prior version)
# ─────────────────────────────────────────────────────────────────────────────

def infer_earnings_trend(stock_data: "StockData") -> str:
    """
    Infer whether earnings momentum is accelerating, stable, or decelerating.

    Uses the last 3 years of annual EPS to compute two sequential YoY rates;
    if the most recent rate exceeds the prior by > 3 pp → "accelerating";
    if it's lower by > 3 pp → "decelerating"; else "stable".

    Falls back to revenue if EPS is unreliable (negative bases).
    """
    inc = getattr(stock_data, "income_statements", []) or []

    def _yoy(series, attr):
        rates = []
        for i in range(len(series) - 1):
            curr = getattr(series[i],   attr, None)
            prev = getattr(series[i+1], attr, None)
            if curr is not None and prev is not None and prev > 0:
                rates.append((curr - prev) / abs(prev))
        return rates

    rates = _yoy(inc[:4], "eps_diluted")
    if len(rates) < 2:
        rates = _yoy(inc[:4], "eps")
    if len(rates) < 2:
        rates = _yoy(inc[:4], "revenue")

    if len(rates) >= 2:
        delta = rates[0] - rates[1]   # most recent minus prior
        if delta > 0.03:
            return "accelerating"
        if delta < -0.03:
            return "decelerating"
    return "stable"


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_scenario_tree(
    macro_regime:   str,
    earnings_trend: str            = "stable",
    current_pe:     Optional[float] = None,
    base_eps:       Optional[float] = None,
    current_price:  Optional[float] = None,
    factor_profile: Optional[object] = None,   # FactorProfile
    hrl_result:     Optional[object] = None,   # HRLResult
    current_ps:     Optional[float] = None,    # P/S for revenue-based path
) -> Optional[ScenarioTree]:
    """
    Build a probabilistic scenario tree (up to 36 leaves, pruned).

    Parameters
    ----------
    macro_regime   : current macro regime string (normalised internally)
    earnings_trend : "accelerating" | "stable" | "decelerating"
                     (use infer_earnings_trend() to derive from StockData)
    current_pe     : trailing P/E; used for P/E method
    base_eps       : base-case EPS; required for P/E method
    current_price  : current stock price (anchor for returns and P/S fallback)
    factor_profile : FactorProfile (adjusts branch probabilities)
    hrl_result     : HRLResult (AR(1) blend, margin slope, MR speed)
    current_ps     : P/S ratio for revenue-path pricing

    Returns None when no price anchor is available.
    """
    anchor = current_price or (
        base_eps * current_pe if (base_eps and base_eps > 0 and current_pe) else None
    )
    if anchor is None or anchor <= 0:
        return None

    # Choose valuation method
    if base_eps is not None and base_eps > 0 and current_pe is not None:
        method = "P/E"
    elif current_ps is not None and current_ps > 0:
        method = "P/S"
    else:
        method = "price"

    rk = _regime_key(macro_regime)

    # ── Macro prior ────────────────────────────────────────────────────────────
    macro_probs = dict(_MACRO_TRANSITION.get(rk, _MACRO_TRANSITION["base"]))

    # Momentum_z shifts macro prior slightly toward expansion / contraction
    if factor_profile is not None:
        mz = getattr(factor_profile, "momentum_z", 0.0) or 0.0
        adj = max(-0.06, min(0.06, mz * 0.025))
        macro_probs["re_acceleration"] = max(0.01, macro_probs["re_acceleration"] + adj)
        macro_probs["recession"]       = max(0.01, macro_probs["recession"]       - adj)
    macro_probs = _normalise(macro_probs)

    # ── Build leaves ───────────────────────────────────────────────────────────
    leaves: List[ScenarioLeaf] = []

    for macro, p_macro in macro_probs.items():
        ep = _exec_probs(macro, factor_profile, hrl_result, earnings_trend)
        for execution, p_exec in ep.items():
            mp = _mult_probs(macro, execution, current_pe, factor_profile, hrl_result)
            for multiple_rxn, p_mult in mp.items():

                joint_p = p_macro * p_exec * p_mult
                if joint_p < 0.002:   # prune negligible leaves
                    continue

                eps_g              = _leaf_eps_growth(macro, execution, earnings_trend, hrl_result)
                target_px, pe_adj, frac = _leaf_target_price(
                    macro, multiple_rxn, anchor,
                    current_pe, base_eps, eps_g, method, current_ps,
                )
                ret = target_px / anchor - 1.0

                leaves.append(ScenarioLeaf(
                    label           = f"{macro.replace('_', '-').title()} / {execution.title()} / {multiple_rxn.title()}",
                    path            = f"{macro}→{execution}→{multiple_rxn}",
                    macro_regime    = macro,
                    execution       = execution,
                    multiple_rxn    = multiple_rxn,
                    probability     = joint_p,
                    eps_growth_adj  = eps_g,
                    multiple_adj    = pe_adj,
                    target_price    = target_px,
                    expected_return = ret,
                    # Legacy aliases
                    terminal_price  = target_px,
                    growth_rate     = eps_g,
                    earnings_change = eps_g,
                    multiple_change = frac,
                ))

    if not leaves:
        return None

    # Renormalise (pruning removes some probability mass)
    total_p = sum(l.probability for l in leaves)
    for leaf in leaves:
        leaf.probability /= total_p

    # Sort by probability descending
    leaves.sort(key=lambda l: l.probability, reverse=True)

    # ── Aggregation ────────────────────────────────────────────────────────────
    base_price     = sum(l.probability * l.terminal_price for l in leaves)
    bear_price     = _price_percentile(leaves, 0.10)
    bull_price     = _price_percentile(leaves, 0.90)
    weighted_ret   = sum(l.probability * l.expected_return for l in leaves)
    scenario_std   = _weighted_std(leaves, weighted_ret)
    var_95         = _return_percentile(leaves, 0.05)
    concentration3 = sum(l.probability for l in leaves[:3])
    upside_mass    = sum(l.probability for l in leaves if l.expected_return > 0.20)
    downside_mass  = sum(l.probability for l in leaves if l.expected_return < -0.20)

    best_case  = max(leaves, key=lambda l: l.expected_return)
    worst_case = min(leaves, key=lambda l: l.expected_return)

    # ── Implied multiples ──────────────────────────────────────────────────────
    def _implied_mult(price: float) -> float:
        if method == "P/E" and base_eps and base_eps > 0:
            return price / base_eps
        elif current_pe and anchor > 0:
            return current_pe * (price / anchor)
        return current_pe or 20.0

    base_mult = _implied_mult(base_price)
    bear_mult = _implied_mult(bear_price)
    bull_mult = _implied_mult(bull_price)

    # ── Shock cluster: recession × miss leaves ─────────────────────────────────
    shock_leaves = [
        l for l in leaves
        if l.macro_regime == "recession" and l.execution == "miss"
    ]
    shock_prob = sum(l.probability for l in shock_leaves)
    shock_mean_growth = (
        sum(l.probability * l.eps_growth_adj for l in shock_leaves) / shock_prob
        if shock_prob > 0 else -0.10
    )

    # ── Logging ────────────────────────────────────────────────────────────────
    bull_pct  = (bull_price / anchor - 1.0) * 100
    bear_pct  = (bear_price / anchor - 1.0) * 100
    print(
        f"  [TREE] regime={rk} trend={earnings_trend} n_leaves={len(leaves)}"
        f" E[R]={weighted_ret:+.1%} std={scenario_std:.1%}"
        f" bear=${bear_price:.1f}({bear_pct:+.0f}%)"
        f" bull=${bull_price:.1f}({bull_pct:+.0f}%)"
        f" shock_p={shock_prob:.0%}"
        f" down_mass={downside_mass:.0%} up_mass={upside_mass:.0%}"
    )
    print(
        f"  [TREE] best={best_case.label}({best_case.expected_return:+.0%})"
        f" worst={worst_case.label}({worst_case.expected_return:+.0%})"
        f" conc3={concentration3:.0%} var95={var_95:.0%}"
    )

    return ScenarioTree(
        leaves             = leaves,
        bear_price         = bear_price,
        base_price         = base_price,
        bull_price         = bull_price,
        bear_multiple      = bear_mult,
        base_multiple      = base_mult,
        bull_multiple      = bull_mult,
        shock_prob         = shock_prob,
        shock_mean_growth  = shock_mean_growth,
        macro_regime       = macro_regime,
        method             = method,
        weighted_return    = weighted_ret,
        best_case          = best_case,
        worst_case         = worst_case,
        var_95             = var_95,
        scenario_std       = scenario_std,
        concentration_3    = concentration3,
        upside_mass        = upside_mass,
        downside_mass      = downside_mass,
    )
