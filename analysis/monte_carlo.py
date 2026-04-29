"""
MonteCarlo
==========
Stochastic equity valuation engine for probabilistic price-target
and portfolio-decision analysis.

Design
------
Growth distribution — asymmetric mixture:
  g ~ (1 − p_shock) × SplitNormal(μ, σ_down, σ_up)
     +       p_shock × Normal(μ_shock, σ_shock)

  SplitNormal is a two-piece normal: σ_down controls the downside half,
  σ_up the upside half.  High-quality companies (GM ≥ 55% or OP ≥ 20%)
  get σ_down < σ_up (right-skewed); cyclicals get the inverse.  p_shock
  adds an explicit earnings-miss / macro-contraction scenario.

Exit multiple distribution — mean-reverting Beta:
  x ~ Beta(α, β) mapped onto [multiple_bear, multiple_bull]

  The Beta mean is set to (current + mr_speed × (fair − current)) where
  fair = growth_rate × 1.5 (PEG=1.5 anchor).  High multiples bias the
  distribution toward compression; low multiples toward expansion.
  Concentration α + β controls spread: sticky (quality) companies use
  higher concentration (tighter); cyclicals use lower (wider).
  A macro rate-adjustment shifts the fair multiple before reversion:
  late-cycle → compress; recession → expand.

Regime and quality adjustments:
  Both distributions are modulated by macro_regime and company quality
  (inferred from gross_margin / operating_margin).  The reporting agent
  re-runs MC with the actual macro regime after the fundamental agent
  completes (initial run uses "Unknown" as a neutral default).

Simulations run in pure Python (stdlib only): no numpy/scipy dependency.
  random.betavariate, random.gauss, and random.triangular are all stdlib.
  10 000 paths complete in ~200–350 ms on a modern CPU.

Integration
-----------
    from analysis.monte_carlo import mc_from_valuation_range
    vr = compute_valuation_range(stock_data, metrics)
    vr.mc = mc_from_valuation_range(vr)        # attach in-place

Portfolio sizing signal
-----------------------
    mc.kelly_fraction       — half-Kelly position size fraction [0, 10%]
    mc.upside_downside      — abs(P95 return / P5 return); > 2.5 = right-skewed
    mc.prob_loss_20         — P(loss > 20%); > 0.30 → reduce position one tier
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Optional


# ── Deterministic seed derivation ─────────────────────────────────────────────

def _content_seed(*values: float) -> int:
    """
    Derive a deterministic 32-bit RNG seed from a sequence of numeric inputs.

    Uses SHA-256 so the same inputs always produce the same seed, independent
    of PYTHONHASHSEED (which randomises Python's built-in hash() for strings
    across interpreter launches).

    Rule: same driver inputs → same seed → same MC path sequence →
    byte-identical percentile outputs across consecutive runs.

    Usage:
        rng = random.Random(_content_seed(price, eps, g_base, pe_lo, pe_hi))
    """
    blob = "|".join(f"{v:.8g}" for v in values).encode("ascii")
    digest = hashlib.sha256(blob).digest()
    return int.from_bytes(digest[:4], "little")


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class MCResult:
    """Full output of a single Monte Carlo run."""

    # ── Simulation metadata ───────────────────────────────────────────────────
    n_sims:        int    # number of paths simulated
    horizon_years: int    # time horizon in years
    method:        str    # "P/E" | "P/S" — primary valuation method used

    # ── Growth assumption (carried for report transparency) ───────────────────
    growth_mean:   float  # base EPS / revenue CAGR, decimal (e.g. 0.12 = 12%)
    growth_std:    float  # growth uncertainty, decimal

    # ── Return distribution ───────────────────────────────────────────────────
    mean_return:   float  # arithmetic mean return
    median_return: float  # 50th percentile return
    p5_return:     float  # 5th percentile — severe downside
    p25_return:    float  # 25th percentile
    p75_return:    float  # 75th percentile
    p95_return:    float  # 95th percentile — strong upside

    skewness:      float  # 3rd standardised moment of return distribution

    # ── Outcome probabilities ─────────────────────────────────────────────────
    prob_positive: float  # P(R > 0)
    prob_20_gain:  float  # P(R > 20%)
    prob_loss:     float  # P(R < 0)
    prob_loss_20:  float  # P(R < −20%) — severe loss probability

    # ── Price distribution (absolute) ─────────────────────────────────────────
    mean_price:    float
    p5_price:      float
    p25_price:     float
    median_price:  float
    p75_price:     float
    p95_price:     float

    # ── Risk-adjusted sizing parameters ──────────────────────────────────────
    kelly_fraction:  float  # half-Kelly fraction (clamped [0, 0.10])
    upside_downside: float  # abs(p95_return / p5_return); > 2.5 = right-skewed

    # ── Derived labels (not serialised as separate fields) ────────────────────
    # These are computed from the numeric fields above and used in the frontend.
    # Access via mc.risk_label / mc.upside_skew_label if you need them server-side.

    @property
    def upside_skew_label(self) -> str:
        r = self.upside_downside
        if r >= 3.0:  return "strongly right-skewed"
        if r >= 2.0:  return "right-skewed"
        if r >= 0.8:  return "roughly symmetric"
        return "left-skewed"

    @property
    def risk_label(self) -> str:
        """Qualitative P(>20% loss) tier: 'high' | 'moderate' | 'low'."""
        if self.prob_loss_20 >= 0.30: return "high"
        if self.prob_loss_20 >= 0.15: return "moderate"
        return "low"


# ── Distribution conviction profile ──────────────────────────────────────────

@dataclass
class DistributionProfile:
    """
    Structured conviction assessment derived from a MCResult distribution.

    Evaluates four orthogonal dimensions of the return distribution and
    collapses them into a single conviction score [0–100], a tier label,
    a continuous size adjustment, and a one-sentence rationale.

    Dimensions
    ──────────
    return_score  (40%)  — attractiveness of E[R] vs an 8% hurdle
    skew_score    (30%)  — asymmetry (upside/downside ratio + 3rd moment)
    risk_score    (30%)  — tail containment + distribution tightness

    Width tiers
    ───────────
    iqr < 20%  → "tight"    high conviction; outcomes cluster near the mean
    iqr < 40%  → "moderate"
    iqr ≥ 40%  → "wide"     low conviction; outcomes are widely scattered

    Conviction tiers
    ────────────────
    ≥ 75  very_high  full allocation appropriate
    ≥ 60  high       standard Buy sizing
    ≥ 45  moderate   staged entry recommended
    ≥ 30  low        reduced position
    < 30  very_low   avoid or minimum tracking position
    """
    # ── Shape metrics ─────────────────────────────────────────────────────────
    iqr:           float   # p75_return − p25_return (interquartile range)
    width_tier:    str     # "tight" | "moderate" | "wide"
    net_prob:      float   # prob_positive − prob_loss ∈ [−1, 1]

    # ── Component scores (0–100) ──────────────────────────────────────────────
    return_score:   float  # E[R] attractiveness vs hurdle
    skew_score:     float  # upside asymmetry composite
    risk_score:     float  # tail safety + distribution tightness

    # ── Composite conviction ──────────────────────────────────────────────────
    conviction_score: float  # weighted composite [0–100]
    conviction_tier:  str    # "very_high" | "high" | "moderate" | "low" | "very_low"

    # ── Position guidance ─────────────────────────────────────────────────────
    size_adjustment: float   # additive percentage-point shift (+/−1.5 pp max)
    size_cap:        float   # hard position cap from tail risk (inf = no cap)

    # ── Explanation ───────────────────────────────────────────────────────────
    rationale: str           # one-sentence synthesis of the dominant driver


def distribution_profile(mc: "MCResult", hurdle: float = 0.08) -> DistributionProfile:
    """
    Compute a DistributionProfile from a completed MCResult.

    Parameters
    ----------
    mc      : MCResult from run_monte_carlo()
    hurdle  : minimum acceptable annual return (default 8% ≈ long-run equity premium)

    The function evaluates the FULL distribution shape — not just E[R] —
    and converts it into a conviction score, tier, and size guidance.
    """

    def _norm(x: float, lo: float, hi: float) -> float:
        """Linearly normalise x ∈ [lo, hi] → [0, 100], clamped."""
        if hi <= lo:
            return 50.0
        return max(0.0, min(100.0, (x - lo) / (hi - lo) * 100.0))

    # ── Shape metrics ──────────────────────────────────────────────────────────
    iqr      = mc.p75_return - mc.p25_return
    net_prob = mc.prob_positive - mc.prob_loss
    width_tier = (
        "tight"    if iqr < 0.20 else
        "moderate" if iqr < 0.40 else
        "wide"
    )

    # ── Return score (40%) ─────────────────────────────────────────────────────
    # Maps E[R] − hurdle from [−15pp, +25pp] → [0, 100]
    # Below hurdle penalty; above 25% = excellent
    return_score = _norm(mc.mean_return - hurdle, -0.15, 0.25)

    # ── Skew score (30%) ───────────────────────────────────────────────────────
    # Upside/downside ratio (50%): measures P95/|P5| asymmetry
    ud_score = _norm(mc.upside_downside, 0.50, 4.0)
    # 3rd moment skewness (30%): negative skew penalised, positive rewarded
    sk_score = _norm(mc.skewness, -2.0, 2.0)
    # Probability of gain (20%): breadth of the winning scenario set
    pg_score = _norm(mc.prob_positive, 0.40, 0.80)
    skew_score = 0.50 * ud_score + 0.30 * sk_score + 0.20 * pg_score

    # ── Risk score (30%) ───────────────────────────────────────────────────────
    # Tail safety: 1 − P(loss > 20%) maps [0.5, 1.0] → [0, 100]
    tail_score  = _norm(1.0 - mc.prob_loss_20, 0.50, 1.00)
    # Width penalty: IQR normalised to [0, 60%], inverted (tight = high score)
    width_score = _norm(1.0 - min(1.0, iqr / 0.60), 0.0, 1.0)
    risk_score  = 0.50 * tail_score + 0.50 * width_score

    # ── Conviction composite ───────────────────────────────────────────────────
    conviction_score = (
        0.40 * return_score +
        0.30 * skew_score   +
        0.30 * risk_score
    )

    if conviction_score >= 75:  conviction_tier = "very_high"
    elif conviction_score >= 60: conviction_tier = "high"
    elif conviction_score >= 45: conviction_tier = "moderate"
    elif conviction_score >= 30: conviction_tier = "low"
    else:                        conviction_tier = "very_low"

    # ── Size adjustment ────────────────────────────────────────────────────────
    # Continuous: (score − 50) / 50 → [−1, +1], scaled to ±1.5 pp
    size_adjustment = ((conviction_score - 50.0) / 50.0) * 1.5

    # P5 hard caps — override when severe tail risk exists
    if mc.p5_return < -0.30:
        size_cap = 1.0
    elif mc.p5_return < -0.20:
        size_cap = 2.0
    elif mc.p5_return < -0.10:
        size_cap = 3.0
    else:
        size_cap = float("inf")

    # ── Rationale — synthesise the dominant dimension ─────────────────────────
    score_deviations = [
        ("return",  return_score  - 50.0, return_score),
        ("skew",    skew_score    - 50.0, skew_score),
        ("risk",    risk_score    - 50.0, risk_score),
    ]
    # Pick the dimension with the largest deviation from neutral (50)
    dominant_dim, dominant_dev, dominant_score = max(
        score_deviations, key=lambda x: abs(x[1])
    )

    _er_pct  = f"~{mc.mean_return * 100:.0f}%"
    _p5_pct  = f"~{mc.p5_return  * 100:.0f}%"
    _p95_pct = f"~{mc.p95_return * 100:.0f}%"
    _iqr_pct = f"~{iqr * 100:.0f}%"
    _ud_str  = f"{mc.upside_downside:.1f}×"

    _tier_label = {
        "very_high": "Very high",
        "high":      "High",
        "moderate":  "Moderate",
        "low":       "Low",
        "very_low":  "Very low",
    }[conviction_tier]

    if conviction_score >= 60:
        if dominant_dim == "return":
            rationale = (
                f"{_tier_label} conviction — attractive expected return ({_er_pct}) "
                f"with {width_tier} distribution (IQR {_iqr_pct}) "
                f"and {_ud_str} upside/downside skew"
            )
        elif dominant_dim == "skew":
            rationale = (
                f"{_tier_label} conviction — right-skewed distribution "
                f"(P5 {_p5_pct} / P95 {_p95_pct}, skew {_ud_str}) "
                f"and E[R] {_er_pct}"
            )
        else:  # risk
            rationale = (
                f"{_tier_label} conviction — contained tail risk "
                f"(P5 {_p5_pct}, P(loss>20%)={mc.prob_loss_20:.0%}) "
                f"with {width_tier} distribution and E[R] {_er_pct}"
            )
    elif conviction_score >= 45:
        rationale = (
            f"Moderate conviction — mixed distribution "
            f"(E[R] {_er_pct}, IQR {_iqr_pct}, P5 {_p5_pct}); "
            f"staged entry reduces timing risk"
        )
    else:
        if dominant_dim == "risk" or size_cap < float("inf"):
            rationale = (
                f"{_tier_label} conviction — elevated tail risk "
                f"(P5 {_p5_pct}, P(loss>20%)={mc.prob_loss_20:.0%}); "
                f"distribution too wide (IQR {_iqr_pct}) to justify full size"
            )
        elif dominant_dim == "skew":
            rationale = (
                f"{_tier_label} conviction — left-skewed distribution "
                f"({_ud_str} skew, P95 {_p95_pct} vs P5 {_p5_pct}); "
                f"limited upside optionality"
            )
        else:
            rationale = (
                f"{_tier_label} conviction — unattractive risk/reward "
                f"(E[R] {_er_pct}, P(gain)={mc.prob_positive:.0%}, "
                f"P5 {_p5_pct})"
            )

    return DistributionProfile(
        iqr             = iqr,
        width_tier      = width_tier,
        net_prob        = net_prob,
        return_score    = return_score,
        skew_score      = skew_score,
        risk_score      = risk_score,
        conviction_score = conviction_score,
        conviction_tier  = conviction_tier,
        size_adjustment  = size_adjustment,
        size_cap         = size_cap,
        rationale        = rationale,
    )


# ── Risk classification ───────────────────────────────────────────────────────

@dataclass
class RiskClassification:
    """
    Distinguishes valuation-driven downside from structural downside.

    For high-quality companies (strong profitability + healthy balance sheet)
    tail risk in the MC distribution often reflects multiple compression —
    a temporary, mean-reverting phenomenon.  Standard P5 hard caps penalise
    these unfairly.  For weak-quality companies, the same tail risk may reflect
    genuine business deterioration (structural, potentially permanent loss).

    This classification relaxes P5 caps for quality companies while tightening
    them for weaker ones, keyed on the scorecard's profitability and financial
    health category scores.
    """
    risk_type:    str    # "valuation_driven" | "structural" | "mixed"
    quality_tier: str    # "high" | "average" | "weak"
    confidence:   str    # "high" | "medium" | "low"
    size_cap:     float  # quality-adjusted position cap (inf = no cap)
    cap_source:   str    # short description of why this cap applies
    explanation:  str    # one-sentence rationale for the classification decision


def classify_downside_risk(
    mc: "MCResult",
    profitability_score: float,
    health_score: float,
    flag_count: int = 0,
) -> "RiskClassification":
    """
    Classify the nature of downside risk and return a quality-adjusted size cap.

    Parameters
    ----------
    mc                  : completed MCResult from run_monte_carlo()
    profitability_score : scorecard category score 0–100 for profitability
    health_score        : scorecard category score 0–100 for financial health
    flag_count          : number of risk flags raised (each adds 5 pt penalty,
                          max 15 pts) — tilts toward structural classification

    Quality tiers
    ─────────────
    high    adj_quality ≥ 65 — durable margins and strong balance sheet;
            downside most likely multiple compression (temporary)
    average 40 < adj_quality < 65 — standard sizing thresholds apply
    weak    adj_quality ≤ 40 — fundamentals fragile; tail risk may be permanent

    Adjusted size caps by quality
    ─────────────────────────────
    Valuation-driven (high quality):
      P5 < −30%  → 2.5%  (default 1.0%)
      P5 < −20%  → 3.5%  (default 2.0%)
      P5 < −10%  → no cap (default 3.0%)

    Structural (weak quality):
      P5 < −10%  → 2.0%
      P5 < −20%  → 1.5%
      P5 < −30%  → 1.0%  (same as default)

    Mixed / average: standard thresholds unchanged.
    """
    # ── Quality tier ──────────────────────────────────────────────────────────
    quality_avg  = (profitability_score + health_score) / 2.0
    flag_penalty = min(flag_count * 5, 15)
    adj_quality  = quality_avg - flag_penalty

    if adj_quality >= 65:
        quality_tier = "high"
    elif adj_quality <= 40:
        quality_tier = "weak"
    else:
        quality_tier = "average"

    # ── Risk type ─────────────────────────────────────────────────────────────
    # upside_downside > 1.5 means upside outweighs downside — distribution still
    # right-skewed even when tail risk exists → consistent with valuation compression
    ud = mc.upside_downside
    p5 = mc.p5_return

    if quality_tier == "high" and ud >= 1.5:
        risk_type  = "valuation_driven"
        confidence = "high" if adj_quality >= 72 and ud >= 2.0 else "medium"
    elif quality_tier == "weak":
        risk_type  = "structural"
        confidence = "high" if adj_quality <= 35 else "medium"
    elif quality_tier == "high" and ud < 1.5:
        # High quality but unfavourable skew — inconclusive
        risk_type  = "mixed"
        confidence = "medium"
    elif quality_tier == "average" and ud >= 2.5:
        # Good distribution shape offsets average fundamentals
        risk_type  = "mixed"
        confidence = "medium"
    else:
        risk_type  = "mixed"
        confidence = "low"

    # ── Quality-adjusted size caps ────────────────────────────────────────────
    _p5_pct = f"{p5 * 100:.0f}%"

    if risk_type == "valuation_driven":
        if p5 < -0.30:
            size_cap    = 2.5
            cap_source  = f"P5≈{_p5_pct} (valuation-driven; relaxed cap 2.5%)"
            explanation = (
                f"Tail risk (P5≈{_p5_pct}) in a high-quality business likely reflects "
                f"multiple compression rather than business deterioration; "
                f"cap relaxed to 2.5% (vs 1.0% structural default)."
            )
        elif p5 < -0.20:
            size_cap    = 3.5
            cap_source  = f"P5≈{_p5_pct} (valuation-driven; relaxed cap 3.5%)"
            explanation = (
                f"Downside tail (P5≈{_p5_pct}) likely reflects valuation multiple compression "
                f"in a high-quality business; cap relaxed to 3.5%."
            )
        elif p5 < -0.10:
            size_cap    = float("inf")
            cap_source  = "no cap — valuation-driven tail, high quality"
            explanation = (
                f"Moderate tail (P5≈{_p5_pct}) in a high-quality business is consistent "
                f"with normal multiple volatility; no hard cap applied."
            )
        else:
            size_cap    = float("inf")
            cap_source  = "no cap — tail risk benign"
            explanation = (
                f"Minimal tail risk (P5≈{_p5_pct}) in a high-quality business; "
                f"distribution fully supports standard sizing."
            )

    elif risk_type == "structural":
        if p5 < -0.30:
            size_cap    = 1.0
            cap_source  = f"P5≈{_p5_pct} (structural risk; hard cap 1.0%)"
            explanation = (
                f"Severe tail risk (P5≈{_p5_pct}) in a business with weak fundamentals "
                f"signals structural downside; hard cap 1.0% to limit permanent-loss exposure."
            )
        elif p5 < -0.20:
            size_cap    = 1.5
            cap_source  = f"P5≈{_p5_pct} (structural risk; cap 1.5%)"
            explanation = (
                f"Elevated tail risk (P5≈{_p5_pct}) combined with weak fundamentals "
                f"indicates structural rather than temporary downside; position capped at 1.5%."
            )
        elif p5 < -0.10:
            size_cap    = 2.0
            cap_source  = f"P5≈{_p5_pct} (structural risk; cap 2.0%)"
            explanation = (
                f"Moderate tail risk (P5≈{_p5_pct}) with weak quality signals "
                f"potential structural deterioration; position capped at 2.0%."
            )
        else:
            size_cap    = float("inf")
            cap_source  = "no cap — tail risk contained"
            explanation = (
                f"Tail risk contained (P5≈{_p5_pct}) even with weaker fundamentals; "
                f"no additional position cap from risk classification."
            )

    else:  # mixed / average — standard thresholds
        if p5 < -0.30:
            size_cap    = 1.0
            cap_source  = f"P5≈{_p5_pct} (mixed profile; standard cap 1.0%)"
            explanation = (
                f"Severe tail risk (P5≈{_p5_pct}) with mixed quality profile; "
                f"standard 1.0% cap applied pending clearer risk attribution."
            )
        elif p5 < -0.20:
            size_cap    = 2.0
            cap_source  = f"P5≈{_p5_pct} (mixed profile; standard cap 2.0%)"
            explanation = (
                f"Elevated tail risk (P5≈{_p5_pct}) with mixed quality signal; "
                f"standard 2.0% cap applied."
            )
        elif p5 < -0.10:
            size_cap    = 3.0
            cap_source  = f"P5≈{_p5_pct} (mixed profile; soft cap 3.0%)"
            explanation = (
                f"Moderate tail (P5≈{_p5_pct}); mixed quality signals; "
                f"soft cap at 3.0%."
            )
        else:
            size_cap    = float("inf")
            cap_source  = "no cap — tail risk contained"
            explanation = (
                f"Tail risk contained (P5≈{_p5_pct}); "
                f"no cap from risk classification."
            )

    return RiskClassification(
        risk_type    = risk_type,
        quality_tier = quality_tier,
        confidence   = confidence,
        size_cap     = size_cap,
        cap_source   = cap_source,
        explanation  = explanation,
    )


# ── Core Compounder classification ────────────────────────────────────────────

@dataclass
class CoreCompounderProfile:
    """
    Identifies high-quality businesses that warrant a minimum portfolio floor.

    Core Compounders are durable businesses with top-quartile profitability,
    strong financial health, stable-to-growing revenue, and no structural red
    flags.  They must maintain at least a 1.5% allocation even when short-term
    signals (momentum weakness, valuation risk) would otherwise push below.

    Overrides are still respected when:
    · P/E > 100x (extreme valuation — no compounding thesis at any price)
    · risk_type == "structural" (business deterioration disqualifies)
    """
    is_core_compounder: bool       # True when all criteria are met
    criteria_met:       list       # human-readable list of satisfied criteria
    criteria_failed:    list       # human-readable list of failed criteria
    floor_applied:      bool       # True when position was raised to floor
    floor_size:         float      # floor value (1.5 if qualified, 0.0 otherwise)
    extreme_val_cap:    bool       # True when PE > 100x overrides the floor
    tag:                str        # "Core Compounder Allocation" | ""
    explanation:        str        # one-sentence rationale for the classification


def classify_core_compounder(
    profitability_score:  float,
    health_score:         float,
    growth_score:         "float | None" = None,
    flag_count:           int = 0,
    risk_type:            str = "mixed",
    pe_val:               "float | None" = None,
) -> "CoreCompounderProfile":
    """
    Classify a business as a Core Compounder and compute its minimum floor.

    Parameters
    ----------
    profitability_score : scorecard category score 0–100 (incorporates margins
                          and returns on capital — ROE/ROIC signal is embedded)
    health_score        : scorecard financial_health category score 0–100
    growth_score        : scorecard growth category score 0–100; None = missing
    flag_count          : number of active risk flags from the scorecard
    risk_type           : from classify_downside_risk — "valuation_driven" |
                          "structural" | "mixed"
    pe_val              : trailing P/E ratio; None if unavailable

    Criteria (all five must pass to qualify)
    ─────────────────────────────────────────
    1. profitability_score ≥ 65   top-quartile margins + strong returns (ROE/ROIC proxy)
    2. health_score ≥ 60          durable balance sheet, low leverage
    3. growth_score ≥ 50          stable or growing revenue (revenue model durability)
    4. flag_count ≤ 1             minimal structural risk flags
    5. risk_type ≠ "structural"   downside is multiple-compression, not business deterioration

    Floor logic
    ───────────
    · Qualified → 1.5% minimum regardless of momentum/valuation step-downs
    · Still capped at 1.5% (no increase) when PE > 100x — extreme valuation
      overrides the compounding thesis but does not penalise below the floor
    · risk_type == "structural" → disqualified entirely (criteria 5 fails)
    """
    criteria_met    : list = []
    criteria_failed : list = []

    # ── Criterion 1: Profitability ────────────────────────────────────────────
    if profitability_score >= 65:
        criteria_met.append(f"High profitability (score {profitability_score:.0f}/100)")
    else:
        criteria_failed.append(
            f"Profitability below threshold ({profitability_score:.0f}/100, need ≥65)"
        )

    # ── Criterion 2: Financial health ────────────────────────────────────────
    if health_score >= 60:
        criteria_met.append(f"Strong financial health (score {health_score:.0f}/100)")
    else:
        criteria_failed.append(
            f"Financial health below threshold ({health_score:.0f}/100, need ≥60)"
        )

    # ── Criterion 3: Revenue model durability ────────────────────────────────
    if growth_score is None:
        # Missing growth data → treat as borderline pass (don't disqualify on absence)
        criteria_met.append("Revenue model durability: data insufficient — assumed stable")
    elif growth_score >= 50:
        criteria_met.append(f"Durable/growing revenue model (growth score {growth_score:.0f}/100)")
    else:
        criteria_failed.append(
            f"Revenue model weakening (growth score {growth_score:.0f}/100, need ≥50)"
        )

    # ── Criterion 4: No structural risk flags ────────────────────────────────
    if flag_count <= 1:
        criteria_met.append(f"Minimal risk flags ({flag_count})")
    else:
        criteria_failed.append(
            f"Too many risk flags ({flag_count}, max 1 for core compounder)"
        )

    # ── Criterion 5: Downside is not structural ───────────────────────────────
    if risk_type != "structural":
        criteria_met.append(
            f"Downside risk classified as {risk_type} (not structural deterioration)"
        )
    else:
        criteria_failed.append(
            "Risk type is structural — business deterioration disqualifies core status"
        )

    is_core_compounder = len(criteria_failed) == 0

    # ── Extreme valuation override (PE > 100x) ───────────────────────────────
    extreme_val_cap = pe_val is not None and pe_val > 100.0

    # ── Floor and tag ─────────────────────────────────────────────────────────
    if is_core_compounder and not extreme_val_cap:
        floor_size  = 1.5
        tag         = "Core Compounder Allocation"
        explanation = (
            "Despite valuation-driven downside risk, high business quality "
            "supports maintaining core exposure — minimum 1.5% floor applied."
        )
    elif is_core_compounder and extreme_val_cap:
        floor_size  = 1.5   # floor still applies — PE>100x cap is also 1.5%
        tag         = "Core Compounder Allocation"
        explanation = (
            "Core Compounder profile confirmed, but extreme valuation (P/E>100×) "
            "prevents sizing above 1.5%; floor and cap converge at 1.5%."
        )
    else:
        floor_size  = 0.0
        tag         = ""
        explanation = (
            f"Does not qualify as Core Compounder: {'; '.join(criteria_failed)}."
        )

    return CoreCompounderProfile(
        is_core_compounder = is_core_compounder,
        criteria_met       = criteria_met,
        criteria_failed    = criteria_failed,
        floor_applied      = False,      # updated by caller when target is raised
        floor_size         = floor_size,
        extreme_val_cap    = extreme_val_cap,
        tag                = tag,
        explanation        = explanation,
    )


# ── Growth distribution (regime-aware asymmetric mixture) ────────────────────

@dataclass
class GrowthDistParams:
    """
    Fully-specified parameters for the asymmetric mixture growth distribution.

    Architecture
    ─────────────
    g ~ (1 − shock_prob) × SplitNormal(growth_mean, sigma_down, sigma_up)
       +       shock_prob × Normal(shock_mean, shock_std)

    SplitNormal (two-piece normal) is asymmetric about the mode:
      · sigma_down < sigma_up  →  right-skewed  (high-quality compounders)
      · sigma_down > sigma_up  →  left-skewed   (cyclicals, late-cycle)

    The shock component models low-probability severe outcomes:
    earnings misses, macro contractions, guidance cuts.
    """
    growth_mean:  float   # regime-adjusted central tendency
    sigma_down:   float   # downside std dev (left half of split normal)
    sigma_up:     float   # upside std dev (right half of split normal)
    shock_prob:   float   # P(drawing from shock component); 0–0.50
    shock_mean:   float   # shock scenario mean growth (typically deeply negative)
    shock_std:    float   # shock scenario std dev
    quality_tier: str     # "high" | "average" | "cyclical"
    macro_regime: str     # raw regime string for audit trail


# ── Regime parameter table ────────────────────────────────────────────────────
# (p_shock_add, mu_adj, sigma_scale, skew_adj)
#   p_shock_add  — added to quality-tier base shock probability
#   mu_adj       — shift on growth_mean in decimal (e.g. -0.02 = −2 pp)
#   sigma_scale  — multiplier on base growth_std before two-piece split
#   skew_adj     — additive adjustment to composite skew (+ = more right-skew)
_REGIME_TABLE: dict[str, tuple[float, float, float, float]] = {
    "expansion":   (0.00,  0.02, 1.00,  0.10),
    "recovery":    (0.00,  0.02, 1.10,  0.10),
    "early_cycle": (0.00,  0.02, 1.20,  0.15),
    "mid_cycle":   (0.00,  0.00, 1.00,  0.00),
    "late_cycle":  (0.08, -0.02, 1.10, -0.10),
    "slowdown":    (0.12, -0.03, 1.20, -0.15),
    "contraction": (0.20, -0.05, 1.30, -0.25),
    "recession":   (0.25, -0.08, 1.40, -0.35),
}

# ── Quality parameter table ───────────────────────────────────────────────────
# (sigma_down_mult, sigma_up_mult, shock_prob_base, skew_base)
#   sigma_down_mult  — applied to adj_std for left half
#   sigma_up_mult    — applied to adj_std for right half
#   shock_prob_base  — base P(shock) before regime add
#   skew_base        — base composite skew (+ = right-skewed)
_QUALITY_TABLE: dict[str, tuple[float, float, float, float]] = {
    "high":     (0.60, 1.30, 0.10,  0.20),  # recurring/high-margin: tight down, wide up
    "average":  (1.00, 1.00, 0.15,  0.00),  # symmetric
    "cyclical": (1.40, 0.85, 0.25, -0.15),  # wide down, compressed up
}


def _regime_key(regime: str) -> str:
    """Normalise a free-form macro regime string to a _REGIME_TABLE key."""
    r = regime.lower().replace(" ", "_").replace("-", "_")
    for k in _REGIME_TABLE:
        if k in r or r in k:
            return k
    return "mid_cycle"


def _infer_quality_tier(
    gross_margin: Optional[float],
    op_margin:    Optional[float],
) -> str:
    """
    Derive quality tier from profitability margins.

      High:     gross_margin ≥ 55% OR op_margin ≥ 20%
                (software, pharma, luxury — sticky margins, recurring economics)
      Cyclical: gross_margin < 30% OR op_margin < 5%
                (materials, autos, commodities — thin or volatile margins)
      Average:  everything else
    """
    high = (gross_margin is not None and gross_margin >= 0.55) or \
           (op_margin   is not None and op_margin   >= 0.20)
    low  = (gross_margin is not None and gross_margin <  0.30) or \
           (op_margin   is not None and op_margin   <  0.05)
    if high:
        return "high"
    if low:
        return "cyclical"
    return "average"


# ── Exit multiple distribution (mean-reverting Beta) ─────────────────────────

@dataclass
class MultipleDistParams:
    """
    Parameters for the mean-reverting Beta exit multiple distribution.

    Architecture
    ─────────────
    x ~ Beta(α, β) mapped onto [low, high]

    The Beta mean is set to a mean-reverted value between the current multiple
    and the growth-implied fair multiple.  The concentration c = α + β controls
    the spread: high c → tight (high-quality sticky multiples), low c → wide.

    Mean reversion: reverted = current + mr_speed × (fair − current)
      · If current > fair: distribution biased toward compression
      · If current < fair: distribution biased toward expansion

    Rate adjustment shifts the fair multiple before reversion is applied:
      · Rising rates (late-cycle): compress fair by rate_adj fraction
      · Falling rates (recession): expand fair

    Correlation with growth (ρ):
      The Beta mean shifts by ρ × Z_growth_corr × corr_sensitivity × span
      where Z_growth_corr is a standard normal correlated at ρ with the
      growth Z-score for that path.  This captures the empirical tendency
      for multiple expansion when growth beats expectations and compression
      when growth disappoints — within a single path.
    """
    low:              float   # hard lower bound = bear multiple
    high:             float   # hard upper bound = bull multiple
    current:          float   # current multiple (= base scenario)
    fair:             float   # growth-implied fair multiple (mean reversion anchor)
    mr_speed:         float   # mean reversion speed [0, 1]
    concentration:    float   # Beta α + β; higher = tighter distribution
    rate_adj:         float   # fractional shift on fair before reversion (−0.15 to +0.10)
    correlation_rho:  float   # growth-multiple correlation ρ ∈ [0, 1]
    corr_sensitivity: float   # multiple shift per unit of corr Z × span fraction
    quality_tier:     str     # "high" | "average" | "cyclical"
    macro_regime:     str     # raw regime string for audit trail


# ── Multiple quality table ────────────────────────────────────────────────────
# (mr_speed, concentration)
#   mr_speed:      0 = no reversion toward fair; 1 = full reversion in one period
#   concentration: α + β for Beta; 3 = wide, 6 = moderate, 10 = tight
_MULT_QUALITY_TABLE: dict[str, tuple[float, float]] = {
    "high":     (0.15, 6.0),   # sticky multiples — slow compression, tighter spread
    "average":  (0.30, 4.0),   # moderate reversion, moderate spread
    "cyclical": (0.45, 3.0),   # fast compression, wide spread
}

# ── Regime → rate-environment adjustment on the fair multiple ─────────────────
# Positive = lower rates implied → expand fair multiple
# Negative = higher rates implied → compress fair multiple
# Rule of thumb: each 100 bps change ≈ 1–2 P/E turns
_REGIME_RATE_ADJ: dict[str, float] = {
    "expansion":   -0.04,   # rates may be rising; moderate compression
    "recovery":    +0.04,   # rates low; supports expansion
    "early_cycle": +0.03,
    "mid_cycle":    0.00,
    "late_cycle":  -0.08,   # rates elevated; meaningful compression
    "slowdown":    -0.06,
    "contraction": -0.05,   # rates high but near peak
    "recession":   +0.06,   # rates falling; expansion
}


# ── Growth-multiple correlation table ────────────────────────────────────────
# (rho_pe, rho_ps, sensitivity)
#   rho_pe:      Pearson ρ between growth Z-score and multiple draw for P/E
#   rho_ps:      same for P/S (revenue multiples are less growth-sensitive)
#   sensitivity: fraction of the multiple span shifted per unit of correlated Z
#                e.g. 0.07 × span=20 → 1.4 P/E turns per unit of Z_corr
#
# Calibration rationale:
#   Cyclicals:  highest ρ — market swings from fear to greed on earnings direction
#   High qual:  moderate ρ — premium is sticky; expansion/compression is slower
#   Average:    intermediate
_CORR_TABLE: dict[str, tuple[float, float, float]] = {
    "high":     (0.45, 0.30, 0.06),
    "average":  (0.40, 0.28, 0.07),
    "cyclical": (0.55, 0.38, 0.09),
}


def _fair_multiple(growth_mean: float, method: str) -> float:
    """
    Growth-implied fair multiple using a PEG=1.5 anchor for P/E,
    and an analogous revenue-growth heuristic for P/S.

    PEG = P/E ÷ growth_rate_pct → fair P/E = growth_rate_pct × 1.5
    Bounds: P/E [10, 50]; P/S [0.5, 15]
    """
    g_pct = growth_mean * 100.0
    if method == "P/E":
        return max(10.0, min(50.0, g_pct * 1.5))
    return max(0.5, min(15.0, g_pct * 0.40))  # P/S heuristic


def build_multiple_params(
    multiple_bear:  float,
    multiple_base:  float,
    multiple_bull:  float,
    growth_mean:    float,
    quality_tier:   str  = "average",
    macro_regime:   str  = "Unknown",
    method:         str  = "P/E",
) -> MultipleDistParams:
    """
    Construct a MultipleDistParams from scenario bounds plus regime/quality context.

    Fair multiple is computed from the growth rate via _fair_multiple().
    Mean reversion then biases the Beta distribution toward that anchor,
    with speed and tightness determined by company quality.
    The macro rate adjustment shifts the anchor before reversion is applied,
    capturing the empirical relationship between interest rates and multiples.
    """
    qt = quality_tier if quality_tier in _MULT_QUALITY_TABLE else "average"
    mr_speed, concentration = _MULT_QUALITY_TABLE[qt]

    rk       = _regime_key(macro_regime)
    rate_adj = _REGIME_RATE_ADJ.get(rk, 0.00)

    rho_pe, rho_ps, sensitivity = _CORR_TABLE.get(qt, (0.40, 0.28, 0.07))
    rho = rho_pe if method == "P/E" else rho_ps

    low  = min(multiple_bear, multiple_base, multiple_bull)
    high = max(multiple_bear, multiple_base, multiple_bull)
    fair = _fair_multiple(growth_mean, method)

    return MultipleDistParams(
        low              = low,
        high             = high,
        current          = multiple_base,
        fair             = fair,
        mr_speed         = mr_speed,
        concentration    = concentration,
        rate_adj         = rate_adj,
        correlation_rho  = rho,
        corr_sensitivity = sensitivity,
        quality_tier     = qt,
        macro_regime     = macro_regime,
    )


def build_growth_params(
    growth_mean:  float,
    growth_std:   float,
    macro_regime: str            = "Unknown",
    gross_margin: Optional[float] = None,
    op_margin:    Optional[float] = None,
) -> GrowthDistParams:
    """
    Construct a GrowthDistParams from base EPS/revenue growth assumptions
    plus regime and company quality context.

    Called by mc_from_valuation_range before each simulation run so that
    distribution shape is always consistent with the current macro environment
    and the company's profitability profile.
    """
    rk = _regime_key(macro_regime)
    p_shock_add, mu_adj, sigma_scale, skew_adj = _REGIME_TABLE.get(
        rk, (0.05, 0.00, 1.00, 0.00)
    )

    qt = _infer_quality_tier(gross_margin, op_margin)
    sd_mult, su_mult, shock_prob_base, skew_base = _QUALITY_TABLE[qt]

    # Composite skew: quality base + regime adjustment, hard-clamped ±0.5
    skew = max(-0.50, min(0.50, skew_base + skew_adj))

    # Regime shifts the location and scales the spread
    adj_mean = growth_mean + mu_adj
    adj_std  = growth_std  * sigma_scale

    # Two-piece widths incorporate quality multipliers and composite skew
    sigma_down = max(adj_std * sd_mult * (1.0 - skew * 0.5), 0.005)
    sigma_up   = max(adj_std * su_mult * (1.0 + skew * 0.5), 0.005)

    # Shock component: severe downside draw anchored well below base
    shock_prob = min(shock_prob_base + p_shock_add, 0.50)
    shock_mean = growth_mean - 2.0 * growth_std - abs(mu_adj) * 2.0
    shock_std  = adj_std * 0.60

    return GrowthDistParams(
        growth_mean  = adj_mean,
        sigma_down   = sigma_down,
        sigma_up     = sigma_up,
        shock_prob   = shock_prob,
        shock_mean   = shock_mean,
        shock_std    = shock_std,
        quality_tier = qt,
        macro_regime = macro_regime,
    )


# ── Primitive samplers (stdlib only) ─────────────────────────────────────────

def _triangular(
    rng:  random.Random,
    low:  float,
    mode: float,
    high: float,
) -> float:
    """Sample from Triangular(low, mode, high). Degrades to mode if degenerate."""
    if low >= high:
        return mode
    # stdlib random.triangular(low, high, mode)
    return rng.triangular(low, high, mode)


def _normal(rng: random.Random, mean: float, std: float) -> float:
    """Sample from Normal(mean, std). Returns mean when std ≤ 0."""
    if std <= 0.0:
        return mean
    return rng.gauss(mean, std)


def _linear_pct(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile. p ∈ [0, 1]."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    idx = p * (n - 1)
    lo  = int(idx)
    hi  = lo + 1
    if hi >= n:
        return sorted_vals[-1]
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def _skewness(vals: list[float], mu: float, sigma: float) -> float:
    """3rd standardised central moment. Returns 0 if degenerate."""
    if sigma <= 0.0 or not vals:
        return 0.0
    n = len(vals)
    return sum(((v - mu) / sigma) ** 3 for v in vals) / n


def _split_normal(
    rng:        random.Random,
    mean:       float,
    sigma_down: float,
    sigma_up:   float,
) -> float:
    """
    Sample from a two-piece (split) normal distribution.

    Draws z ~ N(0,1); maps negative z through sigma_down, positive through
    sigma_up.  Reduces to symmetric Normal when sigma_down == sigma_up.
    """
    z = rng.gauss(0.0, 1.0)
    return mean + (z * sigma_down if z < 0.0 else z * sigma_up)


def _sample_growth(
    rng:    random.Random,
    params: GrowthDistParams,
) -> tuple[float, float]:
    """
    Sample one growth realisation from the mixture distribution.

        g ~ (1 − p_shock) × SplitNormal(μ, σ_down, σ_up)
           +       p_shock × Normal(μ_shock, σ_shock)

    Returns (g, z_eff) where z_eff is the standardised shock fed into the
    correlated multiple sampler.  z_eff = (g − μ) / σ_eff, clamped ±3.
    For shock paths, z_eff is driven deeply negative to propagate the
    correlation that earnings shocks also compress multiples.
    """
    eff_std = (params.sigma_down + params.sigma_up) / 2.0
    if rng.random() < params.shock_prob:
        g = rng.gauss(params.shock_mean, params.shock_std)
    else:
        g = _split_normal(rng, params.growth_mean, params.sigma_down, params.sigma_up)

    z_eff = (g - params.growth_mean) / max(eff_std, 1e-6)
    z_eff = max(-3.0, min(3.0, z_eff))    # clamp to avoid extreme lever arm
    return g, z_eff


def _sample_multiple(
    rng:      random.Random,
    params:   MultipleDistParams,
    growth_z: float = 0.0,
) -> float:
    """
    Sample one exit multiple from the correlated mean-reverting Beta distribution.

    Steps
    ─────
    1. Adjust fair multiple by macro rate environment (rate_adj)
    2. Apply partial mean reversion: reverted = current + mr_speed × (adj_fair − current)
    3. Apply growth-multiple correlation shift via Cholesky decomposition:
         Z_corr = ρ × growth_z + √(1−ρ²) × Z_indep
         shift  = Z_corr × corr_sensitivity × span
         reverted_corr = reverted + shift  (clamped to [low, high])
    4. Normalise reverted_corr to [0, 1] within [low, high]
    5. Sample Beta(α, β) where mean = normalised value and α+β = concentration
    6. Map back to the original multiple range

    Falls back to current multiple if the range is degenerate (low ≈ high).
    growth_z = 0.0 disables correlation (used for fallback / uncorrelated callers).
    """
    span = params.high - params.low
    if span < 1e-6:
        return params.current

    # Step 1: rate-adjusted fair multiple
    adj_fair = max(params.low, min(params.high,
                                   params.fair * (1.0 + params.rate_adj)))

    # Step 2: partial mean reversion toward adj_fair
    reverted = params.current + params.mr_speed * (adj_fair - params.current)
    reverted = max(params.low, min(params.high, reverted))

    # Step 3: Cholesky-correlated shift
    # Z_corr is a standard normal correlated at ρ with growth_z.
    # Positive Z_corr (growth beat) → multiple pushed upward; negative → downward.
    rho    = params.correlation_rho
    z_indep = rng.gauss(0.0, 1.0)
    z_corr  = rho * growth_z + math.sqrt(max(0.0, 1.0 - rho * rho)) * z_indep
    corr_shift = z_corr * params.corr_sensitivity * span
    reverted   = max(params.low, min(params.high, reverted + corr_shift))

    # Steps 4–6: Beta sampling
    mu_norm = max(0.02, min(0.98, (reverted - params.low) / span))
    alpha   = mu_norm * params.concentration
    beta_p  = (1.0 - mu_norm) * params.concentration
    u = rng.betavariate(alpha, beta_p)
    return params.low + u * span


# ── Core simulation ───────────────────────────────────────────────────────────

def run_monte_carlo(
    *,
    current_price:   float,
    metric_current:  float,         # EPS (P/E) | revenue per share (P/S)
    growth_mean:     float,         # base CAGR, decimal
    growth_std:      float,         # uncertainty, decimal
    multiple_bear:   float,         # triangular low
    multiple_base:   float,         # triangular mode
    multiple_bull:   float,         # triangular high
    horizon_years:   int   = 1,
    n_sims:          int   = 10_000,
    method:          str   = "P/E",
    rng_seed:        Optional[int]              = None,
    growth_params:   Optional[GrowthDistParams]   = None,  # asymmetric mixture
    multiple_params: Optional[MultipleDistParams]  = None,  # mean-reverting Beta
) -> MCResult:
    """
    Run N independent paths and return the full return/price distribution.

    Each path:
      1. g   ~ GrowthDist (SplitNormal mixture) or Normal(growth_mean, growth_std)
      2. m_t = metric_current × (1 + g)^t          [clamped at 5% floor]
      3. x   ~ Beta mean-reverting or Triangular(bear, base, bull)
      4. P_t = m_t × x
      5. R   = P_t / current_price − 1
    """
    if current_price <= 0:
        raise ValueError(f"current_price must be > 0, got {current_price}")
    if n_sims < 100:
        raise ValueError(f"n_sims must be ≥ 100, got {n_sims}")
    if metric_current <= 0:
        raise ValueError(f"metric_current must be > 0, got {metric_current}")

    # Deterministic seed: if caller did not supply one, derive from the driver
    # inputs via SHA-256 so the same inputs always produce the same RNG sequence.
    # random.Random(None) would seed from os.urandom() — non-deterministic across
    # consecutive runs on identical inputs.
    _seed = rng_seed if rng_seed is not None else _content_seed(
        current_price, metric_current, growth_mean, growth_std,
        multiple_bear, multiple_base, multiple_bull, float(horizon_years),
    )
    rng      = random.Random(_seed)
    floor    = metric_current * 0.05   # prevent collapse to near-zero
    mult_lo  = min(multiple_bear, multiple_base, multiple_bull)
    mult_hi  = max(multiple_bear, multiple_base, multiple_bull)
    t        = float(horizon_years)

    prices:  list[float] = []
    returns: list[float] = []

    for _ in range(n_sims):
        if growth_params is not None:
            g, g_z = _sample_growth(rng, growth_params)
        else:
            g   = _normal(rng, growth_mean, growth_std)
            g_z = 0.0   # no correlation available — independent sampling

        mt = metric_current * ((1.0 + g) ** t)
        mt = max(mt, floor)

        if multiple_params is not None:
            x = _sample_multiple(rng, multiple_params, growth_z=g_z)
        else:
            x = _triangular(rng, mult_lo, multiple_base, mult_hi)

        p  = max(mt * x, 0.0)
        r  = p / current_price - 1.0
        prices.append(p)
        returns.append(r)

    prices.sort()
    returns.sort()

    n   = len(returns)
    mu  = sum(returns) / n
    var = sum((r - mu) ** 2 for r in returns) / n
    std = math.sqrt(var) if var > 0 else 1e-9
    sk  = _skewness(returns, mu, std)

    # Half-Kelly: E[R] / Var[R] / 2, clamped [0, 10%]
    half_kelly = (mu / var / 2.0) if var > 0 else 0.0
    half_kelly = max(0.0, min(half_kelly, 0.10))

    p5r  = _linear_pct(returns, 0.05)
    p95r = _linear_pct(returns, 0.95)

    # Upside/downside ratio: positive = right-skewed (good for longs)
    if p5r < 0:
        ud = abs(p95r / p5r)
    elif p95r > 0:
        ud = p95r / 0.01   # denominator floored so division is safe
    else:
        ud = 1.0

    return MCResult(
        n_sims        = n_sims,
        horizon_years = horizon_years,
        method        = method,
        growth_mean   = growth_mean,
        growth_std    = growth_std,
        mean_return   = mu,
        median_return = _linear_pct(returns, 0.50),
        p5_return     = p5r,
        p25_return    = _linear_pct(returns, 0.25),
        p75_return    = _linear_pct(returns, 0.75),
        p95_return    = p95r,
        skewness      = sk,
        prob_positive = sum(1 for r in returns if r > 0.0) / n,
        prob_20_gain  = sum(1 for r in returns if r > 0.20) / n,
        prob_loss     = sum(1 for r in returns if r < 0.0) / n,
        prob_loss_20  = sum(1 for r in returns if r < -0.20) / n,
        mean_price    = sum(prices) / n,
        p5_price      = _linear_pct(prices, 0.05),
        p25_price     = _linear_pct(prices, 0.25),
        median_price  = _linear_pct(prices, 0.50),
        p75_price     = _linear_pct(prices, 0.75),
        p95_price     = _linear_pct(prices, 0.95),
        kelly_fraction   = half_kelly,
        upside_downside  = ud,
    )


# ── Convenience wrapper ────────────────────────────────────────────────────────

_DEFAULT_N_SIMS: int = 10_000


def _apply_layer_overrides(
    gparams:          "GrowthDistParams",
    mparams:          "MultipleDistParams",
    g_base:           float,
    macro_regime:     str,
    factor_profile:   object,
    regression_calib: object,
    scenario_tree:    object,
    hrl_result:       object = None,
) -> "tuple[GrowthDistParams, MultipleDistParams, float]":
    """
    Apply factor model, HRL, regression calibration, and scenario tree overrides
    to already-constructed growth and multiple distribution parameters.

    Priority stack (applied in order):
      4. Factor profile        — quality_tier re-derivation from cross-sectional
                                 z-scores (sigma asymmetry + shock_base);
                                 momentum_z → correlation_rho;
                                 profitability_z → shock_prob uplift;
                                 macro_score → growth_mean cyclicality drag
      3. HRL + regression      — three-way growth_mean blend (MC 40% + factor
                                 reg 30% + AR(1) 30%); mr_speed override from
                                 valuation mean-reversion kappa; margin trend
                                 drives sigma asymmetry
      2. Regression (fallback) — two-way blend when HRL not available
      1. Scenario tree         — shock_prob / shock_mean override from the
                                 contraction × miss leaf cluster (most explicit)

    None inputs degrade gracefully — each layer is independently optional.
    """
    # ── Factor profile: quality_tier → sigma asymmetry ────────────────────────
    if factor_profile is not None and getattr(factor_profile, "n_peers", 0) >= 3:
        _qf    = getattr(factor_profile, "quality_factor", 50.0)
        _fp_qt = (
            "high"     if _qf >= 65 else
            "cyclical" if _qf <= 35 else
            "average"
        )
        if _fp_qt != gparams.quality_tier:
            _rk = _regime_key(macro_regime)
            _p_shock_add, _, _, _skew_adj = _REGIME_TABLE.get(
                _rk, (0.05, 0.00, 1.00, 0.00)
            )
            _sd_mult, _su_mult, _shock_base, _skew_base = _QUALITY_TABLE[_fp_qt]
            _skew    = max(-0.50, min(0.50, _skew_base + _skew_adj))
            _adj_std = (gparams.sigma_down + gparams.sigma_up) / 2.0
            _old_qt              = gparams.quality_tier
            gparams.sigma_down   = max(_adj_std * _sd_mult * (1.0 - _skew * 0.5), 0.005)
            gparams.sigma_up     = max(_adj_std * _su_mult * (1.0 + _skew * 0.5), 0.005)
            gparams.shock_prob   = min(_shock_base + _p_shock_add, 0.50)
            gparams.quality_tier = _fp_qt
            print(
                f"  [MC:layer] quality_tier {_old_qt!r}→{_fp_qt!r}"
                f" (factor qfactor={_qf:.0f})"
            )

    # ── Factor profile: momentum_z → correlation_rho ──────────────────────────
    if factor_profile is not None:
        _mom_z = getattr(factor_profile, "momentum_z", 0.0)
        mparams.correlation_rho = min(0.60, max(0.10, 0.35 + _mom_z * 0.08))

    # ── Factor profile: profitability_z → shock_prob uplift ──────────────────
    # Weak profitability (z < -1.0) means thin margins → larger earnings miss
    # probability when macro deteriorates.
    if factor_profile is not None:
        _prof_z = getattr(factor_profile, "profitability_z", None)
        if _prof_z is not None and _prof_z < -1.0:
            _prob_add = min(0.08, (-_prof_z - 1.0) * 0.04)  # max +8pp at z≤-3
            gparams.shock_prob = min(0.45, gparams.shock_prob + _prob_add)
            print(
                f"  [MC:layer] profitability_z={_prof_z:+.2f}"
                f" → shock_prob +{_prob_add:.0%}"
            )

    # ── Factor profile: macro_score → growth_mean cyclicality drag ────────────
    # Highly cyclical stocks (macro_score < 35) get a growth haircut in MC when
    # the macro regime is contractionary; defensive stocks get a small uplift.
    if factor_profile is not None:
        _mac_score = getattr(factor_profile, "macro_score", None)
        if _mac_score is not None:
            _rk_local = _regime_key(macro_regime)
            if _rk_local in ("recession", "slowdown", "late_cycle"):
                if _mac_score < 35:
                    # Cyclical: haircut proportional to cyclicality
                    _drag = (_mac_score - 35.0) / 35.0 * 0.03   # negative
                    _drag = max(-0.05, _drag)
                    gparams.growth_mean = max(-0.30, gparams.growth_mean + _drag)
                    g_base              = gparams.growth_mean
                    print(
                        f"  [MC:layer] macro_score={_mac_score:.0f}"
                        f" (cyclical) → growth_mean drag {_drag:+.1%}"
                    )
                elif _mac_score > 65:
                    # Defensive: modest uplift
                    _lift = (_mac_score - 65.0) / 35.0 * 0.015   # positive, max +1.5%
                    _lift = min(0.015, _lift)
                    gparams.growth_mean = min(0.50, gparams.growth_mean + _lift)
                    g_base              = gparams.growth_mean
                    print(
                        f"  [MC:layer] macro_score={_mac_score:.0f}"
                        f" (defensive) → growth_mean lift +{_lift:.1%}"
                    )

    # ── HRL + regression calibration: growth_mean blend ──────────────────────
    # Three-way blend: MC fundamental (40%) + factor reg (30%) + AR(1) (30%).
    # When HRL result is available and has medium/high confidence, use its
    # pre-computed calibrated_growth_mean directly.  Additionally apply:
    #   · valuation_mr_speed  → mparams.mr_speed override
    #   · margin_trend_slope  → sigma asymmetry (declining margins = fatter left tail)
    # Fall back to the two-way factor regression blend when HRL is absent.
    _hrl_applied = False

    if hrl_result is not None:
        _hrl_conf  = getattr(hrl_result, "hrl_confidence",         "low")
        _hrl_g     = getattr(hrl_result, "calibrated_growth_mean", None)
        _hrl_kappa = getattr(hrl_result, "valuation_mr_speed",     None)
        _hrl_slope = getattr(hrl_result, "margin_trend_slope",     None)

        # ── Three-way growth mean blend ──────────────────────────────────────
        if _hrl_g is not None and _hrl_conf in ("high", "medium"):
            _hrl_w   = 0.55 if _hrl_conf == "high" else 0.40
            _blended = (1.0 - _hrl_w) * gparams.growth_mean + _hrl_w * _hrl_g
            print(
                f"  [MC:layer] HRL growth blend:"
                f" {gparams.growth_mean:+.1%}→{_blended:+.1%}"
                f" (hrl={_hrl_g:+.1%}, w={_hrl_w:.0%}, conf={_hrl_conf})"
            )
            gparams.growth_mean = _blended
            g_base              = _blended
            _hrl_applied        = True

        # ── Mean-reversion speed from valuation model ────────────────────────
        if _hrl_kappa is not None and _hrl_kappa > 0.01:
            _old_mr          = mparams.mr_speed
            mparams.mr_speed = max(0.05, min(0.80, _hrl_kappa))
            print(
                f"  [MC:layer] HRL mr_speed: {_old_mr:.2f}→{mparams.mr_speed:.2f}"
                f" (kappa={_hrl_kappa:.3f})"
            )

        # ── Margin trend → sigma asymmetry ───────────────────────────────────
        # Negative slope (declining margins) widens sigma_down, compresses sigma_up.
        # Positive slope (rising margins) does the reverse.
        if _hrl_slope is not None and abs(_hrl_slope) > 0.003:
            _slope_c = max(-0.05, min(0.05, _hrl_slope))
            _asym    = _slope_c / 0.05 * 0.15   # ±15% maximum asymmetry
            _old_sd  = gparams.sigma_down
            _old_su  = gparams.sigma_up
            gparams.sigma_down = max(0.005, gparams.sigma_down * (1.0 - _asym))
            gparams.sigma_up   = max(0.005, gparams.sigma_up   * (1.0 + _asym))
            print(
                f"  [MC:layer] HRL margin slope={_hrl_slope:+.4f}"
                f" → σ↓ {_old_sd:.1%}→{gparams.sigma_down:.1%}"
                f"  σ↑ {_old_su:.1%}→{gparams.sigma_up:.1%}"
            )

    if not _hrl_applied and regression_calib is not None:
        # ── Fallback: two-way factor regression blend ────────────────────────
        # Used when HRL is absent or low-confidence.
        _conf   = getattr(regression_calib, "confidence", "low")
        _reg_er = getattr(regression_calib, "expected_return", None)
        if _conf in ("high", "medium") and _reg_er is not None:
            _blend_w    = 0.50 if _conf == "high" else 0.30
            _reg_growth = max(-0.30, min(0.50, _reg_er - 0.015))
            _blended    = (1.0 - _blend_w) * gparams.growth_mean + _blend_w * _reg_growth
            print(
                f"  [MC:layer] growth_mean blend (factor reg fallback):"
                f" {gparams.growth_mean:+.1%}→{_blended:+.1%}"
                f" (reg E[R]={_reg_er:+.1%}, w={_blend_w:.0%})"
            )
            gparams.growth_mean = _blended
            g_base              = _blended

    # ── Tracking error → sigma calibration (always when regression_calib available)
    if regression_calib is not None:
        _te = getattr(regression_calib, "tracking_error", None)
        if _te is not None and _te > 0:
            _tw = 0.30
            gparams.sigma_down = (1.0 - _tw) * gparams.sigma_down + _tw * _te * 0.70
            gparams.sigma_up   = (1.0 - _tw) * gparams.sigma_up   + _tw * _te * 1.10

    # ── Scenario tree: shock cluster ──────────────────────────────────────────
    if scenario_tree is not None:
        _st_p = getattr(scenario_tree, "shock_prob",        None)
        _st_g = getattr(scenario_tree, "shock_mean_growth", None)
        if _st_p is not None:
            # Conservative: take the larger shock probability
            gparams.shock_prob = min(0.40, max(gparams.shock_prob, _st_p))
        if _st_g is not None:
            # Take the more pessimistic shock mean
            gparams.shock_mean = min(gparams.shock_mean, _st_g)
        print(
            f"  [MC:layer] tree shock:"
            f" prob={gparams.shock_prob:.0%} mean={gparams.shock_mean:+.1%}"
        )

    return gparams, mparams, g_base


def mc_from_valuation_range(
    vr:               object,
    n_sims:           int             = _DEFAULT_N_SIMS,
    macro_regime:     str             = "Unknown",
    gross_margin:     Optional[float] = None,
    op_margin:        Optional[float] = None,
    stock_data:       object          = None,   # StockData — enables historical calibration
    # ── Alpha engine layer inputs ─────────────────────────────────────────────
    factor_profile:   object          = None,   # FactorProfile  (factor_model.py)
    regression_calib: object          = None,   # RegressionCalibration (regression_calibration.py)
    hrl_result:       object          = None,   # HRLResult (historical_regression.py)
    scenario_tree:    object          = None,   # ScenarioTree   (scenario_tree.py)
) -> Optional[MCResult]:
    """
    Derive Monte Carlo inputs from a completed ValuationRange and run the simulation.

    Path selection priority (mirrors ValuationRange.scenario_primary_method):
      1. P/E  — requires scenario_base_eps > 0 with bear/base/bull P/E multiples
      2. P/S  — requires scenario_ps_rev_per_share > 0 with bear/base/bull P/S multiples
      (EV/EBITDA is not used: converting EV to equity value requires debt/cash
       which is not available on the ValuationRange object post-computation.)

    Growth uncertainty is derived from the scenario EPS spread:
      g_bear ≈ 0         (bear EPS = flat TTM EPS, no growth assumed)
      g_bull = g × 1.30  (bull EPS = 1-year forward at 130% of base CAGR)
      std    = (g_bull − g_bear) / (2 × 1.645)

    This treats bear/bull growth outcomes as approximate P5/P95 of a Normal
    distribution — the same assumption implicitly embedded in the scenario table.

    Returns None when insufficient inputs are available (e.g. negative EPS,
    no revenue per share, or current_price missing).
    """
    price = getattr(vr, "current_price", None)
    if not price or price <= 0:
        return None

    # ── Scenario-derived distribution (highest priority) ──────────────────────
    # When bear/base/bull prices are already computed from the scenario tree,
    # pin the distribution to those exact prices (P5=Bear, P50=Base, P95=Bull).
    # This eliminates the structural contradiction where an independent P/E
    # simulation produces a P50 that deviates from the scenario Base price.
    _bear_px = getattr(vr, "bear_price", None)
    _base_px = getattr(vr, "base_price", None)
    _bull_px = getattr(vr, "bull_price", None)
    if (
        _bear_px is not None and _bear_px > 0
        and _base_px is not None and _base_px > 0
        and _bull_px is not None and _bull_px > 0
    ):
        _base_rg = getattr(vr, "scenario_base_rev_growth", None) or 0.0
        try:
            from analysis.valuation_range import _scenario_derived_mc as _sdmc
            _sd_result = _sdmc(_bear_px, _base_px, _bull_px, price, _base_rg)
            if _sd_result is not None:
                print(
                    f"  [MC:scenario_pass] P5=${_sd_result.p5_price:.2f}"
                    f" P50=${_sd_result.median_price:.2f} (== Base)"
                    f" P95=${_sd_result.p95_price:.2f}"
                )
                return _sd_result
        except Exception as _sd_err:
            print(f"  [MC:scenario_pass] skipped: {_sd_err}")

    horizon = 1  # all scenarios project 1-year forward

    # ── P/E path ──────────────────────────────────────────────────────────────
    eps_base = getattr(vr, "scenario_base_eps", None)
    eps_bear = getattr(vr, "scenario_bear_eps", None)
    pe_bear  = getattr(vr, "scenario_bear_pe",  None)
    pe_base  = getattr(vr, "scenario_base_pe",  None)
    pe_bull  = getattr(vr, "scenario_bull_pe",  None)

    if (
        eps_base is not None and eps_base > 0
        and pe_bear is not None
        and pe_base is not None
        and pe_bull is not None
    ):
        # metric_current: use bear-case EPS (flat TTM) as the anchor.
        # The growth distribution then projects it forward to the 1-year horizon.
        eps0   = (eps_bear if eps_bear and eps_bear > 0 else eps_base)
        g_base = (getattr(vr, "eps_growth_rate", None) or 0.0) / 100.0
        g_bull = g_base * 1.30
        g_bear = 0.0
        # std from scenario spread, floored at 40% of |base| or 3 pp
        g_std  = (g_bull - g_bear) / (2.0 * 1.645)
        g_std  = max(g_std, abs(g_base) * 0.40, 0.03)

        # Fallback quality from stored attrs when caller didn't supply them
        _gm  = gross_margin if gross_margin is not None else getattr(vr, "quality_gross_margin", None)
        _opm = op_margin    if op_margin    is not None else getattr(vr, "quality_op_margin",    None)
        _qt  = _infer_quality_tier(_gm, _opm)

        # ── Calibration path (preferred when stock_data is available) ──────────
        _calib = None
        if stock_data is not None:
            try:
                from analysis.mc_calibration import calibrate_mc_params as _calibrate
                _calib = _calibrate(
                    stock_data,
                    multiple_bear = pe_bear,
                    multiple_base = pe_base,
                    multiple_bull = pe_bull,
                    method        = "P/E",
                    macro_regime  = macro_regime,
                    gross_margin  = _gm,
                    op_margin     = _opm,
                )
                if _calib:
                    print(f"  [MC:calib] growth n={_calib.growth_calib.n_obs}"
                          f" conf={_calib.growth_calib.confidence}"
                          f" mult n={_calib.multiple_calib.n_obs}"
                          f" mr_speed={_calib.multiple_calib.mr_speed:.2f}")
            except Exception as _ce:
                print(f"  [MC:calib] skipped: {_ce}")

        if _calib is not None:
            gparams, mparams = _calib.growth_params, _calib.multiple_params
            # Calibration overrides g_base/g_std for mean_return consistency
            g_base = _calib.growth_calib.mean
            g_std  = _calib.growth_calib.std
        else:
            gparams = build_growth_params(
                growth_mean  = g_base,
                growth_std   = g_std,
                macro_regime = macro_regime,
                gross_margin = _gm,
                op_margin    = _opm,
            )
            mparams = build_multiple_params(
                multiple_bear = pe_bear,
                multiple_base = pe_base,
                multiple_bull = pe_bull,
                growth_mean   = g_base,
                quality_tier  = _qt,
                macro_regime  = macro_regime,
                method        = "P/E",
            )

        # ── Alpha engine layer overrides (factor / regression / HRL / scenario) ─
        gparams, mparams, g_base = _apply_layer_overrides(
            gparams, mparams, g_base, macro_regime,
            factor_profile, regression_calib, scenario_tree,
            hrl_result = hrl_result,
        )

        return run_monte_carlo(
            current_price   = price,
            metric_current  = eps0,
            growth_mean     = g_base,
            growth_std      = g_std,
            multiple_bear   = pe_bear,
            multiple_base   = pe_base,
            multiple_bull   = pe_bull,
            horizon_years   = horizon,
            n_sims          = n_sims,
            method          = "P/E",
            growth_params   = gparams,
            multiple_params = mparams,
        )

    # ── P/S path ──────────────────────────────────────────────────────────────
    rev_sh  = getattr(vr, "scenario_ps_rev_per_share", None)
    ps_bear = getattr(vr, "scenario_bear_ps",          None)
    ps_base = getattr(vr, "scenario_base_ps",          None)
    ps_bull = getattr(vr, "scenario_bull_ps",          None)

    if (
        rev_sh is not None and rev_sh > 0
        and ps_bear is not None
        and ps_base is not None
        and ps_bull is not None
    ):
        g_base = (getattr(vr, "eps_growth_rate", None) or 5.0) / 100.0
        g_bull = g_base * 1.30
        g_bear = 0.0
        g_std  = (g_bull - g_bear) / (2.0 * 1.645)
        g_std  = max(g_std, 0.03)

        _gm  = gross_margin if gross_margin is not None else getattr(vr, "quality_gross_margin", None)
        _opm = op_margin    if op_margin    is not None else getattr(vr, "quality_op_margin",    None)
        _qt  = _infer_quality_tier(_gm, _opm)

        _calib = None
        if stock_data is not None:
            try:
                from analysis.mc_calibration import calibrate_mc_params as _calibrate
                _calib = _calibrate(
                    stock_data,
                    multiple_bear = ps_bear,
                    multiple_base = ps_base,
                    multiple_bull = ps_bull,
                    method        = "P/S",
                    macro_regime  = macro_regime,
                    gross_margin  = _gm,
                    op_margin     = _opm,
                )
            except Exception:
                pass

        if _calib is not None:
            gparams, mparams = _calib.growth_params, _calib.multiple_params
            g_base = _calib.growth_calib.mean
            g_std  = _calib.growth_calib.std
        else:
            gparams = build_growth_params(
                growth_mean  = g_base,
                growth_std   = g_std,
                macro_regime = macro_regime,
                gross_margin = _gm,
                op_margin    = _opm,
            )
            mparams = build_multiple_params(
                multiple_bear = ps_bear,
                multiple_base = ps_base,
                multiple_bull = ps_bull,
                growth_mean   = g_base,
                quality_tier  = _qt,
                macro_regime  = macro_regime,
                method        = "P/S",
            )

        # ── Alpha engine layer overrides (factor / regression / HRL / scenario) ─
        gparams, mparams, g_base = _apply_layer_overrides(
            gparams, mparams, g_base, macro_regime,
            factor_profile, regression_calib, scenario_tree,
            hrl_result = hrl_result,
        )

        return run_monte_carlo(
            current_price   = price,
            metric_current  = rev_sh,
            growth_mean     = g_base,
            growth_std      = g_std,
            multiple_bear   = ps_bear,
            multiple_base   = ps_base,
            multiple_bull   = ps_bull,
            horizon_years   = horizon,
            n_sims          = n_sims,
            method          = "P/S",
            growth_params   = gparams,
            multiple_params = mparams,
        )

    return None


# ── Portfolio sizing adjustment ────────────────────────────────────────────────

def position_adjustment_factor(mc: MCResult) -> float:
    """
    Return a multiplicative adjustment to the base position size [0.5, 1.5].

    Logic (applied in priority order):
      · P(loss > 20%) ≥ 30% → scale down 0.5× (severe tail risk)
      · P(loss > 20%) ≥ 15% → scale down 0.75× (moderate tail risk)
      · Upside/downside ≥ 3.0 AND P(gain) ≥ 60% → scale up 1.25×
      · Upside/downside ≥ 2.5 AND P(gain) ≥ 55% → scale up 1.10×
      · Otherwise: no adjustment (1.0×)

    The caller multiplies this factor against the conviction-tier base size
    produced by the existing position-sizing engine (reporting_agent.py),
    and then re-applies any existing hard caps.
    """
    if mc.prob_loss_20 >= 0.30:
        return 0.50
    if mc.prob_loss_20 >= 0.15:
        return 0.75
    if mc.upside_downside >= 3.0 and mc.prob_positive >= 0.60:
        return 1.25
    if mc.upside_downside >= 2.5 and mc.prob_positive >= 0.55:
        return 1.10
    return 1.0
