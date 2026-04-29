"""
macro_overlay — rule-based macro LEI scoring.

Takes a dict of latest indicator values (from FREDProvider.get_lei_snapshot())
and returns a MacroAssessment with score, regime, risk level, and factor lists.

Scoring is intentionally rule-based and transparent.  Each indicator
contributes a bounded sub-score; the weighted average produces the final
macro_score (0–100).  Tune the thresholds and weights at the top of this file.

Indicator weights (must sum to 1.0)
────────────────────────────────────
  yield_spread    0.30  — strongest single leading indicator for recession
  jobless_claims  0.25  — coincident but fast-moving; good for turns
  housing_starts  0.20  — long-lead indicator, sensitive to rates
  activity_proxy  0.15  — mfg employment as PMI stand-in
  composite_lei   0.10  — OECD CLI or CB LEI when available

Regime thresholds
─────────────────
  score ≥ 65  → Expansion
  score ≥ 50  → Slowdown
  score ≥ 35  → Contraction
  score <  35 → Contraction (severe)

  Recovery is applied when:
    - score is 35–55 AND
    - yield spread is improving (above −0.25 after being negative)
    - OR jobless claims are below 260k
  This avoids Recovery being confused with early Contraction.

Recession risk
──────────────
  macro_score ≥ 65  → Low
  macro_score ≥ 50  → Moderate
  macro_score ≥ 35  → Elevated
  macro_score <  35 → High
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Thresholds ─────────────────────────────────────────────────────────────────

# Yield curve (T10Y2Y, percentage points)
_YIELD_SPREAD_STRONG   =  0.75   # clearly positive → bullish
_YIELD_SPREAD_FLAT     =  0.10   # flat / borderline
_YIELD_SPREAD_INVERTED = -0.10   # inverted → bearish
_YIELD_SPREAD_DEEP     = -0.50   # deeply inverted → very bearish

# Initial jobless claims (seasonally adjusted, thousands of persons)
_CLAIMS_LOW    = 215_000   # historically healthy
_CLAIMS_RISING = 250_000   # starting to concern
_CLAIMS_HIGH   = 300_000   # recession territory
_CLAIMS_SEVERE = 400_000   # crisis-level

# Housing starts (HOUST, thousands of units, SAAR)
_STARTS_STRONG = 1_500     # above-trend
_STARTS_OK     = 1_200     # adequate
_STARTS_WEAK   = 1_000     # below trend
_STARTS_POOR   =   800     # very weak

# Manufacturing employment (MANEMP, thousands, SAAR) — PMI proxy
_MFG_STRONG    = 13_000    # above pre-GFC norms
_MFG_OK        = 12_500
_MFG_WEAK      = 12_000
_MFG_POOR      = 11_500

# OECD CLI (values centred on 100)
_CLI_STRONG    = 100.5
_CLI_OK        = 100.0
_CLI_WEAK      =  99.5
_CLI_POOR      =  99.0

# Weights (must sum to 1.0)
_WEIGHTS = {
    "yield_spread":   0.30,
    "jobless_claims": 0.25,
    "housing_starts": 0.20,
    "activity_proxy": 0.15,
    "composite_lei":  0.10,
}

# Regime thresholds
_EXPANSION_MIN   = 65
_SLOWDOWN_MIN    = 50
_CONTRACTION_MIN = 35

# Sector tilt by regime (fallback when cycle_phase is unknown)
_SECTOR_TILTS = {
    "Expansion":   "Cyclicals, Industrials, Financials",
    "Slowdown":    "Defensives, Healthcare, Consumer Staples",
    "Contraction": "Treasuries, Utilities, Gold proxies",
    "Recovery":    "Small-caps, Cyclicals, Real Estate",
}

# Phase-specific sector tilts (more granular; keyed by (regime, cycle_phase))
_PHASE_SECTOR_TILTS: dict[tuple[str, str], str] = {
    ("Recovery",    "early"):       "Small-caps, Cyclicals, Real Estate",
    ("Expansion",   "mid"):         "Cyclicals, Industrials, Financials",
    ("Expansion",   "late"):        "Quality equities, Dividend payers; reduce high-beta",
    ("Slowdown",    "early"):       "Selective cyclicals, Quality Growth",
    ("Slowdown",    "late"):        "Defensives, Healthcare, Consumer Staples",
    ("Contraction", "contraction"): "Treasuries, Utilities, Gold proxies",
}


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class MacroAssessment:
    macro_score:         float                   # 0–100
    macro_regime:        str                     # Expansion / Slowdown / Contraction / Recovery
    recession_risk_level: str                    # Low / Moderate / Elevated / High
    confidence_modifier: float                   # −0.10 to +0.05; applied to overall eval confidence
    sector_tilt:         str                     # sector preference given current regime + phase
    reasoning_summary:   str
    bullish_macro_factors: list[str] = field(default_factory=list)
    bearish_macro_factors: list[str] = field(default_factory=list)
    data_coverage:       float = 1.0             # fraction of indicators that had real data
    # Phase 1 LEI additions
    cycle_phase:         str = "unknown"         # early / mid / late / contraction / unknown
    lei_trend:           Optional[str] = None    # rising / falling / inflecting / None
    yield_spread_trend:  Optional[str] = None    # rising / falling / inflecting / None
    # Confidence rationale — human-readable explanation of confidence_modifier
    confidence_adjustment_rationale: str = ""


# ── Scoring helpers ────────────────────────────────────────────────────────────

def _score_yield_spread(spread: Optional[float]) -> tuple[float, list[str], list[str]]:
    """Return (sub_score 0–100, bullish_factors, bearish_factors)."""
    if spread is None:
        return 50.0, [], ["Yield curve data not yet available — signal inconclusive"]

    bullish, bearish = [], []

    if spread >= _YIELD_SPREAD_STRONG:
        sub = 85.0
        bullish.append(f"Yield curve positively sloped (+{spread:.2f}pp) — historically expansionary")
    elif spread >= _YIELD_SPREAD_FLAT:
        sub = 65.0
        bullish.append(f"Yield curve modestly positive ({spread:.2f}pp) — neutral to constructive")
    elif spread >= _YIELD_SPREAD_INVERTED:
        sub = 45.0
        bearish.append(f"Yield curve flat ({spread:.2f}pp) — caution warranted")
    elif spread >= _YIELD_SPREAD_DEEP:
        sub = 25.0
        bearish.append(f"Yield curve inverted ({spread:.2f}pp) — recession signal active")
    else:
        sub = 10.0
        bearish.append(f"Yield curve deeply inverted ({spread:.2f}pp) — strong recession warning")

    return sub, bullish, bearish


def _score_jobless_claims(claims: Optional[float]) -> tuple[float, list[str], list[str]]:
    if claims is None:
        return 50.0, [], ["Jobless claims data not yet available — signal inconclusive"]

    bullish, bearish = [], []
    c = claims

    if c <= _CLAIMS_LOW:
        sub = 85.0
        bullish.append(f"Initial jobless claims low ({c:,.0f}) — labour market strong")
    elif c <= _CLAIMS_RISING:
        sub = 65.0
        bullish.append(f"Initial jobless claims healthy ({c:,.0f})")
    elif c <= _CLAIMS_HIGH:
        sub = 40.0
        bearish.append(f"Initial jobless claims elevated ({c:,.0f}) — labour market softening")
    elif c <= _CLAIMS_SEVERE:
        sub = 20.0
        bearish.append(f"Initial jobless claims high ({c:,.0f}) — significant labour market stress")
    else:
        sub = 5.0
        bearish.append(f"Initial jobless claims at crisis levels ({c:,.0f})")

    return sub, bullish, bearish


def _score_housing_starts(starts: Optional[float]) -> tuple[float, list[str], list[str]]:
    if starts is None:
        return 50.0, [], ["Housing starts data not yet available — signal inconclusive"]

    bullish, bearish = [], []
    s = starts

    if s >= _STARTS_STRONG:
        sub = 80.0
        bullish.append(f"Housing starts robust ({s:,.0f}k) — construction cycle healthy")
    elif s >= _STARTS_OK:
        sub = 62.0
        bullish.append(f"Housing starts adequate ({s:,.0f}k)")
    elif s >= _STARTS_WEAK:
        sub = 40.0
        bearish.append(f"Housing starts below trend ({s:,.0f}k) — construction activity slowing")
    elif s >= _STARTS_POOR:
        sub = 22.0
        bearish.append(f"Housing starts weak ({s:,.0f}k) — significant housing contraction")
    else:
        sub = 8.0
        bearish.append(f"Housing starts very weak ({s:,.0f}k) — housing sector in distress")

    return sub, bullish, bearish


def _score_activity_proxy(mfg_emp: Optional[float]) -> tuple[float, list[str], list[str]]:
    """Score manufacturing employment as a PMI/activity proxy."""
    if mfg_emp is None:
        return 50.0, [], ["Manufacturing activity data unavailable — defaulting to neutral"]

    bullish, bearish = [], []
    m = mfg_emp

    if m >= _MFG_STRONG:
        sub = 80.0
        bullish.append(f"Manufacturing employment elevated ({m:,.0f}k) — industrial activity solid")
    elif m >= _MFG_OK:
        sub = 62.0
        bullish.append(f"Manufacturing employment adequate ({m:,.0f}k)")
    elif m >= _MFG_WEAK:
        sub = 42.0
        bearish.append(f"Manufacturing employment below trend ({m:,.0f}k)")
    elif m >= _MFG_POOR:
        sub = 25.0
        bearish.append(f"Manufacturing employment weak ({m:,.0f}k) — industrial contraction underway")
    else:
        sub = 10.0
        bearish.append(f"Manufacturing employment very low ({m:,.0f}k) — industrial recession")

    return sub, bullish, bearish


def _score_composite_lei(cli: Optional[float], lei: Optional[float]) -> tuple[Optional[float], list[str], list[str]]:
    """Use OECD CLI if available; fall back to CB LEI heuristic if not.

    Returns (None, [], [note]) when no CLI/LEI data is present so the caller
    can exclude this indicator from the weighted average entirely, rather than
    silently biasing the score toward neutral.
    """
    # Prefer CLI (normalized around 100)
    value = cli  # may be None
    source = "OECD CLI"
    if value is None and lei is not None:
        # CB LEI is an index level; meaningful as a trend, not absolute level
        # Treat values > 102 as expansionary, < 98 as contractionary for scoring
        value = lei
        source = "Conference Board LEI"

    if value is None:
        return None, [], ["Composite leading index: OECD CLI data too stale to use. Excluded from regime calculation."]

    bullish, bearish = [], []

    if source == "OECD CLI":
        if value >= _CLI_STRONG:
            sub = 80.0
            bullish.append(f"OECD CLI above trend ({value:.2f}) — above-trend growth momentum")
        elif value >= _CLI_OK:
            sub = 60.0
            bullish.append(f"OECD CLI at trend ({value:.2f}) — growth on track")
        elif value >= _CLI_WEAK:
            sub = 40.0
            bearish.append(f"OECD CLI below trend ({value:.2f}) — growth momentum fading")
        else:
            sub = 20.0
            bearish.append(f"OECD CLI well below trend ({value:.2f}) — leading indicator weakening")
    else:
        # CB LEI: rough absolute thresholds
        if value > 102:
            sub = 78.0
            bullish.append(f"Conference Board LEI positive ({value:.1f}) — expansion signal")
        elif value >= 98:
            sub = 55.0
            bullish.append(f"Conference Board LEI near trend ({value:.1f})")
        else:
            sub = 28.0
            bearish.append(f"Conference Board LEI weak ({value:.1f}) — contraction signal")

    return sub, bullish, bearish


def _classify_regime(
    score: float,
    yield_spread: Optional[float],
    claims: Optional[float],
) -> str:
    """
    Classify macro regime from score, with Recovery override logic.
    Recovery is distinguished from early Contraction by:
      - spread recovering toward zero after being negative
      - claims still low-to-moderate
    """
    if score >= _EXPANSION_MIN:
        return "Expansion"

    if score >= _SLOWDOWN_MIN:
        # Could be Slowdown or early Recovery
        if yield_spread is not None and yield_spread >= _YIELD_SPREAD_INVERTED:
            if claims is None or claims <= _CLAIMS_RISING:
                return "Recovery"
        return "Slowdown"

    if score >= _CONTRACTION_MIN:
        # Could be Contraction or early Recovery
        spread_recovering = (
            yield_spread is not None
            and _YIELD_SPREAD_DEEP <= yield_spread < _YIELD_SPREAD_FLAT
        )
        claims_stable = claims is not None and claims <= _CLAIMS_RISING
        if spread_recovering and claims_stable:
            return "Recovery"
        return "Contraction"

    return "Contraction"


def _classify_cycle_phase(
    regime: str,
    cli_trend: Optional[str],
    yield_spread: Optional[float],
    yield_spread_trend: Optional[str],
) -> str:
    """
    Map regime + trend direction to a coarse cycle phase.

    Returns one of: "early" | "mid" | "late" | "contraction" | "unknown"

    Rules are intentionally coarse and transparent.  Each branch is labelled
    with the condition that triggers it so future tuning is legible.

    Recovery  → always "early"   (by definition: trough has passed)
    Slowdown  → usually "late";  "early" only if CLI is inflecting up AND
                                  yield spread is recovering (a turning point)
    Expansion → "mid" by default;
                "late" when yield spread is below 0.25pp OR CLI is falling
                  (spread compression and fading CLI are classic late-cycle tells)
    Contraction → "contraction"  (separate from phase language)
    """
    if regime == "Contraction":
        return "contraction"

    if regime == "Recovery":
        return "early"

    if regime == "Expansion":
        # Late-cycle signals: spread tight/flat, OR CLI momentum is fading
        spread_tight  = yield_spread is not None and yield_spread < 0.25
        cli_fading    = cli_trend == "falling"
        if spread_tight or cli_fading:
            return "late"
        return "mid"

    if regime == "Slowdown":
        # Turning-point signal: CLI inflecting up + spread recovering → early
        cli_turning    = cli_trend == "inflecting"
        spread_turning = yield_spread_trend == "rising"
        if cli_turning and spread_turning:
            return "early"
        return "late"

    return "unknown"


def _recession_risk(score: float) -> str:
    if score >= _EXPANSION_MIN:
        return "Low"
    if score >= _SLOWDOWN_MIN:
        return "Moderate"
    if score >= _CONTRACTION_MIN:
        return "Elevated"
    return "High"


def _confidence_modifier(score: float, data_coverage: float) -> float:
    """
    Positive modifier (+0.05) for strong macro, negative (up to −0.10)
    for weak macro or poor data coverage.
    Clamped to [−0.10, +0.05].
    """
    if data_coverage < 0.4:
        base = -0.05  # too little data to be useful
    elif score >= 70:
        base = 0.05
    elif score >= 55:
        base = 0.02
    elif score >= 40:
        base = -0.02
    elif score >= 25:
        base = -0.06
    else:
        base = -0.10

    # Degrade further for missing data
    coverage_penalty = (1.0 - data_coverage) * 0.05
    return max(-0.10, min(0.05, base - coverage_penalty))


# ── Public entry point ─────────────────────────────────────────────────────────

def score(snapshot: dict) -> MacroAssessment:
    """
    Convert a LEI snapshot dict (from FREDProvider.get_lei_snapshot()) into
    a MacroAssessment.

    Expected snapshot keys (all optional — missing → None):
      yield_spread_10y2y, housing_starts, jobless_claims,
      lei_composite, oecd_cli, mfg_employment,
      oecd_cli_trend, yield_spread_trend          ← Phase 1 additions
    """
    spread    = snapshot.get("yield_spread_10y2y")
    starts    = snapshot.get("housing_starts")
    claims    = snapshot.get("jobless_claims")
    lei       = snapshot.get("lei_composite")
    cli       = snapshot.get("oecd_cli")
    mfg       = snapshot.get("mfg_employment")
    # Phase 1 trend fields — None when FRED unavailable or window too short
    cli_trend    = snapshot.get("oecd_cli_trend")
    spread_trend = snapshot.get("yield_spread_trend")

    print(
        f"  [MACRO] raw inputs:"
        f" yield_spread={spread} housing_starts={starts} jobless_claims={claims}"
        f" lei_composite={lei} oecd_cli={cli} mfg_employment={mfg}"
    )

    # Count how many indicators had real data
    indicators = [spread, starts, claims, mfg, cli if cli is not None else lei]
    available  = sum(1 for v in indicators if v is not None)
    data_coverage = available / len(indicators)

    # Score each indicator
    s_yield,   b_yield,   bad_yield   = _score_yield_spread(spread)
    s_claims,  b_claims,  bad_claims  = _score_jobless_claims(claims)
    s_starts,  b_starts,  bad_starts  = _score_housing_starts(starts)
    s_activity,b_activity,bad_activity= _score_activity_proxy(mfg)
    s_cli,     b_cli,     bad_cli     = _score_composite_lei(cli, lei)

    _cli_str = f"{s_cli:.0f}" if s_cli is not None else "excluded"
    print(
        f"  [MACRO] scores: yield={s_yield:.0f} claims={s_claims:.0f}"
        f" starts={s_starts:.0f} activity={s_activity:.0f} cli={_cli_str}"
    )

    # Weighted macro score — exclude CLI weight when CLI/LEI data is unavailable
    # and redistribute its weight proportionally to the remaining four indicators.
    if s_cli is None:
        _cli_w    = _WEIGHTS["composite_lei"]
        _avail_w  = 1.0 - _cli_w   # = 0.90
        macro_score = (
            s_yield    * (_WEIGHTS["yield_spread"]    / _avail_w)
            + s_claims * (_WEIGHTS["jobless_claims"]  / _avail_w)
            + s_starts * (_WEIGHTS["housing_starts"]  / _avail_w)
            + s_activity * (_WEIGHTS["activity_proxy"] / _avail_w)
        )
    else:
        macro_score = (
            s_yield    * _WEIGHTS["yield_spread"]
            + s_claims * _WEIGHTS["jobless_claims"]
            + s_starts * _WEIGHTS["housing_starts"]
            + s_activity * _WEIGHTS["activity_proxy"]
            + s_cli    * _WEIGHTS["composite_lei"]
        )
    print(
        f"  [MACRO] weighted macro_score={macro_score:.1f}/100"
        f" data_coverage={data_coverage:.0%}"
    )

    # Combine factors
    bullish = b_yield + b_claims + b_starts + b_activity + b_cli
    bearish = bad_yield + bad_claims + bad_starts + bad_activity + bad_cli

    regime      = _classify_regime(macro_score, spread, claims)
    cycle_phase = _classify_cycle_phase(regime, cli_trend, spread, spread_trend)
    risk        = _recession_risk(macro_score)
    conf_mod    = _confidence_modifier(macro_score, data_coverage)
    # Phase-specific sector tilt — falls back to regime-level tilt when phase is unknown
    tilt        = _PHASE_SECTOR_TILTS.get((regime, cycle_phase), _SECTOR_TILTS.get(regime, "No tilt"))

    print(
        f"  [MACRO] cycle_phase={cycle_phase}"
        f"  lei_trend={cli_trend or 'N/A'}"
        f"  spread_trend={spread_trend or 'N/A'}"
    )

    # Confidence rationale
    if data_coverage < 0.4:
        conf_rationale = "Low data coverage — confidence degraded"
    elif macro_score >= 70:
        conf_rationale = f"Strong macro score ({macro_score:.0f}/100) → positive confidence modifier"
    elif macro_score >= 55:
        conf_rationale = f"Solid macro score ({macro_score:.0f}/100) → mild positive modifier"
    elif macro_score >= 40:
        conf_rationale = f"Below-average macro score ({macro_score:.0f}/100) → mild negative modifier"
    elif macro_score >= 25:
        conf_rationale = f"Weak macro score ({macro_score:.0f}/100) → significant negative modifier"
    else:
        conf_rationale = f"Very weak macro score ({macro_score:.0f}/100) → maximum negative modifier"
    if data_coverage < 1.0 and data_coverage >= 0.4:
        conf_rationale += f"; coverage penalty ({data_coverage:.0%} indicators available)"

    # Reasoning summary
    reason_parts = [f"Macro score {macro_score:.0f}/100 → {regime} ({cycle_phase})."]
    if bearish:
        reason_parts.append(f"Key concerns: {bearish[0].split(' —')[0]}.")
    if bullish:
        reason_parts.append(f"Positives: {bullish[0].split(' —')[0]}.")
    reason_parts.append(f"Recession risk: {risk}. Sector tilt: {tilt}.")
    reasoning = " ".join(reason_parts)

    return MacroAssessment(
        macro_score=round(macro_score, 1),
        macro_regime=regime,
        recession_risk_level=risk,
        confidence_modifier=round(conf_mod, 3),
        sector_tilt=tilt,
        reasoning_summary=reasoning,
        bullish_macro_factors=bullish,
        bearish_macro_factors=bearish,
        data_coverage=round(data_coverage, 2),
        cycle_phase=cycle_phase,
        lei_trend=cli_trend,
        yield_spread_trend=spread_trend,
        confidence_adjustment_rationale=conf_rationale,
    )
