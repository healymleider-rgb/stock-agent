"""
macro_overlay — rule-based macro LEI scoring.

Takes a dict of latest indicator values (from FREDProvider.get_lei_snapshot())
and returns a MacroAssessment with score, regime, risk level, and factor lists.

Scoring is intentionally rule-based and transparent.  Each indicator
contributes a bounded sub-score; the weighted average produces the final
macro_score (0–100).  Tune the thresholds and weights at the top of this file.

Indicator weights (must sum to 1.0)
────────────────────────────────────
  Tier 1 — Strongest leading evidence:
    yield_spread     0.22  — T10Y2Y: NY Fed yield-curve recession model
    jobless_claims   0.18  — ICSA: leading at cycle turns

  Tier 2 — Leading, supported by Conference Board LEI methodology:
    consumer_sent    0.15  — UMCSENT: consumer expectations component of LEI
    housing_starts   0.15  — HOUST: building permits / starts component of LEI

  Tier 3 — Coincident, Conference Board CEI components:
    retail_sales     0.12  — RSAFS: personal income proxy
    mfg_prod         0.12  — IPMAN: industrial production (manufacturing)

  Tier 4 — Chronically stale composite:
    oecd_cli         0.06  — OECD CLI: useful when fresh; suppressed when stale

Regime thresholds
─────────────────
  score ≥ 65  → Expansion
  score ≥ 50  → Slowdown / Recovery
  score ≥ 35  → Contraction
  score <  35 → Contraction (severe)

Recession risk
──────────────
  macro_score ≥ 65  → Low
  macro_score ≥ 50  → Moderate
  macro_score ≥ 35  → Elevated
  macro_score <  35 → High
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


# ── Staleness thresholds ───────────────────────────────────────────────────────

_STALE_WEEKLY  = 14    # days: T10Y2Y, ICSA (high-frequency series)
_STALE_MONTHLY = 75    # days: UMCSENT, HOUST, RSAFS, IPMAN (monthly with normal lag)
_STALE_CLI     = 120   # days: OECD CLI (chronically late publisher; suppress beyond this)


# ── Score thresholds ───────────────────────────────────────────────────────────

# Yield curve (T10Y2Y, percentage points)
_YIELD_SPREAD_STRONG   =  0.75   # clearly positive → bullish
_YIELD_SPREAD_FLAT     =  0.10   # flat / borderline
_YIELD_SPREAD_INVERTED = -0.10   # inverted → bearish
_YIELD_SPREAD_DEEP     = -0.50   # deeply inverted → very bearish

# Initial jobless claims (seasonally adjusted, number of persons)
_CLAIMS_LOW    = 215_000
_CLAIMS_RISING = 250_000
_CLAIMS_HIGH   = 300_000
_CLAIMS_SEVERE = 400_000

# University of Michigan Consumer Sentiment (UMCSENT, 0–100 scale)
_SENTIMENT_STRONG = 80.0
_SENTIMENT_OK     = 70.0
_SENTIMENT_WEAK   = 55.0

# Housing starts (HOUST, thousands of units, SAAR)
_STARTS_STRONG = 1_500
_STARTS_OK     = 1_200
_STARTS_WEAK   = 1_000
_STARTS_POOR   =   800

# Retail sales YoY % change (RSAFS)
_RETAIL_STRONG =  5.0
_RETAIL_OK     =  2.0
_RETAIL_FLAT   =  0.0

# Industrial Production: Manufacturing (IPMAN, index 2017=100)
_MFG_STRONG    = 103.0
_MFG_OK        = 100.0
_MFG_WEAK      =  97.0
_MFG_POOR      =  94.0

# OECD Composite Leading Indicator (USALOLITONOSTSAM, centred on 100)
_CLI_STRONG    = 100.5
_CLI_OK        = 100.0
_CLI_WEAK      =  99.5
_CLI_POOR      =  99.0


# Weights (must sum to 1.0)
# Weights derived from tier-based indicator classification.
# Tier hierarchy is supported by LEI/CEI methodology and
# recession forecasting literature (NY Fed yield curve models,
# Conference Board LEI methodology). Within-tier weights are
# design choices and may be adjusted based on regime
# classification performance.
_WEIGHTS = {
    "yield_spread":       0.22,  # Tier 1 — T10Y2Y
    "jobless_claims":     0.18,  # Tier 1 — ICSA
    "consumer_sentiment": 0.15,  # Tier 2 — UMCSENT
    "housing_starts":     0.15,  # Tier 2 — HOUST
    "retail_sales":       0.12,  # Tier 3 — RSAFS
    "mfg_prod":           0.12,  # Tier 3 — IPMAN
    "oecd_cli":           0.06,  # Tier 4 — OECD CLI (suppressed when stale)
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
    macro_score:         float
    macro_regime:        str
    recession_risk_level: str
    confidence_modifier: float
    sector_tilt:         str
    reasoning_summary:   str
    bullish_macro_factors: list[str] = field(default_factory=list)
    bearish_macro_factors: list[str] = field(default_factory=list)
    data_coverage:       float = 1.0
    cycle_phase:         str = "unknown"
    lei_trend:           Optional[str] = None    # OECD CLI trend direction
    yield_spread_trend:  Optional[str] = None
    confidence_adjustment_rationale: str = ""


# ── Scoring helpers ────────────────────────────────────────────────────────────

def _score_yield_spread(spread: float) -> tuple[float, list[str], list[str]]:
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


def _score_jobless_claims(claims: float) -> tuple[float, list[str], list[str]]:
    bullish, bearish = [], []
    if claims <= _CLAIMS_LOW:
        sub = 85.0
        bullish.append(f"Initial jobless claims low ({claims:,.0f}) — labour market strong")
    elif claims <= _CLAIMS_RISING:
        sub = 65.0
        bullish.append(f"Initial jobless claims healthy ({claims:,.0f})")
    elif claims <= _CLAIMS_HIGH:
        sub = 40.0
        bearish.append(f"Initial jobless claims elevated ({claims:,.0f}) — labour market softening")
    elif claims <= _CLAIMS_SEVERE:
        sub = 20.0
        bearish.append(f"Initial jobless claims high ({claims:,.0f}) — significant labour market stress")
    else:
        sub = 5.0
        bearish.append(f"Initial jobless claims at crisis levels ({claims:,.0f})")
    return sub, bullish, bearish


def _score_consumer_sentiment(sentiment: float) -> tuple[float, list[str], list[str]]:
    bullish, bearish = [], []
    if sentiment >= _SENTIMENT_STRONG:
        sub = 80.0
        bullish.append(f"Consumer sentiment elevated ({sentiment:.1f}) — households optimistic")
    elif sentiment >= _SENTIMENT_OK:
        sub = 62.0
        bullish.append(f"Consumer sentiment adequate ({sentiment:.1f})")
    elif sentiment >= _SENTIMENT_WEAK:
        sub = 40.0
        bearish.append(f"Consumer sentiment cautious ({sentiment:.1f}) — household confidence below average")
    else:
        sub = 20.0
        bearish.append(f"Consumer sentiment weak ({sentiment:.1f}) — households pessimistic")
    return sub, bullish, bearish


def _score_housing_starts(starts: float) -> tuple[float, list[str], list[str]]:
    bullish, bearish = [], []
    if starts >= _STARTS_STRONG:
        sub = 80.0
        bullish.append(f"Housing starts robust ({starts:,.0f}k) — construction cycle healthy")
    elif starts >= _STARTS_OK:
        sub = 62.0
        bullish.append(f"Housing starts adequate ({starts:,.0f}k)")
    elif starts >= _STARTS_WEAK:
        sub = 40.0
        bearish.append(f"Housing starts below trend ({starts:,.0f}k) — construction activity slowing")
    elif starts >= _STARTS_POOR:
        sub = 22.0
        bearish.append(f"Housing starts weak ({starts:,.0f}k) — significant housing contraction")
    else:
        sub = 8.0
        bearish.append(f"Housing starts very weak ({starts:,.0f}k) — housing sector in distress")
    return sub, bullish, bearish


def _score_retail_sales(retail_yoy: float) -> tuple[float, list[str], list[str]]:
    bullish, bearish = [], []
    if retail_yoy >= _RETAIL_STRONG:
        sub = 80.0
        bullish.append(f"Retail sales robust (+{retail_yoy:.1f}% YoY) — consumer spending strong")
    elif retail_yoy >= _RETAIL_OK:
        sub = 62.0
        bullish.append(f"Retail sales growing (+{retail_yoy:.1f}% YoY)")
    elif retail_yoy >= _RETAIL_FLAT:
        sub = 40.0
        bearish.append(f"Retail sales flat ({retail_yoy:.1f}% YoY) — consumer spending stalling")
    else:
        sub = 20.0
        bearish.append(f"Retail sales declining ({retail_yoy:.1f}% YoY) — consumer retrenchment")
    return sub, bullish, bearish


def _score_mfg_prod(ipman: float) -> tuple[float, list[str], list[str]]:
    """Score Industrial Production: Manufacturing (IPMAN, index 2017=100)."""
    bullish, bearish = [], []
    if ipman >= _MFG_STRONG:
        sub = 80.0
        bullish.append(f"Industrial production (mfg) elevated ({ipman:.1f}) — manufacturing above trend")
    elif ipman >= _MFG_OK:
        sub = 62.0
        bullish.append(f"Industrial production (mfg) at trend ({ipman:.1f})")
    elif ipman >= _MFG_WEAK:
        sub = 42.0
        bearish.append(f"Industrial production (mfg) below trend ({ipman:.1f}) — manufacturing soft")
    elif ipman >= _MFG_POOR:
        sub = 22.0
        bearish.append(f"Industrial production (mfg) weak ({ipman:.1f}) — industrial contraction underway")
    else:
        sub = 10.0
        bearish.append(f"Industrial production (mfg) very weak ({ipman:.1f}) — significant industrial recession")
    return sub, bullish, bearish


def _score_oecd_cli(cli: float) -> tuple[float, list[str], list[str]]:
    """Score OECD Composite Leading Indicator (USALOLITONOSTSAM, centred on 100)."""
    bullish, bearish = [], []
    if cli >= _CLI_STRONG:
        sub = 80.0
        bullish.append(f"OECD CLI above trend ({cli:.2f}) — above-trend growth momentum")
    elif cli >= _CLI_OK:
        sub = 60.0
        bullish.append(f"OECD CLI at trend ({cli:.2f}) — growth on track")
    elif cli >= _CLI_WEAK:
        sub = 40.0
        bearish.append(f"OECD CLI below trend ({cli:.2f}) — growth momentum fading")
    elif cli >= _CLI_POOR:
        sub = 20.0
        bearish.append(f"OECD CLI weak ({cli:.2f}) — leading composite contracting")
    else:
        sub = 10.0
        bearish.append(f"OECD CLI very weak ({cli:.2f}) — contraction signal")
    return sub, bullish, bearish


def _classify_regime(
    score: float,
    yield_spread: Optional[float],
    claims: Optional[float],
) -> str:
    if score >= _EXPANSION_MIN:
        return "Expansion"

    if score >= _SLOWDOWN_MIN:
        if yield_spread is not None and yield_spread >= _YIELD_SPREAD_INVERTED:
            if claims is None or claims <= _CLAIMS_RISING:
                return "Recovery"
        return "Slowdown"

    if score >= _CONTRACTION_MIN:
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
    cli_trend receives the OECD CLI trend direction (lei_trend from snapshot).

    Recovery  → "early"
    Expansion → "mid" by default; "late" when spread tight (<0.25pp) OR CLI falling
    Slowdown  → "late" by default; "early" if CLI inflecting up AND spread recovering
    Contraction → "contraction"
    """
    if regime == "Contraction":
        return "contraction"
    if regime == "Recovery":
        return "early"
    if regime == "Expansion":
        spread_tight = yield_spread is not None and yield_spread < 0.25
        cli_fading   = cli_trend == "falling"
        if spread_tight or cli_fading:
            return "late"
        return "mid"
    if regime == "Slowdown":
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
    if data_coverage < 0.4:
        base = -0.05
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
    coverage_penalty = (1.0 - data_coverage) * 0.05
    return max(-0.10, min(0.05, base - coverage_penalty))


# ── Public entry point ─────────────────────────────────────────────────────────

def score(snapshot: dict) -> MacroAssessment:
    """
    Convert a LEI snapshot dict (from FREDProvider.get_lei_snapshot()) into
    a MacroAssessment.

    Reads _observation_dates from the snapshot (if present) to apply per-indicator
    staleness thresholds before scoring.  Stale indicators are excluded from the
    weighted average (their weight is redistributed proportionally) and labelled
    "Stale, excluded." in the bearish factors list.

    Expected snapshot keys (all optional — None or missing → excluded):
      yield_spread_10y2y, jobless_claims, consumer_sentiment, housing_starts,
      retail_sales_yoy, mfg_prod, oecd_cli,
      lei_trend, yield_spread_trend,
      _observation_dates   ← injected by FREDProvider; read here, not popped
    """
    obs_dates = snapshot.get("_observation_dates", {})
    today = date.today()

    def _days_old(key: str) -> int:
        d = obs_dates.get(key)
        if not d or not isinstance(d, str):
            return 0
        try:
            return (today - datetime.strptime(d, "%Y-%m-%d").date()).days
        except (ValueError, TypeError):
            return 0

    # Read raw values
    spread    = snapshot.get("yield_spread_10y2y")
    claims    = snapshot.get("jobless_claims")
    sentiment = snapshot.get("consumer_sentiment")
    starts    = snapshot.get("housing_starts")
    retail    = snapshot.get("retail_sales_yoy")
    mfg       = snapshot.get("mfg_prod")
    cli       = snapshot.get("oecd_cli")
    cli_trend    = snapshot.get("lei_trend")
    spread_trend = snapshot.get("yield_spread_trend")

    # Apply per-indicator staleness thresholds.
    # Values that exceed their threshold are nulled and labelled "Stale, excluded."
    # in bearish factors.  Weight is redistributed to the remaining live indicators.
    _stale_notes: list[str] = []

    if spread is not None and _days_old("yield_spread_10y2y") > _STALE_WEEKLY:
        _stale_notes.append("Yield curve (T10Y2Y): Stale, excluded.")
        spread = None
    if claims is not None and _days_old("jobless_claims") > _STALE_WEEKLY:
        _stale_notes.append("Jobless claims (ICSA): Stale, excluded.")
        claims = None
    if sentiment is not None and _days_old("consumer_sentiment") > _STALE_MONTHLY:
        _stale_notes.append("Consumer sentiment (UMCSENT): Stale, excluded.")
        sentiment = None
    if starts is not None and _days_old("housing_starts") > _STALE_MONTHLY:
        _stale_notes.append("Housing starts (HOUST): Stale, excluded.")
        starts = None
    if retail is not None and _days_old("retail_sales_yoy") > _STALE_MONTHLY:
        _stale_notes.append("Retail sales (RSAFS): Stale, excluded.")
        retail = None
    if mfg is not None and _days_old("mfg_prod") > _STALE_MONTHLY:
        _stale_notes.append("Industrial production/mfg (IPMAN): Stale, excluded.")
        mfg = None
    if cli is not None and _days_old("oecd_cli") > _STALE_CLI:
        _stale_notes.append("OECD CLI: Stale, excluded.")
        cli = None

    print(
        f"  [MACRO] raw inputs:"
        f" yield_spread={spread} jobless_claims={claims}"
        f" consumer_sentiment={sentiment} housing_starts={starts}"
        f" retail_sales_yoy={retail} mfg_prod={mfg} oecd_cli={cli}"
        + (f" [{len(_stale_notes)} stale]" if _stale_notes else "")
    )

    # Map weight-keys to (input_value, scored_value, bullish, bearish)
    # Scoring functions are only called for non-None values.
    bullish: list[str] = []
    bearish: list[str] = []
    _vals:   dict[str, Optional[float]] = {}
    _scores: dict[str, float] = {}

    def _score_if_avail(key: str, val: Optional[float], fn) -> None:
        _vals[key] = val
        if val is not None:
            sub, b, bad = fn(val)
            _scores[key] = sub
            bullish.extend(b)
            bearish.extend(bad)
        else:
            _scores[key] = 50.0  # unused (excluded from average)

    _score_if_avail("yield_spread",       spread,    _score_yield_spread)
    _score_if_avail("jobless_claims",     claims,    _score_jobless_claims)
    _score_if_avail("consumer_sentiment", sentiment, _score_consumer_sentiment)
    _score_if_avail("housing_starts",     starts,    _score_housing_starts)
    _score_if_avail("retail_sales",       retail,    _score_retail_sales)
    _score_if_avail("mfg_prod",           mfg,       _score_mfg_prod)
    _score_if_avail("oecd_cli",           cli,       _score_oecd_cli)

    # Add stale notes last so they appear after the scored-indicator notes
    bearish.extend(_stale_notes)

    print(
        f"  [MACRO] sub-scores: spread={_scores['yield_spread']:.0f}"
        f" claims={_scores['jobless_claims']:.0f}"
        f" sentiment={_scores['consumer_sentiment']:.0f}"
        f" housing={_scores['housing_starts']:.0f}"
        f" retail={_scores['retail_sales']:.0f}"
        f" mfg={_scores['mfg_prod']:.0f}"
        f" cli={_scores['oecd_cli']:.0f}"
    )

    # Weighted average — exclude indicators whose input value is None
    _avail_w = sum(w for k, w in _WEIGHTS.items() if _vals[k] is not None)
    if _avail_w == 0:
        macro_score = 50.0
    else:
        macro_score = sum(
            _scores[k] * w / _avail_w
            for k, w in _WEIGHTS.items()
            if _vals[k] is not None
        )

    # data_coverage = fraction of weight that had real data
    available    = sum(1 for v in _vals.values() if v is not None)
    data_coverage = available / len(_vals)

    print(
        f"  [MACRO] weighted macro_score={macro_score:.1f}/100"
        f"  avail_weight={_avail_w:.2f}"
        f"  data_coverage={data_coverage:.0%}"
    )

    regime      = _classify_regime(macro_score, spread, claims)
    cycle_phase = _classify_cycle_phase(regime, cli_trend, spread, spread_trend)
    risk        = _recession_risk(macro_score)
    conf_mod    = _confidence_modifier(macro_score, data_coverage)
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
