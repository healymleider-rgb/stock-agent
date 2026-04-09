"""
Aggregator: combines all CategoryScores into a final Scorecard.

Also assigns bullish/bearish factors and determines overall stance.
"""
from __future__ import annotations

from typing import Optional

from models.scorecard import CategoryScore, Scorecard, Stance
from models.stock_data import StockData
from config import Config


# ── Macro overlay helpers ──────────────────────────────────────────────────────

def _apply_macro_overlay(sc: Scorecard, macro: dict) -> None:
    """
    Adjust assembled category scores, overall_score, and stance based on macro
    regime findings.  Called in two passes inside build_scorecard():

      Pass 1 (before compute_overall_score):
        - Valuation score: soften in Expansion, amplify penalty in Contraction
        - Growth / risk weights: shift slightly toward growth in Expansion,
          toward risk in Contraction

      Pass 2 (after compute_overall_score, before determine_stance):
        - Apply a small macro_score delta (±3 pts max)
        - Downgrade stance one level if recession_risk is "High"

    All adjustments are additive and bounded so nothing goes below 0 or above 100.
    """
    regime       = macro.get("macro_regime") or macro.get("regime") or "Unknown"
    macro_score  = macro.get("macro_score", 50.0)
    rec_risk     = macro.get("recession_risk_level") or macro.get("recession_risk") or "Unknown"
    bullish_macro = macro.get("bullish_macro_factors", [])
    bearish_macro = macro.get("bearish_macro_factors", [])

    # ── DEBUG: pre-overlay state ───────────────────────────────────────────────
    _pre_val  = sc.valuation.score  if sc.valuation  else None
    _pre_grow = sc.growth.score     if sc.growth     else None
    print(
        f"  [MacroOverlay] regime={regime!r}  macro_score={macro_score:.1f}"
        f"  recession_risk={rec_risk!r}"
    )
    print(
        f"  [MacroOverlay] pre-overlay: "
        f"val_score={_pre_val}  growth_score={_pre_grow}"
    )

    # ── Pass 1: category adjustments ──────────────────────────────────────────
    _adjust_for_regime(sc, regime)

    # ── Recompute overall score with mutated weights/scores ───────────────────
    # (base score was already computed in build_scorecard before this call)
    sc.compute_overall_score()
    print(
        f"  [MacroOverlay] after regime adjustment: "
        f"val_score={sc.valuation.score:.2f}  "
        f"overall={sc.overall_score:.2f}"
        if sc.valuation else
        f"  [MacroOverlay] after regime adjustment: val_score=N/A  overall={sc.overall_score:.2f}"
    )

    # ── Pass 2: macro_score delta ─────────────────────────────────────────────
    # (macro_score − 50) / 50 * 3  →  range [−3, +3]
    delta = (macro_score - 50.0) / 50.0 * 3.0
    sc.overall_score = max(0.0, min(100.0, sc.overall_score + delta))
    print(
        f"  [MacroOverlay] after macro delta ({delta:+.2f}): "
        f"overall={sc.overall_score:.2f}"
    )

    # ── Stance ────────────────────────────────────────────────────────────────
    sc.determine_stance()
    stance_before_override = sc.stance.value
    print(f"  [MacroOverlay] stance before recession override: {stance_before_override!r}")

    # Downgrade stance one level on High recession risk
    if rec_risk == "High":
        if sc.stance == Stance.BULLISH:
            sc.stance = Stance.NEUTRAL
            sc.bearish_factors.insert(0, "Macro: High recession risk — Bullish stance tempered to Neutral")
        elif sc.stance == Stance.NEUTRAL:
            sc.stance = Stance.BEARISH
            sc.bearish_factors.insert(0, "Macro: High recession risk — Neutral stance downgraded to Bearish")

    print(
        f"  [MacroOverlay] final stance: {sc.stance.value!r}"
        + (f" (downgraded from {stance_before_override!r})" if sc.stance.value != stance_before_override else "")
    )

    # ── Surface macro factors in the scorecard ────────────────────────────────
    for f in bullish_macro[:2]:
        sc.bullish_factors.append(f"Macro: {f}")
    for f in bearish_macro[:2]:
        sc.bearish_factors.append(f"Macro: {f}")

    # Trim combined lists to 6 each
    sc.bullish_factors = list(dict.fromkeys(sc.bullish_factors))[:6]
    sc.bearish_factors = list(dict.fromkeys(sc.bearish_factors))[:6]


def _adjust_for_regime(sc: Scorecard, regime: str) -> None:
    """
    Mutate CategoryScore weights and scores based on macro regime.

    Expansion:
      - Valuation: soften the penalty for expensive stocks (low scores move up slightly).
        Rationale: high multiples are more justifiable in a growing economy.
      - Growth weight gets +0.03, valuation weight gives −0.03.

    Contraction:
      - Valuation: amplify the penalty for expensive stocks.
        Rationale: multiple compression accelerates in recessions.
      - Growth weight gets −0.03, risk weight gets +0.03.

    Recovery:
      - Growth weight gets +0.02, risk weight gives −0.02.
        Rationale: early recovery rewards growth re-acceleration.

    Slowdown / Unknown: no adjustment.
    """
    val  = sc.valuation
    grow = sc.growth
    risk = sc.risk

    if regime == "Expansion":
        grow_score = grow.score if (grow is not None and grow.data_quality != "missing") else None

        if grow_score is not None and grow_score > 80:
            tier = "HIGH"          # full boost
            val_soften  = 0.12     # soften valuation penalty
            weight_shift = 0.03    # val → growth
        elif grow_score is not None and grow_score >= 65:
            tier = "MID"           # small boost
            val_soften  = 0.0      # no valuation score change
            weight_shift = 0.01
        else:
            tier = "LOW"           # near-zero — defensive/slow-growth names
            val_soften  = 0.0
            weight_shift = 0.0

        print(
            f"  [MacroOverlay/_adjust] Expansion branch: "
            f"growth_score={grow_score}  tier={tier}  "
            f"val_soften={val_soften}  weight_shift={weight_shift}"
        )

        if val_soften > 0 and val is not None and val.data_quality != "missing" and val.score < 50:
            before = val.score
            val.score = min(100.0, val.score + (50.0 - val.score) * val_soften)
            print(
                f"  [MacroOverlay/_adjust] {tier}: "
                f"val_score {before:.2f} → {val.score:.2f} (+{val.score - before:.2f})"
            )

        if weight_shift > 0 and val is not None and grow is not None:
            shift = min(weight_shift, val.weight)
            val.weight  = round(val.weight  - shift, 4)
            grow.weight = round(grow.weight + shift, 4)
            print(
                f"  [MacroOverlay/_adjust] {tier}: weight shift "
                f"val_w={val.weight:.4f}  grow_w={grow.weight:.4f}"
            )

    elif regime == "Contraction":
        # Amplify valuation penalty: expensive stocks get pushed down further
        if val is not None and val.data_quality != "missing" and val.score < 50:
            adjustment = (50.0 - val.score) * 0.18
            val.score = max(0.0, val.score - adjustment)
        # Shift 3% weight from growth → risk
        if grow is not None and risk is not None:
            shift = min(0.03, grow.weight)
            grow.weight = round(grow.weight - shift, 4)
            risk.weight = round(risk.weight + shift, 4)

    elif regime == "Recovery":
        # Shift 2% weight from risk → growth
        if grow is not None and risk is not None:
            shift = min(0.02, risk.weight)
            risk.weight = round(risk.weight - shift, 4)
            grow.weight = round(grow.weight + shift, 4)


def build_scorecard(
    ticker: str,
    valuation: CategoryScore,
    growth: CategoryScore,
    profitability: CategoryScore,
    financial_health: CategoryScore,
    momentum: CategoryScore,
    risk: CategoryScore,
    risk_flags: list[str],
    confidence: float = 0.0,
    macro_findings: Optional[dict] = None,
) -> Scorecard:
    """
    Assemble a fully computed Scorecard.

    macro_findings (optional): payload from MacroLEIAgent.  When present,
    the macro overlay adjusts valuation/growth weights, applies a small
    macro_score delta, and may downgrade stance on high recession risk.
    """
    # !! SENTINEL — if this line never prints, __pycache__ is stale.
    # Fix: find . -name __pycache__ -exec rm -rf {} + && python main.py <ticker>
    print(f"\n  !! [SCORER build_scorecard ENTRY] ticker={ticker}  id(build_scorecard)={id(build_scorecard)}")

    sc = Scorecard(ticker=ticker)
    sc.valuation = valuation
    sc.growth = growth
    sc.profitability = profitability
    sc.financial_health = financial_health
    sc.momentum = momentum
    sc.risk = risk
    sc.risk_flags = risk_flags
    sc.confidence = confidence

    # Compute base score first so it's visible in debug output regardless of
    # whether the overlay runs.  The overlay will call compute_overall_score()
    # again after mutating weights/scores.
    sc.compute_overall_score()
    print(
        f"  [Scorer] base overall_score (pre-macro) = {sc.overall_score:.2f}  "
        f"ticker={ticker}"
    )

    _macro_regime = None
    if macro_findings:
        _macro_regime = macro_findings.get("macro_regime") or macro_findings.get("regime")
    print(f"  [Scorer] resolved macro_regime={_macro_regime!r}")
    _overlay_will_run = bool(macro_findings) and _macro_regime not in (None, "Unknown")
    print(
        f"  [Scorer] macro_findings present={bool(macro_findings)}  "
        f"macro_regime={_macro_regime!r}  overlay_will_run={_overlay_will_run}"
    )

    if _overlay_will_run:
        _apply_macro_overlay(sc, macro_findings)
        print(
            f"  [Scorer] final overall_score (post-macro) = {sc.overall_score:.2f}  "
            f"stance={sc.stance.value!r}"
        )
    else:
        sc.determine_stance()
        print(
            f"  [Scorer] macro overlay skipped — "
            f"{'FRED_API_KEY not set or regime=Unknown; set FRED_API_KEY to enable macro overlay' if not macro_findings or _macro_regime in (None, 'Unknown') else 'no macro findings'}"
        )

    # ── Derive bullish and bearish factors ────────────────────────────────────
    categories = [valuation, growth, profitability, financial_health, momentum, risk]
    for cat in categories:
        strong = cat.score >= 70
        weak = cat.score < 45
        for factor in cat.factors:
            if "[RISK]" in factor:
                sc.bearish_factors.append(factor.replace("[RISK] ", ""))
            elif strong:
                sc.bullish_factors.append(f"{cat.name.title()}: {factor}")
            elif weak:
                sc.bearish_factors.append(f"{cat.name.title()}: {factor}")

    # Deduplicate
    sc.bullish_factors = list(dict.fromkeys(sc.bullish_factors))[:6]
    sc.bearish_factors = list(dict.fromkeys(sc.bearish_factors))[:6]

    # ── Key drivers (top 3 highest-weighted categories) ───────────────────────
    sorted_cats = sorted(categories, key=lambda c: abs(c.score - 50), reverse=True)
    for cat in sorted_cats[:3]:
        sc.key_drivers.append(
            f"{cat.name.replace('_', ' ').title()} ({cat.score:.0f}/100) — {cat.reasoning}"
        )

    # ── What would change the view ────────────────────────────────────────────
    if sc.stance.value == "Bullish":
        sc.what_would_change_view = [
            "Material deterioration in revenue growth or margins",
            "Significant increase in leverage or debt taken on",
            "Breakdown below 200-day moving average on heavy volume",
            "Earnings miss or guidance cut in the next 1-2 quarters",
        ]
    elif sc.stance.value == "Bearish":
        sc.what_would_change_view = [
            "Return to positive free cash flow generation",
            "Evidence of revenue stabilization or reacceleration",
            "Successful balance sheet deleveraging",
            "Multiple positive earnings surprises that reset expectations",
        ]
    else:
        sc.what_would_change_view = [
            "Sustained revenue growth acceleration above 15%",
            "Clear margin expansion over two or more consecutive quarters",
            "A catalyst event: new product, partnership, or market share gain",
            "Significant deterioration in fundamentals would shift view to Bearish",
        ]

    return sc
