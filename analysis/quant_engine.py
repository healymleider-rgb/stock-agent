"""
quant_engine.py
===============
Unified 8-step quantitative investment decision engine.

Integrates all alpha-layer outputs into a single, auditable result that
traces every number back to its source (regression / factor model / MC /
scenario tree).  Designed as a read-only consumer — it does NOT re-run
any computation; it enriches and structures `AlphaPipeline` outputs for
the reporting and API layers.

8-Step Architecture
-------------------
  Step 1  — Factor model enrichment
              Cross-sectional z-scores, top-2 positive/negative factor
              contributions, regime-adjusted composite.
  Step 2  — Historical regression drivers
              Factor-premium drivers ranked by |contribution|, plus
              HRL models: AR(1) rho, margin trend, valuation MR, drawdown.
  Step 3  — MC inputs (regression-derived)
              Three-way growth blend components, sigma asymmetry driver,
              MR speed override, shock floor.  Traces every MC param.
  Step 4  — Scenario tree summary
              Bear/Base/Bull leaves, weighted E[R], downside/upside mass,
              VaR-95, shock probability, top-6 leaves by probability.
  Step 5  — MC distribution
              Full return distribution percentiles, probabilities, conviction
              score from DistributionProfile (return / skew / risk dims).
  Step 6  — Position sizing decision
              6-step audit: Kelly → conviction adjustment → tail cap →
              divergence cap → coherence penalty → final recommendation.
  Step 7  — Coherence assessment
              Coherence issues with per-issue resolution guidance;
              overall signal-reliability rating.
  Step 8  — JSON output layer
              `QuantEngineResult.to_dict()` — all steps, fully serialisable.

Integration
-----------
    from analysis.quant_engine import QuantEngine

    qe = QuantEngine()
    result = qe.build(
        ticker       = "AAPL",
        alpha_outputs = alpha,       # AlphaEngineOutputs from AlphaPipeline.run()
        val_range    = val_range,    # mutated by pipeline — val_range.mc is enriched
    )
    findings["quant_engine"] = result.to_dict()   # store for API
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from analysis.alpha_pipeline import AlphaEngineOutputs
    from analysis.valuation_range import ValuationRange


# ── Resolution guidance for coherence issues ─────────────────────────────────

_COHERENCE_RESOLUTIONS: List[tuple] = [
    (
        "COHERENCE: style=",
        "Verify macro regime assignment. Markov prior may be penalising structural "
        "quality unfairly. If regime is confirmed, quality signal may be lagging — "
        "wait for next earnings report before adjusting weights.",
    ),
    (
        "COHERENCE: margin_trend_slope",
        "Check peer set appropriateness. HRL time-series signals margin recovery "
        "in progress — cross-sectional peer comparison has not yet reflected it. "
        "Consider upgrading profitability weight if slope persists 2+ quarters.",
    ),
    (
        "COHERENCE NOTE: AR(1)",
        "No action required. Structural earnings quality and macro tail risk "
        "co-exist legitimately in late-cycle / contraction regimes. Monitor "
        "shock_prob quarterly; if > 35% for 2+ cycles, consider trimming.",
    ),
    (
        "COHERENCE: MC mean_return",
        "Review macro regime assignment. MC growth inputs and Markov prior may use "
        "inconsistent regime assumptions. Re-run with a single unified regime label "
        "and compare factor vs scenario expected returns again.",
    ),
]


def _resolve_issue(issue_text: str) -> str:
    """Return resolution guidance for a coherence issue string."""
    for prefix, guidance in _COHERENCE_RESOLUTIONS:
        if issue_text.startswith(prefix):
            return guidance
    return (
        "Review underlying data inputs and model assumptions for this cross-layer conflict."
    )


# ── Internal extraction helpers ───────────────────────────────────────────────

def _extract_factor_step(fp) -> Dict[str, Any]:
    """
    Step 1: Factor model enrichment.

    Computes top-2 positive and top-2 negative factor contributions
    (effective_weight × z-score) beyond just strongest/weakest.
    """
    if fp is None:
        return {"available": False}

    # factor_name → FactorProfile z-score attribute
    z_map: Dict[str, float] = {
        "quality":       getattr(fp, "quality_z",       0.0),
        "profitability": getattr(fp, "profitability_z", 0.0),
        "growth":        getattr(fp, "growth_z",        0.0),
        "value":         getattr(fp, "value_z",         0.0),
        "momentum":      getattr(fp, "momentum_z",      0.0),
        "lowvol":        getattr(fp, "stability_z",     0.0),  # stability_z ≡ low-vol factor
        "macro":         getattr(fp, "macro_z",         0.0),
    }

    weights: Dict[str, float] = getattr(fp, "effective_weights", {}) or {}

    contributions: Dict[str, float] = {
        f: weights.get(f, 0.0) * z
        for f, z in z_map.items()
    }

    # Sort all factors by contribution (highest first)
    sorted_c = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)

    top_pos = [
        {
            "factor":       f,
            "contribution": round(c, 4),
            "z_score":      round(z_map[f], 3),
            "weight":       round(weights.get(f, 0.0), 3),
        }
        for f, c in sorted_c
        if c > 0
    ][:2]

    top_neg = [
        {
            "factor":       f,
            "contribution": round(c, 4),
            "z_score":      round(z_map[f], 3),
            "weight":       round(weights.get(f, 0.0), 3),
        }
        for f, c in reversed(sorted_c)
        if c < 0
    ][:2]

    return {
        "available":            True,
        "composite_score":      getattr(fp, "composite_score",     None),
        "style_label":          getattr(fp, "style_label",         ""),
        "confidence":           getattr(fp, "confidence",          "low"),
        "n_peers":              getattr(fp, "n_peers",             0),
        "z_scores":             {f: round(z, 3) for f, z in z_map.items()},
        "cdf_scores": {
            "quality":       getattr(fp, "quality_factor",      None),
            "profitability": getattr(fp, "profitability_score", None),
            "macro":         getattr(fp, "macro_score",         None),
        },
        "effective_weights":    {f: round(w, 4) for f, w in weights.items()},
        "contributions":        {f: round(c, 4) for f, c in contributions.items()},
        "top_positive_factors": top_pos,
        "top_negative_factors": top_neg,
        "pm_interpretation":    getattr(fp, "pm_interpretation", ""),
        "strongest_factor":     getattr(fp, "strongest_factor",  ""),
        "weakest_factor":       getattr(fp, "weakest_factor",    ""),
    }


def _extract_regression_step(rc, hrl) -> Dict[str, Any]:
    """
    Step 2: Historical regression drivers.

    Factor-premium contributions ranked by |contribution| + full HRL model outputs.
    """
    if rc is None and hrl is None:
        return {"available": False}

    result: Dict[str, Any] = {"available": True}

    if rc is not None:
        fb: Dict[str, float] = getattr(rc, "factor_betas", {}) or {}
        total_abs = sum(abs(v) for v in fb.values()) or 1.0
        drivers = sorted(
            [
                {
                    "factor":       f,
                    "contribution": round(v, 4),
                    "pct_of_total": round(abs(v) / total_abs * 100, 1),
                    "direction":    "positive" if v >= 0 else "negative",
                }
                for f, v in fb.items()
            ],
            key=lambda x: abs(x["contribution"]),
            reverse=True,
        )
        result["regression"] = {
            "expected_return":  round(rc.expected_return,  4),
            "tracking_error":   round(rc.tracking_error,   4),
            "alpha":            round(rc.alpha,             4),
            "r_squared":        round(rc.r_squared,         3),
            "n_obs":            rc.n_obs,
            "confidence":       rc.confidence,
            "method":           rc.method,
            "factor_drivers":   drivers,
        }

    if hrl is not None:
        result["hrl"] = {
            # Model A — AR(1) EPS persistence
            "ar1_rho":            round(getattr(hrl, "ar1_eps_persistence", 0.0), 3),
            "ar1_growth_est":     round(getattr(hrl, "ar1_growth_estimate", 0.0), 4),
            "ar1_r2":             round(getattr(hrl, "ar1_r2",              0.0), 3),
            "ar1_n":              getattr(hrl, "ar1_n", 0),
            # Model B — Margin trend
            "margin_slope":       round(getattr(hrl, "margin_trend_slope",  0.0), 4),
            "margin_r2":          round(getattr(hrl, "margin_trend_r2",     0.0), 3),
            "margin_series":      getattr(hrl, "margin_series_used", "none"),
            # Model C — Valuation mean reversion
            "valuation_mr_speed":  round(getattr(hrl, "valuation_mr_speed",  0.0), 3),
            "valuation_mr_target": getattr(hrl, "valuation_mr_target", None),
            "valuation_mr_r2":     round(getattr(hrl, "valuation_mr_r2",    0.0), 3),
            # Model D — Macro sensitivity
            "macro_sensitivity":   round(getattr(hrl, "macro_sensitivity",   0.0), 3),
            "macro_sensitivity_r2":round(getattr(hrl, "macro_sensitivity_r2",0.0), 3),
            "macro_n":             getattr(hrl, "macro_n", 0),
            # Model F — Drawdown
            "max_drawdown_avg":   round(getattr(hrl, "max_drawdown_avg",    0.0), 3),
            "max_drawdown_worst": round(getattr(hrl, "max_drawdown_worst",  0.0), 3),
            # Synthesised
            "calibrated_growth":  round(getattr(hrl, "calibrated_growth_mean", 0.0), 4),
            "confidence":         getattr(hrl, "hrl_confidence", "low"),
            "diagnostics":        dict(getattr(hrl, "diagnostics", {}) or {}),
        }

    return result


def _extract_mc_inputs_step(hrl, rc, val_range) -> Dict[str, Any]:
    """
    Step 3: MC inputs (regression-derived).

    Reconstructs the three-way growth blend and traces every MC parameter
    to its source, mirroring _apply_layer_overrides() in monte_carlo.py.
    """
    if hrl is None and rc is None:
        return {"available": False}

    result: Dict[str, Any] = {"available": True}

    # ── Three-way blend components ────────────────────────────────────────────
    mc_gr_raw = getattr(val_range, "eps_growth_rate", None) if val_range else None
    mc_gr_dec = mc_gr_raw / 100.0 if mc_gr_raw is not None else None
    reg_er    = getattr(rc, "expected_return", None) if rc else None
    ar1_g     = getattr(hrl, "ar1_growth_estimate", None) if hrl else None
    hrl_conf  = getattr(hrl, "hrl_confidence", "low") if hrl else "low"

    # Mirror blend-weight logic from historical_regression.py
    w_mc  = 0.40
    w_reg = 0.30
    w_ar1 = 0.30

    if hrl_conf == "low":
        w_mc  += w_ar1 * 0.50
        w_ar1  = w_ar1 * 0.50

    if rc is None or reg_er is None:
        w_mc  += w_reg
        w_reg  = 0.0

    blend: Dict[str, Any] = {}
    if mc_gr_dec is not None:
        blend["fundamental"] = {
            "value":  round(mc_gr_dec, 4),
            "weight": round(w_mc,      3),
            "source": "ValuationRange.eps_growth_rate",
        }
    if reg_er is not None and w_reg > 0:
        blend["factor_regression"] = {
            "value":  round(reg_er, 4),
            "weight": round(w_reg,  3),
            "source": "RegressionCalibration.expected_return",
        }
    if ar1_g is not None and w_ar1 > 0:
        blend["ar1_hrl"] = {
            "value":  round(ar1_g, 4),
            "weight": round(w_ar1, 3),
            "source": "HRLResult.ar1_growth_estimate",
        }

    # Blended growth mean
    blended = sum(
        comp["value"] * comp["weight"] for comp in blend.values()
    ) if blend else None
    result["blend_components"]    = blend
    result["blended_growth_mean"] = round(blended, 4) if blended is not None else None
    result["hrl_calibrated"]      = round(getattr(hrl, "calibrated_growth_mean", 0.0), 4) if hrl else None

    # ── Sigma asymmetry driver ────────────────────────────────────────────────
    if hrl is not None:
        slope = getattr(hrl, "margin_trend_slope", 0.0)
        result["sigma_asymmetry"] = {
            "margin_trend_slope":  round(slope, 4),
            "interpretation": (
                "Negative trend → σ_down > σ_up (fatter left tail)"
                if slope < 0
                else "Positive trend → σ_up ≥ σ_down (right skew maintained)"
            ),
        }

        result["mr_speed_override"] = {
            "valuation_mr_speed":  round(getattr(hrl, "valuation_mr_speed",  0.0), 3),
            "valuation_mr_target": getattr(hrl, "valuation_mr_target", None),
            "source":              "HRLResult.valuation_mr_speed",
        }

        result["shock_floor"] = {
            "max_drawdown_avg":    round(getattr(hrl, "max_drawdown_avg", 0.0), 3),
            "max_drawdown_worst":  round(getattr(hrl, "max_drawdown_worst", 0.0), 3),
            "source":              "HRLResult.max_drawdown_avg",
            "note":                "Shock scenario floor anchored to realised peak-to-trough history.",
        }

    # ── Regression tracking error → MC sigma baseline ────────────────────────
    if rc is not None:
        result["tracking_error_baseline"] = {
            "value":  round(rc.tracking_error, 4),
            "source": "RegressionCalibration.tracking_error",
            "note":   "Annualised residual std after market attribution — seeds σ calibration.",
        }

    return result


def _extract_scenario_step(st) -> Dict[str, Any]:
    """Step 4: Scenario tree summary — Bear/Base/Bull + dispersion metrics."""
    if st is None:
        return {"available": False}

    leaves = getattr(st, "leaves", []) or []

    result: Dict[str, Any] = {
        "available":        True,
        "macro_regime":     getattr(st, "macro_regime", ""),
        "method":           getattr(st, "method",       ""),
        "n_leaves":         len(leaves),
        "weighted_return":  round(getattr(st, "weighted_return",  0.0), 4),
        "scenario_std":     round(getattr(st, "scenario_std",     0.0), 4),
        "var_95":           round(getattr(st, "var_95",           0.0), 4),
        "downside_mass":    round(getattr(st, "downside_mass",    0.0), 3),
        "upside_mass":      round(getattr(st, "upside_mass",      0.0), 3),
        "concentration_3":  round(getattr(st, "concentration_3",  0.0), 3),
        "shock_prob":       round(getattr(st, "shock_prob",       0.0), 4),
        "shock_mean_growth":round(getattr(st, "shock_mean_growth",0.0), 4),
        # Legacy MC bridge fields
        "bear_price":       getattr(st, "bear_price",   None),
        "base_price":       getattr(st, "base_price",   None),
        "bull_price":       getattr(st, "bull_price",   None),
        "bear_multiple":    getattr(st, "bear_multiple",None),
        "base_multiple":    getattr(st, "base_multiple",None),
        "bull_multiple":    getattr(st, "bull_multiple",None),
    }

    # Bull leaf (best_case)
    bc = getattr(st, "best_case", None)
    if bc is not None:
        result["bull"] = {
            "label":           bc.label,
            "path":            bc.path,
            "probability":     round(bc.probability, 4),
            "target_price":    round(bc.target_price, 2) if bc.target_price else None,
            "expected_return": round(bc.expected_return, 4),
            "eps_growth_adj":  round(bc.eps_growth_adj, 4),
            "multiple_adj":    round(bc.multiple_adj, 2),
        }

    # Bear leaf (worst_case)
    wc = getattr(st, "worst_case", None)
    if wc is not None:
        result["bear"] = {
            "label":           wc.label,
            "path":            wc.path,
            "probability":     round(wc.probability, 4),
            "target_price":    round(wc.target_price, 2) if wc.target_price else None,
            "expected_return": round(wc.expected_return, 4),
            "eps_growth_adj":  round(wc.eps_growth_adj, 4),
            "multiple_adj":    round(wc.multiple_adj, 2),
        }

    # Top-6 leaves by probability for PM drill-down
    sorted_leaves = sorted(leaves, key=lambda l: getattr(l, "probability", 0.0), reverse=True)
    result["top_leaves"] = [
        {
            "label":           getattr(l, "label",           ""),
            "path":            getattr(l, "path",            ""),
            "macro_regime":    getattr(l, "macro_regime",    ""),
            "execution":       getattr(l, "execution",       ""),
            "multiple_rxn":    getattr(l, "multiple_rxn",    ""),
            "probability":     round(getattr(l, "probability",     0.0), 4),
            "expected_return": round(getattr(l, "expected_return", 0.0), 4),
            "target_price":    round(getattr(l, "target_price",    0.0), 2),
            "eps_growth_adj":  round(getattr(l, "eps_growth_adj",  0.0), 4),
            "multiple_adj":    round(getattr(l, "multiple_adj",    0.0), 2),
        }
        for l in sorted_leaves[:6]
    ]

    return result


def _extract_mc_step(mc) -> tuple:
    """
    Step 5: MC distribution.

    Returns (dist_dict, DistributionProfile_or_None).
    DistributionProfile is returned separately so Step 6 can use it.
    """
    if mc is None:
        return {"available": False}, None

    dp = None
    try:
        from analysis.monte_carlo import distribution_profile as _dp_fn
        dp = _dp_fn(mc)
    except Exception:
        pass

    iqr = getattr(mc, "p75_return", 0.0) - getattr(mc, "p25_return", 0.0)

    result: Dict[str, Any] = {
        "available":       True,
        "n_sims":          mc.n_sims,
        "horizon_years":   mc.horizon_years,
        "method":          mc.method,
        # Growth assumptions
        "growth_mean":     round(mc.growth_mean, 4),
        "growth_std":      round(mc.growth_std,  4),
        # Return distribution
        "mean_return":     round(mc.mean_return,   4),
        "median_return":   round(mc.median_return, 4),
        "p5_return":       round(mc.p5_return,     4),
        "p25_return":      round(mc.p25_return,    4),
        "p75_return":      round(mc.p75_return,    4),
        "p95_return":      round(mc.p95_return,    4),
        "skewness":        round(mc.skewness,      3),
        "iqr":             round(iqr,              4),
        # Probabilities
        "prob_positive":   round(mc.prob_positive, 3),
        "prob_20_gain":    round(mc.prob_20_gain,  3),
        "prob_loss":       round(mc.prob_loss,     3),
        "prob_loss_20":    round(mc.prob_loss_20,  3),
        # Price distribution
        "mean_price":      round(mc.mean_price,   2),
        "p5_price":        round(mc.p5_price,     2),
        "p25_price":       round(mc.p25_price,    2),
        "median_price":    round(mc.median_price, 2),
        "p75_price":       round(mc.p75_price,    2),
        "p95_price":       round(mc.p95_price,    2),
        # Sizing signals
        "kelly_fraction":   round(mc.kelly_fraction,  4),
        "upside_downside":  round(mc.upside_downside, 3),
        "upside_skew_label": mc.upside_skew_label,
        "risk_label":        mc.risk_label,
    }

    if dp is not None:
        result["conviction"] = {
            "score":           round(dp.conviction_score, 1),
            "tier":            dp.conviction_tier,
            "iqr":             round(dp.iqr,             4),
            "width_tier":      dp.width_tier,
            "net_prob":        round(dp.net_prob,         3),
            "return_score":    round(dp.return_score,     1),
            "skew_score":      round(dp.skew_score,       1),
            "risk_score":      round(dp.risk_score,       1),
            "size_adjustment": round(dp.size_adjustment,  3),
            "size_cap":        (
                None if dp.size_cap == float("inf") else round(dp.size_cap, 2)
            ),
            "rationale":       dp.rationale,
        }

    return result, dp


def _compute_sizing_step(mc, dp, alpha_outputs, base_pct: float = 2.0) -> Dict[str, Any]:
    """
    Step 6: 6-step position sizing audit trail.

    Starts from MC half-Kelly, applies conviction-tier adjustment, tail-risk
    hard cap, model-divergence cap, coherence-issues penalty, and produces
    a final recommendation with basis-point audit trail.

    base_pct: reference portfolio allocation (default 2% for a diversified fund).
    """
    steps: List[Dict[str, Any]] = []

    if mc is None:
        return {
            "steps":          [],
            "recommendation": "hold",
            "final_pct":      base_pct,
            "base_pct":       base_pct,
            "reasoning":      "MC simulation not available — defaulting to base allocation.",
        }

    # ── Step 1: MC Kelly fraction ─────────────────────────────────────────────
    kelly      = mc.kelly_fraction
    current    = kelly * 100.0   # convert to percentage (e.g. 0.025 → 2.5)
    steps.append({
        "step":       1,
        "name":       "MC Kelly Fraction",
        "input":      f"half-Kelly={kelly:.3f}",
        "output_pct": round(current, 2),
        "note": (
            f"Half-Kelly from {mc.n_sims:,} simulated paths: "
            f"mean={mc.mean_return:+.1%}, P5={mc.p5_return:+.1%}, "
            f"P(gain)={mc.prob_positive:.0%}."
        ),
    })

    # ── Step 2: Conviction tier adjustment ───────────────────────────────────
    if dp is not None:
        adj        = dp.size_adjustment      # ±1.5 pp (from DistributionProfile)
        current   += adj
        steps.append({
            "step":       2,
            "name":       "Conviction Tier Adjustment",
            "input":      f"tier={dp.conviction_tier}, score={dp.conviction_score:.0f}",
            "delta_pp":   round(adj, 2),
            "output_pct": round(current, 2),
            "note":       dp.rationale,
            "applied":    True,
        })
    else:
        steps.append({
            "step":       2,
            "name":       "Conviction Tier Adjustment",
            "input":      "DistributionProfile unavailable",
            "delta_pp":   0,
            "output_pct": round(current, 2),
            "applied":    False,
            "note":       "Cannot compute conviction tier — MC result required.",
        })

    # ── Step 3: Tail risk hard cap ────────────────────────────────────────────
    p5 = mc.p5_return
    if p5 < -0.30:
        cap, cap_note = 1.0, f"P5={p5:.0%} < -30% → hard cap 1.0%"
    elif p5 < -0.20:
        cap, cap_note = 2.0, f"P5={p5:.0%} < -20% → hard cap 2.0%"
    elif p5 < -0.10:
        cap, cap_note = 3.0, f"P5={p5:.0%} < -10% → hard cap 3.0%"
    else:
        cap, cap_note = float("inf"), f"P5={p5:.1%} — above all tail-risk thresholds."

    applied = cap != float("inf")
    if applied:
        current = min(current, cap)
    steps.append({
        "step":       3,
        "name":       "Tail Risk Hard Cap",
        "input":      f"P5={p5:.1%}",
        "cap_pct":    cap if cap != float("inf") else None,
        "output_pct": round(current, 2),
        "applied":    applied,
        "note":       cap_note,
    })

    # ── Step 4: Model divergence cap ─────────────────────────────────────────
    div_label = getattr(alpha_outputs, "divergence_label", "n/a") if alpha_outputs else "n/a"
    div_pct   = getattr(alpha_outputs, "divergence_pct",   None)  if alpha_outputs else None
    if div_label == "significant":
        current = min(current, 1.5)
        div_note = (
            f"Significant model disagreement ({div_pct:.0%} gap). "
            f"Position capped at 1.5% until divergence resolves."
        )
        applied = True
    else:
        div_note = f"No divergence cap — models are {div_label}."
        applied  = False

    steps.append({
        "step":       4,
        "name":       "Model Divergence Cap",
        "input":      f"divergence={div_label}" + (f", gap={div_pct:.0%}" if div_pct is not None else ""),
        "cap_pct":    1.5 if applied else None,
        "output_pct": round(current, 2),
        "applied":    applied,
        "note":       div_note,
    })

    # ── Step 5: Coherence issues penalty ─────────────────────────────────────
    c_issues  = getattr(alpha_outputs, "coherence_issues", []) if alpha_outputs else []
    n_issues  = len(c_issues)
    if n_issues > 0:
        penalty   = min(1.0, n_issues * 0.25)
        current   = max(0.0, current - penalty)
        coh_note  = (
            f"{n_issues} cross-layer coherence issue(s) × 25 bp = −{penalty:.2f}pp applied."
        )
        applied   = True
    else:
        penalty   = 0.0
        coh_note  = "No coherence issues — no penalty."
        applied   = False

    steps.append({
        "step":       5,
        "name":       "Coherence Issues Penalty",
        "input":      f"{n_issues} issue(s)",
        "penalty_pp": round(-penalty, 2) if applied else 0,
        "output_pct": round(current, 2),
        "applied":    applied,
        "note":       coh_note,
    })

    # ── Step 6: Final recommendation ─────────────────────────────────────────
    final_pct = max(0.0, min(10.0, current))   # global clamp [0, 10%]

    if final_pct <= 0.25:
        recommendation = "avoid"
        rec_note = f"Allocation {final_pct:.2f}% — do not initiate a new position."
    elif final_pct < base_pct * 0.60:
        recommendation = "reduce"
        rec_note = f"{final_pct:.2f}% < {base_pct * 0.60:.2f}% (60% of base {base_pct:.1f}%) — reduce or trim."
    elif final_pct > base_pct * 1.30:
        recommendation = "increase"
        rec_note = f"{final_pct:.2f}% > {base_pct * 1.30:.2f}% (130% of base {base_pct:.1f}%) — increase position."
    else:
        recommendation = "hold"
        rec_note = f"{final_pct:.2f}% near base {base_pct:.1f}% — maintain current weighting."

    steps.append({
        "step":           6,
        "name":           "Final Recommendation",
        "output_pct":     round(final_pct, 2),
        "recommendation": recommendation,
        "note":           rec_note,
    })

    return {
        "steps":          steps,
        "recommendation": recommendation,
        "final_pct":      round(final_pct, 2),
        "base_pct":       base_pct,
        "reasoning":      rec_note,
    }


def _assess_coherence_step(coherence_issues: List[str]) -> Dict[str, Any]:
    """
    Step 7: Coherence assessment with per-issue resolution guidance.

    Classifies overall signal reliability from the number and severity of
    cross-layer coherence issues.
    """
    with_resolutions = [
        {
            "issue":      issue,
            "resolution": _resolve_issue(issue),
        }
        for issue in coherence_issues
    ]

    n = len(coherence_issues)
    if n == 0:
        reliability = "high"
        summary     = "All four cross-layer coherence checks passed — signals are consistent."
    elif n == 1:
        reliability = "moderate"
        summary     = "One coherence issue detected — monitor and verify the flagged assumption."
    else:
        reliability = "low"
        summary     = (
            f"{n} coherence issues detected — review all flagged conflicts before "
            f"taking a full position."
        )

    return {
        "n_issues":         n,
        "signal_reliability": reliability,
        "summary":          summary,
        "issues":           with_resolutions,
    }


# ── Main engine class ─────────────────────────────────────────────────────────

class QuantEngine:
    """
    Unified 8-step quantitative investment decision engine.

    Reads AlphaEngineOutputs and a mutated ValuationRange (with enriched
    MC result attached), and structures all layer outputs into a single
    auditable dict.

    Usage
    -----
        qe     = QuantEngine()
        result = qe.build("AAPL", alpha_outputs, val_range)
        findings["quant_engine"] = result
    """

    def build(
        self,
        ticker:        str,
        alpha_outputs: "AlphaEngineOutputs",
        val_range:     Optional["ValuationRange"],
    ) -> Dict[str, Any]:
        """
        Execute all 8 steps and return a fully serialisable dict.

        Parameters
        ----------
        ticker        : stock ticker (for labelling)
        alpha_outputs : result from AlphaPipeline.run()
        val_range     : ValuationRange object; val_range.mc must be the
                        enriched MCResult from the pipeline MC re-run.
        """
        if alpha_outputs is None:
            return {"available": False, "ticker": ticker}

        fp  = alpha_outputs.factor_profile
        rc  = alpha_outputs.regression_calib
        hrl = alpha_outputs.hrl_result
        st  = alpha_outputs.scenario_tree
        mc  = getattr(val_range, "mc", None) if val_range else None

        # ── Steps 1-5 (extraction) ────────────────────────────────────────────
        step1 = _extract_factor_step(fp)
        step2 = _extract_regression_step(rc, hrl)
        step3 = _extract_mc_inputs_step(hrl, rc, val_range)
        step4 = _extract_scenario_step(st)
        step5_dict, dp = _extract_mc_step(mc)

        # ── Step 6: Position sizing ───────────────────────────────────────────
        step6 = _compute_sizing_step(mc, dp, alpha_outputs)

        # ── Step 7: Coherence assessment ──────────────────────────────────────
        step7 = _assess_coherence_step(
            getattr(alpha_outputs, "coherence_issues", []) or []
        )

        # ── Step 8: Divergence metadata (output layer) ────────────────────────
        divergence = {
            "pct":   alpha_outputs.divergence_pct,
            "label": alpha_outputs.divergence_label,
            "flags": list(alpha_outputs.divergence_flags or []),
        }

        return {
            "available":         True,
            "ticker":            ticker,
            "macro_regime":      (alpha_outputs.trace or {}).get("macro_regime", "Unknown"),
            # 8 steps
            "step1_factor":      step1,
            "step2_regression":  step2,
            "step3_mc_inputs":   step3,
            "step4_scenarios":   step4,
            "step5_mc":          step5_dict,
            "step6_sizing":      step6,
            "step7_coherence":   step7,
            "step8_divergence":  divergence,
            # Observability
            "layers_degraded":   list(alpha_outputs.layers_degraded or []),
            "pipeline_trace":    dict(alpha_outputs.trace or {}),
        }
