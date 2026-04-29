"""
alpha_pipeline.py
=================
Ordered execution pipeline for the four alpha engine layers.

Enforces strict execution ordering with explicit data contracts.
Wraps each layer in an independent try/except so failure of any
single layer degrades gracefully without poisoning downstream layers.

Execution order (strictly sequential — each layer requires prior output)
------------------------------------------------------------------------
  1.   Factor model          — 7-factor cross-sectional z-scores, style,
                               composite score, regime + archetype weights
  2.   Regression calibration — factor premium × z-score → expected return;
                               historical vol → tracking error
  2.5. HRL                   — AR(1) EPS, margin trend, valuation MR,
                               macro sensitivity, drawdown, three-way blend
  3.   Scenario tree         — Markov narrative branches, shock cluster,
                               dispersion metrics, best/worst case
  MC   enriched re-run       — integrates all four layers into final
                               GrowthDistParams + run_monte_carlo()
                               → mutates val_range.mc in-place

Divergence detection
--------------------
  |scenario_tree.weighted_return − regression_calib.expected_return|:
    < 5%  → "coherent"      — both models agree; high sizing confidence
    5-12% → "moderate"      — flag in DATA QUALITY; both signals retained
    > 12% → "significant"   — flag + position size capped at 1.5% until
                               divergence resolves (Step 4e of sizing)

Coherence checks
----------------
  Four cross-layer consistency tests surface unexpected conflicts between
  structural (factor model) and empirical (HRL / scenario tree) signals.
  Results surface in the DATA QUALITY FLAGS report section.

Integration
-----------
    from analysis.alpha_pipeline import AlphaPipeline

    outputs = AlphaPipeline().run(
        stock_data   = stock_data,
        val_range    = val_range,        # mutated in-place: val_range.mc updated
        macro_regime = "Late_Cycle",
        peer_rows    = [],
        pe_val       = 28.5,
        price        = 112.0,
    )
    factor_profile   = outputs.factor_profile
    regression_calib = outputs.regression_calib
    hrl_result       = outputs.hrl_result
    scenario_tree    = outputs.scenario_tree
    # outputs.divergence_label, outputs.divergence_flags, outputs.coherence_issues
    # outputs.trace  — structured dict for observability
"""
from __future__ import annotations

import traceback as _tb
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.stock_data import StockData


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AlphaEngineOutputs:
    """
    Consolidated output of a full AlphaPipeline.run() call.

    All layer outputs are None when the corresponding layer failed.
    Downstream consumers must always guard with `is not None` checks.
    """
    # ── Layer outputs ──────────────────────────────────────────────────────
    factor_profile:   Optional[object]   # FactorProfile
    regression_calib: Optional[object]   # RegressionCalibration
    hrl_result:       Optional[object]   # HRLResult
    scenario_tree:    Optional[object]   # ScenarioTree

    # ── Divergence between scenario tree and regression model ──────────────
    divergence_pct:   Optional[float]    # absolute gap (e.g. 0.17 = 17 pp)
    divergence_label: str                # "coherent" | "moderate" | "significant" | "n/a"
    divergence_flags: List[str]          # human-readable flag lines for DATA QUALITY section

    # ── Cross-layer coherence ──────────────────────────────────────────────
    coherence_issues: List[str]          # from _coherence_check()

    # ── Observability ──────────────────────────────────────────────────────
    trace:            Dict[str, Any]     # structured per-run trace
    layers_degraded:  List[str]          # names of layers that returned None


# ─────────────────────────────────────────────────────────────────────────────
# Coherence check
# ─────────────────────────────────────────────────────────────────────────────

def _coherence_check(
    factor:    Optional[object],
    hrl:       Optional[object],
    scenario:  Optional[object],
    mc_result: Optional[object],
) -> List[str]:
    """
    Four cross-layer consistency tests.
    Returns a list of human-readable issue strings (empty list = all clear).

    These are surfaced in the DATA QUALITY FLAGS section of the report.
    A coherence issue is not an error — it is a signal that warrants
    PM awareness or review of an assumption.
    """
    issues: List[str] = []

    if factor is None and scenario is None:
        return issues

    style      = getattr(factor,   "style_label",      "") or "" if factor else ""
    prof_z     = getattr(factor,   "profitability_z",  0.0) or 0.0 if factor else 0.0
    down_mass  = getattr(scenario, "downside_mass",    0.0) or 0.0 if scenario else 0.0
    shock_prob = getattr(scenario, "shock_prob",       0.0) or 0.0 if scenario else 0.0
    scene_er   = getattr(scenario, "weighted_return",  None)       if scenario else None
    mc_mean    = getattr(mc_result, "mean_return",     None)       if mc_result else None

    # Check 1: Compounder style with high downside mass
    # Likely cause: overly contractionary Markov macro prior vs structural quality signal.
    if style == "Compounder" and down_mass > 0.30:
        issues.append(
            f"COHERENCE: style={style!r} but scenario downside_mass={down_mass:.0%} (>30%). "
            f"Structural quality signal and narrative probability distribution diverge. "
            f"Verify macro regime assignment — Markov prior may be penalising this name unfairly."
        )

    # Check 2: Rising margin trend (HRL time-series) + weak profitability z (cross-sectional)
    # Likely cause: peer set overstates industry margin norms, or time-series shows
    # a company catching up to peers (genuine improvement not yet reflected cross-sectionally).
    if hrl is not None:
        slope = getattr(hrl, "margin_trend_slope", 0.0) or 0.0
        if slope > 0.010 and prof_z < -0.5:
            issues.append(
                f"COHERENCE: margin_trend_slope={slope:+.3f}/qtr (rising) but "
                f"profitability_z={prof_z:+.2f} (below peers). Time-series and "
                f"cross-sectional signals diverge. Check whether peer set is appropriate "
                f"or whether margin recovery is in progress."
            )

    # Check 3: High AR(1) persistence + high shock probability
    # Usually NOT incoherence (shock_prob is macro-driven, not earnings-quality-driven).
    # Logged only when rho > 0.85 AND shock_prob > 0.25 — both extreme.
    if hrl is not None:
        rho = getattr(hrl, "ar1_eps_persistence", 0.65) or 0.65
        if rho > 0.85 and shock_prob > 0.25:
            issues.append(
                f"COHERENCE NOTE: AR(1) persistence rho={rho:.2f} (high) alongside "
                f"shock_prob={shock_prob:.0%} (elevated). Shock prob is macro-driven "
                f"(recession ∩ miss cluster) — even persistent earners face tail risk "
                f"in deep recessions. No action required; noted for PM awareness."
            )

    # Check 4: MC mean return and scenario weighted return directionally opposite (>15 pp gap)
    # Likely cause: MC growth_mean is calibrated to factor/HRL signals (expansion-biased)
    # while scenario tree is driven by a contractionary Markov prior.
    if scene_er is not None and mc_mean is not None:
        gap = abs(mc_mean - scene_er)
        if mc_mean > 0.10 and scene_er < -0.05 and gap > 0.15:
            issues.append(
                f"COHERENCE: MC mean_return={mc_mean:+.1%} and scenario "
                f"weighted_return={scene_er:+.1%} have opposite signs (gap={gap:.0%}). "
                f"Review macro regime — MC growth inputs and Markov prior may be using "
                f"inconsistent regime assumptions."
            )

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class AlphaPipeline:
    """
    Ordered alpha engine pipeline with contract validation and coherence checks.

    Instantiate once per evaluation; not thread-safe (traces are per-instance).

    Usage
    -----
        outputs = AlphaPipeline().run(
            stock_data=sd, val_range=vr, macro_regime="Late_Cycle",
            peer_rows=[], pe_val=28.5, price=112.0,
        )
        # val_range.mc is mutated in-place with the enriched MC result.
        # Unpack: outputs.factor_profile, outputs.scenario_tree, etc.
    """

    def run(
        self,
        stock_data:   "StockData",
        val_range:    Optional[object],
        macro_regime: str,
        peer_rows:    List[object],
        pe_val:       Optional[float],
        price:        Optional[float],
    ) -> AlphaEngineOutputs:
        """
        Execute all four alpha layers in strict dependency order, then run
        the enriched MC simulation and compute divergence + coherence.

        val_range is mutated: val_range.mc is overwritten with the enriched
        MCResult when the MC re-run succeeds.
        """
        factor_profile:   Optional[object] = None
        regression_calib: Optional[object] = None
        hrl_result:       Optional[object] = None
        scenario_tree:    Optional[object] = None
        layers_degraded:  List[str]        = []
        trace:            Dict[str, Any]   = {"macro_regime": macro_regime}

        # No val_range → skip all alpha layers (no price target anchor)
        if val_range is None:
            return AlphaEngineOutputs(
                factor_profile   = None,
                regression_calib = None,
                hrl_result       = None,
                scenario_tree    = None,
                divergence_pct   = None,
                divergence_label = "n/a",
                divergence_flags = [],
                coherence_issues = [],
                trace            = trace,
                layers_degraded  = ["factor", "regression", "hrl", "scenario"],
            )

        # ── Layer 1: Factor model ─────────────────────────────────────────────
        # No upstream dependencies — always runs first.
        try:
            from analysis.factor_model import build_factor_profile as _bfp
            factor_profile = _bfp(stock_data, peer_rows, macro_regime=macro_regime)
            if factor_profile is not None:
                trace["factor_style"]     = getattr(factor_profile, "style_label",     "?")
                trace["factor_composite"] = getattr(factor_profile, "composite_score", None)
                trace["factor_n_peers"]   = getattr(factor_profile, "n_peers",         0)
                trace["factor_quality_z"] = getattr(factor_profile, "quality_z",       None)
        except Exception as e:
            print(f"  [PIPELINE:factor] degraded: {e}")
            layers_degraded.append("factor")

        # ── Layer 2: Regression calibration ──────────────────────────────────
        # Contract: requires factor_profile from Layer 1.
        # When factor_profile is None, regression skips (no z-scores to use).
        if factor_profile is not None:
            try:
                from analysis.regression_calibration import calibrate_regression as _creg
                regression_calib = _creg(factor_profile, stock_data, macro_regime)
                if regression_calib is not None:
                    trace["reg_expected_return"] = regression_calib.expected_return
                    trace["reg_confidence"]      = regression_calib.confidence
                    trace["reg_tracking_error"]  = regression_calib.tracking_error
            except Exception as e:
                print(f"  [PIPELINE:regression] degraded: {e}")
                layers_degraded.append("regression")
        else:
            layers_degraded.append("regression")

        # ── Layer 2.5: Historical Regression Layer ────────────────────────────
        # Contract: accepts regression_calib from Layer 2 for three-way blend.
        # When regression_calib is None, HRL redistributes the 30% reg weight
        # to the MC fundamental path (two-way blend instead of three-way).
        try:
            from analysis.historical_regression import (
                run_historical_regression_layer as _hrl_run,
            )
            _mc_gr_raw = getattr(val_range, "eps_growth_rate", None)
            _mc_gr_dec = _mc_gr_raw / 100.0 if _mc_gr_raw is not None else None
            hrl_result = _hrl_run(
                stock_data,
                factor_profile = factor_profile,
                macro_regime   = macro_regime,
                mc_growth_mean = _mc_gr_dec,
                reg_calib      = regression_calib,
            )
            if hrl_result is not None:
                trace["hrl_confidence"]   = hrl_result.hrl_confidence
                trace["hrl_ar1_rho"]      = hrl_result.ar1_eps_persistence
                trace["hrl_calibrated_g"] = hrl_result.calibrated_growth_mean
                trace["hrl_mr_speed"]     = hrl_result.valuation_mr_speed
                trace["hrl_margin_slope"] = hrl_result.margin_trend_slope
                trace["hrl_max_dd_avg"]   = hrl_result.max_drawdown_avg
        except Exception as e:
            print(f"  [PIPELINE:hrl] degraded: {e}")
            print(_tb.format_exc())
            layers_degraded.append("hrl")

        # ── Layer 3: Scenario tree ────────────────────────────────────────────
        # Contract: accepts hrl_result from Layer 2.5 for probability calibration.
        # When hrl_result is None, scenario tree uses regime-table defaults.
        try:
            from analysis.scenario_tree import (
                build_scenario_tree as _bst,
                infer_earnings_trend as _iet,
            )
            _trend = _iet(stock_data)
            scenario_tree = _bst(
                macro_regime   = macro_regime,
                earnings_trend = _trend,
                current_pe     = pe_val,
                base_eps       = getattr(val_range, "scenario_base_eps", None),
                current_price  = price,
                factor_profile = factor_profile,
                hrl_result     = hrl_result,
            )
            if scenario_tree is not None:
                trace["scenario_weighted_return"] = scenario_tree.weighted_return
                trace["scenario_std"]             = scenario_tree.scenario_std
                trace["scenario_shock_prob"]      = scenario_tree.shock_prob
                trace["scenario_n_leaves"]        = len(scenario_tree.leaves)
                trace["scenario_down_mass"]       = scenario_tree.downside_mass
                trace["scenario_up_mass"]         = scenario_tree.upside_mass
                trace["scenario_concentration3"]  = scenario_tree.concentration_3
                trace["scenario_var95"]           = scenario_tree.var_95
        except Exception as e:
            print(f"  [PIPELINE:scenario] degraded: {e}")
            layers_degraded.append("scenario")

        # ── Enriched MC re-run ────────────────────────────────────────────────
        # Integrates all four layers into _apply_layer_overrides() priority stack,
        # then runs 10,000 MC paths. Mutates val_range.mc in-place.
        mc_result = getattr(val_range, "mc", None)
        try:
            from analysis.monte_carlo import mc_from_valuation_range as _rerun_mc
            _mc_enriched = _rerun_mc(
                val_range,
                macro_regime     = macro_regime,
                gross_margin     = getattr(val_range, "quality_gross_margin", None),
                op_margin        = getattr(val_range, "quality_op_margin",    None),
                stock_data       = stock_data,
                factor_profile   = factor_profile,
                regression_calib = regression_calib,
                hrl_result       = hrl_result,
                scenario_tree    = scenario_tree,
            )
            if _mc_enriched is not None:
                val_range.mc = _mc_enriched
                mc_result    = _mc_enriched
                _fp_label    = getattr(factor_profile, "style_label", "?") if factor_profile else "?"
                trace["mc_mean_return"]   = _mc_enriched.mean_return
                trace["mc_p5_return"]     = _mc_enriched.p5_return
                trace["mc_prob_positive"] = _mc_enriched.prob_positive
                print(
                    f"  [PIPELINE:MC] regime={macro_regime!r}"
                    f" style={_fp_label!r}"
                    f" mean={_mc_enriched.mean_return:+.1%}"
                    f" P5={_mc_enriched.p5_return:+.1%}"
                    f" P(gain)={_mc_enriched.prob_positive:.0%}"
                )
        except Exception as e:
            print(f"  [PIPELINE:MC] enriched re-run skipped: {e}")
            layers_degraded.append("mc_enriched")

        # ── Divergence detection ──────────────────────────────────────────────
        # Measure pairwise disagreement across all three model outputs:
        #   · factor_ER   = regression_calib.expected_return   (structural)
        #   · scenario_ER = scenario_tree.weighted_return      (narrative)
        #   · mc_ER       = val_range.mc.mean_return            (probabilistic)
        # model_divergence = max pairwise gap.
        # Thresholds: < 5% coherent, 5-12% moderate, > 12% significant.
        divergence_pct:   Optional[float] = None
        divergence_label: str             = "n/a"
        divergence_flags: List[str]       = []

        reg_er = getattr(regression_calib, "expected_return", None) if regression_calib else None
        sce_er = getattr(scenario_tree,    "weighted_return",  None) if scenario_tree    else None
        mc_er  = getattr(getattr(val_range, "mc", None), "mean_return", None)

        # Collect all available model estimates
        _model_estimates: Dict[str, float] = {}
        if reg_er is not None:
            _model_estimates["factor"]   = reg_er
        if sce_er is not None:
            _model_estimates["scenario"] = sce_er
        if mc_er is not None:
            _model_estimates["mc"]       = mc_er

        if len(_model_estimates) >= 2:
            _keys   = list(_model_estimates.keys())
            _vals   = list(_model_estimates.values())
            _pairs  = [
                (abs(_vals[i] - _vals[j]), _keys[i], _keys[j])
                for i in range(len(_vals)) for j in range(i + 1, len(_vals))
            ]
            _max_pair = max(_pairs, key=lambda x: x[0])
            divergence_pct   = _max_pair[0]
            _pair_label      = f"{_max_pair[1]} vs {_max_pair[2]}"
            trace["divergence_pct"]  = divergence_pct
            trace["divergence_pair"] = _pair_label

            if divergence_pct < 0.05:
                divergence_label = "coherent"
            elif divergence_pct < 0.12:
                divergence_label = "moderate"
                divergence_flags.append(
                    f"MODEL_DIVERGENCE (moderate): "
                    f"{_pair_label} gap = {divergence_pct:.0%}. "
                    f"All model signals retained. "
                    + (f"factor={reg_er:+.1%} " if reg_er is not None else "")
                    + (f"scenario={sce_er:+.1%} " if sce_er is not None else "")
                    + (f"MC={mc_er:+.1%}." if mc_er is not None else "")
                )
            else:
                divergence_label = "significant"
                divergence_flags.append(
                    f"MODEL_DISAGREEMENT ⚠: "
                    f"{_pair_label} gap = {divergence_pct:.0%} (> 12%). "
                    f"Position size capped at 1.5% until divergence resolves. "
                    + (f"factor={reg_er:+.1%} " if reg_er is not None else "")
                    + (f"scenario={sce_er:+.1%} " if sce_er is not None else "")
                    + (f"MC={mc_er:+.1%}. " if mc_er is not None else "")
                    + f"Review macro regime — models may be using inconsistent assumptions."
                )

            print(
                f"  [PIPELINE:divergence]"
                + (f" factor={reg_er:+.1%}" if reg_er is not None else "")
                + (f" scenario={sce_er:+.1%}" if sce_er is not None else "")
                + (f" mc={mc_er:+.1%}" if mc_er is not None else "")
                + f" max_gap={divergence_pct:.0%} ({_pair_label})"
                + f" label={divergence_label}"
            )

        trace["divergence_label"] = divergence_label

        # ── Coherence check ───────────────────────────────────────────────────
        coherence_issues = _coherence_check(
            factor    = factor_profile,
            hrl       = hrl_result,
            scenario  = scenario_tree,
            mc_result = mc_result,
        )
        trace["coherence_n_issues"] = len(coherence_issues)

        # ── Structured trace log ──────────────────────────────────────────────
        def _fmt(v: Any) -> str:
            if v is None:
                return "?"
            if isinstance(v, float):
                return f"{v:+.1%}" if abs(v) < 2.0 else f"{v:.1f}"
            return str(v)

        print(
            f"  [PIPELINE:summary]"
            f" ticker=?"
            f" style={trace.get('factor_style', '?')}"
            f" composite={_fmt(trace.get('factor_composite'))}"
            f" reg_E[R]={_fmt(trace.get('reg_expected_return'))}"
            f" hrl_blend={_fmt(trace.get('hrl_calibrated_g'))}"
            f" scene_E[R]={_fmt(trace.get('scenario_weighted_return'))}"
            f" mc_mean={_fmt(trace.get('mc_mean_return'))}"
            f" divergence={divergence_label}"
            f" coherence_issues={len(coherence_issues)}"
            f" degraded={layers_degraded or 'none'}"
        )

        return AlphaEngineOutputs(
            factor_profile   = factor_profile,
            regression_calib = regression_calib,
            hrl_result       = hrl_result,
            scenario_tree    = scenario_tree,
            divergence_pct   = divergence_pct,
            divergence_label = divergence_label,
            divergence_flags = divergence_flags,
            coherence_issues = coherence_issues,
            trace            = trace,
            layers_degraded  = layers_degraded,
        )
