"""
layer2_processor.py
===================
LAYER 2 — Single-ticker valuation input processor.

Converts a standardized key-value input block (produced by extract_inputs.py
or pasted manually) into a structured JSON output for scenario analysis and
Monte Carlo simulation.

All rules encoded here are the canonical implementation of the Layer 2
system prompt — deterministic Python, no LLM required.

Classification rules
--------------------
  stable_compounder   Revenue growth 8–20%, consistent, beta < 1.2, stable margins
  cyclical            Revenue variance > 20% Y/Y historically, macro-sensitive
  turnaround          EPS_CURRENT < 0, or negative → positive EPS flip
  margin_recovery     GM-NM spread ≥ 20pp (or ≥ 40pp when NM > 20%), or negative net margin
  multiple_rerating   Growth < 10% but avg P/E > 30×
  binary_fat_tailed   Model base PT > 40% below current price, or EPS_2026 < EPS_CURRENT
  acquisition_distorted ACQUISITION_DISTORTED: true override, or revenue growth > 30%

Spread rules
------------
  model_type           growth_spread  margin_spread  multiple_spread
  stable_compounder    ±0.04          ±0.03          ±20%
  cyclical             ±0.08          ±0.05          ±30%
  turnaround           ±0.12          ±0.08          ±40%
  margin_recovery      ±0.10          ±0.08          ±35%
  multiple_rerating    ±0.04          ±0.03          ±35%
  binary_fat_tailed    ±0.20          ±0.12          ±60%
  acquisition_distorted ±0.15         ±0.10          ±50%
  When multiple types apply → use widest spread.

Monte Carlo distribution rules
-------------------------------
  stable_compounder    normal / normal / normal      → normal tail
  cyclical             normal / normal / normal      → wide tail
  turnaround           skewed-right / skewed-right / log-normal → wide tail
  margin_recovery      skewed-right / skewed-right / normal     → wide tail
  multiple_rerating    normal / normal / skewed-right           → normal tail
  binary_fat_tailed    fat-tailed / fat-tailed / fat-tailed     → fat-tailed tail
  acquisition_distorted uniform / uniform / uniform             → fat-tailed tail

Integration
-----------
    from analysis.layer2_processor import process_input_block

    # from key-value dict (extract_inputs.py output)
    result = process_input_block(raw_dict)
    print(json.dumps(result, indent=2))

    # from raw key-value text block
    from analysis.layer2_processor import parse_keyvalue, process_input_block
    result = process_input_block(parse_keyvalue(text_block))
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any, Dict, List, Optional, Tuple


# ── Spread lookup tables ──────────────────────────────────────────────────────

# (growth_spread, margin_spread, multiple_pct_spread)
_SPREAD: Dict[str, Tuple[float, float, float]] = {
    "stable_compounder":    (0.04, 0.03, 0.20),
    "cyclical":             (0.08, 0.05, 0.30),
    "turnaround":           (0.12, 0.08, 0.40),
    "margin_recovery":      (0.10, 0.08, 0.35),
    "multiple_rerating":    (0.04, 0.03, 0.35),
    "binary_fat_tailed":    (0.20, 0.12, 0.60),
    "acquisition_distorted":(0.15, 0.10, 0.50),
}

# growth_dist, margin_dist, multiple_dist, tail_profile
_MC_DIST: Dict[str, Tuple[str, str, str, str]] = {
    "stable_compounder":    ("normal",       "normal",       "normal",      "normal"),
    "cyclical":             ("normal",       "normal",       "normal",      "wide"),
    "turnaround":           ("skewed-right", "skewed-right", "log-normal",  "wide"),
    "margin_recovery":      ("skewed-right", "skewed-right", "normal",      "wide"),
    "multiple_rerating":    ("normal",       "normal",       "skewed-right","normal"),
    "binary_fat_tailed":    ("fat-tailed",   "fat-tailed",   "fat-tailed",  "fat-tailed"),
    "acquisition_distorted":("uniform",      "uniform",      "uniform",     "fat-tailed"),
}

# Priority order for merging distributions when multiple types apply:
# highest priority row wins each field.
_TYPE_PRIORITY = [
    "binary_fat_tailed",
    "acquisition_distorted",
    "turnaround",
    "margin_recovery",
    "cyclical",
    "multiple_rerating",
    "stable_compounder",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sf(v: Any, decimals: int = 4) -> Optional[float]:
    """Safe float with rounding; returns None on failure or NaN."""
    if v is None or v == "null":
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else round(f, decimals)
    except (TypeError, ValueError):
        return None


def parse_keyvalue(text: str) -> Dict[str, Any]:
    """
    Parse a raw key-value block (one 'KEY: value' per line) into a dict.
    Lines starting with # are ignored.
    """
    out: Dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val.lower() == "null":
            out[key] = None
        else:
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = val
    return out


# ── Classification ────────────────────────────────────────────────────────────

def _classify(inp: Dict[str, Any]) -> Tuple[List[str], str, str, List[str]]:
    """
    Returns (model_types, primary_driver, assumption_quality, flags).

    Classification follows the Layer 2 rules exactly; when multiple types
    match, ALL are returned and the widest spread is used downstream.
    """
    eps_cur  = _sf(inp.get("EPS_CURRENT"))
    eps_26   = _sf(inp.get("EPS_2026"))
    eps_27   = _sf(inp.get("EPS_2027"))
    eps_28   = _sf(inp.get("EPS_2028"))
    growth   = _sf(inp.get("PROJECTED_REVENUE_GROWTH"))
    gm       = _sf(inp.get("GROSS_MARGIN_CURRENT"))
    nm       = _sf(inp.get("NET_MARGIN_CURRENT"))
    beta     = _sf(inp.get("BETA"))
    avg_pe   = _sf(inp.get("AVG_PE_RATIO"))
    curr_px  = _sf(inp.get("CURRENT_PRICE"))
    pt_base  = _sf(inp.get("PRICE_TARGET_2026_BASE"))

    types:  List[str] = []
    flags:  List[str] = []

    # ── stable_compounder ─────────────────────────────────────────────────────
    if growth is not None and 0.08 <= growth <= 0.20:
        if beta is None or beta < 1.2:
            types.append("stable_compounder")

    # ── cyclical ─────────────────────────────────────────────────────────────
    # Proxy: beta ≥ 1.4 as macro-sensitivity indicator (no historical rev series)
    if beta is not None and beta >= 1.4:
        types.append("cyclical")
        flags.append(f"Beta={beta:.2f} ≥ 1.4 — classified as cyclical (no historical rev series available).")

    # ── turnaround ────────────────────────────────────────────────────────────
    if eps_cur is not None and eps_cur < 0:
        types.append("turnaround")
        flags.append(f"EPS_CURRENT={eps_cur:.4f} < 0 — turnaround classification.")
    elif eps_cur is not None and eps_26 is not None and eps_26 > abs(eps_cur) * 2:
        # EPS more than doubles (including from very low base) — aggressive inflection
        types.append("turnaround")
        flags.append(
            f"EPS inflection: {eps_cur:.2f} → {eps_26:.2f} — aggressive positive flip."
        )

    # ── margin_recovery ───────────────────────────────────────────────────────
    if gm is not None and nm is not None:
        spread = gm - nm
        # Threshold is 40pp when NM > 20% — avoids misclassifying high-quality
        # companies (e.g. software with structural R&D/SBC cost layers) that have
        # a persistently wide GM-NM gap unrelated to any recovery story.
        mr_threshold = 0.40 if nm > 0.20 else 0.20
        if spread >= mr_threshold:
            types.append("margin_recovery")
            flags.append(
                f"Gross margin ({gm:.1%}) vs net margin ({nm:.1%}): "
                f"{spread:.0%} spread ≥ {mr_threshold:.0%} threshold — margin_recovery classification."
            )
        elif nm < 0:
            types.append("margin_recovery")
            flags.append(f"Net margin negative ({nm:.1%}) — margin_recovery classification.")

    # ── multiple_rerating ─────────────────────────────────────────────────────
    if growth is not None and growth < 0.10 and avg_pe is not None and avg_pe > 30:
        types.append("multiple_rerating")
        flags.append(
            f"Low growth ({growth:.1%}) with high avg P/E ({avg_pe:.1f}×) — "
            f"multiple_rerating classification."
        )

    # ── binary_fat_tailed ────────────────────────────────────────────────────
    if curr_px is not None and pt_base is not None:
        gap = (pt_base - curr_px) / curr_px
        if gap < -0.40:
            types.append("binary_fat_tailed")
            flags.append(
                f"Model base PT (${pt_base:.2f}) is {gap:.0%} below current price "
                f"(${curr_px:.2f}) — binary_fat_tailed classification."
            )
    if eps_cur is not None and eps_26 is not None and eps_26 < eps_cur and eps_cur > 0:
        types.append("binary_fat_tailed")
        flags.append(
            f"EPS_2026 ({eps_26:.2f}) < EPS_CURRENT ({eps_cur:.2f}) despite positive "
            f"growth assumption — binary_fat_tailed classification."
        )

    # ── acquisition_distorted ────────────────────────────────────────────────
    # Manual override: set ACQUISITION_DISTORTED: true in the input block to
    # force this classification regardless of growth rate (e.g. QXO-type stories
    # where M&A is the explicit thesis but reported growth is sub-50%).
    _acq_override = str(inp.get("ACQUISITION_DISTORTED", "")).lower() in ("true", "1", "yes")
    if _acq_override:
        types.append("acquisition_distorted")
        flags.append(
            "ACQUISITION_DISTORTED=true manual override — treating as acquisition-distorted "
            "regardless of reported growth rate."
        )
    # Proxy: growth > 30% suggests acquisition-driven step change (was 50%;
    # lowered because M&A roll-ups rarely show >50% on a single reporting year
    # but routinely show 30–50% as acquired revenue layers in).
    elif growth is not None and growth > 0.30:
        types.append("acquisition_distorted")
        flags.append(
            f"Revenue growth {growth:.1%} > 30% — possible acquisition-distorted; "
            f"verify organic run-rate before modelling as compound growth. "
            f"Set ACQUISITION_DISTORTED: false to suppress if growth is organic."
        )

    # Fallback: at minimum label as stable_compounder
    if not types:
        types.append("stable_compounder")

    # Deduplicate preserving order
    seen: set = set()
    types = [t for t in types if not (t in seen or seen.add(t))]   # type: ignore[func-returns-value]

    # ── Primary driver ────────────────────────────────────────────────────────
    if "turnaround" in types:
        primary = "turnaround"
    elif "margin_recovery" in types:
        # If margin_recovery fired only from the GM-NM gap (nm > 20%) rather than
        # from a genuine recovery story (nm < 0), keep primary as growth/multiple.
        if nm is not None and nm > 0.20:
            primary = "growth"
        else:
            primary = "margin"
    elif "multiple_rerating" in types:
        primary = "multiple"
    elif growth is not None and growth >= 0.15:
        primary = "growth"
    elif gm is not None and nm is not None and (gm - nm) >= 0.15:
        primary = "margin"
    elif avg_pe is not None and avg_pe > 25:
        primary = "multiple"
    else:
        primary = "growth"

    # ── Assumption quality ────────────────────────────────────────────────────
    bad_signals = sum([
        eps_cur is not None and eps_cur < 0,
        growth is not None and growth > 0.40,
        nm is not None and nm < 0,
        "acquisition_distorted" in types,
        # Model PT severely below current price — binary/speculative signal
        "binary_fat_tailed" in types,
        # Turnaround via positive-EPS inflection (EPS doubles from low but positive base)
        # Does NOT double-count: if eps_cur < 0, the first condition fires instead.
        ("turnaround" in types and eps_cur is not None and eps_cur > 0),
        # EPS Y2→Y3 regression — declining forward earnings despite positive growth
        (eps_27 is not None and eps_28 is not None and eps_27 > 0 and eps_28 < eps_27),
    ])
    if bad_signals >= 2:
        quality = "low"
    elif bad_signals == 1:
        quality = "medium"
    else:
        quality = "high"

    # Additional consistency flags
    if curr_px is not None and pt_base is not None:
        upside = (pt_base / curr_px - 1) * 100
        if abs(upside) > 30:
            flags.append(
                f"Model base (${pt_base:.2f}) is {upside:+.1f}% vs current price (${curr_px:.2f})."
            )
    if eps_cur is not None and eps_26 is not None:
        cagr_str = f"${eps_cur:.2f} → ${eps_26:.2f}"
        flags.append(f"EPS Y0 → Y1: {cagr_str}.")
    if eps_26 is not None and eps_28 is not None and eps_26 > 0:
        cagr_2yr = (eps_28 / eps_26) ** 0.5 - 1
        flags.append(f"EPS Y1→Y3 CAGR: {cagr_2yr:.1%} ({eps_26:.2f} → {eps_28:.2f}).")
    if eps_27 is not None and eps_28 is not None and eps_27 > 0 and eps_28 < eps_27:
        flags.append(
            f"EPS Y2→Y3 regression: ${eps_27:.2f} → ${eps_28:.2f} — "
            f"declining forward earnings despite positive growth assumption."
        )

    return types, primary, quality, flags


# ── Consistency check ─────────────────────────────────────────────────────────

def _consistency_check(inp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify: PRICE_TARGET_2026_BASE ≈ EPS_2026 × AVG_PE_RATIO (±5%).
    """
    eps_26  = _sf(inp.get("EPS_2026"))
    avg_pe  = _sf(inp.get("AVG_PE_RATIO"))
    pt_base = _sf(inp.get("PRICE_TARGET_2026_BASE"))

    if eps_26 is None or avg_pe is None or pt_base is None:
        return {
            "eps_times_multiple": None,
            "price_target_base":  pt_base,
            "deviation_pct":      None,
            "passed":             None,
        }

    implied = round(eps_26 * avg_pe, 2)
    dev     = (implied - pt_base) / pt_base if pt_base != 0 else None
    passed  = abs(dev) <= 0.05 if dev is not None else None

    return {
        "eps_times_multiple": implied,
        "price_target_base":  pt_base,
        "deviation_pct":      round(dev, 4) if dev is not None else None,
        "passed":             passed,
    }


# ── Spread selection ──────────────────────────────────────────────────────────

def _widest_spread(types: List[str]) -> Tuple[float, float, float]:
    """
    Return (growth_spread, margin_spread, multiple_pct_spread) as the widest
    values across all active model types.
    """
    gs, ms, mps = 0.0, 0.0, 0.0
    for t in types:
        g, m, mp = _SPREAD.get(t, (0.04, 0.03, 0.20))
        gs  = max(gs,  g)
        ms  = max(ms,  m)
        mps = max(mps, mp)
    return gs, ms, mps


# ── MC distribution selection ─────────────────────────────────────────────────

def _mc_distributions(types: List[str]) -> Tuple[str, str, str, str]:
    """
    Return (growth_dist, margin_dist, multiple_dist, tail_profile).

    When multiple types apply, the highest-priority type (by _TYPE_PRIORITY)
    wins each distribution field.
    """
    # Pick the highest-priority active type
    winner = next((t for t in _TYPE_PRIORITY if t in types), types[0])
    return _MC_DIST.get(winner, ("normal", "normal", "normal", "normal"))


# ── Skew mapping ──────────────────────────────────────────────────────────────

_SKEW_MAP = {
    "normal":       0.0,
    "skewed-right": 0.5,
    "log-normal":   1.0,
    "fat-tailed":   0.0,    # kurtosis rather than skew; 0 is correct for skew param
    "uniform":      0.0,
}


# ── Main processor ────────────────────────────────────────────────────────────

def save_result(result: Dict[str, Any], outdir: str = "data/layer2") -> str:
    """
    Save a Layer 2 result dict to disk.

    Writes to {outdir}/{TICKER}_layer2.json, creating directories as needed.
    Returns the path written.
    """
    import json
    from pathlib import Path

    ticker  = result.get("ticker", "UNKNOWN")
    out     = Path(outdir) / f"{ticker}_layer2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return str(out)


def process_input_block(inp: Dict[str, Any], save: bool = False,
                        outdir: str = "data/layer2") -> Dict[str, Any]:
    """
    Convert a standardized key-value input dict into the Layer 2 JSON schema.

    Parameters
    ----------
    inp    : dict from parse_keyvalue() or extract_inputs.extract()
    save   : if True, write result to {outdir}/{TICKER}_layer2.json
    outdir : output directory when save=True (default: data/layer2)

    Returns
    -------
    dict matching the Layer 2 output schema (JSON-serialisable)
    """
    # ── Raw inputs ────────────────────────────────────────────────────────────
    ticker   = str(inp.get("TICKER", "UNKNOWN")).upper()
    curr_px  = _sf(inp.get("CURRENT_PRICE"))
    eps_cur  = _sf(inp.get("EPS_CURRENT"))
    eps_26   = _sf(inp.get("EPS_2026"))
    eps_27   = _sf(inp.get("EPS_2027"))
    eps_28   = _sf(inp.get("EPS_2028"))
    pt_base  = _sf(inp.get("PRICE_TARGET_2026_BASE"))
    pt_low   = _sf(inp.get("PRICE_TARGET_2026_LOW"))
    pt_high  = _sf(inp.get("PRICE_TARGET_2026_HIGH"))
    avg_pe   = _sf(inp.get("AVG_PE_RATIO"))
    pe_low   = _sf(inp.get("PE_RANGE_LOW"))
    pe_high  = _sf(inp.get("PE_RANGE_HIGH"))
    growth   = _sf(inp.get("PROJECTED_REVENUE_GROWTH"))
    gm       = _sf(inp.get("GROSS_MARGIN_CURRENT"))
    nm       = _sf(inp.get("NET_MARGIN_CURRENT"))
    wacc     = _sf(inp.get("WACC"))
    beta     = _sf(inp.get("BETA"))

    # ── Classification ────────────────────────────────────────────────────────
    types, primary, quality, flags = _classify(inp)

    # ── Spread selection (widest across all active types) ─────────────────────
    gs, ms, mps = _widest_spread(types)

    # ── Normalized assumptions ────────────────────────────────────────────────
    growth_mean   = growth   if growth  is not None else 0.10
    margin_mean   = gm       if gm      is not None else (nm if nm is not None else 0.20)
    multi_center  = avg_pe   if avg_pe  is not None else 20.0
    growth_range  = round(gs  * 2, 4)       # ±spread → total range
    margin_range  = round(ms  * 2, 4)
    multi_range   = round(multi_center * mps * 2, 4)

    # ── Scenario mapping ──────────────────────────────────────────────────────
    bear_g  = round(growth_mean - gs,  4)
    base_g  = round(growth_mean,       4)
    bull_g  = round(growth_mean + gs,  4)

    bear_m  = round(margin_mean - ms,  4)
    base_m  = round(margin_mean,       4)
    bull_m  = round(margin_mean + ms,  4)

    bear_pe = round(multi_center * (1.0 - mps), 2)
    base_pe = round(multi_center,               2)
    bull_pe = round(multi_center * (1.0 + mps), 2)

    def _pt(pe_val: float) -> Optional[float]:
        """Implied price target = EPS_2026 × multiple_scenario."""
        if eps_26 is None:
            return None
        return round(eps_26 * pe_val, 2)

    # ── MC distribution selection ─────────────────────────────────────────────
    g_dist, m_dist, pe_dist, tail = _mc_distributions(types)

    # Sigma = range / 3  (3-sigma covers full scenario range)
    g_sigma  = round(growth_range  / 3, 4)
    m_sigma  = round(margin_range  / 3, 4)
    pe_sigma = round(multi_range   / 3, 4)

    # ── Consistency check ─────────────────────────────────────────────────────
    consistency = _consistency_check(inp)

    # ── Assemble output ───────────────────────────────────────────────────────
    result = {
        "ticker":     ticker,
        "model_date": date.today().isoformat(),
        "excel_base_case": {
            "current_price":        curr_px,
            "revenue_growth":       growth,
            "gross_margin_current": gm,
            "net_margin_current":   nm,
            "eps_current":          eps_cur,
            "eps_forward": {
                "y1_2026": eps_26,
                "y2_2027": eps_27,
                "y3_2028": eps_28,
            },
            "price_target_base":    pt_base,
            "price_target_low":     pt_low,
            "price_target_high":    pt_high,
            "implied_multiple":     avg_pe,
            "pe_range_low":         pe_low,
            "pe_range_high":        pe_high,
            "wacc":                 wacc,
            "beta":                 beta,
        },
        "model_diagnostics": {
            "model_type":         types,
            "primary_driver":     primary,
            "assumption_quality": quality,
            "flags":              flags,
            "consistency_check":  consistency,
        },
        "normalized_assumptions": {
            "growth_mean":    growth_mean,
            "growth_range":   growth_range,
            "margin_mean":    margin_mean,
            "margin_range":   margin_range,
            "multiple_center":multi_center,
            "multiple_range": multi_range,
        },
        "scenario_mapping": {
            "bear": {
                "growth":               bear_g,
                "margin":               bear_m,
                "multiple":             bear_pe,
                "implied_price_target": _pt(bear_pe),
            },
            "base": {
                "growth":               base_g,
                "margin":               base_m,
                "multiple":             base_pe,
                "implied_price_target": _pt(base_pe),
            },
            "bull": {
                "growth":               bull_g,
                "margin":               bull_m,
                "multiple":             bull_pe,
                "implied_price_target": _pt(bull_pe),
            },
        },
        "monte_carlo_mapping": {
            "growth": {
                "distribution": g_dist,
                "mean":         growth_mean,
                "sigma":        g_sigma,
                "skew":         _SKEW_MAP.get(g_dist, 0.0),
            },
            "margin": {
                "distribution": m_dist,
                "mean":         margin_mean,
                "sigma":        m_sigma,
                "skew":         _SKEW_MAP.get(m_dist, 0.0),
            },
            "multiple": {
                "distribution": pe_dist,
                "mean":         multi_center,
                "sigma":        pe_sigma,
                "skew":         _SKEW_MAP.get(pe_dist, 0.0),
            },
            "tail_profile": tail,
        },
    }

    if save:
        path = save_result(result, outdir=outdir)
        import sys
        print(f"  saved → {path}", file=sys.stderr)

    return result
