"""
analysis/ticker_analyzer.py — Solo Mode output producer.

Reads a Layer 2 JSON and a ValidationGate snapshot dict and emits a
dict conforming to schemas/ticker_analysis_v1.schema.json.

All numeric values in the output satisfy:
  N1 — every DerivedValue.formula evaluates to .value within ±0.001 or ±0.5%
  F1 — current_price.vintage within 26h of as_of (equity)
  S1 — every dict with a `value` key carries source or derived: true
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional


def analyze_ticker(layer2: Dict[str, Any], snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a ticker_analysis_v1.json-shaped dict for a single equity.

    Parameters
    ----------
    layer2 : dict
        Contents of data/layer2/{TICKER}_layer2.json. Provides ticker
        identity, model_date, and scenario/monte_carlo mappings.
    snapshot : dict
        Output of ValidationGate.build_snapshot(). Provides live price,
        current percentiles (from valuation_range + Monte Carlo), and
        macro context.

    Returns
    -------
    dict conforming to schemas/ticker_analysis_v1.schema.json.

    Raises
    ------
    ValueError if required fields are missing from inputs.
    """
    # ── Validate required inputs ──────────────────────────────────────────
    _require(snapshot, "price", "distribution", "as_of_date")
    _require(snapshot["price"], "value", "source", "vintage")
    dist = snapshot["distribution"]
    _require(dist, "percentiles", "mode")
    percs = dist["percentiles"]
    _require(percs, "p5", "p25", "p50", "p75", "p95")

    # ── Current price (SourcedValue) ──────────────────────────────────────
    # Taken directly from snapshot — already has value/source/vintage
    # from build_snapshot's price dict (fixed 2026-04-23).
    price_val = float(snapshot["price"]["value"])
    current_price: Dict[str, Any] = {
        "value":   price_val,
        "source":  snapshot["price"]["source"],
        "vintage": snapshot["price"]["vintage"],
    }

    # ── Distribution (five SourcedValues) ────────────────────────────────
    dist_source = (
        "stockeval_monte_carlo"
        if dist["mode"] == "monte_carlo"
        else "stockeval_scenario_anchors"
    )
    dist_vintage = snapshot["as_of_date"]

    def _sv(val: float) -> Dict[str, Any]:
        return {
            "value":   round(val, 2),
            "source":  dist_source,
            "vintage": dist_vintage,
        }

    # Use rounded values so the formula literals and computed value agree exactly.
    p5r  = round(float(percs["p5"]),  2)
    p25r = round(float(percs["p25"]), 2)
    p50r = round(float(percs["p50"]), 2)
    p75r = round(float(percs["p75"]), 2)
    p95r = round(float(percs["p95"]), 2)

    distribution = {
        "p5":  _sv(p5r),
        "p25": _sv(p25r),
        "p50": _sv(p50r),
        "p75": _sv(p75r),
        "p95": _sv(p95r),
    }

    # ── Expected return (DerivedValue) ────────────────────────────────────
    # Weighted: 25% bear (p5) + 50% base (p50) + 25% bull (p95).
    # Formula uses rounded percentile literals and the exact stored price so
    # N1's AST evaluator produces the same result as the stored value.
    er_formula = (
        f"0.25 * ({p5r}/{price_val} - 1) + "
        f"0.50 * ({p50r}/{price_val} - 1) + "
        f"0.25 * ({p95r}/{price_val} - 1)"
    )
    er_value = round(
        0.25 * (p5r  / price_val - 1) +
        0.50 * (p50r / price_val - 1) +
        0.25 * (p95r / price_val - 1),
        4,
    )
    expected_return: Dict[str, Any] = {
        "value":   er_value,
        "formula": er_formula,
        "derived": True,
        "source":  "weighted_scenario_return_formula",
    }

    # ── Zone classification ───────────────────────────────────────────────
    if price_val <= p25r:
        zone = "current_below_p25"
    elif price_val <= p50r:
        zone = "current_between_p25_p50"
    else:
        zone = "current_above_p50"

    # ── Entry strategy ────────────────────────────────────────────────────
    if zone == "current_below_p25":
        # p40 interpolates 60% of the way from p25 to p50 (the 40th percentile
        # in the fallback distribution).  Mirrors _derive_fallback_percentiles.
        p40 = round(p25r + (p50r - p25r) * 0.60, 2)
        strong_buy_zone: Optional[Dict[str, float]] = {
            "low":  p5r,
            "high": round(price_val, 2),
        }
        ideal_buy_price = round(price_val, 2)
        starter_zone: Optional[Dict[str, float]] = {
            "low":  round(price_val, 2),
            "high": p40,
        }

    elif zone == "current_between_p25_p50":
        strong_buy_zone = {"low": p5r, "high": p25r}
        ideal_buy_price = p25r
        starter_zone    = {"low": p25r, "high": round(price_val, 2)}

    else:  # current_above_p50
        strong_buy_zone = None
        ideal_buy_price = p25r
        starter_zone    = None

    entry_strategy: Dict[str, Any] = {
        "current_zone_logic": zone,
        "strong_buy_zone":    strong_buy_zone,
        "ideal_buy_price":    ideal_buy_price,
        "starter_zone":       starter_zone,
        "trim_zone_above":    p75r,
        "exit_zone_above":    p95r,
    }

    # ── Position sizing ───────────────────────────────────────────────────
    exec_block    = snapshot.get("execution") or {}
    raw_target    = exec_block.get("target_size_pct")
    target_val    = float(raw_target) if isinstance(raw_target, (int, float)) else 1.5
    tranche_count = 3

    # target_size_pct: constant in v1; formula is the numeric literal itself.
    target_size_pct: Dict[str, Any] = {
        "value":   target_val,
        "formula": str(target_val),
        "derived": True,
        "source":  "advisor_input_v1",
    }

    no_position = (target_val == 0.0) or (zone == "current_above_p50")
    if no_position:
        rec_now_val     = 0.0
        rec_now_formula = "0.0"
    else:
        rec_now_val     = round(target_val / tranche_count, 4)
        rec_now_formula = f"{target_val} / {tranche_count}"

    recommended_now_pct: Dict[str, Any] = {
        "value":   rec_now_val,
        "formula": rec_now_formula,
        "derived": True,
        "source":  "recommended_now_derivation",
    }

    raw_conviction = exec_block.get("conviction", "")
    conviction = raw_conviction if raw_conviction in ("High", "Medium", "Low") else "Low"

    # entry_style: "wait" when no position is recommended (target=0 or above P50);
    # "staged" otherwise.  Known gap: wire momentum score in a later iteration
    # to distinguish "staged" from "aggressive".
    entry_style = "wait" if no_position else "staged"

    position_sizing: Dict[str, Any] = {
        "target_size_pct":     target_size_pct,
        "recommended_now_pct": recommended_now_pct,
        "conviction":          conviction,
        "entry_style":         entry_style,
        "tranche_count":       tranche_count,
    }

    # ── Upstream audit trail ──────────────────────────────────────────────
    layer2_json = json.dumps(layer2, sort_keys=True, separators=(",", ":"))
    layer2_hash = "sha256:" + hashlib.sha256(layer2_json.encode()).hexdigest()

    raw_vintage = layer2.get("model_date", "")
    layer2_vintage = (raw_vintage + "T00:00:00Z") if raw_vintage and "T" not in raw_vintage else raw_vintage

    upstream: Dict[str, Any] = {
        "stockeval_layer2_hash":    layer2_hash,
        "stockeval_layer2_vintage": layer2_vintage,
    }

    # ── Assemble and return ───────────────────────────────────────────────
    ticker = (
        layer2.get("ticker", "").upper()
        or snapshot.get("ticker", "").upper()
    )
    if not ticker:
        raise ValueError("ticker missing from both layer2 and snapshot")

    return {
        "schema_version":  "1.0",
        "as_of":           snapshot["as_of_date"],
        "ticker":          ticker,
        "kind":            "equity",
        "current_price":   current_price,
        "distribution":    distribution,
        "expected_return": expected_return,
        "entry_strategy":  entry_strategy,
        "position_sizing": position_sizing,
        "upstream":        upstream,
    }


def _require(d: dict, *keys: str) -> None:
    """Raise ValueError if any key is absent or None in d."""
    missing = [k for k in keys if d.get(k) is None]
    if missing:
        raise ValueError(f"Required fields missing from input: {missing}")


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import json as _json
    from jsonschema import Draft202012Validator
    from analysis.validation_gate import ValidationGate

    schema = _json.loads(
        Path("schemas/ticker_analysis_v1.schema.json").read_text()
    )
    validator = Draft202012Validator(schema)

    layer2 = _json.loads(Path("data/layer2/AXON_layer2.json").read_text())

    snapshot = {
        "ticker":     "AXON",
        "as_of_date": "2026-04-24T17:00:00Z",
        "kind":       "equity",
        "price": {
            "value":   395.47,
            "source":  "FMP",
            "vintage": "2026-04-24T16:55:00Z",
        },
        "distribution": {
            "mode": "fallback",
            "percentiles": {
                "p5":  304.62,
                "p25": 338.86,
                "p50": 403.38,
                "p75": 482.43,
                "p95": 544.71,
            },
        },
        "execution": {
            "target_size_pct": 1.5,
            "conviction":      "Low",
        },
    }

    result = analyze_ticker(layer2, snapshot)

    # ── Schema validation ─────────────────────────────────────────────────
    errors = list(validator.iter_errors(result))
    if errors:
        print("SCHEMA VALIDATION FAILED:")
        for e in errors:
            print(f"  - {list(e.absolute_path)}: {e.message}")
    else:
        print("✓ Output validates against ticker_analysis_v1 schema.")

    # ── Rule validation ───────────────────────────────────────────────────
    gate = ValidationGate()
    n1 = gate._b8_numerical_invariants(result)
    f1 = gate._b9_price_freshness(result)
    s1 = gate._b10_source_attribution(result)

    print(f"N1: passed={n1.passed}, corrections={len(n1.corrections)}")
    if n1.corrections:
        for c in n1.corrections:
            print(f"  correction: {c.field} {c.old!r} → {c.new!r}")

    print(f"F1: passed={f1.passed}, prices_checked={f1.metadata.get('prices_checked')}, "
          f"classification={f1.metadata.get('live_count', 0) and 'live' or f1.metadata}")

    print(f"S1: passed={s1.passed}, unsourced_count={s1.metadata.get('unsourced_count')}")
    if s1.metadata.get("unsourced_count", 0) > 0:
        for c in s1.corrections:
            print(f"  unsourced: {c.field}")

    print()
    print(_json.dumps(result, indent=2))
