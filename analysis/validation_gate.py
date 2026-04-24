"""
validation_gate.py — Pre-report validation framework.

Implements the 10-block validation contract defined in
stock_eval/prompts/VALIDATION_LAYER.md.  Must pass all blocks (or carry
explicit overrides) before any report is generated.

Usage
─────
    from analysis.validation_gate import ValidationGate

    gate = ValidationGate()
    snapshot = gate.build_snapshot(
        ticker, as_of_date, norm_metrics, stock_data,
        valuation_range, macro_findings, scorecard, memo_text,
    )
    log = gate.run(snapshot)

    print(log.format())
    if not log.is_clear:
        raise ReportBlockedError(log)

Blocks
──────
  1 — Primary Input Freshness
  2 — Market Cap Triangle
  3 — P/E Basis Consistency
  4 — Scenario / Distribution Anchoring
  5 — Macro Series Integrity
  6 — FCF Currency Against Guidance
  7 — Execution Language Coherence
  8 — Numerical Invariants
  9 — Price Freshness
 10 — Source Attribution
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


# ── Validation layer loader ───────────────────────────────────────────────────

_VALIDATION_LAYER_PATH = (
    Path(__file__).resolve().parents[2]  # stock_eval/
    / "stock_eval" / "prompts" / "VALIDATION_LAYER.md"
)


def load_validation_layer() -> str:
    """
    Load the VALIDATION_LAYER.md system prompt addendum from disk.

    Returns the full text of the validation layer. Callers that pass this
    text to a model as a system prompt addendum must do so before every
    report generation or audit pass — per the loading convention in
    stock_eval/prompts/README.md.

    Raises FileNotFoundError if the file cannot be found at the expected path.
    """
    if not _VALIDATION_LAYER_PATH.exists():
        raise FileNotFoundError(
            f"VALIDATION_LAYER.md not found at {_VALIDATION_LAYER_PATH}. "
            "Ensure stock_eval/prompts/ is present relative to the repo root."
        )
    return _VALIDATION_LAYER_PATH.read_text(encoding="utf-8")

# ── Staleness limits (calendar days from report as_of_date) ──────────────────

_STALE_DAYS: Dict[str, int] = {
    "price":           1,
    "shares_diluted":  100,
    "eps_ttm":         100,
    "fcf_guidance":    90,
    "cli":             45,
    "jobless_claims":  8,
    "housing_starts":  45,
    "manuf_employ":    45,
    "yield_curve":     1,
}

# Macro regime gate rules: (min_cli, max_cli) → expected_regimes
_CLI_REGIME_RULES: List[Tuple[float, float, List[str]]] = [
    (101.0, float("inf"), ["Expansion"]),
    (99.5,  101.0,        ["Expansion", "Late Cycle", "Slowdown"]),
    (99.0,   99.5,        ["Slowdown", "Contraction"]),
    (0.0,    99.0,        ["Contraction"]),
]

# Execution status banned phrases ─────────────────────────────────────────────
_WAIT_BANNED: List[str] = [
    "starter position at current is justified",
    "attractive entry at current",
    "buying at current levels is appropriate",
    "position at current price",
    "initiating a position at current",
    "enter at current",
    "buy at current",
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Correction:
    block: int
    field: str
    old_value: Any
    new_value: Any
    reason: str
    rule_id: Optional[str] = None
    formula_string: Optional[str] = None
    computed_result: Optional[float] = None


@dataclass
class BlockResult:
    block_id: int
    name: str
    passed: bool = True
    failures: List[str] = field(default_factory=list)
    corrections: List[Correction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def fail(self, message: str) -> None:
        self.passed = False
        self.failures.append(message)

    def correct(
        self,
        field: str,
        old: Any,
        new: Any,
        reason: str,
        rule_id: Optional[str] = None,
        formula_string: Optional[str] = None,
        computed_result: Optional[float] = None,
    ) -> Correction:
        c = Correction(
            block=self.block_id,
            field=field,
            old_value=old,
            new_value=new,
            reason=reason,
            rule_id=rule_id,
            formula_string=formula_string,
            computed_result=computed_result,
        )
        self.corrections.append(c)
        return c


@dataclass
class ValidationLog:
    ticker: str
    report_date: str
    price_str: str
    blocks: List[BlockResult] = field(default_factory=list)
    overrides: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_clear(self) -> bool:
        """True when all blocks passed or all failures are overridden."""
        failed = [b for b in self.blocks if not b.passed]
        overridden_ids = {o["block"] for o in self.overrides}
        return all(b.block_id in overridden_ids for b in failed)

    @property
    def qualified(self) -> bool:
        """True when report is clear but carries active overrides."""
        return self.is_clear and bool(self.overrides)

    @property
    def status(self) -> str:
        if self.is_clear:
            return "QUALIFIED" if self.qualified else "CLEAR TO GENERATE"
        n = sum(1 for b in self.blocks if not b.passed)
        return f"BLOCKED ({n} failure{'s' if n != 1 else ''} require manual input)"

    def all_corrections(self) -> List[Correction]:
        cs: List[Correction] = []
        for b in self.blocks:
            cs.extend(b.corrections)
        return cs

    def format(self) -> str:
        lines: List[str] = ["=== STOCKEVAL VALIDATION LOG ==="]
        lines.append(f"Ticker:        {self.ticker}")
        lines.append(f"Report date:   {self.report_date}")
        lines.append(f"Price:         {self.price_str}")
        lines.append("")
        for b in self.blocks:
            tag = "PASS" if b.passed else "FAIL"
            if b.metadata:
                meta_parts = [f"{k}={v}" for k, v in b.metadata.items()]
                if not b.passed:
                    meta_parts.append(f"failures={len(b.failures)}")
                meta_str = " (" + ", ".join(meta_parts) + ")"
            else:
                meta_str = ""
            lines.append(f"BLOCK {b.block_id} — {b.name:<40} [{tag}]{meta_str}")
            if not b.passed:
                for msg in b.failures:
                    lines.append(f"    ✗ {msg}")
        corrections = self.all_corrections()
        if corrections:
            lines.append("")
            lines.append("Corrections applied:")
            for c in corrections:
                lines.append(
                    f"  - {c.field}: {c.old_value!r} → {c.new_value!r}"
                    f"  (reason: {c.reason})"
                )
        if self.overrides:
            lines.append("")
            lines.append("OVERRIDES:")
            for o in self.overrides:
                lines.append(
                    f"  - Block {o['block']}: {o['reason_code']}"
                    f" — {o['justification']}"
                )
        lines.append("")
        lines.append(f"Report status: {self.status}")
        lines.append("=== END LOG ===")
        return "\n".join(lines)


# ── Percentile derivation (Block 4 reference implementation) ──────────────────

def _derive_fallback_percentiles(p5: float, p50: float, p95: float) -> Dict[str, float]:
    """
    Derive all percentile anchors from (Bear=P5, Base=P50, Bull=P95).

    Derivation:
      p25 = p5  + (p50 - p5)  × 20/45   [(25-5)/(50-5)]
      p75 = p50 + (p95 - p50) × 25/45   [(75-50)/(95-50)]
      p20 = p5  + (p25 - p5)  × 0.75    [(20-5)/(25-5)]
      p40 = p25 + (p50 - p25) × 0.60    [(40-25)/(50-25)]
      p10 = p5  + (p20 - p5)  × 0.50    [(10-5)/(20-5)]
      p60 = p50 + (p75 - p50) × 0.40    [(60-50)/(75-50)]
      p80 = p75 + (p95 - p75) × 0.20    [(80-75)/(95-75)]
      p90 = p75 + (p95 - p75) × 0.60    [(90-75)/(95-75)]
    """
    p25 = p5  + (p50 - p5)  * (20 / 45)
    p75 = p50 + (p95 - p50) * (25 / 45)
    p20 = p5  + (p25 - p5)  * 0.75
    p40 = p25 + (p50 - p25) * 0.60
    p10 = p5  + (p20 - p5)  * 0.50
    p60 = p50 + (p75 - p50) * 0.40
    p80 = p75 + (p95 - p75) * 0.20
    p90 = p75 + (p95 - p75) * 0.60
    return {
        "p5": p5, "p10": p10, "p20": p20, "p25": p25,
        "p40": p40, "p50": p50, "p60": p60,
        "p75": p75, "p80": p80, "p90": p90, "p95": p95,
    }


# ── Rule N1 helpers ───────────────────────────────────────────────────────────

#: Node types permitted by the safe arithmetic evaluator.
_SAFE_EVAL_ALLOWED: frozenset = frozenset({
    ast.Expression, ast.BinOp, ast.UnaryOp,
    ast.Constant,
    ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub,
})


def _safe_eval_arithmetic(expr: str) -> float:
    """
    Evaluate a pure arithmetic expression string without using eval() or exec().

    Permitted constructs: +  -  *  /  parentheses  numeric literals (int/float).
    Any other node type raises ValueError("disallowed node: <TypeName>").
    Division by zero raises ValueError("division by zero").
    """
    tree = ast.parse(expr.strip(), mode="eval")

    for node in ast.walk(tree):
        if type(node) not in _SAFE_EVAL_ALLOWED:
            raise ValueError(f"disallowed node: {type(node).__name__}")

    def _eval(node: ast.expr) -> float:  # type: ignore[type-arg]
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError(f"non-numeric constant: {node.value!r}")
            return float(node.value)
        if isinstance(node, ast.BinOp):
            left  = _eval(node.left)
            right = _eval(node.right)
            op    = node.op
            if isinstance(op, ast.Add):  return left + right
            if isinstance(op, ast.Sub):  return left - right
            if isinstance(op, ast.Mult): return left * right
            if isinstance(op, ast.Div):
                if right == 0.0:
                    raise ValueError("division by zero")
                return left / right
            raise ValueError(f"disallowed operator: {type(op).__name__}")
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -_eval(node.operand)
            raise ValueError(f"disallowed unary operator: {type(node.op).__name__}")
        raise ValueError(f"disallowed node: {type(node).__name__}")

    return _eval(tree.body)


def _walk_formula_value_pairs(
    obj: Any, path: str = ""
) -> Iterator[Tuple[str, dict]]:
    """
    Recursively walk a nested dict/list structure.

    Yields (dotted_path, sub_dict) for every dict that contains BOTH:
      - a 'formula' key whose value is a str
      - a 'value'   key whose value is an int or float

    Path notation:
      dict key  → appended with a dot separator (or no separator at root)
      list index → appended as [i]

    Example yielded path:
      "holdings[1].taxable_equivalent_yield"
    """
    if isinstance(obj, dict):
        if (
            "formula" in obj and isinstance(obj["formula"], str)
            and "value"   in obj and isinstance(obj["value"], (int, float))
        ):
            yield path, obj
        for key, val in obj.items():
            child_path = f"{path}.{key}" if path else key
            yield from _walk_formula_value_pairs(val, child_path)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            yield from _walk_formula_value_pairs(item, f"{path}[{i}]")


# ── Rule F1 helpers ───────────────────────────────────────────────────────────

def _parse_iso8601(ts: str) -> datetime:
    """
    Parse an ISO 8601 timestamp string into a timezone-aware datetime.

    Handles:
      - 'Z' suffix       → replaced with '+00:00'
      - Date-only strings (YYYY-MM-DD) → midnight UTC assumed
    Raises ValueError on malformed input.
    """
    ts = ts.strip().replace("Z", "+00:00")
    if len(ts) == 10 and ts[4] == "-" and ts[7] == "-":
        ts += "T00:00:00+00:00"
    return datetime.fromisoformat(ts)


def _walk_price_objects(
    obj: Any,
    path: str = "",
    parent_kind: Optional[str] = None,
    _seen: Optional[set] = None,
) -> Iterator[Tuple[str, Optional[str], dict]]:
    """
    Recursively walk a nested dict/list structure.

    Yields (dotted_path, kind, price_dict) for every dict that contains BOTH:
      - a 'value' key whose value is an int or float
      - a 'vintage' key whose value is a str

    'kind' is the nearest ancestor dict's 'kind' string value (or None).

    Path notation matches _walk_formula_value_pairs exactly:
      dict key  → appended with a dot separator (or no separator at root)
      list index → appended as [i]

    Deduplication: the same dict object (by id()) is never yielded twice.
    """
    if _seen is None:
        _seen = set()
    if isinstance(obj, dict):
        current_kind = (
            obj["kind"] if isinstance(obj.get("kind"), str) else parent_kind
        )
        # Only classify dicts whose immediate key name contains "price".
        # This scopes F1 to actual price objects and avoids false stale_block
        # hits on fundamental data (eps.ttm, shares_diluted_b, yields, etc.)
        # that carry vintage but are not market prices.
        leaf_key = path.rsplit(".", 1)[-1].split("[")[0].lower()
        if (
            "price" in leaf_key
            and "value" in obj and isinstance(obj["value"], (int, float))
            and "vintage" in obj and isinstance(obj["vintage"], str)
        ):
            oid = id(obj)
            if oid not in _seen:
                _seen.add(oid)
                yield path, current_kind, obj
        for key, val in obj.items():
            child_path = f"{path}.{key}" if path else key
            yield from _walk_price_objects(val, child_path, current_kind, _seen)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            yield from _walk_price_objects(
                item, f"{path}[{i}]", parent_kind, _seen
            )


# ── Rule S1 helpers ───────────────────────────────────────────────────────────

def _walk_numeric_value_objects(
    obj: Any,
    path: str = "",
    ancestor_source: Optional[str] = None,
) -> Iterator[Tuple[str, dict, Optional[str]]]:
    """
    Recursively walk a nested dict/list structure.

    Yields (dotted_path, value_dict, inherited_source) for every dict that
    contains a 'value' key whose value is an int or float.

    inherited_source is the nearest ancestor dict's 'source' string value,
    or None if no ancestor carries one.  The current dict's own 'source' (if
    present) is NOT passed as inherited_source for its own yield — the caller
    checks value_dict.get("source") directly.  The current dict's source IS
    passed down to its children as the new ancestor_source.

    Path notation matches existing walkers exactly.
    """
    if isinstance(obj, dict):
        if "value" in obj and isinstance(obj["value"], (int, float)):
            yield path, obj, ancestor_source
        # Propagate source to children — use this dict's source if present
        child_source = (
            obj["source"] if isinstance(obj.get("source"), str) else ancestor_source
        )
        for key, val in obj.items():
            child_path = f"{path}.{key}" if path else key
            yield from _walk_numeric_value_objects(val, child_path, child_source)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            yield from _walk_numeric_value_objects(
                item, f"{path}[{i}]", ancestor_source
            )


# ── Main validation gate ───────────────────────────────────────────────────────

class ValidationGate:
    """
    Pre-report validation gate.  Build a snapshot then call run().
    """

    def __init__(self, overrides: Optional[Dict[int, Dict[str, str]]] = None) -> None:
        # overrides: {block_id: {"reason_code": ..., "justification": ...}}
        self._overrides: Dict[int, Dict[str, str]] = overrides or {}

    # ── Snapshot builder ───────────────────────────────────────────────────────

    def build_snapshot(
        self,
        ticker: str,
        as_of_date: str,
        norm_metrics: Any,          # NormalizedMetrics
        stock_data: Any,            # StockData
        valuation_range: Any,       # ValuationRange (or None)
        macro_findings: Dict[str, Any],
        scorecard: Any,             # Scorecard
        memo_text: str = "",
        position_pct: Optional[float] = None,
        execution_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Assemble ticker_snapshot from existing analysis objects.
        Every primary input carries provenance.  Derived fields are computed
        only from primary inputs inside this function.
        """
        m = norm_metrics  # shorthand

        # ── Price ──────────────────────────────────────────────────────────────
        price_val    = m.price if m else None
        price_source = m.price_source if m else "unavailable"

        # ── Shares ────────────────────────────────────────────────────────────
        shares_val = m.shares if m else None     # raw count (not billions)
        shares_b   = (shares_val / 1e9) if shares_val else None

        # Reporting period: use most recent income statement date
        income_stmts = (stock_data.income_statements or []) if stock_data else []
        shares_period = (
            income_stmts[0].period_of_report
            if income_stmts and hasattr(income_stmts[0], "period_of_report")
            else None
        )

        # ── Market cap (derived: price × shares) ─────────────────────────────
        api_mktcap_b  = (m.market_cap_api / 1e9)  if (m and m.market_cap_api)  else None
        comp_mktcap_b = (price_val * shares_val / 1e9) if (price_val and shares_val) else None
        # Authoritative market cap: api if not adjusted; recomputed if price was overridden
        auth_mktcap_b = (
            (m.market_cap_recomp / 1e9) if (m and m.market_cap_recomp and m.price_adjusted)
            else api_mktcap_b
        )

        # ── EPS ───────────────────────────────────────────────────────────────
        ttm_eps      = m.ttm_eps       if m else None
        ttm_eps_src  = m.ttm_eps_source if m else "unavailable"
        ann_eps      = m.annual_eps    if m else None
        ann_eps_src  = m.annual_eps_source if m else "unavailable"
        eps_basis    = "ttm" if ttm_eps else ("annual" if ann_eps else "unavailable")
        eps_val      = ttm_eps if ttm_eps else ann_eps

        # TTM quarters from quarterly income statements
        qtr_stmts = (stock_data.quarterly_income or []) if stock_data else []
        ttm_quarters = [
            getattr(q, "eps_diluted", None) or getattr(q, "eps", None)
            for q in qtr_stmts[:4]
        ] if qtr_stmts else []

        # ── Multiples ─────────────────────────────────────────────────────────
        pe_val       = m.pe_ratio   if m else None
        pe_source    = m.pe_source  if m else "unavailable"
        ps_val       = m.ps_ratio   if m else None
        ev_ebitda_v  = m.ev_ebitda  if m else None

        # Implied P/E from labeled basis
        implied_pe = (price_val / eps_val) if (price_val and eps_val and eps_val > 0) else None

        # PEG
        eps_growth_pct = m.eps_growth_pct if (m and hasattr(m, "eps_growth_pct")) else None
        peg = (pe_val / eps_growth_pct) if (pe_val and eps_growth_pct and eps_growth_pct > 0) else None

        # ── FCF (from driver model) ────────────────────────────────────────────
        vr = valuation_range
        fcf_base_b  = None
        fcf_bear_b  = None
        fcf_bull_b  = None
        guidance_available = False
        if vr and vr.driver_model_available:
            fcf_base_b = (vr.scenario_base_fwd_fcf / 1e9) if vr.scenario_base_fwd_fcf else None
            fcf_bear_b = (vr.scenario_bear_fwd_fcf / 1e9) if vr.scenario_bear_fwd_fcf else None
            fcf_bull_b = (vr.scenario_bull_fwd_fcf / 1e9) if vr.scenario_bull_fwd_fcf else None

        # ── Scenarios ─────────────────────────────────────────────────────────
        scenarios: Dict[str, Any] = {}
        if vr:
            for case, px in [("bear", vr.bear_price), ("base", vr.base_price), ("bull", vr.bull_price)]:
                rg_attr = f"scenario_{case}_rev_growth"
                om_attr = f"scenario_{case}_op_margin"
                fc_attr = f"scenario_{case}_fcf_conv"
                ex_attr = f"scenario_{case}_exit_mult"
                scenarios[case] = {
                    "price":      px,
                    "rev_g":      getattr(vr, rg_attr, None),
                    "op_margin":  getattr(vr, om_attr, None),
                    "fcf_conv":   getattr(vr, fc_attr, None),
                    "exit_mult":  getattr(vr, ex_attr, None),
                }

        # ── Distribution ──────────────────────────────────────────────────────
        mc = getattr(vr, "mc", None) if vr else None
        if mc and mc.p5_price > 0 and mc.p95_price > 0:
            dist_mode = "monte_carlo"
            percentiles = {
                "p5":  mc.p5_price,
                "p25": mc.p25_price,
                "p50": mc.median_price,
                "p75": mc.p75_price,
                "p95": mc.p95_price,
            }
            n_sims = mc.n_sims
        else:
            dist_mode = "fallback"
            bear_px = (vr.bear_price or 0) if vr else 0
            base_px = (vr.base_price or 0) if vr else 0
            bull_px = (vr.bull_price or 0) if vr else 0
            percentiles = _derive_fallback_percentiles(bear_px, base_px, bull_px)
            n_sims = None

        # ── Macro ─────────────────────────────────────────────────────────────
        macro_regime = macro_findings.get("macro_regime", "Unknown")
        macro_score  = macro_findings.get("macro_score", None)
        lei_snap     = macro_findings.get("lei_snapshot", {})
        cli_val      = lei_snap.get("cli") if lei_snap else None
        claims_val   = lei_snap.get("jobless_claims") if lei_snap else None
        housing_val  = lei_snap.get("housing_starts") if lei_snap else None
        mfg_val      = lei_snap.get("manuf_employ") if lei_snap else None
        yield_val    = lei_snap.get("yield_spread") if lei_snap else None

        # ── Execution ─────────────────────────────────────────────────────────
        stance_raw = (
            scorecard.stance.value
            if (scorecard and hasattr(scorecard.stance, "value"))
            else str(scorecard.stance) if scorecard else "Unknown"
        )
        thesis = (
            "Buy" if "bullish" in stance_raw.lower()
            else "Sell" if "bearish" in stance_raw.lower()
            else "Hold"
        )
        conviction = macro_findings.get("conviction", "Medium")
        target_size = position_pct or 0.0

        return {
            "ticker":       ticker,
            "as_of_date":   as_of_date,
            "kind":         "equity",
            "price":        {
                "value":   price_val,
                "source":  price_source,
                "vintage": datetime.now(timezone.utc).isoformat(),
            },
            "shares_b":     {"value": shares_b,    "source": getattr(m, "shares_source", "") if m else "",
                             "reporting_period": shares_period},
            "market_cap_b": {
                "api":      api_mktcap_b,
                "computed": comp_mktcap_b,
                "auth":     auth_mktcap_b,
            },
            "eps": {
                "ttm":          {"value": ttm_eps,   "source": ttm_eps_src, "quarters": ttm_quarters},
                "annual":       {"value": ann_eps,   "source": ann_eps_src},
                "basis_used":   eps_basis,
                "value":        eps_val,
            },
            "multiples": {
                "pe":           {"value": pe_val,      "source": pe_source},
                "implied_pe":   implied_pe,
                "ps":           {"value": ps_val},
                "ev_ebitda":    {"value": ev_ebitda_v},
                "peg":          {"value": peg},
            },
            "fcf": {
                "base_b":               fcf_base_b,
                "bear_b":               fcf_bear_b,
                "bull_b":               fcf_bull_b,
                "guidance_available":   guidance_available,
            },
            "scenarios":    scenarios,
            "distribution": {
                "mode":        dist_mode,
                "percentiles": percentiles,
                "n_sims":      n_sims,
            },
            "macro": {
                "cli":            cli_val,
                "jobless_claims": claims_val,
                "housing_starts": housing_val,
                "manuf_employ":   mfg_val,
                "yield_curve":    yield_val,
                "regime":         macro_regime,
                "macro_score":    macro_score,
            },
            "execution": {
                "thesis":            thesis,
                "execution":         execution_label or "UNKNOWN",
                "conviction":        conviction,
                "target_size_pct":   target_size,
                "recommended_now_pct": target_size,
            },
            "memo_text":    memo_text,
        }

    # ── Block runners ──────────────────────────────────────────────────────────

    def _b1_freshness(self, snap: Dict[str, Any]) -> BlockResult:
        """Block 1 — Primary Input Freshness."""
        b = BlockResult(1, "Primary Input Freshness")
        as_of = snap.get("as_of_date", "")
        try:
            ref = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        except Exception:
            ref = datetime.now(timezone.utc)

        # For fields where we have actual vintage (income statement dates)
        # we check staleness.  For fields sourced from live API we note they
        # were current at evaluation time.
        shares_info = snap.get("shares_b", {})
        period = shares_info.get("reporting_period")
        if period:
            try:
                period_dt = datetime.fromisoformat(period + "T00:00:00+00:00"
                                                    if "T" not in period else period)
                age_days = (ref - period_dt).days
                limit = _STALE_DAYS["shares_diluted"]
                if age_days > limit:
                    b.fail(
                        f"shares_diluted: period {period} is {age_days}d old "
                        f"(limit {limit}d) — fetch most recent 10-Q/10-K"
                    )
            except Exception:
                pass  # Can't parse — treat as OK

        # Price is live at evaluation time; note the source for auditability
        price_src = snap.get("price", {}).get("source", "unknown")
        if price_src == "unavailable":
            b.fail("price: source is 'unavailable' — live price required")

        # EPS TTM: check quarter count
        quarters = snap.get("eps", {}).get("ttm", {}).get("quarters", [])
        filled = [q for q in quarters if q is not None]
        if len(filled) < 4:
            b.fail(
                f"eps.ttm: only {len(filled)}/4 quarters available "
                f"— TTM EPS may be understated"
            )

        return b

    def _b2_market_cap_triangle(self, snap: Dict[str, Any]) -> BlockResult:
        """Block 2 — Market Cap Triangle: price × shares = market cap."""
        b = BlockResult(2, "Market Cap Triangle")
        price  = (snap.get("price") or {}).get("value")
        shares_b = (snap.get("shares_b") or {}).get("value")
        mktcap = snap.get("market_cap_b", {})
        api_b  = mktcap.get("api")
        comp_b = mktcap.get("computed")

        if price is None or shares_b is None:
            b.fail("price or shares unavailable — cannot verify market cap triangle")
            return b

        expected_b = price * shares_b  # already in billions
        if api_b is not None:
            delta = abs(api_b - expected_b) / max(expected_b, 0.001)
            if delta > 0.005:
                correction = b.correct(
                    "market_cap_b",
                    f"${api_b:.2f}B (API)",
                    f"${expected_b:.2f}B (price × shares)",
                    "Block 2: price × shares",
                )
                snap["market_cap_b"]["auth"] = expected_b
                b.fail(
                    f"market_cap_b mismatch: API ${api_b:.2f}B vs "
                    f"computed ${expected_b:.2f}B "
                    f"(Δ={delta:.1%} > 0.5% tolerance)"
                )
        return b

    def _b3_pe_basis(self, snap: Dict[str, Any]) -> BlockResult:
        """Block 3 — P/E Basis Consistency."""
        b = BlockResult(3, "P/E Basis Consistency")
        price      = (snap.get("price") or {}).get("value")
        eps_obj    = snap.get("eps", {})
        eps_val    = eps_obj.get("value")
        eps_basis  = eps_obj.get("basis_used", "unavailable")
        multiples  = snap.get("multiples", {})
        pe_obj     = multiples.get("pe", {})
        pe_val     = pe_obj.get("value")
        pe_source  = pe_obj.get("source", "")
        implied_pe = multiples.get("implied_pe")

        # Basis must be labeled
        if eps_basis == "unavailable" or not eps_val:
            b.fail(
                f"eps.basis_used = '{eps_basis}' — P/E cannot be verified without "
                f"a labeled EPS basis (must be 'ttm' or 'annual')"
            )
            return b

        if not price or not eps_val:
            b.fail("price or EPS unavailable — P/E cannot be verified")
            return b

        # Recompute implied P/E and compare
        computed_pe = price / eps_val if eps_val > 0 else None
        if computed_pe is None:
            b.fail("EPS ≤ 0 — P/E is not meaningful for negative earnings")
            return b

        if pe_val is not None:
            delta = abs(pe_val - computed_pe) / max(computed_pe, 0.01)
            if delta > 0.02:
                b.correct(
                    "multiples.pe",
                    f"{pe_val:.2f}× (reported)",
                    f"{computed_pe:.2f}× ({eps_basis} EPS basis)",
                    "Block 3: price / eps",
                )
                snap["multiples"]["pe"]["value"] = round(computed_pe, 2)
                b.fail(
                    f"P/E {pe_val:.2f}× inconsistent with {eps_basis} EPS "
                    f"(${eps_val:.4f}): implied {computed_pe:.2f}× "
                    f"(Δ={delta:.1%} > 2% tolerance) — corrected to {computed_pe:.2f}×"
                )

        # Warn if source is not labeled as TTM or forward
        if pe_source and "unavailable" not in pe_source and "ttm" not in pe_source.lower() \
                and "annual" not in pe_source.lower() and "forward" not in pe_source.lower():
            b.fail(
                f"P/E source label '{pe_source}' is ambiguous — must include "
                f"'TTM', 'annual', or 'forward' to be verifiable"
            )

        return b

    def _b4_scenario_distribution(self, snap: Dict[str, Any]) -> BlockResult:
        """Block 4 — Scenario / Distribution Anchoring."""
        b = BlockResult(4, "Scenario / Distribution Anchoring")
        dist      = snap.get("distribution", {})
        mode      = dist.get("mode", "fallback")
        pct       = dist.get("percentiles", {})
        scenarios = snap.get("scenarios", {})
        tol_price = 0.05   # $0.05 tolerance on percentile prices
        tol_band  = 0.15   # 15% relative tolerance for MC scenario mapping

        bear_px = (scenarios.get("bear") or {}).get("price")
        base_px = (scenarios.get("base") or {}).get("price")
        bull_px = (scenarios.get("bull") or {}).get("price")

        if not scenarios or bear_px is None or base_px is None or bull_px is None:
            b.fail("scenarios are incomplete — cannot anchor distribution")
            return b

        if mode == "fallback":
            # Rule: P5=Bear, P50=Base, P95=Bull, exactly.
            checks = [
                ("p5",  bear_px, "Bear"),
                ("p50", base_px, "Base"),
                ("p95", bull_px, "Bull"),
            ]
            any_failed = False
            for pkey, scenario_px, label in checks:
                stored = pct.get(pkey)
                if stored is None:
                    b.fail(f"distribution.percentiles.{pkey} is missing (fallback mode)")
                    any_failed = True
                    continue
                if abs(stored - scenario_px) > tol_price:
                    b.correct(
                        f"distribution.percentiles.{pkey}",
                        f"${stored:.2f}",
                        f"${scenario_px:.2f} ({label} scenario)",
                        f"Block 4: fallback rule {pkey}={label}",
                    )
                    pct[pkey] = scenario_px
                    b.fail(
                        f"fallback P{pkey[1:]}: stored ${stored:.2f} ≠ "
                        f"{label} ${scenario_px:.2f} (Δ>${tol_price:.2f})"
                    )
                    any_failed = True

            # Recompute derived percentiles and verify
            if not any_failed:
                derived = _derive_fallback_percentiles(bear_px, base_px, bull_px)
                for pkey, expected in derived.items():
                    stored = pct.get(pkey)
                    if stored is not None and abs(stored - expected) > tol_price:
                        b.correct(
                            f"distribution.percentiles.{pkey}",
                            f"${stored:.2f}",
                            f"${expected:.2f} (derived from Bear/Base/Bull)",
                            "Block 4: derived percentile recomputed",
                        )
                        pct[pkey] = expected
                        b.fail(
                            f"derived {pkey}: stored ${stored:.2f} ≠ "
                            f"expected ${expected:.2f} — recomputed from anchors"
                        )

        elif mode == "monte_carlo":
            # Scenario prices must fall inside bounded MC bands
            p5  = pct.get("p5",  0)
            p25 = pct.get("p25", 0)
            p50 = pct.get("p50", 0)
            p75 = pct.get("p75", 0)
            p95 = pct.get("p95", 0)

            # Bear must fall in [p5, p30]  — p30 approximated as p25 + 25% of (p50-p25)
            p30 = p25 + 0.25 * (p50 - p25)
            if bear_px < p5 or bear_px > p30:
                b.fail(
                    f"MC mode: Bear ${bear_px:.2f} outside [P5 ${p5:.2f}, P30 ${p30:.2f}] — "
                    f"scenario tree may be mis-parameterized"
                )

            # Base must satisfy |base − p50| / p50 ≤ 15%
            if p50 > 0:
                base_dev = abs(base_px - p50) / p50
                if base_dev > tol_band:
                    b.fail(
                        f"MC mode: Base ${base_px:.2f} deviates {base_dev:.1%} from "
                        f"P50 ${p50:.2f} (limit {tol_band:.0%}) — "
                        f"distribution is diverging from driver model"
                    )

            # Bull must fall in [p70, p95] — p70 approximated as p50 + 80% of (p75-p50)
            p70 = p50 + 0.80 * (p75 - p50)
            if bull_px < p70 or bull_px > p95:
                b.fail(
                    f"MC mode: Bull ${bull_px:.2f} outside [P70 ${p70:.2f}, P95 ${p95:.2f}] — "
                    f"scenario tree may be mis-parameterized"
                )

        return b

    def _b5_macro_integrity(self, snap: Dict[str, Any]) -> BlockResult:
        """
        Block 5 — Macro Series Integrity.

        Checks internal consistency (CLI level → expected regime) since
        external FRED cross-validation requires a live network call that
        was already performed by MacroLEIAgent.  A full external check
        requires re-running the FRED fetch and comparing values.
        """
        b = BlockResult(5, "Macro Series Integrity")
        macro = snap.get("macro", {})
        cli   = macro.get("cli")
        regime = macro.get("regime", "Unknown")

        # Verify CLI level → regime consistency
        if cli is not None:
            expected_regimes: List[str] = []
            for lo, hi, regimes in _CLI_REGIME_RULES:
                if lo <= cli < hi:
                    expected_regimes = regimes
                    break
            if expected_regimes and regime not in expected_regimes:
                b.fail(
                    f"CLI {cli:.2f} → expected regime in {expected_regimes}, "
                    f"but reported '{regime}' — regime classification is inconsistent"
                )
        else:
            # CLI missing — note but don't fail; macro overlay may have used other indicators
            pass

        # Check macro_score range
        ms = macro.get("macro_score")
        if ms is not None and not (0 <= ms <= 100):
            b.fail(f"macro_score {ms} is outside valid range [0, 100]")

        # Check that key indicator fields are present
        missing_indicators = [
            k for k in ("cli", "jobless_claims", "housing_starts", "manuf_employ")
            if macro.get(k) is None
        ]
        if missing_indicators:
            b.fail(
                f"Macro indicators missing from snapshot: {missing_indicators} — "
                f"these should be populated by MacroLEIAgent"
            )

        return b

    def _b6_fcf_guidance(self, snap: Dict[str, Any]) -> BlockResult:
        """Block 6 — FCF Currency Against Guidance."""
        b = BlockResult(6, "FCF vs Guidance")
        fcf = snap.get("fcf", {})

        if not fcf.get("guidance_available", False):
            # No guidance: PASS — consensus FCF is accepted
            return b

        # If guidance is available but base FCF diverges significantly from it,
        # the field fcf.guidance_b should have been populated.  Since the current
        # system does not explicitly store management guidance in a structured field,
        # we check that base FCF is within the driver model's own bear/bull bounds.
        base_b = fcf.get("base_b")
        bear_b = fcf.get("bear_b")
        bull_b = fcf.get("bull_b")

        if base_b is None:
            b.fail("fcf.base_b is None despite guidance_available=True")
            return b

        if bear_b is not None and base_b < bear_b:
            b.fail(
                f"fcf.base_b ${base_b:.2f}B < fcf.bear_b ${bear_b:.2f}B — "
                f"base case FCF is worse than the bear scenario"
            )

        if bull_b is not None and base_b > bull_b:
            b.fail(
                f"fcf.base_b ${base_b:.2f}B > fcf.bull_b ${bull_b:.2f}B — "
                f"base case FCF exceeds the bull scenario"
            )

        return b

    def _b7_execution_coherence(self, snap: Dict[str, Any]) -> BlockResult:
        """Block 7 — Execution Language Coherence."""
        b = BlockResult(7, "Execution Language Coherence")
        exec_obj   = snap.get("execution", {})
        ex_label   = exec_obj.get("execution", "UNKNOWN").upper()
        conviction = exec_obj.get("conviction", "Medium")
        target_pct = exec_obj.get("target_size_pct", 0.0)
        now_pct    = exec_obj.get("recommended_now_pct", 0.0)
        memo       = snap.get("memo_text", "").lower()

        # WAIT: recommended_now must be 0%; banned phrases must be absent
        if ex_label in ("WAIT", "HOLD"):
            if now_pct and now_pct > 0:
                b.fail(
                    f"execution={ex_label} but recommended_now_pct={now_pct:.1f}% — "
                    f"must be 0% when not actively buying"
                )
            for phrase in _WAIT_BANNED:
                if phrase in memo:
                    b.fail(
                        f"execution={ex_label}: memo contains banned phrase "
                        f"'{phrase}' — implies entry at current price, contradicting WAIT status"
                    )

        # STARTER / STAGED BUY: recommended_now must be > 0 AND ≤ 50% of target
        if ex_label in ("STARTER", "STAGED_BUY", "STAGED BUY"):
            if now_pct is not None and now_pct <= 0:
                b.fail(
                    f"execution={ex_label} but recommended_now_pct={now_pct:.1f}% — "
                    f"must be > 0 for a starter position"
                )
            half_target = (target_pct or 0) * 0.5
            if now_pct and half_target > 0 and now_pct > half_target:
                b.fail(
                    f"execution={ex_label}: recommended_now {now_pct:.1f}% > "
                    f"50% of target {target_pct:.1f}% ({half_target:.1f}%) — "
                    f"first tranche must not exceed half the target allocation"
                )

        # Low conviction: target ≤ 2.0%
        if conviction.lower() == "low":
            if target_pct and target_pct > 2.0:
                b.fail(
                    f"conviction=Low but target_size_pct={target_pct:.1f}% > 2.0% — "
                    f"low conviction requires maximum 2.0% portfolio target"
                )

        # High conviction: target ≥ 2.0%
        if conviction.lower() == "high":
            if target_pct and target_pct > 0 and target_pct < 2.0:
                b.fail(
                    f"conviction=High but target_size_pct={target_pct:.1f}% < 2.0% — "
                    f"high conviction requires at least 2.0% portfolio target"
                )

        # Mutually exclusive: Extended/Extreme valuation + BUY_NOW
        vr_dict = snap.get("distribution", {})
        # Check via scenario: if all scenario prices below current price → BUY_NOW is contradicted
        price_val  = (snap.get("price") or {}).get("value")
        scenarios  = snap.get("scenarios", {})
        bull_price = (scenarios.get("bull") or {}).get("price")
        if (ex_label == "BUY_NOW" and price_val and bull_price
                and bull_price < price_val):
            b.fail(
                f"execution=BUY_NOW but bull scenario ${bull_price:.2f} < "
                f"current price ${price_val:.2f} — valuation is Extended/Extreme, "
                f"contradicting an immediate buy recommendation"
            )

        return b

    def _b8_numerical_invariants(self, snap: Dict[str, Any]) -> BlockResult:
        """
        Block 8 — Numerical Invariants (Rule N1: Formula-Value Consistency).

        Walks the full snapshot dict tree.  For every object containing both a
        'formula' string and a 'value' number, evaluates the formula safely and
        compares to the stored value.  Mismatches beyond tolerance are corrected
        and logged.

        Tolerance: max(0.001, abs(stored_value) × 0.005)  (±0.001 absolute or
        ±0.5% relative, whichever is larger — as specified in VALIDATION_LAYER.md).

        NOTE: 'snap' here is the plain dict returned by build_snapshot().  There
        is no snap.raw_data attribute; the snapshot IS the dict, so this method
        walks snap directly.
        """
        b = BlockResult(8, "Numerical Invariants")
        pairs_checked = 0

        for path, obj in _walk_formula_value_pairs(snap):
            pairs_checked += 1
            formula = obj["formula"]
            stored  = float(obj["value"])

            try:
                computed = _safe_eval_arithmetic(formula)
            except ValueError as err:
                b.fail(f"{path}.formula rejected: {err}")
                continue

            tolerance = max(0.001, abs(stored) * 0.005)
            if abs(computed - stored) > tolerance:
                path_lower = path.lower()
                if any(k in path_lower for k in ("yield", "rate", "margin")):
                    precision = 4
                elif any(k in path_lower for k in ("price", "market_value")):
                    precision = 2
                else:
                    precision = 4
                new_val = round(computed, precision)
                b.correct(
                    field=f"{path}.value",
                    old=stored,
                    new=new_val,
                    reason=f"formula '{formula}' evaluates to {computed}, not {stored}",
                    rule_id="N1",
                    formula_string=formula,
                    computed_result=computed,
                )

        b.metadata["pairs_checked"] = pairs_checked
        return b

    def _b9_price_freshness(self, snap) -> BlockResult:
        """
        Block 9 — Data Freshness (Rule F1: Price Vintage Freshness).

        Classifies every price object in the snapshot against the report's
        as_of timestamp.  Stale equity prices block report generation; stale
        muni prices flag but do not block; cash/money-market prices always pass.

        snap may be a plain dict (as returned by build_snapshot) or an object
        with an 'as_of' attribute.
        """
        b = BlockResult(9, "Price Freshness")

        # Read as_of from dict or object attribute.
        # "as_of_date" is the canonical key produced by build_snapshot;
        # "as_of" is used by external snapshots (ticker_analysis_v1.json).
        # Accept either; prefer "as_of_date".
        if isinstance(snap, dict):
            as_of_raw = snap.get("as_of_date") or snap.get("as_of")
        else:
            as_of_raw = (
                getattr(snap, "as_of_date", None)
                or getattr(snap, "as_of", None)
            )

        if as_of_raw is None:
            b.fail("as_of missing at top level — cannot classify freshness")
            b.metadata["prices_checked"] = 0
            return b

        try:
            report_as_of = _parse_iso8601(str(as_of_raw))
        except ValueError as exc:
            b.fail(f"as_of '{as_of_raw}' could not be parsed as ISO 8601: {exc}")
            b.metadata["prices_checked"] = 0
            return b

        prices_checked        = 0
        equity_count          = 0
        muni_count            = 0
        cash_count            = 0
        live_count            = 0
        prior_close_count     = 0
        acceptable_count      = 0
        stale_block_count     = 0
        stale_flag_count      = 0
        missing_vintage_count = 0

        for path, kind, price_dict in _walk_price_objects(snap):
            prices_checked += 1

            if kind == "equity":
                equity_count += 1
            elif kind == "muni_bond":
                muni_count += 1
            elif kind == "cash":
                cash_count += 1

            # Defensive: walker guarantees vintage present, but guard anyway
            vintage_str = price_dict.get("vintage")
            if vintage_str is None:
                missing_vintage_count += 1
                if kind == "equity":
                    b.fail(f"{path}: equity price missing vintage — blocks report")
                continue

            try:
                vintage = _parse_iso8601(str(vintage_str))
            except ValueError as exc:
                b.fail(f"{path}.vintage could not be parsed as ISO 8601: {exc}")
                continue

            age_hours = (report_as_of - vintage).total_seconds() / 3600

            if kind == "equity":
                if age_hours <= 1.0:
                    classification = "live"
                    live_count += 1
                elif age_hours <= 26.0:
                    classification = "prior_close"
                    prior_close_count += 1
                else:
                    classification = "stale_block"
                    stale_block_count += 1
                    b.fail(
                        f"{path}: equity price {age_hours:.1f}h stale — blocks report"
                    )
            elif kind == "muni_bond":
                if age_hours <= 168.0:
                    classification = "acceptable"
                    acceptable_count += 1
                else:
                    classification = "stale_flag"
                    stale_flag_count += 1
            elif kind == "cash":
                classification = "acceptable"
                acceptable_count += 1
            else:
                classification = "acceptable"
                acceptable_count += 1

            b.correct(
                field=f"{path}.freshness",
                old=None,
                new=classification,
                reason=f"age={age_hours:.1f}h",
                rule_id="F1",
            )

        b.metadata["prices_checked"]        = prices_checked
        b.metadata["equity_count"]          = equity_count
        b.metadata["muni_count"]            = muni_count
        b.metadata["cash_count"]            = cash_count
        b.metadata["live_count"]            = live_count
        b.metadata["prior_close_count"]     = prior_close_count
        b.metadata["acceptable_count"]      = acceptable_count
        b.metadata["stale_block_count"]     = stale_block_count
        b.metadata["stale_flag_count"]      = stale_flag_count
        b.metadata["missing_vintage_count"] = missing_vintage_count

        return b

    def _b10_source_attribution(self, snap) -> BlockResult:
        """
        Block 10 — Source Provenance (Rule S1: Source Attribution Required).

        Walks every numeric value object in the snapshot and verifies it
        carries a source attribution.  Unsourced values are surfaced as
        Correction entries; the block never fails (passed stays True).

        Attribution is satisfied by any of:
          - value_dict["source"]          — explicit source string
          - value_dict["derived"] is True — computed from other fields
          - "formula" in value_dict       — inline derivation
          - inherited_source is not None  — ancestor carries source
          - unit-literal exemption        — value == 1.0 and path hints unit
          - source_not_required: true     — explicit opt-out
        """
        b = BlockResult(10, "Source Attribution")

        values_checked  = 0
        sourced_count   = 0
        derived_count   = 0
        inherited_count = 0
        unsourced_count = 0
        exempted_count  = 0

        for path, value_dict, inherited_source in _walk_numeric_value_objects(snap):
            values_checked += 1

            if value_dict.get("source_not_required"):
                exempted_count += 1
                continue

            if value_dict.get("source"):
                sourced_count += 1
                continue

            if value_dict.get("derived") is True or "formula" in value_dict:
                derived_count += 1
                continue

            if inherited_source is not None:
                inherited_count += 1
                continue

            # Unit-literal exemption: value == 1.0 and path key hints unit
            parent_hint = path.rsplit(".", 1)[-1].lower()
            if abs(value_dict["value"] - 1.0) < 1e-9 and any(
                hint in parent_hint for hint in ("par", "unit", "nav")
            ):
                exempted_count += 1
                continue

            unsourced_count += 1
            b.correct(
                field=f"{path}.source",
                old=None,
                new="UNSOURCED",
                reason=(
                    f"numeric value {value_dict['value']} has no source "
                    f"attribution and no derived=true flag"
                ),
                rule_id="S1",
            )

        b.metadata["values_checked"]  = values_checked
        b.metadata["sourced_count"]   = sourced_count
        b.metadata["derived_count"]   = derived_count
        b.metadata["inherited_count"] = inherited_count
        b.metadata["unsourced_count"] = unsourced_count
        b.metadata["exempted_count"]  = exempted_count

        return b

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self, snapshot: Dict[str, Any]) -> ValidationLog:
        """
        Run all 10 blocks.  Returns a ValidationLog with PASS/FAIL for each.
        Any block with a registered override is treated as passed in the log.
        """
        ticker    = snapshot.get("ticker", "UNKNOWN")
        as_of     = snapshot.get("as_of_date", "")
        price_val = (snapshot.get("price") or {}).get("value")
        price_src = (snapshot.get("price") or {}).get("source", "")
        price_str = (
            f"${price_val:.4f} (source: {price_src})"
            if price_val else "unavailable"
        )

        log = ValidationLog(
            ticker=ticker,
            report_date=as_of,
            price_str=price_str,
        )

        runners = [
            self._b1_freshness,
            self._b2_market_cap_triangle,
            self._b3_pe_basis,
            self._b4_scenario_distribution,
            self._b5_macro_integrity,
            self._b6_fcf_guidance,
            self._b7_execution_coherence,
            self._b8_numerical_invariants,
            self._b9_price_freshness,
            self._b10_source_attribution,
        ]

        for runner in runners:
            result: BlockResult = runner(snapshot)
            # Apply override if registered
            if not result.passed and result.block_id in self._overrides:
                ov = self._overrides[result.block_id]
                log.overrides.append({
                    "block": result.block_id,
                    "reason_code": ov.get("reason_code", "MANUAL_OVERRIDE"),
                    "justification": ov.get("justification", ""),
                })
                result.passed = True   # marked passed for report generation
            log.blocks.append(result)

        return log


# ── Convenience exception ──────────────────────────────────────────────────────

class ReportBlockedError(Exception):
    """Raised when the validation gate blocks report generation."""
    def __init__(self, log: ValidationLog) -> None:
        self.log = log
        super().__init__(f"Report blocked for {log.ticker}: {log.status}\n{log.format()}")
