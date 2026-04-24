"""
Tests for Rule N1 — Block 8: Numerical Invariants.

Covers:
  - _safe_eval_arithmetic (unit)
  - _walk_formula_value_pairs (unit)
  - ValidationGate._b8_numerical_invariants (integration via direct call)
"""
import math
import sys
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.validation_gate import (
    ValidationGate,
    _safe_eval_arithmetic,
    _walk_formula_value_pairs,
    _parse_iso8601,
    _walk_price_objects,
    _walk_numeric_value_objects,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_test_snapshot() -> dict:
    """
    Build a minimal but structurally complete snapshot via the real
    ValidationGate.build_snapshot(), using SimpleNamespace mocks for all
    analysis objects.  Reused by the build_snapshot integration tests.
    """
    norm_metrics = SimpleNamespace(
        price=100.0,          price_source="FMP",
        shares=1_000_000_000, shares_source="10-Q",
        market_cap_api=100_000_000_000, market_cap_recomp=None, price_adjusted=False,
        ttm_eps=2.50,         ttm_eps_source="10-Q_TTM",
        annual_eps=2.40,      annual_eps_source="10-K",
        pe_ratio=40.0,        pe_source="TTM",
        ps_ratio=5.0,         ev_ebitda=20.0, eps_growth_pct=None,
    )
    stock_data = SimpleNamespace(
        income_statements=[SimpleNamespace(period_of_report="2026-01-31")],
        quarterly_income=[SimpleNamespace(eps_diluted=0.60)] * 4,
    )
    macro_findings = {
        "macro_regime": "Expansion",
        "macro_score": 69,
        "lei_snapshot": {
            "cli": 100.5, "jobless_claims": 207,
            "housing_starts": 1487, "manuf_employ": 12591, "yield_spread": 0.52,
        },
    }
    scorecard = SimpleNamespace(stance=SimpleNamespace(value="bullish"))

    gate = ValidationGate()
    return gate.build_snapshot(
        ticker="TEST",
        as_of_date="2026-04-22T10:30:00Z",
        norm_metrics=norm_metrics,
        stock_data=stock_data,
        valuation_range=None,
        macro_findings=macro_findings,
        scorecard=scorecard,
    )


def _run_b8(snap: dict):
    """Call _b8_numerical_invariants directly on a plain dict snapshot."""
    gate = ValidationGate()
    return gate._b8_numerical_invariants(snap)


# ── 1. safe_eval: basic arithmetic ───────────────────────────────────────────

def test_safe_eval_simple_arithmetic():
    assert _safe_eval_arithmetic("2 + 2") == 4.0
    assert _safe_eval_arithmetic("10 - 3") == 7.0
    assert _safe_eval_arithmetic("3 * 4") == 12.0
    assert _safe_eval_arithmetic("15 / 3") == 5.0
    result = _safe_eval_arithmetic("0.0481 / (1 - 0.37 - 0.133)")
    assert math.isclose(result, 0.0968, abs_tol=1e-4), f"got {result}"


# ── 2. safe_eval: rejects malicious input ─────────────────────────────────────

def test_safe_eval_rejects_malicious(capsys):
    malicious = [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "globals()",
        "lambda x: x",
    ]
    for expr in malicious:
        with pytest.raises((ValueError, Exception)):
            _safe_eval_arithmetic(expr)

    captured = capsys.readouterr()
    assert "pwned" not in captured.out
    assert "pwned" not in captured.err


# ── 3. walker: finds deeply nested pair ──────────────────────────────────────

def test_walker_finds_nested_pairs():
    data = {
        "level1": {
            "level2": {
                "level3": {
                    "level4": {"formula": "1 + 1", "value": 2.0}
                }
            }
        }
    }
    pairs = list(_walk_formula_value_pairs(data))
    assert len(pairs) == 1
    path, obj = pairs[0]
    assert path == "level1.level2.level3.level4"
    assert obj["formula"] == "1 + 1"
    assert obj["value"] == 2.0


# ── 4. walker: ignores incomplete pairs ──────────────────────────────────────

def test_walker_ignores_incomplete_pairs():
    data = {
        "only_formula": {"formula": "1 + 1"},
        "only_value":   {"value": 2.0},
        "neither":      {"x": 99},
    }
    pairs = list(_walk_formula_value_pairs(data))
    assert len(pairs) == 0


# ── 5. walker: list index notation ───────────────────────────────────────────

def test_walker_handles_list_indices():
    data = {
        "items": [
            {"formula": "1 + 0", "value": 1.0},
            {"formula": "2 + 0", "value": 2.0},
            {"formula": "3 + 0", "value": 3.0},
        ]
    }
    pairs = list(_walk_formula_value_pairs(data))
    assert len(pairs) == 3
    paths = [p for p, _ in pairs]
    assert paths == ["items[0]", "items[1]", "items[2]"]


# ── 6. rule N1: catches TEY mismatch ─────────────────────────────────────────

def test_rule_n1_catches_tey_mismatch():
    # The formula evaluates to ~0.0968, but value is stored as 0.0836
    snap = {
        "taxable_equivalent_yield": {
            "formula": "0.0481 / (1 - 0.37 - 0.133)",
            "value": 0.0836,
        }
    }
    result = _run_b8(snap)

    assert len(result.corrections) == 1
    c = result.corrections[0]
    assert c.rule_id == "N1"
    assert c.old_value == 0.0836
    assert math.isclose(c.new_value, 0.0968, abs_tol=1e-4), f"new_value={c.new_value}"
    assert c.formula_string == "0.0481 / (1 - 0.37 - 0.133)"
    assert math.isclose(c.computed_result, 0.0968, abs_tol=1e-4)


# ── 7. rule N1: passes when formula matches ───────────────────────────────────

def test_rule_n1_passes_when_formula_matches():
    computed = _safe_eval_arithmetic("0.0481 / (1 - 0.37 - 0.133)")
    snap = {
        "taxable_equivalent_yield": {
            "formula": "0.0481 / (1 - 0.37 - 0.133)",
            "value": computed,       # exactly what the formula evaluates to
        }
    }
    result = _run_b8(snap)

    assert result.passed is True
    assert len(result.corrections) == 0


# ── 8. rule N1: multiple violations ──────────────────────────────────────────

def test_rule_n1_multiple_violations():
    snap = {
        "holding_a": {
            "taxable_equivalent_yield": {
                "formula": "0.05 / (1 - 0.37 - 0.133)",
                "value": 0.01,          # wrong
            }
        },
        "holding_b": {
            "some_yield": {
                "formula": "0.04 / (1 - 0.37 - 0.133)",
                "value": 0.01,          # wrong
            }
        },
    }
    result = _run_b8(snap)

    assert len(result.corrections) == 2
    paths = {c.field for c in result.corrections}
    assert "holding_a.taxable_equivalent_yield.value" in paths
    assert "holding_b.some_yield.value" in paths


# ── 9. rule N1: vacuous pass reports pairs_checked=0 ─────────────────────────

def test_rule_n1_vacuous_pass_reports_pairs_checked_zero():
    snap = {
        "ticker": "NFLX",
        "price": {"value": 92.58, "source": "FMP"},   # no formula — not caught
        "multiples": {"pe": {"value": 29.9}},
    }
    result = _run_b8(snap)

    assert result.passed is True
    assert result.metadata["pairs_checked"] == 0


# ── 10. rule N1: active pass reports correct count ────────────────────────────

def test_rule_n1_active_pass_reports_count():
    v1 = _safe_eval_arithmetic("1 + 1")
    v2 = _safe_eval_arithmetic("2 * 3")
    v3 = _safe_eval_arithmetic("10 / 4")
    snap = {
        "a": {"formula": "1 + 1",  "value": v1},
        "b": {"formula": "2 * 3",  "value": v2},
        "c": {"formula": "10 / 4", "value": v3},
    }
    result = _run_b8(snap)

    assert result.passed is True
    assert len(result.corrections) == 0
    assert result.metadata["pairs_checked"] == 3


# ── Rule F1: Price Vintage Freshness ─────────────────────────────────────────

def _price_snap(kind, vintage_dt, now=None):
    """Build a minimal snap dict with one price object."""
    if now is None:
        now = datetime.now(timezone.utc)
    return {
        "as_of": now.isoformat(),
        "asset": {
            "kind": kind,
            "price": {
                "value": 100.0,
                "vintage": vintage_dt.isoformat(),
            },
        },
    }


# ── 11. F1: fresh equity (< 1h) passes as "live" ─────────────────────────────

def test_f1_fresh_equity_passes():
    now = datetime.now(timezone.utc)
    snap = _price_snap("equity", now - timedelta(minutes=30), now)
    result = ValidationGate()._b9_price_freshness(snap)

    assert result.passed
    assert len(result.failures) == 0
    assert any(c.new_value == "live" for c in result.corrections)


# ── 12. F1: equity 18h old passes as "prior_close" ───────────────────────────

def test_f1_prior_close_equity_passes():
    now = datetime.now(timezone.utc)
    snap = _price_snap("equity", now - timedelta(hours=18), now)
    result = ValidationGate()._b9_price_freshness(snap)

    assert result.passed
    assert any(c.new_value == "prior_close" for c in result.corrections)


# ── 13. F1: equity 30h old blocks as "stale_block" ───────────────────────────

def test_f1_stale_equity_blocks():
    now = datetime.now(timezone.utc)
    snap = _price_snap("equity", now - timedelta(hours=30), now)
    result = ValidationGate()._b9_price_freshness(snap)

    assert not result.passed
    assert len(result.failures) == 1
    assert any(c.new_value == "stale_block" for c in result.corrections)


# ── 14. F1: muni within 7 days passes as "acceptable" ────────────────────────

def test_f1_muni_within_7d_passes():
    now = datetime.now(timezone.utc)
    snap = _price_snap("muni_bond", now - timedelta(hours=48), now)
    result = ValidationGate()._b9_price_freshness(snap)

    assert result.passed
    assert any(c.new_value == "acceptable" for c in result.corrections)


# ── 15. F1: muni over 7 days flags but does not block ────────────────────────

def test_f1_muni_over_7d_flags():
    now = datetime.now(timezone.utc)
    snap = _price_snap("muni_bond", now - timedelta(hours=200), now)
    result = ValidationGate()._b9_price_freshness(snap)

    assert result.passed is True
    assert any(c.new_value == "stale_flag" for c in result.corrections)


# ── 16. F1: cash with any age always passes ───────────────────────────────────

def test_f1_cash_never_flagged():
    now = datetime.now(timezone.utc)
    snap = _price_snap("cash", now - timedelta(days=100), now)
    result = ValidationGate()._b9_price_freshness(snap)

    assert result.passed
    assert len(result.failures) == 0


# ── 17. F1: missing as_of blocks immediately ─────────────────────────────────

def test_f1_missing_as_of_blocks():
    snap = {
        "asset": {
            "kind": "equity",
            "price": {"value": 100.0, "vintage": "2026-01-01T00:00:00+00:00"},
        }
    }
    result = ValidationGate()._b9_price_freshness(snap)

    assert not result.passed
    assert any("as_of" in msg for msg in result.failures)


# ── 26. F1: reads "as_of_date" key (build_snapshot output shape) ─────────────

def test_f1_reads_as_of_date_field_from_build_snapshot():
    """Block 9 must not fail when the snapshot uses 'as_of_date' (build_snapshot key)."""
    now = datetime.now(timezone.utc)
    snap = {
        "as_of_date": now.isoformat(),   # ← build_snapshot key
        "asset": {
            "kind": "equity",
            "price": {
                "value": 150.0,
                "vintage": (now - timedelta(minutes=45)).isoformat(),
            },
        },
    }
    result = ValidationGate()._b9_price_freshness(snap)

    assert result.passed, f"Block 9 failed on as_of_date snap: {result.failures}"
    assert result.metadata.get("prices_checked", 0) >= 1
    assert any(c.new_value == "live" for c in result.corrections)


# ── 27. F1: reads "as_of" key (external snapshot shape) ──────────────────────

def test_f1_reads_as_of_field_from_external_snapshot():
    """Block 9 must also accept 'as_of' (used by ticker_analysis_v1.json)."""
    now = datetime.now(timezone.utc)
    snap = {
        "as_of": now.isoformat(),        # ← external / Solo Mode key
        "asset": {
            "kind": "equity",
            "price": {
                "value": 150.0,
                "vintage": (now - timedelta(hours=20)).isoformat(),
            },
        },
    }
    result = ValidationGate()._b9_price_freshness(snap)

    assert result.passed, f"Block 9 failed on as_of snap: {result.failures}"
    assert result.metadata.get("prices_checked", 0) >= 1
    assert any(c.new_value == "prior_close" for c in result.corrections)


# ── Rule S1: Source Attribution Required ─────────────────────────────────────

def _run_b10(snap: dict):
    return ValidationGate()._b10_source_attribution(snap)


# ── 18. S1: explicit source satisfies requirement ─────────────────────────────

def test_s1_sourced_value_passes_clean():
    snap = {"metric": {"value": 42.0, "source": "FRED:X",
                       "vintage": "2026-04-22T00:00:00Z"}}
    result = _run_b10(snap)

    assert result.passed
    assert result.metadata["sourced_count"] == 1
    assert result.metadata["unsourced_count"] == 0
    assert len(result.corrections) == 0


# ── 19. S1: formula field satisfies derived requirement ──────────────────────

def test_s1_derived_with_formula_exempt():
    snap = {"ratio": {"value": 0.5, "formula": "1 / 2"}}
    result = _run_b10(snap)

    assert result.passed
    assert result.metadata["derived_count"] == 1
    assert result.metadata["unsourced_count"] == 0


# ── 20. S1: ancestor source propagates to child ──────────────────────────────

def test_s1_inherited_source_exempt():
    snap = {"bucket": {"source": "S", "inner": {"value": 3.14}}}
    result = _run_b10(snap)

    assert result.passed
    assert result.metadata["inherited_count"] == 1
    assert result.metadata["unsourced_count"] == 0


# ── 21. S1: bare numeric value emits UNSOURCED correction ────────────────────

def test_s1_unsourced_emits_correction():
    snap = {"anon": {"value": 99.9}}
    result = _run_b10(snap)

    assert result.passed                          # block never fails
    assert result.metadata["unsourced_count"] == 1
    assert len(result.corrections) == 1
    c = result.corrections[0]
    assert c.rule_id == "S1"
    assert c.new_value == "UNSOURCED"
    assert c.old_value is None


# ── 22. S1: value == 1.0 under a par/unit/nav key is exempted ────────────────

def test_s1_unit_literal_exempt():
    snap = {"cash": {"par": {"value": 1.0}}}
    result = _run_b10(snap)

    assert result.passed
    assert result.metadata["exempted_count"] >= 1
    assert result.metadata["unsourced_count"] == 0


# ── 23. S1: source_not_required flag opts out ─────────────────────────────────

def test_s1_explicit_exempt_flag():
    snap = {"x": {"value": 0.0, "source_not_required": True,
                  "reason": "placeholder"}}
    result = _run_b10(snap)

    assert result.passed
    assert result.metadata["exempted_count"] == 1
    assert result.metadata["unsourced_count"] == 0


# ── 24. S1: block never fails even when all values are unsourced ──────────────

def test_s1_block_never_fails():
    snap = {"a": {"value": 1}, "b": {"value": 2}, "c": {"value": 3}}
    result = _run_b10(snap)

    assert result.passed is True
    assert len(result.corrections) == 3
    assert result.metadata["unsourced_count"] == 3


# ── 25. S1: counter identity holds across mixed portfolio ─────────────────────

def test_s1_nested_portfolio_coverage():
    snap = {
        "price":      {"value": 185.0, "source": "FMP"},
        "eps":        {"value": 6.43,  "source": "FMP"},
        "pe":         {"value": 28.8,  "formula": "185.0 / 6.43"},
        "forward_pe": {"value": 25.1,  "derived": True},
        "book_value": {"value": 4.0},
        "nav":        {"par": {"value": 1.0}},
        "bucket": {
            "source": "FRED",
            "cli":          {"value": 100.5},
            "yield_spread": {"value": -0.5},
        },
        "mystery": {"value": 0.0, "source_not_required": True,
                    "reason": "placeholder"},
    }
    result = _run_b10(snap)

    m = result.metadata
    assert result.passed
    assert (
        m["values_checked"]
        == m["sourced_count"] + m["derived_count"]
           + m["inherited_count"] + m["unsourced_count"] + m["exempted_count"]
    )


# ── build_snapshot integration tests (Blocks 8 / 9 / 10) ─────────────────────

def test_b9_works_on_real_build_snapshot_output():
    """
    Regression: Block 9 previously read 'as_of' when build_snapshot produces
    'as_of_date', causing a permanent false-fail on every real evaluation.
    This test exercises the real path: build_snapshot → _b9_price_freshness.

    build_snapshot now emits 'vintage' on snap['price'] (current UTC time)
    and 'kind': 'equity' at the top level, so the walker classifies the
    price and prices_checked must be >= 1.
    """
    snap = _build_test_snapshot()

    assert "as_of_date" in snap, "build_snapshot must emit 'as_of_date'"
    assert snap["as_of_date"] == "2026-04-22T10:30:00Z"
    assert snap.get("kind") == "equity", "build_snapshot must emit kind='equity'"
    assert "vintage" in snap["price"], "build_snapshot must emit vintage on price dict"

    gate = ValidationGate()
    result = gate._b9_price_freshness(snap)

    assert result.passed, (
        f"Block 9 failed on real build_snapshot output. Failures: {result.failures}"
    )
    assert "as_of missing" not in " ".join(result.failures)

    # Price was fetched live (vintage ≈ as_of_date), so it must be classified
    # as "live" (age < 1h) or "prior_close" (< 26h), never stale_block.
    prices_checked = result.metadata.get("prices_checked", 0)
    assert prices_checked >= 1, (
        f"Expected at least 1 price classified; got {prices_checked}. "
        "Check that snap['price'] has both 'value' and 'vintage' fields."
    )
    assert result.metadata.get("stale_block_count", 0) == 0
    assert any(
        c.new_value in ("live", "prior_close") for c in result.corrections
    ), f"Expected live or prior_close classification; got: {result.corrections}"


def test_b8_works_on_real_build_snapshot_output():
    """
    Regression: ensure _b8_numerical_invariants runs cleanly on a real
    build_snapshot output.  build_snapshot uses flat numbers (no formula+value
    pairs), so pairs_checked should be 0 and passed should be True.

    If pairs_checked becomes non-zero in the future, a formula field was added
    to build_snapshot — that's intentional and this test should be updated.
    """
    snap = _build_test_snapshot()

    gate = ValidationGate()
    result = gate._b8_numerical_invariants(snap)

    assert result.passed, f"Block 8 failed on build_snapshot output: {result.failures}"
    assert result.metadata.get("pairs_checked") == 0, (
        f"Expected 0 formula/value pairs in build_snapshot output; "
        f"got {result.metadata.get('pairs_checked')}. "
        "Update test if build_snapshot intentionally gained formula fields."
    )


def test_b10_works_on_real_build_snapshot_output():
    """
    Regression: ensure _b10_source_attribution walks a real build_snapshot.
    Block 10 never fails by design (corrections only), so this test documents
    the current baseline unsourced_count.

    Baseline (2026-04-22): unsourced_count=3
      - eps            (snap['eps'] has 'value' key but no top-level source)
      - multiples.ps   (no source on ps dict)
      - multiples.ev_ebitda (no source on ev_ebitda dict)

    If this count changes, review whether new fields are being correctly
    sourced (improvement) or previously-sourced fields lost attribution (regression).
    """
    snap = _build_test_snapshot()

    gate = ValidationGate()
    result = gate._b10_source_attribution(snap)

    assert result.passed, "Block 10 must never fail (by design)"

    m = result.metadata
    # Identity invariant: every walked value is classified exactly once
    assert (
        m["values_checked"]
        == m["sourced_count"] + m["derived_count"]
           + m["inherited_count"] + m["unsourced_count"] + m["exempted_count"]
    )

    # Baseline unsourced fields — document, do not suppress
    unsourced_fields = sorted(c.field for c in result.corrections)
    assert m["unsourced_count"] == 3, (
        f"Baseline unsourced_count changed from 3 to {m['unsourced_count']}. "
        f"Fields: {unsourced_fields}. "
        "Update baseline if this is an intentional improvement or structural change."
    )
    assert unsourced_fields == ["eps.source", "multiples.ev_ebitda.source", "multiples.ps.source"], (
        f"Unsourced field set changed: {unsourced_fields}"
    )
