"""
tests/test_api_macro_flow.py

Tests for the macro data-flow from MacroLEIAgent payload
through web_api._extract_macro() to the API response dict.

These tests run under the .venv (which has fastapi installed).
Run with:
    .venv/bin/python -m pytest tests/test_api_macro_flow.py -v

They do NOT start the FastAPI server — they import _extract_macro
directly and feed it a mock state object that mirrors the real
OrchestratorAgent state structure.
"""
from __future__ import annotations

import pytest
from web_api import _extract_macro


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_state(macro_payload: dict | None):
    """Mock OrchestratorAgent state with agent_findings["macro"]."""
    class _State:
        agent_findings = {"macro": macro_payload or {}}
    return _State()


def _full_payload(**overrides) -> dict:
    """
    Full MacroLEIAgent payload — mirrors what MacroLEIAgent.process_message()
    puts into state.agent_findings["macro"].
    """
    base = {
        "macro_score":            61.4,
        "macro_regime":           "Recovery",
        "recession_risk_level":   "Moderate",
        "confidence_modifier":    -0.02,
        "sector_tilt":            "Small-caps, Cyclicals, Real Estate",
        "reasoning_summary":      "Macro score 61/100 → Recovery (early). Recession risk: Moderate.",
        "bullish_macro_factors":  ["Yield curve modestly positive (0.50pp)"],
        "bearish_macro_factors":  ["OECD CLI below trend (99.85) — growth momentum fading"],
        "data_coverage":          1.0,
        # Phase 1 LEI fields
        "cycle_phase":            "early",
        "lei_trend":              "rising",
        "yield_spread_trend":     "falling",
        "snapshot":               {"yield_spread_10y2y": 0.5},
        "observation_dates":      {"oecd_cli": "2024-01-01", "yield_spread_10y2y": "2026-04-08"},
    }
    base.update(overrides)
    return base


# ── Group 16: _extract_macro() serialisation ──────────────────────────────────

class TestExtractMacroFields:
    """All Phase 1 LEI fields must survive _extract_macro() unchanged."""

    def test_available_true_when_payload_present(self):
        result = _extract_macro(_make_state(_full_payload()))
        assert result["available"] is True

    def test_core_fields_pass_through(self):
        result = _extract_macro(_make_state(_full_payload()))
        assert result["macro_regime"]         == "Recovery"
        assert result["macro_score"]          == 61.4
        assert result["recession_risk_level"] == "Moderate"
        assert result["sector_tilt"]          == "Small-caps, Cyclicals, Real Estate"
        assert result["bullish_macro_factors"] == ["Yield curve modestly positive (0.50pp)"]
        assert result["bearish_macro_factors"] == ["OECD CLI below trend (99.85) — growth momentum fading"]

    def test_cycle_phase_passes_through(self):
        result = _extract_macro(_make_state(_full_payload()))
        assert result["cycle_phase"] == "early", (
            f"cycle_phase stripped by _extract_macro — got {result.get('cycle_phase')!r}"
        )

    def test_lei_trend_passes_through(self):
        result = _extract_macro(_make_state(_full_payload()))
        assert result["lei_trend"] == "rising", (
            f"lei_trend stripped by _extract_macro — got {result.get('lei_trend')!r}"
        )

    def test_yield_spread_trend_passes_through(self):
        result = _extract_macro(_make_state(_full_payload()))
        assert result["yield_spread_trend"] == "falling", (
            f"yield_spread_trend stripped — got {result.get('yield_spread_trend')!r}"
        )

    def test_reasoning_summary_passes_through(self):
        result = _extract_macro(_make_state(_full_payload()))
        assert "Macro score" in result.get("reasoning_summary", ""), (
            f"reasoning_summary missing or empty: {result.get('reasoning_summary')!r}"
        )

    def test_observation_dates_passes_through(self):
        result = _extract_macro(_make_state(_full_payload()))
        obs = result.get("observation_dates", {})
        assert obs.get("oecd_cli") == "2024-01-01"
        assert obs.get("yield_spread_10y2y") == "2026-04-08"

    def test_data_coverage_passes_through(self):
        result = _extract_macro(_make_state(_full_payload()))
        assert result["data_coverage"] == 1.0

    def test_all_required_keys_present(self):
        """Every key the frontend MacroData interface expects must be present."""
        required = {
            "available", "macro_regime", "macro_score", "recession_risk_level",
            "sector_tilt", "bullish_macro_factors", "bearish_macro_factors",
            "data_coverage", "cycle_phase", "lei_trend", "yield_spread_trend",
            "reasoning_summary", "observation_dates",
        }
        result = _extract_macro(_make_state(_full_payload()))
        missing = required - set(result.keys())
        assert not missing, f"Keys missing from _extract_macro output: {missing}"


class TestExtractMacroEdgeCases:
    """Backward compatibility and missing-data safety."""

    def test_available_false_when_no_payload(self):
        result = _extract_macro(_make_state(None))
        assert result["available"] is False

    def test_available_false_when_empty_dict(self):
        result = _extract_macro(_make_state({}))
        assert result["available"] is False

    def test_cycle_phase_none_when_key_absent(self):
        """Payload from before Phase 1 LEI additions — cycle_phase key missing."""
        old_payload = {k: v for k, v in _full_payload().items()
                       if k not in ("cycle_phase", "lei_trend", "yield_spread_trend")}
        result = _extract_macro(_make_state(old_payload))
        assert result["available"] is True
        assert result["cycle_phase"] is None
        assert result["lei_trend"] is None
        assert result["yield_spread_trend"] is None

    def test_cycle_phase_unknown_passes_through(self):
        """'unknown' phase must reach frontend so it can suppress the badge."""
        result = _extract_macro(_make_state(_full_payload(cycle_phase="unknown")))
        assert result["cycle_phase"] == "unknown"

    def test_no_crash_on_exception(self):
        """_extract_macro must return {"available": False} on any internal error."""
        class _BrokenState:
            @property
            def agent_findings(self):
                raise RuntimeError("simulated state corruption")
        result = _extract_macro(_BrokenState())
        assert result == {"available": False}

    def test_late_cycle_payload(self):
        result = _extract_macro(_make_state(_full_payload(
            macro_regime="Expansion",
            cycle_phase="late",
            lei_trend="falling",
            yield_spread_trend="falling",
        )))
        assert result["cycle_phase"] == "late"
        assert result["lei_trend"]   == "falling"
        assert result["macro_regime"] == "Expansion"

    def test_contraction_payload(self):
        result = _extract_macro(_make_state(_full_payload(
            macro_regime="Contraction",
            cycle_phase="contraction",
            macro_score=28.0,
            recession_risk_level="High",
            lei_trend="falling",
            yield_spread_trend=None,
        )))
        assert result["cycle_phase"]           == "contraction"
        assert result["recession_risk_level"]  == "High"
        assert result["yield_spread_trend"]    is None   # None must survive, not be dropped
