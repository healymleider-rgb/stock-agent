"""
MacroLEIAgent — macroeconomic leading indicator assessment.

Fetches macro indicator data from FRED and scores the current economic regime
using analysis.macro_overlay.  Returns a structured payload that the
Orchestrator stores in state.agent_findings["macro"].

Message contract
────────────────
  Receives : ANALYSIS_REQUEST
  Sends    : ANALYSIS_RESPONSE

  Payload returned:
    macro_score           float       0–100
    macro_regime          str         Expansion / Slowdown / Contraction / Recovery
    recession_risk_level  str         Low / Moderate / Elevated / High
    confidence_modifier   float       applied to overall eval confidence
    sector_tilt           str         preferred sectors given current regime
    reasoning_summary     str
    bullish_macro_factors list[str]
    bearish_macro_factors list[str]
    data_coverage         float       0–1, fraction of indicators that had data
    snapshot              dict        raw FRED values for audit / debug

Availability
────────────
  If FRED_API_KEY is not set, the agent returns a neutral assessment
  (score=50, regime="Unknown", recession_risk_level="Unknown") with
  confidence_modifier=0 rather than failing the evaluation.
"""
from __future__ import annotations

from analysis.macro_overlay import score as overlay_score
from agents.base_agent import BaseAgent
from api.fred_provider import FREDProvider
from models.message import AgentMessage, MessageType
from utils.logger import logger


class MacroLEIAgent(BaseAgent):
    name = "macro_lei_agent"

    def __init__(self) -> None:
        self._fred = FREDProvider()

    def process_message(self, message: AgentMessage) -> AgentMessage:
        if message.message_type != MessageType.ANALYSIS_REQUEST:
            return self._error_response(message, "Expected ANALYSIS_REQUEST")

        if not self._fred.is_available():
            logger.warning(
                "MacroLEIAgent: FRED_API_KEY not set — returning neutral macro assessment"
            )
            print(
                "  [MacroLEI] FRED_API_KEY not configured — "
                "macro analysis skipped, neutral defaults applied"
            )
            return self._neutral_response(message)

        # Fetch all tracked indicators in one round-trip bundle
        print(f"  [MacroLEI] Fetching LEI snapshot from FRED ...")
        snapshot = self._fred.get_lei_snapshot()

        # Extract observation dates (injected by FREDProvider.get_lei_snapshot)
        obs_dates: dict = snapshot.pop("_observation_dates", {})

        # Log what we got — include observation date so staleness is visible
        data_keys = [k for k in snapshot if not k.startswith("_")]
        available = sum(1 for k in data_keys if snapshot[k] is not None)
        total     = len(data_keys)
        print(f"  [MacroLEI] FRED snapshot — {available}/{total} indicators available:")
        for k in data_keys:
            v    = snapshot[k]
            d    = obs_dates.get(k, "?")
            vstr = f"{v:.4f}" if isinstance(v, float) else "N/A"
            print(f"    {k}: {vstr}  (obs date: {d})")

        # Score using the rule-based overlay (overlay_score only uses numeric keys)
        assessment = overlay_score(snapshot)

        print(
            f"  [MacroLEI] Regime={assessment.macro_regime}  "
            f"Score={assessment.macro_score:.0f}/100  "
            f"RecessionRisk={assessment.recession_risk_level}  "
            f"ConfMod={assessment.confidence_modifier:+.3f}"
        )

        payload = {
            "macro_score":            assessment.macro_score,
            "macro_regime":           assessment.macro_regime,
            "recession_risk_level":   assessment.recession_risk_level,
            "confidence_modifier":    assessment.confidence_modifier,
            "sector_tilt":            assessment.sector_tilt,
            "reasoning_summary":      assessment.reasoning_summary,
            "bullish_macro_factors":  assessment.bullish_macro_factors,
            "bearish_macro_factors":  assessment.bearish_macro_factors,
            "data_coverage":          assessment.data_coverage,
            "snapshot":               snapshot,
            "observation_dates":      obs_dates,
        }

        confidence = min(0.90, 0.50 + assessment.data_coverage * 0.40)

        return self._reply(
            message,
            MessageType.ANALYSIS_RESPONSE,
            payload=payload,
            confidence=round(confidence, 2),
            reasoning_summary=assessment.reasoning_summary,
        )

    # ── Fallback when FRED is not configured ──────────────────────────────────

    def _neutral_response(self, message: AgentMessage) -> AgentMessage:
        payload = {
            "macro_score":            50.0,
            "macro_regime":           "Unknown",
            "recession_risk_level":   "Unknown",
            "confidence_modifier":    0.0,
            "sector_tilt":            "No tilt — macro data unavailable",
            "reasoning_summary":      "FRED API key not configured; macro analysis unavailable.",
            "bullish_macro_factors":  [],
            "bearish_macro_factors":  ["FRED_API_KEY not set — macro indicators not evaluated"],
            "data_coverage":          0.0,
            "snapshot":               {},
        }
        return self._reply(
            message,
            MessageType.ANALYSIS_RESPONSE,
            payload=payload,
            confidence=0.0,
            reasoning_summary="Macro analysis skipped — FRED_API_KEY not configured.",
        )
