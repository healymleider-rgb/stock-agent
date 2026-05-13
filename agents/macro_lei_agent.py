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

from datetime import date, datetime

from analysis.macro_overlay import score as overlay_score
from agents.base_agent import BaseAgent
from api.fred_provider import FREDProvider
from models.message import AgentMessage, MessageType
from utils.logger import logger

_STALENESS_WARN_DAYS = 60  # warn if any indicator observation is older than this


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

        print(f"  [MacroLEI] Fetching LEI snapshot from FRED ...")
        snapshot = self._fred.get_lei_snapshot()

        # Log what we got — include observation date so staleness is visible
        _snap_obs = snapshot.get("_observation_dates", {})
        data_keys = [k for k in snapshot if not k.startswith("_") and "_trend" not in k]
        available = sum(1 for k in data_keys if snapshot[k] is not None)
        total     = len(data_keys)
        print(f"  [MacroLEI] FRED snapshot — {available}/{total} indicators available:")
        for k in data_keys:
            v    = snapshot[k]
            d    = _snap_obs.get(k, "?")
            vstr = f"{v:.4f}" if isinstance(v, float) else "N/A"
            print(f"    {k}: {vstr}  (obs date: {d})")

        # Warn if any indicator observation is stale (single combined message)
        stale_indicators: list[str] = []
        today = date.today()
        for k, obs_date_str in _snap_obs.items():
            if obs_date_str is None or snapshot.get(k) is None:
                continue
            try:
                obs_dt   = datetime.strptime(obs_date_str, "%Y-%m-%d").date()
                days_old = (today - obs_dt).days
                if days_old > _STALENESS_WARN_DAYS:
                    stale_indicators.append(f"{k} ({days_old}d old)")
            except (ValueError, TypeError):
                pass
        if stale_indicators:
            stale_msg = "Macro indicators stale: " + ", ".join(stale_indicators)
            logger.warning("MacroLEIAgent: %s", stale_msg)
            print(f"  [MacroLEI] *** {stale_msg} ***")

        # Score using the rule-based overlay — _observation_dates must still be in snapshot
        assessment = overlay_score(snapshot)

        # Pop observation dates AFTER scoring so score() could read them for staleness
        obs_dates: dict = snapshot.pop("_observation_dates", {})

        print(
            f"  [MacroLEI] Regime={assessment.macro_regime}  "
            f"CyclePhase={assessment.cycle_phase}  "
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
            # Phase 1 LEI additions
            "cycle_phase":            assessment.cycle_phase,
            "lei_trend":              assessment.lei_trend,
            "yield_spread_trend":     assessment.yield_spread_trend,
            # Traceability
            "confidence_adjustment_rationale": assessment.confidence_adjustment_rationale,
            # Actual level values for display
            "oecd_cli_level":         snapshot.get("oecd_cli"),
            "mfg_prod_level":         snapshot.get("mfg_prod"),
            "yield_curve_level":      snapshot.get("yield_spread_10y2y"),
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
            "sector_tilt":            "",
            "reasoning_summary":      "Macro overlay not available — macro signals are inconclusive.",
            "bullish_macro_factors":  [],
            "bearish_macro_factors":  [],
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
