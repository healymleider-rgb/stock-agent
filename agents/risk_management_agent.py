"""
RiskManagementAgent

Identifies material risks across leverage, earnings quality,
margin pressure, volatility, and valuation extremes.

Extracts the NormalizedMetrics object from prior fundamental findings so
market_cap, pe_ratio, and debt_to_equity are sourced from the same
validated pipeline used by all other report sections.
"""
from __future__ import annotations

from analysis.risk import score_risk
from agents.base_agent import BaseAgent
from config import Config
from models.message import AgentMessage, MessageType
from models.stock_data import StockData


class RiskManagementAgent(BaseAgent):
    name = "risk_management_agent"

    def process_message(self, message: AgentMessage) -> AgentMessage:
        if message.message_type != MessageType.ANALYSIS_REQUEST:
            return self._error_response(message, "Expected ANALYSIS_REQUEST")

        stock_data: StockData = message.payload.get("stock_data")
        if stock_data is None:
            return self._error_response(message, "No stock_data in payload")

        # Extract NormalizedMetrics from prior fundamental findings.
        # These were computed once in FundamentalAnalysisAgent and flow forward
        # via prior_findings so the risk scorer uses the same market_cap, PE,
        # and D/E values as the valuation scorecard and report header.
        prior_findings = message.payload.get("prior_findings", {})
        fund_findings  = prior_findings.get("fundamental", {})
        norm_metrics   = fund_findings.get("normalized_metrics")
        validation     = fund_findings.get("validation")

        if norm_metrics is not None:
            print(
                f"  [RISK AGENT] using NormalizedMetrics:"
                f" mktcap={norm_metrics.market_cap}({norm_metrics.market_cap_source})"
                f" pe={norm_metrics.pe_ratio}({norm_metrics.pe_source})"
                f" de={norm_metrics.debt_to_equity}"
            )
        else:
            print("  [RISK AGENT] NormalizedMetrics not available — falling back to raw stock_data")

        if validation is not None:
            print(
                f"  [RISK AGENT] ValidationResult:"
                f" flags={len(validation.flags)}"
                f" conviction_penalty={validation.conviction_penalty}"
                f" coverage={validation.data_coverage:.0%}"
            )

        weight                 = Config.SCORE_WEIGHTS["risk"]
        risk_score, risk_flags = score_risk(
            stock_data, weight=weight, metrics=norm_metrics, validation=validation
        )

        critical = [
            f for f in risk_flags
            if any(
                kw in f.lower()
                for kw in ["negative fcf", "cannot service", "dangerously", "very high"]
            )
        ]
        elevated = [f for f in risk_flags if f not in critical]
        severity = "critical" if critical else "elevated" if elevated else "normal"

        confidence = (
            0.85 if (stock_data.latest_ratios and stock_data.latest_balance) else 0.45
        )

        return self._reply(
            message,
            MessageType.ANALYSIS_RESPONSE,
            payload={
                "analysis_type":  "risk",
                "risk":            risk_score,
                "risk_flags":      risk_flags,
                "critical_flags":  critical,
                "elevated_flags":  elevated,
                "severity":        severity,
            },
            confidence=confidence,
            reasoning_summary=(
                f"Risk assessment: {severity.upper()}. "
                f"{len(risk_flags)} flag(s) — score {risk_score.score:.0f}/100. "
                f"{risk_score.reasoning}"
            ),
        )
