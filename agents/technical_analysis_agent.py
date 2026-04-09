"""
TechnicalAnalysisAgent

Analyzes price behavior using moving averages, RSI, MACD,
and 6/12-month returns. Produces a momentum CategoryScore.

Extracts NormalizedMetrics from prior fundamental findings so the
resolved price (metrics.price) is available to the momentum scorer.
"""
from __future__ import annotations

from analysis.momentum import score_momentum
from agents.base_agent import BaseAgent
from config import Config
from models.message import AgentMessage, MessageType
from models.stock_data import StockData


class TechnicalAnalysisAgent(BaseAgent):
    name = "technical_analysis_agent"

    def process_message(self, message: AgentMessage) -> AgentMessage:
        if message.message_type != MessageType.ANALYSIS_REQUEST:
            return self._error_response(message, "Expected ANALYSIS_REQUEST")

        stock_data: StockData = message.payload.get("stock_data")
        if stock_data is None:
            return self._error_response(message, "No stock_data in payload")

        # Extract NormalizedMetrics from prior fundamental findings.
        # The momentum scorer uses price history (raw, unchanged) but having
        # norm_metrics available lets it use the validated resolved price
        # and keeps the door open for future technical enhancements.
        prior_findings = message.payload.get("prior_findings", {})
        norm_metrics = prior_findings.get("fundamental", {}).get("normalized_metrics")

        weight         = Config.SCORE_WEIGHTS["momentum"]
        momentum_score = score_momentum(stock_data, weight=weight, metrics=norm_metrics)

        confidence = 0.85 if momentum_score.data_quality == "good" else 0.35
        ph         = stock_data.price_history
        days       = len(ph) if ph else 0

        return self._reply(
            message,
            MessageType.ANALYSIS_RESPONSE,
            payload={
                "analysis_type":    "technical",
                "momentum":          momentum_score,
                "price_data_points": days,
            },
            confidence=confidence,
            reasoning_summary=(
                f"Technical analysis complete using {days} days of price data. "
                f"Momentum: {momentum_score.score:.0f}/100 — {momentum_score.reasoning}"
            ),
        )
