"""
SentimentAnalysisAgent

V1: Clean scaffold with clear extension points.
Returns neutral sentiment with data source notes.

Extension points for V2+:
─────────────────────────
  1. Finnhub news sentiment
     GET https://finnhub.io/api/v1/news-sentiment?symbol={ticker}&token={key}

  2. Finnhub social sentiment (premium)
     GET https://finnhub.io/api/v1/stock/social-sentiment?symbol={ticker}

  3. SEC EDGAR 8-K filings (material events)
     GET https://data.sec.gov/submissions/CIK{padded_cik}.json

  4. NLP on earnings call transcripts

  5. Options flow (unusual activity) — put/call ratios
"""
from __future__ import annotations

from agents.base_agent import BaseAgent
from models.message import AgentMessage, MessageType


class SentimentAnalysisAgent(BaseAgent):
    name = "sentiment_analysis_agent"

    def process_message(self, message: AgentMessage) -> AgentMessage:
        if message.message_type != MessageType.ANALYSIS_REQUEST:
            return self._error_response(message, "Expected ANALYSIS_REQUEST")

        ticker           = message.ticker
        sentiment_result = self._run_v1_scaffold(ticker)

        return self._reply(
            message,
            MessageType.ANALYSIS_RESPONSE,
            payload={
                "analysis_type": "sentiment",
                "findings":       sentiment_result,
            },
            confidence=0.3,
            reasoning_summary=(
                "Sentiment module is scaffolded for V1. "
                "Integrate Finnhub/EDGAR in V2 for real signal."
            ),
        )

    def _run_v1_scaffold(self, ticker: str) -> dict:
        return {
            "ticker":            ticker,
            "overall_sentiment": "neutral",
            "sentiment_score":   0.0,
            "news_sentiment":    None,
            "social_sentiment":  None,
            "sec_8k_recent":     None,
            "earnings_tone":     None,
            "put_call_ratio":    None,
            "data_source":       "v1_scaffold",
            "note": (
                "Sentiment analysis requires Finnhub and/or EDGAR integration. "
                "See agents/sentiment_analysis_agent.py for extension points."
            ),
        }

    # ── V2 extension stubs ─────────────────────────────────────────────────────

    def _fetch_finnhub_news_sentiment(self, ticker: str) -> dict:
        raise NotImplementedError(
            "Finnhub sentiment not yet implemented — see V2 roadmap"
        )

    def _fetch_sec_8k_filings(self, cik: str) -> list[dict]:
        raise NotImplementedError(
            "EDGAR 8-K fetch not yet implemented — see V2 roadmap"
        )
