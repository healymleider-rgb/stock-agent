"""
MarketAnalystAgent

Reviews macroeconomic conditions, sector context,
analyst consensus, and price targets to add external context
to the evaluation.

V1: uses FMP sector performance data + analyst recommendations.
"""
from __future__ import annotations

from models.message import AgentMessage, MessageType
from agents.base_agent import BaseAgent
from models.stock_data import StockData
from utils.helpers import safe_float


class MarketAnalystAgent(BaseAgent):
    name = "market_analyst_agent"

    def process_message(self, message: AgentMessage) -> AgentMessage:
        if message.message_type != MessageType.ANALYSIS_REQUEST:
            return self._error_response(message, "Expected ANALYSIS_REQUEST")

        stock_data: StockData = message.payload.get("stock_data")
        if stock_data is None:
            return self._error_response(message, "No stock_data in payload")

        findings: dict = {}
        commentary: list[str] = []

        # ── Sector performance ─────────────────────────────────────────────────
        sector_findings = self._analyze_sector(stock_data)
        findings["sector"] = sector_findings
        if sector_findings.get("note"):
            commentary.append(sector_findings["note"])

        # ── Analyst consensus ──────────────────────────────────────────────────
        analyst_findings = self._analyze_analyst_consensus(stock_data)
        findings["analyst"] = analyst_findings
        if analyst_findings.get("note"):
            commentary.append(analyst_findings["note"])

        # ── Price target vs current price ─────────────────────────────────────
        pt_findings = self._analyze_price_target(stock_data)
        findings["price_target"] = pt_findings
        if pt_findings.get("note"):
            commentary.append(pt_findings["note"])

        # ── Macro summary (V1: text only; V2 can pull live macro data) ─────────
        findings["macro_note"] = (
            "Macro context not fetched in V1. "
            "Extend MarketAnalystAgent to call FRED, BLS, or a macro API."
        )

        confidence = 0.60 if (
            stock_data.analyst_recommendations or stock_data.sector_performance
        ) else 0.30

        summary = " | ".join(commentary) if commentary else "Market context analyzed"

        return self._reply(
            message,
            MessageType.ANALYSIS_RESPONSE,
            payload={"analysis_type": "market", "findings": findings},
            confidence=confidence,
            reasoning_summary=summary,
        )

    # ── Sub-analyzers ──────────────────────────────────────────────────────────

    def _analyze_sector(self, stock_data: StockData) -> dict:
        if not stock_data.sector_performance or not stock_data.profile:
            return {"note": "Sector performance data unavailable"}
        sector = stock_data.profile.sector
        for row in stock_data.sector_performance:
            if row.get("sector", "").lower() == sector.lower():
                # stable: "changePercentage" (no trailing 's') — v3 used "changesPercentage"
                chg = safe_float(row.get("changePercentage")) or 0.0
                direction = "outperforming" if chg > 1 else "underperforming" if chg < -1 else "flat"
                return {
                    "sector": sector,
                    "sector_change_pct": chg,
                    "note": f"Sector '{sector}' is {direction} ({chg:+.1f}% recent)",
                }
        return {"sector": sector, "note": f"Sector '{sector}' performance not found in data"}

    def _analyze_analyst_consensus(self, stock_data: StockData) -> dict:
        recs = stock_data.analyst_recommendations
        if not recs:
            return {"note": "No analyst recommendations available"}
        latest = recs[0] if recs else {}
        # stable /grades-consensus fields: strongBuy, buy, hold, sell, strongSell
        # (v3 used analystRatingsStrongBuy / analystRatingsBuy / … — now removed)
        strong_buy = int(latest.get("strongBuy", 0) or 0)
        buy = int(latest.get("buy", 0) or 0)
        hold = int(latest.get("hold", 0) or 0)
        sell = int(latest.get("sell", 0) or 0)
        strong_sell = int(latest.get("strongSell", 0) or 0)
        total = strong_buy + buy + hold + sell + strong_sell
        if total == 0:
            return {"note": "Analyst count is zero"}
        bull_pct = (strong_buy + buy) / total * 100
        bear_pct = (sell + strong_sell) / total * 100
        sentiment = "bullish" if bull_pct > 60 else "bearish" if bear_pct > 40 else "mixed"
        return {
            "strong_buy": strong_buy,
            "buy": buy,
            "hold": hold,
            "sell": sell,
            "strong_sell": strong_sell,
            "total_analysts": total,
            "bull_pct": round(bull_pct, 1),
            "bear_pct": round(bear_pct, 1),
            "consensus": sentiment,
            "note": (
                f"Analyst consensus: {sentiment.upper()} "
                f"({bull_pct:.0f}% bullish, {bear_pct:.0f}% bearish, {total} analysts)"
            ),
        }

    def _analyze_price_target(self, stock_data: StockData) -> dict:
        pt = stock_data.price_targets
        price = stock_data.current_price
        if not pt or price is None:
            return {"note": "Price target data unavailable"}
        target = safe_float(pt.get("targetConsensus"))
        high = safe_float(pt.get("targetHigh"))
        low = safe_float(pt.get("targetLow"))
        if target is None:
            return {"note": "No consensus price target"}
        upside = (target - price) / price * 100
        direction = "upside" if upside > 0 else "downside"
        return {
            "current_price": price,
            "consensus_target": target,
            "target_high": high,
            "target_low": low,
            "upside_pct": round(upside, 1),
            "note": (
                f"Analyst consensus target ${target:.2f} implies "
                f"{abs(upside):.1f}% {direction} from ${price:.2f}"
            ),
        }
