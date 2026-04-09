"""
FundamentalAnalysisAgent

Evaluates company financial health using income, balance, and cash flow data.
Runs valuation, growth, profitability, and financial health scorers.

All four scorers receive the same NormalizedMetrics object so every metric
they use is sourced from the same validated, logged pipeline — never from
independent reads of raw stock_data fields.

Confidence is derived from NormalizedMetrics quality (compute_confidence),
not from scorer data_quality flags, so it reflects actual data resolution.
"""
from __future__ import annotations

from analysis.growth import score_growth
from analysis.health import score_financial_health
from analysis.metrics import compute_confidence, compute_core_metrics
from analysis.profitability import score_profitability
from analysis.valuation import score_valuation
from analysis.valuation_range import compute_valuation_range
from agents.base_agent import BaseAgent
from config import Config
from models.message import AgentMessage, MessageType
from models.stock_data import StockData


class FundamentalAnalysisAgent(BaseAgent):
    name = "fundamental_analysis_agent"

    def process_message(self, message: AgentMessage) -> AgentMessage:
        if message.message_type != MessageType.ANALYSIS_REQUEST:
            return self._error_response(message, "Expected ANALYSIS_REQUEST")

        stock_data: StockData = message.payload.get("stock_data")
        if stock_data is None:
            return self._error_response(message, "No stock_data in payload")

        weights = Config.SCORE_WEIGHTS

        # ── Debug: show what financial fields are available ────────────────────
        income  = stock_data.latest_income
        ratios  = stock_data.latest_ratios
        print(
            f"  [FUND DEBUG] income_statements={len(stock_data.income_statements)}"
            f"  ratios={'present' if ratios else 'None'}"
            f"  price={stock_data.current_price}"
            f"  market_cap={stock_data.market_cap}"
            f"  quarterly={len(stock_data.quarterly_income)}"
        )
        if income:
            print(
                f"  [FUND DEBUG] net_income={income.net_income}"
                f"  revenue={income.revenue}"
                f"  ebitda={income.ebitda}"
                f"  gross_profit={income.gross_profit}"
                f"  operating_income={income.operating_income}"
            )
        else:
            print("  [FUND DEBUG] income=None — all scorers will use neutral fallbacks")

        # ── Compute NormalizedMetrics ONCE — single source of truth ───────────
        # All four scorers below consume this object exclusively.
        # No scorer is allowed to re-derive PE, market cap, margins, etc.
        norm_metrics = compute_core_metrics(stock_data)

        # ── Run all four fundamental scorers — all fed normalized metrics ──────
        val_score = score_valuation(
            stock_data, weight=weights["valuation"], metrics=norm_metrics
        )
        gro_score = score_growth(
            stock_data, weight=weights["growth"], metrics=norm_metrics
        )
        pro_score = score_profitability(
            stock_data, weight=weights["profitability"], metrics=norm_metrics
        )
        hlt_score = score_financial_health(
            stock_data, weight=weights["financial_health"], metrics=norm_metrics
        )

        # ── Confidence from NormalizedMetrics quality ─────────────────────────
        # compute_confidence() measures what fraction of key metrics resolved
        # and applies a penalty for any provider-vs-computed divergence warnings.
        confidence = compute_confidence(norm_metrics)

        # ── Debug: log each scorer's output and data_quality ──────────────────
        for s in [val_score, gro_score, pro_score, hlt_score]:
            print(
                f"  [FUND SCORE] {s.name}: score={s.score:.1f}/100"
                f" quality={s.data_quality}"
                f" weight={s.weight}"
            )
            for f in s.factors[:3]:
                print(f"    factor: {f}")

        # ── Reasoning narrative from top signals ───────────────────────────────
        scores = [val_score, gro_score, pro_score, hlt_score]
        top_scores = sorted(scores, key=lambda s: abs(s.score - 50), reverse=True)
        summary = "; ".join(
            f"{s.name.replace('_', ' ')} {s.score:.0f}/100 ({s.reasoning})"
            for s in top_scores[:2]
        )

        # ── Valuation range (scenario analysis) ───────────────────────────────
        # Isolated in try/except: a crash here must NOT prevent scores from
        # being returned.  Scores are more important than scenario analysis.
        try:
            val_range = compute_valuation_range(stock_data, metrics=norm_metrics)
        except Exception as exc:
            import traceback as _tb
            print(f"  [FUND WARN] compute_valuation_range failed: {exc}")
            print(_tb.format_exc())
            val_range = None

        return self._reply(
            message,
            MessageType.ANALYSIS_RESPONSE,
            payload={
                "analysis_type":     "fundamental",
                "valuation":          val_score,
                "growth":             gro_score,
                "profitability":      pro_score,
                "financial_health":   hlt_score,
                "valuation_range":    val_range,
                # NormalizedMetrics flows downstream so risk/reporting agents
                # and web_api can reuse it without recomputing.
                "normalized_metrics": norm_metrics,
            },
            confidence=confidence,
            reasoning_summary=f"Fundamental analysis complete. {summary}",
        )
