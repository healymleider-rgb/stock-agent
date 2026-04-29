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
from analysis.trend import detect_trends
from analysis.valuation import score_valuation
from analysis.valuation_range import compute_valuation_range
from utils.helpers import clamp
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

        # ── Data Integrity check ───────────────────────────────────────────────
        # Runs after NormalizedMetrics so all source annotations are available.
        # ValidationResult flows downstream to risk scorer and reporting agent.
        validation_result = None
        try:
            from analysis.data_integrity import run_data_integrity_check
            validation_result = run_data_integrity_check(norm_metrics, stock_data)
            # Apply integrity corrections back to NormalizedMetrics so ALL downstream
            # scorers automatically use the corrected values (price override,
            # recomputed PE/PS/EV). This is the "Metric Reliability Engine" coupling:
            # the integrity check feeds corrections back into the single source of truth.
            if validation_result is not None:
                norm_metrics.apply_integrity_corrections(validation_result)
        except Exception as _die:
            print(f"  [INTEGRITY] check skipped: {_die}")

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

        # ── Trend detection + score adjustments ───────────────────────────────
        trends = None
        try:
            trends = detect_trends(stock_data)
            if trends.growth_adj != 0.0:
                gro_score.score = clamp(gro_score.score + trends.growth_adj)
                gro_score.factors.append(
                    f"Trend adjustment ({trends.revenue_growth} revenue trend): "
                    f"{trends.growth_adj:+.0f}pts"
                )
                print(
                    f"  [TREND] growth_adj={trends.growth_adj:+.0f}"
                    f" revenue={trends.revenue_growth}"
                )
            if trends.profitability_adj != 0.0:
                pro_score.score = clamp(pro_score.score + trends.profitability_adj)
                pro_score.factors.append(
                    f"Trend adjustment ({trends.op_margin} margin trend): "
                    f"{trends.profitability_adj:+.0f}pts"
                )
                print(
                    f"  [TREND] prof_adj={trends.profitability_adj:+.0f}"
                    f" op_margin={trends.op_margin}"
                )
            # Apply confidence penalty for volatile metrics
            if trends.confidence_penalty > 0.0:
                confidence = max(0.05, confidence - trends.confidence_penalty)
                print(
                    f"  [TREND] volatile metrics → confidence_penalty={trends.confidence_penalty:.2f}"
                    f" → confidence={confidence:.3f}"
                )
        except Exception as _te:
            print(f"  [TREND] detection skipped: {_te}")

        # ── Valuation range (scenario analysis) ───────────────────────────────
        # Isolated in try/except: a crash here must NOT prevent scores from
        # being returned.  Scores are more important than scenario analysis.
        try:
            val_range = compute_valuation_range(stock_data, metrics=norm_metrics, trends=trends)
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
                # ValidationResult carries integrity flags and per-metric
                # confidence for risk scoring and report DATA QUALITY section.
                "validation": validation_result,
                # TrendResult carries per-metric trends and signals for display
                "trends": trends,
            },
            confidence=confidence,
            reasoning_summary=f"Fundamental analysis complete. {summary}",
        )
