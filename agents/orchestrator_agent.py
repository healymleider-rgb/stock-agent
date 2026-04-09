"""
OrchestratorAgent — the system's central coordinator.

Controls the full evaluation lifecycle:
  1. Build an evaluation plan based on the ticker
  2. Run a reasoning loop: decide → fetch → analyze → refine
  3. Route messages to the appropriate specialist agents
  4. Track confidence and iterate until confident or at the iteration limit
  5. Trigger final reporting

Reasoning Loop
──────────────
  while not state.is_complete():
      action = _decide_next_action(state)
      message = _create_message(action)
      response = _route(message)
      _update_state(state, response)
      state.iteration += 1

The loop is intentionally transparent — every decision is logged to
state.reasoning_log so you can trace exactly what happened and why.
"""
from __future__ import annotations

from agents.data_retrieval_agent import DataRetrievalAgent
from agents.fundamental_analysis_agent import FundamentalAnalysisAgent
from agents.macro_lei_agent import MacroLEIAgent
from agents.market_analyst_agent import MarketAnalystAgent
from agents.reporting_agent import ReportingAgent
from agents.risk_management_agent import RiskManagementAgent
from agents.sentiment_analysis_agent import SentimentAnalysisAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from models.message import AgentMessage, MessageType
from models.scorecard import Scorecard
from models.state import EvaluationState
from utils.logger import logger


class OrchestratorAgent:
    """
    Not a BaseAgent subclass — the Orchestrator orchestrates; it doesn't
    respond to messages from another orchestrator.
    """

    def __init__(self) -> None:
        # Instantiate all specialist agents
        self._data = DataRetrievalAgent()
        self._fundamental = FundamentalAnalysisAgent()
        self._technical = TechnicalAnalysisAgent()
        self._market = MarketAnalystAgent()
        self._sentiment = SentimentAnalysisAgent()
        self._risk = RiskManagementAgent()
        self._macro = MacroLEIAgent()
        self._reporting = ReportingAgent()

    # ── Public entry point ────────────────────────────────────────────────────

    def evaluate(self, ticker: str) -> tuple[Scorecard, str, EvaluationState]:
        """
        Run a full evaluation for `ticker`.

        Returns
        -------
        scorecard : Scorecard
            Fully computed scorecard with subscores and stance.
        memo : str
            Human-readable investment memo.
        state : EvaluationState
            Full evaluation state including reasoning log.
        """
        ticker = ticker.upper().strip()
        state = EvaluationState(ticker=ticker)
        state.log(f"Evaluation started for {ticker}")

        self._run_reasoning_loop(state)

        # Generate final report
        scorecard, memo = self._request_final_report(state)
        state.scorecard = scorecard
        state.log(f"Evaluation complete — overall score {scorecard.overall_score:.0f}/100, stance {scorecard.stance.value}")

        return scorecard, memo, state

    # ── Reasoning loop ────────────────────────────────────────────────────────

    def _run_reasoning_loop(self, state: EvaluationState) -> None:
        """
        Core agentic loop.
        Each iteration decides the best next action, executes it,
        and updates state.
        """
        while not state.is_complete():
            state.iteration += 1
            action = self._decide_next_action(state)

            if action is None:
                state.log("No further actions identified — exiting loop early.")
                break

            state.log(f"Action selected: {action}")
            self._execute_action(action, state)
            self._update_confidence(state)

            logger.info(
                "Orchestrator iter=%d action=%s confidence=%.2f",
                state.iteration,
                action,
                state.confidence,
            )

    def _decide_next_action(self, state: EvaluationState) -> str | None:
        """
        Decision function: return the most valuable next action given current state.

        Priority order:
          1. Fetch profile (always first — needed to understand the company)
          2. Fetch financials (needed for fundamental analysis)
          3. Run fundamental analysis (highest-weight categories)
          4. Run macro LEI analysis (independent of stock data — can run early)
          5. Fetch price history (needed for technical analysis)
          6. Run technical analysis
          7. Fetch sector + analyst data (context layer)
          8. Run market analysis
          9. Run sentiment analysis (scaffold in V1)
          10. Run risk analysis
          11. Fetch quarterly data (refinement if confidence is low)
          12. Fetch earnings data (additional context)
          13. Done
        """
        d = state.data_fetched
        a = state.analyses_completed

        # ── Data fetches ───────────────────────────────────────────────────────
        if not d["profile"]:
            return "fetch:profile"
        if not d["financials"]:
            return "fetch:financials"

        # ── Quarterly data — always fetch before fundamental analysis ─────────
        # quarterly_income is required for TTM EPS computation in valuation_range
        # and metrics.compute_core_metrics. Do not defer it to the low-confidence
        # refinement block; without it, TTM P/E cannot be computed.
        if d["financials"] and not d["quarterly"]:
            return "fetch:quarterly"

        # ── Core analyses (require financials + quarterly) ─────────────────────
        if d["financials"] and d["quarterly"] and not a["fundamental"]:
            return "analyze:fundamental"

        # ── Macro LEI (independent of stock data — FRED-sourced) ─────────────
        if not a["macro"]:
            return "analyze:macro"

        if not d["price_history"]:
            return "fetch:price_history"
        if d["price_history"] and not a["technical"]:
            return "analyze:technical"

        # ── Context layer ──────────────────────────────────────────────────────
        if not d["analyst"]:
            return "fetch:analyst"
        if not d["sector"]:
            return "fetch:sector"
        if d["analyst"] and d["sector"] and not a["market"]:
            return "analyze:market"

        # ── Sentiment (scaffold) ───────────────────────────────────────────────
        if not a["sentiment"]:
            return "analyze:sentiment"

        # ── Risk analysis ──────────────────────────────────────────────────────
        if not a["risk"]:
            return "analyze:risk"

        # ── Refinement: fetch more data if confidence is still low ─────────────
        if state.confidence < state.confidence_threshold:
            if not d["earnings"]:
                return "fetch:earnings"

        return None  # All done

    def _execute_action(self, action: str, state: EvaluationState) -> None:
        """Dispatch the chosen action to the appropriate agent."""
        kind, name = action.split(":", 1)

        if kind == "fetch":
            self._do_fetch(name, state)
        elif kind == "analyze":
            self._do_analyze(name, state)
        else:
            state.log(f"Unknown action kind: {action}")

    # ── Data fetch dispatch ───────────────────────────────────────────────────

    def _do_fetch(self, data_type: str, state: EvaluationState) -> None:
        reason_map = {
            "profile":       "Need company identity, sector, and market cap before any analysis",
            "financials":    "Core annual financials required for fundamental analysis",
            "price_history": "Need OHLCV data for technical and momentum analysis",
            "analyst":       "Analyst consensus and price targets add market context",
            "sector":        "Sector performance provides macro backdrop",
            "earnings":      "Earnings history adds precision to growth and quality assessment",
            "quarterly":     "Quarterly data adds recency signal to trend analysis",
        }

        request = AgentMessage(
            sender="orchestrator",
            recipient="data_retrieval_agent",
            ticker=state.ticker,
            message_type=MessageType.DATA_REQUEST,
            payload={
                "data_type": data_type,
                "reason": reason_map.get(data_type, "requested"),
            },
        )
        response = self._data.handle(request)
        state.log(f"Data fetch '{data_type}': {response.reasoning_summary}")

        if response.is_error():
            err_msg = response.payload.get('error', 'unknown error')
            state.log(f"  ERROR: {err_msg}")
            state.mark_data_fetched(data_type)
            # Record in data_sources so reporting can show accurate completeness.
            # HTTP 402/403 means plan access restriction, not missing ticker data.
            if "402" in err_msg or "403" in err_msg:
                state.stock_data.data_sources[data_type] = "access_restricted"
            else:
                state.stock_data.data_sources[data_type] = "unavailable"
            print(f"\n  !! [ORCH FETCH FAILED] {data_type}: {err_msg}\n")
            return

        try:
            self._ingest_data_response(data_type, response.payload, state)
        except Exception as exc:  # noqa: BLE001
            # Ingest errors must NOT prevent mark_data_fetched — without it the
            # orchestrator retries the same endpoint every iteration until the
            # loop limit, overwriting any partially-ingested state each time.
            logger.exception(
                "Orchestrator: unexpected ingest error for '%s': %s", data_type, exc
            )
            print(f"\n  !! [ORCH INGEST ERROR] {data_type}: {exc}\n")
            state.mark_data_fetched(data_type)
            return

        state.mark_data_fetched(data_type)

        # Record provider attribution in StockData
        source = response.payload.get("source", "FMP")
        field_sources = response.payload.get("field_sources", {})
        state.stock_data.data_sources[data_type] = source
        if field_sources:
            state.stock_data.data_sources.update(field_sources)

    def _ingest_data_response(
        self, data_type: str, payload: dict, state: EvaluationState
    ) -> None:
        """Update the shared StockData from a DATA_RESPONSE payload."""
        sd = state.stock_data

        if data_type == "profile":
            # Guard: only assign non-None values so a missing key never
            # silently overwrites a field that was successfully populated earlier.
            incoming_profile        = payload.get("profile")
            incoming_price          = payload.get("current_price")
            incoming_mktcap         = payload.get("market_cap")
            incoming_shares         = payload.get("shares_outstanding")
            incoming_mktcap_cmp     = payload.get("market_cap_computed")
            if incoming_profile is not None:
                sd.profile = incoming_profile
            if incoming_price is not None:
                sd.current_price = incoming_price
            if incoming_mktcap is not None:
                sd.market_cap = incoming_mktcap
            if incoming_shares is not None:
                sd.shares_outstanding = incoming_shares
            if incoming_mktcap_cmp is not None:
                sd.market_cap_computed = incoming_mktcap_cmp

            company  = sd.profile.company_name if sd.profile else "NONE"
            sector   = sd.profile.sector       if sd.profile else "NONE"
            industry = sd.profile.industry     if sd.profile else "NONE"
            price    = sd.current_price
            mktcap   = sd.market_cap
            shares   = sd.shares_outstanding
            mktcap_c = sd.market_cap_computed

            # ── Run audit: profile inputs ────────────────────────────────────
            print(
                f"  [ORCH INGEST] profile → "
                f"ticker={state.ticker!r} company={company!r} "
                f"sector={sector!r} industry={industry!r}"
            )
            print(
                f"  [ORCH AUDIT] price={price}  "
                f"shares_outstanding={shares}  "
                f"market_cap_api={mktcap}  "
                f"market_cap_computed={mktcap_c}"
            )
            if mktcap and mktcap_c and mktcap_c > 0:
                diff_pct = abs(mktcap - mktcap_c) / mktcap_c * 100
                flag = "  *** DISCREPANCY ***" if diff_pct > 10 else ""
                print(f"  [ORCH AUDIT] market_cap diff = {diff_pct:.1f}%{flag}")

            if sd.profile:
                mktcap_str = f" | Market cap: ${mktcap/1e9:.2f}B" if mktcap else ""
                state.log(
                    f"  Profile: {company} | Sector: {sector} | Industry: {industry}{mktcap_str}"
                )
            else:
                print(
                    f"  !! [ORCH INGEST] profile payload delivered but sd.profile is None"
                    f" — payload keys: {list(payload.keys())}"
                )

        elif data_type == "financials":
            sd.income_statements = payload.get("income_statements", [])
            sd.balance_sheets = payload.get("balance_sheets", [])
            sd.cash_flows = payload.get("cash_flows", [])
            sd.ratios = payload.get("ratios", [])
            print(
                f"  [ORCH INGEST] financials → "
                f"income={len(sd.income_statements)} "
                f"balance={len(sd.balance_sheets)} "
                f"cashflow={len(sd.cash_flows)} "
                f"ratios={len(sd.ratios)}"
            )
            state.log(
                f"  Financials loaded: {len(sd.income_statements)} years of income data"
            )
            if not sd.income_statements:
                print(f"  !! [ORCH INGEST] income_statements is EMPTY after ingest — payload keys: {list(payload.keys())}")
            # Adaptive reasoning: check for red flags early
            self._early_risk_scan(sd, state)

        elif data_type == "quarterly":
            sd.quarterly_income = payload.get("quarterly_income", [])

        elif data_type == "price_history":
            sd.price_history = payload.get("price_history")
            ph = sd.price_history
            days = len(ph) if ph else 0
            print(f"  [ORCH INGEST] price_history → {days} trading days")
            if ph:
                state.log(f"  Price history: {days} trading days loaded")

        elif data_type == "earnings":
            sd.earnings = payload.get("earnings", [])
            sd.earnings_surprises = payload.get("earnings_surprises", [])

        elif data_type == "analyst":
            sd.analyst_recommendations = payload.get("analyst_recommendations", [])
            sd.price_targets = payload.get("price_targets", {})

        elif data_type == "sector":
            sd.sector_performance = payload.get("sector_performance", [])

    def _early_risk_scan(self, sd, state: EvaluationState) -> None:
        """
        Quick scan after financials are loaded.
        Allows the orchestrator to prioritize risk analysis if red flags appear.
        This is the 'interpret as we go' behavior.
        """
        cf = sd.latest_cashflow
        ratios = sd.latest_ratios
        if cf and cf.free_cash_flow is not None and cf.free_cash_flow < 0:
            state.log("  [Early signal] Negative free cash flow detected → risk analysis prioritized")
        if ratios and ratios.debt_to_equity is not None and ratios.debt_to_equity > 2.0:
            state.log("  [Early signal] High D/E ratio detected → risk analysis prioritized")

    # ── Analysis dispatch ─────────────────────────────────────────────────────

    def _do_analyze(self, analysis_type: str, state: EvaluationState) -> None:
        agent_map = {
            "fundamental": (self._fundamental, "Analyze income, balance, cash flow, and ratios"),
            "technical":   (self._technical,   "Compute momentum and technical indicators from price data"),
            "market":      (self._market,       "Assess sector context, analyst consensus, and price targets"),
            "sentiment":   (self._sentiment,    "Analyze news and social sentiment (V1 scaffold)"),
            "risk":        (self._risk,         "Identify all material financial and structural risks"),
            "macro":       (self._macro,        "Assess macroeconomic regime via FRED leading indicators"),
        }

        if analysis_type not in agent_map:
            state.log(f"Unknown analysis type: {analysis_type}")
            return

        agent, reason = agent_map[analysis_type]

        request = AgentMessage(
            sender="orchestrator",
            recipient=agent.name,
            ticker=state.ticker,
            message_type=MessageType.ANALYSIS_REQUEST,
            payload={
                "stock_data": state.stock_data,
                "reason": reason,
                "prior_findings": state.agent_findings,
            },
        )
        response = agent.handle(request)
        state.log(
            f"Analysis '{analysis_type}' complete — confidence {response.confidence:.2f}: "
            f"{response.reasoning_summary}"
        )

        if not response.is_error():
            state.agent_findings[analysis_type] = response.payload
            # Apply macro confidence modifier immediately so subsequent
            # agents and confidence tracking see the adjusted baseline.
            if analysis_type == "macro":
                modifier = response.payload.get("confidence_modifier", 0.0)
                if modifier != 0.0:
                    state.confidence = max(0.0, min(0.95, state.confidence + modifier))
                    regime = response.payload.get("macro_regime", "Unknown")
                    state.log(
                        f"  Macro modifier applied: {modifier:+.3f} "
                        f"(regime={regime}, new confidence={state.confidence:.2f})"
                    )

        state.mark_analysis_complete(analysis_type)

    # ── Final report ──────────────────────────────────────────────────────────

    def _request_final_report(
        self, state: EvaluationState
    ) -> tuple[Scorecard, str]:
        # Prefer metrics-based confidence (reflects actual data quality) over
        # the orchestrator's completion heuristic (reflects pipeline completion).
        # If fundamental analysis succeeded and stored NormalizedMetrics, use
        # compute_confidence() on that object for the final report confidence.
        from analysis.metrics import compute_confidence as _compute_confidence
        fund_findings = state.agent_findings.get("fundamental", {})
        norm_metrics  = fund_findings.get("normalized_metrics")
        if norm_metrics is not None:
            final_confidence = _compute_confidence(norm_metrics)
            print(
                f"  [ORCH] final_confidence from NormalizedMetrics:"
                f" {final_confidence:.3f} (orchestrator heuristic was {state.confidence:.3f})"
            )
        else:
            # Fundamental analysis failed — confidence must reflect missing categories.
            # Cap at 0.40 when fundamental findings are absent (4 of 6 categories missing).
            final_confidence = min(state.confidence, 0.40)
            print(
                f"  [ORCH] final_confidence capped at 0.40:"
                f" no normalized_metrics in fund_findings"
                f" (orchestrator heuristic was {state.confidence:.3f})"
            )

        request = AgentMessage(
            sender="orchestrator",
            recipient="reporting_agent",
            ticker=state.ticker,
            message_type=MessageType.FINAL_SUMMARY_REQUEST,
            payload={
                "stock_data": state.stock_data,
                "agent_findings": state.agent_findings,
                "confidence": final_confidence,
                "reasoning_log": state.reasoning_log,
            },
        )
        response = self._reporting.handle(request)
        if response.is_error():
            raise RuntimeError(f"Reporting agent failed: {response.payload.get('error')}")

        scorecard: Scorecard = response.payload["scorecard"]
        memo: str = response.payload["memo"]
        return scorecard, memo

    # ── Confidence tracking ───────────────────────────────────────────────────

    def _update_confidence(self, state: EvaluationState) -> None:
        """
        Confidence is a weighted composite of:
          - completion ratio of analysis pipeline
          - average confidence of individual agent responses (approximated)
          - data quality (penalize missing data)
        """
        completion = state.completion_ratio()

        # Simple heuristic: confidence tracks completion with a cap
        base = completion * 0.85

        # Bonus for having high-quality fundamental data
        sd = state.stock_data
        if sd.income_statements and sd.balance_sheets and sd.ratios:
            base = min(base + 0.10, 0.95)

        # Bonus for price history
        if sd.price_history and len(sd.price_history) >= 200:
            base = min(base + 0.05, 0.95)

        state.confidence = base

    @property
    def api_call_log(self) -> list[str]:
        return self._data.api_call_log
