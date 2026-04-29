"""
DataRetrievalAgent — the only agent permitted to call external data APIs.

Accepts DATA_REQUEST messages from the Orchestrator, fetches the requested
data slice via the DataBroker (FMP primary + optional fallbacks), and returns
a DATA_RESPONSE including source attribution.

Supported request types (payload["data_type"]):
  "profile"       → company profile + current quote
  "financials"    → income, balance, cash flow, ratios (annual)
  "quarterly"     → quarterly income statements
  "price_history" → OHLCV price series
  "earnings"      → earnings history + surprises
  "analyst"       → analyst recommendations + price targets
  "sector"        → sector performance

Each response payload includes:
  "source"        → provider name that supplied the dataset ("FMP", etc.)
  "field_sources" → {prefix.field_name: provider} for any field-level fills
"""
from __future__ import annotations

from api.data_broker import DataBroker
from api.fmp_client import FMPError
from agents.base_agent import BaseAgent
from config import Config
from models.message import AgentMessage, MessageType
from models.stock_data import BalanceSheet, CompanyProfile, FinancialRatios, IncomeStatement
from utils.logger import logger


def _make_fallback_profile(ticker: str) -> tuple[CompanyProfile, float, float]:
    """
    Synthetic profile for offline valuation / PEG testing.

    price=100.0 and market_cap=100B give shares≈1B, which lets the
    EV/EBITDA path compute equity-per-share from the fallback balance sheet.
    """
    profile = CompanyProfile(
        symbol=ticker,
        company_name=ticker,
        sector="Unknown",
        industry="Unknown",
        description="",
        exchange="",
        country="",
        market_cap=100_000_000_000.0,
        price=100.0,
    )
    return profile, 100.0, 100_000_000_000.0


def _make_fallback_financials() -> tuple[
    list[IncomeStatement], list[BalanceSheet], list, list[FinancialRatios]
]:
    """
    Synthetic financial data for offline valuation / PEG testing.

    Mimics a mid-cap, profitable, high-quality company:
      EPS growth ≈ 10 % annualised (3.18 → 3.50 over one year)
      Revenue 45 B  |  EBITDA 12 B  |  Net income 9.1 B
      P/E 22  |  EV/EBITDA 14  |  P/S 5
      Operating margin 25 %  →  quality multipliers use ±15 % range

    These are clearly labelled "fallback_test_data" in the source field so
    the reporting layer can display a prominent disclaimer.
    """
    income = [
        IncomeStatement(
            date="2024-12-31",
            period="FY",
            revenue=45_000_000_000.0,
            gross_profit=18_000_000_000.0,
            operating_income=11_250_000_000.0,
            net_income=9_100_000_000.0,
            ebitda=12_000_000_000.0,
            eps=3.50,
            eps_diluted=3.50,
            gross_profit_ratio=0.40,
            operating_income_ratio=0.25,
            net_income_ratio=0.202,
        ),
        IncomeStatement(
            date="2023-12-31",
            period="FY",
            revenue=40_900_000_000.0,
            gross_profit=16_360_000_000.0,
            operating_income=10_225_000_000.0,
            net_income=8_272_000_000.0,
            ebitda=10_900_000_000.0,
            eps=3.18,
            eps_diluted=3.18,
            gross_profit_ratio=0.40,
            operating_income_ratio=0.25,
            net_income_ratio=0.202,
        ),
    ]
    balance = [
        BalanceSheet(
            date="2024-12-31",
            period="FY",
            total_assets=55_000_000_000.0,
            total_liabilities=20_000_000_000.0,
            total_equity=35_000_000_000.0,
            total_debt=10_000_000_000.0,
            short_term_debt=2_000_000_000.0,
            long_term_debt=8_000_000_000.0,
            cash_and_equivalents=5_000_000_000.0,
            total_current_assets=15_000_000_000.0,
            total_current_liabilities=8_000_000_000.0,
        ),
    ]
    ratios = [
        FinancialRatios(
            date="2024-12-31",
            period="FY",
            pe_ratio=22.0,
            ps_ratio=5.0,
            ev_to_ebitda=14.0,
            roe=0.26,
            roa=0.165,
            operating_margin=0.25,
            net_margin=0.202,
            gross_margin=0.40,
            current_ratio=1.88,
            debt_to_equity=0.29,
        ),
    ]
    return income, balance, [], ratios


class DataRetrievalAgent(BaseAgent):
    name = "data_retrieval_agent"

    def __init__(self) -> None:
        self._broker = DataBroker()

    # ── Message handler ────────────────────────────────────────────────────────

    def process_message(self, message: AgentMessage) -> AgentMessage:
        if message.message_type != MessageType.DATA_REQUEST:
            return self._error_response(message, "Unexpected message type")

        ticker = message.ticker
        data_type = message.payload.get("data_type", "")
        reason = message.payload.get("reason", "requested by orchestrator")
        logger.info(
            "DataRetrieval: fetching '%s' for %s — reason: %s",
            data_type, ticker, reason,
        )

        handler = self._dispatch.get(data_type)
        if handler is None:
            return self._error_response(message, f"Unknown data_type: '{data_type}'")

        try:
            payload, confidence, summary = handler(self, ticker)
        except FMPError as exc:
            err = f"FMP API error: {exc}"
            print(f"\n  !! [DATA-AGENT FAILED] {data_type}: {err}\n")
            return self._error_response(message, err)
        except Exception as exc:  # noqa: BLE001
            err = f"Unexpected fetch error: {exc}"
            print(f"\n  !! [DATA-AGENT FAILED] {data_type}: {err}\n")
            return self._error_response(message, err)

        if "error" in payload:
            print(f"\n  !! [DATA-AGENT FAILED] {data_type}: {payload['error']}\n")
        else:
            source = payload.get("source", "FMP")
            field_sources = payload.get("field_sources", {})
            _dbg = {
                k: (f"list[{len(v)}]" if isinstance(v, list) else type(v).__name__)
                for k, v in payload.items()
                if k not in ("source", "field_sources")
            }
            fallback_note = (
                f" [fallback={source}]" if source not in ("FMP", "unavailable") else ""
            )
            field_note = (
                f" [field-fills: {len(field_sources)}]" if field_sources else ""
            )
            print(f"  [DATA-AGENT OK] {data_type}{fallback_note}{field_note}: {_dbg}")

        return self._reply(
            message,
            MessageType.DATA_RESPONSE,
            payload={"data_type": data_type, **payload},
            confidence=confidence,
            reasoning_summary=summary,
        )

    # ── Per-type fetch methods ─────────────────────────────────────────────────

    def _fetch_profile(self, ticker: str) -> tuple[dict, float, str]:
        result, source, field_sources = self._broker.get_profile(ticker)
        if result.is_empty():
            logger.warning(
                "DataRetrieval [%s]: no profile from any provider — using fallback", ticker
            )
            profile, price, market_cap = _make_fallback_profile(ticker)
            print(
                f"  [DATA-AGENT] profile missing for {ticker}"
                " — injecting fallback test data"
            )
            return (
                {
                    "profile": profile,
                    "current_price": price,
                    "market_cap": market_cap,
                    "quote": {},
                    "source": "fallback_test_data",
                    "field_sources": {},
                },
                0.3,
                f"Using fallback profile for {ticker} — valuation/PEG testing only",
            )
        if result.current_price is None:
            logger.warning(
                "DataRetrieval [%s]: profile OK but no price in quote", ticker
            )
        logger.info(
            "DataRetrieval [%s]: profile OK — company=%s price=%s mktcap=%s shares=%s via %s",
            ticker, result.profile.company_name, result.current_price,
            result.market_cap, result.shares_outstanding, source,
        )
        print(
            f"  [DATA-AGENT AUDIT] {ticker} profile raw inputs:"
            f"  price={result.current_price}"
            f"  shares_outstanding={result.shares_outstanding}"
            f"  market_cap_api={result.market_cap}"
            f"  market_cap_computed={result.market_cap_computed}"
        )
        return (
            {
                "profile": result.profile,
                "current_price": result.current_price,
                "market_cap": result.market_cap,
                "shares_outstanding": result.shares_outstanding,
                "market_cap_computed": result.market_cap_computed,
                "quote": result.quote,
                "source": source,
                "field_sources": field_sources,
                "shares_source":            result.shares_source,
                "shares_filing_period_end": result.shares_filing_period_end,
                "shares_filing_url":        result.shares_filing_url,
                "shares_data_refreshed_at": result.shares_data_refreshed_at,
            },
            0.9,
            f"Fetched profile for {result.profile.company_name} ({ticker}) via {source}",
        )

    def _fetch_financials(self, ticker: str) -> tuple[dict, float, str]:
        limit = Config.FINANCIAL_STATEMENT_LIMIT
        result, source, field_sources = self._broker.get_financials(ticker, limit)
        income   = result.income_statements
        balance  = result.balance_sheets
        cashflow = result.cash_flows
        ratios   = result.ratios
        logger.info(
            "DataRetrieval [%s]: financials — income=%d balance=%d"
            " cashflow=%d ratios=%d via %s",
            ticker, len(income), len(balance), len(cashflow), len(ratios), source,
        )
        for name, lst in [
            ("income", income), ("balance", balance),
            ("cashflow", cashflow), ("ratios", ratios),
        ]:
            if not lst:
                logger.warning("DataRetrieval [%s]: %s returned EMPTY", ticker, name)
        if not income:
            income, balance, cashflow, ratios = _make_fallback_financials()
            print(
                f"  [DATA-AGENT] financials empty for {ticker}"
                " — injecting fallback test data (valuation/PEG testing only)"
            )
            return (
                {
                    "income_statements": income,
                    "balance_sheets": balance,
                    "cash_flows": cashflow,
                    "ratios": ratios,
                    "source": "fallback_test_data",
                    "field_sources": {},
                },
                0.3,
                "Using fallback test data — valuation/PEG testing only",
            )
        conf = 0.9 if (income and balance and cashflow and ratios) else 0.5
        return (
            {
                "income_statements": income,
                "balance_sheets": balance,
                "cash_flows": cashflow,
                "ratios": ratios,
                "source": source,
                "field_sources": field_sources,
            },
            conf,
            f"Fetched {len(income)} years of annual financials for {ticker} via {source}",
        )

    def _fetch_quarterly(self, ticker: str) -> tuple[dict, float, str]:
        quarterly, source, field_sources = self._broker.get_quarterly(ticker)
        return (
            {
                "quarterly_income": quarterly,
                "source": source,
                "field_sources": field_sources,
            },
            0.8 if quarterly else 0.2,
            f"Fetched {len(quarterly)} quarters of income data via {source}",
        )

    def _fetch_price_history(self, ticker: str) -> tuple[dict, float, str]:
        ph, source, field_sources = self._broker.get_price_history(
            ticker, Config.PRICE_HISTORY_DAYS
        )
        if ph is None or len(ph) < 20:
            return (
                {"error": "Insufficient price history"},
                0.2,
                "Price history unavailable",
            )
        return (
            {
                "price_history": ph,
                "source": source,
                "field_sources": field_sources,
            },
            0.95,
            f"Fetched {len(ph)} trading days of price history for {ticker} via {source}",
        )

    def _fetch_earnings(self, ticker: str) -> tuple[dict, float, str]:
        (earnings, surprises), source, field_sources = self._broker.get_earnings(ticker)
        return (
            {
                "earnings": earnings,
                "earnings_surprises": surprises,
                "source": source,
                "field_sources": field_sources,
            },
            0.8 if earnings else 0.3,
            f"Fetched {len(earnings)} earnings records via {source}",
        )

    def _fetch_analyst(self, ticker: str) -> tuple[dict, float, str]:
        result, source, field_sources = self._broker.get_analyst_data(ticker)
        return (
            {
                "analyst_recommendations": result.recommendations,
                "price_targets": result.price_targets,
                "source": source,
                "field_sources": field_sources,
            },
            0.7 if result.recommendations else 0.4,
            f"Fetched analyst data: {len(result.recommendations)} recommendations via {source}",
        )

    def _fetch_sector(self, ticker: str) -> tuple[dict, float, str]:
        sector_data, source, field_sources = self._broker.get_sector_performance()
        return (
            {
                "sector_performance": sector_data,
                "source": source,
                "field_sources": field_sources,
            },
            0.8 if sector_data else 0.3,
            f"Fetched sector performance data via {source}",
        )

    _dispatch = {
        "profile":       _fetch_profile,
        "financials":    _fetch_financials,
        "quarterly":     _fetch_quarterly,
        "price_history": _fetch_price_history,
        "earnings":      _fetch_earnings,
        "analyst":       _fetch_analyst,
        "sector":        _fetch_sector,
    }

    # ── Call audit ─────────────────────────────────────────────────────────────

    @property
    def api_call_log(self) -> list[str]:
        return self._broker.fmp_call_log
