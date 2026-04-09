"""
AlphaVantageProvider — financial statements fallback via Alpha Vantage API.

Scope
-----
This provider covers financial statement data only:
  - INCOME_STATEMENT  → IncomeStatement rows
  - BALANCE_SHEET     → BalanceSheet rows
  - CASH_FLOW         → CashFlowStatement rows

It does NOT provide:
  - Profile / quote data (no FMP-equivalent endpoint)
  - FinancialRatios (AV ratios live in OVERVIEW — not implemented here)
  - Price history
  - Earnings surprises
  - Analyst recommendations
  - Sector performance

All unimplemented methods return empty / None so the DataBroker waterfall
continues to the next provider without errors.

Activation
----------
Set ALPHA_VANTAGE_API_KEY in your .env file.
Without this key, is_available() returns False and this provider is skipped.

API
---
Alpha Vantage REST API: https://www.alphavantage.co/documentation/
Base URL: https://www.alphavantage.co/query
Auth: ?apikey=<key> query parameter

Rate limits (as of 2024):
  Free tier  : 25 requests / day, 5 / minute
  Premium    : 75 requests / minute (and above, depending on plan)

  Each call to get_financials() makes 3 requests (income + balance + cashflow).
  With a free key this exhausts ~12% of the daily quota per ticker.
  Consider caching results if you evaluate multiple tickers per session.

Field mapping
-------------
Alpha Vantage returns all numeric fields as strings.
Their null sentinel is the string "None" — safe_float() handles this correctly
by catching ValueError from float("None") and returning None.

  AV field                              → Our model field
  ─────────────────────────────────────────────────────────
  INCOME_STATEMENT
  fiscalDateEnding                      → date
  totalRevenue                          → revenue
  grossProfit                           → gross_profit
  operatingIncome                       → operating_income
  netIncome                             → net_income
  ebitda                                → ebitda
  researchAndDevelopment                → rd_expenses
  sellingGeneralAndAdministrative       → selling_expenses
  interestExpense                       → interest_expense

  BALANCE_SHEET
  fiscalDateEnding                      → date
  totalAssets                           → total_assets
  totalLiabilities                      → total_liabilities
  totalShareholderEquity                → total_equity
  shortLongTermDebtTotal                → total_debt
  shortTermDebt                         → short_term_debt
  longTermDebtNoncurrent                → long_term_debt
  cashAndCashEquivalentsAtCarryingValue → cash_and_equivalents
  shortTermInvestments                  → short_term_investments
  totalCurrentAssets                    → total_current_assets
  totalCurrentLiabilities               → total_current_liabilities
  goodwill                              → goodwill
  intangibleAssets                      → intangible_assets
  retainedEarnings                      → retained_earnings

  CASH_FLOW
  fiscalDateEnding                      → date
  operatingCashflow                     → operating_cash_flow
  capitalExpenditures                   → capital_expenditure
  (computed: operating − capex)         → free_cash_flow
  profitLoss                            → net_income
  depreciationDepletionAndAmortization  → depreciation_amortization
  dividendPayoutCommonStock             → dividends_paid
  paymentsForRepurchaseOfCommonStock    → common_stock_repurchased
"""
from __future__ import annotations

import time
from typing import Any, Optional

import requests

from api.provider_base import AnalystResult, DataProvider, FinancialsResult, ProfileResult
from config import Config
from models.stock_data import (
    BalanceSheet,
    CashFlowStatement,
    IncomeStatement,
    PriceHistory,
)
from utils.helpers import safe_float
from utils.logger import logger

_AV_BASE = "https://www.alphavantage.co/query"
_REQUEST_TIMEOUT = 2   # seconds — fail fast; AV is a fallback, not the primary source

# Free tier: 5 requests/minute → enforce at least 13s between calls.
# Each get_financials() makes 3 requests; without pacing all three land
# within 1 second and reliably trigger the rate limit.
_MIN_REQUEST_INTERVAL = 13.0  # seconds

# Seconds to wait before the single 429 retry.
_RATE_LIMIT_WAIT = 2


class AlphaVantageProvider(DataProvider):
    """
    Alpha Vantage fallback — supplies income, balance sheet, and cash flow
    statements when FMP returns empty or access-restricted results.
    """

    name = "AlphaVantage"

    # When True, DataBroker skips this provider in the field-level fill loop
    # so it is not called when FMP has already returned a non-empty dataset.
    skip_when_fmp_has_data = True

    def __init__(self) -> None:
        self._api_key = Config.ALPHA_VANTAGE_API_KEY
        self._session = requests.Session()
        self._last_request_ts: float = 0.0  # unix timestamp of the last completed request
        # Set to True after the first in-band rate-limit message so all
        # subsequent calls within this provider instance are skipped immediately.
        self._rate_limited: bool = False

    def is_available(self) -> bool:
        available = bool(self._api_key)
        if not available:
            logger.debug(
                "AlphaVantageProvider: ALPHA_VANTAGE_API_KEY not set — provider disabled"
            )
        return available

    # ── DataProvider interface ─────────────────────────────────────────────────

    def get_profile(self, ticker: str) -> ProfileResult:
        # Not implemented — AV OVERVIEW endpoint is not mapped to ProfileResult.
        # Use FMP for profile data.
        return ProfileResult()

    def get_financials(self, ticker: str, limit: int = 5) -> FinancialsResult:
        income   = self._fetch_income_statements(ticker, limit)
        balance  = self._fetch_balance_sheets(ticker, limit)
        cashflow = self._fetch_cash_flows(ticker, limit)
        return FinancialsResult(
            income_statements=income,
            balance_sheets=balance,
            cash_flows=cashflow,
            # Alpha Vantage does not provide FinancialRatios via the statement
            # endpoints. The OVERVIEW endpoint covers some ratios but requires
            # a separate implementation. Left empty so the broker's field-level
            # merge logic simply skips it.
            ratios=[],
        )

    def get_quarterly(self, ticker: str) -> list[IncomeStatement]:
        # Not implemented — AV quarterly income is available but not mapped yet.
        # TODO: fetch annualReports → quarterlyReports from INCOME_STATEMENT
        return []

    def get_price_history(self, ticker: str, days: int = 400) -> Optional[PriceHistory]:
        # Not implemented — use FMP for price history.
        return None

    def get_earnings(self, ticker: str) -> tuple[list[dict], list[dict]]:
        # Not implemented — use FMP for earnings.
        return [], []

    def get_analyst_data(self, ticker: str) -> AnalystResult:
        # Not implemented — use FMP for analyst data.
        return AnalystResult()

    def get_sector_performance(self) -> list[dict[str, Any]]:
        # Not implemented — AV has no sector performance endpoint.
        return []

    # ── Private fetch methods ──────────────────────────────────────────────────

    def _get(self, function: str, symbol: str) -> Any:
        """
        Rate-limited GET to the Alpha Vantage query endpoint.

        Pacing: enforces _MIN_REQUEST_INTERVAL between successive calls so the
        free-tier limit (5 req/min) is never exceeded even when get_financials()
        fires three requests back-to-back.

        Retry: handles both HTTP 429 and AV's in-band rate-limit messages
        ("Information" / "Note" keys) with short backoffs.
        """
        # ── Abort immediately if this instance already hit a rate limit ──────
        if self._rate_limited:
            logger.debug(
                "AlphaVantageProvider: skipping %s/%s — rate limit already hit this run",
                function, symbol,
            )
            return None

        # ── Inter-request pacing ───────────────────────────────────────────────
        elapsed = time.time() - self._last_request_ts
        if elapsed < _MIN_REQUEST_INTERVAL:
            wait = _MIN_REQUEST_INTERVAL - elapsed
            print(
                f"  [AV] pacing {function}/{symbol}"
                f" — waiting {wait:.1f}s (rate limit guard)"
            )
            time.sleep(wait)

        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self._api_key,
        }
        logger.info("AlphaVantageProvider: requesting %s for %s", function, symbol)

        # ── Request with 429 retry (wait 2s, retry once) ──────────────────────
        for _attempt in range(2):
            self._last_request_ts = time.time()
            try:
                resp = self._session.get(_AV_BASE, params=params, timeout=_REQUEST_TIMEOUT)
            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "AlphaVantageProvider: request error for %s/%s — %s",
                    function, symbol, exc,
                )
                return None

            if resp.status_code == 429:
                if _attempt == 0:
                    print(
                        f"  [AV] {function}/{symbol}: HTTP 429 — rate limit,"
                        f" waiting {_RATE_LIMIT_WAIT}s, retrying once"
                    )
                    logger.warning(
                        "AlphaVantageProvider: 429 for %s/%s — sleeping %ds",
                        function, symbol, _RATE_LIMIT_WAIT,
                    )
                    time.sleep(_RATE_LIMIT_WAIT)
                    continue
                print(f"  [AV] {function}/{symbol}: HTTP 429 — rate limit after retry")
                logger.warning("AlphaVantageProvider: 429 for %s/%s — giving up", function, symbol)
                return None

            try:
                resp.raise_for_status()
            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "AlphaVantageProvider: HTTP error for %s/%s — %s",
                    function, symbol, exc,
                )
                return None

            break  # successful HTTP response

        data = resp.json()

        # ── AV in-band rate-limit / error messages ─────────────────────────────
        if "Information" in data:
            self._rate_limited = True
            logger.warning(
                "AlphaVantageProvider: API notice for %s/%s — %s (aborting further AV calls)",
                function, symbol, data["Information"],
            )
            print(
                f"  [AV] {function}/{symbol}: rate limit or plan restriction"
                f" — {data['Information'][:120]}"
                f" (AV disabled for remainder of this run)"
            )
            return None
        if "Note" in data:
            self._rate_limited = True
            logger.warning(
                "AlphaVantageProvider: rate-limit note for %s/%s — %s (aborting further AV calls)",
                function, symbol, data["Note"],
            )
            print(
                f"  [AV] {function}/{symbol}: rate limit — {data['Note'][:120]}"
                f" (AV disabled for remainder of this run)"
            )
            return None
        if "Error Message" in data:
            logger.warning(
                "AlphaVantageProvider: error for %s/%s — %s",
                function, symbol, data["Error Message"],
            )
            return None

        _data_len = len(data) if isinstance(data, (list, dict)) else "?"
        print(f"  [AV] {function}/{symbol} → {type(data).__name__}[{_data_len}]")
        return data

    # ── Statement parsers ──────────────────────────────────────────────────────

    def _fetch_income_statements(
        self, symbol: str, limit: int
    ) -> list[IncomeStatement]:
        data = self._get("INCOME_STATEMENT", symbol)
        if not data:
            return []
        reports = data.get("annualReports", [])
        if not reports:
            logger.warning(
                "AlphaVantageProvider: INCOME_STATEMENT/%s — no annualReports", symbol
            )
            return []
        results = []
        for raw in reports[:limit]:
            results.append(IncomeStatement(
                date=raw.get("fiscalDateEnding", ""),
                period="FY",
                revenue=safe_float(raw.get("totalRevenue")),
                gross_profit=safe_float(raw.get("grossProfit")),
                operating_income=safe_float(raw.get("operatingIncome")),
                net_income=safe_float(raw.get("netIncome")),
                ebitda=safe_float(raw.get("ebitda")),
                # AV does not include EPS in INCOME_STATEMENT (use EARNINGS endpoint)
                eps=None,
                eps_diluted=None,
                # Margin ratios are not provided by AV statements — compute later
                # via field-level merge or leave for FMP ratios endpoint
                gross_profit_ratio=None,
                operating_income_ratio=None,
                net_income_ratio=None,
                rd_expenses=safe_float(raw.get("researchAndDevelopment")),
                selling_expenses=safe_float(raw.get("sellingGeneralAndAdministrative")),
                interest_expense=safe_float(raw.get("interestExpense")),
            ))
        logger.info(
            "AlphaVantageProvider: INCOME_STATEMENT/%s — %d rows", symbol, len(results)
        )
        return results

    def _fetch_balance_sheets(
        self, symbol: str, limit: int
    ) -> list[BalanceSheet]:
        data = self._get("BALANCE_SHEET", symbol)
        if not data:
            return []
        reports = data.get("annualReports", [])
        if not reports:
            logger.warning(
                "AlphaVantageProvider: BALANCE_SHEET/%s — no annualReports", symbol
            )
            return []
        results = []
        for raw in reports[:limit]:
            results.append(BalanceSheet(
                date=raw.get("fiscalDateEnding", ""),
                period="FY",
                total_assets=safe_float(raw.get("totalAssets")),
                total_liabilities=safe_float(raw.get("totalLiabilities")),
                total_equity=safe_float(raw.get("totalShareholderEquity")),
                # AV provides shortLongTermDebtTotal as the combined debt figure
                total_debt=safe_float(raw.get("shortLongTermDebtTotal")),
                short_term_debt=safe_float(raw.get("shortTermDebt")),
                # AV splits long-term debt into current and noncurrent portions
                long_term_debt=safe_float(raw.get("longTermDebtNoncurrent")),
                cash_and_equivalents=safe_float(
                    raw.get("cashAndCashEquivalentsAtCarryingValue")
                ),
                short_term_investments=safe_float(raw.get("shortTermInvestments")),
                total_current_assets=safe_float(raw.get("totalCurrentAssets")),
                total_current_liabilities=safe_float(raw.get("totalCurrentLiabilities")),
                goodwill=safe_float(raw.get("goodwill")),
                intangible_assets=safe_float(raw.get("intangibleAssets")),
                retained_earnings=safe_float(raw.get("retainedEarnings")),
            ))
        logger.info(
            "AlphaVantageProvider: BALANCE_SHEET/%s — %d rows", symbol, len(results)
        )
        return results

    def _fetch_cash_flows(
        self, symbol: str, limit: int
    ) -> list[CashFlowStatement]:
        data = self._get("CASH_FLOW", symbol)
        if not data:
            return []
        reports = data.get("annualReports", [])
        if not reports:
            logger.warning(
                "AlphaVantageProvider: CASH_FLOW/%s — no annualReports", symbol
            )
            return []
        results = []
        for raw in reports[:limit]:
            operating = safe_float(raw.get("operatingCashflow"))
            capex = safe_float(raw.get("capitalExpenditures"))
            # AV does not expose free cash flow directly; compute it when both
            # components are available.
            if operating is not None and capex is not None:
                free_cf = operating - abs(capex)
            else:
                free_cf = None
            results.append(CashFlowStatement(
                date=raw.get("fiscalDateEnding", ""),
                period="FY",
                operating_cash_flow=operating,
                capital_expenditure=capex,
                free_cash_flow=free_cf,
                net_income=safe_float(raw.get("profitLoss")),
                depreciation_amortization=safe_float(
                    raw.get("depreciationDepletionAndAmortization")
                ),
                # AV does not break out stock-based compensation in CASH_FLOW
                stock_based_compensation=None,
                dividends_paid=safe_float(raw.get("dividendPayoutCommonStock")),
                # AV does not expose acquisitionsNet in CASH_FLOW
                acquisitions=None,
                common_stock_repurchased=safe_float(
                    raw.get("paymentsForRepurchaseOfCommonStock")
                ),
            ))
        logger.info(
            "AlphaVantageProvider: CASH_FLOW/%s — %d rows", symbol, len(results)
        )
        return results
