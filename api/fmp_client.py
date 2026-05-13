"""
Financial Modeling Prep API client — Stable API.

This is the ONLY module that makes HTTP requests to FMP.
All other code receives normalized domain models from here.

Migration note
--------------
This client targets the FMP Stable API:
  https://financialmodelingprep.com/stable/

The legacy /v3/ endpoints were retired after August 31 2025.

Structural changes from v3
--------------------------
  - Symbol is always a QUERY PARAMETER (?symbol=AAPL), never a path segment.
  - Cash flow path is "/cash-flow-statement" (with all hyphens).
  - ROE / ROA / ROIC moved from /ratios to /key-metrics.
  - PE / PS / PB ratios are in /ratios only; they were removed from /key-metrics.
  - EV/EBITDA lives in both: ratios uses "enterpriseValueMultiple",
    key-metrics uses "evToEBITDA".
  - Historical price requires ?from=DATE&to=DATE (no more ?timeseries=N).
  - Sector performance requires an explicit ?date=YYYY-MM-DD.
  - Analyst grades replace the old analyst-stock-recommendations endpoint.
  - Price-target-consensus replaces the individual price-target endpoint.

Stable endpoints used
---------------------
  /profile                          company overview
  /quote                            real-time price + market cap
  /income-statement                 annual & quarterly income statements
  /balance-sheet-statement          annual balance sheets
  /cash-flow-statement              annual & quarterly cash flows
  /ratios                           annual financial ratios
  /ratios-ttm                       TTM fallback for valuation metrics
  /key-metrics                      annual operational key metrics
  /key-metrics-ttm                  TTM key metrics fallback
  /historical-price-eod/full        full OHLCV price history
  /historical/earning-calendar      per-company earnings history
  /earnings-surprises               EPS actuals vs estimates
  /grades                           individual analyst grade changes
  /price-target-consensus           aggregated analyst price target
  /sector-performance-snapshot      sector performance for a given date
"""
from __future__ import annotations

import datetime
import re
import time
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import Config
from models.stock_data import (
    BalanceSheet,
    CashFlowStatement,
    CompanyProfile,
    FinancialRatios,
    IncomeStatement,
    PriceHistory,
    StockData,
)
from utils.helpers import safe_float, safe_get
from utils.logger import logger


class FMPError(Exception):
    """Raised when the FMP API returns an error or unexpected response."""


# Seconds to sleep between every FMP request (rate-limit pacing).
_REQUEST_DELAY = 0.75

# Seconds to wait before the single 429 retry.
_RATE_LIMIT_WAIT = 2


class FMPClient:
    """
    Thread-safe FMP Stable API client with:
      - in-memory response cache (keyed by endpoint + params)
      - automatic retries with exponential back-off
      - response normalization into domain models
    """

    # Stable API base — all paths are relative to this prefix
    _STABLE_BASE: str = "https://financialmodelingprep.com/stable"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or Config.FMP_API_KEY
        self._cache: dict[str, Any] = {}
        self._session = self._build_session()
        self._call_log: list[str] = []
        # Always visible on startup so there's no ambiguity about which API is in use
        logger.info("FMPClient initialized — stable base: %s", self._STABLE_BASE)
        print(f"  [FMP] Stable API base: {self._STABLE_BASE}")

    # ── Session & HTTP plumbing ────────────────────────────────────────────────

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=Config.MAX_RETRIES,
            backoff_factor=Config.RETRY_BACKOFF,
            # 429 is handled manually in _get() with longer waits;
            # keep 5xx here for fast server-error retries.
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _get(self, path: str, params: Optional[dict] = None, timeout: Optional[int] = None) -> Any:
        """
        Authenticated GET against the stable base with caching.
        Prints every raw response so the pipeline is never opaque.
        Returns parsed JSON or raises FMPError.
        """
        params = params or {}
        params["apikey"] = self._api_key

        cache_key = path + str(sorted(params.items()))
        if cache_key in self._cache:
            logger.debug("FMP cache hit: %s", path)
            return self._cache[cache_key]

        url = f"{self._STABLE_BASE}{path}"
        display_params = {k: v for k, v in params.items() if k != "apikey"}
        logger.info("FMP stable request: %s  params=%s", path, display_params)
        self._call_log.append(path)

        request_timeout = timeout if timeout is not None else Config.REQUEST_TIMEOUT

        # Short delay between every request to avoid hammering the API.
        time.sleep(_REQUEST_DELAY)

        # Make request; if 429, wait 2s and retry once.
        for _attempt in range(2):
            try:
                resp = self._session.get(url, params=params, timeout=request_timeout)
            except requests.exceptions.RequestException as exc:
                raise FMPError(f"Request failed for {path}: {exc}") from exc

            if resp.status_code == 429:
                if _attempt == 0:
                    print(
                        f"  [FMP] 429 rate limit: {path}"
                        f" — waiting {_RATE_LIMIT_WAIT}s, retrying once"
                    )
                    logger.warning("FMP: 429 for %s — sleeping %ds, retrying", path, _RATE_LIMIT_WAIT)
                    time.sleep(_RATE_LIMIT_WAIT)
                    continue
                raise FMPError(f"HTTP 429 for {path} — rate limit after retry")

            try:
                resp.raise_for_status()
            except requests.exceptions.HTTPError as exc:
                raise FMPError(f"HTTP {exc.response.status_code} for {path}") from exc

            break  # successful response

        data = resp.json()

        # Always print raw response — critical for diagnosing empty / error results
        _data_len = len(data) if isinstance(data, (list, dict)) else "?"
        print(
            f"  [FMP] {path}"
            f" → {type(data).__name__}[{_data_len}]"
            f" | {str(data)[:300]}"
        )

        # FMP encodes errors as {"Error Message": "..."} in 200 responses
        if isinstance(data, dict) and "Error Message" in data:
            raise FMPError(f"FMP API error for {path}: {data['Error Message']}")

        if isinstance(data, list) and len(data) == 0:
            logger.warning("FMP returned empty list for %s — key may lack access or ticker has no data", path)

        self._cache[cache_key] = data
        return data

    @property
    def call_log(self) -> list[str]:
        return list(self._call_log)

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _first_valid(*raw_values: Any) -> Optional[float]:
        """Return safe_float of the first value that is not None after conversion."""
        for v in raw_values:
            result = safe_float(v)
            if result is not None:
                return result
        return None

    # ── Public fetch methods ───────────────────────────────────────────────────

    def fetch_profile(self, symbol: str) -> Optional[CompanyProfile]:
        data = self._get("/profile", {"symbol": symbol})
        if not data or not isinstance(data, list):
            logger.warning("fetch_profile [%s]: returned %s — no profile data", symbol, repr(data)[:120])
            return None
        raw = data[0]
        logger.debug("fetch_profile [%s]: company=%s sector=%s price=%s",
                     symbol, raw.get("companyName"), raw.get("sector"), raw.get("price"))
        return CompanyProfile(
            symbol=raw.get("symbol", symbol),
            company_name=raw.get("companyName", ""),
            sector=raw.get("sector", ""),
            industry=raw.get("industry", ""),
            description=raw.get("description", ""),
            exchange=raw.get("exchangeShortName", ""),
            country=raw.get("country", ""),
            # stable: mktCap and marketCap are both accepted
            market_cap=self._first_valid(raw.get("mktCap"), raw.get("marketCap")),
            price=safe_float(raw.get("price")),
            beta=safe_float(raw.get("beta")),
            volume_avg=int(raw["volAvg"]) if raw.get("volAvg") else None,
            employees=int(raw["fullTimeEmployees"]) if raw.get("fullTimeEmployees") else None,
            website=raw.get("website", ""),
            ceo=raw.get("ceo", ""),
            ipo_date=raw.get("ipoDate", ""),
            currency=(raw.get("currency") or "USD").strip().upper(),
        )

    def fetch_quote(self, symbol: str) -> dict[str, Any]:
        data = self._get("/quote", {"symbol": symbol})
        if not data or not isinstance(data, list):
            logger.warning("fetch_quote [%s]: returned %s — no quote data", symbol, repr(data)[:120])
            return {}
        q = data[0]
        logger.debug(
            "fetch_quote [%s]: price=%s marketCap=%s sharesOutstanding=%s",
            symbol, q.get("price"), q.get("marketCap"), q.get("sharesOutstanding"),
        )
        return q

    def fetch_income_statements(
        self, symbol: str, limit: int = 5, period: str = "annual"
    ) -> list[IncomeStatement]:
        data = self._get(
            "/income-statement",
            {"symbol": symbol, "limit": limit, "period": period},
        )
        if not isinstance(data, list):
            logger.warning("fetch_income_statements [%s]: unexpected response type %s",
                           symbol, type(data).__name__)
            return []
        if not data:
            logger.warning("fetch_income_statements [%s]: empty list returned", symbol)
        else:
            logger.debug("fetch_income_statements [%s]: %d rows, first date=%s revenue=%s",
                         symbol, len(data), data[0].get("date"), data[0].get("revenue"))
        results = []
        for raw in data:
            results.append(IncomeStatement(
                date=raw.get("date", ""),
                period=raw.get("period", "FY"),
                revenue=safe_float(raw.get("revenue")),
                gross_profit=safe_float(raw.get("grossProfit")),
                operating_income=safe_float(raw.get("operatingIncome")),
                net_income=safe_float(raw.get("netIncome")),
                ebitda=safe_float(raw.get("ebitda")),
                eps=safe_float(raw.get("eps")),
                # stable uses epsDiluted (camelCase); v3 used epsdiluted (all lower)
                eps_diluted=self._first_valid(raw.get("epsDiluted"), raw.get("epsdiluted")),
                gross_profit_ratio=safe_float(raw.get("grossProfitRatio")),
                operating_income_ratio=safe_float(raw.get("operatingIncomeRatio")),
                net_income_ratio=safe_float(raw.get("netIncomeRatio")),
                rd_expenses=safe_float(raw.get("researchAndDevelopmentExpenses")),
                selling_expenses=safe_float(raw.get("sellingGeneralAndAdministrativeExpenses")),
                interest_expense=safe_float(raw.get("interestExpense")),
                filing_date=raw.get("filingDate") or None,
                reported_currency=(raw.get("reportedCurrency") or "USD").strip().upper(),
            ))
        return results

    def fetch_balance_sheets(
        self, symbol: str, limit: int = 5, period: str = "annual"
    ) -> list[BalanceSheet]:
        data = self._get(
            "/balance-sheet-statement",
            {"symbol": symbol, "limit": limit, "period": period},
        )
        if not isinstance(data, list):
            return []
        results = []
        for raw in data:
            results.append(BalanceSheet(
                date=raw.get("date", ""),
                period=raw.get("period", "FY"),
                total_assets=safe_float(raw.get("totalAssets")),
                total_liabilities=safe_float(raw.get("totalLiabilities")),
                # stable has both totalStockholdersEquity and totalEquity
                total_equity=self._first_valid(
                    raw.get("totalStockholdersEquity"),
                    raw.get("totalEquity"),
                ),
                total_debt=safe_float(raw.get("totalDebt")),
                short_term_debt=safe_float(raw.get("shortTermDebt")),
                long_term_debt=safe_float(raw.get("longTermDebt")),
                cash_and_equivalents=safe_float(raw.get("cashAndCashEquivalents")),
                short_term_investments=safe_float(raw.get("shortTermInvestments")),
                total_current_assets=safe_float(raw.get("totalCurrentAssets")),
                total_current_liabilities=safe_float(raw.get("totalCurrentLiabilities")),
                goodwill=safe_float(raw.get("goodwill")),
                intangible_assets=safe_float(raw.get("intangibleAssets")),
                retained_earnings=safe_float(raw.get("retainedEarnings")),
            ))
        return results

    def fetch_cash_flows(
        self, symbol: str, limit: int = 5, period: str = "annual"
    ) -> list[CashFlowStatement]:
        data = self._get(
            "/cash-flow-statement",
            {"symbol": symbol, "limit": limit, "period": period},
        )
        if not isinstance(data, list):
            return []
        results = []
        for raw in data:
            results.append(CashFlowStatement(
                date=raw.get("date", ""),
                period=raw.get("period", "FY"),
                # stable has both operatingCashFlow and netCashProvidedByOperatingActivities
                operating_cash_flow=self._first_valid(
                    raw.get("operatingCashFlow"),
                    raw.get("netCashProvidedByOperatingActivities"),
                ),
                # stable has capitalExpenditure and investmentsInPropertyPlantAndEquipment
                capital_expenditure=self._first_valid(
                    raw.get("capitalExpenditure"),
                    raw.get("investmentsInPropertyPlantAndEquipment"),
                ),
                free_cash_flow=safe_float(raw.get("freeCashFlow")),
                net_income=safe_float(raw.get("netIncome")),
                depreciation_amortization=safe_float(raw.get("depreciationAndAmortization")),
                stock_based_compensation=safe_float(raw.get("stockBasedCompensation")),
                # stable: dividendsPaid split into commonDividendsPaid / netDividendsPaid
                dividends_paid=self._first_valid(
                    raw.get("commonDividendsPaid"),
                    raw.get("dividendsPaid"),
                    raw.get("netDividendsPaid"),
                ),
                acquisitions=safe_float(raw.get("acquisitionsNet")),
                common_stock_repurchased=safe_float(raw.get("commonStockRepurchased")),
            ))
        return results

    def fetch_ratios(
        self, symbol: str, limit: int = 5
    ) -> list[FinancialRatios]:
        """
        Fetch annual ratios with layered fallback for stable API.

        Stable field name changes from v3
        ----------------------------------
          priceEarningsRatio      → priceToEarningsRatio   (in /ratios)
          enterpriseValueOverEBITDA → enterpriseValueMultiple  (in /ratios)
          debtEquityRatio         → debtToEquityRatio
          totalDebtToAssetsRatio  → debtToAssetsRatio
          interestCoverage        → interestCoverageRatio
          payoutRatio             → dividendPayoutRatio
          returnOnEquity/ROA/ROIC → moved to /key-metrics (removed from /ratios)

        Source priority for each field:
          1. /ratios           — annual ratios (stable field names)
          2. /ratios-ttm       — TTM ratios (same field names, TTM values)
          3. /key-metrics      — ROE, ROA, ROIC, EV/EBITDA (annual)
          4. /key-metrics-ttm  — ROE, ROA, ROIC, EV/EBITDA (TTM fallback)

        If annual ratios return empty, synthesize a TTM stub row so that
        latest_ratios is never None when TTM data is available.
        """
        annual_raw = self._get("/ratios", {"symbol": symbol, "limit": limit, "period": "annual"})
        annual: list[dict] = annual_raw if isinstance(annual_raw, list) else []
        if not annual:
            logger.warning("fetch_ratios [%s]: /ratios returned empty — attempting TTM fallback", symbol)

        # ── Source 2: ratios TTM ───────────────────────────────────────────────
        # Same field names as /ratios but TTM values; no "TTM" suffix on fields.
        ttm_raw = self._get("/ratios-ttm", {"symbol": symbol})
        ttm: dict[str, Any] = {}
        if isinstance(ttm_raw, list) and ttm_raw:
            ttm = ttm_raw[0]
        elif isinstance(ttm_raw, dict):
            ttm = ttm_raw

        # ── Source 3: key-metrics annual ──────────────────────────────────────
        km_raw = self._get("/key-metrics", {"symbol": symbol, "limit": limit, "period": "annual"})
        km_list: list[dict] = km_raw if isinstance(km_raw, list) else []
        km_by_date: dict[str, dict] = {r.get("date", ""): r for r in km_list}

        # ── Source 4: key-metrics TTM ─────────────────────────────────────────
        km_ttm_raw = self._get("/key-metrics-ttm", {"symbol": symbol})
        km_ttm: dict[str, Any] = {}
        if isinstance(km_ttm_raw, list) and km_ttm_raw:
            km_ttm = km_ttm_raw[0]
        elif isinstance(km_ttm_raw, dict):
            km_ttm = km_ttm_raw

        # Synthesize a TTM stub row when annual ratios are absent
        if not annual and (ttm or km_list or km_ttm):
            ttm_date = (
                ttm.get("date")
                or km_ttm.get("date")
                or (km_list[0].get("date") if km_list else "TTM")
            )
            logger.warning(
                "fetch_ratios [%s]: annual ratios empty — synthesizing TTM row (date=%s)",
                symbol, ttm_date,
            )
            annual = [{"date": ttm_date, "period": "TTM"}]

        results = []
        for i, raw in enumerate(annual):
            date = raw.get("date", "")
            is_most_recent = (i == 0)

            # Key-metrics: date match first, positional fallback for slight date skew
            km = km_by_date.get(date) or (km_list[i] if i < len(km_list) else {})

            # ── pe_ratio ──────────────────────────────────────────────────────
            # Stable: priceToEarningsRatio in /ratios and /ratios-ttm.
            # For the most recent row, PREFER TTM over annual:
            #   Annual P/E uses fiscal-year-end price × FY EPS (stale if stock
            #   has moved significantly since year-end).
            #   TTM P/E uses trailing 12-month EPS × a more current price.
            # For historical rows, only annual data is meaningful.
            pe_annual = raw.get("priceToEarningsRatio")
            pe_ttm    = ttm.get("priceToEarningsRatio") if is_most_recent else None
            if is_most_recent:
                pe_ratio = self._first_valid(pe_ttm, pe_annual)   # TTM first
            else:
                pe_ratio = self._first_valid(pe_annual)
            logger.debug("pe_ratio [%s %s] annual=%s  ttm=%s  → %s  (ttm_preferred=%s)",
                         symbol, date, pe_annual, pe_ttm, pe_ratio, is_most_recent)

            # ── ps_ratio ──────────────────────────────────────────────────────
            ps_annual = raw.get("priceToSalesRatio")
            ps_ttm    = ttm.get("priceToSalesRatio") if is_most_recent else None
            if is_most_recent:
                ps_ratio = self._first_valid(ps_ttm, ps_annual)   # TTM first
            else:
                ps_ratio = self._first_valid(ps_annual)
            logger.debug("ps_ratio [%s %s] annual=%s  ttm=%s  → %s  (ttm_preferred=%s)",
                         symbol, date, ps_annual, ps_ttm, ps_ratio, is_most_recent)

            # ── ev_to_ebitda ──────────────────────────────────────────────────
            ev_annual = raw.get("enterpriseValueMultiple")
            ev_ttm    = ttm.get("enterpriseValueMultiple") if is_most_recent else None
            ev_km     = km.get("evToEBITDA")
            ev_km_ttm = km_ttm.get("evToEBITDA") if is_most_recent else None
            if is_most_recent:
                ev_to_ebitda = self._first_valid(ev_ttm, ev_km_ttm, ev_annual, ev_km)
            else:
                ev_to_ebitda = self._first_valid(ev_annual, ev_km)
            logger.debug("ev_to_ebitda [%s %s] annual=%s  ttm=%s  km=%s  km_ttm=%s  → %s  (ttm_preferred=%s)",
                         symbol, date, ev_annual, ev_ttm, ev_km, ev_km_ttm, ev_to_ebitda, is_most_recent)

            results.append(FinancialRatios(
                date=date,
                period=raw.get("period", "FY"),
                pe_ratio=pe_ratio,
                pb_ratio=self._first_valid(
                    raw.get("priceToBookRatio"),
                    ttm.get("priceToBookRatio") if is_most_recent else None,
                ),
                ps_ratio=ps_ratio,
                ev_to_ebitda=ev_to_ebitda,
                # EV/revenue moved to key-metrics in stable as "evToSales"
                ev_to_revenue=self._first_valid(
                    km.get("evToSales"),
                    km_ttm.get("evToSales") if is_most_recent else None,
                ),
                # ROE / ROA / ROIC removed from /ratios in stable — now in /key-metrics
                roe=self._first_valid(
                    km.get("returnOnEquity"),
                    km_ttm.get("returnOnEquity") if is_most_recent else None,
                ),
                roa=self._first_valid(
                    km.get("returnOnAssets"),
                    km_ttm.get("returnOnAssets") if is_most_recent else None,
                ),
                roic=self._first_valid(
                    km.get("returnOnInvestedCapital"),
                    km_ttm.get("returnOnInvestedCapital") if is_most_recent else None,
                    km.get("returnOnCapitalEmployed"),
                    km_ttm.get("returnOnCapitalEmployed") if is_most_recent else None,
                ),
                # Stable renamed debtEquityRatio → debtToEquityRatio
                debt_to_equity=self._first_valid(
                    raw.get("debtToEquityRatio"),
                    km.get("debtToEquity"),
                ),
                # Stable renamed totalDebtToAssetsRatio → debtToAssetsRatio
                debt_to_assets=self._first_valid(
                    raw.get("debtToAssetsRatio"),
                    km.get("debtToAssets"),
                ),
                current_ratio=self._first_valid(
                    raw.get("currentRatio"),
                    km.get("currentRatio"),
                ),
                quick_ratio=safe_float(raw.get("quickRatio")),
                # Stable renamed interestCoverage → interestCoverageRatio
                interest_coverage=self._first_valid(
                    raw.get("interestCoverageRatio"),
                    raw.get("debtServiceCoverageRatio"),
                ),
                gross_margin=safe_float(raw.get("grossProfitMargin")),
                operating_margin=safe_float(raw.get("operatingProfitMargin")),
                net_margin=safe_float(raw.get("netProfitMargin")),
                fcf_yield=self._first_valid(
                    km.get("freeCashFlowYield"),
                    km_ttm.get("freeCashFlowYield") if is_most_recent else None,
                ),
                dividend_yield=self._first_valid(
                    raw.get("dividendYield"),
                    raw.get("dividendYieldPercentage"),
                ),
                # Stable renamed payoutRatio → dividendPayoutRatio
                payout_ratio=self._first_valid(
                    raw.get("dividendPayoutRatio"),
                    raw.get("payoutRatio"),
                ),
            ))
        return results

    def fetch_key_metrics(self, symbol: str, limit: int = 5) -> list[dict[str, Any]]:
        data = self._get("/key-metrics", {"symbol": symbol, "limit": limit, "period": "annual"})
        return data if isinstance(data, list) else []

    def fetch_price_history(
        self, symbol: str, timeseries: int = 400
    ) -> Optional[PriceHistory]:
        # Stable uses ?from=DATE&to=DATE instead of ?timeseries=N.
        # Multiply trading-day count by 1.5 to convert to calendar days
        # (accounts for weekends + ~10 holidays per year).
        today = datetime.date.today()
        from_date = today - datetime.timedelta(days=int(timeseries * 1.5))
        data = self._get(
            "/historical-price-eod/full",
            {"symbol": symbol, "from": str(from_date), "to": str(today)},
        )
        # Stable /historical-price-eod/full returns a flat list directly (no wrapper dict).
        # Fall back to safe_get("historical") in case the format ever reverts.
        if isinstance(data, list):
            historicals = data
        else:
            historicals = safe_get(data, "historical")
        if not historicals or not isinstance(historicals, list):
            logger.warning("fetch_price_history [%s]: unexpected response shape — type=%s",
                           symbol, type(data).__name__)
            return None

        # FMP returns newest first
        ph = PriceHistory()
        for bar in historicals:
            ph.dates.append(bar.get("date", ""))
            ph.closes.append(safe_float(bar.get("close"), 0.0))
            ph.highs.append(safe_float(bar.get("high"), 0.0))
            ph.lows.append(safe_float(bar.get("low"), 0.0))
            ph.volumes.append(safe_float(bar.get("volume"), 0.0))
        return ph

    def fetch_earnings(self, symbol: str) -> list[dict[str, Any]]:
        # Stable equivalent: /historical/earning-calendar
        # Note: field names changed — date, epsEstimated, eps, revenueEstimated, revenue
        data = self._get("/historical/earning-calendar", {"symbol": symbol})
        if not isinstance(data, list):
            logger.warning(
                "fetch_earnings [%s]: /historical/earning-calendar returned %s — no earnings data",
                symbol, type(data).__name__,
            )
            return []
        return data

    def fetch_earnings_surprises(self, symbol: str) -> list[dict[str, Any]]:
        # Stable field name changes:
        #   "actual"    (v3) → "actualEarningResult"  (stable)
        #   "estimated" (v3) → "estimatedEarning"     (stable)
        data = self._get("/earnings-surprises", {"symbol": symbol})
        return data if isinstance(data, list) else []

    def fetch_analyst_recommendations(self, symbol: str) -> list[dict[str, Any]]:
        # Stable equivalent: /grades-consensus — returns aggregate buy/hold/sell counts.
        # Fields: symbol, consensus, strongBuy, buy, hold, sell, strongSell
        # NOTE: v3 used /analyst-stock-recommendations with "analystRatingsStrongBuy" etc.
        #       Stable uses shorter field names: strongBuy, buy, hold, sell, strongSell.
        #       The caller (MarketAnalystAgent) reads these names directly from recs[0].
        data = self._get("/grades-consensus", {"symbol": symbol})
        if isinstance(data, list) and data:
            return data
        if isinstance(data, dict) and data:
            return [data]   # wrap single object so callers can always do recs[0]
        logger.warning(
            "fetch_analyst_recommendations [%s]: /grades-consensus returned %s — no analyst data",
            symbol, type(data).__name__,
        )
        return []

    def fetch_price_targets(self, symbol: str) -> dict[str, Any]:
        # Stable equivalent: /price-target-consensus
        # Returns: {symbol, targetHigh, targetLow, targetConsensus, targetMedian}
        data = self._get("/price-target-consensus", {"symbol": symbol})
        if isinstance(data, list) and data:
            return data[0]
        if isinstance(data, dict) and data:
            return data
        return {}

    def fetch_sector_performance(self) -> list[dict[str, Any]]:
        # Stable requires an explicit date. Try today then walk back up to 5 days
        # to skip weekends and market holidays. Use a short timeout (8 s) because
        # this endpoint can hang; sector data is context-only and not worth waiting for.
        today = datetime.date.today()
        for days_back in range(6):
            candidate = today - datetime.timedelta(days=days_back)
            candidate_str = candidate.isoformat()
            try:
                data = self._get(
                    "/sector-performance-snapshot",
                    {"date": candidate_str},
                    timeout=8,
                )
            except FMPError as exc:
                logger.warning("fetch_sector_performance: error for date=%s — %s", candidate_str, exc)
                return []
            if not isinstance(data, list):
                logger.warning(
                    "fetch_sector_performance: unexpected response type %s for date=%s",
                    type(data).__name__, candidate_str,
                )
                return []
            if data:
                return data
            logger.debug("fetch_sector_performance: empty for date=%s — trying previous day", candidate_str)
        logger.warning("fetch_sector_performance: no sector data found in last 5 trading days")
        return []

    def fetch_peers(self, symbol: str) -> list[str]:
        """
        Return peer ticker symbols from FMP /stock-peers.
        FMP curates same-sector, same-industry peers for each symbol.
        Returns an empty list on any error so callers can fall back silently.
        """
        try:
            data = self._get("/stock-peers", {"symbol": symbol})
        except FMPError:
            return []
        if not data or not isinstance(data, list):
            return []
        first = data[0] if data else {}
        # Legacy FMP API: [{peersList: ["AAPL", ...]}, ...]
        # Stable FMP API: [{symbol: "AAPL", companyName: ..., price: ..., mktCap: ...}, ...]
        if isinstance(first, dict) and "peersList" in first:
            peers = first.get("peersList", [])
            return [str(t) for t in peers if t and t != symbol]
        return [
            str(r["symbol"]) for r in data
            if isinstance(r, dict) and r.get("symbol") and r["symbol"] != symbol
        ]

    def fetch_shares_float(self, symbol: str) -> Optional[dict[str, Any]]:
        """
        Return SEC-sourced share count via FMP's /shares-float endpoint.

        FMP sources this data directly from SEC EDGAR filings and includes
        a direct filing URL, making every derived value fully auditable.

        Returns dict with:
          {
            "shares":             float,        # outstandingShares from SEC filing
            "source":             str,           # "FMP/shares-float (SEC EDGAR)"
            "filing_url":         str,           # Direct SEC EDGAR URL
            "filing_period_end":  str | None,    # Fiscal period end, e.g. "2025-12-31"
                                                 # (parsed from filing_url path)
            "data_refreshed_at":  str,           # When FMP last refreshed this record
                                                 # (NOT the SEC filing date)
            "fetched_at":         str,           # ISO timestamp of this fetch
          }

        Returns None on miss or if outstandingShares is absent.
        """
        data = self._get("/shares-float", {"symbol": symbol})

        record: Optional[dict[str, Any]] = None
        if isinstance(data, dict):
            record = data
        elif isinstance(data, list) and data:
            record = data[0]

        if not record:
            return None

        shares = safe_float(record.get("outstandingShares"))
        if shares is None:
            return None

        # FMP stores the SEC EDGAR document URL in the "source" field.
        filing_url = str(record.get("source", ""))

        # Parse fiscal period end from the URL path (e.g. "pypl-20251231.htm" → "2025-12-31").
        _m = re.search(r"(\d{4})(\d{2})(\d{2})\.htm", filing_url)
        filing_period_end: Optional[str] = (
            f"{_m.group(1)}-{_m.group(2)}-{_m.group(3)}" if _m else None
        )

        return {
            "shares":            shares,
            "source":            "FMP/shares-float (SEC EDGAR)",
            "filing_url":        filing_url,
            "filing_period_end": filing_period_end,
            # FMP's "date" is their data-refresh timestamp, not the SEC filing date.
            "data_refreshed_at": str(record.get("date", "")),
            "fetched_at":        datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    def fetch_screener(
        self,
        sector: str = "",
        industry: str = "",
        min_mkt_cap: Optional[float] = None,
        max_mkt_cap: Optional[float] = None,
        country: str = "US",
        limit: int = 10,
    ) -> list[str]:
        """
        Return ticker symbols from FMP /stock-screener filtered by sector/industry/mktcap.
        Returns [] on any error or if the endpoint is gated.
        """
        params: dict[str, Any] = {"country": country, "limit": limit}
        if sector:
            params["sector"] = sector
        if industry:
            params["industry"] = industry
        if min_mkt_cap is not None:
            params["marketCapMoreThan"] = int(min_mkt_cap)
        if max_mkt_cap is not None:
            params["marketCapLowerThan"] = int(max_mkt_cap)
        try:
            data = self._get("/stock-screener", params)
        except FMPError:
            return []
        if not data or not isinstance(data, list):
            return []
        return [str(r["symbol"]) for r in data if isinstance(r, dict) and r.get("symbol")]

    # ── Bulk convenience method ────────────────────────────────────────────────

    def fetch_all_financials(self, symbol: str) -> StockData:
        """
        Convenience: populate a StockData with all commonly needed data.
        Use for quick one-shot fetches outside the agentic loop.
        """
        sd = StockData(ticker=symbol)
        limit = Config.FINANCIAL_STATEMENT_LIMIT
        sd.profile = self.fetch_profile(symbol)
        quote = self.fetch_quote(symbol)
        sd.current_price = safe_float(quote.get("price"))
        sd.market_cap = safe_float(quote.get("marketCap"))
        _ts = quote.get("timestamp")
        if _ts:
            try:
                sd.quote_date = datetime.datetime.utcfromtimestamp(int(_ts)).strftime("%Y-%m-%d")
            except Exception:
                pass
        sd.income_statements = self.fetch_income_statements(symbol, limit)
        sd.quarterly_income = self.fetch_income_statements(symbol, 4, "quarter")
        sd.balance_sheets = self.fetch_balance_sheets(symbol, limit)
        sd.cash_flows = self.fetch_cash_flows(symbol, limit)
        sd.ratios = self.fetch_ratios(symbol, limit)
        sd.price_history = self.fetch_price_history(symbol, Config.PRICE_HISTORY_DAYS)
        sd.earnings = self.fetch_earnings(symbol)
        sd.earnings_surprises = self.fetch_earnings_surprises(symbol)
        sd.analyst_recommendations = self.fetch_analyst_recommendations(symbol)
        sd.price_targets = self.fetch_price_targets(symbol)
        return sd
