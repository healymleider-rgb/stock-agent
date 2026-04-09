"""
AlphaSpreadProvider — optional fallback data provider scaffold.

CURRENT STATE
-------------
All methods return empty results. This provider is disabled by default.
It activates only when ALPHASPREAD_API_KEY is set in the environment.
The DataBroker silently skips it when is_available() returns False.

WHY THIS IS A SCAFFOLD
----------------------
Alpha Spread (alphaspread.com) focuses on intrinsic value and DCF models.
As of this writing, they do not offer a documented public REST API.
Integration options:

  Option A — Official API (if released)
    Set ALPHASPREAD_API_KEY=<key> in your .env file.
    Alpha Spread has hinted at an API roadmap for institutional subscribers.
    This adapter is pre-wired to use that key once available.

  Option B — Session-based access (browser)
    Similar to YCharts Option A: extract a session token from DevTools.
    Set ALPHASPREAD_SESSION_TOKEN in .env and implement using self._session.
    WARNING: Terms of Service restrictions apply. Verify before scraping.

  Option C — CSV data export
    Alpha Spread supports data exports for subscribers. Parse those files
    here if placed in a configured directory (ALPHASPREAD_DATA_DIR).

HOW TO IMPLEMENT
----------------
1. Set ALPHASPREAD_API_KEY (or ALPHASPREAD_SESSION_TOKEN) in .env
2. Update is_available() to check the relevant env var
3. Implement the individual get_* methods below
4. No other changes required — DataBroker picks it up automatically

WHAT ALPHA SPREAD COULD FILL
-----------------------------
Alpha Spread specialises in intrinsic value and DCF-derived metrics:
  - get_financials: normalized financials with 10-year history
  - get_ratios: WACC, intrinsic value estimates, DCF-derived fair value
  - get_analyst_data: consensus price targets aggregated across brokers
"""
from __future__ import annotations

from typing import Any, Optional

from api.provider_base import AnalystResult, DataProvider, FinancialsResult, ProfileResult
from config import Config
from models.stock_data import IncomeStatement, PriceHistory
from utils.logger import logger


class AlphaSpreadProvider(DataProvider):
    """
    Alpha Spread fallback provider.
    Returns empty results for all methods until implemented.
    """

    name = "AlphaSpread"

    def __init__(self) -> None:
        # TODO: initialise a requests.Session with API key or session token
        # once a stable programmatic interface is available.
        #
        # Example (API key approach):
        #   import requests
        #   self._session = requests.Session()
        #   self._session.headers["X-Api-Key"] = Config.ALPHASPREAD_API_KEY
        pass

    def is_available(self) -> bool:
        available = bool(Config.ALPHASPREAD_API_KEY)
        if not available:
            logger.debug("AlphaSpreadProvider: ALPHASPREAD_API_KEY not set — provider disabled")
        return available

    # ── Scaffold methods ───────────────────────────────────────────────────────

    def get_profile(self, ticker: str) -> ProfileResult:
        # TODO: retrieve company overview data.
        logger.debug("AlphaSpreadProvider.get_profile [%s]: not implemented", ticker)
        return ProfileResult()

    def get_financials(self, ticker: str, limit: int = 5) -> FinancialsResult:
        # TODO: retrieve normalized financials.
        #       Alpha Spread normalizes statements across GAAP / IFRS,
        #       making it useful for international tickers FMP sometimes misses.
        logger.debug("AlphaSpreadProvider.get_financials [%s]: not implemented", ticker)
        return FinancialsResult()

    def get_quarterly(self, ticker: str) -> list[IncomeStatement]:
        logger.debug("AlphaSpreadProvider.get_quarterly [%s]: not implemented", ticker)
        return []

    def get_price_history(self, ticker: str, days: int = 400) -> Optional[PriceHistory]:
        logger.debug("AlphaSpreadProvider.get_price_history [%s]: not implemented", ticker)
        return None

    def get_earnings(self, ticker: str) -> tuple[list[dict], list[dict]]:
        logger.debug("AlphaSpreadProvider.get_earnings [%s]: not implemented", ticker)
        return [], []

    def get_analyst_data(self, ticker: str) -> AnalystResult:
        # TODO: Alpha Spread aggregates analyst consensus targets.
        #       Could supplement FMP's grades-consensus data.
        logger.debug("AlphaSpreadProvider.get_analyst_data [%s]: not implemented", ticker)
        return AnalystResult()

    def get_sector_performance(self) -> list[dict[str, Any]]:
        logger.debug("AlphaSpreadProvider.get_sector_performance: not implemented")
        return []
