"""
YChartsProvider — optional fallback data provider scaffold.

CURRENT STATE
-------------
All methods return empty results. This provider is disabled by default.
It activates only when YCHARTS_SESSION_TOKEN is set in the environment.
The DataBroker silently skips it when is_available() returns False.

WHY THIS IS A SCAFFOLD
----------------------
YCharts (ycharts.com) does not expose a documented public REST API.
Integration requires one of the following approaches:

  Option A — Session token (browser-based)
    1. Log in to ycharts.com in a browser
    2. Extract the session cookie or bearer token from DevTools
    3. Set YCHARTS_SESSION_TOKEN=<token> in your .env file
    WARNING: Session tokens expire. This approach requires manual renewal.
    NOTE: Scraping is against YCharts' Terms of Service — use only if you
    have explicit written permission or a data license agreement.

  Option B — Official data API (if released)
    If YCharts releases an official API, implement using an API key stored
    in YCHARTS_API_KEY. Update is_available() and the constructor accordingly.

  Option C — Exported CSV files
    YCharts supports bulk CSV exports for subscribers. If you place exported
    files in a configured directory (YCHARTS_DATA_DIR), implement get_*
    methods to parse them instead of making HTTP requests.

HOW TO IMPLEMENT
----------------
1. Set YCHARTS_SESSION_TOKEN (or YCHARTS_API_KEY) in .env
2. Implement the individual get_* methods below using self._session
3. No other changes required — DataBroker picks it up automatically

WHAT YCHARTS COULD FILL
-----------------------
Fields that FMP sometimes misses and YCharts typically provides:
  - get_financials: longer financial history (10+ years)
  - get_ratios: ROIC, EV/EBITDA for smaller/international tickers
  - get_analyst_data: analyst consensus aggregates
  - get_price_history: extended price history for delisted / older tickers
"""
from __future__ import annotations

from typing import Any, Optional

from api.provider_base import AnalystResult, DataProvider, FinancialsResult, ProfileResult
from config import Config
from models.stock_data import IncomeStatement, PriceHistory
from utils.logger import logger


class YChartsProvider(DataProvider):
    """
    YCharts fallback provider.
    Returns empty results for all methods until implemented.
    """

    name = "YCharts"

    def __init__(self) -> None:
        # TODO: initialise a requests.Session here with the session token
        # as an Authorization header or cookie once implemented.
        #
        # Example:
        #   import requests
        #   self._session = requests.Session()
        #   self._session.headers["Authorization"] = f"Bearer {Config.YCHARTS_SESSION_TOKEN}"
        pass

    def is_available(self) -> bool:
        available = bool(Config.YCHARTS_SESSION_TOKEN)
        if not available:
            logger.debug("YChartsProvider: YCHARTS_SESSION_TOKEN not set — provider disabled")
        return available

    # ── Scaffold methods ───────────────────────────────────────────────────────
    # Each docstring describes what data YCharts could supply.

    def get_profile(self, ticker: str) -> ProfileResult:
        # TODO: hit /companies/<ticker>/ to retrieve sector, description,
        #       market cap, employee count.
        logger.debug("YChartsProvider.get_profile [%s]: not implemented", ticker)
        return ProfileResult()

    def get_financials(self, ticker: str, limit: int = 5) -> FinancialsResult:
        # TODO: retrieve income, balance, and cash-flow statements.
        #       YCharts is particularly useful for ROIC and extended history.
        logger.debug("YChartsProvider.get_financials [%s]: not implemented", ticker)
        return FinancialsResult()

    def get_quarterly(self, ticker: str) -> list[IncomeStatement]:
        logger.debug("YChartsProvider.get_quarterly [%s]: not implemented", ticker)
        return []

    def get_price_history(self, ticker: str, days: int = 400) -> Optional[PriceHistory]:
        # TODO: retrieve OHLCV data, useful for extended or delisted tickers.
        logger.debug("YChartsProvider.get_price_history [%s]: not implemented", ticker)
        return None

    def get_earnings(self, ticker: str) -> tuple[list[dict], list[dict]]:
        logger.debug("YChartsProvider.get_earnings [%s]: not implemented", ticker)
        return [], []

    def get_analyst_data(self, ticker: str) -> AnalystResult:
        # TODO: YCharts aggregates analyst ratings and consensus targets.
        logger.debug("YChartsProvider.get_analyst_data [%s]: not implemented", ticker)
        return AnalystResult()

    def get_sector_performance(self) -> list[dict[str, Any]]:
        logger.debug("YChartsProvider.get_sector_performance: not implemented")
        return []
