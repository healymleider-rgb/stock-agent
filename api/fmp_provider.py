"""
FMPProvider — DataProvider adapter wrapping the existing FMPClient.

FMP is the primary (required) provider.
This class is a thin pass-through so FMPClient stays unchanged.
"""
from __future__ import annotations

from typing import Any, Optional

from api.fmp_client import FMPClient, FMPError
from api.provider_base import AnalystResult, DataProvider, FinancialsResult, ProfileResult
from utils.logger import logger
from models.stock_data import IncomeStatement, PriceHistory
from utils.helpers import safe_float


class FMPProvider(DataProvider):
    """Delegates every call directly to the existing FMPClient."""

    name = "FMP"

    def __init__(self) -> None:
        self._client = FMPClient()

    # is_available() inherits True — FMP is always required.

    def get_profile(self, ticker: str) -> ProfileResult:
        profile = self._client.fetch_profile(ticker)

        # /quote may be gated on lower-tier FMP plans (HTTP 402).
        # /profile already includes price and mktCap — use those as fallback
        # so profile data is never lost just because the quote endpoint fails.
        try:
            quote = self._client.fetch_quote(ticker)
        except FMPError as exc:
            logger.warning(
                "FMPProvider [%s]: /quote failed (%s) — using profile.price / "
                "profile.market_cap as fallback",
                ticker, exc,
            )
            print(
                f"  [FMP] /quote unavailable for {ticker} ({exc})"
                f" — falling back to profile price/mktcap"
            )
            quote = {}

        # Prefer live quote values; fall back to profile fields when quote is empty.
        current_price = safe_float(quote.get("price"))
        if current_price is None and profile is not None:
            current_price = profile.price

        market_cap = safe_float(quote.get("marketCap"))
        if market_cap is None and profile is not None:
            market_cap = profile.market_cap

        # Extract shares outstanding and compute cross-check market cap.
        # FMP /quote includes sharesOutstanding (shares × current price = market cap).
        shares_outstanding = safe_float(quote.get("sharesOutstanding"))
        market_cap_computed: Optional[float] = None
        if shares_outstanding and current_price and current_price > 0:
            market_cap_computed = round(shares_outstanding * current_price, 0)

        # Log raw inputs and flag if API market cap differs materially from computed.
        ticker_sym = profile.symbol if profile else "?"
        print(
            f"  [FMP PROFILE AUDIT] {ticker_sym}: "
            f"price={current_price}  "
            f"shares={shares_outstanding}  "
            f"mktcap_api={market_cap}  "
            f"mktcap_computed={market_cap_computed}"
        )
        if market_cap and market_cap_computed and market_cap_computed > 0:
            discrepancy = abs(market_cap - market_cap_computed) / market_cap_computed
            if discrepancy > 0.10:
                logger.warning(
                    "FMPProvider [%s]: market cap discrepancy %.1f%% "
                    "(api=%s  computed=%s) — using api value",
                    ticker_sym, discrepancy * 100, market_cap, market_cap_computed,
                )
                print(
                    f"  [FMP PROFILE WARN] {ticker_sym}: market cap discrepancy "
                    f"{discrepancy:.1%} (api={market_cap:.0f}  computed={market_cap_computed:.0f})"
                )

        return ProfileResult(
            profile=profile,
            current_price=current_price,
            market_cap=market_cap,
            shares_outstanding=shares_outstanding,
            market_cap_computed=market_cap_computed,
            quote=quote,
        )

    def get_financials(self, ticker: str, limit: int = 5) -> FinancialsResult:
        return FinancialsResult(
            income_statements=self._client.fetch_income_statements(ticker, limit),
            balance_sheets=self._client.fetch_balance_sheets(ticker, limit),
            cash_flows=self._client.fetch_cash_flows(ticker, limit),
            ratios=self._client.fetch_ratios(ticker, limit),
        )

    def get_quarterly(self, ticker: str) -> list[IncomeStatement]:
        return self._client.fetch_income_statements(ticker, 4, "quarter")

    def get_price_history(self, ticker: str, days: int = 400) -> Optional[PriceHistory]:
        return self._client.fetch_price_history(ticker, days)

    def get_earnings(self, ticker: str) -> tuple[list[dict], list[dict]]:
        return (
            self._client.fetch_earnings(ticker),
            self._client.fetch_earnings_surprises(ticker),
        )

    def get_peers(self, ticker: str) -> list[str]:
        """Return FMP-curated peer tickers (same sector/industry)."""
        try:
            return self._client.fetch_peers(ticker)
        except FMPError:
            return []

    def get_screener(
        self,
        sector: str = "",
        industry: str = "",
        min_mkt_cap: Optional[float] = None,
        max_mkt_cap: Optional[float] = None,
        limit: int = 10,
    ) -> list[str]:
        """Return tickers from FMP /stock-screener matching sector/industry/size."""
        return self._client.fetch_screener(
            sector=sector,
            industry=industry,
            min_mkt_cap=min_mkt_cap,
            max_mkt_cap=max_mkt_cap,
            limit=limit,
        )

    def get_analyst_data(self, ticker: str) -> AnalystResult:
        return AnalystResult(
            recommendations=self._client.fetch_analyst_recommendations(ticker),
            price_targets=self._client.fetch_price_targets(ticker),
        )

    def get_sector_performance(self) -> list[dict[str, Any]]:
        return self._client.fetch_sector_performance()

    @property
    def call_log(self) -> list[str]:
        return self._client.call_log
