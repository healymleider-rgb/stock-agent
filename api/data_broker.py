"""
DataBroker — multi-provider waterfall with field-level merge.

Priority order
--------------
  1. FMP           (always required, primary source)
  2. AlphaVantage  (optional fallback — enabled via ALPHA_VANTAGE_API_KEY)
  3. YCharts       (optional fallback — enabled via YCHARTS_SESSION_TOKEN)
  4. AlphaSpread   (optional fallback — enabled via ALPHASPREAD_API_KEY)

Merge behaviour
---------------
Dataset level:
  If FMP returns an empty/None result for a data type (e.g. no income
  statements), the broker tries each fallback provider in order until one
  returns data. The first non-empty result is used and its provider name
  is recorded in the returned source string.

  Critically: if FMP raises FMPError (e.g. HTTP 402 access-restricted),
  the broker catches it, logs it, and proceeds to the fallback loop
  exactly as if FMP returned empty data. The error does NOT propagate
  to the DataRetrievalAgent — doing so would bypass all fallbacks.

Field level (within FinancialsResult):
  Even when FMP returns income/balance/cashflow/ratios, individual fields
  in those objects may be None (FMP doesn't cover everything).
  After the FMP fetch, each fallback provider is asked for the same data.
  Any None field in the FMP result is filled in-place from the fallback
  if the fallback has a non-None value for that field.
  Field-level fills are logged individually and recorded in field_sources.

FMP always wins:
  A valid (non-None) FMP value is NEVER replaced by a fallback value.
  Only None fields are eligible for fallback filling.

Return format
-------------
Every public method returns a 3-tuple:
  (result, dataset_source, field_sources)

  result         — typed result container (ProfileResult, FinancialsResult …)
  dataset_source — name of the provider that supplied the core dataset
  field_sources  — {prefix.field_name: provider_name} for field-level fills
                   Empty dict when all data came from the primary source.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional, TypeVar

from api.alpha_vantage_provider import AlphaVantageProvider
from api.alphaspread_provider import AlphaSpreadProvider
from api.fmp_client import FMPError
from api.fmp_provider import FMPProvider
from api.provider_base import AnalystResult, DataProvider, FinancialsResult, ProfileResult
from api.ycharts_provider import YChartsProvider
from models.stock_data import IncomeStatement, PriceHistory
from utils.logger import logger

T = TypeVar("T")


# ── Field-level merge helper ───────────────────────────────────────────────────

def _merge_dataclass_fields(
    primary: Any,
    secondary: Any,
    secondary_name: str,
    field_prefix: str,
) -> dict[str, str]:
    """
    Fill None (or empty-string) fields in *primary* from *secondary* in-place.
    Fields named 'date' and 'period' are never overwritten.

    Returns a dict mapping field_prefix.field_name → secondary_name
    for every field that was filled.  Empty dict if nothing was filled.

    This function works with any dataclass — IncomeStatement, BalanceSheet,
    CashFlowStatement, FinancialRatios, etc.
    """
    filled: dict[str, str] = {}
    for f in dataclasses.fields(primary):
        if f.name in ("date", "period"):
            continue
        primary_val = getattr(primary, f.name)
        # Treat None and empty string as "missing"
        is_missing = primary_val is None or (isinstance(primary_val, str) and not primary_val)
        if is_missing:
            secondary_val = getattr(secondary, f.name, None)
            has_value = secondary_val is not None and not (
                isinstance(secondary_val, str) and not secondary_val
            )
            if has_value:
                setattr(primary, f.name, secondary_val)
                key = f"{field_prefix}.{f.name}"
                filled[key] = secondary_name
                logger.info(
                    "DataBroker: filled %s from %s (was None, now %s)",
                    key, secondary_name, secondary_val,
                )
    return filled


def _is_access_restricted(exc: FMPError) -> bool:
    """Return True if the FMPError indicates a plan/access restriction (402/403)."""
    msg = str(exc)
    return "402" in msg or "403" in msg


# ── DataBroker ─────────────────────────────────────────────────────────────────

class DataBroker:
    """
    Orchestrates the FMP-first waterfall and optional field-level fallback fills.

    Usage (same call sites as FMPClient but returns 3-tuples):
        result, source, field_sources = broker.get_profile(ticker)
        result, source, field_sources = broker.get_financials(ticker, limit)
        ...
    """

    def __init__(self) -> None:
        self._fmp = FMPProvider()

        # Build optional fallback list in priority order.
        # A provider is included only when is_available() returns True,
        # which requires the appropriate env var to be set.
        self._fallbacks: list[DataProvider] = []

        # Alpha Vantage — documented REST API, real data, registered key required.
        # Covers income / balance / cashflow statements. Comes first because it
        # provides the most reliable fallback for financial statement data.
        alphavantage = AlphaVantageProvider()
        if alphavantage.is_available():
            self._fallbacks.append(alphavantage)
            logger.info("DataBroker: AlphaVantage fallback enabled")
        else:
            logger.debug("DataBroker: AlphaVantage fallback not configured")

        ycharts = YChartsProvider()
        if ycharts.is_available():
            self._fallbacks.append(ycharts)
            logger.info("DataBroker: YCharts fallback enabled")
        else:
            logger.debug("DataBroker: YCharts fallback not configured")

        alphaspread = AlphaSpreadProvider()
        if alphaspread.is_available():
            self._fallbacks.append(alphaspread)
            logger.info("DataBroker: AlphaSpread fallback enabled")
        else:
            logger.debug("DataBroker: AlphaSpread fallback not configured")

        providers_str = "FMP" + (
            f" + {', '.join(p.name for p in self._fallbacks)}"
            if self._fallbacks else " only"
        )
        print(f"  [DataBroker] Active providers: {providers_str}")

    # ── Public API (mirrors FMPProvider, returns 3-tuples) ─────────────────────

    def get_profile(
        self, ticker: str
    ) -> tuple[ProfileResult, str, dict[str, str]]:
        """
        FMP first. FMPError is caught and treated as empty so fallbacks can run.
        Returns (ProfileResult, dataset_source, field_sources).
        """
        result = ProfileResult()
        source = "FMP"
        try:
            result = self._fmp.get_profile(ticker)
        except FMPError as exc:
            restriction = _is_access_restricted(exc)
            source = "access_restricted" if restriction else "FMP_error"
            logger.warning(
                "DataBroker: FMP profile [%s] failed (%s) — trying fallbacks", ticker, exc
            )
            print(
                f"  [DataBroker] FMP /profile {'access-restricted' if restriction else 'error'}"
                f" for {ticker} — trying fallbacks"
            )

        if result.is_empty():
            for provider in self._fallbacks:
                try:
                    fb = provider.get_profile(ticker)
                    if not fb.is_empty():
                        result = fb
                        source = provider.name
                        logger.info(
                            "DataBroker: profile for %s filled from %s fallback",
                            ticker, provider.name,
                        )
                        print(
                            f"  [DataBroker] profile for {ticker} — using {provider.name} fallback"
                        )
                        break
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "DataBroker: %s.get_profile [%s] failed — %s",
                        provider.name, ticker, exc,
                    )

        if result.is_empty():
            source = "unavailable"

        return result, source, {}

    def get_financials(
        self, ticker: str, limit: int = 5
    ) -> tuple[FinancialsResult, str, dict[str, str]]:
        """
        FMP first for the dataset.

        If FMP raises FMPError (including HTTP 402 access-restricted), the error
        is caught here and the fallback loop runs immediately. This is the key
        fix — previously FMPError propagated past the fallback loop entirely.

        If FMP returns data, field-level fill from fallbacks fills any None fields.
        If FMP is empty or errors, try fallbacks for the whole dataset.
        """
        result = FinancialsResult()
        source = "FMP"
        field_sources: dict[str, str] = {}
        fmp_failed = False

        try:
            result = self._fmp.get_financials(ticker, limit)
        except FMPError as exc:
            fmp_failed = True
            restriction = _is_access_restricted(exc)
            source = "access_restricted" if restriction else "FMP_error"
            logger.warning(
                "DataBroker: FMP financials [%s] failed (%s) — trying fallbacks",
                ticker, exc,
            )
            print(
                f"\n  [DataBroker] FMP financials "
                f"{'access-restricted (HTTP 402)' if restriction else 'error'}"
                f" for {ticker} — attempting fallback providers"
            )

        if result.is_empty():
            # Dataset-level fallback — first non-empty provider wins
            for provider in self._fallbacks:
                logger.info(
                    "DataBroker: attempting %s for financials [%s]%s",
                    provider.name, ticker,
                    " (FMP access-restricted)" if fmp_failed else " (FMP empty)",
                )
                print(
                    f"  [DataBroker] financials fallback: trying {provider.name} for {ticker}"
                )
                try:
                    fb = provider.get_financials(ticker, limit)
                    if not fb.is_empty():
                        result = fb
                        source = provider.name
                        logger.info(
                            "DataBroker: financials for %s — %s provided"
                            " income=%d balance=%d cashflow=%d",
                            ticker, provider.name,
                            len(fb.income_statements),
                            len(fb.balance_sheets),
                            len(fb.cash_flows),
                        )
                        print(
                            f"  [DataBroker] financials for {ticker} — "
                            f"{provider.name} supplied "
                            f"income={len(fb.income_statements)} "
                            f"balance={len(fb.balance_sheets)} "
                            f"cashflow={len(fb.cash_flows)}"
                        )
                        break
                    else:
                        logger.info(
                            "DataBroker: %s returned empty financials for %s — trying next",
                            provider.name, ticker,
                        )
                        print(
                            f"  [DataBroker] {provider.name} returned empty financials"
                            f" for {ticker} — trying next provider"
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "DataBroker: %s.get_financials [%s] failed — %s",
                        provider.name, ticker, exc,
                    )
                    print(
                        f"  [DataBroker] {provider.name} financials error"
                        f" for {ticker}: {exc}"
                    )
        else:
            # FMP returned data — field-level fill from fallbacks for any None fields.
            # Skip providers marked skip_when_fmp_has_data (e.g. AlphaVantage) to
            # avoid the 13s inter-request pacing delay when FMP already succeeded.
            for provider in self._fallbacks:
                if getattr(provider, "skip_when_fmp_has_data", False):
                    logger.debug(
                        "DataBroker: skipping %s field-fill for %s — FMP has data",
                        provider.name, ticker,
                    )
                    continue
                try:
                    fb = provider.get_financials(ticker, limit)
                    if fb.is_empty():
                        continue
                    field_sources.update(
                        _merge_rows(result.income_statements, fb.income_statements,
                                    provider.name, "income")
                    )
                    field_sources.update(
                        _merge_rows(result.balance_sheets, fb.balance_sheets,
                                    provider.name, "balance")
                    )
                    field_sources.update(
                        _merge_rows(result.cash_flows, fb.cash_flows,
                                    provider.name, "cashflow")
                    )
                    field_sources.update(
                        _merge_rows(result.ratios, fb.ratios,
                                    provider.name, "ratios")
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "DataBroker: %s.get_financials field-fill [%s] failed — %s",
                        provider.name, ticker, exc,
                    )

        if result.is_empty():
            source = "unavailable"
            logger.warning(
                "DataBroker: financials for %s — no provider returned data", ticker
            )
            print(f"  [DataBroker] financials for {ticker} — all providers exhausted, no data")

        return result, source, field_sources

    def get_quarterly(
        self, ticker: str
    ) -> tuple[list[IncomeStatement], str, dict[str, str]]:
        result: list[IncomeStatement] = []
        source = "FMP"
        field_sources: dict[str, str] = {}

        try:
            result = self._fmp.get_quarterly(ticker)
        except FMPError as exc:
            logger.warning(
                "DataBroker: FMP quarterly [%s] failed (%s) — trying fallbacks", ticker, exc
            )

        if not result:
            for provider in self._fallbacks:
                try:
                    fb = provider.get_quarterly(ticker)
                    if fb:
                        result = fb
                        source = provider.name
                        break
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "DataBroker: %s.get_quarterly [%s] failed — %s",
                        provider.name, ticker, exc,
                    )
        else:
            for provider in self._fallbacks:
                try:
                    fb = provider.get_quarterly(ticker)
                    if fb:
                        field_sources.update(
                            _merge_rows(result, fb, provider.name, "quarterly_income")
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "DataBroker: %s.get_quarterly field-fill [%s] failed — %s",
                        provider.name, ticker, exc,
                    )

        if not result:
            source = "unavailable"

        return result, source, field_sources

    def get_price_history(
        self, ticker: str, days: int = 400
    ) -> tuple[Optional[PriceHistory], str, dict[str, str]]:
        result: Optional[PriceHistory] = None
        source = "FMP"

        try:
            result = self._fmp.get_price_history(ticker, days)
        except FMPError as exc:
            logger.warning(
                "DataBroker: FMP price history [%s] failed (%s) — trying fallbacks", ticker, exc
            )

        if result is None:
            for provider in self._fallbacks:
                try:
                    fb = provider.get_price_history(ticker, days)
                    if fb is not None:
                        result = fb
                        source = provider.name
                        logger.info(
                            "DataBroker: price history for %s filled from %s fallback",
                            ticker, provider.name,
                        )
                        break
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "DataBroker: %s.get_price_history [%s] failed — %s",
                        provider.name, ticker, exc,
                    )

        if result is None:
            source = "unavailable"

        return result, source, {}

    def get_earnings(
        self, ticker: str
    ) -> tuple[tuple[list, list], str, dict[str, str]]:
        earnings: list = []
        surprises: list = []
        source = "FMP"

        try:
            earnings, surprises = self._fmp.get_earnings(ticker)
        except FMPError as exc:
            logger.warning(
                "DataBroker: FMP earnings [%s] failed (%s) — trying fallbacks", ticker, exc
            )

        if not earnings:
            for provider in self._fallbacks:
                try:
                    fb_earnings, fb_surprises = provider.get_earnings(ticker)
                    if fb_earnings:
                        earnings = fb_earnings
                        surprises = fb_surprises or surprises
                        source = provider.name
                        break
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "DataBroker: %s.get_earnings [%s] failed — %s",
                        provider.name, ticker, exc,
                    )

        if not earnings:
            source = "unavailable"

        return (earnings, surprises), source, {}

    def get_analyst_data(
        self, ticker: str
    ) -> tuple[AnalystResult, str, dict[str, str]]:
        result = AnalystResult()
        source = "FMP"

        try:
            result = self._fmp.get_analyst_data(ticker)
        except FMPError as exc:
            logger.warning(
                "DataBroker: FMP analyst data [%s] failed (%s) — trying fallbacks", ticker, exc
            )

        if result.is_empty():
            for provider in self._fallbacks:
                try:
                    fb = provider.get_analyst_data(ticker)
                    if not fb.is_empty():
                        result = fb
                        source = provider.name
                        break
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "DataBroker: %s.get_analyst_data [%s] failed — %s",
                        provider.name, ticker, exc,
                    )

        if result.is_empty():
            source = "unavailable"

        return result, source, {}

    def get_sector_performance(
        self,
    ) -> tuple[list[dict[str, Any]], str, dict[str, str]]:
        result: list[dict[str, Any]] = []
        source = "FMP"

        try:
            result = self._fmp.get_sector_performance()
        except FMPError as exc:
            logger.warning(
                "DataBroker: FMP sector performance failed (%s) — trying fallbacks", exc
            )

        if not result:
            for provider in self._fallbacks:
                try:
                    fb = provider.get_sector_performance()
                    if fb:
                        result = fb
                        source = provider.name
                        break
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "DataBroker: %s.get_sector_performance failed — %s",
                        provider.name, exc,
                    )

        if not result:
            source = "unavailable"

        return result, source, {}

    def get_peers(self, ticker: str) -> list[str]:
        """
        Return FMP-curated peer tickers (best-effort, no fallback).
        Empty list on any error.
        """
        try:
            return self._fmp.get_peers(ticker)
        except Exception as exc:  # noqa: BLE001
            logger.debug("DataBroker: get_peers [%s] failed — %s", ticker, exc)
            return []

    # ── Passthrough ────────────────────────────────────────────────────────────

    @property
    def fmp_call_log(self) -> list[str]:
        """API call log from the FMP provider (for audit / debug)."""
        return self._fmp.call_log


# ── Internal helper ────────────────────────────────────────────────────────────

def _merge_rows(
    primary_rows: list,
    secondary_rows: list,
    secondary_name: str,
    prefix: str,
) -> dict[str, str]:
    """
    Merge field-level fallback data across parallel lists of dataclass objects.
    Rows are matched by position (index). Extra secondary rows are ignored.
    Returns all field-level source attributions from every row.
    """
    filled: dict[str, str] = {}
    for i, primary_row in enumerate(primary_rows):
        if i >= len(secondary_rows):
            break
        filled.update(
            _merge_dataclass_fields(
                primary_row, secondary_rows[i], secondary_name, prefix
            )
        )
    return filled
