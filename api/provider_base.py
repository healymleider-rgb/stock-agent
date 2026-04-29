"""
DataProvider interface — abstract base class for all data providers.

All providers (FMP, YCharts, AlphaSpread) implement this interface.
The DataBroker calls each in configured priority order and merges results.

Return contract
---------------
Every get_* method returns a typed result container or empty/None.
Providers MUST NOT raise exceptions — catch internally and return empty.
An empty result signals "no data from this provider"; the broker tries next.

Adding a new provider
---------------------
1. Create  api/<name>_provider.py  subclassing DataProvider
2. Set     name = "<Name>"
3. Override is_available() to check for required credentials / tokens
4. Implement all get_* methods (return empty for any unimplemented ones)
5. Register in DataBroker.__init__() as an optional fallback in priority order
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from models.stock_data import (
    BalanceSheet,
    CashFlowStatement,
    CompanyProfile,
    FinancialRatios,
    IncomeStatement,
    PriceHistory,
)


# ── Typed result containers ────────────────────────────────────────────────────

@dataclass
class ProfileResult:
    """Returned by get_profile()."""
    profile: Optional[CompanyProfile] = None
    current_price: Optional[float] = None
    market_cap: Optional[float] = None
    shares_outstanding: Optional[float] = None   # raw shares from /quote or /shares-float
    market_cap_computed: Optional[float] = None  # price × shares cross-check
    quote: dict[str, Any] = field(default_factory=dict)
    # Provenance for shares_outstanding — populated when /shares-float (SEC EDGAR) is used
    shares_source: str = ""                            # e.g. "FMP/shares-float (SEC EDGAR)"
    shares_filing_period_end: Optional[str] = None    # fiscal period end, e.g. "2025-12-31"
    shares_filing_date: Optional[str] = None          # real SEC filing date — set later from income_statements
    shares_filing_url: Optional[str] = None           # direct SEC EDGAR document URL
    shares_data_refreshed_at: Optional[str] = None    # when FMP last refreshed the float record

    def is_empty(self) -> bool:
        return self.profile is None


@dataclass
class FinancialsResult:
    """Returned by get_financials()."""
    income_statements: list[IncomeStatement] = field(default_factory=list)
    balance_sheets: list[BalanceSheet] = field(default_factory=list)
    cash_flows: list[CashFlowStatement] = field(default_factory=list)
    ratios: list[FinancialRatios] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.income_statements


@dataclass
class AnalystResult:
    """Returned by get_analyst_data()."""
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    price_targets: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.recommendations and not self.price_targets


# ── Abstract base ──────────────────────────────────────────────────────────────

class DataProvider(ABC):
    """
    Abstract interface every data provider must implement.

    The DataBroker uses this to call providers uniformly, apply the
    FMP-first waterfall, and merge results with source attribution.
    """

    #: Human-readable name used in logs and source attribution records.
    name: str = "base"

    def is_available(self) -> bool:
        """
        Return True if this provider is configured and ready to use.
        The DataBroker silently skips unavailable providers.
        Default: True. Override to check credential environment variables.
        """
        return True

    @abstractmethod
    def get_profile(self, ticker: str) -> ProfileResult: ...

    @abstractmethod
    def get_financials(self, ticker: str, limit: int = 5) -> FinancialsResult: ...

    @abstractmethod
    def get_quarterly(self, ticker: str) -> list[IncomeStatement]: ...

    @abstractmethod
    def get_price_history(self, ticker: str, days: int = 400) -> Optional[PriceHistory]: ...

    @abstractmethod
    def get_earnings(
        self, ticker: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]: ...

    @abstractmethod
    def get_analyst_data(self, ticker: str) -> AnalystResult: ...

    @abstractmethod
    def get_sector_performance(self) -> list[dict[str, Any]]: ...
