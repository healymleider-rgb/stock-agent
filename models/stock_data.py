"""
Normalized domain models for stock data.

FMPClient produces these; all agents consume them.
Raw API dicts never leave the data retrieval layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ── Company Overview ───────────────────────────────────────────────────────────

@dataclass
class CompanyProfile:
    symbol: str
    company_name: str
    sector: str
    industry: str
    description: str
    exchange: str
    country: str
    market_cap: Optional[float] = None
    price: Optional[float] = None
    beta: Optional[float] = None
    volume_avg: Optional[int] = None
    employees: Optional[int] = None
    website: str = ""
    ceo: str = ""
    ipo_date: str = ""


# ── Income Statement ───────────────────────────────────────────────────────────

@dataclass
class IncomeStatement:
    date: str
    period: str = "FY"           # "FY" or "Q1"/"Q2" etc.
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    ebitda: Optional[float] = None
    eps: Optional[float] = None
    eps_diluted: Optional[float] = None
    gross_profit_ratio: Optional[float] = None
    operating_income_ratio: Optional[float] = None
    net_income_ratio: Optional[float] = None
    rd_expenses: Optional[float] = None
    selling_expenses: Optional[float] = None
    interest_expense: Optional[float] = None


# ── Balance Sheet ──────────────────────────────────────────────────────────────

@dataclass
class BalanceSheet:
    date: str
    period: str = "FY"
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    total_debt: Optional[float] = None
    short_term_debt: Optional[float] = None
    long_term_debt: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    short_term_investments: Optional[float] = None
    total_current_assets: Optional[float] = None
    total_current_liabilities: Optional[float] = None
    goodwill: Optional[float] = None
    intangible_assets: Optional[float] = None
    retained_earnings: Optional[float] = None


# ── Cash Flow Statement ────────────────────────────────────────────────────────

@dataclass
class CashFlowStatement:
    date: str
    period: str = "FY"
    operating_cash_flow: Optional[float] = None
    capital_expenditure: Optional[float] = None
    free_cash_flow: Optional[float] = None
    net_income: Optional[float] = None
    depreciation_amortization: Optional[float] = None
    stock_based_compensation: Optional[float] = None
    dividends_paid: Optional[float] = None
    acquisitions: Optional[float] = None
    common_stock_repurchased: Optional[float] = None


# ── Ratios & Metrics ───────────────────────────────────────────────────────────

@dataclass
class FinancialRatios:
    date: str
    period: str = "FY"
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    ev_to_revenue: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    roic: Optional[float] = None
    debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    interest_coverage: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    fcf_yield: Optional[float] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None


# ── Price History ──────────────────────────────────────────────────────────────

@dataclass
class PriceHistory:
    dates: list[str] = field(default_factory=list)     # newest → oldest
    closes: list[float] = field(default_factory=list)
    highs: list[float] = field(default_factory=list)
    lows: list[float] = field(default_factory=list)
    volumes: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.closes)

    @property
    def latest_price(self) -> Optional[float]:
        return self.closes[0] if self.closes else None

    @property
    def price_6m_ago(self) -> Optional[float]:
        """Price ~126 trading days ago."""
        idx = min(126, len(self.closes) - 1)
        return self.closes[idx] if len(self.closes) > idx else None

    @property
    def price_12m_ago(self) -> Optional[float]:
        """Price ~252 trading days ago."""
        idx = min(252, len(self.closes) - 1)
        return self.closes[idx] if len(self.closes) > idx else None


# ── Top-Level Container ────────────────────────────────────────────────────────

@dataclass
class StockData:
    """
    Master container that flows through the system.
    The DataRetrievalAgent populates this incrementally
    as the Orchestrator requests different data slices.
    """
    ticker: str

    profile: Optional[CompanyProfile] = None
    income_statements: list[IncomeStatement] = field(default_factory=list)
    quarterly_income: list[IncomeStatement] = field(default_factory=list)
    balance_sheets: list[BalanceSheet] = field(default_factory=list)
    cash_flows: list[CashFlowStatement] = field(default_factory=list)
    ratios: list[FinancialRatios] = field(default_factory=list)
    price_history: Optional[PriceHistory] = None
    current_price: Optional[float] = None
    market_cap: Optional[float] = None
    shares_outstanding: Optional[float] = None   # from /quote sharesOutstanding
    market_cap_computed: Optional[float] = None  # price × shares_outstanding
    earnings: list[dict[str, Any]] = field(default_factory=list)
    earnings_surprises: list[dict[str, Any]] = field(default_factory=list)
    analyst_recommendations: list[dict[str, Any]] = field(default_factory=list)
    price_targets: dict[str, Any] = field(default_factory=dict)
    sector_performance: list[dict[str, Any]] = field(default_factory=list)

    # Provider attribution — populated by OrchestratorAgent during ingestion.
    # Keys are data-type names ("profile", "financials", …) for dataset-level
    # attribution, and "prefix.field_name" for field-level fallback fills.
    # Values are provider names ("FMP", "YCharts", "AlphaSpread", "unavailable").
    data_sources: dict[str, str] = field(default_factory=dict)

    # Derived convenience accessors

    @property
    def latest_income(self) -> Optional[IncomeStatement]:
        return self.income_statements[0] if self.income_statements else None

    @property
    def latest_balance(self) -> Optional[BalanceSheet]:
        return self.balance_sheets[0] if self.balance_sheets else None

    @property
    def latest_cashflow(self) -> Optional[CashFlowStatement]:
        return self.cash_flows[0] if self.cash_flows else None

    @property
    def latest_ratios(self) -> Optional[FinancialRatios]:
        return self.ratios[0] if self.ratios else None
