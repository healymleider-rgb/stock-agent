"""
Peer comparison analysis.

Fetches P/E, P/S, EV/EBITDA, EPS growth, and PEG for up to _MAX_PEERS peers
and builds a comparative table alongside the target stock.

Candidate pool — 3 tiers, merged in priority order:
  0. FMP /stock-peers   — dynamic, same sector + industry (most precise)
  1. _TICKER_PEERS      — curated industry-specific maps for known tickers
  2. FMP /stock-screener — adaptive market cap band (archetype-specific)

  Sector-level (_PEER_UNIVERSE) and global (_GLOBAL_FALLBACK) tiers are removed.
  PeerSelectionEngine is the ONLY source of peers. No sector/industry fallbacks.
  If no usable peers are found → "No valid peers available." Never substitute.

Adaptive filtering (via PeerSelectionEngine):
  - Market cap bands are archetype-specific (tight for Banks; very wide for Fintech)
  - Structural constraints: revenue similarity ±50%, margin similarity ±20pp
  - Geographic preference: same region sorted higher (US > DEV > EM for US targets)
  - Auto-expand: if <3 peers survive structural filter, constraints relax once
  - Economic-similarity ranking: weighted Euclidean distance replaces completeness sort

Metric derivation per peer:
  - P/E  : price / eps_diluted  (or net_income / shares when eps absent)
  - P/S  : mkt_cap / revenue
  - EV/EBITDA: (mkt_cap + debt - cash) / ebitda
  - EPS CAGR: annualised over min(3,n-1) years of eps history
  - PEG  : P/E / EPS CAGR%

A peer with ANY computable metric (pe, ps, ev_ebitda, growth) is kept.
The peer section is shown if ≥1 usable peer is found.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agents.peer_selection_engine import (
    Archetype,
    ClassificationResult,
    GeoRegion,
    PeerSelectionEngine,
    get_engine,
)
from api.fmp_client import FMPError
from api.fmp_provider import FMPProvider
from config import Config


# ── Ticker-specific peer maps (tier 1) ────────────────────────────────────────
_TICKER_PEERS: dict[str, list[str]] = {
    # ── Ad-tech / DSP ─────────────────────────────────────────────────────────
    "TTD":  ["APP",  "MGNI", "PUBM", "DV",   "IAS"],
    "MGNI": ["TTD",  "PUBM", "APP",  "DV",   "IAS"],
    "PUBM": ["TTD",  "MGNI", "APP",  "DV",   "IAS"],
    "DV":   ["TTD",  "MGNI", "IAS",  "APP",  "PUBM"],
    "IAS":  ["TTD",  "MGNI", "DV",   "APP",  "PUBM"],
    "APP":  ["TTD",  "MGNI", "DV",   "IAS",  "PUBM"],
    # ── Marketing-tech / CRM ─────────────────────────────────────────────────
    "HUBS": ["CRM",  "MNDY", "ZI",   "BRZE", "NOW"],
    "CRM":  ["SAP",  "ORCL", "HUBS", "NOW",  "WDAY"],
    "BRZE": ["HUBS", "CRM",  "ZI",   "MNDY", "NOW"],
    # ── Cloud infrastructure ──────────────────────────────────────────────────
    "AMZN": ["MSFT", "GOOGL", "META", "WMT",  "SHOP"],
    "SNOW": ["DDOG", "MDB",   "ESTC", "DT",   "SPLK"],
    "DDOG": ["SNOW", "ESTC",  "DT",   "NR",   "SPLK"],
    "MDB":  ["SNOW", "DDOG",  "ESTC", "CASS", "PTC"],
    # ── Cybersecurity ─────────────────────────────────────────────────────────
    "CRWD": ["S",    "PANW",  "FTNT", "ZS",   "OKTA"],
    "PANW": ["CRWD", "FTNT",  "ZS",   "S",    "OKTA"],
    "ZS":   ["CRWD", "PANW",  "FTNT", "S",    "NET"],
    "S":    ["CRWD", "PANW",  "ZS",   "FTNT", "NET"],
    "OKTA": ["CRWD", "ZS",    "PANW", "SAIL", "NET"],
    # ── Payments / fintech ────────────────────────────────────────────────────
    "V":    ["MA",   "PYPL",  "AXP",  "FIS",  "FISV"],
    "MA":   ["V",    "PYPL",  "AXP",  "FIS",  "FISV"],
    "PYPL": ["V",    "MA",    "SQ",   "AFRM", "SEZL"],
    "SQ":   ["PYPL", "AFRM",  "FOUR", "SEZL", "V"],
    "AFRM": ["SQ",   "PYPL",  "SEZL", "FOUR", "MA"],
    # ── SaaS ─────────────────────────────────────────────────────────────────
    "NOW":  ["CRM",  "WDAY",  "HUBS", "SAP",  "ORCL"],
    "WDAY": ["NOW",  "CRM",   "SAP",  "INTU", "ADSK"],
    "INTU": ["WDAY", "ADP",   "PCTY", "PAYX", "NOW"],
    "SHOP": ["AMZN", "BIGC",  "WIX",  "SQSP", "ETSY"],
    # ── Creative / design software ────────────────────────────────────────────
    "ADBE": ["CRM",  "NOW",   "MSFT", "ORCL", "INTU"],
    "ADSK": ["ADBE", "PTC",   "ANSYS","CDNS", "SNPS"],
    "ANSS": ["ADSK", "CDNS",  "SNPS", "PTC",  "ADBE"],
    "CDNS": ["ANSS", "SNPS",  "ADSK", "PTC",  "KLAC"],
    "SNPS": ["CDNS", "ANSS",  "ADSK", "KLAC", "MRVL"],
    # ── Athletic / footwear ───────────────────────────────────────────────────
    "NKE":  ["LULU", "DECK",  "CROX", "SKX",  "ON"],
    "LULU": ["NKE",  "DECK",  "CROX", "SKX",  "ON"],
    "DECK": ["NKE",  "LULU",  "CROX", "SKX",  "ON"],
    "CROX": ["NKE",  "LULU",  "DECK", "SKX",  "ON"],
    "SKX":  ["NKE",  "LULU",  "DECK", "CROX", "ON"],
    # ── Mega-cap tech ─────────────────────────────────────────────────────────
    "AAPL": ["MSFT", "GOOGL", "META", "NVDA", "AMZN"],
    "MSFT": ["AAPL", "GOOGL", "META", "AMZN", "NOW"],
    "GOOGL":["AAPL", "MSFT",  "META", "AMZN", "TTD"],
    "META": ["AAPL", "MSFT",  "GOOGL","SNAP", "PINS"],
    "SNAP": ["META", "PINS",  "TTD",  "MGNI", "DV"],
    "PINS": ["META", "SNAP",  "TTD",  "GOOGL","APP"],
    # ── Semiconductors ────────────────────────────────────────────────────────
    "NVDA": ["AMD",  "INTC",  "AVGO", "QCOM", "MRVL"],
    "AMD":  ["NVDA", "INTC",  "AVGO", "QCOM", "MRVL"],
    "INTC": ["NVDA", "AMD",   "AVGO", "QCOM", "TSM"],
    "AVGO": ["NVDA", "AMD",   "QCOM", "MRVL", "AMAT"],
    "QCOM": ["NVDA", "AMD",   "AVGO", "MRVL", "MTSI"],
    # ── Beverages / consumer staples ─────────────────────────────────────────
    "KO":   ["PEP",  "MDLZ",  "GIS",  "HSY",  "MKC"],
    "PEP":  ["KO",   "MDLZ",  "GIS",  "HSY",  "MKC"],
    # ── QSR ──────────────────────────────────────────────────────────────────
    "MCD":  ["SBUX", "YUM",   "QSR",  "WEN",  "DPZ"],
    "SBUX": ["MCD",  "YUM",   "QSR",  "DNKN", "DPZ"],
    # ── E-commerce ────────────────────────────────────────────────────────────
    "ETSY": ["SHOP", "AMZN",  "EBAY", "W",    "POSHM"],
    "EBAY": ["AMZN", "ETSY",  "SHOP", "WMT",  "W"],
    # ── Large banks ───────────────────────────────────────────────────────────
    "JPM":  ["BAC",  "WFC",   "GS",   "MS",   "C"],
    "BAC":  ["JPM",  "WFC",   "GS",   "C",    "MS"],
    "GS":   ["MS",   "JPM",   "BAC",  "WFC",  "C"],
    "MS":   ["GS",   "JPM",   "BAC",  "WFC",  "C"],
    # ── Oil majors ────────────────────────────────────────────────────────────
    "XOM":  ["CVX",  "COP",   "BP",   "SHEL", "TTE"],
    "CVX":  ["XOM",  "COP",   "BP",   "SHEL", "TTE"],
    # ── Pharma ───────────────────────────────────────────────────────────────
    "JNJ":  ["PFE",  "ABBV",  "MRK",  "BMY",  "LLY"],
    "PFE":  ["JNJ",  "ABBV",  "MRK",  "BMY",  "LLY"],
    "ABBV": ["JNJ",  "PFE",   "MRK",  "BMY",  "LLY"],
    "LLY":  ["NVO",  "ABBV",  "JNJ",  "MRK",  "BMY"],
    # ── EVs ──────────────────────────────────────────────────────────────────
    "TSLA": ["RIVN", "LCID",  "F",    "GM",   "NIO"],
    # ── Streaming ────────────────────────────────────────────────────────────
    "NFLX": ["DIS",  "WBD",   "PARA", "AMZN", "SPOT"],
    "SPOT": ["NFLX", "IHRT",  "SXM",  "AMZN", "DIS"],
    # ── Retail ───────────────────────────────────────────────────────────────
    "WMT":  ["COST", "TGT",   "AMZN", "KR",   "DG"],
    "COST": ["WMT",  "TGT",   "BJ",   "AMZN", "KR"],
    "TGT":  ["WMT",  "COST",  "KSS",  "M",    "DG"],
}

# ── Sector-level peer universe (tier 3) ────────────────────────────────────────

# _PEER_UNIVERSE (tier 3) and _GLOBAL_FALLBACK (tier 4) DELETED.
# Sector/global fallback tiers bypass PeerSelectionEngine's archetype logic and
# are the root cause of wrong peer groups (e.g. banks for data companies).
# PeerSelectionEngine is the ONLY source of peers. No fallbacks.
# If tiers 0–2 produce no usable peers → output is "No valid peers available."

_MAX_PEERS              = 5    # maximum peers shown in the table
_MAX_CANDIDATES         = 15   # maximum tickers evaluated before stopping
_MIN_VALID_FOR_CONCLUSION = 3  # rows needed before making a relative-stat claim
_USABLE_PEER_TARGET     = 4    # ideal peers to find
_MIN_USABLE_PEERS       = 1    # show section with any usable peer

# ── Relaxation cascade constants ───────────────────────────────────────────────
# Adjacent archetype pairs — allowed when same-arch pool is exhausted (Step 4).
# Key = target archetype int; value = set of adjacent peer archetype ints.
# Hard exclusions (Banks, Insurance, Asset Managers, Other, Fintech) are never
# adjacent regardless of peer scarcity — their primary economic drivers are
# structurally incompatible (NIM, reserves, AUM float, take-rate).
_ADJACENT: dict[int, set[int]] = {
    int(Archetype.FINANCIAL_DATA):   {int(Archetype.EXCHANGES)},
    int(Archetype.EXCHANGES):        {int(Archetype.FINANCIAL_DATA), int(Archetype.INVESTMENT_BANKS)},
    int(Archetype.INVESTMENT_BANKS): {int(Archetype.EXCHANGES)},
}

# Adjacent-arch curated supplement — tickers that are high-confidence adjacent-arch
# proxies for each archetype.  These are injected into the candidate pool as Tier 3
# when the general pool (tiers 0–2) does not naturally contain adjacent-arch companies.
#
# Purpose: Step 4 of the relaxation cascade requires adj_arch_pool to be populated.
# The FMP /stock-peers API and screener are sector-scoped — they may return only
# same-arch candidates, leaving adj_arch_pool empty and Step 4 unreachable.
# This map guarantees at least a minimal adjacent-arch candidate set is evaluated.
#
# Tickers chosen for: archetype alignment, fee-based margins, and consistent FMP coverage.
# The _is_fee_based() gate (gm > 55% OR om > 15%) provides the final economic filter.
_ADJACENT_CURATED: dict[int, list[str]] = {
    # FINANCIAL_DATA target → adjacent: Exchanges / Market Infrastructure
    int(Archetype.FINANCIAL_DATA): [
        "ICE",   # Intercontinental Exchange — largest data + exchange conglomerate
        "CME",   # CME Group — derivatives exchange with substantial data revenue
        "CBOE",  # Cboe Global Markets — options exchange + market data
        "NDAQ",  # Nasdaq — exchange + financial technology / data services
        "MKTX",  # MarketAxess — electronic bond trading (high recurring revenue)
    ],
    # EXCHANGES target → adjacent: Financial Data + Investment Banks (fee-based only)
    int(Archetype.EXCHANGES): [
        "SPGI",  # S&P Global — ratings + data, adjacent subscription economics
        "MSCI",  # MSCI — index + analytics, high ARR; strong proxy for exchange data
        "FDS",   # FactSet — financial data platform, fee-based
        "TRI",   # Thomson Reuters — data / workflow (fee-based)
        "VRSK",  # Verisk — data analytics, insurance / energy verticals
        "EVR",   # Evercore — pure advisory boutique; high operating margins
        "LAZ",   # Lazard — advisory + asset management; fee-revenue mix
    ],
    # INVESTMENT_BANKS target → adjacent: Exchanges / Market Infrastructure
    int(Archetype.INVESTMENT_BANKS): [
        "ICE",   # Intercontinental Exchange
        "CME",   # CME Group
        "CBOE",  # Cboe Global Markets
        "NDAQ",  # Nasdaq
        "MKTX",  # MarketAxess
    ],
}

# Core metrics: derivable from income statement alone; universally available.
# Used in Steps 3–4 of the relaxation cascade.
_CORE_METRICS:     list[str] = ["revenue_growth", "operating_margin", "gross_margin"]

# Extended metric pool: any of these satisfies the ≥2-metric gate in Step 2.
_EXTENDED_METRICS: list[str] = [
    "pe", "ps", "ev_ebitda", "growth_pct",
    "gross_margin", "operating_margin", "revenue_growth",
]

# Sanity caps — ratios beyond these are not meaningful comparisons
_PE_MAX       = 500.0
_PS_MAX       = 100.0
_EV_EBITDA_MAX = 200.0
_PEG_MAX      = 100.0


# ── Per-fiscal-year snapshot for historical table ─────────────────────────────

@dataclass
class HistoricalYear:
    """One fiscal year of per-company metrics for the 5-year peer history table."""
    label:          str             # "Current", "FY-1", "FY-2", …
    fiscal_year:    str  = ""       # calendar year string, e.g. "2024"
    revenue_growth: Optional[float] = None   # YoY %
    eps_growth:     Optional[float] = None   # YoY %
    op_margin:      Optional[float] = None   # decimal
    net_margin:     Optional[float] = None   # decimal
    roe:            Optional[float] = None   # decimal
    roic:           Optional[float] = None   # decimal
    ebitda_growth:  Optional[float] = None   # YoY %


def _derive_historical(
    income_statements: list,
    ratios:            list | None,
    n:                 int = 5,
) -> list[HistoricalYear]:
    """
    Build per-fiscal-year historical metric snapshots from annual statements.

    Parameters
    ----------
    income_statements : annual statements, newest first (standard pipeline order)
    ratios            : annual ratio objects, newest first; None if unavailable
    n                 : number of periods to return (default 5)

    Returns a list of HistoricalYear objects, newest first (index 0 = Current).
    YoY growth rates require the NEXT (older) statement, so the oldest period
    returned will have None for all growth fields.
    Never raises.
    """
    if not income_statements:
        return []
    n_inc = len(income_statements)
    n_rat = len(ratios) if ratios else 0
    result: list[HistoricalYear] = []

    for i in range(min(n, n_inc)):
        inc = income_statements[i]
        rat = ratios[i] if (ratios and i < n_rat) else None

        # Fiscal year from date field ("2024-09-28" → "2024")
        fy    = (inc.date or "")[:4]
        label = "Current" if i == 0 else f"FY-{i}"

        # ── YoY growth rates (vs next/older period) ───────────────────────────
        def _yoy_pct(v0, v1, cap=300.0) -> Optional[float]:
            if v0 is None or not v1 or v1 == 0:
                return None
            raw = (v0 / v1 - 1) * 100
            return round(raw, 1) if abs(raw) <= cap else None

        prior = income_statements[i + 1] if i + 1 < n_inc else None
        rev_g  = _yoy_pct(inc.revenue, prior.revenue if prior else None)
        ebi_g  = _yoy_pct(inc.ebitda,  prior.ebitda  if prior else None)

        # EPS growth — positive base required for meaningful %
        eps_g = None
        if prior is not None:
            e0 = inc.eps_diluted   or inc.eps
            e1 = prior.eps_diluted or prior.eps
            if e0 is not None and e1 is not None and e1 > 0 and e0 > 0:
                raw = (e0 / e1 - 1) * 100
                eps_g = round(raw, 1) if abs(raw) <= 200 else None

        # ── Margin: ratio API preferred, income statement fallback ────────────
        def _op_mg(inc, rat) -> Optional[float]:
            if rat and rat.operating_margin is not None:
                return round(rat.operating_margin, 4)
            if getattr(inc, "operating_income_ratio", None) is not None:
                return round(inc.operating_income_ratio, 4)
            if inc.operating_income and inc.revenue and inc.revenue > 0:
                return round(inc.operating_income / inc.revenue, 4)
            return None

        def _net_mg(inc, rat) -> Optional[float]:
            if rat and rat.net_margin is not None:
                return round(rat.net_margin, 4)
            if getattr(inc, "net_income_ratio", None) is not None:
                return round(inc.net_income_ratio, 4)
            if inc.net_income and inc.revenue and inc.revenue > 0:
                return round(inc.net_income / inc.revenue, 4)
            return None

        result.append(HistoricalYear(
            label          = label,
            fiscal_year    = fy,
            revenue_growth = rev_g,
            eps_growth     = eps_g,
            op_margin      = _op_mg(inc, rat),
            net_margin     = _net_mg(inc, rat),
            roe            = round(rat.roe,  4) if (rat and rat.roe  is not None) else None,
            roic           = round(rat.roic, 4) if (rat and rat.roic is not None) else None,
            ebitda_growth  = ebi_g,
        ))

    return result


# ── Internal metric bundle (returned by _derive_metrics) ───────────────────────

@dataclass
class _DM:
    """All derived metrics for one peer row. Internal use only."""
    pe:               Optional[float] = None
    ps:               Optional[float] = None
    ev_ebitda:        Optional[float] = None
    growth_pct:       Optional[float] = None   # EPS CAGR %
    peg:              Optional[float] = None
    # Growth
    revenue_growth:   Optional[float] = None   # YoY %, e.g. 15.5
    ebitda_growth:    Optional[float] = None   # YoY %, e.g. 10.2
    # Profitability
    gross_margin:     Optional[float] = None   # ratio 0-1
    operating_margin: Optional[float] = None
    net_margin:       Optional[float] = None
    roe:              Optional[float] = None
    roic:             Optional[float] = None
    # Financial health
    debt_equity:       Optional[float] = None
    current_ratio:     Optional[float] = None
    interest_coverage: Optional[float] = None
    # Market / risk
    beta:              Optional[float] = None


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class PeerRow:
    ticker:     str
    name:       str            = ""     # company name for display
    market_cap: Optional[float] = None  # raw market cap for display
    # Valuation
    pe:         Optional[float] = None
    ps:         Optional[float] = None
    ev_ebitda:  Optional[float] = None
    growth_pct: Optional[float] = None   # annualised EPS CAGR, e.g. 12.5
    peg:        Optional[float] = None
    # Growth
    revenue_growth:   Optional[float] = None   # YoY revenue growth %
    ebitda_growth:    Optional[float] = None   # YoY EBITDA growth %
    # Profitability
    gross_margin:     Optional[float] = None   # ratio 0-1
    operating_margin: Optional[float] = None
    net_margin:       Optional[float] = None
    roe:              Optional[float] = None
    roic:             Optional[float] = None
    # Financial health
    debt_equity:       Optional[float] = None
    current_ratio:     Optional[float] = None
    interest_coverage: Optional[float] = None
    # Market / risk
    beta:             Optional[float] = None
    # Metadata
    is_target:       bool           = False
    tier:            int            = 0      # which candidate tier supplied this peer
    justification:   str            = ""    # why this peer is a relevant comparable
    peer_archetype:  Optional[int]  = None  # Archetype int value; set after classification
    # Source classification metadata — fetched from FMP profile, used in classify()
    sector:          str            = ""    # peer's sector string (e.g. "Financial Services")
    industry:        str            = ""    # peer's industry string (e.g. "Financial Data")
    # Peer quality fields — set by build_peer_comparison() after relaxation cascade
    is_proxy:        bool           = False  # True when peer is from adjacent archetype (Step 4)
    quality_score:   Optional[float] = None  # 0–100; reflects archetype alignment + data completeness
    peer_confidence: str            = ""    # "High" | "Medium" | "Low" | "Proxy"
    # 5-year historical metrics — populated from annual income statements + ratios
    historical: list[HistoricalYear] = field(default_factory=list)


@dataclass
class PeerComparison:
    target_ticker: str
    rows:          list[PeerRow] = field(default_factory=list)
    insights:      list[str]    = field(default_factory=list)
    has_peers:     bool = False
    # Selection quality metadata — set by build_peer_comparison()
    peer_level:    int  = 1     # 1 = direct, 2 = relaxed, 3 = proxy (adjacent arch)
    section_label: str  = "Peer Comparison"
    proxy_note:    str  = ""    # non-empty only at Level 3


# ── Metric derivation ──────────────────────────────────────────────────────────

def _derive_metrics(
    price:             Optional[float],
    mkt_cap:           Optional[float],
    income_statements: list,
    balance_sheets:    list | None = None,
    ratios:            list | None = None,
    profile_beta:      Optional[float] = None,
) -> "_DM":
    """
    Derive the full _DM metric bundle from raw fetched data.
    Applies sanity caps; returns None for any uncomputable or out-of-range metric.
    Never raises.

    PEG basis:
      - P/E   : trailing (price / most-recent-annual EPS diluted)
      - growth : 3-year annualised EPS CAGR (annualised over min(3, n-1) years)
      - PEG   : trailing P/E / EPS CAGR%  — only shown when CAGR > 5% (low-growth PEG is misleading)

    Growth outlier caps (applied before storing):
      - Revenue / EBITDA growth : ±300% — extreme base-year swings excluded
      - EPS CAGR                : ±150% — unreliable at extremes; PEG also cleared

    Interest coverage: set to None when ratios API returns 0.0 (data absent, not true zero).
    """
    dm = _DM()
    if not income_statements:
        return dm

    inc     = income_statements[0]
    bal     = (balance_sheets[0] if balance_sheets else None)
    rat     = (ratios[0] if ratios else None)
    shares  = (mkt_cap / price) if (mkt_cap and price and price > 0) else None

    # ── EPS (prefer diluted) ──────────────────────────────────────────────────
    eps = inc.eps_diluted or inc.eps
    if eps is None and inc.net_income and shares and shares > 0:
        eps = inc.net_income / shares

    # ── P/E ──────────────────────────────────────────────────────────────────
    if price and eps and eps > 0:
        raw = price / eps
        if 0 < raw <= _PE_MAX:
            dm.pe = round(raw, 2)

    # ── P/S ──────────────────────────────────────────────────────────────────
    if mkt_cap and inc.revenue and inc.revenue > 0:
        raw = mkt_cap / inc.revenue
        if 0 < raw <= _PS_MAX:
            dm.ps = round(raw, 2)

    # ── EV/EBITDA ────────────────────────────────────────────────────────────
    if mkt_cap and inc.ebitda and inc.ebitda > 0:
        debt = (bal.total_debt or 0.0) if bal else 0.0
        cash = (bal.cash_and_equivalents or 0.0) if bal else 0.0
        ev   = mkt_cap + debt - cash
        if ev > 0:
            raw = ev / inc.ebitda
            if 0 < raw <= _EV_EBITDA_MAX:
                dm.ev_ebitda = round(raw, 2)

    # ── EPS CAGR & PEG ───────────────────────────────────────────────────────
    eps_series: list[float] = []
    for stmt in income_statements:
        e = stmt.eps_diluted or stmt.eps
        if e is None and shares and shares > 0 and stmt.net_income:
            e = stmt.net_income / shares
        if e is not None:
            eps_series.append(e)

    if len(eps_series) >= 2:
        n      = min(len(eps_series) - 1, 3)
        oldest = eps_series[n]
        latest = eps_series[0]
        if oldest > 0 and latest > 0:
            cagr          = (latest / oldest) ** (1.0 / n) - 1.0
            dm.growth_pct = round(cagr * 100, 1)
            if dm.pe and dm.growth_pct and dm.growth_pct > 0:
                raw_peg = dm.pe / dm.growth_pct
                if 0 < raw_peg <= _PEG_MAX:
                    dm.peg = round(raw_peg, 2)

    # ── Revenue growth (YoY) ─────────────────────────────────────────────────
    if len(income_statements) >= 2:
        r0 = income_statements[0].revenue
        r1 = income_statements[1].revenue
        if r0 is not None and r1 and r1 > 0:
            dm.revenue_growth = round((r0 / r1 - 1) * 100, 1)

    # ── EBITDA growth (YoY) ──────────────────────────────────────────────────
    if len(income_statements) >= 2:
        e0 = income_statements[0].ebitda
        e1 = income_statements[1].ebitda
        if e0 is not None and e1 and e1 > 0:
            dm.ebitda_growth = round((e0 / e1 - 1) * 100, 1)

    # ── Profitability margins ─────────────────────────────────────────────────
    # Prefer ratio API values (most reliable); fall back to income statement ratios
    if rat and rat.gross_margin is not None:
        dm.gross_margin = round(rat.gross_margin, 4)
    elif inc.gross_profit_ratio is not None:
        dm.gross_margin = round(inc.gross_profit_ratio, 4)
    elif inc.gross_profit and inc.revenue and inc.revenue > 0:
        dm.gross_margin = round(inc.gross_profit / inc.revenue, 4)

    if rat and rat.operating_margin is not None:
        dm.operating_margin = round(rat.operating_margin, 4)
    elif inc.operating_income_ratio is not None:
        dm.operating_margin = round(inc.operating_income_ratio, 4)
    elif inc.operating_income and inc.revenue and inc.revenue > 0:
        dm.operating_margin = round(inc.operating_income / inc.revenue, 4)

    if rat and rat.net_margin is not None:
        dm.net_margin = round(rat.net_margin, 4)
    elif inc.net_income_ratio is not None:
        dm.net_margin = round(inc.net_income_ratio, 4)
    elif inc.net_income and inc.revenue and inc.revenue > 0:
        dm.net_margin = round(inc.net_income / inc.revenue, 4)

    # ── Return metrics (ratios API only — too fragile to derive) ─────────────
    if rat:
        if rat.roe is not None:
            dm.roe = round(rat.roe, 4)
        if rat.roic is not None:
            dm.roic = round(rat.roic, 4)

    # ── Financial health ─────────────────────────────────────────────────────
    if rat and rat.debt_to_equity is not None:
        dm.debt_equity = round(rat.debt_to_equity, 3)
    elif bal and bal.total_equity and bal.total_equity > 0 and bal.total_debt is not None:
        dm.debt_equity = round(bal.total_debt / bal.total_equity, 3)

    if rat and rat.current_ratio is not None:
        dm.current_ratio = round(rat.current_ratio, 2)
    elif bal and bal.total_current_liabilities and bal.total_current_liabilities > 0 and bal.total_current_assets:
        dm.current_ratio = round(bal.total_current_assets / bal.total_current_liabilities, 2)

    if rat and rat.interest_coverage is not None and rat.interest_coverage > 0:
        dm.interest_coverage = round(rat.interest_coverage, 2)
    elif inc.operating_income and inc.interest_expense and inc.interest_expense > 0:
        dm.interest_coverage = round(inc.operating_income / inc.interest_expense, 2)

    # ── Beta ─────────────────────────────────────────────────────────────────
    if profile_beta is not None:
        dm.beta = round(profile_beta, 2)

    # ── Sanity-check margins (must be in [-1, 1]) ─────────────────────────────
    # Ratios API sometimes emits values outside this range on data errors.
    if dm.gross_margin is not None and not (-1.0 <= dm.gross_margin <= 1.0):
        dm.gross_margin = None
    if dm.operating_margin is not None and not (-2.0 <= dm.operating_margin <= 1.0):
        dm.operating_margin = None
    if dm.net_margin is not None and not (-2.0 <= dm.net_margin <= 1.0):
        dm.net_margin = None

    # ── Cap growth extremes before storing (outliers distort medians) ─────────
    # Revenue / EBITDA growth beyond ±300% is almost always a base-year anomaly.
    _GROWTH_CAP = 300.0
    if dm.revenue_growth is not None and abs(dm.revenue_growth) > _GROWTH_CAP:
        dm.revenue_growth = None
    if dm.ebitda_growth is not None and abs(dm.ebitda_growth) > _GROWTH_CAP:
        dm.ebitda_growth = None
    # EPS CAGR beyond ±150% is unreliable; PEG based on it is also unreliable.
    if dm.growth_pct is not None and abs(dm.growth_pct) > 150.0:
        dm.growth_pct = None
        dm.peg = None

    # ── Validate current ratio and D/E (must be non-negative) ────────────────
    if dm.current_ratio is not None and dm.current_ratio < 0:
        dm.current_ratio = None
    if dm.debt_equity is not None and dm.debt_equity < 0:
        dm.debt_equity = None

    return dm


# ── Median helpers ─────────────────────────────────────────────────────────────

def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return (s[n // 2 - 1] + s[n // 2]) / 2 if n % 2 == 0 else s[n // 2]


def _filtered_median(values: list[float], low: float = -300.0, high: float = 300.0) -> Optional[float]:
    """Compute median after excluding values outside [low, high]. Returns None if no values remain."""
    filtered = [v for v in values if low <= v <= high]
    if not filtered:
        return None
    return _median(filtered)


# ── Peer quality scoring ──────────────────────────────────────────────────────

def _score_completeness(row: PeerRow) -> int:
    """Higher = more data available. Used to rank peers when truncating to _MAX_PEERS."""
    s = 0
    # Valuation
    if row.pe is not None:            s += 3
    if row.ps is not None:            s += 2
    if row.ev_ebitda is not None:     s += 2
    if row.growth_pct is not None:    s += 2
    if row.peg is not None and _peg_is_meaningful_check(row.growth_pct, row.peg): s += 3
    # Growth
    if row.revenue_growth is not None:   s += 2
    if row.ebitda_growth is not None:    s += 1
    # Profitability
    if row.gross_margin is not None:     s += 2
    if row.operating_margin is not None: s += 2
    if row.net_margin is not None:       s += 2
    if row.roe is not None:              s += 2
    if row.roic is not None:             s += 2
    # Financial health
    if row.debt_equity is not None:      s += 2
    if row.current_ratio is not None:    s += 1
    if row.interest_coverage is not None: s += 1
    # Market
    if row.beta is not None:             s += 1
    return s


def _is_usable(row: PeerRow) -> bool:
    """A peer is usable if it has at least ONE valid metric."""
    return any(v is not None for v in [
        row.pe, row.ps, row.ev_ebitda, row.growth_pct,
        row.gross_margin, row.operating_margin, row.revenue_growth,
        row.debt_equity, row.current_ratio,
    ])


def _peg_is_meaningful_check(growth_pct: Optional[float], peg: Optional[float]) -> bool:
    return (
        peg is not None
        and growth_pct is not None
        and growth_pct > 5.0
        and peg < 100
    )


# ── Insight generation ─────────────────────────────────────────────────────────

def _peg_is_meaningful(row: PeerRow) -> bool:
    return _peg_is_meaningful_check(row.growth_pct, row.peg)


def _rel(target_val: Optional[float], peer_vals: list[float], high_is_good: bool = True) -> str:
    """Return 'premium'/'discount'/'in line' or 'above'/'below'/'in line' label.

    Uses (target - median) / |median| as the relative difference. This works
    correctly when either value is negative — ratio-based comparison (target / median)
    is wrong when median is negative, e.g. target=+19.4, median=-14.5 gives
    ratio=-1.34 which falsely reads as "below median".
    """
    if target_val is None or not peer_vals:
        return ""
    med = _median(peer_vals)
    abs_med = abs(med)
    if abs_med == 0:
        if high_is_good:
            return "above median" if target_val > 0 else "below median"
        return "premium to peers" if target_val > 0 else "discount to peers"
    pct_diff = (target_val - med) / abs_med
    if high_is_good:
        if pct_diff > 0.15:   return "above median"
        if pct_diff < -0.15:  return "below median"
        return "in line with median"
    else:
        if pct_diff > 0.15:   return "premium to peers"
        if pct_diff < -0.15:  return "discount to peers"
        return "in line with peers"


def _generate_insights(rows: list[PeerRow]) -> list[str]:
    """
    Generate one insight per analytical dimension (valuation, growth,
    profitability, financial health, market). Returned as a 5-element list
    (one per section); empty string if no data for that section.
    """
    if len(rows) < 2:
        return ["Insufficient peer data for comparison."]

    target   = rows[0]
    peers    = [r for r in rows if not r.is_target]
    insights: list[str] = []

    # ── 1. Valuation ─────────────────────────────────────────────────────────
    peg_rows  = [r for r in rows if _peg_is_meaningful(r)]
    peer_pes  = [r.pe for r in peers if r.pe is not None]
    val_parts: list[str] = []

    if target.pe is not None and len(peer_pes) >= 2:
        med_pe = _median(peer_pes)
        label  = _rel(target.pe, peer_pes, high_is_good=False)
        val_parts.append(f"P/E of {target.pe:.1f}x is {label} ({med_pe:.1f}x median)")

    if _peg_is_meaningful(target) and len(peg_rows) >= _MIN_VALID_FOR_CONCLUSION:
        peer_pegs = [r.peg for r in peg_rows if not r.is_target and r.peg is not None]
        if peer_pegs:
            med_peg = _median(peer_pegs)
            # Absolute PEG interpretation takes precedence over relative-to-peers label.
            # PEG ≈ 1.0 is the standard "fairly valued" threshold — do NOT call this "undervalued".
            if 0.8 <= target.peg <= 1.2:
                peg_abs_label = "fairly valued relative to growth"
            elif target.peg < 0.8:
                peg_abs_label = "growth-adjusted discount"
            elif target.peg < 2.0:
                peg_abs_label = "premium to growth"
            else:
                peg_abs_label = "significantly premium to growth"
            # Add relative-to-peers context when meaningful
            if target.peg < med_peg * 0.85:
                val_parts.append(f"PEG {target.peg:.2f}x — {peg_abs_label} (cheapest vs. peers)")
            elif target.peg > med_peg * 1.15:
                val_parts.append(f"PEG {target.peg:.2f}x — {peg_abs_label} (most expensive vs. peers)")
            else:
                val_parts.append(f"PEG {target.peg:.2f}x — {peg_abs_label}")

    if val_parts:
        insights.append(". ".join(val_parts) + ".")
    else:
        insights.append("")

    # ── 2. Growth ─────────────────────────────────────────────────────────────
    peer_rev_g = [r.revenue_growth for r in peers if r.revenue_growth is not None]
    peer_eps_g = [r.growth_pct for r in peers if r.growth_pct is not None]
    grow_parts: list[str] = []

    if target.revenue_growth is not None and len(peer_rev_g) >= 2:
        med = _filtered_median(peer_rev_g)
        if med is not None:
            label = _rel(target.revenue_growth, peer_rev_g, high_is_good=True)
            grow_parts.append(f"Revenue growth {target.revenue_growth:+.1f}% vs. peer median {med:+.1f}% — {label}")

    if target.growth_pct is not None and len(peer_eps_g) >= 2:
        med = _filtered_median(peer_eps_g, low=-150.0, high=150.0)
        if med is not None:
            label = _rel(target.growth_pct, peer_eps_g, high_is_good=True)
            grow_parts.append(f"EPS CAGR {target.growth_pct:.1f}% — {label} ({med:.1f}%)")

    if grow_parts:
        insights.append(". ".join(grow_parts) + ".")
    else:
        insights.append("")

    # ── 3. Profitability ──────────────────────────────────────────────────────
    peer_gm  = [r.gross_margin for r in peers if r.gross_margin is not None]
    peer_om  = [r.operating_margin for r in peers if r.operating_margin is not None]
    peer_roe = [r.roe for r in peers if r.roe is not None]
    prof_parts: list[str] = []

    if target.gross_margin is not None and len(peer_gm) >= 2:
        med   = _median(peer_gm)
        label = _rel(target.gross_margin, peer_gm, high_is_good=True)
        prof_parts.append(f"Gross margin {target.gross_margin*100:.1f}% — {label} ({med*100:.1f}%)")

    if target.operating_margin is not None and len(peer_om) >= 2:
        med   = _median(peer_om)
        label = _rel(target.operating_margin, peer_om, high_is_good=True)
        prof_parts.append(f"Operating margin {target.operating_margin*100:.1f}% vs. {med*100:.1f}%")

    if target.roe is not None and len(peer_roe) >= 2:
        med   = _median(peer_roe)
        label = _rel(target.roe, peer_roe, high_is_good=True)
        prof_parts.append(f"ROE {target.roe*100:.1f}% — {label}")

    if prof_parts:
        insights.append(". ".join(prof_parts) + ".")
    else:
        insights.append("")

    # ── 4. Financial health ───────────────────────────────────────────────────
    peer_de  = [r.debt_equity for r in peers if r.debt_equity is not None]
    peer_cr  = [r.current_ratio for r in peers if r.current_ratio is not None]
    health_parts: list[str] = []

    if target.debt_equity is not None and len(peer_de) >= 2:
        med   = _median(peer_de)
        label = "below median (less leveraged)" if target.debt_equity < med * 0.85 else \
                "above median (more leveraged)" if target.debt_equity > med * 1.15 else \
                "in line with median"
        health_parts.append(f"D/E of {target.debt_equity:.2f}x — {label} ({med:.2f}x)")

    if target.current_ratio is not None and len(peer_cr) >= 2:
        med = _median(peer_cr)
        health_parts.append(f"current ratio {target.current_ratio:.2f}x vs. peer median {med:.2f}x")

    if health_parts:
        insights.append(". ".join(health_parts) + ".")
    else:
        insights.append("")

    # ── 5. Market / Risk ──────────────────────────────────────────────────────
    peer_beta = [r.beta for r in peers if r.beta is not None]
    mkt_parts: list[str] = []

    if target.beta is not None:
        if len(peer_beta) >= 2:
            med = _median(peer_beta)
            label = "higher-beta (more volatile)" if target.beta > med * 1.2 else \
                    "lower-beta (more defensive)" if target.beta < med * 0.8 else \
                    "similar volatility profile"
            mkt_parts.append(f"Beta {target.beta:.2f} — {label} vs. peer median {med:.2f}")
        else:
            risk_label = "high-beta" if target.beta > 1.5 else "low-beta" if target.beta < 0.8 else "market-correlated"
            mkt_parts.append(f"Beta {target.beta:.2f} — {risk_label} stock")

    if mkt_parts:
        insights.append(". ".join(mkt_parts) + ".")
    else:
        insights.append("")

    return insights   # Always 5 elements


# ── Per-ticker fetch ────────────────────────────────────────────────────────────

def _row_from_dm(
    pticker:  str,
    name:     str,
    mkt_cap:  Optional[float],
    dm:       "_DM",
    tier:     int,
    sector:   str = "",
    industry: str = "",
) -> PeerRow:
    return PeerRow(
        ticker            = pticker,
        name              = name,
        market_cap        = mkt_cap,
        pe                = dm.pe,
        ps                = dm.ps,
        ev_ebitda         = dm.ev_ebitda,
        growth_pct        = dm.growth_pct,
        peg               = dm.peg,
        revenue_growth    = dm.revenue_growth,
        ebitda_growth     = dm.ebitda_growth,
        gross_margin      = dm.gross_margin,
        operating_margin  = dm.operating_margin,
        net_margin        = dm.net_margin,
        roe               = dm.roe,
        roic              = dm.roic,
        debt_equity       = dm.debt_equity,
        current_ratio     = dm.current_ratio,
        interest_coverage = dm.interest_coverage,
        beta              = dm.beta,
        tier              = tier,
        sector            = sector,
        industry          = industry,
    )


def _fetch_peer_row(
    pticker: str, fmp: FMPProvider, limit: int, tier: int = 0
) -> "PeerRow | None":
    """
    Fetch profile + financials for one peer.
    Returns None on any error or if the peer has zero computable metrics.
    Never raises — all errors are caught and logged.
    """
    try:
        profile_result = fmp.get_profile(pticker)
        if profile_result.is_empty():
            print(f"  [PEER] {pticker}: no profile — skipping")
            return None

        price    = profile_result.current_price
        mkt_cap  = profile_result.market_cap
        name     = getattr(profile_result.profile, "company_name", pticker) or pticker
        beta     = getattr(profile_result.profile, "beta", None)
        _psect   = getattr(profile_result.profile, "sector",   "") or ""
        _pind    = getattr(profile_result.profile, "industry", "") or ""

        fin_result = fmp.get_financials(pticker, limit)
        income  = fin_result.income_statements
        balance = fin_result.balance_sheets
        ratios  = fin_result.ratios

        dm = _derive_metrics(price, mkt_cap, income, balance, ratios, beta)
        row = _row_from_dm(pticker, name, mkt_cap, dm, tier,
                           sector=_psect, industry=_pind)
        row.historical = _derive_historical(income, ratios, n=6)
        score  = _score_completeness(row)
        usable = _is_usable(row)
        print(
            f"  [PEER] {pticker} (tier={tier}): "
            f"PE={dm.pe} PS={dm.ps} EV/EBITDA={dm.ev_ebitda} growth={dm.growth_pct}% "
            f"rev_g={dm.revenue_growth}% gm={dm.gross_margin} om={dm.operating_margin} "
            f"de={dm.debt_equity} cr={dm.current_ratio} beta={dm.beta} "
            f"score={score} usable={usable}"
        )
        return row if usable else None

    except FMPError as exc:
        print(f"  [PEER] {pticker}: FMPError — {exc} — skipping")
        raise   # propagate so caller can distinguish rate-limit errors
    except Exception as exc:
        print(f"  [PEER] {pticker}: error — {exc} — skipping")
        return None


# ── Candidate pool builder ─────────────────────────────────────────────────────

def _build_candidate_pool(
    target_ticker:   str,
    sector:          str,
    industry:        str,
    target_mkt_cap:  Optional[float],
    fmp:             FMPProvider,
    # Adaptive market cap band multipliers (from PeerSelectionEngine)
    screener_lo:     float = 0.33,
    screener_hi:     float = 3.0,
) -> list[tuple[str, int]]:
    """
    Return a list of (ticker, tier) candidates, deduplicated, target excluded.
    Tiers:
      0 = FMP /stock-peers (dynamic, most precise)
      1 = _TICKER_PEERS (curated)
      2 = FMP /stock-screener (sector+industry, adaptive mktcap band)

    Tiers 3 and 4 are permanently removed. No sector or global fallback.
    screener_lo / screener_hi are set by the caller from PeerSelectionEngine
    archetype bands — tighter for Banks, wider for Fintech/Exchanges.
    """
    seen: set[str] = {target_ticker.upper()}
    pool: list[tuple[str, int]] = []

    def _add(tickers: list[str], tier: int) -> None:
        for t in tickers:
            u = t.upper()
            if u not in seen:
                seen.add(u)
                pool.append((t, tier))

    # Tier 0: FMP /stock-peers
    tier0: list[str] = []
    try:
        tier0 = fmp.get_peers(target_ticker)
        print(
            f"  [PEER] {target_ticker}: tier-0 /stock-peers → "
            f"{len(tier0)} peers: {tier0}"
        )
    except Exception as exc:
        print(f"  [PEER] {target_ticker}: tier-0 /stock-peers failed ({exc})")
    _add(tier0, 0)

    # Tier 1: curated ticker-specific
    tier1 = _TICKER_PEERS.get(target_ticker.upper(), [])
    _add(tier1, 1)

    # Tier 2: screener — same sector + industry, adaptive mktcap band
    tier2: list[str] = []
    if sector or industry:
        min_cap = (target_mkt_cap * screener_lo) if target_mkt_cap else None
        max_cap = (target_mkt_cap * screener_hi) if target_mkt_cap else None
        try:
            raw = fmp.get_screener(
                sector=sector, industry=industry,
                min_mkt_cap=min_cap, max_mkt_cap=max_cap, limit=15,
            )
            tier2 = raw
            print(
                f"  [PEER] {target_ticker}: tier-2 screener "
                f"({sector}/{industry}, band {screener_lo:.2f}x–{screener_hi:.1f}x)"
                f" → {len(tier2)} candidates"
            )
        except Exception as exc:
            print(f"  [PEER] {target_ticker}: tier-2 screener failed ({exc})")
    _add(tier2, 2)

    # Tiers 3 and 4 (sector universe + global fallback) are REMOVED.
    # PeerSelectionEngine is the sole source of peers. If tiers 0–2 produce
    # no usable candidates the section reports "No valid peers available" —
    # sector or global fallback tickers are never substituted.

    print(
        f"  [PEER] {target_ticker}: total pool={len(pool)} "
        f"(t0={len(tier0)} t1={len(tier1)} t2={len(tier2)}) "
        f"| sector={sector!r} industry={industry!r}"
    )
    return pool


# ── Relaxation cascade helpers ────────────────────────────────────────────────

def _has_min_metrics(row: PeerRow, min_count: int, metric_pool: list[str]) -> bool:
    """Return True if row has at least min_count non-None metrics from metric_pool."""
    return sum(1 for m in metric_pool if getattr(row, m, None) is not None) >= min_count


def _is_fee_based(row: PeerRow) -> bool:
    """
    Return True when the peer's margin profile is consistent with a fee-based /
    asset-light revenue model — not balance-sheet-driven intermediation.

    Applied exclusively as an additional gate for Step 4 (adjacent-archetype proxy)
    peers to enforce the "fee-based or recurring/volume-driven revenue" requirement.
    Either condition is sufficient; both absent → reject (conservative default).

    Thresholds:
      · gross_margin > 55%  — software-like data / exchange economics
      · operating_margin > 15% — operating leverage indicates asset-light model
    """
    gm = row.gross_margin
    om = row.operating_margin
    return (gm is not None and gm > 0.55) or (om is not None and om > 0.15)


def _select_peers_with_relaxation(
    target_row:     PeerRow,
    same_arch:      list[PeerRow],
    adj_arch:       list[PeerRow],
    target_arch:    Archetype,
    pse:            PeerSelectionEngine,
    target_mkt_cap: Optional[float],
    target_ps:      Optional[float],
    min_peers:      int = 3,
) -> tuple[list[PeerRow], str]:
    """
    Controlled relaxation cascade — returns (selected_peers, fallback_level_used).

    Steps run in order; stops at first step yielding >= min_peers peers:
      0  step0_baseline              structural filter (strict)  + same-arch + full ranking
      1  step1_expanded_band         structural filter (expanded) + same-arch + full ranking
      2  step2_no_structural_filter  no filter + same-arch (>=2 of 7 extended metrics)
      3  step3_simplified_metrics    no filter + same-arch (>=1 core metric) + simplified ranking
      4  step4_adjacent_archetype_proxy  same+adj arch (>=1 core metric) + simplified ranking

    Adjacent-arch peers are flagged by the caller (build_peer_comparison) via the
    [PROXY PEER] justification prefix.  This function never touches justification.
    """
    def _struct(candidates: list[PeerRow], expanded: bool) -> list[PeerRow]:
        return pse.apply_structural_filters(
            target_row,
            candidates,
            target_arch,
            target_mkt_cap=target_mkt_cap,
            target_ps=target_ps,
            target_geo=GeoRegion.US,
            expanded=expanded,
        )

    # Step 0 — baseline: strict structural filter + same-arch + full ranking
    filtered = _struct(same_arch, expanded=False)
    ranked   = pse.rank_peers(target_row, filtered, target_arch)
    if len(ranked) >= min_peers:
        print(f"  [PSE/relax] step0_baseline → {len(ranked)} peers")
        return ranked[:_MAX_PEERS], "step0_baseline"

    # Step 1 — expanded mktcap band: relaxed structural filter + same-arch
    filtered = _struct(same_arch, expanded=True)
    ranked   = pse.rank_peers(target_row, filtered, target_arch)
    if len(ranked) >= min_peers:
        print(f"  [PSE/relax] step1_expanded_band → {len(ranked)} peers")
        return ranked[:_MAX_PEERS], "step1_expanded_band"

    # Step 2 — skip structural filter: same-arch + >=2 of 7 extended metrics
    usable = [r for r in same_arch if _has_min_metrics(r, 2, _EXTENDED_METRICS)]
    ranked  = pse.rank_peers(target_row, usable, target_arch)
    if len(ranked) >= min_peers:
        print(f"  [PSE/relax] step2_no_structural_filter → {len(ranked)} peers")
        return ranked[:_MAX_PEERS], "step2_no_structural_filter"

    # Step 3 — simplified metrics: same-arch + >=1 core metric + simplified ranking
    usable = [r for r in same_arch if _has_min_metrics(r, 1, _CORE_METRICS)]
    ranked  = pse.rank_peers_simplified(target_row, usable)
    if len(ranked) >= min_peers:
        print(f"  [PSE/relax] step3_simplified_metrics → {len(ranked)} peers")
        return ranked[:_MAX_PEERS], "step3_simplified_metrics"

    # Step 4 — adjacent archetypes: last resort, any result is acceptable.
    # Gate: adjacent peers must have ≥1 core metric AND a fee-based margin profile
    # (gross_margin > 55% OR operating_margin > 15%) to confirm economic alignment.
    # This enforces the "fee-based model or recurring/volume-driven revenue" requirement
    # and prevents balance-sheet-driven intermediaries from slipping through.
    adj_usable = [
        r for r in adj_arch
        if _has_min_metrics(r, 1, _CORE_METRICS) and _is_fee_based(r)
    ]
    combined   = [r for r in same_arch if _has_min_metrics(r, 1, _CORE_METRICS)] + adj_usable
    ranked     = pse.rank_peers_simplified(target_row, combined)
    if ranked:
        print(
            f"  [PSE/relax] step4_adjacent_archetype_proxy → {len(ranked)} peers "
            f"(adj_fee_qualified={len(adj_usable)} of {len(adj_arch)} adjacent)"
        )
        return ranked[:_MAX_PEERS], "step4_adjacent_archetype_proxy"

    return [], "no_peers_available"


# ── Public entry point ─────────────────────────────────────────────────────────

def build_peer_comparison(
    target_ticker:       str,
    target_pe:           Optional[float],
    target_ps:           Optional[float],
    target_growth:       Optional[float],
    target_peg:          Optional[float],
    target_ev_ebitda:    Optional[float] = None,
    sector:              str = "",
    industry:            str = "",
    target_mkt_cap:      Optional[float] = None,
    target_company_name: str = "",
    # Extended target metrics for multi-dimensional peer comparison
    target_revenue_growth:    Optional[float] = None,
    target_gross_margin:      Optional[float] = None,
    target_operating_margin:  Optional[float] = None,
    target_net_margin:        Optional[float] = None,
    target_roe:               Optional[float] = None,
    target_roic:              Optional[float] = None,
    target_debt_equity:       Optional[float] = None,
    target_current_ratio:     Optional[float] = None,
    target_interest_coverage: Optional[float] = None,
    target_beta:              Optional[float] = None,
    # StockData for deriving target's 5-year historical (no extra API calls)
    target_stock_data:        Optional[object] = None,
) -> PeerComparison:
    """
    Build a PeerComparison for the target ticker.

    NEVER returns has_peers=False unless there is a truly fatal error —
    the 5-tier candidate pool + relaxed size filter guarantee at least one peer
    for any US-listed stock with FMP coverage.

    Fallback passes:
      Pass 1: Candidates from pool where size is within ±67% of target.
      Pass 2: If <_USABLE_PEER_TARGET found, evaluate size-deferred tickers.
      Pass 3: If still short, evaluate remaining pool without size filter at all.
    """
    print(
        f"\n  [PEER START] ticker={target_ticker!r}"
        f" PE={target_pe} PS={target_ps} EV/EBITDA={target_ev_ebitda}"
        f" growth={target_growth}% PEG={target_peg}"
        f" mkt_cap={target_mkt_cap}"
        f" sector={sector!r} industry={industry!r}"
        f" company={target_company_name!r}"
    )

    target_row = PeerRow(
        ticker            = target_ticker,
        name              = target_company_name or target_ticker,
        market_cap        = target_mkt_cap,
        pe                = target_pe,
        ps                = target_ps,
        ev_ebitda         = target_ev_ebitda,
        growth_pct        = target_growth,
        peg               = target_peg,
        revenue_growth    = target_revenue_growth,
        gross_margin      = target_gross_margin,
        operating_margin  = target_operating_margin,
        net_margin        = target_net_margin,
        roe               = target_roe,
        roic              = target_roic,
        debt_equity       = target_debt_equity,
        current_ratio     = target_current_ratio,
        interest_coverage = target_interest_coverage,
        beta              = target_beta,
        is_target         = True,
    )
    # Populate target historical from already-fetched StockData (no API call)
    if target_stock_data is not None:
        try:
            target_row.historical = _derive_historical(
                getattr(target_stock_data, "income_statements", []) or [],
                getattr(target_stock_data, "ratios", []) or [],
                n=6,
            )
        except Exception:
            pass

    _empty = PeerComparison(target_ticker=target_ticker, rows=[target_row], has_peers=False)

    # Gate: target must have at least one metric to compare against
    if not any(v is not None for v in [target_pe, target_ps, target_ev_ebitda, target_growth]):
        print(f"  [PEER] {target_ticker}: no target metrics available — omitting section")
        return _empty

    try:
        fmp = FMPProvider()
    except Exception as exc:
        print(f"  [PEER] FMPProvider init failed — {exc}")
        return _empty

    # ── Classify target archetype → get adaptive mktcap bands ─────────────────
    _pse = get_engine()
    _classification = _pse.classify(
        ticker           = target_ticker,
        sector           = sector,
        industry         = industry,
        gross_margin     = target_gross_margin,
        operating_margin = target_operating_margin,
        net_margin       = target_net_margin,
        debt_equity      = target_debt_equity,
    )
    _bands = _pse.get_adaptive_bands(_classification.archetype)

    print(
        f"  [PSE] archetype={_classification.archetype.name!r} "
        f"method={_classification.method!r} "
        f"mktcap_band=[{_bands.lo:.2f}x, {_bands.hi:.1f}x] "
        f"(confidence={_classification.confidence:.2f})"
    )

    pool = _build_candidate_pool(
        target_ticker,
        sector,
        industry,
        target_mkt_cap,
        fmp,
        screener_lo=_bands.lo,
        screener_hi=_bands.hi,
    )

    # ── Adjacent-arch curated supplement (Tier 3) ─────────────────────────────
    # The FMP /stock-peers API and sector screener are scoped to the same
    # sector/industry as the target — they naturally return same-arch candidates.
    # If no adjacent-arch companies make it into the pool, adj_arch_pool (populated
    # later via pre-classification) will be empty and Step 4 of the relaxation
    # cascade can never fire.
    #
    # Fix: inject a small set of curated adjacent-arch tickers as Tier 3 candidates.
    # They are appended to the END of the pool so they are evaluated last (Pass 2/3),
    # never displacing same-arch peers. The _is_fee_based() gate in Step 4 provides
    # the final economic filter before any of these are accepted.
    _adj_curated = _ADJACENT_CURATED.get(int(_classification.archetype), [])
    if _adj_curated:
        _pool_seen: set[str] = {t.upper() for t, _ in pool} | {target_ticker.upper()}
        _injected = 0
        for _cticker in _adj_curated:
            _cu = _cticker.upper()
            if _cu not in _pool_seen:
                _pool_seen.add(_cu)
                pool.append((_cticker, 3))   # Tier 3 = adjacent-arch curated
                _injected += 1
        if _injected:
            print(
                f"  [PEER] adjacent-arch curated supplement: "
                f"+{_injected} Tier-3 candidates for {_classification.archetype.name!r} target"
            )

    if not pool:
        print(f"  [PEER] {target_ticker}: pool is empty — omitting")
        return _empty

    # Need 7 statements for 6 years of YoY growth (oldest period has no prior for growth calc).
    # Same API call, larger response — no extra round trips.
    limit = min(Config.FINANCIAL_STATEMENT_LIMIT, 7)
    usable_rows:  list[PeerRow] = []
    size_deferred: list[tuple[str, int]] = []
    later_pool:    list[tuple[str, int]] = []
    candidates_tried = 0

    # ── Pass 1: size-filtered (adaptive band from archetype) ─────────────────
    _pass1_lo = _bands.lo
    _pass1_hi = _bands.hi
    print(
        f"  [PEER] === PASS 1: size-filtered "
        f"({_pass1_lo:.2f}x–{_pass1_hi:.1f}x of {target_mkt_cap}) ==="
    )
    for pticker, tier in pool:
        if len(usable_rows) >= _USABLE_PEER_TARGET:
            # Keep remaining for later passes if needed
            later_pool.append((pticker, tier))
            continue
        if candidates_tried >= _MAX_CANDIDATES:
            later_pool.append((pticker, tier))
            continue

        candidates_tried += 1

        # Market cap size filter
        if target_mkt_cap:
            try:
                pf = fmp.get_profile(pticker)
                if pf.is_empty():
                    print(f"  [PEER] {pticker}: no profile — skipping")
                    continue
                peer_mc = pf.market_cap
                # Sector consistency guard: tier-0 (FMP /stock-peers) sometimes
                # returns cross-sector results. If we have a known target sector,
                # reject peers from a completely different sector when we still
                # have room to find better comps (i.e. not desperate for any peer).
                if sector and len(usable_rows) < _USABLE_PEER_TARGET:
                    peer_sector = getattr(getattr(pf, "profile", None), "sector", "") or ""
                    if peer_sector and peer_sector.lower() != sector.lower():
                        print(
                            f"  [PEER] {pticker}: sector mismatch "
                            f"({peer_sector!r} vs target {sector!r}) — skipping"
                        )
                        candidates_tried -= 1
                        continue
                if peer_mc and target_mkt_cap:
                    ratio = peer_mc / target_mkt_cap
                    if not (_pass1_lo <= ratio <= _pass1_hi):
                        print(
                            f"  [PEER] {pticker}: size ratio={ratio:.2f} outside "
                            f"[{_pass1_lo:.2f}, {_pass1_hi:.1f}] — deferred"
                        )
                        size_deferred.append((pticker, tier))
                        candidates_tried -= 1
                        continue
                # Reuse fetched profile
                price    = pf.current_price
                mkt_cap  = pf.market_cap
                name     = getattr(pf.profile, "company_name", pticker) or pticker
                beta     = getattr(pf.profile, "beta", None)
                _psect   = getattr(pf.profile, "sector",   "") or ""
                _pind    = getattr(pf.profile, "industry", "") or ""
                fin_result = fmp.get_financials(pticker, limit)
                income  = fin_result.income_statements
                balance = fin_result.balance_sheets
                ratios  = fin_result.ratios
                dm  = _derive_metrics(price, mkt_cap, income, balance, ratios, beta)
                row = _row_from_dm(pticker, name, mkt_cap, dm, tier,
                                   sector=_psect, industry=_pind)
                row.historical = _derive_historical(income, ratios, n=6)
                score  = _score_completeness(row)
                usable = _is_usable(row)
                print(
                    f"  [PEER] {pticker} (tier={tier}): "
                    f"PE={dm.pe} PS={dm.ps} EV/EBITDA={dm.ev_ebitda} growth={dm.growth_pct}% "
                    f"gm={dm.gross_margin} om={dm.operating_margin} de={dm.debt_equity} "
                    f"beta={dm.beta} score={score} usable={usable}"
                )
                if usable:
                    usable_rows.append(row)
                else:
                    print(f"  [PEER] {pticker}: no computable metrics — excluded")
            except FMPError as exc:
                print(f"  [PEER] {pticker}: FMP error — {exc} — skipping")
            except Exception as exc:
                print(f"  [PEER] {pticker}: error — {exc}")
        else:
            try:
                row = _fetch_peer_row(pticker, fmp, limit, tier)
                if row is not None:
                    usable_rows.append(row)
            except FMPError as exc:
                print(f"  [PEER] {pticker}: FMP error — {exc} — skipping")

    print(f"  [PEER] Pass 1 complete: {len(usable_rows)} usable, {len(size_deferred)} deferred")

    # ── Pass 2: size-deferred (relaxed size) ──────────────────────────────────
    if len(usable_rows) < _USABLE_PEER_TARGET and size_deferred:
        print(f"  [PEER] === PASS 2: size-deferred ({len(size_deferred)} tickers) ===")
        for pticker, tier in size_deferred:
            if len(usable_rows) >= _USABLE_PEER_TARGET:
                break
            try:
                row = _fetch_peer_row(pticker, fmp, limit, tier)
                if row is not None:
                    usable_rows.append(row)
            except FMPError as exc:
                print(f"  [PEER] {pticker}: FMP error (pass2) — {exc} — skipping")
        print(f"  [PEER] Pass 2 complete: {len(usable_rows)} usable")

    # ── Pass 3: remaining pool, no size constraint ────────────────────────────
    if len(usable_rows) < _USABLE_PEER_TARGET and later_pool:
        print(f"  [PEER] === PASS 3: remaining pool ({len(later_pool)} tickers, no size filter) ===")
        for pticker, tier in later_pool:
            if len(usable_rows) >= _USABLE_PEER_TARGET:
                break
            try:
                row = _fetch_peer_row(pticker, fmp, limit, tier)
                if row is not None:
                    usable_rows.append(row)
            except FMPError as exc:
                print(f"  [PEER] {pticker}: FMP error (pass3) — {exc} — skipping")
        print(f"  [PEER] Pass 3 complete: {len(usable_rows)} usable")

    # ── Final gate: require at least 1 usable peer ────────────────────────────
    if len(usable_rows) < _MIN_USABLE_PEERS:
        print(
            f"  [PEER] {len(usable_rows)} usable peer(s) — below minimum {_MIN_USABLE_PEERS} — omitting"
        )
        return _empty

    # ── Pre-classify all usable rows → partition into same-arch / adjacent pools ─
    target_arch = _classification.archetype
    _adj_set    = _ADJACENT.get(int(target_arch), set())
    same_arch_pool: list[PeerRow] = []
    adj_arch_pool:  list[PeerRow] = []

    for row in usable_rows:
        peer_cls = _pse.classify(
            ticker           = row.ticker,
            sector           = row.sector,     # from FMP profile — improves classification
            industry         = row.industry,   # from FMP profile — reduces fallback to OTHER
            gross_margin     = row.gross_margin,
            operating_margin = row.operating_margin,
            net_margin       = row.net_margin,
            debt_equity      = row.debt_equity,
        )
        row.peer_archetype = int(peer_cls.archetype)
        if peer_cls.archetype == target_arch:
            same_arch_pool.append(row)
        elif int(peer_cls.archetype) in _adj_set:
            adj_arch_pool.append(row)
        # Excluded archetypes (Banks, Insurance, etc.) are not added to any pool

    print(
        f"  [PSE] archetype pools for {target_ticker}: "
        f"same={len(same_arch_pool)} adj={len(adj_arch_pool)} "
        f"excluded={len(usable_rows) - len(same_arch_pool) - len(adj_arch_pool)} "
        f"(target_arch={target_arch.name!r})"
    )

    # ── Relaxation cascade ────────────────────────────────────────────────────
    selected, fallback_level = _select_peers_with_relaxation(
        target_row     = target_row,
        same_arch      = same_arch_pool,
        adj_arch       = adj_arch_pool,
        target_arch    = target_arch,
        pse            = _pse,
        target_mkt_cap = target_mkt_cap,
        target_ps      = target_ps,
        min_peers      = 3,
    )

    print(
        f"  [PSE] {target_ticker}: peers selected at level={fallback_level!r} "
        f"({len(selected)} peers)"
    )

    if fallback_level == "no_peers_available":
        print(f"  [PEER] {target_ticker}: no peers found at any relaxation level — omitting")
        _empty.insights = ["No valid peers available — insufficient coverage in tiers 0–2."]
        return _empty

    # ── Quality score and confidence level per fallback stage ────────────────
    # Scores reflect archetype alignment and data richness at each cascade step.
    # These are written to PeerRow fields consumed by the renderer and the API.
    _STAGE_QUALITY: dict[str, float] = {
        "step0_baseline":              100.0,  # same-arch, strict filter, full metrics
        "step1_expanded_band":          88.0,  # same-arch, wider band, full metrics
        "step2_no_structural_filter":   72.0,  # same-arch, no filter, ≥2 metrics
        "step3_simplified_metrics":     58.0,  # same-arch, no filter, ≥1 core metric
        "step4_adjacent_archetype_proxy": 38.0,  # adjacent-arch, proxy comp
    }
    _STAGE_CONFIDENCE: dict[str, str] = {
        "step0_baseline":              "High",
        "step1_expanded_band":         "High",
        "step2_no_structural_filter":  "Medium",
        "step3_simplified_metrics":    "Medium",
        "step4_adjacent_archetype_proxy": "Low",
    }
    _base_quality    = _STAGE_QUALITY.get(fallback_level, 50.0)
    _base_confidence = _STAGE_CONFIDENCE.get(fallback_level, "Medium")

    # ── Generate justifications + annotate quality fields ─────────────────────
    for row in selected:
        peer_arch = Archetype(row.peer_archetype) if row.peer_archetype else target_arch
        just = _pse.justify_peer(
            target_ticker    = target_ticker,
            target_archetype = target_arch,
            peer_ticker      = row.ticker,
            peer_name        = row.name,
            peer_archetype   = peer_arch,
            peer_mkt_cap     = row.market_cap or 0.0,
        )

        # Proxy peers (adjacent-archetype, intentionally included at Step 4)
        _is_proxy = (peer_arch != target_arch and int(peer_arch) in _adj_set)
        if _is_proxy:
            row.justification   = "[PROXY PEER] " + just
            row.is_proxy        = True
            row.peer_confidence = "Low"
        else:
            row.justification   = just
            row.is_proxy        = False
            row.peer_confidence = _base_confidence

        # Quality score: start from stage baseline, adjust for metric completeness.
        # Completeness bonus: +0 to +8 points based on data richness (0–28 point scale).
        completeness = _score_completeness(row)
        completeness_bonus = round(min(completeness / 28.0, 1.0) * 8.0, 1)
        row.quality_score = round(
            min(100.0, (_base_quality if not _is_proxy else 38.0) + completeness_bonus),
            1,
        )

    print(
        f"  [PEER] FINAL: selected {len(selected)} peers for {target_ticker} "
        f"(archetype={_classification.archetype.name}):"
    )
    for r in selected:
        print(
            f"    {r.ticker} (tier={r.tier}): "
            f"PE={r.pe} PS={r.ps} EV/EBITDA={r.ev_ebitda} "
            f"growth={r.growth_pct}% PEG={r.peg} completeness={_score_completeness(r)}"
        )

    # ── Map cascade step → 3-tier public level ───────────────────────────────
    _LEVEL_MAP: dict[str, int] = {
        "step0_baseline":                1,
        "step1_expanded_band":           1,
        "step2_no_structural_filter":    2,
        "step3_simplified_metrics":      2,
        "step4_adjacent_archetype_proxy": 3,
    }
    _peer_level = _LEVEL_MAP.get(fallback_level, 1)

    _section_label = (
        "Proxy Peer Comparison (Adjacent Business Models)"
        if _peer_level == 3
        else "Peer Comparison"
    )
    _proxy_note = (
        "Peers selected based on economic similarity rather than strict classification. "
        "No direct archetype match was available; adjacent fee-based models are used as proxies."
        if _peer_level == 3
        else ""
    )

    print(
        f"  [PEER] level={_peer_level} section={_section_label!r} "
        f"proxy_note={'yes' if _proxy_note else 'no'}"
    )

    pc = PeerComparison(target_ticker=target_ticker)
    pc.rows          = [target_row] + selected
    pc.has_peers     = True
    pc.insights      = _generate_insights(pc.rows)
    pc.peer_level    = _peer_level
    pc.section_label = _section_label
    pc.proxy_note    = _proxy_note

    print(f"  [PEER] Insights: {pc.insights}")
    return pc
