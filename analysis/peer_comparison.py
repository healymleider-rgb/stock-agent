"""
Peer comparison analysis.

Fetches P/E, P/S, EV/EBITDA, EPS growth, and PEG for up to _MAX_PEERS peers
and builds a comparative table alongside the target stock.

Candidate pool — 5 tiers, merged in priority order:
  0. FMP /stock-peers       — dynamic, same sector + industry (most precise)
  1. _TICKER_PEERS          — curated industry-specific maps for known tickers
  2. FMP /stock-screener    — same sector + industry, ±67% market cap
  3. _PEER_UNIVERSE         — sector-level broad fallback
  4. _GLOBAL_FALLBACK       — large-cap last resort (prevents empty sections)

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
_PEER_UNIVERSE: dict[str, list[str]] = {
    "Technology":             ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AVGO", "QCOM", "ORCL", "CRM", "ADBE"],
    "Consumer Cyclical":      ["AMZN", "TSLA", "HD",    "NKE",  "MCD",  "SBUX", "LOW",  "TJX",  "BKNG", "MAR"],
    "Consumer Defensive":     ["WMT",  "KO",   "PEP",   "COST", "PG",   "MO",   "PM",   "CL",   "KMB",  "GIS"],
    "Healthcare":             ["UNH",  "JNJ",  "LLY",   "ABBV", "MRK",  "PFE",  "ABT",  "TMO",  "DHR",  "AMGN"],
    "Financial Services":     ["JPM",  "BAC",  "WFC",   "GS",   "MS",   "BLK",  "C",    "SCHW", "AXP",  "V"],
    "Energy":                 ["XOM",  "CVX",  "COP",   "EOG",  "SLB",  "PSX",  "VLO",  "MPC",  "OXY",  "HAL"],
    "Industrials":            ["CAT",  "GE",   "HON",   "RTX",  "UPS",  "DE",   "BA",   "LMT",  "NOC",  "EMR"],
    "Communication Services": ["GOOGL","META", "NFLX",  "DIS",  "CMCSA","T",    "VZ",   "CHTR", "TMUS", "WBD"],
    "Basic Materials":        ["LIN",  "APD",  "SHW",   "ECL",  "NEM",  "FCX",  "NUE",  "DOW",  "LYB",  "ALB"],
    "Real Estate":            ["AMT",  "PLD",  "EQIX",  "SPG",  "O",    "WY",   "DLR",  "PSA",  "EXR",  "AVB"],
    "Utilities":              ["NEE",  "DUK",  "SO",    "D",    "EXC",  "AEP",  "SRE",  "PCG",  "XEL",  "WEC"],
}

# ── Global last-resort fallback (tier 4) ──────────────────────────────────────
_GLOBAL_FALLBACK: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "AVGO",  "JPM",  "V",
    "WMT",  "UNH",  "LLY",   "XOM",  "MA",
]

_MAX_PEERS              = 4    # maximum peers shown in the table
_MAX_CANDIDATES         = 15   # maximum tickers evaluated before stopping
_MIN_VALID_FOR_CONCLUSION = 3  # rows needed before making a relative-stat claim
_USABLE_PEER_TARGET     = 4    # ideal peers to find
_MIN_USABLE_PEERS       = 1    # show section with any usable peer

# Sanity caps — ratios beyond these are not meaningful comparisons
_PE_MAX       = 500.0
_PS_MAX       = 100.0
_EV_EBITDA_MAX = 200.0
_PEG_MAX      = 100.0


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class PeerRow:
    ticker:     str
    name:       str            = ""     # company name for display
    pe:         Optional[float] = None
    ps:         Optional[float] = None
    ev_ebitda:  Optional[float] = None
    growth_pct: Optional[float] = None   # annualised EPS CAGR, e.g. 12.5
    peg:        Optional[float] = None
    is_target:  bool           = False
    tier:       int            = 0      # which candidate tier supplied this peer


@dataclass
class PeerComparison:
    target_ticker: str
    rows:      list[PeerRow] = field(default_factory=list)
    insights:  list[str]    = field(default_factory=list)
    has_peers: bool = False


# ── Metric derivation ──────────────────────────────────────────────────────────

def _derive_metrics(
    price:             Optional[float],
    mkt_cap:           Optional[float],
    income_statements: list,
    balance_sheets:    list | None = None,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Return (pe, ps, ev_ebitda, growth_pct, peg) from raw data.
    Applies sanity caps; returns None for any uncomputable or out-of-range metric.
    """
    if not income_statements:
        return None, None, None, None, None

    inc     = income_statements[0]
    bal     = (balance_sheets[0] if balance_sheets else None)
    shares  = (mkt_cap / price) if (mkt_cap and price and price > 0) else None

    # ── EPS (prefer diluted) ──────────────────────────────────────────────────
    eps = inc.eps_diluted or inc.eps
    if eps is None and inc.net_income and shares and shares > 0:
        eps = inc.net_income / shares

    # ── P/E ──────────────────────────────────────────────────────────────────
    pe: Optional[float] = None
    if price and eps and eps > 0:
        raw = price / eps
        if 0 < raw <= _PE_MAX:
            pe = round(raw, 2)

    # ── P/S ──────────────────────────────────────────────────────────────────
    ps: Optional[float] = None
    if mkt_cap and inc.revenue and inc.revenue > 0:
        raw = mkt_cap / inc.revenue
        if 0 < raw <= _PS_MAX:
            ps = round(raw, 2)

    # ── EV/EBITDA ────────────────────────────────────────────────────────────
    ev_ebitda: Optional[float] = None
    if mkt_cap and inc.ebitda and inc.ebitda > 0:
        debt = (bal.total_debt or 0.0) if bal else 0.0
        cash = (bal.cash_and_equivalents or 0.0) if bal else 0.0
        ev   = mkt_cap + debt - cash
        if ev > 0:
            raw = ev / inc.ebitda
            if 0 < raw <= _EV_EBITDA_MAX:
                ev_ebitda = round(raw, 2)

    # ── EPS CAGR ──────────────────────────────────────────────────────────────
    eps_series: list[float] = []
    for stmt in income_statements:
        e = stmt.eps_diluted or stmt.eps
        if e is None and shares and shares > 0 and stmt.net_income:
            e = stmt.net_income / shares
        if e is not None:
            eps_series.append(e)

    growth_pct: Optional[float] = None
    peg:        Optional[float] = None
    if len(eps_series) >= 2:
        n      = min(len(eps_series) - 1, 3)
        oldest = eps_series[n]
        latest = eps_series[0]
        if oldest > 0 and latest > 0:
            cagr       = (latest / oldest) ** (1.0 / n) - 1.0
            growth_pct = round(cagr * 100, 1)
            if pe and growth_pct and growth_pct > 0:
                raw_peg = pe / growth_pct
                if 0 < raw_peg <= _PEG_MAX:
                    peg = round(raw_peg, 2)

    return pe, ps, ev_ebitda, growth_pct, peg


# ── Median helper ──────────────────────────────────────────────────────────────

def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return (s[n // 2 - 1] + s[n // 2]) / 2 if n % 2 == 0 else s[n // 2]


# ── Peer quality scoring ──────────────────────────────────────────────────────

def _score_completeness(row: PeerRow) -> int:
    """Higher = more data available. Used to rank peers when truncating to _MAX_PEERS."""
    s = 0
    if row.pe is not None:                                          s += 3
    if row.ps is not None:                                          s += 2
    if row.ev_ebitda is not None:                                   s += 2
    if row.growth_pct is not None:                                  s += 2
    if row.peg is not None and _peg_is_meaningful_check(row.growth_pct, row.peg): s += 3
    return s


def _is_usable(row: PeerRow) -> bool:
    """A peer is usable if it has at least ONE valid metric."""
    return any(v is not None for v in [row.pe, row.ps, row.ev_ebitda, row.growth_pct])


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


def _generate_insights(rows: list[PeerRow]) -> list[str]:
    if len(rows) < 2:
        return []

    target   = rows[0]
    insights: list[str] = []

    # PEG
    peg_rows = [r for r in rows if _peg_is_meaningful(r)]
    if _peg_is_meaningful(target) and len(peg_rows) >= _MIN_VALID_FOR_CONCLUSION:
        by_peg    = sorted(peg_rows, key=lambda r: r.peg)
        cheapest  = by_peg[0]
        costliest = by_peg[-1]
        if cheapest.ticker == target.ticker:
            insights.append(
                f"{target.ticker} has the lowest PEG ({target.peg:.2f}x) — "
                "appears cheapest on a growth-adjusted basis."
            )
        elif costliest.ticker == target.ticker:
            insights.append(
                f"{target.ticker} has the highest PEG ({target.peg:.2f}x) — "
                "appears most expensive on a growth-adjusted basis."
            )
        else:
            insights.append(
                f"{target.ticker} PEG of {target.peg:.2f}x falls mid-range "
                f"(lowest: {cheapest.ticker} at {cheapest.peg:.2f}x, "
                f"highest: {costliest.ticker} at {costliest.peg:.2f}x)."
            )
    elif target.peg is not None and not _peg_is_meaningful(target):
        insights.append("PEG not meaningful — low or negative EPS growth.")

    # P/E vs peer median
    pe_rows  = [r for r in rows if r.pe is not None]
    peer_pes = [r.pe for r in pe_rows if not r.is_target]
    if target.pe is not None and len(pe_rows) >= _MIN_VALID_FOR_CONCLUSION and peer_pes:
        med_pe = _median(peer_pes)
        if target.pe < med_pe * 0.85:
            insights.append(
                f"P/E of {target.pe:.1f}x sits below the peer median of {med_pe:.1f}x — discount to group."
            )
        elif target.pe > med_pe * 1.15:
            insights.append(
                f"P/E of {target.pe:.1f}x sits above the peer median of {med_pe:.1f}x — premium to group."
            )
        else:
            insights.append(
                f"P/E of {target.pe:.1f}x broadly in line with the peer median of {med_pe:.1f}x."
            )

    # EPS growth vs peer median
    g_rows  = [r for r in rows if r.growth_pct is not None]
    peer_gs = [r.growth_pct for r in g_rows if not r.is_target]
    if target.growth_pct is not None and len(g_rows) >= _MIN_VALID_FOR_CONCLUSION and peer_gs:
        med_g = _median(peer_gs)
        if target.growth_pct > med_g * 1.20:
            insights.append(
                f"EPS growth of {target.growth_pct:.1f}% leads the peer median of {med_g:.1f}%."
            )
        elif target.growth_pct < med_g * 0.80:
            insights.append(
                f"EPS growth of {target.growth_pct:.1f}% lags the peer median of {med_g:.1f}%."
            )

    # Sparse notice
    valid_count = sum(1 for r in rows if _is_usable(r))
    if valid_count < _MIN_VALID_FOR_CONCLUSION and not insights:
        insights.append("Peer data too sparse for strong relative conclusions.")

    return insights[:3]


# ── Per-ticker fetch ────────────────────────────────────────────────────────────

def _fetch_peer_row(
    pticker: str, fmp: FMPProvider, limit: int, tier: int = 0
) -> "PeerRow | None":
    """
    Fetch profile + financials + balance sheets for one peer.
    Returns None on any error or if the peer has zero computable metrics.
    Never raises — all errors are caught and logged.
    """
    try:
        profile_result = fmp.get_profile(pticker)
        if profile_result.is_empty():
            print(f"  [PEER] {pticker}: no profile — skipping")
            return None

        price   = profile_result.current_price
        mkt_cap = profile_result.market_cap
        name    = getattr(profile_result.profile, "company_name", pticker) or pticker

        fin_result = fmp.get_financials(pticker, limit)
        income  = fin_result.income_statements
        balance = fin_result.balance_sheets

        pe, ps, ev_ebitda, growth, peg = _derive_metrics(price, mkt_cap, income, balance)
        row = PeerRow(
            ticker    = pticker,
            name      = name,
            pe        = pe,
            ps        = ps,
            ev_ebitda = ev_ebitda,
            growth_pct= growth,
            peg       = peg,
            tier      = tier,
        )
        score = _score_completeness(row)
        usable = _is_usable(row)
        print(
            f"  [PEER] {pticker} (tier={tier}): "
            f"PE={pe} PS={ps} EV/EBITDA={ev_ebitda} growth={growth}% PEG={peg} "
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
    target_ticker: str,
    sector:        str,
    industry:      str,
    target_mkt_cap: Optional[float],
    fmp: FMPProvider,
) -> list[tuple[str, int]]:
    """
    Return a list of (ticker, tier) candidates, deduplicated, target excluded.
    Tiers:
      0 = FMP /stock-peers (dynamic, most precise)
      1 = _TICKER_PEERS (curated)
      2 = FMP /stock-screener (sector+industry filter)
      3 = _PEER_UNIVERSE (sector)
      4 = _GLOBAL_FALLBACK (last resort)
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

    # Tier 2: screener — same sector + industry, ±67% market cap
    tier2: list[str] = []
    if sector or industry:
        min_cap = (target_mkt_cap * 0.33) if target_mkt_cap else None
        max_cap = (target_mkt_cap * 3.0)  if target_mkt_cap else None
        try:
            raw = fmp.get_screener(
                sector=sector, industry=industry,
                min_mkt_cap=min_cap, max_mkt_cap=max_cap, limit=15,
            )
            tier2 = raw
            print(
                f"  [PEER] {target_ticker}: tier-2 screener "
                f"({sector}/{industry}) → {len(tier2)} candidates"
            )
        except Exception as exc:
            print(f"  [PEER] {target_ticker}: tier-2 screener failed ({exc})")
    _add(tier2, 2)

    # Tier 3: sector universe
    tier3 = _PEER_UNIVERSE.get(sector, [])
    _add(tier3, 3)

    # Tier 4: global fallback
    _add(_GLOBAL_FALLBACK, 4)

    print(
        f"  [PEER] {target_ticker}: total pool={len(pool)} "
        f"(t0={len(tier0)} t1={len(tier1)} t2={len(tier2)} "
        f"t3={len(tier3)} t4={len(_GLOBAL_FALLBACK)}) "
        f"| sector={sector!r} industry={industry!r}"
    )
    return pool


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
        ticker    = target_ticker,
        name      = target_company_name or target_ticker,
        pe        = target_pe,
        ps        = target_ps,
        ev_ebitda = target_ev_ebitda,
        growth_pct= target_growth,
        peg       = target_peg,
        is_target = True,
    )
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

    pool = _build_candidate_pool(target_ticker, sector, industry, target_mkt_cap, fmp)
    if not pool:
        print(f"  [PEER] {target_ticker}: pool is empty — omitting")
        return _empty

    limit = min(Config.FINANCIAL_STATEMENT_LIMIT, 4)
    usable_rows:  list[PeerRow] = []
    size_deferred: list[tuple[str, int]] = []
    later_pool:    list[tuple[str, int]] = []
    candidates_tried = 0

    # ── Pass 1: size-filtered ─────────────────────────────────────────────────
    print(f"  [PEER] === PASS 1: size-filtered (±67% of {target_mkt_cap}) ===")
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
                if peer_mc and target_mkt_cap:
                    ratio = peer_mc / target_mkt_cap
                    if not (0.33 <= ratio <= 3.0):
                        print(f"  [PEER] {pticker}: size ratio={ratio:.2f} — deferred")
                        size_deferred.append((pticker, tier))
                        candidates_tried -= 1
                        continue
                # Reuse fetched profile
                price   = pf.current_price
                mkt_cap = pf.market_cap
                name    = getattr(pf.profile, "company_name", pticker) or pticker
                fin_result = fmp.get_financials(pticker, limit)
                income  = fin_result.income_statements
                balance = fin_result.balance_sheets
                pe, ps, ev_ebitda, growth, peg = _derive_metrics(price, mkt_cap, income, balance)
                row = PeerRow(
                    ticker=pticker, name=name,
                    pe=pe, ps=ps, ev_ebitda=ev_ebitda,
                    growth_pct=growth, peg=peg, tier=tier,
                )
                score = _score_completeness(row)
                usable = _is_usable(row)
                print(
                    f"  [PEER] {pticker} (tier={tier}): "
                    f"PE={pe} PS={ps} EV/EBITDA={ev_ebitda} growth={growth}% PEG={peg} "
                    f"score={score} usable={usable}"
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

    # Sort by completeness score descending, keep top _MAX_PEERS
    usable_rows.sort(key=_score_completeness, reverse=True)
    selected = usable_rows[:_MAX_PEERS]

    print(
        f"  [PEER] FINAL: selected {len(selected)} peers for {target_ticker}:"
    )
    for r in selected:
        print(
            f"    {r.ticker} (tier={r.tier}): "
            f"PE={r.pe} PS={r.ps} EV/EBITDA={r.ev_ebitda} "
            f"growth={r.growth_pct}% PEG={r.peg} score={_score_completeness(r)}"
        )

    pc = PeerComparison(target_ticker=target_ticker)
    pc.rows      = [target_row] + selected
    pc.has_peers = True
    pc.insights  = _generate_insights(pc.rows)

    print(f"  [PEER] Insights: {pc.insights}")
    return pc
