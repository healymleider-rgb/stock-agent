"""
Core metrics computation — single source of truth for all displayed metrics.

NormalizedMetrics is computed ONCE per evaluation and consumed by:
  - analysis.valuation.score_valuation  (scorecard valuation category)
  - web_api._extract_stock_info         (report header)
  - web_api._extract_peer_comparison    (peer target row)
  - analysis.valuation_range            (scenario analysis)

Design rules
────────────
Market cap
  FMP /quote `marketCap` is authoritative — it is computed live by the exchange
  feed as price × shares at the moment of the quote.  We only recompute it when
  the quote price was overridden by a fresher price-history close.  We NEVER
  substitute income-statement-derived shares for the authoritative market cap,
  because that derivation is fragile across stock splits, heavy SBC dilution,
  and basic vs diluted share count conventions.

Shares outstanding
  Used only for EV/EBITDA and P/S per-share methods inside valuation_range.
  Source: FMP /quote `sharesOutstanding` (actual count, not derived).

P/E
  Always computed fresh: price / TTM_EPS (sum of last 4 quarterly eps_diluted).
  Provider TTM P/E from /ratios-ttm is used as fallback and for audit.
  The computed value is preferred because TTM EPS is more current than FY EPS.

P/S and EV/EBITDA
  Computed from market_cap (authoritative) and revenue/EBITDA from income statement.
  Provider values from /ratios are used as fallback and for audit.

Validation
  Only reject values that are physically impossible:
    P/E  ≤ 0 or > 500
    P/S  ≤ 0 or > 100
    EV/EBITDA ≤ 0 or > 300
  Never null a metric because two sources disagree — log the divergence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from models.stock_data import StockData


# ── Sanity caps (physically impossible values) ─────────────────────────────────
_PE_CAP        = 500.0
_PS_CAP        = 100.0
_EV_EBITDA_CAP = 300.0
_PEG_CAP       = 100.0

# Log-only thresholds (divergence triggers a warning but no rejection)
_PRICE_DIV_LOG  = 0.05   # log when quote and price_history diverge > 5%
_MKTCAP_DIV_LOG = 0.20   # log when computed and API market cap diverge > 20%
_PE_DIV_LOG     = 0.25   # log when computed and provider PE diverge > 25%


@dataclass
class NormalizedMetrics:
    """
    Validated, sourced metrics for a single ticker.

    Every display-facing field has a corresponding _source attribute so any
    number in the report can be fully explained.  Raw intermediate values are
    preserved for debugging.
    """
    ticker: str

    # ── Price ─────────────────────────────────────────────────────────────────
    price:         Optional[float] = None
    price_source:  str = ""       # "price_history" | "quote" | "profile" | "unavailable"
    price_quote:   Optional[float] = None
    price_hist:    Optional[float] = None
    price_adjusted: bool = False   # True when price_history override was applied

    # ── Shares (for per-share valuation methods only) ─────────────────────────
    shares:        Optional[float] = None
    shares_source: str = ""       # "quote" | "income_diluted" | "income_basic" | "unavailable"

    # ── Market cap ────────────────────────────────────────────────────────────
    market_cap:          Optional[float] = None
    market_cap_source:   str = ""         # "api" | "recomputed" | "unavailable"
    market_cap_api:      Optional[float] = None   # FMP /quote marketCap
    market_cap_recomp:   Optional[float] = None   # price × shares (when price was adjusted)

    # ── EPS ───────────────────────────────────────────────────────────────────
    ttm_eps:           Optional[float] = None
    ttm_eps_source:    str = ""           # "4Q_eps_diluted" | "4Q_net_income" | "unavailable"
    annual_eps:        Optional[float] = None
    annual_eps_source: str = ""           # "eps_diluted" | "eps" | "derived"

    # ── P/E ───────────────────────────────────────────────────────────────────
    pe_ratio:         Optional[float] = None
    pe_source:        str = ""    # "computed_ttm" | "provider_ttm" | "computed_annual" | "unavailable"
    pe_computed_ttm:  Optional[float] = None   # price / ttm_eps
    pe_provider_ttm:  Optional[float] = None   # from /ratios or /ratios-ttm
    pe_computed_ann:  Optional[float] = None   # price / annual_eps

    # ── P/S ───────────────────────────────────────────────────────────────────
    ps_ratio:     Optional[float] = None
    ps_source:    str = ""        # "computed" | "provider" | "unavailable"
    ps_computed:  Optional[float] = None
    ps_provider:  Optional[float] = None

    # ── EV/EBITDA ─────────────────────────────────────────────────────────────
    ev_ebitda:          Optional[float] = None
    ev_ebitda_source:   str = ""  # "computed" | "provider" | "unavailable"
    ev_ebitda_computed: Optional[float] = None
    ev_ebitda_provider: Optional[float] = None

    # ── Financial ingredients (raw, for downstream use) ───────────────────────
    revenue:    Optional[float] = None
    ebitda:     Optional[float] = None
    net_income: Optional[float] = None

    # ── EPS growth & PEG (one definition everywhere) ─────────────────────────
    eps_growth_pct: Optional[float] = None   # annualized %, e.g. 12.5 means 12.5%
    peg:            Optional[float] = None

    # ── Provider-only ratios (no computed alternative) ────────────────────────
    roe:              Optional[float] = None
    roa:              Optional[float] = None
    gross_margin:     Optional[float] = None
    net_margin:       Optional[float] = None
    operating_margin: Optional[float] = None
    debt_to_equity:   Optional[float] = None
    current_ratio:    Optional[float] = None
    pb_ratio:         Optional[float] = None
    dividend_yield:   Optional[float] = None

    # ── Audit trail ───────────────────────────────────────────────────────────
    log: list[str] = field(default_factory=list)


def compute_core_metrics(stock_data: StockData) -> NormalizedMetrics:
    """
    Compute and validate all core metrics from fully-populated StockData.

    Call this after the orchestrator loop so quarterly_income and
    price_history are both populated.
    """
    m = NormalizedMetrics(ticker=stock_data.ticker)

    def _log(msg: str) -> None:
        m.log.append(msg)
        print(f"  [METRICS] {msg}")

    income  = stock_data.latest_income
    balance = stock_data.latest_balance
    ratios  = stock_data.latest_ratios

    _log(f"=== {stock_data.ticker} ===")
    _log(
        f"inputs: income={'ok' if income else 'None'}"
        f" balance={'ok' if balance else 'None'}"
        f" ratios={'ok' if ratios else 'None'}"
        f" quarterly={len(stock_data.quarterly_income)}"
        f" price_history={'ok' if stock_data.price_history else 'None'}"
    )
    if income:
        _log(
            f"income raw: revenue={income.revenue}"
            f" net_income={income.net_income}"
            f" ebitda={income.ebitda}"
            f" eps={income.eps}"
            f" eps_diluted={income.eps_diluted}"
        )
    if ratios:
        _log(
            f"ratios raw: pe={ratios.pe_ratio}"
            f" ps={ratios.ps_ratio}"
            f" ev_ebitda={ratios.ev_to_ebitda}"
        )

    # ── 1. PRICE ──────────────────────────────────────────────────────────────
    # Prefer price_history close[0] when it diverges materially from the quote,
    # because the quote price may be from a different session than the history
    # endpoint.  For the common case (intraday quote), quote == history close.
    m.price_quote = stock_data.current_price
    if stock_data.price_history and stock_data.price_history.closes:
        m.price_hist = stock_data.price_history.closes[0]

    _log(f"PRICE: quote={m.price_quote}  history={m.price_hist}")

    if m.price_quote and m.price_hist and m.price_hist > 0:
        div = abs(m.price_quote - m.price_hist) / m.price_hist
        if div > _PRICE_DIV_LOG:
            _log(
                f"PRICE: quote={m.price_quote} vs history={m.price_hist}"
                f" divergence={div:.1%} > {_PRICE_DIV_LOG:.0%}"
                f" → using price_history (fresher)"
            )
            m.price = m.price_hist
            m.price_source = "price_history"
            m.price_adjusted = True
        else:
            m.price = m.price_quote
            m.price_source = "quote"
            _log(f"PRICE: quote≈history Δ={div:.1%} → using quote={m.price_quote}")
    elif m.price_hist and not m.price_quote:
        m.price = m.price_hist
        m.price_source = "price_history"
        m.price_adjusted = True
        _log(f"PRICE: no quote → using price_history={m.price_hist}")
    elif m.price_quote:
        m.price = m.price_quote
        m.price_source = "quote"
        _log(f"PRICE: using quote={m.price_quote} (no history)")
    elif stock_data.profile and stock_data.profile.price:
        m.price = stock_data.profile.price
        m.price_source = "profile"
        m.price_adjusted = True
        _log(f"PRICE: using profile.price={m.price}")
    else:
        m.price_source = "unavailable"
        _log("PRICE: unavailable")

    # ── 2. SHARES OUTSTANDING (for per-share valuation methods only) ──────────
    # Primary: FMP /quote sharesOutstanding — the actual reported share count.
    # This is used only by valuation_range for EV/EBITDA and P/S per-share
    # scenario computations, NOT for market cap.
    if stock_data.shares_outstanding and stock_data.shares_outstanding > 0:
        m.shares = stock_data.shares_outstanding
        m.shares_source = "quote"
        _log(f"SHARES: /quote sharesOutstanding={m.shares:,.0f}")
    elif income and income.net_income and income.eps_diluted:
        # Fallback: derive from income statement
        ni, epsd = income.net_income, income.eps_diluted
        if epsd != 0:
            derived = abs(ni / epsd)
            # Sanity: derived shares should be in a plausible range (1M–100B)
            if 1e6 <= derived <= 1e11:
                m.shares = derived
                m.shares_source = "income_diluted"
                _log(f"SHARES: derived net_income/eps_diluted={m.shares:,.0f}")
            else:
                _log(
                    f"SHARES: derived={derived:.0f} outside plausible range"
                    f" — skipping income_diluted method"
                )
    if m.shares is None and income and income.net_income and income.eps:
        ni, eps = income.net_income, income.eps
        if eps != 0:
            derived = abs(ni / eps)
            if 1e6 <= derived <= 1e11:
                m.shares = derived
                m.shares_source = "income_basic"
                _log(f"SHARES: derived net_income/eps_basic={m.shares:,.0f}")
    if m.shares is None:
        m.shares_source = "unavailable"
        _log("SHARES: unavailable from all sources")

    # ── 3. MARKET CAP ─────────────────────────────────────────────────────────
    # FMP /quote marketCap is authoritative — computed live by the exchange feed.
    # Exception: if price was adjusted from price_history, recompute to stay
    # internally consistent (market_cap = adjusted_price × shares_outstanding).
    m.market_cap_api = stock_data.market_cap
    _log(f"MARKET_CAP: api={m.market_cap_api}")

    if m.price_adjusted and m.price and m.shares and m.shares > 0:
        m.market_cap_recomp = round(m.price * m.shares, 0)
        m.market_cap = m.market_cap_recomp
        m.market_cap_source = "recomputed"
        _log(
            f"MARKET_CAP: price was adjusted → recomputed={m.market_cap_recomp:,.0f}"
            f" (price={m.price} × shares={m.shares:,.0f})"
        )
        if m.market_cap_api:
            div = abs(m.market_cap_recomp - m.market_cap_api) / max(m.market_cap_api, 1)
            flag = "  *** LARGE DIVERGENCE ***" if div > _MKTCAP_DIV_LOG else ""
            _log(
                f"MARKET_CAP: api={m.market_cap_api:,.0f}"
                f" recomputed={m.market_cap_recomp:,.0f}"
                f" Δ={div:.1%}{flag}"
            )
    elif m.market_cap_api:
        m.market_cap = m.market_cap_api
        m.market_cap_source = "api"
        _log(f"MARKET_CAP: using api={m.market_cap_api:,.0f} (price not adjusted)")
    elif m.price and m.shares and m.shares > 0:
        m.market_cap_recomp = round(m.price * m.shares, 0)
        m.market_cap = m.market_cap_recomp
        m.market_cap_source = "recomputed"
        _log(f"MARKET_CAP: api unavailable → recomputed={m.market_cap_recomp:,.0f}")
    else:
        m.market_cap_source = "unavailable"
        _log("MARKET_CAP: unavailable")

    # ── 4. FINANCIAL INGREDIENTS ──────────────────────────────────────────────
    if income:
        m.revenue    = income.revenue
        m.ebitda     = income.ebitda
        m.net_income = income.net_income
        _log(
            f"FINANCIALS: revenue={m.revenue}"
            f" ebitda={m.ebitda}"
            f" net_income={m.net_income}"
        )

    # ── 5. TTM EPS — sum of last 4 quarterly EPS ──────────────────────────────
    quarters = stock_data.quarterly_income[:4] if stock_data.quarterly_income else []
    _log(f"TTM_EPS: {len(quarters)} quarterly statements available")

    if len(quarters) >= 4:
        eps_vals: list[float] = []
        ni_vals:  list[float] = []
        for q in quarters:
            e = q.eps_diluted or q.eps
            if e is not None:
                eps_vals.append(e)
            if q.net_income is not None:
                ni_vals.append(q.net_income)

        if len(eps_vals) >= 4:
            m.ttm_eps = sum(eps_vals[:4])
            m.ttm_eps_source = "4Q_eps_diluted"
            _log(f"TTM_EPS: sum({[round(e,4) for e in eps_vals[:4]]}) = {m.ttm_eps:.4f}")
        elif len(ni_vals) >= 4 and m.shares and m.shares > 0:
            ttm_ni = sum(ni_vals[:4])
            m.ttm_eps = ttm_ni / m.shares
            m.ttm_eps_source = "4Q_net_income"
            _log(
                f"TTM_EPS: sum(4Q net_income={ttm_ni:,.0f})"
                f" / shares={m.shares:,.0f}"
                f" = {m.ttm_eps:.4f}"
            )
        else:
            _log(
                f"TTM_EPS: {len(eps_vals)} EPS vals + {len(ni_vals)} NI vals"
                f" — insufficient for 4Q sum"
            )
    else:
        _log(f"TTM_EPS: {len(quarters)} quarters < 4 — TTM EPS unavailable")

    # ── 6. ANNUAL EPS ─────────────────────────────────────────────────────────
    if income:
        if income.eps_diluted is not None:
            m.annual_eps = income.eps_diluted
            m.annual_eps_source = "eps_diluted"
        elif income.eps is not None:
            m.annual_eps = income.eps
            m.annual_eps_source = "eps"
        elif income.net_income and m.shares and m.shares > 0:
            m.annual_eps = income.net_income / m.shares
            m.annual_eps_source = "derived"
        if m.annual_eps is not None:
            _log(f"ANNUAL_EPS: {m.annual_eps:.4f} (source: {m.annual_eps_source})")

    # Provider ratios for cross-validation and passthrough
    if ratios:
        m.pe_provider_ttm    = ratios.pe_ratio
        m.ps_provider        = ratios.ps_ratio
        m.ev_ebitda_provider = ratios.ev_to_ebitda
        m.roe                = ratios.roe
        m.roa                = ratios.roa
        m.gross_margin       = ratios.gross_margin
        m.net_margin         = ratios.net_margin
        m.operating_margin   = ratios.operating_margin
        m.debt_to_equity     = ratios.debt_to_equity
        m.current_ratio      = ratios.current_ratio
        m.pb_ratio           = ratios.pb_ratio
        m.dividend_yield     = ratios.dividend_yield

    # Derive margins from statements when ratios are absent
    if income and income.revenue and income.revenue > 0:
        rev = income.revenue
        if m.gross_margin is None and income.gross_profit is not None:
            m.gross_margin = income.gross_profit / rev
        if m.net_margin is None and income.net_income is not None:
            m.net_margin = income.net_income / rev
        if m.operating_margin is None and income.operating_income is not None:
            m.operating_margin = income.operating_income / rev

    # ── 7. P/E ────────────────────────────────────────────────────────────────
    # Computed TTM P/E: price / sum(last 4 quarterly EPS)
    if m.price and m.ttm_eps is not None and m.ttm_eps != 0:
        raw = m.price / m.ttm_eps
        if 0 < raw <= _PE_CAP:
            m.pe_computed_ttm = round(raw, 2)
            _log(
                f"PE: price={m.price} / ttm_eps={m.ttm_eps:.4f}"
                f" → computed_ttm={m.pe_computed_ttm:.2f}"
            )
        else:
            _log(
                f"PE: computed_ttm={raw:.2f} outside valid range"
                f" (0, {_PE_CAP}] — excluded"
            )

    # Computed annual P/E fallback
    if m.price and m.annual_eps and m.annual_eps != 0:
        raw = m.price / m.annual_eps
        if 0 < raw <= _PE_CAP:
            m.pe_computed_ann = round(raw, 2)
            _log(f"PE: price/annual_eps → computed_ann={m.pe_computed_ann:.2f}")
        else:
            _log(f"PE: computed_ann={raw:.2f} outside valid range — excluded")

    if m.pe_provider_ttm:
        _log(f"PE: provider_ttm={m.pe_provider_ttm:.2f}")

    # Selection: provider_ttm first (authoritative FMP TTM), then computed_ttm,
    # then annual.  Divergence triggers a warning + confidence note but NEVER
    # nulls a valid value.  Negative computed TTM EPS means computed_ttm is
    # excluded from the valid set, but provider/annual paths remain open.
    if m.pe_provider_ttm and 0 < m.pe_provider_ttm <= _PE_CAP:
        m.pe_ratio  = m.pe_provider_ttm
        m.pe_source = "provider_ttm"
        _log(f"PE: using provider_ttm={m.pe_provider_ttm:.2f}")
        if m.pe_computed_ttm is not None:
            div = abs(m.pe_provider_ttm - m.pe_computed_ttm) / m.pe_computed_ttm
            if div > _PE_DIV_LOG:
                _log(
                    f"PE: provider_ttm={m.pe_provider_ttm:.2f}"
                    f" vs computed_ttm={m.pe_computed_ttm:.2f}"
                    f" Δ={div:.1%} — WARN: sources diverge"
                    f" (keeping provider, confidence reduced)"
                )
    elif m.pe_computed_ttm is not None:
        m.pe_ratio  = m.pe_computed_ttm
        m.pe_source = "computed_ttm"
        _log(f"PE: no valid provider_ttm → using computed_ttm={m.pe_computed_ttm:.2f}")
    elif m.pe_computed_ann is not None:
        m.pe_ratio  = m.pe_computed_ann
        m.pe_source = "computed_annual"
        _log(f"PE: no TTM available → using computed_annual={m.pe_computed_ann:.2f}")
    else:
        m.pe_source = "unavailable"
        _log("PE: unavailable from all sources")

    # ── 8. P/S ────────────────────────────────────────────────────────────────
    if m.market_cap and m.revenue and m.revenue > 0:
        raw = m.market_cap / m.revenue
        if 0 < raw <= _PS_CAP:
            m.ps_computed = round(raw, 2)
            _log(
                f"PS: mktcap={m.market_cap:,.0f} / revenue={m.revenue:,.0f}"
                f" → computed={m.ps_computed:.2f}"
            )
        else:
            _log(f"PS: computed={raw:.2f} outside valid range — excluded")

    if m.ps_provider:
        _log(f"PS: provider={m.ps_provider:.2f}")

    if m.ps_computed is not None:
        m.ps_ratio  = m.ps_computed
        m.ps_source = "computed"
        if m.ps_provider and m.ps_provider > 0:
            div = abs(m.ps_provider - m.ps_computed) / m.ps_computed
            if div > 0.15:
                _log(
                    f"PS: computed={m.ps_computed:.2f}"
                    f" vs provider={m.ps_provider:.2f}"
                    f" Δ={div:.1%} (using computed)"
                )
    elif m.ps_provider and 0 < m.ps_provider <= _PS_CAP:
        m.ps_ratio  = m.ps_provider
        m.ps_source = "provider"
        _log(f"PS: no computed → using provider={m.ps_provider:.2f}")
    else:
        m.ps_source = "unavailable"
        _log("PS: unavailable")

    # ── 9. EV/EBITDA ──────────────────────────────────────────────────────────
    if m.market_cap and m.ebitda and m.ebitda > 0:
        debt = (balance.total_debt or 0.0) if balance else 0.0
        cash = (balance.cash_and_equivalents or 0.0) if balance else 0.0
        ev   = m.market_cap + debt - cash
        if ev > 0:
            raw = ev / m.ebitda
            if 0 < raw <= _EV_EBITDA_CAP:
                m.ev_ebitda_computed = round(raw, 2)
                _log(
                    f"EV_EBITDA: EV={ev:,.0f}"
                    f" (mktcap={m.market_cap:,.0f}+debt={debt:,.0f}-cash={cash:,.0f})"
                    f" / EBITDA={m.ebitda:,.0f}"
                    f" → computed={m.ev_ebitda_computed:.2f}"
                )
            else:
                _log(f"EV_EBITDA: computed={raw:.2f} outside valid range — excluded")
        else:
            _log(f"EV_EBITDA: EV={ev:,.0f} ≤ 0 → computed unavailable")

    if m.ev_ebitda_provider:
        _log(f"EV_EBITDA: provider={m.ev_ebitda_provider:.2f}")

    if m.ev_ebitda_computed is not None:
        m.ev_ebitda        = m.ev_ebitda_computed
        m.ev_ebitda_source = "computed"
        if m.ev_ebitda_provider and m.ev_ebitda_provider > 0:
            div = abs(m.ev_ebitda_provider - m.ev_ebitda_computed) / m.ev_ebitda_computed
            if div > 0.15:
                _log(
                    f"EV_EBITDA: computed={m.ev_ebitda_computed:.2f}"
                    f" vs provider={m.ev_ebitda_provider:.2f}"
                    f" Δ={div:.1%} (using computed)"
                )
    elif m.ev_ebitda_provider and 0 < m.ev_ebitda_provider <= _EV_EBITDA_CAP:
        m.ev_ebitda        = m.ev_ebitda_provider
        m.ev_ebitda_source = "provider"
        _log(f"EV_EBITDA: no computed → using provider={m.ev_ebitda_provider:.2f}")
    else:
        m.ev_ebitda_source = "unavailable"
        _log("EV_EBITDA: unavailable")

    # ── 10. EPS GROWTH — one methodology everywhere ───────────────────────────
    # Annualized CAGR over min(3, n-1) annual periods.
    # Uses TTM EPS as the "current" endpoint when available.
    stmts = stock_data.income_statements
    if len(stmts) >= 2:
        eps_series: list[float] = []
        for stmt in stmts:
            e = stmt.eps_diluted or stmt.eps
            if e is None and m.shares and m.shares > 0 and stmt.net_income:
                e = stmt.net_income / m.shares
            if e is not None:
                eps_series.append(e)

        if len(eps_series) >= 2:
            n = min(len(eps_series) - 1, 3)
            eps_latest = m.ttm_eps if (m.ttm_eps is not None and m.ttm_eps != 0) else eps_series[0]
            eps_label  = "TTM" if (m.ttm_eps is not None) else "annual[0]"
            eps_oldest = eps_series[n]

            if eps_oldest != 0 and eps_latest != 0 and (
                (eps_oldest > 0 and eps_latest > 0) or
                (eps_oldest < 0 and eps_latest < 0)
            ):
                cagr = (abs(eps_latest) / abs(eps_oldest)) ** (1.0 / n) - 1.0
                if eps_latest < 0:
                    cagr = -cagr
                m.eps_growth_pct = round(cagr * 100, 1)
                _log(
                    f"EPS_GROWTH: {eps_latest:.4f}({eps_label})"
                    f" vs {eps_oldest:.4f}(annual[{n}y])"
                    f" → CAGR={m.eps_growth_pct:.1f}%"
                )
            else:
                _log(
                    f"EPS_GROWTH: mixed signs or zero"
                    f" (latest={eps_latest}, oldest={eps_oldest}) → skipped"
                )
        else:
            _log(f"EPS_GROWTH: {len(eps_series)} EPS data points — need ≥2")
    else:
        _log(f"EPS_GROWTH: {len(stmts)} income statements — need ≥2")

    # ── 11. PEG ───────────────────────────────────────────────────────────────
    if m.pe_ratio and m.eps_growth_pct and m.eps_growth_pct > 0:
        raw_peg = m.pe_ratio / m.eps_growth_pct
        if 0 < raw_peg <= _PEG_CAP:
            m.peg = round(raw_peg, 2)
            _log(f"PEG: {m.pe_ratio:.2f} / {m.eps_growth_pct:.1f}% = {m.peg:.2f}")
        else:
            _log(f"PEG: raw={raw_peg:.2f} outside valid range → skipped")
    elif m.pe_ratio and m.eps_growth_pct is not None and m.eps_growth_pct <= 0:
        _log(f"PEG: EPS growth ≤0 ({m.eps_growth_pct:.1f}%) → PEG not meaningful")
    else:
        _log(f"PEG: unavailable (pe={m.pe_ratio}, growth={m.eps_growth_pct})")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    sh_str = f"{m.shares:,.0f}" if m.shares else "N/A"
    mc_str = f"{m.market_cap:,.0f}" if m.market_cap else "N/A"
    _log(
        f"FINAL:"
        f" price={m.price}({m.price_source})"
        f" shares={sh_str}({m.shares_source})"
        f" mktcap={mc_str}({m.market_cap_source})"
        f" | pe={m.pe_ratio}({m.pe_source})"
        f" ps={m.ps_ratio}({m.ps_source})"
        f" ev_ebitda={m.ev_ebitda}({m.ev_ebitda_source})"
        f" | eps_growth={m.eps_growth_pct}%"
        f" peg={m.peg}"
    )
    return m


# ── Confidence ─────────────────────────────────────────────────────────────────
#
# Weights reflect how much each resolved metric contributes to a complete,
# trustworthy analysis.  Key valuation inputs (PE, PS, EV/EBITDA) together
# carry 35%; market cap 10%; growth 20%; profitability 20%; health 15%.
# Weights intentionally sum to 1.0 so confidence is a true percentage.

_CONFIDENCE_WEIGHTS: dict[str, float] = {
    # Valuation
    "pe_ratio":       0.15,
    "ps_ratio":       0.10,
    "ev_ebitda":      0.10,
    # Market cap
    "market_cap":     0.10,
    # Growth
    "eps_growth_pct": 0.12,
    "peg":            0.08,
    # Profitability
    "gross_margin":   0.07,
    "net_margin":     0.07,
    "roe":            0.06,
    # Financial health
    "debt_to_equity": 0.08,
    "current_ratio":  0.07,
}


def compute_signal_confidence(
    categories: dict,
) -> tuple[float, str]:
    """
    Derive confidence from SIGNAL AGREEMENT across category scores.

    High confidence (>0.85) requires directional alignment — most major
    factors pointing the same way.  Confidence is penalised when:
      - fundamental categories (valuation, growth, profitability, health)
        disagree directionally with momentum
      - score dispersion across categories is wide (>30 pts spread)

    ``categories`` should be a dict of name → CategoryScore-like object
    with ``.score`` and ``.data_quality`` attributes.

    Returns (confidence_0_to_1, explanation_string).
    """
    # Filter to non-missing categories
    valid: dict[str, float] = {}
    for name, cat in categories.items():
        if cat is None:
            continue
        dq = getattr(cat, "data_quality", "good")
        if dq != "missing":
            valid[name] = float(cat.score)

    if not valid:
        return 0.30, "Insufficient category data for signal agreement assessment."

    def _dir(s: float) -> str:
        return "bull" if s >= 55 else "bear" if s < 45 else "neutral"

    directions = {name: _dir(s) for name, s in valid.items()}
    bull_count = sum(1 for d in directions.values() if d == "bull")
    bear_count = sum(1 for d in directions.values() if d == "bear")
    n = len(valid)

    # Agreement ratio: fraction of categories pointing the dominant direction
    max_consensus = max(bull_count, bear_count)
    agreement_ratio = max_consensus / n if n > 0 else 0.5

    # Dispersion penalty — large spread means less conviction even if majority agrees
    all_scores = list(valid.values())
    dispersion = max(all_scores) - min(all_scores)
    # Penalty scales from 0 at 30 pts spread to 0.20 at 80 pts spread
    dispersion_penalty = max(0.0, min(0.20, (dispersion - 30.0) / 250.0))

    # Fundamental vs momentum conflict penalty
    fundamental_names = {"valuation", "growth", "profitability", "financial_health"}
    fund_scores = [s for name, s in valid.items() if name in fundamental_names]
    mom_score = valid.get("momentum")

    conflict_penalty = 0.0
    conflict_note = ""
    if fund_scores and mom_score is not None:
        fund_avg = sum(fund_scores) / len(fund_scores)
        fund_dir = _dir(fund_avg)
        mom_dir = _dir(mom_score)
        if fund_dir != mom_dir and fund_dir != "neutral" and mom_dir != "neutral":
            conflict_penalty = 0.10
            conflict_note = (
                f"momentum ({mom_score:.0f}/100) diverges from"
                f" fundamentals ({fund_avg:.0f}/100 avg)"
            )

    # Base confidence from agreement: range [0.40, 0.90]
    base = 0.40 + agreement_ratio * 0.50
    conf = max(0.0, min(1.0, base - dispersion_penalty - conflict_penalty))
    conf = round(conf, 3)

    # Build explanation
    bull_names = [n for n, d in directions.items() if d == "bull"]
    bear_names = [n for n, d in directions.items() if d == "bear"]

    if conflict_note:
        explanation = f"Mixed signals reduce conviction — {conflict_note}."
    elif agreement_ratio >= 0.80 and dispersion < 35:
        if bull_count >= bear_count:
            named = ", ".join(n.replace("_", " ") for n in bull_names[:3])
            explanation = f"High signal agreement across {named} — conviction supported."
        else:
            named = ", ".join(n.replace("_", " ") for n in bear_names[:3])
            explanation = f"High agreement on weakness across {named}."
    elif dispersion > 40:
        explanation = (
            f"Wide score dispersion ({dispersion:.0f} pts) across factors"
            " limits conviction despite directional majority."
        )
    elif agreement_ratio >= 0.60:
        explanation = "Moderate signal agreement — most factors aligned but some divergence."
    else:
        explanation = "Signals are mixed — no clear directional consensus across factors."

    print(
        f"  [SIGNAL_CONF] agreement={agreement_ratio:.2f}"
        f" dispersion={dispersion:.1f} conflict_penalty={conflict_penalty:.2f}"
        f" → conf={conf:.3f} | {explanation}"
    )
    return conf, explanation


def compute_confidence(metrics: NormalizedMetrics) -> float:
    """
    Derive a confidence score (0.0–1.0) directly from NormalizedMetrics quality.

    Rules
    -----
    Base score  : weighted fraction of key metrics that resolved to a value.
    TTM bonus   : +5 pp when PE came from a TTM source (fresher than annual).
    Divergence  : −3 pp per provider-vs-computed divergence warning logged.

    The result is clamped to [0.0, 1.0].  High confidence (>0.90) requires
    at least most key metrics present and no major contradictions.
    """
    total_w = sum(_CONFIDENCE_WEIGHTS.values())
    achieved = sum(
        w for field, w in _CONFIDENCE_WEIGHTS.items()
        if getattr(metrics, field) is not None
    )
    base = achieved / total_w

    # Bonus for having TTM-quality PE (more current than annual)
    if metrics.pe_source in ("provider_ttm", "computed_ttm"):
        base = min(base + 0.05, 1.0)

    # Penalty per logged divergence warning
    divergences = sum(1 for e in metrics.log if "WARN: sources diverge" in e)
    base = max(0.0, base - divergences * 0.03)

    result = round(base, 3)
    print(
        f"  [CONFIDENCE] {metrics.ticker}:"
        f" achieved={achieved:.3f}/{total_w:.3f}"
        f" base={base:.3f}"
        f" divergences={divergences}"
        f" → confidence={result:.3f}"
    )
    return result
