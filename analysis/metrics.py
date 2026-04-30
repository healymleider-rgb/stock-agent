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
    shares:             Optional[float] = None
    shares_source:      str = ""       # "quote" | "FMP/shares-float (SEC EDGAR)" | "income_diluted" | ...
    shares_filing_date: Optional[str] = None   # SEC filing date when shares_source is /shares-float
    shares_filing_url:  Optional[str] = None   # Direct SEC EDGAR document URL

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

    # ── FCF ───────────────────────────────────────────────────────────────────
    ttm_fcf:        Optional[float] = None
    ttm_fcf_source: str = ""        # "cash_flow_statement" | "unavailable"

    # ── EPS growth & PEG (one definition everywhere) ─────────────────────────
    eps_growth_pct: Optional[float] = None   # annualized %, e.g. 12.5 means 12.5%
    peg:            Optional[float] = None
    peg_method:     str = ""    # "eps_cagr" | "revenue_cagr" | "not_meaningful"
    peg_note:       str = ""    # set when method != eps_cagr

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

    def apply_integrity_corrections(self, validation: object) -> None:
        """
        Apply adjusted_metrics from a ValidationResult back into this object.

        Called by FundamentalAnalysisAgent after run_data_integrity_check()
        so that all downstream scorers (valuation, growth, etc.) automatically
        use the corrected values without any code change in those scorers.

        Corrects:
          price      → recompute pe_ratio and ps_ratio using corrected price
          pe_ratio   → override with integrity-checked computed value
          ev_ebitda  → override with integrity-checked computed value
        """
        adj = getattr(validation, "adjusted_metrics", {}) or {}
        if not adj:
            return

        def _log(msg: str) -> None:
            self.log.append(f"[CORRECTION] {msg}")
            print(f"  [METRICS:CORRECTION] {msg}")

        # Price override (from market cap identity constraint)
        if "price" in adj:
            old_price   = self.price
            self.price  = adj["price"]
            self.price_source = "integrity_corrected"
            self.price_adjusted = True
            _log(f"price {old_price} → {self.price:.4f} (market cap identity)")

            # Recompute ratio numerators that use price
            if self.ttm_eps and self.ttm_eps != 0:
                old_pe         = self.pe_ratio
                new_pe         = self.price / self.ttm_eps
                import math as _math
                if 0 < new_pe < 500:
                    self.pe_computed_ttm = new_pe
                    self.pe_ratio        = new_pe
                    self.pe_source       = "computed_ttm_corrected"
                    _log(f"pe_ratio {old_pe} → {new_pe:.2f} (price corrected)")

            if self.market_cap and self.market_cap > 0 and self.revenue and self.revenue > 0:
                old_ps         = self.ps_ratio
                new_ps         = self.market_cap / self.revenue
                if 0 < new_ps < 100:
                    self.ps_computed = new_ps
                    self.ps_ratio    = new_ps
                    self.ps_source   = "computed_corrected"
                    _log(f"ps_ratio {old_ps} → {new_ps:.2f} (price corrected)")

        # PE override (from integrity check cross-validation)
        if "pe_ratio" in adj and "price" not in adj:   # price takes precedence
            old_pe          = self.pe_ratio
            self.pe_ratio   = adj["pe_ratio"]
            self.pe_source  = "computed_ttm"
            _log(f"pe_ratio {old_pe} → {self.pe_ratio:.2f} (integrity computed)")

        # EV/EBITDA override
        if "ev_ebitda" in adj:
            old_ev              = self.ev_ebitda
            self.ev_ebitda      = adj["ev_ebitda"]
            self.ev_ebitda_source = "computed"
            _log(f"ev_ebitda {old_ev} → {self.ev_ebitda:.2f} (integrity computed)")


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
        # Use the source label that came from the provider (may be /shares-float or /quote).
        # Fall back to "quote" for backwards compatibility with data fetched before the
        # provenance fields were added.
        m.shares_source      = stock_data.shares_source or "quote"
        m.shares_filing_date = stock_data.shares_filing_date
        m.shares_filing_url  = stock_data.shares_filing_url
        _log(f"SHARES: {m.shares_source}={m.shares:,.0f}")
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
    # Always compute market_cap = price × shares_outstanding.
    # The FMP /quote marketCap field is stored for reference and divergence
    # logging only — it can lag the current price by minutes, use a different
    # share count than the latest 10-Q, or reflect a pre-market snapshot.
    # Computing from price × shares guarantees internal consistency: every
    # ratio derived from market_cap (P/S, EV) uses the same price the rest
    # of the report uses.
    m.market_cap_api = stock_data.market_cap
    _log(f"MARKET_CAP: api={m.market_cap_api}")

    if m.price and m.shares and m.shares > 0:
        m.market_cap_recomp = round(m.price * m.shares, 0)
        m.market_cap = m.market_cap_recomp
        m.market_cap_source = "recomputed"
        _log(
            f"MARKET_CAP: recomputed={m.market_cap_recomp:,.0f}"
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
        _log(f"MARKET_CAP: shares unavailable → using api={m.market_cap_api:,.0f}")
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

    # ── 5.5. TTM FCF — single authoritative source for FCF driver chain ─────────
    # Reads the most recent annual cash flow statement (FMP `freeCashFlow` field).
    # Stored here so all downstream users (valuation_range, exit multiple) read
    # from this attribute rather than re-reading stock_data.cash_flows directly.
    _cfs = stock_data.cash_flows
    if _cfs and _cfs[0].free_cash_flow is not None:
        m.ttm_fcf = _cfs[0].free_cash_flow
        m.ttm_fcf_source = "cash_flow_statement"
        _log(f"TTM_FCF: {m.ttm_fcf:,.0f} (source: cash_flow_statement)")
    else:
        m.ttm_fcf_source = "unavailable"
        _log("TTM_FCF: unavailable — cash_flows missing or freeCashFlow is None")

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

    # Selection: computed_ttm first (price / sum of last 4 quarterly EPS — primary source),
    # then provider_ttm (FMP field — useful cross-check but can lag or use a different EPS base),
    # then annual.  The computed path is authoritative: it uses the actual quarterly filings
    # we have in hand, so it cannot drift from reality the way a cached API field can.
    if m.pe_computed_ttm is not None:
        m.pe_ratio  = m.pe_computed_ttm
        m.pe_source = "computed_ttm"
        _log(f"PE: using computed_ttm={m.pe_computed_ttm:.2f} (price / 4Q eps sum)")
        if m.pe_provider_ttm and 0 < m.pe_provider_ttm <= _PE_CAP:
            div = abs(m.pe_computed_ttm - m.pe_provider_ttm) / max(m.pe_provider_ttm, 0.01)
            if div > _PE_DIV_LOG:
                _log(
                    f"PE: computed_ttm={m.pe_computed_ttm:.2f}"
                    f" vs provider_ttm={m.pe_provider_ttm:.2f}"
                    f" Δ={div:.1%} — provider diverges (stale EPS base?); using computed"
                )
    elif m.pe_provider_ttm and 0 < m.pe_provider_ttm <= _PE_CAP:
        m.pe_ratio  = m.pe_provider_ttm
        m.pe_source = "provider_ttm"
        _log(f"PE: no computed_ttm → using provider_ttm={m.pe_provider_ttm:.2f}")
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
    _eps_cv          = 0.0
    _has_negative_eps = False
    _eps_series_full: list[float] = []   # kept for volatility check in step 11

    if len(stmts) >= 2:
        eps_series: list[float] = []
        for stmt in stmts:
            e = stmt.eps_diluted or stmt.eps
            if e is None and m.shares and m.shares > 0 and stmt.net_income:
                e = stmt.net_income / m.shares
            if e is not None:
                eps_series.append(e)
        _eps_series_full = eps_series

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

    # EPS volatility stats (used by step 11 to decide PEG reliability)
    _all_eps = (
        ([m.ttm_eps] if m.ttm_eps is not None else []) + _eps_series_full
    )[:6]
    _has_negative_eps = any(e < 0 for e in _all_eps)
    if len(_all_eps) >= 3:
        import statistics as _stats
        _eps_mean = _stats.mean(_all_eps)
        _eps_cv   = (_stats.stdev(_all_eps) / abs(_eps_mean)) if _eps_mean != 0 else float("inf")
    _log(
        f"EPS_VOLATILITY: cv={_eps_cv:.2f}"
        f" has_negative={_has_negative_eps}"
        f" n={len(_all_eps)}"
    )

    # Revenue CAGR (for PEG fallback when EPS is unreliable)
    _rev_cagr_pct: Optional[float] = None
    _rev_series = [stmt.revenue for stmt in stmts[:5] if getattr(stmt, "revenue", None) and stmt.revenue > 0]
    if len(_rev_series) >= 2:
        _n_rev   = min(len(_rev_series) - 1, 3)
        _r_cagr  = (_rev_series[0] / _rev_series[_n_rev]) ** (1.0 / _n_rev) - 1.0
        _rev_cagr_pct = round(_r_cagr * 100, 1)
        _log(f"REV_CAGR: {_rev_cagr_pct:.1f}% ({_n_rev}y, ${_rev_series[_n_rev]:,.0f} → ${_rev_series[0]:,.0f})")

    # ── 11. PEG ───────────────────────────────────────────────────────────────
    # Volatility guard: use revenue CAGR when EPS history is unreliable.
    # Conditions: CV > 0.5, any negative EPS, EPS CAGR > 50%, or
    # EPS/revenue CAGR divergence > 20 pp.
    _eps_cagr_ok = True
    _unreliable_reasons: list[str] = []

    if _eps_cv > 0.5:
        _eps_cagr_ok = False
        _unreliable_reasons.append(f"CV={_eps_cv:.2f}")
    if _has_negative_eps:
        _eps_cagr_ok = False
        _unreliable_reasons.append("negative EPS in history")
    if m.eps_growth_pct is not None and m.eps_growth_pct > 50:
        _eps_cagr_ok = False
        _unreliable_reasons.append(f"EPS CAGR {m.eps_growth_pct:.0f}% likely volatile-base")
    if (
        m.eps_growth_pct is not None and _rev_cagr_pct is not None
        and abs(m.eps_growth_pct - _rev_cagr_pct) > 20
    ):
        _eps_cagr_ok = False
        _unreliable_reasons.append(
            f"EPS/rev CAGR diverge {abs(m.eps_growth_pct - _rev_cagr_pct):.0f}pp"
        )

    if m.pe_ratio and m.eps_growth_pct and m.eps_growth_pct > 0:
        _eps_raw_peg = m.pe_ratio / m.eps_growth_pct

        if _eps_cagr_ok:
            # Standard EPS-based PEG
            if 0 < _eps_raw_peg <= _PEG_CAP:
                m.peg        = round(_eps_raw_peg, 2)
                m.peg_method = "eps_cagr"
                _log(f"PEG: {m.pe_ratio:.2f} / {m.eps_growth_pct:.1f}% = {m.peg:.2f} (eps_cagr)")
            else:
                _log(f"PEG: raw={_eps_raw_peg:.2f} outside valid range → skipped")
        else:
            # EPS CAGR unreliable — fall back to revenue CAGR
            _reason_str = "; ".join(_unreliable_reasons)
            if _rev_cagr_pct is not None and _rev_cagr_pct >= 5.0:
                _rev_raw_peg = m.pe_ratio / _rev_cagr_pct
                if 0 < _rev_raw_peg <= _PEG_CAP:
                    m.peg        = round(_rev_raw_peg, 2)
                    m.peg_method = "revenue_cagr"
                    m.peg_note   = (
                        f"EPS-based PEG ({round(_eps_raw_peg, 2):.2f}) unreliable "
                        f"({_reason_str}). Using revenue CAGR ({_rev_cagr_pct:.0f}%) "
                        f"instead. EPS-based PEG shown for reference only."
                    )
                    _log(
                        f"PEG: {m.pe_ratio:.2f} / {_rev_cagr_pct:.1f}% = {m.peg:.2f}"
                        f" (revenue_cagr; eps_peg={round(_eps_raw_peg,2):.2f} suppressed)"
                    )
                else:
                    m.peg        = None
                    m.peg_method = "not_meaningful"
                    m.peg_note   = f"PEG not meaningful ({_reason_str}; revenue CAGR outside range)"
                    _log(f"PEG: not_meaningful ({_reason_str})")
            else:
                m.peg        = None
                m.peg_method = "not_meaningful"
                rev_note = (
                    f"rev CAGR {_rev_cagr_pct:.0f}% too low" if _rev_cagr_pct is not None
                    else "revenue CAGR unavailable"
                )
                m.peg_note   = f"PEG not meaningful for this ticker ({_reason_str}; {rev_note})."
                _log(f"PEG: not_meaningful ({_reason_str}; {rev_note})")
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
    macro: dict | None = None,
) -> tuple[float, str]:
    """
    Confidence Model V2 — structured, explainable signal confidence.

    Combines three independent components:
      1. Agreement score   — fraction of valid categories pointing the same direction.
      2. Conflict penalty  — named penalties for specific signal conflicts:
                             * strong fundamentals vs weak momentum
                             * strong fundamentals vs expensive valuation
                             * strong fundamentals vs adverse macro regime
                             * wide score dispersion (> 45 pts)
      3. Completeness adj  — small bonus/penalty for data coverage vs baseline.

    Final confidence = base(agreement) − conflict_total + completeness_adj,
    clamped to [0.25, 0.92].

    ``categories``  dict of name → CategoryScore-like object with .score and
                    .data_quality attributes (missing entries are excluded).
    ``macro``       optional macro_findings dict from MacroLEIAgent; used only
                    to detect regime / recession-risk conflict.  Safe to omit.

    Returns (confidence_0_to_1, explanation_string).
    """
    # ── Step 0: filter to non-missing categories ──────────────────────────────
    EXPECTED_N = 6  # valuation, growth, profitability, financial_health, momentum, risk
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

    def _join(names: list[str]) -> str:
        if not names:
            return ""
        if len(names) == 1:
            return names[0]
        return ", ".join(names[:-1]) + f" and {names[-1]}"

    # ── Step 1: Agreement score ───────────────────────────────────────────────
    directions   = {name: _dir(s) for name, s in valid.items()}
    bull_count   = sum(1 for d in directions.values() if d == "bull")
    bear_count   = sum(1 for d in directions.values() if d == "bear")
    n            = len(valid)
    max_consensus = max(bull_count, bear_count)
    agreement    = max_consensus / n if n > 0 else 0.5

    all_scores   = list(valid.values())
    dispersion   = max(all_scores) - min(all_scores)

    # ── Step 2: Conflict penalty — named, typed penalties ─────────────────────
    FUND_NAMES   = {"growth", "profitability", "financial_health"}
    fund_scores  = [s for nm, s in valid.items() if nm in FUND_NAMES]
    val_score    = valid.get("valuation")
    mom_score    = valid.get("momentum")

    fund_avg     = sum(fund_scores) / len(fund_scores) if fund_scores else None
    fund_strong  = fund_avg is not None and fund_avg >= 62  # clear fundamental strength
    fund_weak    = fund_avg is not None and fund_avg < 42   # clear fundamental weakness

    # Assess macro regime and recession risk from macro dict (optional)
    macro_regime  = (macro or {}).get("macro_regime", "")
    macro_risk    = (macro or {}).get("recession_risk_level", "")
    macro_adverse = macro_regime in ("Contraction", "Slowdown") or macro_risk in ("High", "Elevated")
    macro_avail   = bool(macro_regime)

    # Named conflicts: (description, penalty_amount)
    conflicts: list[tuple[str, float]] = []

    # A. Strong fundamentals vs weak momentum — price not confirming the thesis
    if fund_strong and mom_score is not None and mom_score < 45:
        conflicts.append((
            f"momentum ({mom_score:.0f}/100) not confirming strong fundamentals ({fund_avg:.0f}/100 avg)",
            0.10,
        ))
    # B. Weak fundamentals vs strong momentum — price run ahead of fundamentals
    elif fund_weak and mom_score is not None and mom_score >= 65:
        conflicts.append((
            f"momentum ({mom_score:.0f}/100) running ahead of weak fundamentals ({fund_avg:.0f}/100 avg)",
            0.08,
        ))
    # C. General directional mismatch (catches neutral-boundary cases not covered above)
    elif fund_avg is not None and mom_score is not None:
        fund_dir = _dir(fund_avg)
        mom_dir  = _dir(mom_score)
        if fund_dir != "neutral" and mom_dir != "neutral" and fund_dir != mom_dir:
            conflicts.append((
                f"momentum ({mom_score:.0f}/100) diverges from fundamentals ({fund_avg:.0f}/100 avg)",
                0.06,
            ))

    # D. Strong fundamentals vs stretched valuation
    if fund_strong and val_score is not None and val_score < 40:
        conflicts.append((
            f"valuation stretched (score {val_score:.0f}/100) against strong underlying fundamentals",
            0.08,
        ))

    # E. Strong fundamentals vs adverse macro regime
    if fund_strong and macro_avail and macro_adverse:
        regime_str = macro_regime if macro_regime else f"recession risk {macro_risk.lower()}"
        conflicts.append((
            f"macro headwind ({regime_str.lower()}) limits conviction despite strong fundamentals",
            0.07,
        ))

    # F. Wide dispersion — score spread reduces reliability regardless of direction
    if dispersion > 45:
        conflicts.append((
            f"wide score spread ({dispersion:.0f} pts) across categories",
            0.05,
        ))

    conflict_total = min(0.28, sum(p for _, p in conflicts))

    # ── Step 3: Completeness adjustment ──────────────────────────────────────
    # Neutral at ~4/6 categories; max ±0.04
    completeness    = n / EXPECTED_N
    completeness_adj = (completeness - (4 / EXPECTED_N)) * 0.10
    completeness_adj = max(-0.04, min(0.04, completeness_adj))

    # ── Step 4: Combine ───────────────────────────────────────────────────────
    base = 0.40 + agreement * 0.50      # [0.40, 0.90]
    conf = base - conflict_total + completeness_adj
    conf = max(0.25, min(0.92, conf))
    conf = round(conf, 3)

    # ── Step 5: Build structured explanation ─────────────────────────────────
    bull_names = [nm.replace("_", " ") for nm, d in directions.items() if d == "bull"]
    bear_names = [nm.replace("_", " ") for nm, d in directions.items() if d == "bear"]

    if conflicts:
        # Lead with what's agreeing, then name the conflicts
        if bull_count >= bear_count and bull_names:
            agree_part = f"Strong agreement across {_join(bull_names[:3])}"
        elif bear_names:
            agree_part = f"Broad bearish alignment across {_join(bear_names[:3])}"
        else:
            agree_part = "Signals partially aligned"

        if len(conflicts) == 1:
            conflict_part = conflicts[0][0]
            explanation = f"{agree_part}, but {conflict_part} — mixed signals reduce conviction."
        else:
            # Name the two most impactful conflicts
            top2 = sorted(conflicts, key=lambda x: x[1], reverse=True)[:2]
            c1, c2 = top2[0][0], top2[1][0]
            explanation = (
                f"{agree_part}; key conflicts — {c1}; and {c2} — conviction limited."
            )
    elif agreement >= 0.80 and dispersion < 35:
        # High conviction, clean signal
        if bull_count >= bear_count and bull_names:
            named = _join(bull_names[:3])
            explanation = f"High signal agreement across {named} — conviction well supported."
        elif bear_names:
            named = _join(bear_names[:3])
            explanation = f"High agreement on weakness across {named} — bearish conviction well supported."
        else:
            explanation = "Strong signal alignment — conviction well supported."
    elif agreement >= 0.60:
        if bull_count >= bear_count and bull_names:
            named = _join(bull_names[:3])
            explanation = f"Most factors ({named}) lean bullish — moderate conviction."
        elif bear_names:
            named = _join(bear_names[:3])
            explanation = f"Most factors ({named}) lean bearish — moderate conviction."
        else:
            explanation = "Majority of factors in agreement — moderate conviction."
    else:
        # Low agreement — name the poles
        _top = sorted(valid.items(), key=lambda x: x[1], reverse=True)
        _bot = sorted(valid.items(), key=lambda x: x[1])
        strong_str = _join([nm.replace("_", " ") for nm, _ in _top[:2]])
        weak_str   = _join([nm.replace("_", " ") for nm, _ in _bot[:2]])
        explanation = (
            f"Mixed signals — strong {strong_str} offset by weak {weak_str}"
            f" ({dispersion:.0f} pts spread); no clear directional consensus."
        )

    # Append macro note when macro is available but not already in conflict list
    macro_in_conflicts = any("macro" in desc for desc, _ in conflicts)
    if macro_avail and not macro_in_conflicts:
        if macro_adverse:
            regime_str = macro_regime if macro_regime else f"recession risk {macro_risk.lower()}"
            explanation += f" Macro ({regime_str.lower()}) adds a modest headwind."
        elif macro_regime in ("Expansion", "Recovery") and macro_risk not in ("High", "Elevated"):
            explanation += f" Macro backdrop ({macro_regime.lower()}) supports conviction."

    print(
        f"  [SIGNAL_CONF v2] agreement={agreement:.2f}"
        f" conflicts={len(conflicts)} total_penalty={conflict_total:.2f}"
        f" completeness={completeness:.2f} → conf={conf:.3f}"
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
