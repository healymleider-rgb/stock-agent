"""
ValuationRange — multi-method fair value estimation with PEG validation.

Generates bear/base/bull price targets using three independent methods:
  P/E-based  : current EPS × multiple scenarios
  EV/EBITDA  : current EBITDA × multiple → equity value per share
  P/S        : revenue per share × multiple scenarios

Multiple compression/expansion range is quality-adjusted:
  High quality (op. margin ≥ 20% or ROE ≥ 20%) → ±15% (narrower)
  Low  quality (op. margin < 10%, low ROE)      → ±25% (wider)
  Otherwise                                      → ±20% (default)

Outlier methods (base price > 2.5× or < 0.4× the median) are excluded
from aggregation when ≥ 2 methods are available.

Aggregate bear/base/bull targets use a profitability-weighted average.
PEG ratio (P/E ÷ annualised EPS growth rate) validates the multiple.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from models.stock_data import FinancialRatios, StockData

if TYPE_CHECKING:
    from analysis.metrics import NormalizedMetrics


# Base-case scale factor (always 1.0 — current multiple anchors the base)
_BASE_MULT = 1.00


@dataclass
class ValuationRange:
    """Output of compute_valuation_range()."""

    # Per-method implied prices (None when method is not computable)
    pe_bear:  Optional[float] = None
    pe_base:  Optional[float] = None
    pe_bull:  Optional[float] = None

    ev_bear:  Optional[float] = None
    ev_base:  Optional[float] = None
    ev_bull:  Optional[float] = None

    ps_bear:  Optional[float] = None
    ps_base:  Optional[float] = None
    ps_bull:  Optional[float] = None

    # Aggregate: profitability-weighted average across available (non-outlier) methods
    bear_price: Optional[float] = None
    base_price: Optional[float] = None
    bull_price: Optional[float] = None

    # PEG ratio validation
    peg_ratio:          Optional[float] = None
    eps_growth_rate:    Optional[float] = None   # annualised %, e.g. 12.5 = 12.5 %
    peg_interpretation: str = ""

    # One-line upside/downside summary
    upside_context: str = ""

    # Context
    current_price:  Optional[float] = None
    methods_used:   list[str] = field(default_factory=list)
    data_quality:   str = "missing"    # "good" | "partial" | "missing"

    # ── Scenario assumptions — per-scenario inputs for full driver transparency ─
    # Every number in the scenario table must be traceable to an explicit input.
    # Populated by compute_valuation_range(); consumed by _build_valuation_range_section.

    # Quality-adjusted multiple compression/expansion applied to all methods
    scenario_bear_mult: Optional[float] = None   # e.g. 0.80 for ±20% quality tier
    scenario_bull_mult: Optional[float] = None   # e.g. 1.20

    # Which method is featured as the primary driver (most appropriate for the company)
    scenario_primary_method: str = ""    # "P/E" | "EV/EBITDA" | "P/S"

    # P/E per-scenario inputs — multiple AND earnings project forward
    #   Bear: compressed multiple × flat EPS (no growth assumed)
    #   Base: current multiple × 1-year forward EPS at base CAGR
    #   Bull: expanded multiple × 1-year forward EPS at 130% of CAGR
    scenario_bear_pe: Optional[float] = None   # bear P/E = pe × bear_mult
    scenario_base_pe: Optional[float] = None   # base P/E = pe
    scenario_bull_pe: Optional[float] = None   # bull P/E = pe × bull_mult
    scenario_bear_eps: Optional[float] = None  # EPS in bear (TTM, no growth)
    scenario_base_eps: Optional[float] = None  # EPS in base (1yr fwd at base CAGR)
    scenario_bull_eps: Optional[float] = None  # EPS in bull (1yr fwd, 130% CAGR)

    # Legacy base-case aliases (kept for backward compat with API consumers)
    scenario_pe_multiple: Optional[float] = None   # = scenario_base_pe
    scenario_pe_eps:      Optional[float] = None   # = scenario_base_eps

    # EV/EBITDA per-scenario multiples (EV/EBITDA varies; EBITDA held flat for simplicity)
    scenario_bear_ev:   Optional[float] = None   # = ev_ebitda × bear_mult
    scenario_base_ev:   Optional[float] = None   # = ev_ebitda
    scenario_bull_ev:   Optional[float] = None   # = ev_ebitda × bull_mult
    # Legacy alias
    scenario_ev_multiple:   Optional[float] = None
    scenario_ev_ebitda_val: Optional[float] = None

    # P/S per-scenario multiples (P/S varies; rev/share held flat for simplicity)
    scenario_bear_ps:   Optional[float] = None   # = ps × bear_mult
    scenario_base_ps:   Optional[float] = None   # = ps
    scenario_bull_ps:   Optional[float] = None   # = ps × bull_mult
    # Legacy alias
    scenario_ps_multiple:      Optional[float] = None
    scenario_ps_rev_per_share: Optional[float] = None

    # Growth rate carried for PEG display
    scenario_growth_rate: Optional[float] = None   # annualised EPS CAGR %


# ── Quality multipliers ────────────────────────────────────────────────────────

def _quality_multipliers(ratios) -> tuple[float, float]:
    """
    Derive (bear_mult, bull_mult) from company quality.

    High quality (op. margin ≥ 20% or ROE ≥ 20%) → ±15%
      Rationale: premium multiples are stickier for high-quality businesses.
    Low  quality (op. margin < 10% or ROE < 5%)   → ±25%
      Rationale: multiple compression risk is higher for weaker businesses.
    Otherwise                                       → ±20% (default)
    """
    if ratios is None:
        return 0.80, 1.20

    op_margin = ratios.operating_margin   # decimal, e.g. 0.25 for 25 %
    roe       = ratios.roe

    high_quality = (
        (op_margin is not None and op_margin >= 0.20)
        or (roe is not None and roe >= 0.20)
    )
    low_quality = (
        (op_margin is not None and op_margin < 0.10)
        or (roe is not None and roe < 0.05)
    )

    if high_quality:
        return 0.85, 1.15
    if low_quality:
        return 0.75, 1.25
    return 0.80, 1.20


# ── Outlier filtering ──────────────────────────────────────────────────────────

def _simple_median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return (s[n // 2 - 1] + s[n // 2]) / 2 if n % 2 == 0 else s[n // 2]


def _filter_outliers(
    pe_base: Optional[float],
    ev_base: Optional[float],
    ps_base: Optional[float],
) -> set[str]:
    """
    Return the set of method keys to exclude as outliers before aggregation.

    A method is an outlier if its base-case implied price is > 2.5× or
    < 0.4× the median of all available base prices.
    Only applied when ≥ 2 methods have data.
    """
    candidates = {
        k: v for k, v in [("pe", pe_base), ("ev", ev_base), ("ps", ps_base)]
        if v is not None
    }
    if len(candidates) < 2:
        return set()

    med = _simple_median(list(candidates.values()))
    if med == 0:
        return set()

    return {k for k, v in candidates.items() if v / med > 2.5 or v / med < 0.4}


# ── Aggregation ────────────────────────────────────────────────────────────────

def _weighted_aggregate(
    pe_val: Optional[float],
    ev_val: Optional[float],
    ps_val: Optional[float],
    is_profitable: bool,
) -> Optional[float]:
    """
    Compute a weighted average across available valuation method outputs.

    Profitable company (positive earnings):
      P/E = 40%  |  EV/EBITDA = 40%  |  P/S = 20%

    Unprofitable company:
      EV/EBITDA = 30%  |  P/S = 70%  |  P/E excluded

    Weights are re-normalised over whichever methods have non-None values so
    a missing method does not silently pull the result toward zero.
    """
    if is_profitable:
        target_weights = {"pe": 0.40, "ev": 0.40, "ps": 0.20}
    else:
        target_weights = {"pe": 0.00, "ev": 0.30, "ps": 0.70}

    candidates = {"pe": pe_val, "ev": ev_val, "ps": ps_val}
    available  = {k: v for k, v in candidates.items() if v is not None and target_weights[k] > 0}

    if not available:
        return None

    total_w = sum(target_weights[k] for k in available)
    if total_w == 0:
        return None

    result = sum(v * target_weights[k] / total_w for k, v in available.items())
    return round(result, 2)


# ── Ratio derivation from statements ──────────────────────────────────────────

def _derive_ratios(stock_data: StockData) -> Optional[FinancialRatios]:
    """
    Synthesise key valuation multiples from raw statements when ratios are absent
    (e.g. AlphaVantage only returns statement data, not a ratios endpoint).

    Returns a FinancialRatios with only the fields we can derive; all others
    remain None so existing None-checks downstream are unaffected.
    Returns None when no multiples can be computed at all.
    """
    income  = stock_data.latest_income
    balance = stock_data.latest_balance
    price   = stock_data.current_price
    mkt_cap = stock_data.market_cap

    if income is None:
        return None

    print(
        f"  [VAL DEBUG] income fields — "
        f"revenue={income.revenue}, net_income={income.net_income}, "
        f"ebitda={income.ebitda}, eps={income.eps}, eps_diluted={income.eps_diluted}"
    )
    print(
        f"  [VAL DEBUG] context — "
        f"price={price}, market_cap={mkt_cap}, "
        f"balance={'ok' if balance else 'None'}"
    )
    if balance:
        print(
            f"  [VAL DEBUG] balance fields — "
            f"total_debt={balance.total_debt}, cash={balance.cash_and_equivalents}, "
            f"total_equity={balance.total_equity}"
        )

    # ── Shares outstanding (needed for per-share metrics) ─────────────────────
    shares: Optional[float] = None
    if income.eps_diluted and income.eps_diluted > 0 and income.net_income and income.net_income > 0:
        shares = abs(income.net_income / income.eps_diluted)
        print(f"  [VAL DEBUG] shares from income (net/eps_diluted): {shares:,.0f}")
    elif income.eps and income.eps > 0 and income.net_income and income.net_income > 0:
        shares = abs(income.net_income / income.eps)
        print(f"  [VAL DEBUG] shares from income (net/eps): {shares:,.0f}")
    elif mkt_cap and price and price > 0:
        shares = mkt_cap / price
        print(f"  [VAL DEBUG] shares from mkt_cap/price: {shares:,.0f}")
    if not shares:
        print(f"  [VAL DEBUG] shares=None ← MISSING (eps_diluted={income.eps_diluted}, mkt_cap={mkt_cap}, price={price})")

    # ── EPS: prefer stated value, fall back to net_income / shares ───────────
    eps = income.eps_diluted or income.eps
    if eps is None and income.net_income and shares and shares > 0:
        eps = income.net_income / shares
    print(f"  [VAL DEBUG] derived EPS={eps}"
          + ("" if eps else
             f" ← MISSING (net_income={income.net_income}, shares={shares})"))

    # ── P/E ───────────────────────────────────────────────────────────────────
    pe_ratio: Optional[float] = None
    if not price:
        print("  [VAL] skip P/E: price is None")
    elif not eps or eps <= 0:
        print(f"  [VAL] skip P/E: EPS is {eps}")
    else:
        pe_ratio = round(price / eps, 2)
        print(f"  [VAL] computed P/E={pe_ratio:.2f} (price={price}, eps={eps:.2f})")

    # ── P/S ───────────────────────────────────────────────────────────────────
    ps_ratio: Optional[float] = None
    if not mkt_cap:
        print("  [VAL] skip P/S: market_cap is None")
    elif not income.revenue or income.revenue <= 0:
        print(f"  [VAL] skip P/S: revenue is {income.revenue}")
    else:
        ps_ratio = round(mkt_cap / income.revenue, 2)
        print(f"  [VAL] computed P/S={ps_ratio:.2f}"
              f" (mkt_cap={mkt_cap:.0f}, rev={income.revenue:.0f})")

    # ── EV / EBITDA ───────────────────────────────────────────────────────────
    ev_to_ebitda: Optional[float] = None
    if not mkt_cap:
        print("  [VAL] skip EV/EBITDA: market_cap is None")
    elif not income.ebitda or income.ebitda <= 0:
        print(f"  [VAL] skip EV/EBITDA: EBITDA is {income.ebitda}")
    else:
        debt = (balance.total_debt or 0.0) if balance else 0.0
        cash = (balance.cash_and_equivalents or 0.0) if balance else 0.0
        ev = mkt_cap + debt - cash
        if ev > 0:
            ev_to_ebitda = round(ev / income.ebitda, 2)
            print(f"  [VAL] computed EV/EBITDA={ev_to_ebitda:.2f}"
                  f" (EV={ev:.0f}, EBITDA={income.ebitda:.0f})")
        else:
            print(f"  [VAL] skip EV/EBITDA: EV={ev} (mkt_cap={mkt_cap},"
                  f" debt={debt}, cash={cash})")

    # ── Operating margin + ROE (used by _quality_multipliers) ─────────────────
    op_margin: Optional[float] = None
    if income.operating_income and income.revenue and income.revenue > 0:
        op_margin = income.operating_income / income.revenue

    roe: Optional[float] = None
    if (income.net_income and balance
            and balance.total_equity and balance.total_equity > 0):
        roe = income.net_income / balance.total_equity

    if pe_ratio is None and ps_ratio is None and ev_to_ebitda is None:
        return None  # nothing computable — let caller handle missing ratios

    return FinancialRatios(
        date=income.date or "derived",
        period="FY",
        pe_ratio=pe_ratio,
        ps_ratio=ps_ratio,
        ev_to_ebitda=ev_to_ebitda,
        operating_margin=op_margin,
        roe=roe,
    )


# ── Public entry point ─────────────────────────────────────────────────────────

def compute_valuation_range(
    stock_data: StockData,
    metrics: "Optional[NormalizedMetrics]" = None,
) -> ValuationRange:
    """
    Build a ValuationRange from StockData.

    If *metrics* is provided (a pre-computed NormalizedMetrics from
    analysis.metrics.compute_core_metrics), its validated price, shares,
    TTM EPS, PE, PS, and EV/EBITDA values are used directly — no re-derivation.

    This guarantees that the scenario analysis is anchored to exactly the
    same numbers shown in the report header and peer comparison table.

    Returns a ValuationRange with data_quality="missing" if insufficient
    inputs are available.
    """
    vr = ValuationRange()

    income  = stock_data.latest_income
    balance = stock_data.latest_balance

    # ── Resolve inputs: prefer pre-computed metrics, fall back to self-derive ──
    if metrics is not None:
        price      = metrics.price
        shares     = metrics.shares
        market_cap = metrics.market_cap
        ttm_eps    = metrics.ttm_eps
        pe         = metrics.pe_ratio
        ps         = metrics.ps_ratio
        ev_ebitda  = metrics.ev_ebitda
        shares_str = f"{shares:,.0f}" if shares else "N/A"
        print(
            f"  [VAL] using NormalizedMetrics:"
            f" price={price}({metrics.price_source})"
            f" pe={pe}({metrics.pe_source})"
            f" ps={ps}({metrics.ps_source})"
            f" ev_ebitda={ev_ebitda}({metrics.ev_ebitda_source})"
            f" shares={shares_str}"
            f" ttm_eps={ttm_eps}"
        )
        # Ratios still needed for quality multipliers
        ratios = stock_data.latest_ratios or _derive_ratios(stock_data)
    else:
        # Self-derive (legacy path — used when metrics is not passed)
        print("  [VAL] NormalizedMetrics not provided — self-deriving inputs")
        price      = stock_data.current_price
        market_cap = stock_data.market_cap
        ratios     = stock_data.latest_ratios or _derive_ratios(stock_data)

        # Price cross-check
        ph_price: Optional[float] = None
        if stock_data.price_history and stock_data.price_history.closes:
            ph_price = stock_data.price_history.closes[0]
        if price and ph_price and ph_price > 0:
            divergence = abs(price - ph_price) / ph_price
            if divergence > 0.10:
                print(
                    f"  [VAL WARN] price divergence {divergence:.1%}:"
                    f" quote={price} history={ph_price} — using history"
                )
                price = ph_price
            else:
                print(f"  [VAL] price check OK: quote={price} history={ph_price} Δ={divergence:.1%}")
        elif ph_price and not price:
            price = ph_price

        # Shares
        shares = None
        if income and income.net_income and income.eps_diluted:
            ni, epsd = income.net_income, income.eps_diluted
            if (ni > 0 and epsd > 0) or (ni < 0 and epsd < 0):
                shares = abs(ni / epsd)
                print(f"  [VAL] shares from net_income/eps_diluted: {shares:,.0f}")
        if shares is None and income and income.net_income and income.eps:
            ni, eps = income.net_income, income.eps
            if (ni > 0 and eps > 0) or (ni < 0 and eps < 0):
                shares = abs(ni / eps)
                print(f"  [VAL] shares from net_income/eps: {shares:,.0f}")
        if shares is None and market_cap and price and price > 0:
            shares = market_cap / price
            print(f"  [VAL] shares from mktcap/price: {shares:,.0f}")
        if shares is None and stock_data.shares_outstanding:
            shares = stock_data.shares_outstanding
            print(f"  [VAL] shares from /quote: {shares:,.0f}")

        # TTM EPS
        ttm_eps = None
        quarters = stock_data.quarterly_income[:4] if stock_data.quarterly_income else []
        if len(quarters) >= 4:
            q_eps = [q.eps_diluted or q.eps for q in quarters if (q.eps_diluted or q.eps) is not None]
            q_ni  = [q.net_income for q in quarters if q.net_income is not None]
            if len(q_eps) == 4:
                ttm_eps = sum(q_eps)
            elif len(q_ni) == 4 and shares:
                ttm_eps = sum(q_ni) / shares
        if ttm_eps is not None:
            print(f"  [VAL] TTM EPS={ttm_eps:.4f}")

        # Multiples
        pe        = ratios.pe_ratio      if ratios else None
        ps        = ratios.ps_ratio      if ratios else None
        ev_ebitda = ratios.ev_to_ebitda  if ratios else None

    vr.current_price = price

    # ── Extract EPS growth rate early — used in forward EPS projections ────────
    # When metrics is provided (normal pipeline path) we use the system-wide CAGR.
    # On the legacy self-derive path, growth rate will be computed later in the
    # PEG section; forward projections will fall back to flat EPS in that case.
    _eps_growth_pct: Optional[float] = (
        metrics.eps_growth_pct if metrics is not None else None
    )

    # ── Profitability flag (drives aggregation weights) ────────────────────────
    net_income_val = income.net_income if income else None
    _eps_val       = (income.eps_diluted or income.eps) if income else None
    is_profitable  = (
        (net_income_val is not None and net_income_val > 0)
        or (_eps_val is not None and _eps_val > 0)
    )
    if metrics is not None and metrics.ttm_eps is not None:
        is_profitable = metrics.ttm_eps > 0

    # Quality-adjusted multiple range
    ratios_for_quality = stock_data.latest_ratios or _derive_ratios(stock_data)
    bear_mult, bull_mult = _quality_multipliers(ratios_for_quality)

    # Record multipliers for report transparency
    vr.scenario_bear_mult = bear_mult
    vr.scenario_bull_mult = bull_mult

    # Upper clamp for implied prices
    price_cap = (price * 5.0) if price else None

    # ── EPS for P/E method ────────────────────────────────────────────────────
    # Use TTM EPS so the multiple is consistent with the displayed P/E.
    eps = ttm_eps
    eps_source = "TTM" if eps is not None else ""
    if eps is None:
        eps = (income.eps_diluted or income.eps) if income else None
        eps_source = "annual"
    if eps is None and income and income.net_income and shares and shares > 0:
        eps = income.net_income / shares
        eps_source = "derived"

    # ── P/E method — two-dimensional scenario: multiple AND EPS vary ─────────
    # Bear : compressed multiple × flat TTM EPS (no growth — most conservative)
    # Base : current multiple × 1-year forward EPS at the annualised CAGR
    # Bull : expanded multiple × 1-year forward EPS at 130 % of the CAGR
    # This produces distinct, independently motivated scenarios rather than
    # a single EPS held constant while only the multiple shifts.
    if not pe or pe <= 0:
        print(f"  [VAL] skip P/E method: pe={pe}")
    elif not eps or eps <= 0:
        print(f"  [VAL] skip P/E method: EPS={eps} (source: {eps_source})")
    else:
        _g = max(0.0, _eps_growth_pct / 100) if (_eps_growth_pct and _eps_growth_pct > 0) else 0.0
        _bear_pe  = round(pe * bear_mult, 2)
        _base_pe  = round(pe, 2)
        _bull_pe  = round(pe * bull_mult, 2)
        _bear_eps = round(eps, 4)                      # flat — no growth assumed
        _base_eps = round(eps * (1.0 + _g), 4)        # 1-year forward at CAGR
        _bull_eps = round(eps * (1.0 + _g * 1.3), 4)  # 1-year forward, 30% acceleration

        def _pp(mult: float, e: float) -> float:
            raw = max(0.0, mult * e)
            return round(min(raw, price_cap) if price_cap else raw, 2)

        vr.pe_bear = _pp(_bear_pe, _bear_eps)
        vr.pe_base = _pp(_base_pe, _base_eps)
        vr.pe_bull = _pp(_bull_pe, _bull_eps)
        vr.methods_used.append("P/E")

        # Per-scenario inputs for driver table
        vr.scenario_bear_pe  = _bear_pe
        vr.scenario_base_pe  = _base_pe
        vr.scenario_bull_pe  = _bull_pe
        vr.scenario_bear_eps = _bear_eps
        vr.scenario_base_eps = _base_eps
        vr.scenario_bull_eps = _bull_eps
        # Legacy aliases
        vr.scenario_pe_multiple = _base_pe
        vr.scenario_pe_eps      = _base_eps
        print(
            f"  [VAL] P/E method: pe={pe:.2f} eps={eps:.4f}({eps_source})"
            f" g={_g*100:.1f}% | bear={_bear_pe}×${_bear_eps:.2f}=${vr.pe_bear}"
            f" base={_base_pe}×${_base_eps:.2f}=${vr.pe_base}"
            f" bull={_bull_pe}×${_bull_eps:.2f}=${vr.pe_bull}"
        )

    # ── EV/EBITDA method — multiple varies, EBITDA held flat ─────────────────
    ebitda     = income.ebitda if income else None
    total_debt = balance.total_debt if balance else None
    cash_val   = balance.cash_and_equivalents if balance else None

    if not ev_ebitda or ev_ebitda <= 0:
        print(f"  [VAL] skip EV/EBITDA method: ev_ebitda={ev_ebitda}")
    elif not ebitda or ebitda <= 0:
        print(f"  [VAL] skip EV/EBITDA method: EBITDA={ebitda}")
    elif not shares or shares <= 0:
        print(f"  [VAL] skip EV/EBITDA method: shares unavailable")
    else:
        print(f"  [VAL] EV/EBITDA method: ev_ebitda={ev_ebitda} EBITDA={ebitda:.0f} shares={shares:.0f}")
        vr.ev_bear, vr.ev_base, vr.ev_bull = _ev_scenario_prices(
            ev_ebitda, ebitda, total_debt, cash_val, shares, bear_mult, bull_mult, price_cap
        )
        vr.methods_used.append("EV/EBITDA")
        vr.scenario_bear_ev     = round(ev_ebitda * bear_mult, 2)
        vr.scenario_base_ev     = round(ev_ebitda, 2)
        vr.scenario_bull_ev     = round(ev_ebitda * bull_mult, 2)
        # Legacy aliases
        vr.scenario_ev_multiple   = round(ev_ebitda, 2)
        vr.scenario_ev_ebitda_val = round(ebitda, 0)

    # ── P/S method — multiple varies, rev/share held flat ────────────────────
    revenue = income.revenue if income else None

    if not ps or ps <= 0:
        print(f"  [VAL] skip P/S method: ps={ps}")
    elif not revenue or revenue <= 0:
        print(f"  [VAL] skip P/S method: revenue={revenue}")
    elif not shares or shares <= 0:
        print(f"  [VAL] skip P/S method: shares unavailable")
    else:
        rev_per_share = revenue / shares
        print(f"  [VAL] P/S method: ps={ps} revenue={revenue:.0f} shares={shares:.0f}")
        vr.ps_bear, vr.ps_base, vr.ps_bull = _scenario_prices(
            ps, rev_per_share, bear_mult, bull_mult, price_cap
        )
        vr.methods_used.append("P/S")
        vr.scenario_bear_ps         = round(ps * bear_mult, 2)
        vr.scenario_base_ps         = round(ps, 2)
        vr.scenario_bull_ps         = round(ps * bull_mult, 2)
        # Legacy aliases
        vr.scenario_ps_multiple      = round(ps, 2)
        vr.scenario_ps_rev_per_share = round(rev_per_share, 4)

    # ── Outlier filtering ─────────────────────────────────────────────────────
    outliers = _filter_outliers(vr.pe_base, vr.ev_base, vr.ps_base)
    for key in outliers:
        setattr(vr, f"{key}_bear", None)
        setattr(vr, f"{key}_base", None)
        setattr(vr, f"{key}_bull", None)
    if outliers:
        print(f"  [VAL] outlier methods removed: {outliers}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    vr.bear_price = _weighted_aggregate(vr.pe_bear, vr.ev_bear, vr.ps_bear, is_profitable)
    vr.base_price = _weighted_aggregate(vr.pe_base, vr.ev_base, vr.ps_base, is_profitable)
    vr.bull_price = _weighted_aggregate(vr.pe_bull, vr.ev_bull, vr.ps_bull, is_profitable)

    # ── Data quality ──────────────────────────────────────────────────────────
    filtered = []
    if vr.pe_base is not None:
        filtered.append("P/E")
    if vr.ev_base is not None:
        filtered.append("EV/EBITDA")
    if vr.ps_base is not None:
        filtered.append("P/S")
    vr.methods_used = filtered
    n = len(filtered)
    vr.data_quality = "good" if n >= 2 else "partial" if n == 1 else "missing"

    # ── Primary method selection — AFTER outlier filtering ───────────────────
    # Must run after filtering so it only selects methods whose implied prices
    # actually survived.  P/E is preferred (most intuitive); EV/EBITDA is used
    # when EPS is negative; P/S is the last resort for pre-profit companies.
    if vr.pe_base is not None and vr.scenario_bear_eps is not None and (vr.scenario_bear_eps or 0) > 0:
        vr.scenario_primary_method = "P/E"
    elif vr.ev_base is not None:
        vr.scenario_primary_method = "EV/EBITDA"
    elif vr.ps_base is not None:
        vr.scenario_primary_method = "P/S"

    # ── Upside context ────────────────────────────────────────────────────────
    if vr.base_price is not None and price and price > 0:
        upside_pct = (vr.base_price / price - 1.0) * 100
        vr.upside_context = _upside_context_line(upside_pct)

    # ── PEG ──────────────────────────────────────────────────────────────────
    # Use pre-computed values when available to keep methodology consistent
    if metrics is not None and metrics.eps_growth_pct is not None:
        vr.eps_growth_rate = metrics.eps_growth_pct
        vr.peg_ratio       = metrics.peg
        if metrics.peg is not None:
            _g = metrics.eps_growth_pct
            vr.peg_interpretation = _peg_interpretation(metrics.peg, _g)
        else:
            vr.peg_interpretation = "PEG not computable"
    else:
        vr.peg_ratio, vr.eps_growth_rate, vr.peg_interpretation = _compute_peg(
            stock_data.income_statements, pe, shares=shares, ttm_eps=ttm_eps
        )

    # Carry growth rate as a named scenario field for report transparency
    vr.scenario_growth_rate = vr.eps_growth_rate

    return vr


# ── Helpers ────────────────────────────────────────────────────────────────────

def _scenario_prices(
    multiple: float,
    per_share_metric: float,
    bear_mult: float,
    bull_mult: float,
    price_cap: Optional[float],
) -> tuple[float, float, float]:
    """Return (bear, base, bull) implied prices."""
    def _p(factor: float) -> float:
        raw = max(0.0, multiple * factor * per_share_metric)
        return round(min(raw, price_cap) if price_cap else raw, 2)

    return _p(bear_mult), _p(_BASE_MULT), _p(bull_mult)


def _ev_scenario_prices(
    ev_ebitda: float,
    ebitda: float,
    total_debt: Optional[float],
    cash: Optional[float],
    shares: float,
    bear_mult: float,
    bull_mult: float,
    price_cap: Optional[float],
) -> tuple[float, float, float]:
    """
    Derive implied equity price per share from EV/EBITDA scenarios.

    implied EV       = ev_multiple × EBITDA
    implied equity   = implied EV − total_debt + cash
    implied price    = equity / shares
    """
    debt = total_debt or 0.0
    cash = cash or 0.0

    def _p(factor: float) -> float:
        equity_val = ev_ebitda * factor * ebitda - debt + cash
        raw = max(0.0, equity_val / shares)
        return round(min(raw, price_cap) if price_cap else raw, 2)

    return _p(bear_mult), _p(_BASE_MULT), _p(bull_mult)


def _upside_context_line(upside_pct: float) -> str:
    """One-sentence summary of base-case implied upside/downside."""
    sign = "+" if upside_pct >= 0 else ""
    pct_str = f"{sign}{upside_pct:.0f}%"

    if upside_pct < -10:
        label = "downside risk at current multiples"
    elif upside_pct <= 10:
        label = "roughly fairly valued at current multiples"
    elif upside_pct <= 25:
        label = "moderate upside to base case"
    else:
        label = "meaningful upside to base case"

    return f"Base case implies {pct_str} vs current price — {label}."


def _peg_interpretation(peg: float, growth_pct: float) -> str:
    """Return a PEG interpretation string given peg and growth rate."""
    if peg < 1.0:
        band = "undervalued relative to growth"
    elif peg < 1.5:
        band = "reasonable relative to growth"
    elif peg < 2.5:
        band = "slightly expensive relative to growth"
    else:
        band = "expensive relative to growth"
    interp = f"PEG {peg:.2f} — {band}"
    if growth_pct < 5.0:
        interp += " (low growth makes PEG less meaningful)"
    elif growth_pct > 25.0:
        interp += " (high growth may justify elevated multiples)"
    return interp


def _compute_peg(
    income_statements: list,
    current_pe: Optional[float],
    shares: Optional[float] = None,
    ttm_eps: Optional[float] = None,
) -> tuple[Optional[float], Optional[float], str]:
    """
    Compute PEG ratio = current_pe / EPS_growth_rate_pct.

    Uses the annualised EPS CAGR over the most recent 2–3 years.
    If ttm_eps is provided it is used as the "latest" EPS point, which
    makes the growth rate more current than the last annual report.
    When eps fields are absent, falls back to net_income / shares.
    Returns (peg_ratio, eps_growth_pct, interpretation_string).
    """
    if not income_statements or current_pe is None or current_pe <= 0:
        return None, None, "PEG not computable — P/E unavailable"

    # Collect EPS from annual statements, newest first.
    # Fall back to net_income / shares when eps fields are absent.
    eps_values: list[float] = []
    for stmt in income_statements:
        eps = stmt.eps_diluted or stmt.eps
        if eps is None and shares and shares > 0 and stmt.net_income:
            eps = stmt.net_income / shares
        if eps is not None:
            eps_values.append(eps)

    if len(eps_values) < 2:
        return None, None, "Insufficient EPS history for PEG calculation"

    n_years = min(len(eps_values) - 1, 3)
    # Use TTM EPS as the most-recent point when available; otherwise use annual
    if ttm_eps is not None and ttm_eps > 0:
        eps_latest = ttm_eps
        eps_oldest = eps_values[n_years]
        print(f"  [VAL] PEG using TTM EPS={ttm_eps:.4f} as latest, annual[{n_years}]={eps_oldest:.4f}")
    else:
        eps_latest = eps_values[0]
        eps_oldest = eps_values[n_years]

    if eps_oldest <= 0 or eps_latest <= 0:
        return None, None, "Negative or zero EPS in history — PEG not meaningful"

    growth_rate = (eps_latest / eps_oldest) ** (1.0 / n_years) - 1.0
    growth_pct  = round(growth_rate * 100, 1)   # e.g. 12.5
    print(f"  [VAL] computed growth={growth_pct:.1f}% annualised"
          f" (EPS {eps_oldest:.2f} → {eps_latest:.2f} over {n_years}y)")

    if growth_pct <= 0:
        return None, growth_pct, (
            f"Negative EPS growth ({growth_pct:.1f}% annualised) — PEG not meaningful"
        )

    peg = round(current_pe / growth_pct, 2)

    # ── Four-band interpretation ───────────────────────────────────────────────
    if peg < 1.0:
        band = "undervalued relative to growth"
    elif peg < 1.5:
        band = "reasonable relative to growth"
    elif peg < 2.5:
        band = "slightly expensive relative to growth"
    else:
        band = "expensive relative to growth"

    interp = f"PEG {peg:.2f} — {band}"

    # Growth rate qualifiers
    if growth_pct < 5.0:
        interp += " (low growth makes PEG less meaningful)"
    elif growth_pct > 25.0:
        interp += " (high growth may justify elevated multiples)"

    return peg, growth_pct, interp
