"""
ValuationRange — driver-based fair value estimation with multi-method cross-checks.

Primary model (driver-based):
  Revenue growth → operating margin → FCF conversion → exit multiple → equity/share
  Each scenario shows what changed (growth deceleration, margin compression, re-rating)
  rather than anchoring on the current P/E.

Cross-check methods (secondary, not aggregated into primary output):
  P/E-based  : current EPS × multiple scenarios
  EV/EBITDA  : current EBITDA × multiple → equity value per share
  P/S        : revenue per share × multiple scenarios

Falls back to the profitability-weighted average of cross-check methods when
driver inputs are insufficient (negative op margin, no revenue, no shares).

Multiple compression/expansion range is quality-adjusted:
  High quality (op. margin ≥ 20% or ROE ≥ 20%) → ±15% (narrower)
  Low  quality (op. margin < 10%, low ROE)      → ±25% (wider)
  Otherwise                                      → ±20% (default)

Outlier methods (base price > 2.5× or < 0.4× the median) are excluded from
the fallback aggregation when ≥ 2 methods are available.

PEG ratio (P/E ÷ annualised EPS growth rate) validates the multiple.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Optional

from models.stock_data import FinancialRatios, StockData

if TYPE_CHECKING:
    from analysis.metrics import NormalizedMetrics
    from analysis.monte_carlo import MCResult

# ── Guidance overrides ────────────────────────────────────────────────────────
# Management guidance file — loaded once at module import.
# Format: data/guidance_overrides.json
# Callers pass the ticker; we return guidance that is ≤ 90 days old.

_GUIDANCE_MAX_AGE_DAYS = 90
_GUIDANCE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "data", "guidance_overrides.json"
)


def _load_guidance_overrides() -> dict:
    try:
        with open(_GUIDANCE_FILE, "r") as fh:
            raw = json.load(fh)
        return {k: v for k, v in raw.items() if not k.startswith("_")}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


_GUIDANCE_OVERRIDES: dict = _load_guidance_overrides()


def _get_guidance(ticker: str, as_of: date | None = None) -> dict | None:
    """
    Return guidance entry for *ticker* if it exists and is ≤ 90 days old.
    Returns None when no valid guidance is found.
    """
    entry = _GUIDANCE_OVERRIDES.get(ticker.upper())
    if not entry:
        return None
    try:
        guidance_date = datetime.strptime(entry["date"], "%Y-%m-%d").date()
    except (KeyError, ValueError):
        return None
    ref_date = as_of or date.today()
    age = (ref_date - guidance_date).days
    if age > _GUIDANCE_MAX_AGE_DAYS:
        print(
            f"  [GUIDANCE] {ticker}: guidance from {guidance_date} is {age}d old "
            f"(>{_GUIDANCE_MAX_AGE_DAYS}d) — ignoring"
        )
        return None
    return entry


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

    # ── Driver-based scenario model — fundamental projections ─────────────────
    # Populated when op_margin > 0 and revenue history is available.
    # When available, these override the multiple-based weighted aggregate
    # as bear_price / base_price / bull_price.
    driver_model_available: bool = False

    # Per-scenario revenue growth (decimal, e.g. 0.10 = 10 %)
    scenario_bear_rev_growth: Optional[float] = None
    scenario_base_rev_growth: Optional[float] = None
    scenario_bull_rev_growth: Optional[float] = None

    # Per-scenario operating margin (decimal)
    scenario_bear_op_margin: Optional[float] = None
    scenario_base_op_margin: Optional[float] = None
    scenario_bull_op_margin: Optional[float] = None

    # Per-scenario FCF conversion (FCF / EBIT, decimal, 0.30–0.95)
    scenario_bear_fcf_conv: Optional[float] = None
    scenario_base_fcf_conv: Optional[float] = None
    scenario_bull_fcf_conv: Optional[float] = None

    # Per-scenario exit multiple (EV/FCF equivalent)
    scenario_bear_exit_mult: Optional[float] = None
    scenario_base_exit_mult: Optional[float] = None
    scenario_bull_exit_mult: Optional[float] = None

    # Forward projections (absolute $, for display in report)
    scenario_bear_fwd_rev: Optional[float] = None
    scenario_base_fwd_rev: Optional[float] = None
    scenario_bull_fwd_rev: Optional[float] = None

    scenario_bear_fwd_fcf: Optional[float] = None
    scenario_base_fwd_fcf: Optional[float] = None
    scenario_bull_fwd_fcf: Optional[float] = None

    # Attribution label — one line per scenario showing what changed
    scenario_bear_label: str = ""
    scenario_base_label: str = ""
    scenario_bull_label: str = ""

    # ── Trend-driven adjustments applied to base-case driver assumptions ───────
    # Populated when compute_valuation_range() receives a TrendResult.
    # Used by the reporting agent to render "Trend Impact on Valuation".
    trend_margin_adj:    float       = 0.0    # pp applied to base op_margin (decimal)
    trend_rev_adj:       float       = 0.0    # rate applied to base rev_growth (decimal)
    trend_impact_lines:  list[str]   = field(default_factory=list)

    # ── Company quality inputs — stored for MC regime re-runs ─────────────────
    # Set by compute_valuation_range() from ratios_for_quality so that the
    # reporting agent can call mc_from_valuation_range() with macro regime
    # without needing access to the raw ratios object.
    quality_gross_margin: Optional[float] = field(default=None)
    quality_op_margin:    Optional[float] = field(default=None)

    # ── Monte Carlo simulation results ────────────────────────────────────────
    # Populated by compute_valuation_range() after the deterministic scenarios
    # are finalised.  None when inputs are insufficient (negative EPS, no price).
    mc: Optional["MCResult"] = field(default=None)


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
    Derive quality-scoring ratios (operating_margin, roe) from raw statements.

    SSOT contract
    ─────────────
    P/E, P/S, and EV/EBITDA are NEVER computed here.  Those are the exclusive
    domain of NormalizedMetrics (analysis.metrics.compute_core_metrics).  Any
    caller that needs valuation multiples must pass a NormalizedMetrics instance
    to compute_valuation_range(); the metrics=None legacy path is the only
    consumer of this function's FinancialRatios output.

    This function exists solely to populate the quality-multiplier inputs
    (operating_margin, roe) when stock_data.latest_ratios is absent.  It must
    not introduce an alternative source for any metric that appears in the report.
    """
    income  = stock_data.latest_income
    balance = stock_data.latest_balance

    if income is None:
        return None

    # ── Operating margin (used by _quality_multipliers) ────────────────────────
    op_margin: Optional[float] = None
    if income.operating_income and income.revenue and income.revenue > 0:
        op_margin = income.operating_income / income.revenue

    # ── ROE (used by _quality_multipliers) ────────────────────────────────────
    roe: Optional[float] = None
    if (income.net_income and balance
            and balance.total_equity and balance.total_equity > 0):
        roe = income.net_income / balance.total_equity

    if op_margin is None and roe is None:
        return None  # nothing to offer for quality assessment

    return FinancialRatios(
        date=income.date or "derived",
        period="FY",
        pe_ratio=None,          # SSOT: read from NormalizedMetrics.pe_ratio
        ps_ratio=None,          # SSOT: read from NormalizedMetrics.ps_ratio
        ev_to_ebitda=None,      # SSOT: read from NormalizedMetrics.ev_ebitda
        operating_margin=op_margin,
        roe=roe,
    )


# ── Driver-based scenario model helpers ───────────────────────────────────────

def _rev_cagr_3y(income_statements: list) -> Optional[float]:
    """
    Compute revenue CAGR from annual income statements (newest first).
    Prefers 3-year CAGR; falls back to 1-year growth when only 2 periods exist.
    """
    stmts = income_statements or []
    if len(stmts) >= 4:
        r0 = stmts[0].revenue
        r3 = stmts[3].revenue
        if r0 and r3 and r3 > 0:
            return (r0 / r3) ** (1.0 / 3) - 1.0
    if len(stmts) >= 2:
        r0 = stmts[0].revenue
        r1 = stmts[1].revenue
        if r0 and r1 and r1 > 0:
            return r0 / r1 - 1.0
    return None


def _fcf_conv_from_statements(cash_flows: list, operating_income: Optional[float]) -> Optional[float]:
    """
    Derive FCF conversion = TTM FCF / TTM EBIT.

    Returns a rate clamped to [0.30, 0.98].
    Returns None when EBIT is non-positive or data is absent.
    """
    if not cash_flows or not operating_income or operating_income <= 0:
        return None
    ttm_fcf = cash_flows[0].free_cash_flow if cash_flows else None
    if ttm_fcf is None:
        return None
    ratio = ttm_fcf / operating_income
    return max(0.30, min(0.98, ratio))


def _ev_fcf_multiple(
    market_cap: Optional[float],
    net_debt: float,
    ttm_fcf: Optional[float],
    pe_ratio: Optional[float],
) -> float:
    """
    Derive EV/FCF exit multiple (primary), with P/E-based fallback.

    Priority:
    1. EV / TTM FCF when TTM FCF > 0
    2. P/E ÷ 0.75  — assumes ~75 % FCF/earnings conversion rate
    3. 18× default — broad-market median for quality businesses
    """
    if market_cap and ttm_fcf and ttm_fcf > 0:
        ev = market_cap + net_debt
        mult = ev / ttm_fcf
        return max(5.0, min(60.0, mult))
    if pe_ratio and pe_ratio > 0:
        return max(5.0, min(60.0, pe_ratio / 0.75))
    return 18.0


def _driver_scenario_price(
    base_revenue:  float,
    rev_growth:    float,
    op_margin:     float,
    fcf_conv:      float,
    exit_mult:     float,
    shares:        float,
    net_debt:      float = 0.0,
    price_cap:     Optional[float] = None,
) -> tuple[float, float, float]:
    """
    Fundamental driver chain: revenue → EBIT → FCF → equity value → price.

    fwd_revenue = base_revenue × (1 + rev_growth)
    fwd_ebit    = fwd_revenue  × op_margin
    fwd_fcf     = fwd_ebit     × fcf_conv
    equity_val  = fwd_fcf × exit_mult − net_debt
    price       = max(0, equity_val) / shares

    Returns (fwd_revenue, fwd_fcf, price_per_share).
    """
    fwd_rev   = base_revenue * (1.0 + rev_growth)
    fwd_ebit  = fwd_rev      * op_margin
    fwd_fcf   = fwd_ebit     * fcf_conv
    equity    = max(0.0, fwd_fcf * exit_mult - net_debt)
    raw_price = equity / shares if shares > 0 else 0.0
    price     = round(min(raw_price, price_cap) if price_cap else raw_price, 2)
    return fwd_rev, fwd_fcf, price


def _driver_mc(
    base_revenue: float,
    bear_rev_g:   float,
    base_rev_g:   float,
    bull_rev_g:   float,
    bear_op_mg:   float,
    base_op_mg:   float,
    bull_op_mg:   float,
    bear_fcf:     float,
    base_fcf:     float,
    bull_fcf:     float,
    bear_exit:    float,
    base_exit:    float,
    bull_exit:    float,
    shares:       float,
    net_debt:     float,
    current_price: float,
    price_cap:    Optional[float] = None,
    n_sims:       int = 750,
    base_price:   Optional[float] = None,  # deterministic base-case for anchoring
) -> Optional["MCResult"]:
    """
    Driver-chain Monte Carlo using triangular distributions over the four
    fundamental inputs (Rev Growth, Op Margin, FCF Conversion, Exit Multiple).

    Each path: sample g, m, f, x from triangular(bear, base, bull) then apply
    the same driver chain as _driver_scenario_price().  Returns an MCResult
    with the same structure as the P/E-based Monte Carlo so all downstream
    sizing / reporting code is unchanged.

    Uses stdlib random only — no numpy/scipy.
    """
    import random  as _rnd
    import math    as _math
    import hashlib as _hl

    if current_price <= 0 or shares <= 0 or base_revenue <= 0:
        return None

    # Deterministic seed derived from driver inputs — same inputs → same MC path.
    # Uses SHA-256 (not hash()) so the seed is stable regardless of PYTHONHASHSEED.
    _seed_blob = (
        f"{base_revenue:.8g}|{base_rev_g:.8g}|{base_op_mg:.8g}"
        f"|{base_fcf:.8g}|{base_exit:.8g}|{current_price:.8g}"
    ).encode("ascii")
    _seed = int.from_bytes(_hl.sha256(_seed_blob).digest()[:4], "little")
    rng = _rnd.Random(_seed)

    prices:  list[float] = []
    returns: list[float] = []

    for _ in range(n_sims):
        # Clamp so low <= mode <= high for triangular (bear may exceed base when
        # the trend adjustment shifted the base below the raw bear scenario).
        _rg_lo  = min(bear_rev_g, base_rev_g, bull_rev_g)
        _rg_hi  = max(bear_rev_g, base_rev_g, bull_rev_g)
        _rg_md  = base_rev_g  # mode = base case

        _om_lo  = min(bear_op_mg, base_op_mg, bull_op_mg)
        _om_hi  = max(bear_op_mg, base_op_mg, bull_op_mg)
        _om_md  = base_op_mg

        _fc_lo  = min(bear_fcf, base_fcf, bull_fcf)
        _fc_hi  = max(bear_fcf, base_fcf, bull_fcf)
        _fc_md  = base_fcf

        _ex_lo  = min(bear_exit, base_exit, bull_exit)
        _ex_hi  = max(bear_exit, base_exit, bull_exit)
        _ex_md  = base_exit

        # If all three are identical, triangular(x, x, x) = x (degenerate but valid)
        rev_g = rng.triangular(_rg_lo, _rg_hi, _rg_md)
        op_mg = rng.triangular(_om_lo, _om_hi, _om_md)
        fcf_c = rng.triangular(_fc_lo, _fc_hi, _fc_md)
        ex_mt = rng.triangular(_ex_lo, _ex_hi, _ex_md)

        # Enforce economic floor: op_margin must be positive for FCF to be positive
        op_mg = max(op_mg, 0.001)
        fcf_c = max(fcf_c, 0.05)
        ex_mt = max(ex_mt, 1.0)

        _, _, px = _driver_scenario_price(
            base_revenue, rev_g, op_mg, fcf_c, ex_mt,
            shares, net_debt, price_cap,
        )
        ret = px / current_price - 1.0
        prices.append(px)
        returns.append(ret)

    # ── PART 1: Anchor distribution to base-case scenario ────────────────────
    # Triangular distribution mean = (lo + mode + hi) / 3 ≠ mode.
    # Nonlinear driver chain further shifts the expected price away from the
    # deterministic base case.  Rescale if deviation > 10% so the distribution
    # is centred on the scenario base case, not the triangular mean.
    if base_price is not None and base_price > 0 and len(prices) > 0:
        _raw_mean_px = sum(prices) / len(prices)
        if _raw_mean_px > 0:
            _deviation = abs(_raw_mean_px / base_price - 1.0)
            if _deviation > 0.10:
                _scale = base_price / _raw_mean_px
                prices  = [p * _scale for p in prices]
                print(
                    f"  [MC:anchor] raw_mean={_raw_mean_px:.2f} base={base_price:.2f}"
                    f" deviation={_deviation:.1%} → rescaled ×{_scale:.4f}"
                )
        # Recompute returns from (possibly rescaled) prices before sorting
        returns = [p / current_price - 1.0 for p in prices]

    prices.sort()
    returns.sort()

    n   = len(returns)
    mu  = sum(returns) / n
    var = sum((r - mu) ** 2 for r in returns) / n
    std = _math.sqrt(var) if var > 0 else 1e-9

    # 3rd standardised moment (skewness)
    sk  = sum((r - mu) ** 3 for r in returns) / (n * std ** 3) if std > 1e-12 else 0.0

    # Half-Kelly: E[R] / Var[R] / 2, clamped [0, 10%]
    half_kelly = (mu / var / 2.0) if var > 0 else 0.0
    half_kelly = max(0.0, min(half_kelly, 0.10))

    def _pct(lst: list[float], p: float) -> float:
        """Linear-interpolation percentile."""
        _n = len(lst)
        if _n == 0:
            return 0.0
        idx = p * (_n - 1)
        lo  = int(idx)
        hi  = min(lo + 1, _n - 1)
        return lst[lo] + (idx - lo) * (lst[hi] - lst[lo])

    p5r  = _pct(returns, 0.05)
    p95r = _pct(returns, 0.95)
    ud   = abs(p95r / p5r) if p5r < 0 else (p95r / 0.01 if p95r > 0 else 1.0)

    from analysis.monte_carlo import MCResult as _MCResult
    return _MCResult(
        n_sims        = n_sims,
        horizon_years = 1,
        method        = "driver",
        growth_mean   = base_rev_g,
        growth_std    = (_rg_hi - _rg_lo) / (2.0 * 1.645),
        mean_return   = mu,
        median_return = _pct(returns, 0.50),
        p5_return     = p5r,
        p25_return    = _pct(returns, 0.25),
        p75_return    = _pct(returns, 0.75),
        p95_return    = p95r,
        skewness      = sk,
        prob_positive = sum(1 for r in returns if r > 0.0) / n,
        prob_20_gain  = sum(1 for r in returns if r > 0.20) / n,
        prob_loss     = sum(1 for r in returns if r < 0.0) / n,
        prob_loss_20  = sum(1 for r in returns if r < -0.20) / n,
        mean_price    = sum(prices) / n,
        p5_price      = _pct(prices, 0.05),
        p25_price     = _pct(prices, 0.25),
        median_price  = _pct(prices, 0.50),
        p75_price     = _pct(prices, 0.75),
        p95_price     = _pct(prices, 0.95),
        kelly_fraction  = half_kelly,
        upside_downside = ud,
    )


# ── Scenario-derived distribution (replaces independent MC) ───────────────────

def _scenario_derived_mc(
    bear_price: float,
    base_price: float,
    bull_price: float,
    current_price: float,
    base_rev_g: float = 0.0,
) -> "Optional[MCResult]":
    """
    Build a MCResult directly from the scenario-tree prices so that the
    distribution section of the report is always consistent with the
    scenario section.

    Mapping:  Bear → P5,  Base → P50,  Bull → P95
    Interpolation (mathematically exact given uniform spacing assumption):
      P25 = P5  + (P50 − P5)  × (20/45)
      P75 = P50 + (P95 − P50) × (25/45)

    The CDF is piecewise-linear between (P5, 0.05), (P25, 0.25),
    (P50, 0.50), (P75, 0.75), (P95, 0.95).  All probabilities are
    derived from this CDF analytically — no random sampling.

    This guarantees: P50 == Base scenario price (to machine precision).
    There is exactly ONE forecast in the system, expressed two ways.
    """
    import math as _math

    if current_price <= 0 or bear_price <= 0 or base_price <= 0 or bull_price <= 0:
        return None

    # ── Percentile anchors ────────────────────────────────────────────────────
    p5  = bear_price
    p50 = base_price
    p95 = bull_price

    p25 = p5  + (p50 - p5)  * (20.0 / 45.0)
    p75 = p50 + (p95 - p50) * (25.0 / 45.0)

    # Returns at each anchor
    def _r(px: float) -> float:
        return px / current_price - 1.0

    r5, r25, r50, r75, r95 = _r(p5), _r(p25), _r(p50), _r(p75), _r(p95)

    # ── Mean return: canonical 3-scenario probability-weighted formula ────────
    # Bear→P5, Base→P50, Bull→P95 with scenario tree weights (0.25, 0.50, 0.25).
    # Using cumulative probabilities as weights (the prior bug) gives weights that
    # sum to 2.5, producing an impossibly large E[R].  This formula sums to 1.0.
    _P_BEAR, _P_BASE, _P_BULL = 0.25, 0.50, 0.25
    mean_r = _P_BEAR * r5 + _P_BASE * r50 + _P_BULL * r95

    # Validation: E[R] must lie within [min_scenario, max_scenario]
    _r_min = min(r5, r50, r95)
    _r_max = max(r5, r50, r95)
    if not (_r_min <= mean_r <= _r_max):
        raise AssertionError(
            f"_scenario_derived_mc: E[R] {mean_r:.4f} outside scenario range "
            f"[{_r_min:.4f}, {_r_max:.4f}] — formula error"
        )

    # ── Variance and skewness using the same 3-scenario weights ───────────────
    var  = (_P_BEAR * (r5  - mean_r) ** 2
          + _P_BASE * (r50 - mean_r) ** 2
          + _P_BULL * (r95 - mean_r) ** 2)
    std  = _math.sqrt(var) if var > 0 else 1e-9
    skew = (
        _P_BEAR * ((r5  - mean_r) / std) ** 3 +
        _P_BASE * ((r50 - mean_r) / std) ** 3 +
        _P_BULL * ((r95 - mean_r) / std) ** 3
    ) if std > 1e-12 else 0.0

    # ── Half-Kelly: E[R] / Var[R] / 2, clamped [0, 10%] ─────────────────────
    half_kelly = (mean_r / var / 2.0) if var > 0 else 0.0
    half_kelly = max(0.0, min(half_kelly, 0.10))

    # ── Probability estimates from piecewise-linear CDF ─────────────────────
    # CDF nodes: (price, cumulative_prob)
    _cdf_nodes = [(p5, 0.05), (p25, 0.25), (p50, 0.50), (p75, 0.75), (p95, 0.95)]

    def _cdf(px: float) -> float:
        """Piecewise-linear CDF interpolation."""
        if px <= _cdf_nodes[0][0]:
            return 0.05
        if px >= _cdf_nodes[-1][0]:
            return 0.95
        for i in range(len(_cdf_nodes) - 1):
            lo_px, lo_p = _cdf_nodes[i]
            hi_px, hi_p = _cdf_nodes[i + 1]
            if lo_px <= px <= hi_px:
                if hi_px == lo_px:
                    return lo_p
                t = (px - lo_px) / (hi_px - lo_px)
                return lo_p + t * (hi_p - lo_p)
        return 0.95

    prob_loss    = _cdf(current_price)
    prob_loss_20 = _cdf(current_price * 0.80)
    prob_positive = 1.0 - prob_loss
    prob_20_gain  = 1.0 - _cdf(current_price * 1.20)

    p95r = r95
    ud   = abs(p95r / r5) if r5 < 0 else (p95r / 0.01 if p95r > 0 else 1.0)

    # ── Growth std approximation from the price range ─────────────────────────
    growth_std = (bull_price - bear_price) / (base_price * 2 * 1.645) if base_price > 0 else 0.05

    from analysis.monte_carlo import MCResult as _MCResult
    return _MCResult(
        n_sims        = 0,          # 0 = derived, not sampled
        horizon_years = 1,
        method        = "scenario_derived",
        growth_mean   = base_rev_g,
        growth_std    = round(growth_std, 4),
        mean_return   = round(mean_r,  6),
        median_return = round(r50,     6),
        p5_return     = round(r5,      6),
        p25_return    = round(r25,     6),
        p75_return    = round(r75,     6),
        p95_return    = round(r95,     6),
        skewness      = round(skew,    4),
        prob_positive = round(prob_positive, 4),
        prob_20_gain  = round(prob_20_gain,  4),
        prob_loss     = round(prob_loss,     4),
        prob_loss_20  = round(prob_loss_20,  4),
        mean_price    = round((p5 + p25 + p50 + p75 + p95) / 5.0, 2),
        p5_price      = round(p5,  2),
        p25_price     = round(p25, 2),
        median_price  = round(p50, 2),
        p75_price     = round(p75, 2),
        p95_price     = round(p95, 2),
        kelly_fraction  = round(half_kelly, 4),
        upside_downside = round(ud, 4),
    )


# ── Public entry point ─────────────────────────────────────────────────────────

def compute_valuation_range(
    stock_data: StockData,
    metrics: "Optional[NormalizedMetrics]" = None,
    trends: "Optional[object]" = None,   # TrendResult from analysis.trend
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
        # LEGACY PATH — metrics=None violates the SSOT contract.
        # All key metrics (price, market_cap, EPS, P/E, P/S, EV/EBITDA, FCF)
        # must flow from compute_core_metrics() → NormalizedMetrics → here.
        # This branch is retained only for isolated unit tests of valuation logic
        # that construct StockData directly without running the full pipeline.
        import warnings as _warnings
        _warnings.warn(
            "compute_valuation_range() called without NormalizedMetrics. "
            "Metric values derived here may differ from those shown in the report. "
            "Always pass metrics=compute_core_metrics(stock_data) in production.",
            UserWarning, stacklevel=2,
        )
        print(
            "  [VAL WARNING] NormalizedMetrics not provided — SSOT contract violated. "
            "Self-deriving inputs from raw statements. Values may differ from report header."
        )
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

    # Store quality margins for MC regime re-runs in reporting_agent
    if ratios_for_quality is not None:
        vr.quality_gross_margin = getattr(ratios_for_quality, "gross_margin",    None)
        vr.quality_op_margin    = getattr(ratios_for_quality, "operating_margin", None)

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

    # ── Multiple-based aggregate (fallback / cross-check) ────────────────────
    # Always computed; overridden by the driver model when available.
    vr.bear_price = _weighted_aggregate(vr.pe_bear, vr.ev_bear, vr.ps_bear, is_profitable)
    vr.base_price = _weighted_aggregate(vr.pe_base, vr.ev_base, vr.ps_base, is_profitable)
    vr.bull_price = _weighted_aggregate(vr.pe_bull, vr.ev_bull, vr.ps_bull, is_profitable)

    # ── Driver-based scenario model ───────────────────────────────────────────
    # Primary valuation: revenue growth → op margin → FCF conversion → exit mult.
    # Overrides the multiple-based aggregate when inputs are sufficient.
    # Falls back silently when op_margin ≤ 0 (unprofitable) or data is missing.

    _ttm_rev: Optional[float] = (
        income.revenue if income and income.revenue and income.revenue > 0 else None
    )
    _cur_op_mg: Optional[float] = None
    if income and income.operating_income and income.revenue and income.revenue > 0:
        _cur_op_mg = income.operating_income / income.revenue

    _rev_cagr = _rev_cagr_3y(stock_data.income_statements)

    _cfs = stock_data.cash_flows
    # SSOT: FCF comes from NormalizedMetrics when available.  Falling back to
    # stock_data.cash_flows is only permitted on the metrics=None legacy path.
    _ttm_fcf_abs: Optional[float] = (
        metrics.ttm_fcf if metrics is not None
        else (_cfs[0].free_cash_flow if _cfs else None)
    )
    # FCF conversion ratio derived from the SSOT FCF value — never re-reads raw
    _op_income_for_fcf = income.operating_income if income else None
    _fcf_conv_base: float = (
        max(0.30, min(0.98, _ttm_fcf_abs / _op_income_for_fcf))
        if (_ttm_fcf_abs is not None
            and _op_income_for_fcf is not None and _op_income_for_fcf > 0)
        else None
    ) or 0.75

    _net_debt: float = 0.0
    if balance:
        _net_debt = (balance.total_debt or 0.0) - (balance.cash_and_equivalents or 0.0)

    _exit_mult_base = _ev_fcf_multiple(market_cap, _net_debt, _ttm_fcf_abs, pe)

    _driver_ok = (
        _ttm_rev is not None and _ttm_rev > 0
        and _cur_op_mg is not None and _cur_op_mg > 0.005   # at least minimally profitable
        and shares is not None and shares > 0
    )

    if _driver_ok:
        # ── Scenario-specific driver assumptions ─────────────────────────────
        # Revenue growth: trend ±5pp; clamp bear ≥ −10%, bull ≤ 50%
        _base_rev_g = _rev_cagr if _rev_cagr is not None else 0.04
        _bear_rev_g = max(-0.10, _base_rev_g - 0.05)
        _bull_rev_g = min(0.50,  _base_rev_g + 0.05)

        # Op margin: bear −3pp, bull +2pp; clamp to [0.5%, 60%]
        _bear_op_mg = max(0.005, _cur_op_mg - 0.03)
        _base_op_mg = _cur_op_mg
        _bull_op_mg = min(0.60,  _cur_op_mg + 0.02)

        # ── Apply trend-driven adjustments to BASE-CASE assumptions ──────────
        # Bear and bull bounds shift proportionally because they are derived
        # from the new base, so the spread direction is preserved.
        _trend_margin_adj = 0.0
        _trend_rev_adj    = 0.0
        _trend_impact_lines: list[str] = []

        if trends is not None:
            _trend_margin_adj = getattr(trends, "valuation_margin_adj", 0.0)
            _trend_rev_adj    = getattr(trends, "valuation_rev_adj",    0.0)

            if _trend_margin_adj != 0.0:
                _raw_base_op_mg = _base_op_mg
                _base_op_mg = max(0.005, min(0.60, _base_op_mg + _trend_margin_adj))
                # Recalculate bear/bull from updated base
                _bear_op_mg = max(0.005, _base_op_mg - 0.03)
                _bull_op_mg = min(0.60,  _base_op_mg + 0.02)
                _direction  = "Expanding" if _trend_margin_adj > 0 else "Deteriorating"
                _trend_impact_lines.append(
                    f"Op margin {_direction.lower()} → base margin "
                    f"{_base_op_mg:.1%} ({_trend_margin_adj*100:+.0f}pp vs current {_raw_base_op_mg:.1%})"
                )

            if _trend_rev_adj != 0.0:
                _raw_base_rev_g = _base_rev_g
                _base_rev_g = max(-0.10, _base_rev_g + _trend_rev_adj)
                _bear_rev_g = max(-0.10, _base_rev_g - 0.05)
                _bull_rev_g = min(0.50,  _base_rev_g + 0.05)
                _rev_trend  = getattr(trends, "revenue_growth", "Deteriorating")
                _trend_impact_lines.append(
                    f"Revenue trend {_rev_trend.lower()} → base rev growth "
                    f"{_base_rev_g:+.0%} ({_trend_rev_adj*100:+.0f}pp vs raw CAGR {_raw_base_rev_g:+.0%})"
                )

            if not _trend_impact_lines:
                # Stable/no-adjustment: emit neutral note
                _trend_impact_lines.append("Trend signals neutral — no adjustment to base-case drivers")

            vr.trend_margin_adj   = _trend_margin_adj
            vr.trend_rev_adj      = _trend_rev_adj
            vr.trend_impact_lines = _trend_impact_lines

            print(
                f"  [TREND→VAL] margin_adj={_trend_margin_adj*100:+.0f}pp"
                f" rev_adj={_trend_rev_adj*100:+.0f}pp"
                f" → base_op_mg={_base_op_mg:.1%} base_rev_g={_base_rev_g:+.1%}"
            )

        # ── Management guidance override for FCF ─────────────────────────────
        # If a recent earnings release provides an explicit full-year FCF
        # target (≤ 90 days old), derive the implied FCF conversion ratio
        # from that guidance and use it as the base — overriding the TTM
        # historical ratio.  Bear and bull ratios are rescaled proportionally
        # so the bear/base/bull spread is preserved.
        _guidance = _get_guidance(stock_data.ticker)
        _fcf_guidance_used = False
        _fcf_conv_adj = _fcf_conv_base   # may be replaced below
        if (
            _guidance is not None
            and _guidance.get("fcf") is not None
            and _ttm_rev is not None and _ttm_rev > 0
            and _base_op_mg > 0
        ):
            _guide_fcf_abs = float(_guidance["fcf"])
            # Forward EBIT = projected revenue × base op margin
            _fwd_rev_est   = _ttm_rev * (1.0 + _base_rev_g)
            _fwd_ebit_est  = _fwd_rev_est * _base_op_mg
            if _fwd_ebit_est > 0:
                _fcf_conv_adj = _guide_fcf_abs / _fwd_ebit_est
                # Clamp to economically plausible range [0.30, 0.95]
                _fcf_conv_adj = max(0.30, min(0.95, _fcf_conv_adj))
                print(
                    f"  [GUIDANCE] {stock_data.ticker}: FCF guidance ${_guide_fcf_abs/1e9:.1f}B "
                    f"→ implied fcf_conv={_fcf_conv_adj:.3f} "
                    f"(was {_fcf_conv_base:.3f}, src: {_guidance.get('source','?')})"
                )
                _fcf_guidance_used = True
            else:
                print(
                    f"  [GUIDANCE] {stock_data.ticker}: guidance present but fwd_ebit≤0 "
                    f"— ignoring guidance"
                )

        # FCF conversion: bear −10pp, bull +5pp; clamp to [0.30, 0.95]
        # Rescale bear/bull proportionally when guidance changed the base.
        if _fcf_guidance_used and _fcf_conv_base > 0:
            _scale = _fcf_conv_adj / _fcf_conv_base
            _bear_fcf = max(0.30, min(0.95, (max(0.30, _fcf_conv_base - 0.10)) * _scale))
            _base_fcf = _fcf_conv_adj
            _bull_fcf = max(0.30, min(0.95, (min(0.95, _fcf_conv_base + 0.05)) * _scale))
        else:
            _bear_fcf = max(0.30, _fcf_conv_base - 0.10)
            _base_fcf = _fcf_conv_base
            _bull_fcf = min(0.95, _fcf_conv_base + 0.05)

        # Exit multiple: quality-adjusted compression/expansion
        _bear_exit = max(5.0, _exit_mult_base * bear_mult)
        _base_exit = _exit_mult_base
        _bull_exit = min(80.0, _exit_mult_base * bull_mult)

        # ── Driver chain for each scenario ───────────────────────────────────
        _drv_bear_rev, _drv_bear_fcf, _drv_bear_px = _driver_scenario_price(
            _ttm_rev, _bear_rev_g, _bear_op_mg, _bear_fcf,
            _bear_exit, shares, _net_debt, price_cap,
        )
        _drv_base_rev, _drv_base_fcf, _drv_base_px = _driver_scenario_price(
            _ttm_rev, _base_rev_g, _base_op_mg, _base_fcf,
            _base_exit, shares, _net_debt, price_cap,
        )
        _drv_bull_rev, _drv_bull_fcf, _drv_bull_px = _driver_scenario_price(
            _ttm_rev, _bull_rev_g, _bull_op_mg, _bull_fcf,
            _bull_exit, shares, _net_debt, price_cap,
        )

        # ── Attribution labels — what changed per scenario ───────────────────
        _mg_bear_delta = (_cur_op_mg - _bear_op_mg) * 100
        _mg_bull_delta = (_bull_op_mg - _cur_op_mg) * 100
        vr.scenario_bear_label = (
            f"Revenue {_bear_rev_g:+.0%} (trend −5pp), "
            f"margin {_bear_op_mg:.1%} (−{_mg_bear_delta:.0f}pp), "
            f"exit {_bear_exit:.1f}× (de-rates {(1-bear_mult):.0%})"
        )
        vr.scenario_base_label = (
            f"Revenue {_base_rev_g:+.0%} (3Y trend"
            + (f", trend-adj {_trend_rev_adj*100:+.0f}pp" if _trend_rev_adj != 0.0 else "")
            + f"), margin {_base_op_mg:.1%}"
            + (f" (trend-adj {_trend_margin_adj*100:+.0f}pp)" if _trend_margin_adj != 0.0 else " (unchanged)")
            + f", exit {_base_exit:.1f}×"
        )
        vr.scenario_bull_label = (
            f"Revenue {_bull_rev_g:+.0%} (trend +5pp), "
            f"margin {_bull_op_mg:.1%} (+{_mg_bull_delta:.0f}pp), "
            f"exit {_bull_exit:.1f}× (re-rates {(bull_mult-1):.0%})"
        )

        # ── Override aggregate with driver results ────────────────────────────
        vr.bear_price = _drv_bear_px
        vr.base_price = _drv_base_px
        vr.bull_price = _drv_bull_px
        vr.driver_model_available = True

        # Store all driver inputs and projections
        vr.scenario_bear_rev_growth = round(_bear_rev_g, 4)
        vr.scenario_base_rev_growth = round(_base_rev_g, 4)
        vr.scenario_bull_rev_growth = round(_bull_rev_g, 4)

        vr.scenario_bear_op_margin  = round(_bear_op_mg, 4)
        vr.scenario_base_op_margin  = round(_base_op_mg, 4)
        vr.scenario_bull_op_margin  = round(_bull_op_mg, 4)

        vr.scenario_bear_fcf_conv   = round(_bear_fcf, 3)
        vr.scenario_base_fcf_conv   = round(_base_fcf, 3)
        vr.scenario_bull_fcf_conv   = round(_bull_fcf, 3)

        vr.scenario_bear_exit_mult  = round(_bear_exit, 2)
        vr.scenario_base_exit_mult  = round(_base_exit, 2)
        vr.scenario_bull_exit_mult  = round(_bull_exit, 2)

        vr.scenario_bear_fwd_rev    = round(_drv_bear_rev, 0)
        vr.scenario_base_fwd_rev    = round(_drv_base_rev, 0)
        vr.scenario_bull_fwd_rev    = round(_drv_bull_rev, 0)

        vr.scenario_bear_fwd_fcf    = round(_drv_bear_fcf, 0)
        vr.scenario_base_fwd_fcf    = round(_drv_base_fcf, 0)
        vr.scenario_bull_fwd_fcf    = round(_drv_bull_fcf, 0)

        _fcf_src = "EV/FCF" if (_ttm_fcf_abs and _ttm_fcf_abs > 0 and market_cap) else "P/E-implied"
        print(
            f"  [DRIVER] rev_cagr={_base_rev_g:+.1%}"
            f" op_margin={_cur_op_mg:.1%}"
            f" fcf_conv={_fcf_conv_base:.2f}"
            f" exit_mult={_exit_mult_base:.1f}× ({_fcf_src})"
        )
        print(
            f"  [DRIVER] bear={_drv_bear_px} base={_drv_base_px} bull={_drv_bull_px}"
        )
    else:
        _driver_skip_reasons = []
        if not _ttm_rev or _ttm_rev <= 0:
            _driver_skip_reasons.append("no revenue")
        if _cur_op_mg is None or _cur_op_mg <= 0.005:
            _driver_skip_reasons.append(f"op_margin={_cur_op_mg}")
        if not shares or shares <= 0:
            _driver_skip_reasons.append("no shares")
        print(
            f"  [DRIVER] skipped ({'; '.join(_driver_skip_reasons)}) "
            f"— using multiple-based aggregate"
        )

    # ── Data quality ──────────────────────────────────────────────────────────
    filtered = []
    if vr.driver_model_available:
        filtered.append("driver")
    if vr.pe_base is not None:
        filtered.append("P/E")
    if vr.ev_base is not None:
        filtered.append("EV/EBITDA")
    if vr.ps_base is not None:
        filtered.append("P/S")
    vr.methods_used = filtered
    n_check = len(filtered)
    # Driver model alone counts as "good" — it uses multiple inputs internally.
    # Without driver, need ≥ 2 multiple-based methods for "good".
    if vr.driver_model_available:
        vr.data_quality = "good"
    elif n_check >= 2:
        vr.data_quality = "good"
    elif n_check == 1:
        vr.data_quality = "partial"
    else:
        vr.data_quality = "missing"

    # ── Primary method selection ──────────────────────────────────────────────
    # Driver model is primary when available.  Falls back to the most
    # informative multiple-based method (P/E > EV/EBITDA > P/S).
    if vr.driver_model_available:
        vr.scenario_primary_method = "driver"
    elif vr.pe_base is not None and vr.scenario_bear_eps is not None and (vr.scenario_bear_eps or 0) > 0:
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

    # ── Probabilistic distribution — derived from scenario tree ──────────────
    # The distribution is built DIRECTLY from the three scenario prices:
    #   Bear → P5,  Base → P50,  Bull → P95
    # Intermediate percentiles are interpolated.  This guarantees:
    #   (1) P50 == Base scenario price (no divergence possible)
    #   (2) P95 ≥ P50 ≥ P5 always holds (structural consistency)
    #   (3) Fully deterministic — same inputs → byte-identical outputs
    #
    # The previous approach ran an independent triangular-distribution MC
    # with tighter bounds (±3pp vs ±5pp), which caused P50 to drift above
    # Scenario Bull in asymmetric scenarios (e.g. NFLX 4/21/26 P50=$141.55
    # vs Bull=$144.43).  There is now exactly ONE forecast in the system,
    # expressed two ways (scenario table + percentile distribution).
    if vr.current_price and vr.current_price > 0:
        try:
            if (
                vr.bear_price is not None and vr.bear_price > 0
                and vr.base_price is not None and vr.base_price > 0
                and vr.bull_price is not None and vr.bull_price > 0
            ):
                _base_rg_for_mc = vr.scenario_base_rev_growth or 0.0
                vr.mc = _scenario_derived_mc(
                    bear_price    = vr.bear_price,
                    base_price    = vr.base_price,
                    bull_price    = vr.bull_price,
                    current_price = vr.current_price,
                    base_rev_g    = _base_rg_for_mc,
                )

            if vr.mc:
                print(
                    f"  [MC] scenario_derived: "
                    f"P5=${vr.mc.p5_price:.2f} "
                    f"P50=${vr.mc.median_price:.2f} (== Base) "
                    f"P95=${vr.mc.p95_price:.2f} "
                    f"P(gain)={vr.mc.prob_positive:.0%}"
                )
        except Exception as _mc_err:
            print(f"  [MC] distribution skipped: {_mc_err}")

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
    if peg < 0.8:
        band = "growth-adjusted discount"
    elif peg <= 1.2:
        band = "fairly valued relative to growth"
    elif peg < 2.0:
        band = "moderate premium to growth"
    else:
        band = "significant premium to growth"
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
    if peg < 0.8:
        band = "growth-adjusted discount"
    elif peg <= 1.2:
        band = "fairly valued relative to growth"
    elif peg < 2.0:
        band = "moderate premium to growth"
    else:
        band = "significant premium to growth"

    interp = f"PEG {peg:.2f} — {band}"

    # Growth rate qualifiers
    if growth_pct < 5.0:
        interp += " (low growth makes PEG less meaningful)"
    elif growth_pct > 25.0:
        interp += " (high growth may justify elevated multiples)"

    return peg, growth_pct, interp
