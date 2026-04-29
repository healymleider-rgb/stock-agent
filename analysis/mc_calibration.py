"""
mc_calibration.py
=================
Data-driven calibration of Monte Carlo growth and multiple distributions.

Derives GrowthDistParams and MultipleDistParams directly from a company's
historical income statements and ratio history — replacing the fixed regime /
quality tables with empirically grounded parameters where data is sufficient.

Design
------
Growth calibration (EPS / revenue)
  · Year-over-year growth rates are extracted from annual income statements.
  · The two-piece (split) normal is fitted directly: sigma_down from
    below-mean observations, sigma_up from above-mean observations.
  · Shock probability is the empirical fraction of years with growth below
    (mean − 1.5 × sigma_down) — the historical left-tail frequency.
  · Falls back to revenue when EPS series is too short or has negative bases.

Multiple calibration (P/E or P/S)
  · Mean reversion speed is estimated via OLS AR(1) on the historical series.
  · Beta concentration is calibrated from the empirical spread of the series
    within the bear/bull bounds, using the identity
    Var_Beta = μ(1−μ)/(c+1) solved for c.
  · Historical p10/p90 become the "high confidence" bounds.

Peer adjustment
  · A configurable shrinkage weight blends the company's own statistics
    toward the peer median (default 30% peer weight).

Confidence
  · "high"   : n_obs ≥ 8 years
  · "medium" : n_obs ≥ 4 years
  · "low"    : n_obs < 4 years — parameters are returned but flagged

Integration
-----------
    from analysis.mc_calibration import calibrate_mc_params
    from analysis.monte_carlo import run_monte_carlo

    calib = calibrate_mc_params(stock_data, multiple_bear=pe_bear,
                                multiple_base=pe_base, multiple_bull=pe_bull,
                                method="P/E")
    if calib:
        result = run_monte_carlo(
            ...,
            growth_params   = calib.growth_params,
            multiple_params = calib.multiple_params,
        )
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.stock_data import StockData
    from analysis.monte_carlo import GrowthDistParams, MultipleDistParams


# ── Minimum observations to produce calibrated parameters ─────────────────────
_MIN_OBS_HARD   = 3   # below this → return None (use defaults)
_MIN_OBS_MEDIUM = 4   # below this → "low" confidence
_MIN_OBS_HIGH   = 8   # at or above → "high" confidence

# Peer shrinkage weight: 0 = ignore peers, 1 = use only peer median
_PEER_WEIGHT: float = 0.30


# ── Output dataclasses ─────────────────────────────────────────────────────────

@dataclass
class GrowthCalibration:
    """Empirical statistics derived from a historical growth series."""
    n_obs:       int     # number of valid YoY observations
    mean:        float   # arithmetic mean growth (decimal)
    median:      float   # median growth
    std:         float   # sample standard deviation
    skewness:    float   # 3rd standardised central moment
    p5:          float   # 5th percentile — historical severe downside
    p95:         float   # 95th percentile — historical strong upside
    sigma_down:  float   # std dev of below-mean observations (SplitNormal left)
    sigma_up:    float   # std dev of above-mean observations (SplitNormal right)
    shock_prob:  float   # empirical P(growth < mean − 1.5 × sigma_down)
    source:      str     # "eps" | "revenue" | "net_income"
    confidence:  str     # "high" | "medium" | "low"


@dataclass
class MultipleCalibration:
    """Empirical statistics derived from a historical valuation multiple series."""
    n_obs:         int
    mean:          float   # historical average multiple
    std:           float   # historical standard deviation
    p10:           float   # 10th percentile (historical trough)
    p90:           float   # 90th percentile (historical peak)
    mr_speed:      float   # mean reversion speed [0, 1] estimated from AR(1)
    mr_halflife:   float   # implied half-life in years
    concentration: float   # Beta α + β calibrated from empirical spread
    source:        str     # "pe" | "ps" | "ev_ebitda"
    confidence:    str


@dataclass
class MCCalibrationResult:
    """Complete calibration output — ready to pass to run_monte_carlo()."""
    growth_calib:    GrowthCalibration
    multiple_calib:  MultipleCalibration
    growth_params:   "GrowthDistParams"
    multiple_params: "MultipleDistParams"
    notes:           list[str] = field(default_factory=list)


# ── Statistics helpers (stdlib only) ──────────────────────────────────────────

def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


def _std(vals: list[float], mu: Optional[float] = None) -> float:
    if len(vals) < 2:
        return 0.0
    mu = mu if mu is not None else _mean(vals)
    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(var)


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile. p ∈ [0, 1]."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    idx = p * (n - 1)
    lo  = int(idx)
    hi  = lo + 1
    if hi >= n:
        return sorted_vals[-1]
    return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])


def _skewness(vals: list[float], mu: float, sigma: float) -> float:
    if sigma <= 0 or len(vals) < 3:
        return 0.0
    n = len(vals)
    return sum(((v - mu) / sigma) ** 3 for v in vals) / n


def _confidence(n: int) -> str:
    if n >= _MIN_OBS_HIGH:   return "high"
    if n >= _MIN_OBS_MEDIUM: return "medium"
    return "low"


# ── Growth series extraction ───────────────────────────────────────────────────

def _extract_yoy_growth(
    statements: list,
    field_name: str,
) -> list[float]:
    """
    Compute year-over-year growth rates from an ordered (newest→oldest) list
    of annual income statements.  Observations are excluded when the base
    year value is ≤ 0 (avoids meaningless rates from negative denominators).
    """
    vals: list[tuple[str, float]] = []
    for stmt in statements:
        v = getattr(stmt, field_name, None)
        if v is not None and (stmt.period or "FY").startswith("FY"):
            vals.append((stmt.date, v))

    if len(vals) < 2:
        return []

    # Oldest → newest for growth computation
    vals_sorted = sorted(vals, key=lambda x: x[0])
    rates: list[float] = []
    for i in range(1, len(vals_sorted)):
        base = vals_sorted[i - 1][1]
        curr = vals_sorted[i][1]
        if base > 0:
            rates.append((curr - base) / base)
    return rates


# ── Growth calibration ─────────────────────────────────────────────────────────

def calibrate_growth(
    statements:        list,
    peer_growth_rates: Optional[list[float]] = None,
) -> Optional[GrowthCalibration]:
    """
    Fit growth distribution parameters from historical annual income statements.

    Priority: EPS → revenue (fallback when EPS series is insufficient).
    Returns None when fewer than _MIN_OBS_HARD valid observations exist
    across both sources.
    """
    # Try EPS first (most relevant for P/E MC)
    eps_rates = _extract_yoy_growth(statements, "eps_diluted")
    if len(eps_rates) < _MIN_OBS_HARD:
        eps_rates = _extract_yoy_growth(statements, "eps")
    source = "eps"

    rates = eps_rates
    if len(rates) < _MIN_OBS_HARD:
        rates  = _extract_yoy_growth(statements, "revenue")
        source = "revenue"
    if len(rates) < _MIN_OBS_HARD:
        rates  = _extract_yoy_growth(statements, "net_income")
        source = "net_income"
    if len(rates) < _MIN_OBS_HARD:
        return None

    # Peer shrinkage toward median peer growth
    if peer_growth_rates and len(peer_growth_rates) >= 2:
        peer_med = _median(peer_growth_rates)
        rates = [r * (1 - _PEER_WEIGHT) + peer_med * _PEER_WEIGHT for r in rates]

    sorted_rates = sorted(rates)
    n   = len(rates)
    mu  = _mean(rates)
    sig = _std(rates, mu)

    # SplitNormal: separate above- and below-mean observations
    below = [r for r in rates if r < mu]
    above = [r for r in rates if r >= mu]
    sigma_down = (_std(below, mu) if len(below) >= 2 else sig * 0.9)
    sigma_up   = (_std(above, mu) if len(above) >= 2 else sig * 1.1)
    sigma_down = max(sigma_down, 0.005)
    sigma_up   = max(sigma_up,   0.005)

    # Shock probability: empirical fraction below left-tail threshold
    threshold  = mu - 1.5 * sigma_down
    shock_prob = sum(1 for r in rates if r < threshold) / n
    shock_prob = max(shock_prob, 0.05)   # floor — at least 5% shock risk

    sk = _skewness(rates, mu, sig)

    return GrowthCalibration(
        n_obs      = n,
        mean       = mu,
        median     = _median(rates),
        std        = sig,
        skewness   = sk,
        p5         = _percentile(sorted_rates, 0.05),
        p95        = _percentile(sorted_rates, 0.95),
        sigma_down = sigma_down,
        sigma_up   = sigma_up,
        shock_prob = shock_prob,
        source     = source,
        confidence = _confidence(n),
    )


# ── Multiple series extraction ─────────────────────────────────────────────────

def _extract_multiple_series(
    ratios_list: list,
    field_name:  str,
) -> list[float]:
    """Extract a valid (positive) historical multiple series from a ratios list."""
    vals: list[tuple[str, float]] = []
    for r in ratios_list:
        v = getattr(r, field_name, None)
        if v is not None and v > 0 and (r.period or "FY").startswith("FY"):
            vals.append((r.date, v))
    return [v for _, v in sorted(vals, key=lambda x: x[0])]


# ── AR(1) mean-reversion speed ─────────────────────────────────────────────────

def _ols_ar1_speed(series: list[float]) -> float:
    """
    Estimate mean-reversion speed from an AR(1) process via OLS.

    Model: Δx_t = α × (x_{t-1} − μ) + ε_t
    The mean-reversion speed is −α (should be positive for mean-reversion).
    Half-life = ln(2) / speed.
    Returns speed clamped to [0.05, 0.70].
    """
    if len(series) < 3:
        return 0.30   # fallback to moderate reversion
    mu     = _mean(series)
    diffs  = [series[i] - series[i - 1] for i in range(1, len(series))]
    lagged = [series[i - 1] - mu         for i in range(1, len(series))]
    n  = len(diffs)
    cov = sum(diffs[i] * lagged[i] for i in range(n)) / n
    var = sum(lagged[i] ** 2       for i in range(n)) / n
    if var <= 0:
        return 0.30
    alpha = cov / var   # should be negative for mean-reverting series
    speed = max(0.05, min(0.70, -alpha))
    return speed


# ── Beta concentration from empirical spread ───────────────────────────────────

def _calibrate_concentration(
    hist_mean:  float,
    hist_std:   float,
    low_bound:  float,
    high_bound: float,
) -> float:
    """
    Solve Beta concentration c = α + β from empirical variance.

    Beta variance: Var = μ_n × (1 − μ_n) / (c + 1)
    → c = μ_n × (1 − μ_n) / Var_n − 1

    where μ_n and Var_n are normalised to [low_bound, high_bound].
    Returns c clamped to [2, 12].
    """
    span = high_bound - low_bound
    if span <= 0 or hist_std <= 0:
        return 4.0   # fallback

    mu_n  = max(0.05, min(0.95, (hist_mean - low_bound) / span))
    var_n = (hist_std / span) ** 2
    if var_n <= 0:
        return 4.0

    c = mu_n * (1.0 - mu_n) / var_n - 1.0
    return max(2.0, min(12.0, c))


# ── Multiple calibration ───────────────────────────────────────────────────────

def calibrate_multiple(
    ratios_list:          list,
    multiple_bear:        float,
    multiple_bull:        float,
    method:               str   = "P/E",
    peer_multiples:       Optional[list[float]] = None,
) -> Optional[MultipleCalibration]:
    """
    Fit multiple distribution parameters from historical ratio data.

    Returns None when fewer than _MIN_OBS_HARD valid observations exist.
    """
    field_map = {"P/E": "pe_ratio", "P/S": "ps_ratio", "EV/EBITDA": "ev_to_ebitda"}
    field = field_map.get(method, "pe_ratio")
    series = _extract_multiple_series(ratios_list, field)

    if len(series) < _MIN_OBS_HARD:
        return None

    # Peer shrinkage toward peer median
    if peer_multiples and len(peer_multiples) >= 2:
        peer_med = _median(peer_multiples)
        series = [v * (1 - _PEER_WEIGHT) + peer_med * _PEER_WEIGHT for v in series]

    sorted_s = sorted(series)
    n    = len(series)
    mu   = _mean(series)
    sig  = _std(series, mu)
    p10  = _percentile(sorted_s, 0.10)
    p90  = _percentile(sorted_s, 0.90)

    # Clamp p10/p90 within supplied bear/bull bounds
    p10 = max(p10, multiple_bear * 0.80)
    p90 = min(p90, multiple_bull * 1.20)

    mr_speed   = _ols_ar1_speed(series)
    halflife   = math.log(2) / mr_speed if mr_speed > 0 else 99.0
    conc       = _calibrate_concentration(mu, sig, multiple_bear, multiple_bull)

    return MultipleCalibration(
        n_obs         = n,
        mean          = mu,
        std           = sig,
        p10           = p10,
        p90           = p90,
        mr_speed      = mr_speed,
        mr_halflife   = halflife,
        concentration = conc,
        source        = field,
        confidence    = _confidence(n),
    )


# ── Parameter assembly ─────────────────────────────────────────────────────────

def _assemble_growth_params(
    gc:            GrowthCalibration,
    macro_regime:  str,
    quality_tier:  str,
) -> "GrowthDistParams":
    """
    Convert a GrowthCalibration into a GrowthDistParams, applying only
    regime mean-shift and scale (not quality tier — that's already in the
    empirical sigma_down/sigma_up).
    """
    from analysis.monte_carlo import GrowthDistParams, _REGIME_TABLE, _regime_key

    rk = _regime_key(macro_regime)
    p_shock_add, mu_adj, sigma_scale, _ = _REGIME_TABLE.get(
        rk, (0.05, 0.00, 1.00, 0.00)
    )

    adj_mean  = gc.mean + mu_adj
    shock_std = max(gc.sigma_down * 0.60 * sigma_scale, 0.005)

    return GrowthDistParams(
        growth_mean  = adj_mean,
        sigma_down   = max(gc.sigma_down * sigma_scale, 0.005),
        sigma_up     = max(gc.sigma_up   * sigma_scale, 0.005),
        shock_prob   = min(gc.shock_prob + p_shock_add, 0.50),
        shock_mean   = gc.mean - 2.0 * gc.std - abs(mu_adj) * 2.0,
        shock_std    = shock_std,
        quality_tier = quality_tier,
        macro_regime = macro_regime,
    )


def _assemble_multiple_params(
    mc:            MultipleCalibration,
    gc:            GrowthCalibration,
    multiple_bear: float,
    multiple_bull: float,
    multiple_base: float,
    macro_regime:  str,
    quality_tier:  str,
    method:        str,
) -> "MultipleDistParams":
    """Convert a MultipleCalibration into a MultipleDistParams."""
    from analysis.monte_carlo import (
        MultipleDistParams, _REGIME_RATE_ADJ, _CORR_TABLE, _regime_key
    )

    rk       = _regime_key(macro_regime)
    rate_adj = _REGIME_RATE_ADJ.get(rk, 0.00)

    rho_pe, rho_ps, sensitivity = _CORR_TABLE.get(quality_tier, (0.40, 0.28, 0.07))
    rho = rho_pe if method == "P/E" else rho_ps

    return MultipleDistParams(
        low              = multiple_bear,
        high             = multiple_bull,
        current          = multiple_base,
        fair             = mc.mean,           # empirical long-run mean IS the fair anchor
        mr_speed         = mc.mr_speed,
        concentration    = mc.concentration,
        rate_adj         = rate_adj,
        correlation_rho  = rho,
        corr_sensitivity = sensitivity,
        quality_tier     = quality_tier,
        macro_regime     = macro_regime,
    )


# ── Public entry point ─────────────────────────────────────────────────────────

def calibrate_mc_params(
    stock_data:          "StockData",
    multiple_bear:       float,
    multiple_base:       float,
    multiple_bull:       float,
    method:              str             = "P/E",
    macro_regime:        str             = "Unknown",
    gross_margin:        Optional[float] = None,
    op_margin:           Optional[float] = None,
    peer_growth_rates:   Optional[list[float]] = None,
    peer_multiples:      Optional[list[float]] = None,
) -> Optional[MCCalibrationResult]:
    """
    Derive fully-calibrated GrowthDistParams and MultipleDistParams from
    a StockData object.

    Parameters
    ----------
    stock_data          : StockData with income_statements and ratios populated
    multiple_bear/base/bull : scenario multiple bounds (from ValuationRange)
    method              : "P/E" | "P/S" — determines which multiple series to use
    macro_regime        : current macro regime string (adjusts location/scale)
    gross_margin        : for quality tier inference
    op_margin           : for quality tier inference
    peer_growth_rates   : list of peer median annual growth rates (decimal)
    peer_multiples      : list of peer current multiples for shrinkage anchor

    Returns None when historical data is insufficient for both growth
    and multiple calibration.
    """
    from analysis.monte_carlo import _infer_quality_tier

    qt = _infer_quality_tier(gross_margin, op_margin)

    gc = calibrate_growth(
        stock_data.income_statements,
        peer_growth_rates = peer_growth_rates,
    )
    mc = calibrate_multiple(
        stock_data.ratios,
        multiple_bear   = multiple_bear,
        multiple_bull   = multiple_bull,
        method          = method,
        peer_multiples  = peer_multiples,
    )

    if gc is None and mc is None:
        return None

    notes: list[str] = []

    # Fall back to table-based defaults when calibration is insufficient
    if gc is None:
        from analysis.monte_carlo import build_growth_params
        gp = build_growth_params(0.10, 0.06, macro_regime, gross_margin, op_margin)
        gc = GrowthCalibration(
            n_obs=0, mean=0.10, median=0.10, std=0.06, skewness=0.0,
            p5=-0.08, p95=0.28, sigma_down=0.05, sigma_up=0.07,
            shock_prob=0.15, source="default", confidence="low",
        )
        notes.append("Growth: insufficient history — using regime/quality table defaults")
    else:
        gp = _assemble_growth_params(gc, macro_regime, qt)
        if gc.confidence == "low":
            notes.append(f"Growth: low confidence — only {gc.n_obs} observations")

    if mc is None:
        from analysis.monte_carlo import build_multiple_params
        mp = build_multiple_params(
            multiple_bear, multiple_base, multiple_bull,
            gc.mean, qt, macro_regime, method,
        )
        mc_dummy = MultipleCalibration(
            n_obs=0, mean=multiple_base, std=0.0, p10=multiple_bear,
            p90=multiple_bull, mr_speed=0.30, mr_halflife=2.3,
            concentration=4.0, source="default", confidence="low",
        )
        mc = mc_dummy
        notes.append("Multiple: insufficient history — using regime/quality table defaults")
    else:
        mp = _assemble_multiple_params(
            mc, gc, multiple_bear, multiple_bull, multiple_base,
            macro_regime, qt, method,
        )
        if mc.confidence == "low":
            notes.append(f"Multiple: low confidence — only {mc.n_obs} observations")

    if peer_growth_rates:
        notes.append(f"Growth anchored to peer median with {_PEER_WEIGHT:.0%} weight")
    if peer_multiples:
        notes.append(f"Multiple anchored to peer median with {_PEER_WEIGHT:.0%} weight")

    return MCCalibrationResult(
        growth_calib    = gc,
        multiple_calib  = mc,
        growth_params   = gp,
        multiple_params = mp,
        notes           = notes,
    )
