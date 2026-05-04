"""
historical_regression.py
========================
Historical Regression Layer (HRL) for the institutional equity research platform.

Pure Python OLS/Ridge kernel using Gaussian elimination — no numpy/scipy.
Five regression models produce calibrated parameters that feed the Monte Carlo
simulation alongside the valuation-range fundamental path and factor-premium
regression calibration.

Models
------
  A  — AR(1) EPS persistence (quarterly or annual series)
  B  — Margin trend regression (linear trend in operating/net margins)
  C  — Valuation mean reversion (P/E series → OU kappa + long-run theta)
  D  — Macro sensitivity (12-month lagged return regression proxy)
  F  — Drawdown analysis (peak-to-trough risk from price history)

Three-way growth_mean blend (consumed by _apply_layer_overrides in monte_carlo.py)
  · MC fundamental path        40%  (ValuationRange scenarios)
  · Factor regression E[R]     30%  (RegressionCalibration.expected_return)
  · AR(1) HRL persistence      30%  (HRLResult.ar1_growth_estimate)
  Weights shift by per-source confidence.

Integration
-----------
    from analysis.historical_regression import run_historical_regression_layer
    hrl = run_historical_regression_layer(
        stock_data, factor_profile, macro_regime,
        mc_growth_mean=0.08, reg_calib=regression_calib
    )
    # hrl.calibrated_growth_mean → three-way blend for MC
    # hrl.valuation_mr_speed     → mean-reversion speed for multiple distribution
    # hrl.margin_trend_slope     → sigma asymmetry (neg slope → fatter left tail)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from models.stock_data import StockData
    from analysis.factor_model import FactorProfile


# ─────────────────────────────────────────────────────────────────────────────
# Pure Python linear algebra kernel
# ─────────────────────────────────────────────────────────────────────────────

def _matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Matrix multiply A × B."""
    m, k = len(A), len(A[0])
    n = len(B[0])
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k):
                s += A[i][p] * B[p][j]
            C[i][j] = s
    return C


def _T(A: List[List[float]]) -> List[List[float]]:
    """Transpose matrix A."""
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def _solve(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """
    Solve Ax = b via Gaussian elimination with partial pivoting.
    Returns None if singular or near-singular.
    """
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            return None
        M[col], M[pivot] = M[pivot], M[col]

        for row in range(col + 1, n):
            if abs(M[col][col]) < 1e-12:
                return None
            factor = M[row][col] / M[col][col]
            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(M[i][i]) < 1e-12:
            return None
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]

    return x


def ols(X: List[List[float]], y: List[float]) -> Optional[List[float]]:
    """OLS: β = (X'X)^{-1} X'y. Returns None if X'X is singular."""
    Xt    = _T(X)
    XtX   = _matmul(Xt, X)
    Xty_m = _matmul(Xt, [[yi] for yi in y])
    Xty   = [row[0] for row in Xty_m]
    return _solve(XtX, Xty)


def ridge(
    X: List[List[float]], y: List[float], lam: float
) -> Optional[List[float]]:
    """Ridge regression: β = (X'X + λI)^{-1} X'y."""
    k   = len(X[0])
    Xt  = _T(X)
    XtX = _matmul(Xt, X)
    for i in range(k):
        XtX[i][i] += lam
    Xty_m = _matmul(Xt, [[yi] for yi in y])
    Xty   = [row[0] for row in Xty_m]
    return _solve(XtX, Xty)


def select_ridge_lambda(
    X:          List[List[float]],
    y:          List[float],
    candidates: Tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0),
) -> float:
    """LOO cross-validation to pick ridge λ with minimum MSE."""
    n = len(y)
    if n < 4:
        return 1.0
    best_lam, best_mse = 1.0, float("inf")
    for lam in candidates:
        errs: List[float] = []
        for i in range(n):
            X_tr = [X[j] for j in range(n) if j != i]
            y_tr = [y[j] for j in range(n) if j != i]
            if len(X_tr) < len(X[0]):
                continue
            beta = ridge(X_tr, y_tr, lam)
            if beta is None:
                continue
            pred = sum(X[i][k] * beta[k] for k in range(len(beta)))
            errs.append((pred - y[i]) ** 2)
        if errs:
            mse = sum(errs) / len(errs)
            if mse < best_mse:
                best_mse, best_lam = mse, lam
    return best_lam


def _r_squared(y: List[float], y_pred: List[float]) -> float:
    """R² = 1 − SS_res / SS_tot."""
    n = len(y)
    if n < 2:
        return 0.0
    y_bar  = sum(y) / n
    ss_tot = sum((yi - y_bar) ** 2 for yi in y)
    ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y, y_pred))
    if ss_tot < 1e-12:
        return 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HRLResult:
    """
    Historical Regression Layer consolidated output.

    Consumed by _apply_layer_overrides() in monte_carlo.py:
      calibrated_growth_mean → replaces / blends MC growth_mean
      valuation_mr_speed     → overrides MultipleDistParams.mr_speed
      margin_trend_slope     → drives sigma asymmetry (neg = fatter left tail)
      max_drawdown_avg       → informs floor for shock scenarios
    """
    # Model A — AR(1) EPS persistence
    ar1_eps_persistence: float        # AR(1) coefficient (rho)
    ar1_growth_estimate: float        # implied 1-yr EPS growth (decimal)
    ar1_r2:              float
    ar1_n:               int

    # Model B — Margin trend (quarterly or annual)
    margin_trend_slope:  float        # slope of margin vs time (decimal/period)
    margin_trend_r2:     float
    margin_series_used:  str          # "operating" | "net" | "none"

    # Model C — Valuation mean reversion
    valuation_mr_speed:  float        # kappa: 0 = no MR, 1 = full MR in one period
    valuation_mr_target: Optional[float]   # historical fair-value multiple (theta)
    valuation_mr_r2:     float

    # Model D — Macro sensitivity proxy
    macro_sensitivity:    float       # beta to 12-mo lagged return
    macro_sensitivity_r2: float
    macro_n:              int

    # Model F — Drawdown
    max_drawdown_avg:    float        # average of top-N historical drawdowns
    max_drawdown_worst:  float        # worst single observed drawdown

    # Synthesised
    calibrated_growth_mean: float     # three-way blend for MC
    hrl_confidence:         str       # "high" | "medium" | "low"
    diagnostics:            Dict[str, str] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Model A — AR(1) EPS persistence
# ─────────────────────────────────────────────────────────────────────────────

def _ar1_regression(series: List[float]) -> Tuple[float, float, float, int]:
    """
    Fit AR(1): y_t = alpha + rho * y_{t-1} + eps_t (oldest → newest).
    Returns (rho, intercept, r2, n_obs).
    """
    if len(series) < 4:
        return 0.5, 0.0, 0.0, 0

    y_lag  = series[:-1]
    y_curr = series[1:]
    n      = len(y_curr)

    X    = [[1.0, y_lag[i]] for i in range(n)]
    beta = ols(X, y_curr)
    if beta is None:
        return 0.5, 0.0, 0.0, n

    intercept = beta[0]
    rho       = max(-0.97, min(0.97, beta[1]))  # bound to stationary region

    y_pred = [intercept + rho * y_lag[i] for i in range(n)]
    r2     = _r_squared(y_curr, y_pred)

    return rho, intercept, r2, n


def _extract_eps_series(stock_data: "StockData") -> List[float]:
    """
    Extract EPS series (oldest → newest) from quarterly or annual statements.
    """
    eps: List[float] = []

    qinc = getattr(stock_data, "quarterly_income", []) or []
    if qinc:
        for stmt in reversed(qinc):   # FMP: newest-first → reverse for time order
            v = getattr(stmt, "eps", None)
            if v is not None:
                eps.append(float(v))
        if len(eps) >= 4:
            return eps

    # Fall back to annual
    eps = []
    for stmt in reversed(getattr(stock_data, "income_statements", []) or []):
        v = getattr(stmt, "eps", None)
        if v is not None:
            eps.append(float(v))
    return eps


def _ar1_growth_estimate(
    rho: float, intercept: float, current_eps: float
) -> float:
    """
    Four-quarter-ahead AR(1) forecast → annualised EPS growth.
    E[EPS_{t+4}] = intercept*(1+rho+rho²+rho³) + rho^4 * EPS_t
    """
    if current_eps is None or current_eps == 0:
        return 0.0
    e1 = intercept + rho * current_eps
    e2 = intercept + rho * e1
    e3 = intercept + rho * e2
    e4 = intercept + rho * e3

    if current_eps < 0:
        if e4 > 0:
            return 0.10   # recovering from loss → moderate positive
        return max(-0.60, (e4 - current_eps) / abs(current_eps))

    growth = e4 / current_eps - 1.0
    return max(-0.60, min(1.0, growth))


# ─────────────────────────────────────────────────────────────────────────────
# Model B — Margin trend
# ─────────────────────────────────────────────────────────────────────────────

def _linear_trend(values: List[float]) -> Tuple[float, float, float]:
    """
    Fit y = a + b*t (t = 0, 1, 2, …). Returns (slope, intercept, r2).
    """
    n = len(values)
    if n < 3:
        return 0.0, 0.0, 0.0
    X    = [[1.0, float(i)] for i in range(n)]
    beta = ols(X, values)
    if beta is None:
        return 0.0, 0.0, 0.0
    a, b   = beta[0], beta[1]
    y_pred = [a + b * i for i in range(n)]
    return b, a, _r_squared(values, y_pred)


def _extract_margin_series(
    stock_data: "StockData",
) -> Tuple[List[float], str]:
    """
    Build operating-margin or net-margin time series (oldest → newest).
    Returns (series, label) where label ∈ {"operating", "net", "none"}.
    """
    def _build(stmts: list, newest_first: bool) -> Tuple[List[float], List[float]]:
        it = reversed(stmts) if newest_first else stmts
        ops, nets = [], []
        for stmt in it:
            rev = getattr(stmt, "revenue", None) or 0
            if rev <= 0:
                continue
            oi = getattr(stmt, "operating_income", None)
            ni = getattr(stmt, "net_income", None)
            if oi is not None:
                ops.append(oi / rev)
            if ni is not None:
                nets.append(ni / rev)
        return ops, nets

    qinc = getattr(stock_data, "quarterly_income", []) or []
    if qinc:
        ops, nets = _build(qinc, newest_first=True)
        if len(ops) >= 4:
            return ops, "operating"
        if len(nets) >= 4:
            return nets, "net"

    ainc = getattr(stock_data, "income_statements", []) or []
    if ainc:
        ops, nets = _build(ainc, newest_first=True)
        if len(ops) >= 3:
            return ops, "operating"
        if len(nets) >= 3:
            return nets, "net"

    return [], "none"


# ─────────────────────────────────────────────────────────────────────────────
# Model C — Valuation mean reversion
# ─────────────────────────────────────────────────────────────────────────────

def _mean_reversion_speed(
    pe_series: List[float],
) -> Tuple[float, Optional[float], float]:
    """
    Estimate OU mean-reversion speed (kappa) from historical P/E series.

    AR(1) on level: PE_t = alpha + rho * PE_{t-1} + eps
    kappa = 1 - rho   (rate of pull toward long-run mean)
    theta = alpha / kappa  (long-run mean)
    Returns (kappa, theta, r2).
    """
    if len(pe_series) < 5:
        return 0.15, None, 0.0

    # Light winsorization at 5th/95th percentile
    s = sorted(pe_series)
    lo = s[max(0, int(len(s) * 0.05))]
    hi = s[min(len(s) - 1, int(len(s) * 0.95))]
    pe_clean = [max(lo, min(hi, p)) for p in pe_series]

    rho, intercept, r2, _ = _ar1_regression(pe_clean)
    kappa = max(0.0, min(1.0, 1.0 - rho))
    if kappa > 0.01:
        theta = intercept / kappa
    else:
        theta = sum(pe_clean) / len(pe_clean)

    return kappa, theta, r2


def _extract_pe_series(stock_data: "StockData") -> List[float]:
    """
    Build historical P/E series (oldest → newest) by pairing annual EPS
    with the approximate year-end price from price_history.
    """
    ainc   = getattr(stock_data, "income_statements", []) or []
    ph     = getattr(stock_data, "price_history", None)
    closes = getattr(ph, "closes", []) if ph else []

    eps_vals: List[float] = []
    for stmt in ainc:     # newest-first from FMP
        eps = getattr(stmt, "eps", None)
        if eps is not None and eps > 0:
            eps_vals.append(float(eps))

    pe_series: List[float] = []
    for i, eps in enumerate(eps_vals):
        price_idx = i * 252   # approximately i years back
        if price_idx < len(closes) and closes[price_idx] > 0:
            pe = closes[price_idx] / eps
            if 3.0 < pe < 200.0:
                pe_series.append(pe)

    return list(reversed(pe_series))   # oldest → newest


# ─────────────────────────────────────────────────────────────────────────────
# Model D — Macro sensitivity (lagged momentum proxy)
# ─────────────────────────────────────────────────────────────────────────────

def _macro_sensitivity_regression(
    price_closes: List[float],
) -> Tuple[float, float, int]:
    """
    Estimate macro sensitivity as the stock's response to its own 12-month
    lagged return (a cyclicality / momentum-regime proxy).

      r_t = alpha + beta * lagged_12m_return_{t} + eps_t

    Returns (beta, r2, n_obs). price_closes is newest-first.
    """
    if len(price_closes) < 60:
        return 1.0, 0.0, 0

    step    = 21
    monthly = [price_closes[i] for i in range(0, len(price_closes), step)]
    monthly = list(reversed(monthly))   # oldest → newest

    if len(monthly) < 18:
        return 1.0, 0.0, 0

    monthly_rets: List[float] = []
    for i in range(1, len(monthly)):
        prev = monthly[i - 1]
        monthly_rets.append(monthly[i] / prev - 1.0 if prev > 0 else 0.0)

    if len(monthly_rets) < 15:
        return 1.0, 0.0, 0

    y: List[float]          = []
    X_feat: List[List[float]] = []
    for i in range(12, len(monthly_rets)):
        lag_ret = sum(monthly_rets[i - 12:i])
        y.append(monthly_rets[i])
        X_feat.append([1.0, lag_ret])

    if len(y) < 4:
        return 1.0, 0.0, 0

    beta_vec = ols(X_feat, y)
    if beta_vec is None:
        return 1.0, 0.0, len(y)

    beta   = max(-3.0, min(3.0, beta_vec[1]))
    y_pred = [beta_vec[0] + beta * X_feat[i][1] for i in range(len(y))]
    r2     = _r_squared(y, y_pred)

    return beta, r2, len(y)


# ─────────────────────────────────────────────────────────────────────────────
# Model F — Drawdown analysis
# ─────────────────────────────────────────────────────────────────────────────

def _drawdown_analysis(
    price_closes: List[float],
    n_drawdowns:  int = 3,
) -> Tuple[float, float]:
    """
    Compute average and worst historical peak-to-trough drawdowns.
    price_closes is newest-first. Returns (avg_top_n, worst) as positive decimals.
    """
    if len(price_closes) < 63:
        return 0.25, 0.35

    closes = list(reversed(price_closes))   # oldest → newest

    peak         = closes[0]
    current_dd   = 0.0
    drawdowns: List[float] = []

    for price in closes[1:]:
        if price > peak:
            if current_dd > 0.01:
                drawdowns.append(current_dd)
            peak       = price
            current_dd = 0.0
        else:
            dd         = (peak - price) / peak
            current_dd = max(current_dd, dd)

    if current_dd > 0.01:
        drawdowns.append(current_dd)

    if not drawdowns:
        return 0.15, 0.20

    drawdowns_sorted = sorted(drawdowns, reverse=True)
    worst  = drawdowns_sorted[0]
    top_n  = drawdowns_sorted[:n_drawdowns]
    avg    = sum(top_n) / len(top_n)

    return avg, worst


# ─────────────────────────────────────────────────────────────────────────────
# Three-way growth blend
# ─────────────────────────────────────────────────────────────────────────────

def _three_way_growth_blend(
    mc_growth:      Optional[float],
    factor_er:      Optional[float],
    ar1_growth:     Optional[float],
    ar1_r2:         float,
    reg_confidence: str,
) -> float:
    """
    Blend three growth signals with confidence-adjusted weights.

    Base: MC fundamental 40%, factor reg 30%, AR(1) 30%.
    Weight redistribution:
      · AR(1) R² < 0.10 or None → shift AR(1) weight to MC
      · reg confidence "low" or None → shift factor weight to MC
    """
    w_mc  = 0.40
    w_reg = 0.30
    w_ar1 = 0.30

    if ar1_r2 < 0.10 or ar1_growth is None:
        w_mc += w_ar1
        w_ar1 = 0.0

    if reg_confidence == "low" or factor_er is None:
        w_mc += w_reg
        w_reg  = 0.0

    # Translate factor E[R] → approximate growth (strip out ~1.5% yield)
    reg_growth = (
        max(-0.30, min(0.50, factor_er - 0.015))
        if factor_er is not None else None
    )

    numerator = 0.0
    active_w  = 0.0

    if mc_growth is not None and w_mc > 0:
        numerator += w_mc * mc_growth
        active_w  += w_mc
    elif w_mc > 0:
        # MC missing — redistribute to available sources
        extra = w_mc / max(1, (1 if reg_growth is not None else 0) + (1 if ar1_growth is not None else 0))
        if reg_growth is not None:
            w_reg += extra
        if ar1_growth is not None:
            w_ar1 += extra

    if reg_growth is not None and w_reg > 0:
        numerator += w_reg * reg_growth
        active_w  += w_reg

    if ar1_growth is not None and w_ar1 > 0:
        numerator += w_ar1 * ar1_growth
        active_w  += w_ar1

    if active_w < 1e-6:
        return mc_growth if mc_growth is not None else 0.05

    return max(-0.30, min(0.80, numerator / active_w))


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_historical_regression_layer(
    stock_data:     "StockData",
    factor_profile: Optional["FactorProfile"] = None,
    macro_regime:   str                        = "Unknown",
    mc_growth_mean: Optional[float]            = None,
    reg_calib:      object                     = None,
) -> Optional[HRLResult]:
    """
    Run all HRL models and return a consolidated HRLResult.

    Parameters
    ----------
    stock_data     : StockData — source for all historical series
    factor_profile : FactorProfile (optional, for logging context)
    macro_regime   : current macro regime string
    mc_growth_mean : growth_mean from ValuationRange (decimal, e.g. 0.08)
    reg_calib      : RegressionCalibration (optional, for three-way blend)

    Returns None only when all data sources are empty and mc_growth_mean is None.
    """
    diag: Dict[str, str] = {}

    # ── Model A: AR(1) EPS persistence ───────────────────────────────────────
    eps_series = _extract_eps_series(stock_data)
    if len(eps_series) >= 4:
        rho, intercept, ar1_r2, ar1_n = _ar1_regression(eps_series)
        current_eps = eps_series[-1]
        ar1_growth  = _ar1_growth_estimate(rho, intercept, current_eps)
        diag["ar1"] = (
            f"rho={rho:.3f} r2={ar1_r2:.2f} n={ar1_n}"
            f" current_eps={current_eps:.2f} est_g={ar1_growth:+.1%}"
        )
    else:
        rho, intercept, ar1_r2, ar1_n = 0.5, 0.0, 0.0, 0
        ar1_growth = None
        diag["ar1"] = f"insufficient EPS data (n={len(eps_series)})"

    print(f"  [HRL:A] {diag['ar1']}")

    # ── Model B: Margin trend ─────────────────────────────────────────────────
    margin_series, margin_label = _extract_margin_series(stock_data)
    if margin_series:
        mslope, _, mr2 = _linear_trend(margin_series)
        diag["margin"] = f"slope={mslope:+.4f} r2={mr2:.2f} n={len(margin_series)} ({margin_label})"
    else:
        mslope, mr2 = 0.0, 0.0
        margin_label = "none"
        diag["margin"] = "no margin data"

    print(f"  [HRL:B] {diag['margin']}")

    # ── Model C: Valuation mean reversion ─────────────────────────────────────
    pe_series = _extract_pe_series(stock_data)
    if len(pe_series) >= 5:
        kappa, theta, mr_r2 = _mean_reversion_speed(pe_series)
        diag["mr"] = (
            f"kappa={kappa:.3f} theta={theta:.1f}" if theta is not None
            else f"kappa={kappa:.3f} theta=None"
        ) + f" r2={mr_r2:.2f}"
    else:
        kappa, theta, mr_r2 = 0.15, None, 0.0
        diag["mr"] = f"insufficient PE history (n={len(pe_series)})"

    print(f"  [HRL:C] {diag['mr']}")

    # ── Model D: Macro sensitivity ─────────────────────────────────────────────
    ph     = getattr(stock_data, "price_history", None)
    closes = getattr(ph, "closes", []) if ph else []

    if len(closes) >= 60:
        macro_beta, macro_r2, macro_n = _macro_sensitivity_regression(closes)
        diag["macro_sens"] = f"beta={macro_beta:.3f} r2={macro_r2:.2f} n={macro_n}"
    else:
        macro_beta, macro_r2, macro_n = 1.0, 0.0, 0
        diag["macro_sens"] = f"insufficient price history (n={len(closes)})"

    print(f"  [HRL:D] {diag['macro_sens']}")

    # ── Model F: Drawdown ─────────────────────────────────────────────────────
    if len(closes) >= 63:
        dd_avg, dd_worst = _drawdown_analysis(closes)
        diag["drawdown"] = f"avg_top3={dd_avg:.1%} worst={dd_worst:.1%}"
    else:
        dd_avg, dd_worst = 0.25, 0.35
        diag["drawdown"] = "using defaults (insufficient price history)"

    print(f"  [HRL:F] {diag['drawdown']}")

    # ── Factor expected return ────────────────────────────────────────────────
    factor_er = None
    reg_conf  = "low"
    if reg_calib is not None:
        factor_er = getattr(reg_calib, "expected_return", None)
        reg_conf  = getattr(reg_calib, "confidence",      "low")

    # ── Three-way growth blend ────────────────────────────────────────────────
    calibrated_growth = _three_way_growth_blend(
        mc_growth      = mc_growth_mean,
        factor_er      = factor_er,
        ar1_growth     = ar1_growth,
        ar1_r2         = ar1_r2,
        reg_confidence = reg_conf,
    )

    diag["blend"] = (
        (f"mc={mc_growth_mean:+.1%}" if mc_growth_mean is not None else "mc=None")
        + (f" reg={factor_er:+.1%}" if factor_er is not None else " reg=None")
        + (f" ar1={ar1_growth:+.1%}" if ar1_growth is not None else " ar1=None")
        + f" → blend={calibrated_growth:+.1%}"
    )
    print(f"  [HRL:blend] {diag['blend']}")

    # ── Overall HRL confidence ────────────────────────────────────────────────
    has_eps    = ar1_n >= 6
    has_margin = len(margin_series) >= 6
    has_price  = len(closes) >= 252

    if has_eps and has_margin and has_price:
        hrl_conf = "high"
    elif has_eps or (has_margin and has_price):
        hrl_conf = "medium"
    else:
        hrl_conf = "low"

    # Return None only when there is truly nothing to offer
    if ar1_n == 0 and len(closes) < 30 and mc_growth_mean is None:
        return None

    return HRLResult(
        ar1_eps_persistence    = rho,
        ar1_growth_estimate    = ar1_growth if ar1_growth is not None else 0.0,
        ar1_r2                 = ar1_r2,
        ar1_n                  = ar1_n,
        margin_trend_slope     = mslope,
        margin_trend_r2        = mr2,
        margin_series_used     = margin_label,
        valuation_mr_speed     = kappa,
        valuation_mr_target    = theta,
        valuation_mr_r2        = mr_r2,
        macro_sensitivity      = macro_beta,
        macro_sensitivity_r2   = macro_r2,
        macro_n                = macro_n,
        max_drawdown_avg       = dd_avg,
        max_drawdown_worst     = dd_worst,
        calibrated_growth_mean = calibrated_growth,
        hrl_confidence         = hrl_conf,
        diagnostics            = diag,
    )
