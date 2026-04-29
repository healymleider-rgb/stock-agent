"""
regression_calibration.py
=========================
Factor-premium expected return estimation for the alpha decision engine.

Architecture
------------
Rather than a full panel regression (which would require historical factor
exposures for every peer at every period — data we don't reliably have),
this module uses a regime-aware factor premium table calibrated to academic
and practitioner estimates.

Expected return = market_premium × beta
               + Σ (factor_premium_i × factor_z_i)
               + idiosyncratic_alpha

Tracking error is estimated from the stock's own realised annual return
history:
    TE ≈ √(realised_vol² − (beta × market_vol)²)

This is the component of volatility not explained by market exposure —
the residual standard deviation that feeds GrowthDistParams.sigma calibration.

Factor premia (annualised, long-run academic estimates)
-------------------------------------------------------
Adjusted by macro regime:
  Expansion   — quality, momentum, growth premia elevated
  Contraction — quality premium stays high; momentum and growth turn negative
  Late_Cycle  — quality elevated, momentum fades, growth dims

Integration
-----------
    from analysis.regression_calibration import calibrate_regression
    from analysis.factor_model import build_factor_profile

    fp    = build_factor_profile(stock_data, peer_rows)
    calib = calibrate_regression(fp, stock_data, macro_regime="Expansion")

    if calib:
        # Use calib.expected_return to override growth_mean in MC
        # Use calib.tracking_error to calibrate sigma
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.stock_data import StockData
    from analysis.factor_model import FactorProfile


# ── Factor premium tables ─────────────────────────────────────────────────────
# All values are annualised decimal returns.
# Source: academic long-run estimates (Fama-French, AQR research) adjusted
# for current macro regime.  These are PREMIA (excess over market), not
# total returns.

_MARKET_PREMIUM:  float = 0.055   # long-run equity risk premium
_MARKET_VOL:      float = 0.160   # long-run annual market volatility

_FACTOR_PREMIA: dict[str, dict[str, float]] = {
    "expansion": {
        "quality":       0.042,   # quality premium elevated in risk-on environment
        "value":         0.028,   # value rerating in growth phase
        "momentum":      0.035,   # trend-following works in directional markets
        "growth":        0.022,   # growth rewarded in expanding economy
        "stability":     0.012,   # low-vol premium compresses in risk-on
        "profitability": 0.018,   # Novy-Marx gross profitability — modest in risk-on
        "macro_defense": 0.004,   # defensive premium near-zero when beta rewarded
    },
    "recovery": {
        "quality":       0.038,
        "value":         0.040,   # value outperforms most in recoveries
        "momentum":      0.030,
        "growth":        0.025,
        "stability":     0.010,
        "profitability": 0.020,
        "macro_defense": 0.006,
    },
    "early_cycle": {
        "quality":       0.035,
        "value":         0.035,
        "momentum":      0.032,
        "growth":        0.028,
        "stability":     0.010,
        "profitability": 0.022,
        "macro_defense": 0.005,
    },
    "mid_cycle": {
        "quality":       0.040,
        "value":         0.022,
        "momentum":      0.025,
        "growth":        0.018,
        "stability":     0.015,
        "profitability": 0.024,
        "macro_defense": 0.008,
    },
    "late_cycle": {
        "quality":       0.050,   # quality premium spikes as selectivity rises
        "value":         0.018,
        "momentum":      0.012,   # momentum fades near turns
        "growth":        0.008,   # growth stocks de-rate
        "stability":     0.025,   # defensive premium rises
        "profitability": 0.028,   # margin leaders favoured as cycle matures
        "macro_defense": 0.015,   # rotation into defensives begins
    },
    "slowdown": {
        "quality":       0.055,
        "value":         0.010,
        "momentum":      -0.005,  # momentum reversal risk in slowdowns
        "growth":        -0.008,
        "stability":     0.030,
        "profitability": 0.030,   # high-GM names weather demand softness better
        "macro_defense": 0.022,   # flight-to-defensives accelerates
    },
    "contraction": {
        "quality":       0.062,   # quality flight-to-safety
        "value":         0.005,
        "momentum":      -0.015,  # trend breaks down
        "growth":        -0.018,
        "stability":     0.038,
        "profitability": 0.032,   # profitable businesses survive; unprofitable de-rate hard
        "macro_defense": 0.030,   # staples/utilities outperform structurally
    },
    "recession": {
        "quality":       0.068,
        "value":         -0.005,  # value traps in recessions
        "momentum":      -0.025,
        "growth":        -0.025,
        "stability":     0.048,
        "profitability": 0.035,   # highest premium: only profitable names survive cuts
        "macro_defense": 0.038,   # maximum defensive tilt — staples/healthcare dominant
    },
    "unknown": {             # neutral / no macro view
        "quality":       0.045,
        "value":         0.022,
        "momentum":      0.020,
        "growth":        0.015,
        "stability":     0.018,
        "profitability": 0.022,
        "macro_defense": 0.012,
    },
}

# Idiosyncratic alpha per unit of quality z-score (50bp per σ of quality)
_ALPHA_PER_QUALITY_Z: float = 0.005


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RegressionCalibration:
    """
    Factor-decomposed expected return and risk estimate for a single stock.

    expected_return  — total annualised expected return (decimal, e.g. 0.12)
    alpha            — idiosyncratic expected alpha beyond market + factors
    factor_betas     — regime-specific factor premia × z-scores  (contribution)
    r_squared        — fraction of variance explained by market factor alone
    tracking_error   — annualised residual std after market attribution
    n_obs            — number of annual return periods available
    confidence       — "high" (≥5 years), "medium" (≥3), "low" (<3)
    method           — always "factor_premium" in this implementation
    """
    alpha:           float
    factor_betas:    dict    # {factor_name: annualised_contribution}
    expected_return: float
    r_squared:       float
    tracking_error:  float
    n_obs:           int
    confidence:      str
    method:          str = "factor_premium"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _regime_key(regime: str) -> str:
    """Normalise free-form macro regime to _FACTOR_PREMIA key."""
    r = regime.lower().replace(" ", "_").replace("-", "_")
    for k in _FACTOR_PREMIA:
        if k in r or r in k:
            return k
    return "unknown"


def _annual_returns_from_history(closes: list, max_years: int = 7) -> list:
    """
    Compute annual returns from a price history (newest → oldest).

    Takes one return per year: r_t = closes[0] / closes[252] − 1,
    then rolls 252 bars at a time.  Returns up to max_years values.
    """
    results = []
    for i in range(0, min(len(closes) - 252, max_years * 252), 252):
        prev = closes[i + 252]
        if prev and prev > 0:
            results.append(closes[i] / prev - 1.0)
    return results


# ── Public entry point ────────────────────────────────────────────────────────

def calibrate_regression(
    factor_profile: "FactorProfile",
    stock_data:     "StockData",
    macro_regime:   str = "Unknown",
) -> Optional[RegressionCalibration]:
    """
    Estimate expected return and tracking error using a factor premium model.

    Parameters
    ----------
    factor_profile : FactorProfile from build_factor_profile()
    stock_data     : StockData for historical vol calibration
    macro_regime   : current macro regime string (matched against _FACTOR_PREMIA)

    Returns None only when the FactorProfile has no valid z-scores
    (all zeros with low confidence and no price history available).
    """
    rk     = _regime_key(macro_regime)
    premia = _FACTOR_PREMIA[rk]

    beta = getattr(
        getattr(stock_data, "profile", None), "beta", None
    ) or 1.0
    beta = max(0.2, min(3.0, beta))   # clamp to sensible range

    # ── Market component ──────────────────────────────────────────────────────
    market_contribution = _MARKET_PREMIUM * beta

    # ── Factor contributions ──────────────────────────────────────────────────
    factor_contributions = {
        "quality":       premia["quality"]                       * factor_profile.quality_z,
        "value":         premia["value"]                         * factor_profile.value_z,
        "momentum":      premia["momentum"]                      * factor_profile.momentum_z,
        "growth":        premia["growth"]                        * factor_profile.growth_z,
        "stability":     premia["stability"]                     * factor_profile.stability_z,
        "profitability": premia.get("profitability", 0.022)      * factor_profile.profitability_z,
        "macro_defense": premia.get("macro_defense",  0.012)     * factor_profile.macro_z,
    }

    # ── Idiosyncratic alpha (quality-driven structural edge) ──────────────────
    alpha = _ALPHA_PER_QUALITY_Z * factor_profile.quality_z

    expected_return = (
        market_contribution
        + sum(factor_contributions.values())
        + alpha
    )

    # ── Historical vol → tracking error ──────────────────────────────────────
    ph = getattr(stock_data, "price_history", None)
    tracking_error = 0.18   # default: broad-market TE for unknown history
    r_squared      = 0.25   # default: market explains ~25% of individual stock var
    n_obs          = 0

    if ph and getattr(ph, "closes", None) and len(ph.closes) >= 253:
        annual_rets = _annual_returns_from_history(ph.closes)
        n_obs = len(annual_rets)

        if n_obs >= 2:
            mu_r    = sum(annual_rets) / n_obs
            var_r   = sum((r - mu_r) ** 2 for r in annual_rets) / (n_obs - 1)
            vol_r   = math.sqrt(var_r)

            # Systematic variance = (β × σ_market)²
            systematic_var = (beta * _MARKET_VOL) ** 2
            residual_var   = max(0.0, var_r - systematic_var)
            tracking_error = math.sqrt(residual_var)

            # R² — fraction of variance from market alone
            r_squared = min(0.80, systematic_var / max(var_r, 1e-8))

    confidence = (
        "high"   if n_obs >= 5 else
        "medium" if n_obs >= 3 else
        "low"
    )

    print(
        f"  [REG] regime={rk} beta={beta:.2f}"
        f" E[R]={expected_return:+.1%} alpha={alpha:+.1%}"
        f" TE={tracking_error:.1%} R²={r_squared:.2f}"
        f" n_obs={n_obs} conf={confidence}"
    )

    return RegressionCalibration(
        alpha            = alpha,
        factor_betas     = factor_contributions,
        expected_return  = expected_return,
        r_squared        = r_squared,
        tracking_error   = tracking_error,
        n_obs            = n_obs,
        confidence       = confidence,
    )
