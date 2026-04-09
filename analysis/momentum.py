"""
Momentum and technical scoring module.

Computes price-based indicators from raw OHLCV data:
  - 6-month and 12-month returns
  - 50-day / 200-day SMA crossover
  - RSI (14-day)
  - MACD signal

When NormalizedMetrics is supplied, it is available for future use (e.g.
resolved price for sanity checks).  Price-history-derived indicators always
come from raw OHLCV data because NormalizedMetrics does not carry them.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from models.scorecard import CategoryScore
from models.stock_data import PriceHistory, StockData
from utils.helpers import clamp

if TYPE_CHECKING:
    from analysis.metrics import NormalizedMetrics


# ── Technical indicator helpers ────────────────────────────────────────────────

def compute_sma(closes: list[float], period: int) -> Optional[float]:
    if len(closes) < period:
        return None
    return sum(closes[:period]) / period


def compute_ema(closes: list[float], period: int) -> Optional[float]:
    """EMA on list ordered newest -> oldest. Returns the current (newest) EMA."""
    if len(closes) < period:
        return None
    data = list(reversed(closes[:period * 3]))
    k    = 2.0 / (period + 1)
    ema  = sum(data[:period]) / period
    for price in data[period:]:
        ema = price * k + ema * (1 - k)
    return ema


def compute_rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """
    RSI-14. closes ordered newest -> oldest.
    Requires at least period + 1 data points.
    """
    if len(closes) < period + 1:
        return None
    prices = list(reversed(closes[:period * 3]))
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains  = [max(d, 0)      for d in deltas]
    losses = [abs(min(d, 0)) for d in deltas]

    avg_gain = sum(gains[:period])  / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i])  / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_macd(closes: list[float]) -> Optional[tuple[float, float]]:
    """Returns (MACD line, signal placeholder). MACD = EMA12 - EMA26."""
    ema12 = compute_ema(closes, 12)
    ema26 = compute_ema(closes, 26)
    if ema12 is None or ema26 is None:
        return None
    return ema12 - ema26, 0.0


# ── Score helpers ──────────────────────────────────────────────────────────────

def _return_score(ret: Optional[float], label: str) -> tuple[float, str]:
    if ret is None:
        return 50.0, f"{label}: N/A"
    pct = ret * 100
    if ret > 0.40:
        score = 95.0
    elif ret > 0.20:
        score = 85.0
    elif ret > 0.10:
        score = 75.0
    elif ret > 0.0:
        score = 62.0
    elif ret > -0.10:
        score = 45.0
    elif ret > -0.25:
        score = 30.0
    else:
        score = 15.0
    direction = "up" if ret > 0 else "down"
    return score, f"{label}: {pct:+.1f}% ({direction})"


def score_momentum(
    stock_data: StockData,
    weight: float = 0.10,
    metrics: "Optional[NormalizedMetrics]" = None,
) -> CategoryScore:
    """
    Compute a 0-100 momentum score from price history.
    100 = strong uptrend with healthy technical structure.
    """
    ph: Optional[PriceHistory] = stock_data.price_history

    if ph is None or len(ph) < 20:
        return CategoryScore(
            name="momentum",
            score=50.0,
            weight=weight,
            factors=["Insufficient price history for technical analysis"],
            reasoning="Cannot compute momentum — price data unavailable or too short.",
            data_quality="missing",
        )

    closes = ph.closes
    factors: list[str] = []
    sub_scores: list[tuple[float, float]] = []

    # ── 6-month return ─────────────────────────────────────────────────────────
    ret_6m = None
    if ph.price_6m_ago and ph.price_6m_ago > 0:
        ret_6m = (closes[0] - ph.price_6m_ago) / ph.price_6m_ago
    r6_s, r6_f = _return_score(ret_6m, "6M return")
    sub_scores.append((r6_s, 0.25))
    factors.append(r6_f)

    # ── 12-month return ────────────────────────────────────────────────────────
    ret_12m = None
    if ph.price_12m_ago and ph.price_12m_ago > 0:
        ret_12m = (closes[0] - ph.price_12m_ago) / ph.price_12m_ago
    r12_s, r12_f = _return_score(ret_12m, "12M return")
    sub_scores.append((r12_s, 0.25))
    factors.append(r12_f)

    # ── Price vs 50-day SMA ────────────────────────────────────────────────────
    sma50 = compute_sma(closes, 50)
    if sma50 and closes[0]:
        pct_diff = (closes[0] - sma50) / sma50 * 100
        if closes[0] > sma50:
            sub_scores.append((78.0, 0.15))
            factors.append(f"Price {pct_diff:+.1f}% above 50-day MA — bullish structure")
        else:
            sub_scores.append((32.0, 0.15))
            factors.append(f"Price {pct_diff:+.1f}% below 50-day MA — bearish structure")
    else:
        sub_scores.append((50.0, 0.15))
        factors.append("50-day MA: N/A")

    # ── Price vs 200-day SMA ───────────────────────────────────────────────────
    sma200 = compute_sma(closes, 200)
    if sma200 and closes[0]:
        pct_diff = (closes[0] - sma200) / sma200 * 100
        if closes[0] > sma200:
            sub_scores.append((80.0, 0.15))
            factors.append(f"Price {pct_diff:+.1f}% above 200-day MA — long-term uptrend")
        else:
            sub_scores.append((28.0, 0.15))
            factors.append(f"Price {pct_diff:+.1f}% below 200-day MA — long-term downtrend")
    else:
        sub_scores.append((50.0, 0.15))
        factors.append("200-day MA: N/A")

    # ── RSI ────────────────────────────────────────────────────────────────────
    rsi = compute_rsi(closes)
    if rsi is not None:
        if rsi < 25:
            rsi_s, rsi_note = 70.0, f"RSI {rsi:.1f} — oversold, potential reversal setup"
        elif rsi < 35:
            rsi_s, rsi_note = 65.0, f"RSI {rsi:.1f} — approaching oversold"
        elif rsi < 50:
            rsi_s, rsi_note = 52.0, f"RSI {rsi:.1f} — neutral to slightly weak"
        elif rsi < 65:
            rsi_s, rsi_note = 65.0, f"RSI {rsi:.1f} — bullish momentum"
        elif rsi < 75:
            rsi_s, rsi_note = 55.0, f"RSI {rsi:.1f} — extended, watch for consolidation"
        else:
            rsi_s, rsi_note = 38.0, f"RSI {rsi:.1f} — overbought territory"
        sub_scores.append((rsi_s, 0.10))
        factors.append(rsi_note)
    else:
        sub_scores.append((50.0, 0.10))

    # ── MACD ───────────────────────────────────────────────────────────────────
    macd_result = compute_macd(closes)
    if macd_result is not None:
        macd_line, _ = macd_result
        if macd_line > 0:
            sub_scores.append((72.0, 0.10))
            factors.append(f"MACD positive ({macd_line:.2f}) — bullish momentum")
        else:
            sub_scores.append((35.0, 0.10))
            factors.append(f"MACD negative ({macd_line:.2f}) — bearish momentum")
    else:
        sub_scores.append((50.0, 0.10))

    total_w   = sum(w for _, w in sub_scores)
    composite = sum(s * w for s, w in sub_scores) / total_w

    if composite >= 75:
        reasoning = "Strong upward momentum with bullish technical structure."
    elif composite >= 55:
        reasoning = "Mixed technicals — some positive signals but not a clear trend."
    elif composite >= 35:
        reasoning = "Momentum is weak or negative — price action lagging."
    else:
        reasoning = "Bearish technical picture with negative momentum."

    return CategoryScore(
        name="momentum",
        score=clamp(composite),
        weight=weight,
        factors=factors,
        reasoning=reasoning,
        data_quality="good",
    )
