"""
Utility functions shared across the codebase.
"""
from __future__ import annotations

import math
from typing import Any, Optional


def safe_get(data: dict, *keys, default: Any = None) -> Any:
    """Safely traverse nested dicts. Returns default on any missing key."""
    cursor = data
    for key in keys:
        if not isinstance(cursor, dict):
            return default
        cursor = cursor.get(key, default)
        if cursor is None:
            return default
    return cursor


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert value to float, returning default on failure or non-finite result."""
    if value is None:
        return default
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def safe_divide(
    numerator: Any, denominator: Any, default: Optional[float] = None
) -> Optional[float]:
    """Divide two numbers, returning default on zero denominator or None inputs."""
    n = safe_float(numerator)
    d = safe_float(denominator)
    if n is None or d is None or d == 0:
        return default
    return n / d


def pct_change(new_val: Any, old_val: Any) -> Optional[float]:
    """Compute percentage change from old_val to new_val as a ratio (0.10 = 10%)."""
    n = safe_float(new_val)
    o = safe_float(old_val)
    if n is None or o is None or o == 0:
        return None
    return (n - o) / abs(o)


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp a float to [lo, hi]."""
    return max(lo, min(hi, value))


def format_large_number(value: Optional[float]) -> str:
    """Format large numbers with T / B / M / K suffixes."""
    if value is None:
        return "N/A"
    abs_val = abs(value)
    sign    = "-" if value < 0 else ""
    if abs_val >= 1e12:
        return f"{sign}{abs_val / 1e12:.2f}T"
    if abs_val >= 1e9:
        return f"{sign}{abs_val / 1e9:.2f}B"
    if abs_val >= 1e6:
        return f"{sign}{abs_val / 1e6:.2f}M"
    if abs_val >= 1e3:
        return f"{sign}{abs_val / 1e3:.2f}K"
    return f"{sign}{abs_val:.2f}"


def format_pct(value: Optional[float], decimals: int = 1) -> str:
    """Format a ratio (0.15) as a percentage string ('15.0%')."""
    if value is None:
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def score_from_thresholds(value: Optional[float], thresholds: list[tuple]) -> float:
    """
    Map a value to a score using an ordered list of (threshold, score) pairs.
    Thresholds must be ordered from highest to lowest.
    Returns the score for the first threshold the value meets or exceeds.

    Example:
        thresholds = [(0.20, 90), (0.10, 75), (0.05, 60), (0.0, 45)]
        score_from_thresholds(0.12, thresholds) -> 75
    """
    if value is None:
        return 50.0
    for threshold, score in thresholds:
        if value >= threshold:
            return float(score)
    return float(thresholds[-1][1]) if thresholds else 50.0
