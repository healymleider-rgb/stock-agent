"""
FREDProvider — macroeconomic leading indicator data via the FRED API.

Scope
-----
This provider fetches macroeconomic time-series from the Federal Reserve Bank
of St. Louis (FRED).  It is NOT a DataProvider subclass — it has no concept of
a stock ticker.  Future agents (e.g. MacroLEIAgent) instantiate it directly.

Activation
----------
Set FRED_API_KEY in your .env file.
Without this key, is_available() returns False and every method returns None/[].

Free API key: https://fred.stlouisfed.org/docs/api/api_key.html
Generous rate limit: 120 requests per 60 seconds (as of 2024).

FRED API reference: https://fred.stlouisfed.org/docs/api/fred/
Base URL: https://api.stlouisfed.org/fred
Auth: ?api_key=<key>&file_type=json query parameters

Supported indicators
-----------------------------------
Series IDs are FRED's canonical identifiers.  Pass them directly to the generic
methods, or use the named constants and convenience methods below.

  Indicator                         FRED Series ID       Frequency
  ─────────────────────────────────────────────────────────────────
  10-Year Treasury yield            GS10                 Monthly
  2-Year Treasury yield             GS2                  Monthly
  Yield curve spread (10Y-2Y)       T10Y2Y               Daily (FRED-computed)
  OECD Composite Leading Indicator  USALOLITONOSTSAM     Monthly  [typically ~2 months stale]
  Industrial Production: Mfg        IPMAN                Monthly  (index 2017=100)
  Housing starts (total, SAAR)      HOUST                Monthly
  Initial jobless claims            ICSA                 Weekly
  Retail sales (total, NSA)         RSAFS                Monthly
  University of Michigan sentiment  UMCSENT              Monthly

Notes
-----
- USALOLITONOSTSAM (OECD CLI) is freely redistributable via FRED; may run
  ~2 months behind real-time.  The staleness gate in macro_overlay.py
  suppresses it if the observation date is older than 120 days.
- USSLIND (Conference Board LEI) requires a redistribution agreement not
  available via the free FRED API and is NOT used.
- T10Y2Y is FRED's own daily spread series; it's preferred over computing
  GS10 - GS2 manually because FRED interpolates missing business days.
- All numeric values in FRED responses are strings; _safe_float() handles
  the sentinel value "." (missing) as well as normal parse failures.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional, Tuple

import requests

from config import Config
from utils.logger import logger


# ── FRED series ID constants ───────────────────────────────────────────────────

# Yield curve
SERIES_YIELD_10Y          = "GS10"          # 10-Year Treasury Constant Maturity Rate
SERIES_YIELD_2Y           = "GS2"           # 2-Year Treasury Constant Maturity Rate
SERIES_YIELD_SPREAD_10Y2Y = "T10Y2Y"        # 10-Year minus 2-Year (FRED-computed, daily)

# OECD Composite Leading Indicator
SERIES_OECD_CLI           = "USALOLITONOSTSAM"  # OECD CLI for US (monthly, centred on 100)

# Industrial Production: Manufacturing
SERIES_MFG_PROD           = "IPMAN"             # Industrial Production: Mfg (monthly, 2017=100)

# Housing (display only — not used in scoring weights)
SERIES_HOUSING_STARTS     = "HOUST"         # Housing starts: total (monthly, SAAR)

# Labour market
SERIES_JOBLESS_CLAIMS     = "ICSA"          # Initial claims, seasonally adjusted (weekly)

# Consumer spending
SERIES_RETAIL_SALES       = "RSAFS"         # Retail and food services sales (monthly)

# Consumer confidence
SERIES_CONSUMER_SENTIMENT = "UMCSENT"       # U of Michigan Consumer Sentiment (monthly)

_FRED_BASE = "https://api.stlouisfed.org/fred"
_REQUEST_TIMEOUT = 15  # seconds


def _safe_float(value: Any) -> Optional[float]:
    """
    Convert a FRED value string to float.
    FRED uses "." as the missing-data sentinel.
    Returns None on any conversion failure.
    """
    if value is None:
        return None
    s = str(value).strip()
    if s in (".", "", "N/A", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


# ── FREDProvider ──────────────────────────────────────────────────────────────

class FREDProvider:
    """
    Thin client for the FRED REST API.

    Generic methods
    ───────────────
    get_series(series_id)              → list of {date, value} dicts (full history)
    get_recent_values(series_id, n)    → last n observations as list[dict]
    get_latest_value(series_id)        → most recent float value, or None

    Named convenience methods
    ─────────────────────────
    get_yield_spread()                 → latest T10Y2Y spread (float or None)
    get_housing_starts()               → latest HOUST value (float or None)
    get_jobless_claims()               → latest ICSA value (float or None)
    get_oecd_cli()                     → latest USALOLITONOSTSAM value (float or None)
    get_mfg_prod()                     → latest IPMAN value (float or None)
    get_retail_sales()                 → latest RSAFS value (float or None)
    get_consumer_sentiment()           → latest UMCSENT value (float or None)
    get_lei_snapshot()                 → dict of all key indicators (latest values)
    """

    name = "FRED"

    def __init__(self) -> None:
        self._api_key = Config.FRED_API_KEY
        self._session = requests.Session()

    def is_available(self) -> bool:
        available = bool(self._api_key)
        if not available:
            logger.debug(
                "FREDProvider: FRED_API_KEY not set — provider disabled"
            )
        return available

    # ── Generic series methods ─────────────────────────────────────────────────

    def get_series(
        self,
        series_id: str,
        years_back: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Fetch recent observations for a FRED series.

        years_back — how many calendar years of history to request (default 3).
        Setting this limits payload size for high-frequency series like ICSA
        (weekly since 1967 → 3 000+ rows) to a manageable ~150 rows.
        Pass years_back=0 to fetch all history (slow for weekly/daily series).

        Returns a list of dicts:
          [{"date": "2024-01-01", "value": 3.87}, ...]
        Missing observations (FRED sentinel ".") have value=None.
        Returns [] on error or if unavailable.
        """
        if not self.is_available():
            return []
        obs_start = ""
        if years_back > 0:
            obs_start = str(date.today() - timedelta(days=365 * years_back))
        data = self._get_observations(series_id, observation_start=obs_start)
        if data is None:
            return []
        return [
            {"date": obs["date"], "value": _safe_float(obs.get("value"))}
            for obs in data
        ]

    def get_recent_values(
        self, series_id: str, limit: int = 12
    ) -> list[dict[str, Any]]:
        """
        Return the most recent *limit* observations for a series.
        Observations are sorted ascending (oldest first) as FRED returns them.
        Returns [] on error.
        """
        all_obs = self.get_series(series_id)
        if not all_obs:
            return []
        return all_obs[-limit:]

    def get_latest_value(self, series_id: str) -> Optional[float]:
        """
        Return the single most recent non-null observation value, or None.
        Skips trailing null (".") observations FRED sometimes appends.
        """
        value, _date = self.get_latest_with_date(series_id)
        return value

    def get_latest_with_date(self, series_id: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Return (value, date) for the most recent non-null observation.
        Date is a YYYY-MM-DD string.  Both are None if no data is available.
        """
        obs = self.get_series(series_id)
        if not obs:
            return None, None
        for entry in reversed(obs):
            if entry["value"] is not None:
                return entry["value"], entry.get("date")
        return None, None

    # ── Named convenience methods ──────────────────────────────────────────────

    def get_yield_spread(self) -> Optional[float]:
        """Return the latest 10Y-2Y Treasury yield spread (T10Y2Y)."""
        return self.get_latest_value(SERIES_YIELD_SPREAD_10Y2Y)

    def get_housing_starts(self) -> Optional[float]:
        """Return the latest US housing starts (HOUST), thousands of units, SAAR."""
        return self.get_latest_value(SERIES_HOUSING_STARTS)

    def get_jobless_claims(self) -> Optional[float]:
        """Return the latest initial jobless claims (ICSA), seasonally adjusted."""
        return self.get_latest_value(SERIES_JOBLESS_CLAIMS)

    def get_oecd_cli(self) -> Optional[float]:
        """Return the latest OECD Composite Leading Indicator for US (USALOLITONOSTSAM)."""
        return self.get_latest_value(SERIES_OECD_CLI)

    def get_mfg_prod(self) -> Optional[float]:
        """Return the latest Industrial Production: Manufacturing index (IPMAN, 2017=100)."""
        return self.get_latest_value(SERIES_MFG_PROD)

    def get_retail_sales(self) -> Optional[float]:
        """Return the latest retail sales level (RSAFS, millions USD)."""
        return self.get_latest_value(SERIES_RETAIL_SALES)

    def get_consumer_sentiment(self) -> Optional[float]:
        """Return the latest University of Michigan Consumer Sentiment (UMCSENT)."""
        return self.get_latest_value(SERIES_CONSUMER_SENTIMENT)

    def _compute_retail_yoy(self) -> Optional[float]:
        """
        Compute year-over-year % change in retail sales (RSAFS).
        Requires at least 13 non-null monthly observations (current month + 12 prior).
        Returns None if insufficient data.
        """
        obs = self.get_series(SERIES_RETAIL_SALES, years_back=2)
        non_null = [o for o in obs if o["value"] is not None]
        if len(non_null) < 13:
            return None
        latest_val = non_null[-1]["value"]
        prior_val  = non_null[-13]["value"]
        if not prior_val:
            return None
        return (latest_val - prior_val) / abs(prior_val) * 100

    def get_series_trend(
        self,
        series_id: str,
        n_periods: int = 3,
        noise_threshold: float = 0.10,
    ) -> Optional[str]:
        """
        Compute the trend direction for a series from the last n_periods + 1
        non-null observations.

        Returns:
          "rising"     — last value > first value by more than noise_threshold
          "falling"    — last value < first value by more than noise_threshold
          "inflecting" — direction reversed within the window (mid-point reversal)
          None         — insufficient data or provider unavailable

        noise_threshold prevents false signals from minor data revisions.
        Recommended values:
          CB LEI (USSLIND)            : 0.10  (index level ~100-120)
          Yield spread (T10Y2Y)       : 0.10  (percentage points)
        """
        if not self.is_available():
            return None

        obs = self.get_recent_values(series_id, limit=n_periods + 1)
        # Keep only non-null values
        values = [o["value"] for o in obs if o["value"] is not None]

        if len(values) < 2:
            return None

        first, last = values[0], values[-1]
        delta = last - first

        if abs(delta) <= noise_threshold:
            # Flat — but check for inflection within the window
            if len(values) >= 3:
                mid = values[len(values) // 2]
                mid_delta = mid - first
                if abs(mid_delta) > noise_threshold and (mid_delta * delta) < 0:
                    return "inflecting"
            return None  # truly flat, no trend signal

        # Check for inflection: net direction disagrees with mid-point direction
        if len(values) >= 3:
            mid = values[len(values) // 2]
            mid_delta = mid - first
            if abs(mid_delta) > noise_threshold and (mid_delta * delta) < 0:
                return "inflecting"

        return "rising" if delta > 0 else "falling"

    def get_lei_snapshot(self) -> dict[str, Any]:
        """
        Fetch the latest value for every tracked leading indicator in one call.
        Returns a dict keyed by friendly name.  Each value is a float or None.

        Also returns "_observation_dates" sub-dict with the FRED observation date
        for each indicator.

        Example return:
          {
            "yield_spread_10y2y":  0.42,
            "housing_starts":      1423.0,
            "jobless_claims":      215000.0,
            "oecd_cli":            100.2,
            "mfg_prod":            97.5,
            "retail_sales_yoy":    3.1,
            "consumer_sentiment":  74.5,
            "lei_trend":           "rising",
            "yield_spread_trend":  "flat",
            "_observation_dates": {
              "yield_spread_10y2y": "2026-02-14",
              "housing_starts":     "2026-01-01",
              ...
            },
          }
        """
        snapshot: dict[str, Any] = {}
        obs_dates: dict[str, Optional[str]] = {}

        checks = [
            ("yield_spread_10y2y", SERIES_YIELD_SPREAD_10Y2Y),
            ("housing_starts",     SERIES_HOUSING_STARTS),
            ("jobless_claims",     SERIES_JOBLESS_CLAIMS),
            ("oecd_cli",           SERIES_OECD_CLI),
            ("mfg_prod",           SERIES_MFG_PROD),
            ("consumer_sentiment", SERIES_CONSUMER_SENTIMENT),
        ]
        for name, series_id in checks:
            value, obs_date = self.get_latest_with_date(series_id)
            snapshot[name] = value
            obs_dates[name] = obs_date
            status = (
                f"{value}  (obs date: {obs_date})" if value is not None
                else "unavailable"
            )
            logger.info("FREDProvider: %s (%s) → %s", name, series_id, status)
            print(f"  [FRED] {name} ({series_id}) → {status}")

        # Retail sales YoY — requires 13 months so computed separately
        retail_yoy = self._compute_retail_yoy()
        _, retail_date = self.get_latest_with_date(SERIES_RETAIL_SALES)
        snapshot["retail_sales_yoy"] = retail_yoy
        obs_dates["retail_sales_yoy"] = retail_date
        retail_status = (
            f"{retail_yoy:.1f}% YoY  (obs date: {retail_date})"
            if retail_yoy is not None else "unavailable"
        )
        logger.info("FREDProvider: retail_sales_yoy (%s) → %s", SERIES_RETAIL_SALES, retail_status)
        print(f"  [FRED] retail_sales_yoy ({SERIES_RETAIL_SALES}) → {retail_status}")

        snapshot["_observation_dates"] = obs_dates

        # Trend directions: OECD CLI (monthly, 3-period) and yield spread (daily, 20-period)
        lei_trend    = self.get_series_trend(
            SERIES_OECD_CLI,           n_periods=3,  noise_threshold=0.10
        )
        spread_trend = self.get_series_trend(
            SERIES_YIELD_SPREAD_10Y2Y, n_periods=20, noise_threshold=0.10
        )
        snapshot["lei_trend"]          = lei_trend
        snapshot["yield_spread_trend"] = spread_trend

        trend_log = (
            f"lei_trend={lei_trend or 'N/A'}"
            f"  yield_spread_trend={spread_trend or 'N/A'}"
        )
        logger.info("FREDProvider: trends — %s", trend_log)
        print(f"  [FRED] trends — {trend_log}")

        return snapshot

    # ── Private HTTP layer ─────────────────────────────────────────────────────

    def _get_observations(
        self,
        series_id: str,
        observation_start: str = "",
    ) -> Optional[list[dict]]:
        """
        GET /fred/series/observations for the given series_id.

        observation_start — YYYY-MM-DD.  When provided, only observations on or
        after this date are returned.  Dramatically reduces payload size for
        high-frequency series (e.g. ICSA has 3 000+ rows since 1967).

        Returns the list of observation dicts, or None on any error.
        """
        url = f"{_FRED_BASE}/series/observations"
        params: dict[str, str] = {
            "series_id":  series_id,
            "api_key":    self._api_key,
            "file_type":  "json",
            "sort_order": "asc",
        }
        if observation_start:
            params["observation_start"] = observation_start
        logger.info(
            "FREDProvider: fetching series %s (start=%s)",
            series_id, observation_start or "all",
        )
        print(
            f"  [FRED] fetching series {series_id}"
            + (f" from {observation_start}" if observation_start else "")
            + " ..."
        )

        try:
            resp = self._session.get(url, params=params, timeout=_REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as exc:
            logger.warning("FREDProvider: request error for %s — %s", series_id, exc)
            print(f"  [FRED] request error for {series_id}: {exc}")
            return None

        if resp.status_code == 403:
            logger.warning(
                "FREDProvider: 403 for %s — series may require a redistribution agreement",
                series_id,
            )
            print(
                f"  [FRED] 403 for {series_id} — "
                "this series may not be redistributable via the free FRED API"
            )
            return None

        if resp.status_code == 400:
            # FRED returns 400 for unknown series IDs with a JSON error body
            try:
                err = resp.json()
                logger.warning(
                    "FREDProvider: 400 for %s — %s", series_id, err.get("error_message", "")
                )
            except Exception:
                logger.warning("FREDProvider: 400 for %s", series_id)
            return None

        if not resp.ok:
            logger.warning(
                "FREDProvider: HTTP %d for %s", resp.status_code, series_id
            )
            return None

        try:
            data = resp.json()
        except ValueError as exc:
            logger.warning("FREDProvider: JSON parse error for %s — %s", series_id, exc)
            return None

        observations = data.get("observations")
        if observations is None:
            logger.warning(
                "FREDProvider: unexpected response shape for %s — no 'observations' key",
                series_id,
            )
            return None

        count = len(observations)
        print(f"  [FRED] {series_id} → {count} observations")
        logger.info("FREDProvider: %s — %d observations returned", series_id, count)
        return observations
