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

Supported indicators (initial set)
-----------------------------------
Series IDs are FRED's canonical identifiers.  Pass them directly to the generic
methods, or use the named constants and convenience methods below.

  Indicator                         FRED Series ID   Frequency
  ─────────────────────────────────────────────────────────────
  10-Year Treasury yield            GS10             Monthly
  2-Year Treasury yield             GS2              Monthly
  Yield curve spread (10Y-2Y)       T10Y2Y           Daily (FRED-computed)
  ISM Manufacturing PMI proxy       MANEMP           Monthly (mfg employment)
  Housing starts (total, SAAR)      HOUST            Monthly
  Initial jobless claims            ICSA             Weekly
  Conference Board LEI              USSLIND          Monthly  [requires subscription]
  OECD CLI for USA                  USALOLITONOSTSAM Monthly

Notes
-----
- USSLIND (Conference Board LEI) may return 403 if FRED's redistribution
  agreement is not in place.  get_lei_composite() will return None and log a
  warning rather than raising.
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

# Activity / PMI proxy
SERIES_MFG_EMPLOYMENT     = "MANEMP"        # Manufacturing employees (monthly, SAAR)

# Housing
SERIES_HOUSING_STARTS     = "HOUST"         # Housing starts: total (monthly, SAAR)

# Labour market
SERIES_JOBLESS_CLAIMS     = "ICSA"          # Initial claims, seasonally adjusted (weekly)

# Composite leading indicators
SERIES_CONF_BOARD_LEI     = "USSLIND"       # Conference Board LEI (may need subscription)
SERIES_OECD_CLI_USA       = "USALOLITONOSTSAM"  # OECD CLI for United States

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
    get_lei_composite()                → latest USSLIND value (float or None)
    get_oecd_cli()                     → latest OECD CLI value (float or None)
    get_mfg_employment()               → latest MANEMP value (float or None)
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
        """
        Return the latest 10Y-2Y Treasury yield spread (T10Y2Y).
        A negative value signals yield curve inversion.
        """
        return self.get_latest_value(SERIES_YIELD_SPREAD_10Y2Y)

    def get_housing_starts(self) -> Optional[float]:
        """
        Return the latest US housing starts (HOUST), thousands of units, SAAR.
        """
        return self.get_latest_value(SERIES_HOUSING_STARTS)

    def get_jobless_claims(self) -> Optional[float]:
        """
        Return the latest initial jobless claims (ICSA), seasonally adjusted.
        Unit: number of persons filing.
        """
        return self.get_latest_value(SERIES_JOBLESS_CLAIMS)

    def get_lei_composite(self) -> Optional[float]:
        """
        Return the latest Conference Board LEI index value (USSLIND).
        May return None if FRED redistribution license is not in place (403).
        """
        return self.get_latest_value(SERIES_CONF_BOARD_LEI)

    def get_oecd_cli(self) -> Optional[float]:
        """
        Return the latest OECD Composite Leading Indicator for the USA.
        Values > 100 indicate above-trend growth momentum.
        """
        return self.get_latest_value(SERIES_OECD_CLI_USA)

    def get_mfg_employment(self) -> Optional[float]:
        """
        Return the latest manufacturing employees count (MANEMP, thousands, SAAR).
        Proxy for ISM / PMI-adjacent manufacturing activity when direct PMI is
        unavailable from FRED.
        """
        return self.get_latest_value(SERIES_MFG_EMPLOYMENT)

    def get_lei_snapshot(self) -> dict[str, Any]:
        """
        Fetch the latest value for every tracked leading indicator in one call.
        Returns a dict keyed by friendly name.  Each value is a float or None.

        Also returns "_observation_dates" sub-dict with the FRED observation date
        for each indicator — lets callers log staleness without changing the
        shape of the primary value fields.

        Example return:
          {
            "yield_spread_10y2y": -0.42,
            "housing_starts":     1423.0,
            "jobless_claims":     215000.0,
            "lei_composite":      None,
            "oecd_cli":           100.12,
            "mfg_employment":     12872.0,
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
            ("lei_composite",      SERIES_CONF_BOARD_LEI),
            ("oecd_cli",           SERIES_OECD_CLI_USA),
            ("mfg_employment",     SERIES_MFG_EMPLOYMENT),
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
        snapshot["_observation_dates"] = obs_dates
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
