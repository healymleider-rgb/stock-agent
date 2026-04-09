"""
Central configuration — loaded once at startup.
All API keys, thresholds, and scoring weights live here.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── API keys ───────────────────────────────────────────────────────────────
    FMP_API_KEY: str = os.getenv("FMP_API_KEY", "")
    FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    # FRED — Federal Reserve Economic Data (macroeconomic indicators).
    # Free API key: https://fred.stlouisfed.org/docs/api/api_key.html
    FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")

    # ── Optional fallback data providers ──────────────────────────────────────
    # These are disabled by default. Set the relevant env var to activate.
    # See the corresponding api/*_provider.py for details on each.
    #
    # Alpha Vantage — documented REST API for financial statements.
    # Free tier: 25 req/day. https://www.alphavantage.co/support/#api-key
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    #
    # YCharts — session token extracted from a logged-in browser session.
    # NOTE: Verify YCharts Terms of Service before use.
    YCHARTS_SESSION_TOKEN: str = os.getenv("YCHARTS_SESSION_TOKEN", "")
    #
    # Alpha Spread — API key (future official API) or session token.
    ALPHASPREAD_API_KEY: str = os.getenv("ALPHASPREAD_API_KEY", "")

    # ── Base URLs ──────────────────────────────────────────────────────────────
    FMP_BASE_URL: str = "https://financialmodelingprep.com/stable"  # legacy /api retired Aug 2025
    FINNHUB_BASE_URL: str = "https://finnhub.io/api/v1"
    EDGAR_BASE_URL: str = "https://data.sec.gov"
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"

    # ── Scoring weights (must sum to 1.0) ──────────────────────────────────────
    SCORE_WEIGHTS: dict = {
        "valuation": 0.20,
        "growth": 0.20,
        "profitability": 0.20,
        "financial_health": 0.20,
        "momentum": 0.10,
        "risk": 0.10,
    }

    # ── Reasoning loop controls ────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float = 0.75
    MAX_ITERATIONS: int = 12

    # ── HTTP client settings ───────────────────────────────────────────────────
    REQUEST_TIMEOUT: int = 20          # seconds
    MAX_RETRIES: int = 3
    RETRY_BACKOFF: float = 1.5         # multiplier between retries

    # ── Data fetch limits ──────────────────────────────────────────────────────
    FINANCIAL_STATEMENT_LIMIT: int = 5  # years of annual statements
    PRICE_HISTORY_DAYS: int = 400       # trading days (~16 months for 12-month returns)

    # ── Memory / persistence ───────────────────────────────────────────────────
    MEMORY_DIR: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "memory", "store"
    )

    # ── Logging ────────────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "stock_eval.log")

    @classmethod
    def validate(cls) -> None:
        """Raise early if required keys are missing."""
        if not cls.FMP_API_KEY:
            raise EnvironmentError(
                "FMP_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        total_weight = sum(cls.SCORE_WEIGHTS.values())
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(
                f"SCORE_WEIGHTS must sum to 1.0 — current sum: {total_weight}"
            )
