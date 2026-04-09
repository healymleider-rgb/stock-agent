"""
EvaluationState — the mutable working memory of the Orchestrator.

Passed by reference through the reasoning loop.
All agents read from or write via the Orchestrator into this object.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from config import Config
from models.stock_data import StockData
from models.scorecard import Scorecard


@dataclass
class EvaluationState:
    ticker: str

    # Core data and scoring containers
    stock_data: StockData = field(init=False)
    scorecard: Scorecard = field(init=False)

    # Which data slices have been fetched
    data_fetched: dict[str, bool] = field(default_factory=lambda: {
        "profile": False,
        "financials": False,
        "quarterly": False,
        "price_history": False,
        "earnings": False,
        "analyst": False,
        "sector": False,
    })

    # Which analysis agents have completed
    analyses_completed: dict[str, bool] = field(default_factory=lambda: {
        "fundamental": False,
        "technical": False,
        "market": False,
        "sentiment": False,
        "risk": False,
        "macro": False,
    })

    # Raw findings keyed by agent name
    agent_findings: dict[str, Any] = field(default_factory=dict)

    # Step-by-step reasoning log visible in --verbose mode
    reasoning_log: list[str] = field(default_factory=list)

    # Loop controls
    confidence: float = 0.0
    iteration: int = 0
    max_iterations: int = field(default_factory=lambda: Config.MAX_ITERATIONS)
    confidence_threshold: float = field(
        default_factory=lambda: Config.CONFIDENCE_THRESHOLD
    )

    def __post_init__(self) -> None:
        self.stock_data = StockData(ticker=self.ticker)
        self.scorecard = Scorecard(ticker=self.ticker)

    # ── Convenience helpers ────────────────────────────────────────────────────

    def is_complete(self) -> bool:
        all_analysed = all(self.analyses_completed.values())
        confident = self.confidence >= self.confidence_threshold
        at_limit = self.iteration >= self.max_iterations
        return (all_analysed and confident) or at_limit

    def analyses_pending(self) -> list[str]:
        return [k for k, done in self.analyses_completed.items() if not done]

    def data_pending(self) -> list[str]:
        return [k for k, done in self.data_fetched.items() if not done]

    def mark_data_fetched(self, key: str) -> None:
        if key in self.data_fetched:
            self.data_fetched[key] = True

    def mark_analysis_complete(self, name: str) -> None:
        if name in self.analyses_completed:
            self.analyses_completed[name] = True

    def log(self, message: str) -> None:
        self.reasoning_log.append(f"[iter={self.iteration}] {message}")

    def completion_ratio(self) -> float:
        """How much of the analysis pipeline is done (0–1)."""
        done = sum(self.analyses_completed.values())
        total = len(self.analyses_completed)
        return done / total if total else 0.0
