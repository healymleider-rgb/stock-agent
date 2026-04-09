"""
Scorecard and CategoryScore models.

The Scorecard is assembled incrementally as each analysis agent
reports back. The Reporting Agent reads the final state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Stance(str, Enum):
    BULLISH = "Bullish"
    NEUTRAL = "Neutral"
    BEARISH = "Bearish"


@dataclass
class CategoryScore:
    """Score for a single analysis dimension."""
    name:         str
    score:        float                              # 0-100
    weight:       float                              # contribution weight (all weights sum to 1.0)
    factors:      list[str] = field(default_factory=list)
    reasoning:    str  = ""
    data_quality: str  = "good"                     # "good" | "partial" | "missing"


@dataclass
class Scorecard:
    """
    Master scorecard assembled from all agent outputs.

    Typical usage
    -------------
    1. ReportingAgent sets each category score field.
    2. Call compute_overall_score() to calculate the weighted average.
    3. Call determine_stance() to set Bullish / Neutral / Bearish.
    """
    ticker: str

    # Category scores — populated by analysis agents via ReportingAgent
    valuation:        Optional[CategoryScore] = None
    growth:           Optional[CategoryScore] = None
    profitability:    Optional[CategoryScore] = None
    financial_health: Optional[CategoryScore] = None
    momentum:         Optional[CategoryScore] = None
    risk:             Optional[CategoryScore] = None

    # Summary fields — set after all categories are populated
    overall_score: float  = 0.0
    stance:        Stance = Stance.NEUTRAL
    confidence:    float  = 0.0

    bullish_factors:        list[str] = field(default_factory=list)
    bearish_factors:        list[str] = field(default_factory=list)
    key_drivers:            list[str] = field(default_factory=list)
    what_would_change_view: list[str] = field(default_factory=list)
    risk_flags:             list[str] = field(default_factory=list)

    # Signal agreement explanation — set by ReportingAgent after blending
    # data-availability confidence with cross-factor agreement confidence.
    confidence_explanation: str = ""

    def compute_overall_score(self) -> float:
        """Weighted average of all non-None category scores."""
        categories = [
            self.valuation,
            self.growth,
            self.profitability,
            self.financial_health,
            self.momentum,
            self.risk,
        ]
        weighted_sum = 0.0
        total_weight = 0.0
        for cat in categories:
            if cat is not None:
                weighted_sum += cat.score * cat.weight
                total_weight += cat.weight
        self.overall_score = (
            weighted_sum / total_weight if total_weight > 0 else 0.0
        )
        return self.overall_score

    def determine_stance(self) -> Stance:
        """
        Assign stance from overall_score.
        >= 65 -> Bullish
        >= 45 -> Neutral
        <  45 -> Bearish
        """
        if self.overall_score >= 65:
            self.stance = Stance.BULLISH
        elif self.overall_score >= 45:
            self.stance = Stance.NEUTRAL
        else:
            self.stance = Stance.BEARISH
        return self.stance

    def category_scores_dict(self) -> dict[str, Optional[float]]:
        """Flat dict of category name -> score, useful for serialisation."""
        return {
            "valuation":        self.valuation.score        if self.valuation        else None,
            "growth":           self.growth.score           if self.growth           else None,
            "profitability":    self.profitability.score    if self.profitability    else None,
            "financial_health": self.financial_health.score if self.financial_health else None,
            "momentum":         self.momentum.score         if self.momentum         else None,
            "risk":             self.risk.score             if self.risk             else None,
        }
