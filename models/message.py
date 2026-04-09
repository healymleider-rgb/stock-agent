"""
Standardized inter-agent message contract.

All communication between agents flows through AgentMessage.
The design is intentionally transport-agnostic — swap the in-process
dispatch for REST, WebSockets, or a message queue without touching agent logic.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MessageType(str, Enum):
    # Data layer
    DATA_REQUEST  = "DATA_REQUEST"
    DATA_RESPONSE = "DATA_RESPONSE"

    # Analysis layer
    ANALYSIS_REQUEST  = "ANALYSIS_REQUEST"
    ANALYSIS_RESPONSE = "ANALYSIS_RESPONSE"

    # Risk alerts (can be emitted at any point)
    RISK_ALERT = "RISK_ALERT"

    # Score updates from any analysis agent
    SCORE_UPDATE = "SCORE_UPDATE"

    # Final reporting
    FINAL_SUMMARY_REQUEST  = "FINAL_SUMMARY_REQUEST"
    FINAL_SUMMARY_RESPONSE = "FINAL_SUMMARY_RESPONSE"

    # Orchestration control
    PLAN_CREATED       = "PLAN_CREATED"
    ITERATION_COMPLETE = "ITERATION_COMPLETE"


@dataclass
class AgentMessage:
    """
    Single message unit passed between agents.

    Fields
    ------
    sender            : name of the originating agent
    recipient         : name of the target agent
    ticker            : stock symbol being evaluated
    message_type      : one of MessageType enum values
    payload           : arbitrary structured data (keep serialisable)
    confidence        : sender's confidence in the payload content (0-1)
    reasoning_summary : one- or two-sentence description of why this was sent
    timestamp         : UTC creation time
    message_id        : short unique identifier for tracing
    """

    sender:            str
    recipient:         str
    ticker:            str
    message_type:      MessageType
    payload:           dict[str, Any]
    confidence:        float = 0.0
    reasoning_summary: str   = ""
    timestamp:  datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    message_id: str = field(
        default_factory=lambda: str(uuid.uuid4())[:8]
    )

    def is_error(self) -> bool:
        return bool(self.payload.get("error"))

    def __repr__(self) -> str:
        return (
            f"[{self.message_id}] {self.sender} -> {self.recipient} "
            f"({self.message_type.value}) ticker={self.ticker} "
            f"conf={self.confidence:.2f}"
        )
