"""
BaseAgent — abstract foundation for all agents.

Every agent must implement process_message().
The base class handles message validation, logging, and error wrapping.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from models.message import AgentMessage, MessageType
from utils.logger import logger


class BaseAgent(ABC):
    """
    Abstract base class for all stock evaluation agents.

    Subclasses
    ----------
    - Must implement process_message(msg) -> AgentMessage
    - Should declare a class-level `name` attribute
    - Should never call FMP directly (only DataRetrievalAgent may do so)
    """

    name: str = "base_agent"

    def handle(self, message: AgentMessage) -> AgentMessage:
        """
        Public entry point.
        Wraps process_message with logging and error handling.
        """
        logger.info(
            "Agent %-28s received %s from %s",
            self.name,
            message.message_type.value,
            message.sender,
        )
        try:
            response = self.process_message(message)
        except Exception as exc:
            logger.exception("Agent %s raised an error: %s", self.name, exc)
            response = self._error_response(message, str(exc))

        logger.info(
            "Agent %-28s sending  %s -> %s  conf=%.2f",
            self.name,
            response.message_type.value,
            response.recipient,
            response.confidence,
        )
        return response

    @abstractmethod
    def process_message(self, message: AgentMessage) -> AgentMessage:
        """Core agent logic. Must return an AgentMessage."""

    # ── Response builder helpers ───────────────────────────────────────────────

    def _reply(
        self,
        original: AgentMessage,
        message_type: MessageType,
        payload: dict,
        confidence: float = 0.0,
        reasoning_summary: str = "",
    ) -> AgentMessage:
        return AgentMessage(
            sender=self.name,
            recipient=original.sender,
            ticker=original.ticker,
            message_type=message_type,
            payload=payload,
            confidence=confidence,
            reasoning_summary=reasoning_summary,
        )

    def _error_response(self, original: AgentMessage, error: str) -> AgentMessage:
        return AgentMessage(
            sender=self.name,
            recipient=original.sender,
            ticker=original.ticker,
            message_type=original.message_type,
            payload={"error": error},
            confidence=0.0,
            reasoning_summary=f"Error in {self.name}: {error}",
        )
