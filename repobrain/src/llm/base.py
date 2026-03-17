from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send a prompt and return the model's text response."""
        ...
