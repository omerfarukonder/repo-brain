from __future__ import annotations

from .base import LLMClient


class OpenAIClient(LLMClient):
    """LLM client that calls the OpenAI API."""

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
        except ImportError as e:
            raise ImportError("openai package is required: pip install openai") from e
        self.model = model

    def complete(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"OpenAI error: {e}"
