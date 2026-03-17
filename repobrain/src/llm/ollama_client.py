from __future__ import annotations

import requests

from .base import LLMClient


class OllamaClient(LLMClient):
    """LLM client that calls a locally running Ollama instance."""

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            return (
                "Unable to connect to Ollama. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except Exception as e:
            return f"Ollama error: {e}"
