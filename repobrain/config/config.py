from pathlib import Path
import yaml


def _find_config_file() -> Path:
    """Walk up from this file's location to find config.yaml in the project root."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        candidate = current / "config.yaml"
        if candidate.exists():
            return candidate
        current = current.parent
    raise FileNotFoundError(
        "config.yaml not found. Please create one in the project root."
    )


_config_cache: dict | None = None


def get_config() -> dict:
    """Load and return the configuration from config.yaml.

    Returns a dict with all configuration values. The result is cached
    after the first call.
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    config_path = _find_config_file()
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    # Apply defaults for any missing keys
    defaults = {
        "llm_provider": "local",
        "model": "llama3",
        "openai_api_key": "",
        "openai_model": "gpt-4o",
        "ollama_base_url": "http://localhost:11434",
        "embedding_model": "all-MiniLM-L6-v2",
        "chroma_persist_dir": "./.chroma",
        "analysis_output_dir": "./analysis",
    }
    for key, value in defaults.items():
        data.setdefault(key, value)

    _config_cache = data
    return _config_cache


def reload_config() -> dict:
    """Force reload of configuration from disk."""
    global _config_cache
    _config_cache = None
    return get_config()
