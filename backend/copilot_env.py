"""
Copilot-backed local LLM defaults.

MiroFish talks to LLMs through OpenAI-compatible clients. The local Copilot
adapter exposes that same API surface, but it does not need a real API key.
This helper fills the placeholder values expected by the OpenAI SDK and CAMEL
when LLM_PROVIDER=copilot is selected.
"""

from __future__ import annotations

import os

DEFAULT_COPILOT_BASE_URL = "http://127.0.0.1:8787/v1"
DEFAULT_COPILOT_MODEL = "gpt-5.5"
COPILOT_PLACEHOLDER_KEY = "copilot-local"


def is_copilot_provider() -> bool:
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    base_url = os.environ.get("LLM_BASE_URL", "").strip()
    return provider == "copilot" or "127.0.0.1:8787" in base_url or "localhost:8787" in base_url


def apply_copilot_llm_defaults() -> None:
    if not is_copilot_provider():
        return

    os.environ.setdefault("LLM_PROVIDER", "copilot")
    os.environ.setdefault("LLM_BASE_URL", DEFAULT_COPILOT_BASE_URL)
    os.environ.setdefault("LLM_MODEL_NAME", os.environ.get("COPILOT_MODEL", DEFAULT_COPILOT_MODEL))
    os.environ.setdefault("LLM_API_KEY", COPILOT_PLACEHOLDER_KEY)

    # CAMEL/OASIS reads OpenAI env vars directly in the simulation scripts.
    os.environ.setdefault("OPENAI_API_KEY", os.environ["LLM_API_KEY"])
    os.environ.setdefault("OPENAI_API_BASE_URL", os.environ["LLM_BASE_URL"])
