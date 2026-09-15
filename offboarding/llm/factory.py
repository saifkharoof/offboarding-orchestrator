"""Selects the LLM provider from configuration.

Defaults to the deterministic fake. A real model is opt-in via
``OFFBOARDING_LLM_PROVIDER=gemini``, so the test suite and a fresh checkout
never require a key.
"""

from __future__ import annotations

import os

from offboarding.domain.errors import PermanentToolError
from offboarding.llm.fake import FakeLLM
from offboarding.llm.provider import LLMProvider


def build_llm(provider: str | None = None) -> LLMProvider:
    """Return the configured provider.

    Raises:
        PermanentToolError: if an unknown provider is named.
    """
    choice = (
        provider or os.environ.get("OFFBOARDING_LLM_PROVIDER", "fake")
    ).lower()

    if choice == "fake":
        return FakeLLM()
    if choice == "gemini":
        from offboarding.llm.gemini import GeminiLLM

        return GeminiLLM()
    raise PermanentToolError(
        f"unknown LLM provider {choice!r}; expected 'fake' or 'gemini'"
    )
