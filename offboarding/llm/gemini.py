"""Gemini-backed provider.

Imported lazily by :func:`offboarding.llm.factory.build_llm` so that neither the
test suite nor a default run needs ``langchain-google-genai`` installed or a
``GEMINI_API_KEY`` set.

Transport failures are mapped to :class:`TransientToolError` so the step's retry
policy applies to them; a response that does not fit the schema is permanent,
because asking the same question again is not a fix.
"""

from __future__ import annotations

import os
from typing import Any

from offboarding.domain.errors import PermanentToolError, TransientToolError
from offboarding.llm.provider import DeprovisioningPlan, parse_plan

SYSTEM_PROMPT = """\
You are an IT offboarding assistant. Given an employee record, produce a \
deprovisioning checklist.

Rules:
- Produce exactly one item per system listed in the employee's `systems` field.
- Never invent a system that is not in that list.
- risk_level is "high" when lingering access would expose production systems, \
customer data or money; otherwise "medium" or "low".
- `action` is a single concrete instruction for an IT administrator.
"""


class GeminiLLM:
    """Generates the deprovisioning plan with Gemini structured output."""

    name = "gemini"

    def __init__(
        self, *, model: str | None = None, api_key: str | None = None
    ) -> None:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise PermanentToolError(
                "the gemini provider requires the 'gemini' extra: "
                "pip install -e '.[gemini]'"
            ) from exc

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise PermanentToolError(
                "OFFBOARDING_LLM_PROVIDER=gemini but GEMINI_API_KEY is not set"
            )

        self._model_name = model or os.environ.get(
            "GEMINI_MODEL", "gemini-2.5-flash"
        )
        # with_structured_output makes the schema the model's contract, so we
        # never parse free text out of a chat response.
        self._client = ChatGoogleGenerativeAI(
            model=self._model_name, google_api_key=key, temperature=0
        ).with_structured_output(DeprovisioningPlan)

    def generate_plan(self, employee: dict[str, Any]) -> DeprovisioningPlan:
        """Ask Gemini for a plan and validate it against the employee record."""
        prompt = (
            f"{SYSTEM_PROMPT}\n\nEmployee record:\n"
            f"- department: {employee.get('department')}\n"
            f"- last day: {employee.get('last_day')}\n"
            f"- systems: {employee.get('systems')}\n"
            f"- company hardware outstanding: "
            f"{employee.get('has_company_hardware')}\n"
        )
        try:
            response = self._client.invoke(prompt)
        except Exception as exc:
            # Rate limits, timeouts and 5xx all arrive here. Treating them as
            # transient lets the step's retry policy handle them.
            raise TransientToolError(
                f"gemini request failed: {type(exc).__name__}: {exc}"
            ) from exc

        return parse_plan(response, employee)
