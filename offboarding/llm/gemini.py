"""Gemini-backed provider, on the `google-genai` Interactions API.

Imported lazily by :func:`offboarding.llm.factory.build_llm` so that neither the
test suite nor a default run needs ``google-genai`` installed or a
``GEMINI_API_KEY`` set.

Model: ``gemini-3.5-flash-lite``, the lowest-cost current model, chosen because
this call is one lightweight structured-output request and the user is on the
free tier. ``gemini-2.5-*`` is legacy as of the current API and is not used.

Transport failures are mapped to :class:`TransientToolError` so the step's retry
policy applies to them; a response that does not fit the schema is permanent,
because asking the same question again is not a fix.

Interactions are stored server-side by default (free tier: 1 day). This prompt
carries employee PII (name, email, department) and the call is one-shot -- we
never use ``previous_interaction_id`` -- so storage is disabled explicitly
rather than left at a default we don't need.
"""

from __future__ import annotations

import json
import os
from typing import Any

from offboarding.domain.errors import PermanentToolError, TransientToolError
from offboarding.llm.provider import DeprovisioningPlan, parse_plan

DEFAULT_MODEL = "gemini-3.5-flash-lite"

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
            from google import genai
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

        self._model_name = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self._client = genai.Client(api_key=key)
        self._schema = DeprovisioningPlan.model_json_schema()

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
            interaction = self._client.interactions.create(
                model=self._model_name,
                input=prompt,
                store=False,  # one-shot call; no need to retain employee PII
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": self._schema,
                },
            )
        except Exception as exc:
            # Rate limits, timeouts and 5xx all arrive here. Treating them as
            # transient lets the step's retry policy handle them.
            raise TransientToolError(
                f"gemini request failed: {type(exc).__name__}: {exc}"
            ) from exc

        # json.loads (not model_validate_json) so a malformed response raises
        # here, in our control, and can be mapped to PermanentToolError -- the
        # same schema/allowlist validation every provider's output goes
        # through in parse_plan, rather than a raw pydantic ValidationError
        # escaping uncaught.
        try:
            payload = json.loads(interaction.output_text)
        except json.JSONDecodeError as exc:
            raise PermanentToolError(
                f"gemini returned a response that is not valid JSON: {exc}"
            ) from exc

        return parse_plan(payload, employee)
