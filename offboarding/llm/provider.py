"""The LLM boundary: one protocol, two implementations, validated output.

The agent's one genuinely non-deterministic step is generating the
deprovisioning checklist. Everything about that step is designed around the
assumption that the model can be wrong:

* it returns **structured** output, validated by :class:`DeprovisioningPlan`;
* it may only name systems the HR record actually lists, so the model cannot
  invent a system to revoke;
* a plan that fails either check is an :class:`InvalidPlan`, which fails the
  run rather than proceeding on a plan we do not trust.

The model proposes; the schema and the allowlist dispose.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from offboarding.domain.errors import InvalidPlan

RiskLevel = Literal["low", "medium", "high"]


class ChecklistItem(BaseModel):
    """One system to deprovision."""

    system: str = Field(description="System identifier, e.g. 'github'.")
    action: str = Field(description="What to do, e.g. 'Revoke org membership'.")
    risk_level: RiskLevel = Field(
        description="How urgent revoking this access is."
    )


class DeprovisioningPlan(BaseModel):
    """The checklist the agent will act on."""

    items: list[ChecklistItem]
    notes: str = ""


class LLMProvider(Protocol):
    """Generates a deprovisioning plan for an employee record."""

    name: str

    def generate_plan(self, employee: dict[str, Any]) -> DeprovisioningPlan:
        """Return a validated plan, or raise :class:`InvalidPlan`."""
        ...


def validate_plan(
    plan: DeprovisioningPlan, employee: dict[str, Any]
) -> DeprovisioningPlan:
    """Reject a plan that does not match the employee's actual access.

    Raises:
        InvalidPlan: if the plan is empty, names a system the employee does
            not have, or omits one they do. Retrying the same prompt is not a
            fix, so this is permanent by construction.
    """
    known = set(employee.get("systems", []))
    proposed = {item.system for item in plan.items}

    if not plan.items:
        raise InvalidPlan("the model returned an empty plan")

    invented = proposed - known
    if invented:
        raise InvalidPlan(
            f"plan names systems the employee does not have: "
            f"{sorted(invented)}"
        )

    missed = known - proposed
    if missed:
        raise InvalidPlan(
            f"plan omits systems the employee still has access to: "
            f"{sorted(missed)}"
        )

    return plan


def parse_plan(payload: Any, employee: dict[str, Any]) -> DeprovisioningPlan:
    """Parse and validate raw model output.

    Raises:
        InvalidPlan: if the payload does not fit the schema.
    """
    try:
        plan = (
            payload
            if isinstance(payload, DeprovisioningPlan)
            else DeprovisioningPlan.model_validate(payload)
        )
    except ValidationError as exc:
        raise InvalidPlan(
            f"the model returned a plan that does not fit the schema: {exc}"
        ) from exc
    return validate_plan(plan, employee)
