"""Deterministic provider used by the test suite and by default.

The brief requires the tests to run without spending money on an external API,
so this is the default. It produces the same plan every time for a given
employee record, which also makes the demo transcript reproducible.
"""

from __future__ import annotations

from typing import Any

from offboarding.llm.provider import ChecklistItem, DeprovisioningPlan, validate_plan

#: Systems where lingering access is most dangerous after someone leaves.
HIGH_RISK_SYSTEMS = frozenset({"aws", "github", "vpn", "netsuite", "salesforce"})

ACTIONS = {
    "github": "Remove from all organisations and revoke personal access tokens",
    "aws": "Delete IAM user and rotate any shared keys they held",
    "slack": "Deactivate account and transfer owned channels",
    "jira": "Deactivate account and reassign open issues",
    "vpn": "Revoke client certificate and remove from the access group",
    "netsuite": "Disable login and revoke approval authority",
    "salesforce": "Freeze user and reassign owned opportunities",
    "zoom": "Deactivate account and release the licence",
}


class FakeLLM:
    """Builds a plan from the employee record with no network call."""

    name = "fake"

    def generate_plan(self, employee: dict[str, Any]) -> DeprovisioningPlan:
        """Return one checklist item per system the employee has access to."""
        items = [
            ChecklistItem(
                system=system,
                action=ACTIONS.get(system, f"Revoke access to {system}"),
                risk_level="high" if system in HIGH_RISK_SYSTEMS else "medium",
            )
            for system in employee.get("systems", [])
        ]
        plan = DeprovisioningPlan(
            items=items,
            notes=(
                f"{len(items)} systems to deprovision before "
                f"{employee.get('last_day', 'the last day')}."
                + (
                    " Company hardware is outstanding and must be collected."
                    if employee.get("has_company_hardware")
                    else ""
                )
            ),
        )
        # Run the same validation the real provider is held to, so the fake
        # cannot drift into producing plans the real path would reject.
        return validate_plan(plan, employee)
