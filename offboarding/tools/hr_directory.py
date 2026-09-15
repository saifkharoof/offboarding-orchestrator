"""Read tool: the HR system of record.

Pure retrieval, no side effects, so it needs no ledger protection -- re-running
it on a resume is harmless by construction.
"""

from __future__ import annotations

from typing import Any

from offboarding.domain.errors import PermanentToolError
from offboarding.tools.base import ToolSpec

#: Fixed directory. A real deployment would call an HR API here; the shape of
#: the record is what the rest of the workflow depends on.
EMPLOYEE_DIRECTORY: dict[str, dict[str, Any]] = {
    "emp-001": {
        "employee_id": "emp-001",
        "name": "Dana Okafor",
        "email": "dana.okafor@example.com",
        "department": "Engineering",
        "manager": "Priya Raman",
        "last_day": "2026-09-30",
        "systems": ["github", "aws", "slack", "jira", "vpn"],
        "has_company_hardware": True,
    },
    "emp-002": {
        "employee_id": "emp-002",
        "name": "Luis Moreno",
        "email": "luis.moreno@example.com",
        "department": "Finance",
        "manager": "Aisha Bakr",
        "last_day": "2026-10-15",
        "systems": ["netsuite", "slack", "vpn"],
        "has_company_hardware": False,
    },
    "emp-003": {
        "employee_id": "emp-003",
        "name": "Wei Zhang",
        "email": "wei.zhang@example.com",
        "department": "Sales",
        "manager": "Tom Becker",
        "last_day": "2026-09-22",
        "systems": ["salesforce", "slack", "zoom", "vpn"],
        "has_company_hardware": True,
    },
}


class HrDirectoryTool:
    """Fetch an employee record by id."""

    spec = ToolSpec(
        name="hr_directory.get_employee",
        description="Fetch an employee record from the HR system of record.",
        side_effecting=False,
    )

    def call(self, *, employee_id: str) -> dict[str, Any]:
        """Return the employee record.

        Raises:
            PermanentToolError: if no such employee exists. Retrying cannot
                conjure one, so this must not be treated as transient.
        """
        record = EMPLOYEE_DIRECTORY.get(employee_id)
        if record is None:
            raise PermanentToolError(f"no employee record for {employee_id!r}")
        return dict(record)
