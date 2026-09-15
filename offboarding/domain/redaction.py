"""Redaction applied to everything written to the execution trace.

The brief asks that traces avoid logging secrets. Rather than trusting each
call site to remember, every payload written to ``step_trace.detail`` passes
through :func:`redact`, so a credential that appears in a tool result is masked
by default and a new tool cannot leak one by omission.
"""

from __future__ import annotations

from typing import Any

#: Substrings that mark a key as sensitive. Matched case-insensitively against
#: the key name, so ``api_key``, ``Authorization`` and ``refresh_token`` all hit.
SENSITIVE_KEY_MARKERS: tuple[str, ...] = (
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "private_key",
    "session",
)

REDACTED = "***redacted***"


def _is_sensitive(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in SENSITIVE_KEY_MARKERS)


def redact(value: Any) -> Any:
    """Return ``value`` with sensitive-looking fields masked, recursively.

    Structure is preserved so the trace still shows the shape of what happened;
    only the values behind sensitive keys are replaced.
    """
    if isinstance(value, dict):
        return {
            k: (REDACTED if _is_sensitive(str(k)) else redact(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value
