"""The tool executor: the single place a tool is ever invoked.

It is deliberately the only path to a tool, because four concerns have to hold
for *every* call and none of them should be a node's responsibility:

1. **Allowlist** -- the tool must be one the step declared.
2. **Budget** -- the run must not have exhausted its tool-call allowance.
3. **Idempotency** -- side-effecting calls go through the ledger.
4. **Trace** -- one row per attempt, including the failed ones.

Retries are implemented here rather than with LangGraph's ``RetryPolicy`` on
purpose. ``RetryPolicy`` retries invisibly, and the brief requires retry
information in the trace; a loop we own writes a trace row per attempt, so
"this succeeded on attempt 3 after two 503s" is readable afterwards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from offboarding.domain.errors import (
    BudgetExceeded,
    PermanentToolError,
    SideEffectReconciliationRequired,
    ToolNotAllowed,
    TransientToolError,
)
from offboarding.domain.redaction import redact
from offboarding.persistence.repositories import RunRepository, TraceRepository
from offboarding.tools.base import ToolRegistry
from offboarding.tools.ledger import SideEffectLedger, build_key


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff settings for transient failures."""

    max_attempts: int = 3
    initial_backoff: float = 0.2
    multiplier: float = 2.0
    max_backoff: float = 5.0

    def backoff_for(self, attempt: int) -> float:
        """Seconds to wait before ``attempt`` + 1 (attempts are 1-based)."""
        delay = self.initial_backoff * (self.multiplier ** (attempt - 1))
        return min(delay, self.max_backoff)


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Outcome of one ``execute`` call, across however many attempts it took."""

    result: dict[str, Any]
    attempts: int
    replayed: bool = False
    reconciled: bool = False


class ToolExecutor:
    """Invokes tools under allowlist, budget, idempotency and trace control."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        ledger: SideEffectLedger,
        runs: RunRepository,
        trace: TraceRepository,
        policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._registry = registry
        self._ledger = ledger
        self._runs = runs
        self._trace = trace
        self._policy = policy or RetryPolicy()
        # Injected so tests do not spend real seconds proving backoff happened.
        self._sleep = sleep

    def execute(
        self,
        *,
        run_id: str,
        step_name: str,
        tool_name: str,
        allowlist: frozenset[str],
        kwargs: dict[str, Any],
        operation: str | None = None,
    ) -> ExecutionResult:
        """Call a tool, retrying transient failures, and trace every attempt.

        Args:
            operation: Names the logical side effect for the idempotency key.
                Required for side-effecting tools; a step performing two
                distinct effects gives them different operation names.

        Raises:
            ToolNotAllowed: the tool is outside the step's allowlist.
            BudgetExceeded: the run has spent its tool-call allowance.
            TransientToolError: every attempt failed transiently.
            PermanentToolError: the tool failed in a way retrying cannot fix.
            SideEffectReconciliationRequired: an interrupted side effect could
                not be reconciled.
        """
        try:
            tool = self._registry.resolve(tool_name, allowlist)
        except ToolNotAllowed as exc:
            # Traced as a step attempt so a guardrail refusal is visible in the
            # trace rather than only in an exception the caller may swallow.
            self._trace_refusal(run_id, step_name, tool_name, exc)
            raise

        if tool.spec.side_effecting and operation is None:
            raise ValueError(
                f"tool {tool_name!r} is side-effecting and requires an "
                "operation name for its idempotency key"
            )

        last_error: Exception | None = None

        for attempt in range(1, self._policy.max_attempts + 1):
            self._check_budget(run_id, step_name, tool_name)

            row = self._trace.start_attempt(
                run_id, step_name, tool_invoked=tool_name
            )
            self._runs.bump_counters(run_id, tool_calls=1)

            try:
                if tool.spec.side_effecting:
                    outcome = self._ledger.run_once(
                        key=build_key(run_id, step_name, operation or ""),
                        run_id=run_id,
                        step_name=step_name,
                        tool=tool,
                        kwargs=kwargs,
                    )
                    result, replayed, reconciled = (
                        outcome.result,
                        outcome.replayed,
                        outcome.reconciled,
                    )
                else:
                    result = tool.call(**kwargs)
                    replayed = reconciled = False

            except TransientToolError as exc:
                last_error = exc
                self._trace.fail_attempt(
                    row.id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    detail={
                        "retryable": True,
                        "attempt": attempt,
                        "max_attempts": self._policy.max_attempts,
                    },
                )
                if attempt < self._policy.max_attempts:
                    self._sleep(self._policy.backoff_for(attempt))
                    continue
                raise

            except (PermanentToolError, SideEffectReconciliationRequired) as exc:
                self._trace.fail_attempt(
                    row.id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    detail={"retryable": False, "attempt": attempt},
                )
                raise

            self._trace.complete_attempt(
                row.id,
                detail=redact(
                    {
                        "attempt": attempt,
                        "replayed": replayed,
                        "reconciled": reconciled,
                        "result": result,
                    }
                ),
            )
            return ExecutionResult(
                result=result,
                attempts=attempt,
                replayed=replayed,
                reconciled=reconciled,
            )

        # Unreachable: the loop either returns or raises.
        raise AssertionError(f"retry loop exited without result: {last_error}")

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------

    def _check_budget(self, run_id: str, step_name: str, tool_name: str) -> None:
        """Refuse a call that would exceed the run's tool-call allowance."""
        run = self._runs.get(run_id)
        if run.tool_call_count >= run.max_tool_calls:
            exc = BudgetExceeded(
                f"run {run_id} has used its {run.max_tool_calls} tool calls"
            )
            self._trace_refusal(run_id, step_name, tool_name, exc)
            raise exc

    def _trace_refusal(
        self, run_id: str, step_name: str, tool_name: str, exc: Exception
    ) -> None:
        row = self._trace.start_attempt(
            run_id, step_name, tool_invoked=tool_name
        )
        self._trace.fail_attempt(
            row.id,
            error_type=type(exc).__name__,
            error_message=str(exc),
            detail={"retryable": False, "guardrail": True},
        )
