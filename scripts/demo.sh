#!/usr/bin/env bash
#
# Runnable demo of the offboarding orchestrator. Its stdout, redirected to a
# file, IS docs/demo-transcript.txt -- regenerate it with:
#
#   scripts/demo.sh > docs/demo-transcript.txt
#
# Three parts:
#   1. The CLI lifecycle: start -> duplicate-request refusal -> approve -> sign.
#      Every `offboarding` call below is launched as a real background
#      process and its PID is printed and waited on -- not asserted, shown --
#      so "no state survives between commands" is something you can verify,
#      not just read.
#   2. A literal `kill -9` of a real process. The CLI can't demonstrate this
#      meaningfully: a single invocation starts and exits before you could
#      type a kill command. The API server is the one long-running process in
#      this project, so that's what gets killed here, for real, mid-run.
#   3. The retry-then-succeed path, triggered by an env var -- no code change.
#
# Nothing here is simulated. Every PID printed is real; every "server is
# dead" check is a real failed connection, not a comment saying so.

set -uo pipefail  # deliberately not -e: some commands below are meant to fail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFBOARDING="$REPO_ROOT/.venv/bin/offboarding"
UVICORN="$REPO_ROOT/.venv/bin/uvicorn"
PORT="${DEMO_PORT:-8799}"

if [[ ! -x "$OFFBOARDING" ]]; then
  echo "error: $OFFBOARDING not found. Run 'pip install -e \".[dev]\"' from $REPO_ROOT first." >&2
  exit 1
fi

WORKDIR="$(mktemp -d)"
SERVER_PID=""
# Guard against ever calling `kill -9 0`: an unset/empty pid is a bare arg to
# kill(1), but "0" is POSIX for "every process in this process group" --
# passing that through the trap would take down the script's own shell.
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill -9 "$SERVER_PID" 2>/dev/null
  fi
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

export OFFBOARDING_LLM_PROVIDER=fake   # explicit: no API key needed for this demo

# Runs a CLI command as a real background process, prints its PID, waits for
# it, and preserves its exit code -- so a "duplicate request refused" step
# still shows the real (nonzero) exit status instead of being masked.
run() {
  # printf %q quotes each arg for display, so a copy-pasted line (e.g. the
  # --note text below, which contains spaces) reproduces the same argv --
  # "$*" alone would silently lose that quoting in the printed transcript.
  printf '$ %s' "${OFFBOARDING##*/}"
  printf ' %q' "$@"
  printf '\n'
  "$OFFBOARDING" "$@" &
  local pid=$!
  echo "  (pid: $pid)"
  wait "$pid"
  local status=$?
  if [[ $status -ne 0 ]]; then
    echo "  (exit code: $status)"
  fi
  return $status
}

section() { echo; echo "================================================================"; echo "# $*"; echo "================================================================"; echo; }
note()    { echo "# $*"; }

# ==========================================================================
# Part 1 -- CLI lifecycle
# ==========================================================================

section "PART 1 -- full lifecycle via the CLI"

export OFFBOARDING_DB_PATH="$WORKDIR/part1.db"
note "Database: a fresh temp file (\$OFFBOARDING_DB_PATH), same as any real deployment."
echo

note "1. Start a run. Fetches the HR record, generates a checklist with the LLM,"
note "   and pauses at the first interrupt: HR approval before the high-risk step."
run start emp-001
RUN=$("$OFFBOARDING" list | head -1 | awk '{print $1}')
echo

note "2. Inspect it. The trace shows every step so far, including the pause itself."
run trace "$RUN"
echo

note "3. A duplicate/out-of-order request is refused before it ever reaches the"
note "   graph -- this is the state check in RunService, not the ledger; we have"
note "   not approved yet, so signing is not a valid operation on this run."
run sign "$RUN" --document-id doc_too_early
echo

note "4. Approve. This revokes access and sends the exit paperwork -- the two"
note "   side-effecting steps -- then pauses again waiting for the signed document."
run approve "$RUN" --approver priya.raman --note "Confirmed, last day 9/30."
echo

note "5. Sign. Note the PID above and the PID below: every command in this"
note "   script -- start, trace, approve, sign -- runs as its own OS process."
note "   Nothing about run $RUN lived in memory between any of them; every"
note "   fact RunService needed came back from \$OFFBOARDING_DB_PATH alone."
run sign "$RUN" --document-id doc_final_signed
echo

note "6. Full trace. Attempt #2 on both pause gates (interrupt() re-runs the"
note "   node from the top on every resume) -- but the side-effecting steps"
note "   below them still show exactly ONE completed attempt each."
run trace "$RUN"
echo

note "7. The idempotency ledger: one row per side effect, each fired exactly"
note "   once, regardless of which process was alive when it happened."
run side-effects "$RUN"

# ==========================================================================
# Part 2 -- a real kill -9, on the one process that's actually killable
# ==========================================================================

section "PART 2 -- kill -9 a real process, restart, resume"

export OFFBOARDING_DB_PATH="$WORKDIR/part2.db"
note "A fresh CLI command starts and exits before you could type a kill"
note "command -- there is no PID worth killing mid-flight. The API server is"
note "the one long-running process here, so that's what actually dies below."
echo

note "Starting the API server in the background."
echo "\$ uvicorn offboarding.api.main:app --port $PORT"
"$UVICORN" offboarding.api.main:app --port "$PORT" > "$WORKDIR/server1.log" 2>&1 < /dev/null &
SERVER_PID=$!
disown
sleep 2
echo "  (pid: $SERVER_PID)"
echo

note "Starting a run on it."
RESP=$(curl -s -X POST "http://localhost:$PORT/runs" -H "Content-Type: application/json" -d '{"employee_id":"emp-002"}')
RUN2=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['run_id'])")
echo "\$ curl -s -X POST http://localhost:$PORT/runs -d '{\"employee_id\": \"emp-002\"}'"
echo "$RESP" | python3 -m json.tool
echo

OLD_PID="$SERVER_PID"
echo "\$ kill -9 $OLD_PID"
kill -9 "$OLD_PID"
SERVER_PID=""   # it's dead; don't let the exit trap try to kill it again
sleep 1
echo "\$ curl http://localhost:$PORT/"
curl -s -o /dev/null -w "  connection: %{http_code} (000 = confirmed dead, this is not a simulation)\n" \
  "http://localhost:$PORT/" --max-time 2 || true
echo

note "A brand-new server process, same database file, different PID."
echo "\$ uvicorn offboarding.api.main:app --port $PORT"
"$UVICORN" offboarding.api.main:app --port "$PORT" > "$WORKDIR/server2.log" 2>&1 < /dev/null &
SERVER_PID=$!
disown
sleep 2
echo "  (pid: $SERVER_PID -- was $OLD_PID before the kill; genuinely different)"
echo

echo "\$ curl http://localhost:$PORT/runs/$RUN2"
curl -s "http://localhost:$PORT/runs/$RUN2" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"  the new process sees it: status={d['status']!r} pause_reason={d.get('pause_reason')!r}\")
"
echo

note "Finish it, on the process that did not start it."
echo "\$ curl -X POST http://localhost:$PORT/runs/$RUN2/approve -d '{\"approver\": \"aisha.bakr\"}'"
curl -s -X POST "http://localhost:$PORT/runs/$RUN2/approve" -H "Content-Type: application/json" -d '{"approver":"aisha.bakr"}' > /dev/null
echo "\$ curl -X POST http://localhost:$PORT/runs/$RUN2/sign -d '{\"document_id\": \"doc_killed_and_restarted\"}'"
curl -s -X POST "http://localhost:$PORT/runs/$RUN2/sign" -H "Content-Type: application/json" -d '{"document_id":"doc_killed_and_restarted"}' \
  | python3 -c "import sys,json; print('  final status:', json.load(sys.stdin)['status'])"

kill -9 "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null
SERVER_PID=""

# ==========================================================================
# Part 3 -- retry behaviour, triggered by config alone
# ==========================================================================

section "PART 3 -- retry-then-succeed, no code change"

export OFFBOARDING_DB_PATH="$WORKDIR/part3.db"
export OFFBOARDING_IAM_FAIL_TIMES=2
note "Make the IAM tool fail transiently twice before succeeding."
echo "\$ export OFFBOARDING_IAM_FAIL_TIMES=2"
echo

run start emp-003 > /dev/null
RUN3=$("$OFFBOARDING" list | head -1 | awk '{print $1}')
run approve "$RUN3" --approver tom.becker
echo

note "Attempts 1-2 fail transiently and are retried with backoff; attempt 3"
note "succeeds. Same idempotency key throughout -- one ledger row, not three."
run trace "$RUN3"
echo "(revoke_access rows only:)"
"$OFFBOARDING" trace "$RUN3" | grep revoke_access
