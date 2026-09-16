#!/usr/bin/env python3
"""Runnable demo of the offboarding orchestrator. Cross-platform: Windows,
macOS, and Linux, with no bash, curl, or POSIX job control required -- only
the Python standard library and the project's own installed executables.

Its stdout, redirected to a file, IS docs/demo-transcript.txt:

    python scripts/demo.py > docs/demo-transcript.txt

Three parts:
  1. The CLI lifecycle: start -> duplicate-request refusal -> approve -> sign.
     Every `offboarding` call below is launched as a real child process
     (subprocess.Popen, never shell=True) and its actual PID is printed and
     waited on -- not asserted, shown -- so "no state survives between
     commands" is something you can verify, not just read.
  2. A literal, forceful termination of a real process. The CLI can't
     demonstrate this meaningfully: a single invocation starts and exits
     before you could interrupt it. The API server is the one long-running
     process in this project, so that's what dies here, for real, mid-run.
     Popen.kill() sends SIGKILL on POSIX and calls TerminateProcess on
     Windows -- on both platforms, the real, immediate, no-cleanup
     termination, not a graceful shutdown request.
  3. The retry-then-succeed path, triggered by an environment variable --
     no code change.

Nothing here is simulated. Every PID printed is real; every "server is dead"
check is a real failed connection, not a comment saying so.
"""

from __future__ import annotations

import atexit
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IS_WINDOWS = os.name == "nt"
PORT = int(os.environ.get("DEMO_PORT", "8799"))

_workdir: Path | None = None
_server_procs: list[subprocess.Popen] = []


def venv_bin(name: str) -> Path:
    """Path to an executable pip installed into .venv -- Scripts/ + .exe on
    Windows, bin/ with no extension everywhere else."""
    if IS_WINDOWS:
        return REPO_ROOT / ".venv" / "Scripts" / f"{name}.exe"
    return REPO_ROOT / ".venv" / "bin" / name


OFFBOARDING = venv_bin("offboarding")
UVICORN = venv_bin("uvicorn")


def _cleanup() -> None:
    for proc in _server_procs:
        if proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
    if _workdir is not None:
        shutil.rmtree(_workdir, ignore_errors=True)


atexit.register(_cleanup)


# --------------------------------------------------------------------------
# Small display / process helpers
# --------------------------------------------------------------------------


def display_cmd(args: list[str]) -> str:
    """A copy-pasteable version of args for this platform's own shell."""
    if IS_WINDOWS:
        return subprocess.list2cmdline(args)
    return shlex.join(args)


def section(title: str) -> None:
    print()
    print("=" * 64)
    print(f"# {title}")
    print("=" * 64)
    print()


def note(*lines: str) -> None:
    for line in lines:
        print(f"# {line}")


def run(args: list[str], env: dict[str, str]) -> tuple[int, str]:
    """Run one `offboarding` CLI command as its own child process.

    Prints the command, its real PID, and its output, then waits for it to
    exit and returns (exit_code, combined_output). A nonzero exit code is
    printed but not raised -- some steps in this demo are meant to fail.
    """
    print(f"$ {display_cmd(['offboarding', *args])}")
    proc = subprocess.Popen(
        [str(OFFBOARDING), *args],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"  (pid: {proc.pid})")
    out, _ = proc.communicate()
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    if proc.returncode != 0:
        print(f"  (exit code: {proc.returncode})")
    return proc.returncode, out


def latest_run_id(env: dict[str, str]) -> str:
    """Run `offboarding list` quietly and return the newest run's id.

    Not itself shown as a transcript step -- same role as capturing `$(...)`
    output in a shell script, not a step a reader needs to see.
    """
    proc = subprocess.run(
        [str(OFFBOARDING), "list"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    first_line = proc.stdout.strip().splitlines()[0]
    return first_line.split()[0]


# --------------------------------------------------------------------------
# HTTP helpers (stdlib only -- no curl, no requests)
# --------------------------------------------------------------------------


def http_get(path: str) -> dict:
    with urllib.request.urlopen(f"http://localhost:{PORT}{path}", timeout=10) as resp:
        return json.loads(resp.read())


def http_post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://localhost:{PORT}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def wait_for_server(timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/openapi.json", timeout=1)
            return
        except Exception as exc:  # noqa: BLE001 - genuinely any error means "not up yet"
            last_error = exc
            time.sleep(0.2)
    raise RuntimeError(f"server on port {PORT} never came up: {last_error}")


def confirm_dead() -> None:
    """Prove the killed server is actually gone -- a real connection attempt,
    not an assertion. An HTTP error response still means something answered
    the socket, so only a connection-level failure counts as dead."""
    try:
        urllib.request.urlopen(f"http://localhost:{PORT}/", timeout=2)
        status = "still responding (unexpected!)"
    except urllib.error.HTTPError:
        status = "still responding (unexpected!)"
    except (urllib.error.URLError, ConnectionError, OSError):
        status = "connection refused (confirmed dead, this is not a simulation)"
    print(f"  {status}")


def start_server(log_path: Path, env: dict[str, str]) -> subprocess.Popen:
    args = ["uvicorn", "offboarding.api.main:app", "--port", str(PORT)]
    print(f"$ {display_cmd(args)}")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [str(UVICORN), "offboarding.api.main:app", "--port", str(PORT)],
        cwd=REPO_ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    _server_procs.append(proc)
    wait_for_server()
    print(f"  (pid: {proc.pid})")
    return proc


def main() -> None:
    global _workdir

    if not OFFBOARDING.exists():
        print(
            f"error: {OFFBOARDING} not found. Run "
            f'`pip install -e ".[dev]"` from {REPO_ROOT} first.',
            file=sys.stderr,
        )
        raise SystemExit(1)

    _workdir = Path(tempfile.mkdtemp(prefix="offboarding-demo-"))
    base_env = os.environ.copy()
    base_env["OFFBOARDING_LLM_PROVIDER"] = "fake"  # explicit: no API key needed

    # ======================================================================
    # Part 1 -- CLI lifecycle
    # ======================================================================

    section("PART 1 -- full lifecycle via the CLI")

    env1 = {**base_env, "OFFBOARDING_DB_PATH": str(_workdir / "part1.db")}
    note("Database: a fresh temp file ($OFFBOARDING_DB_PATH), same as any real deployment.")
    print()

    note(
        "1. Start a run. Fetches the HR record, generates a checklist with the LLM,",
        "   and pauses at the first interrupt: HR approval before the high-risk step.",
    )
    run(["start", "emp-001"], env1)
    run_id = latest_run_id(env1)
    print()

    note("2. Inspect it. The trace shows every step so far, including the pause itself.")
    run(["trace", run_id], env1)
    print()

    note(
        "3. A duplicate/out-of-order request is refused before it ever reaches the",
        "   graph -- this is the state check in RunService, not the ledger; we have",
        "   not approved yet, so signing is not a valid operation on this run.",
    )
    run(["sign", run_id, "--document-id", "doc_too_early"], env1)
    print()

    note(
        "4. Approve. This revokes access and sends the exit paperwork -- the two",
        "   side-effecting steps -- then pauses again waiting for the signed document.",
    )
    run(
        ["approve", run_id, "--approver", "priya.raman", "--note", "Confirmed, last day 9/30."],
        env1,
    )
    print()

    note(
        "5. Sign. Note the PID above and the PID below: every command in this",
        "   script -- start, trace, approve, sign -- runs as its own OS process.",
        f"   Nothing about run {run_id} lived in memory between any of them; every",
        "   fact RunService needed came back from $OFFBOARDING_DB_PATH alone.",
    )
    run(["sign", run_id, "--document-id", "doc_final_signed"], env1)
    print()

    note(
        "6. Full trace. Attempt #2 on both pause gates (interrupt() re-runs the",
        "   node from the top on every resume) -- but the side-effecting steps",
        "   below them still show exactly ONE completed attempt each.",
    )
    run(["trace", run_id], env1)
    print()

    note(
        "7. The idempotency ledger: one row per side effect, each fired exactly",
        "   once, regardless of which process was alive when it happened.",
    )
    run(["side-effects", run_id], env1)

    # ======================================================================
    # Part 2 -- a real, forceful kill, on the one process that's actually
    # long-running
    # ======================================================================

    section("PART 2 -- kill a real process, restart, resume")

    env2 = {**base_env, "OFFBOARDING_DB_PATH": str(_workdir / "part2.db")}
    note(
        "A fresh CLI command starts and exits before you could interrupt it -- there",
        "is no PID worth killing mid-flight. The API server is the one long-running",
        "process here, so that's what actually dies below.",
    )
    print()

    note("Starting the API server in the background.")
    server1 = start_server(_workdir / "server1.log", env2)
    print()

    note("Starting a run on it.")
    print('$ POST /runs {"employee_id": "emp-002"}')
    resp = http_post("/runs", {"employee_id": "emp-002"})
    print(json.dumps(resp, indent=4))
    run2_id = resp["run_id"]
    print()

    old_pid = server1.pid
    print(f"$ kill -9 {old_pid}   # Popen.kill(): SIGKILL on POSIX, TerminateProcess on Windows")
    server1.kill()
    server1.wait(timeout=10)
    _server_procs.remove(server1)
    time.sleep(0.5)
    print("$ GET /")
    confirm_dead()
    print()

    note("A brand-new server process, same database file, different PID.")
    server2 = start_server(_workdir / "server2.log", env2)
    print(f"  (was {old_pid} before the kill; genuinely different)")
    print()

    print(f"$ GET /runs/{run2_id}")
    seen = http_get(f"/runs/{run2_id}")
    print(f"  the new process sees it: status={seen['status']!r} pause_reason={seen.get('pause_reason')!r}")
    print()

    note("Finish it, on the process that did not start it.")
    print(f'$ POST /runs/{run2_id}/approve {{"approver": "aisha.bakr"}}')
    http_post(f"/runs/{run2_id}/approve", {"approver": "aisha.bakr"})
    print(f'$ POST /runs/{run2_id}/sign {{"document_id": "doc_killed_and_restarted"}}')
    final = http_post(f"/runs/{run2_id}/sign", {"document_id": "doc_killed_and_restarted"})
    print(f"  final status: {final['status']}")

    server2.kill()
    server2.wait(timeout=10)
    _server_procs.remove(server2)

    # ======================================================================
    # Part 3 -- retry behaviour, triggered by config alone
    # ======================================================================

    section("PART 3 -- retry-then-succeed, no code change")

    env3 = {
        **base_env,
        "OFFBOARDING_DB_PATH": str(_workdir / "part3.db"),
        "OFFBOARDING_IAM_FAIL_TIMES": "2",
    }
    note("Make the IAM tool fail transiently twice before succeeding.")
    print("$ set OFFBOARDING_IAM_FAIL_TIMES=2" if IS_WINDOWS else "$ export OFFBOARDING_IAM_FAIL_TIMES=2")
    print()

    run(["start", "emp-003"], env3)
    run3_id = latest_run_id(env3)
    run(["approve", run3_id, "--approver", "tom.becker"], env3)
    print()

    note(
        "Attempts 1-2 fail transiently and are retried with backoff; attempt 3",
        "succeeds. Same idempotency key throughout -- one ledger row, not three.",
    )
    _, trace_out = run(["trace", run3_id], env3)
    print("(revoke_access rows only, from the trace above:)")
    for line in trace_out.splitlines():
        if "revoke_access" in line:
            print(line)


if __name__ == "__main__":
    main()
