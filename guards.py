"""
Code-enforced guardrails. Every rule in this file is a hard, programmatic
check — no model prose, no tone, no trust.

Guard 1 — verify_test_run(): exit code 0 is NOT sufficient. Many runners
  exit 0 while executing zero tests (unittest "Ran 0 tests", go test with no
  test files, jest --passWithNoTests, a bare shell command). We regex-parse
  the runner's output and require the number of EXECUTED tests to be > 0.
  Unparseable output is treated as failure: unverifiable == rejected.

Guard 2 — validate_plan(): the Senior's JSON plan is structurally validated
  and its file scopes are checked for strict exclusivity BEFORE any worker
  is called. Any overlap (same file, or one entry being a parent directory
  of another subtask's entry) rejects the whole plan.
"""
import os.path
import re


# --------------------------------------------------------------------------
# Guard 1: strict test-output verification (no empty-suite bypass)
# --------------------------------------------------------------------------

def executed_test_count(output: str) -> int | None:
    """Parses test-runner output and returns the number of tests actually
    executed, or None if no known runner summary could be found.

    Recognizes: pytest, unittest, jest/vitest, mocha, cargo test, TAP.
    When several summaries match (e.g. pytest wrapping unittest), the
    largest count wins — we only care whether it is strictly > 0.
    """
    counts: list[int] = []

    # pytest: "== 2 passed, 1 failed, 1 error in 0.12s ==" (any subset of parts)
    passed = re.search(r"(\d+) passed", output)
    failed = re.search(r"(\d+) failed", output)
    errors = re.search(r"(\d+) error", output)
    if passed or failed or errors:
        counts.append(sum(int(m.group(1)) for m in (passed, failed, errors) if m))

    # unittest: "Ran 3 tests in 0.001s" (written to stderr — pass combined output)
    m = re.search(r"^Ran (\d+) tests? in ", output, re.MULTILINE)
    if m:
        counts.append(int(m.group(1)))

    # jest / vitest: "Tests:       3 passed, 3 total"
    m = re.search(r"^\s*Tests:.*?(\d+) total", output, re.MULTILINE)
    if m:
        counts.append(int(m.group(1)))

    # mocha: "  3 passing (12ms)"
    m = re.search(r"^\s*(\d+) passing\b", output, re.MULTILINE)
    if m:
        counts.append(int(m.group(1)))

    # cargo test: "test result: ok. 3 passed; 0 failed; ..."
    m = re.search(r"test result: \w+\. (\d+) passed; (\d+) failed", output)
    if m:
        counts.append(int(m.group(1)) + int(m.group(2)))

    # Maven Surefire/Failsafe: "Tests run: 5, Failures: 0, Errors: 0, Skipped: 1"
    # The summary line repeats per-class and once as a total — max() handles it.
    for m in re.finditer(r"Tests run: (\d+), Failures: \d+, Errors: \d+(?:, Skipped: (\d+))?", output):
        counts.append(int(m.group(1)) - int(m.group(2) or 0))

    # Gradle: "5 tests completed, 1 failed"
    m = re.search(r"(\d+) tests? completed", output)
    if m:
        counts.append(int(m.group(1)))

    # TAP: "# tests 3"
    m = re.search(r"^# tests (\d+)", output, re.MULTILINE)
    if m:
        counts.append(int(m.group(1)))

    if counts:
        return max(counts)

    # Explicit "nothing ran" markers with no numeric summary at all.
    if re.search(r"no tests ran|collected 0 items|no test files", output, re.IGNORECASE):
        return 0

    return None


def verify_test_run(exit_code: int, output: str) -> tuple[bool, str]:
    """The single decision point for 'did the tests really pass'.
    Returns (ok, reason). ok requires BOTH exit code 0 AND a parsed
    executed-test count strictly greater than 0."""
    if exit_code != 0:
        return False, f"test command exited {exit_code}"
    n = executed_test_count(output)
    if n is None:
        return False, ("exit code was 0 but no test count could be parsed from the "
                       "output — unverifiable runs are rejected (empty-suite bypass guard)")
    if n == 0:
        return False, "exit code was 0 but 0 tests were executed — empty suite is an automatic reject"
    return True, f"{n} test(s) executed, exit code 0"


# --------------------------------------------------------------------------
# Guard 2: Senior plan validation + pre-flight scope exclusivity
# --------------------------------------------------------------------------

_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")  # ids become git branch components


def _norm(path: str) -> str:
    return os.path.normpath(path.strip()).replace("\\", "/")


def _covers(a: str, b: str) -> bool:
    """True if scope entry `a` covers path `b` — same path, or `a` is a
    parent directory of `b`. Mirrors git_ops.check_scope_violation semantics."""
    return a == b or b.startswith(a.rstrip("/") + "/")


def find_scope_conflicts(subtasks: list[dict]) -> list[str]:
    """Pairwise-compares every scope entry across DIFFERENT subtasks.
    Any overlap is a conflict. Empty list = scopes are strictly exclusive."""
    entries: list[tuple[str, str]] = []
    for t in subtasks:
        for s in t.get("file_scope") or []:
            entries.append((str(t.get("id")), _norm(str(s))))

    conflicts = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            (id_a, a), (id_b, b) = entries[i], entries[j]
            if id_a == id_b:
                continue  # redundancy within one worker's own scope can't cause a cross-worker conflict
            if _covers(a, b) or _covers(b, a):
                conflicts.append(f"scope overlap between '{id_a}' and '{id_b}': '{a}' vs '{b}'")
    return conflicts


def validate_plan(plan: dict) -> list[str]:
    """Full pre-flight validation of the Senior's plan. Returns a list of
    human-readable problems; empty list = plan accepted. Must be called
    BEFORE any worker is spawned."""
    problems: list[str] = []

    subtasks = plan.get("subtasks") if isinstance(plan, dict) else None
    if not isinstance(subtasks, list) or not subtasks:
        return ["plan has no non-empty 'subtasks' list"]

    seen_ids: set[str] = set()
    for i, t in enumerate(subtasks):
        if not isinstance(t, dict):
            problems.append(f"subtask #{i} is not a JSON object")
            continue
        tid = t.get("id")
        label = tid if isinstance(tid, str) and tid else f"#{i}"
        if not isinstance(tid, str) or not _ID_RE.match(tid or ""):
            problems.append(f"subtask {label}: 'id' must be a slug of [A-Za-z0-9._-] (it becomes a branch name)")
        elif tid in seen_ids:
            problems.append(f"duplicate subtask id '{tid}'")
        else:
            seen_ids.add(tid)

        owner = t.get("owner")
        if not isinstance(owner, str) or not _ID_RE.match(owner or ""):
            problems.append(f"subtask {label}: 'owner' must be a slug of [A-Za-z0-9._-] (it becomes a branch name)")

        if not isinstance(t.get("description"), str) or not t["description"].strip():
            problems.append(f"subtask {label}: missing non-empty 'description'")

        scope = t.get("file_scope")
        if not isinstance(scope, list) or not scope or not all(isinstance(s, str) and s.strip() for s in scope):
            problems.append(f"subtask {label}: 'file_scope' must be a non-empty list of path strings")
            continue
        for s in scope:
            n = _norm(s)
            if os.path.isabs(n) or n == "." or n == ".." or n.startswith("../"):
                problems.append(f"subtask {label}: illegal scope path '{s}' (absolute, or escapes the repo)")

    problems.extend(find_scope_conflicts([t for t in subtasks if isinstance(t, dict)]))
    return problems
