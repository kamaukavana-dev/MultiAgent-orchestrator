"""
Main loop. Run this file directly.

The chain of custody that matters:
  Senior emits a plan --> guards.validate_plan() rejects any scope overlap
  BEFORE a single worker is called (pre-flight exclusivity check)
  --> worker claims "done" --> orchestrator RUNS THE ACTUAL TEST COMMAND
  --> guards.verify_test_run() requires exit code 0 AND >0 tests executed
  (an empty suite that exits 0 is an automatic reject)
  --> orchestrator checks the ACTUAL git diff for scope violations
  --> only THEN does the Reviewer see real evidence, never the worker's prose
  --> approved branches merge one at a time, never concurrently

If you remember one thing from this file: nowhere does the orchestrator
accept "I finished, tests pass" as fact. It always re-derives that from
a subprocess exit code AND a parsed test count.
"""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import git_ops
import guards
import repo_context
import roles
from state import TaskGraph, Subtask


def write_worker_files(worktree_path: str, files: dict[str, str]):
    """Writes worker output into its worktree — and ONLY its worktree.
    A path that is absolute or resolves outside the worktree (../ traversal)
    would land outside git's view and never appear in the scope-check diff,
    silently bypassing the whole guard chain. Hard error, not a skip."""
    root = os.path.realpath(worktree_path)
    for rel_path, content in files.items():
        full_path = os.path.realpath(os.path.join(root, rel_path))
        if not (full_path == root or full_path.startswith(root + os.sep)):
            raise RuntimeError(f"worker path escapes its worktree: {rel_path!r}")
        os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)


def run_tests(worktree_path: str, test_cmd: list[str]) -> tuple[int, str]:
    """Runs the REAL test command and returns (exit_code, combined output).
    Pass/fail is decided by guards.verify_test_run() on this output — never
    by the exit code alone."""
    result = subprocess.run(test_cmd, cwd=worktree_path, capture_output=True, text=True)
    return result.returncode, (result.stdout + "\n" + result.stderr)


def _feedback_text(reason: str, required_changes: list[str]) -> str:
    """Combines the reviewer's reason + itemized required changes into the
    single string the worker sees on its next attempt."""
    parts = []
    if reason:
        parts.append(reason)
    if required_changes:
        parts.append("Required changes:\n" + "\n".join(f"  - {c}" for c in required_changes))
    return "\n".join(parts) or "unspecified"


def process_subtask(graph: TaskGraph, subtask: Subtask, test_cmd: list[str]) -> bool:
    """Runs one subtask through implement -> test -> review, looping on the
    reviewer's 'revise' verdict with concrete required changes fed back to the
    same worker. The FINAL attempt escalates to a stronger model (config.
    ESCALATION_MODEL) if one is configured. Returns True iff approved."""

    while subtask.attempts < config.MAX_REASSIGN_ATTEMPTS:
        subtask.attempts += 1
        subtask.status = "in_progress"
        # Escalation (#4): on the last allowed attempt, hand the work to a
        # stronger model instead of burning the try on the same one that has
        # already failed twice. Rescues the exact 3-strikes death we saw.
        is_last = subtask.attempts == config.MAX_REASSIGN_ATTEMPTS
        model_override = config.ESCALATION_MODEL if (is_last and config.ESCALATION_MODEL) else None
        subtask.log("assigned", f"attempt {subtask.attempts}"
                    + (f" [escalated -> {model_override}]" if model_override else ""))
        if model_override:
            print(f"[{subtask.id}] ESCALATING final attempt to model '{model_override}'.")

        branch = f"{subtask.owner}/{subtask.id}"
        subtask.branch = branch
        worktree_path = git_ops.create_worker_worktree(config.REPO_ROOT, config.WORKTREE_ROOT, branch)

        # --- Worker implements, seeing the CURRENT contents of its own files
        # (so it modifies rather than blindly overwrites) plus any read-only
        # context files the Senior assigned. Built from the fresh worktree,
        # which is current main at attempt start. ---
        context = repo_context.worker_context(
            worktree_path, subtask.file_scope, subtask.context_files,
        )
        result = roles.worker_implement(
            subtask.owner, subtask.description, subtask.file_scope,
            context=context,
            rejection_reason=subtask.last_rejection_reason,
            model_override=model_override,
        )
        write_worker_files(worktree_path, result.get("files", {}))
        git_ops.commit_worker_changes(worktree_path, f"{subtask.id}: {result.get('notes','')}")
        subtask.log("implemented", result.get("notes", ""))

        # --- Real test run (never trust worker's self-report) ---
        exit_code, test_output = run_tests(worktree_path, test_cmd)
        tests_ok, tests_reason = guards.verify_test_run(exit_code, test_output)
        subtask.log("tested", f"exit_code={exit_code} verdict={tests_reason}")

        # --- Scope violation check on the REAL diff (needed for the reviewer) ---
        violations = git_ops.check_scope_violation(config.REPO_ROOT, branch, subtask.file_scope)
        diff_text = git_ops.get_diff_text(config.REPO_ROOT, branch)

        # The reviewer now sees the SUBTASK, the repo CONTEXT, the real diff, and
        # the real test result together — so it can judge completeness/wiring, not
        # just "did it compile". A failing test run is passed through as context
        # rather than short-circuiting, so the reviewer states WHAT to fix.
        verdict = roles.reviewer_verdict(
            subtask.description, diff_text, exit_code, test_output, violations,
            context=context,
        )
        # Belt-and-suspenders: tests must pass to APPROVE, no matter what the
        # reviewer says. A green-looking review over a red test run downgrades
        # to 'revise' — the empty-suite / exit-code guard is never overridden.
        if not tests_ok and verdict.get("verdict") == "approve":
            verdict = {"verdict": "revise",
                       "reason": f"Test verification failed: {tests_reason}",
                       "required_changes": [f"Make the test run pass and execute >0 tests. {tests_reason}"]}
        subtask.status = "reviewing"
        subtask.log("reviewed", str(verdict))

        v = verdict.get("verdict")
        if v == "approve":
            subtask.status = "approved"
            print(f"[{subtask.id}] APPROVED ({tests_reason}).")
            return True

        # 'revise' and 'reject' both loop back to the same worker with concrete
        # feedback. 'reject' additionally means the approach was wrong, but the
        # worker still gets the reasoning and another attempt within the cap.
        feedback = _feedback_text(verdict.get("reason", ""), verdict.get("required_changes", []))
        if not tests_ok:
            feedback += f"\n\nTest output (tail):\n{test_output[-1500:]}"
        subtask.status = "rejected"
        subtask.last_rejection_reason = feedback
        subtask.log(v or "rejected", feedback[:400])
        print(f"[{subtask.id}] {(v or 'REJECTED').upper()} — {verdict.get('reason','')[:160]} "
              f"(attempt {subtask.attempts}/{config.MAX_REASSIGN_ATTEMPTS}).")

    subtask.status = "failed"
    print(f"[{subtask.id}] FAILED after {config.MAX_REASSIGN_ATTEMPTS} attempts. Needs human review.")
    return False


def run(goal: str, test_cmd: list[str]):
    if not test_cmd:
        # No test command means no verifiable evidence — refuse to run rather
        # than silently treat "nothing happened" as success.
        raise SystemExit("TEST_CMD is required (e.g. export TEST_CMD='pytest'). "
                         "Refusing to run without a verifiable test command.")

    git_ops.ensure_repo_initialized(config.REPO_ROOT)

    print(f"=== Senior decomposing goal ===\n{goal}\n")

    # Snapshot the REAL repo (tree + current contents) so the Senior plans
    # against actual paths and package names, never an imagined layout.
    snapshot = repo_context.repo_snapshot(config.REPO_ROOT)

    # --- PRE-FLIGHT: validate the plan BEFORE any worker is called. A rejected
    # plan is fed back to the Senior with the exact problems, up to the same
    # attempt cap workers get; only a still-broken plan aborts the run. ---
    plan = None
    feedback = ""
    for plan_attempt in range(1, config.MAX_REASSIGN_ATTEMPTS + 1):
        candidate = roles.senior_decompose(goal, max_workers=config.MAX_WORKERS,
                                           repo_snapshot=snapshot, plan_feedback=feedback)
        problems = guards.validate_plan(candidate)
        if isinstance(candidate, dict) and len(candidate.get("subtasks") or []) > config.MAX_WORKERS:
            problems.append(f"plan has {len(candidate['subtasks'])} subtasks, max is {config.MAX_WORKERS}")
        if not problems:
            plan = candidate
            break
        for p in problems:
            print(f"  PLAN REJECTED: {p}")
        feedback = "\n".join(problems)
        print(f"  Re-asking Senior with the validation errors "
              f"(attempt {plan_attempt}/{config.MAX_REASSIGN_ATTEMPTS}).")
    if plan is None:
        raise SystemExit("Senior's plan failed pre-flight validation (scope overlap "
                         "or malformed structure) on every attempt. Aborting before any worker runs.")

    # Dynamic 1..N subtasks — however many the validated plan contains.
    # context_files is optional and read-only, so a missing/malformed value
    # degrades to "no extra context" rather than failing the plan.
    subtasks = [Subtask(id=t["id"], owner=t["owner"], description=t["description"],
                        file_scope=t["file_scope"],
                        context_files=[c for c in (t.get("context_files") or []) if isinstance(c, str)])
                for t in plan["subtasks"]]
    graph = TaskGraph(goal=goal, subtasks=subtasks)
    graph.save("task_graph.json")

    # Balance the worker API-key pool across the plan's owners (round-robin).
    assignments = config.assign_worker_providers([s.owner for s in subtasks])
    print(f"Plan accepted: {len(subtasks)} subtask(s), scopes strictly exclusive.")
    for s in subtasks:
        prov = assignments[s.owner]
        print(f"  - {s.id} -> {s.owner} [key slot: {prov.name}]: {s.description}  scope={s.file_scope}")

    # Process each subtask (implement/test/review/reassign) BEFORE any merging.
    # Workers are physically isolated — own worktree, own branch, own API-key
    # slot — so they run CONCURRENTLY (this is the "several models at once").
    # Only merging stays strictly sequential. ORCH_PARALLELISM caps how many
    # run at a time (default = number of subtasks, i.e. all at once).
    approved = []
    if config.PARALLELISM <= 1 or len(graph.subtasks) <= 1:
        for subtask in graph.subtasks:
            if process_subtask(graph, subtask, test_cmd):
                approved.append(subtask)
            graph.save("task_graph.json")
    else:
        max_workers = min(config.PARALLELISM, len(graph.subtasks))
        print(f"Running {len(graph.subtasks)} workers concurrently "
              f"(up to {max_workers} at a time).")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(process_subtask, graph, s, test_cmd): s
                       for s in graph.subtasks}
            for fut in as_completed(futures):
                subtask = futures[fut]
                try:
                    if fut.result():
                        approved.append(subtask)
                except Exception as e:  # a worker crash must not sink the whole run
                    subtask.status = "failed"
                    subtask.log("error", f"{type(e).__name__}: {e}")
                    print(f"[{subtask.id}] ERRORED — {type(e).__name__}: {e}")
                graph.save("task_graph.json")
        # Merge in the plan's original order, not the order they happened to finish.
        approved.sort(key=lambda s: [x.id for x in graph.subtasks].index(s.id))

    # Sequential merge — one at a time, in the order approved, never concurrent.
    for subtask in approved:
        result = git_ops.merge_branch_to_main(config.REPO_ROOT, subtask.branch)
        if result.returncode != 0:
            print(f"[{subtask.id}] MERGE CONFLICT despite scope enforcement — inspect manually: {result.stderr}")
            subtask.status = "merge_failed"
        else:
            subtask.status = "merged"
            print(f"[{subtask.id}] merged to main cleanly.")
        graph.merge_order.append(subtask.id)
        graph.save("task_graph.json")

    # --- FINAL INTEGRATION GATE: per-branch tests ran in isolation, so cross-
    # subtask interface drift (worker A exporting sub() while worker B calls
    # subtract()) only becomes observable here, on the merged whole. Same rule
    # as everywhere else: the verdict comes from a real subprocess run.
    if approved:
        exit_code, test_output = run_tests(config.REPO_ROOT, test_cmd)
        ok, reason = guards.verify_test_run(exit_code, test_output)
        print(f"\n=== Integration test on merged main: "
              f"{'PASS' if ok else 'FAIL'} ({reason}) ===")
        if not ok:
            print(test_output[-2000:])
            print("Merged branches passed in isolation but the COMBINED suite fails — "
                  "likely an interface mismatch between subtasks. Needs human review.")
            raise SystemExit(1)

    print("\n=== Done. See task_graph.json for full audit trail. ===")


if __name__ == "__main__":
    import sys
    goal = sys.argv[1] if len(sys.argv) > 1 else "Build a simple two-file Python CLI: core logic module + CLI entrypoint module."
    test_cmd = os.environ.get("TEST_CMD", "").split()
    run(goal, test_cmd)
