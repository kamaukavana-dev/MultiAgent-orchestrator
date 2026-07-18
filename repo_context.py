"""
Read-only repo context for the roles.

The Senior plans against the repo's REAL file tree and current contents, and
every worker receives the CURRENT contents of the files it owns (plus any
read-only context files the Senior assigns), so nobody codes against an
imagined codebase — the exact failure mode of planning 'com/example/auth'
paths into a repo whose real package is something else entirely.

Everything in this module is strictly read-only. Workers still change files
ONLY through the orchestrator's scope-checked write path; showing a worker a
context file grants zero write access to it.
"""
import os
import subprocess

# Noise that wastes context budget without informing anyone's plan.
SKIP_BASENAMES = {
    "mvnw", "mvnw.cmd", "gradlew", "gradlew.bat",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "Cargo.lock", ".gitattributes",
}
SKIP_EXTS = {
    ".jar", ".class", ".pyc", ".png", ".jpg", ".jpeg", ".gif", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".woff", ".woff2", ".ttf", ".eot",
}

FILE_CHAR_CAP = 12_000        # per-file cap; big files get truncated, not dropped
SNAPSHOT_CHAR_BUDGET = 60_000  # total budget for the Senior's whole-repo snapshot


def tracked_files(repo_root: str) -> list[str]:
    r = subprocess.run(["git", "ls-files"], cwd=repo_root, capture_output=True, text=True)
    return [f for f in r.stdout.splitlines() if f]


def _wanted(rel_path: str) -> bool:
    base = os.path.basename(rel_path)
    if base in SKIP_BASENAMES:
        return False
    return os.path.splitext(base)[1].lower() not in SKIP_EXTS


def read_file(root: str, rel_path: str, cap: int = FILE_CHAR_CAP) -> str | None:
    """Returns the file's text (truncated at `cap` chars), or None if it does
    not exist or is unreadable. errors='replace' keeps a stray non-UTF8 byte
    from killing the whole context build."""
    path = os.path.join(root, rel_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read(cap + 1)
    except OSError:
        return None
    if len(text) > cap:
        text = text[:cap] + "\n... [truncated]"
    return text


def repo_snapshot(repo_root: str, budget: int = SNAPSHOT_CHAR_BUDGET) -> str:
    """Full-repo view for the Senior: the real tree, then file contents until
    the budget runs out. Files past the budget are still LISTED (in the tree)
    so plans can scope them — they just aren't inlined."""
    files = tracked_files(repo_root)
    parts = ["REPO FILE TREE (every git-tracked file — use these REAL paths):"]
    parts += [f"  {f}" for f in files]
    parts.append("\nCURRENT FILE CONTENTS:")
    used = sum(len(p) + 1 for p in parts)
    omitted = []
    for f in files:
        if not _wanted(f):
            continue
        text = read_file(repo_root, f)
        if text is None:
            continue
        block = f"\n--- {f} ---\n{text}"
        if used + len(block) > budget:
            omitted.append(f)
            continue
        parts.append(block)
        used += len(block)
    if omitted:
        parts.append(f"\n[content omitted for budget, tree still authoritative: {', '.join(omitted)}]")
    return "\n".join(parts)


def worker_context(worktree_path: str, file_scope: list[str], context_files: list[str]) -> str:
    """Per-worker view, built from the worker's own worktree (== current main
    at attempt start): the repo tree, the CURRENT contents of every file the
    worker owns, and the Senior-assigned read-only context files."""
    parts = ["REPO FILE TREE (your branch already contains all of these):"]
    parts += [f"  {f}" for f in tracked_files(worktree_path)]

    parts.append("\nCURRENT CONTENTS OF YOUR ASSIGNED FILES "
                 "(return the ENTIRE new content for any you change):")
    for f in file_scope:
        text = read_file(worktree_path, f)
        parts.append(f"\n--- {f} ---\n{text if text is not None else '[does not exist yet — you will create it]'}")

    if context_files:
        parts.append("\nREAD-ONLY CONTEXT FILES (match these interfaces/conventions exactly; "
                     "you may NOT modify them):")
        for f in context_files:
            text = read_file(worktree_path, f)
            if text is not None:
                parts.append(f"\n--- {f} ---\n{text}")
    return "\n".join(parts)
