"""
Git operations.

Two jobs this file does, and they are the entire reason merge conflicts
get eliminated instead of just "hopefully avoided":

1. Each worker gets an isolated `git worktree` on its own branch, so two
   workers never touch the same working directory at the same time.
2. Before any merge, we diff the worker's branch against the base and
   check every changed file is inside the worker's declared file_scope.
   If it isn't -> automatic reject. No exceptions. This is what makes
   "strict" real instead of a personality trait in a prompt.

Merges to main happen ONE AT A TIME, sequentially, controlled by the
Senior's merge_order — never concurrently.
"""
import subprocess
import os


def run(cmd: list[str], cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def ensure_repo_initialized(repo_root: str):
    if not os.path.isdir(os.path.join(repo_root, ".git")):
        os.makedirs(repo_root, exist_ok=True)
        run(["git", "init"], cwd=repo_root)
    # Ignore runtime artifacts — otherwise pytest's __pycache__ gets committed
    # and every subtask trips the scope check on .pyc files. Checked on EVERY
    # run, not just fresh init: a repo initialized before this guard existed
    # (or with a hand-rolled .gitignore missing these entries) would otherwise
    # fail every subtask forever.
    gitignore_path = os.path.join(repo_root, ".gitignore")
    required = ["__pycache__/", "*.pyc", ".pytest_cache/", "node_modules/"]
    existing = ""
    if os.path.isfile(gitignore_path):
        with open(gitignore_path) as f:
            existing = f.read()
    missing = [r for r in required if r not in existing.splitlines()]
    if missing:
        with open(gitignore_path, "a") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write("\n".join(missing) + "\n")
        run(["git", "add", ".gitignore"], cwd=repo_root)
        run(["git", "commit", "-m", "chore: ignore runtime artifacts"], cwd=repo_root)


def create_worker_worktree(repo_root: str, worktree_root: str, branch: str) -> str:
    """Creates a new branch off main and checks it out into its own directory.
    Paths are made absolute — git resolves relative paths against ITS cwd
    (repo_root), while callers resolve them against the process cwd; mixing
    the two silently splits the worktree from where files get written."""
    worktree_root = os.path.abspath(worktree_root)
    os.makedirs(worktree_root, exist_ok=True)
    path = os.path.join(worktree_root, branch)
    if os.path.exists(os.path.join(path, ".git")):  # .git is a FILE in a linked worktree
        # Reused on reassignment: reset to main so leftovers from the previous
        # rejected attempt (including out-of-scope files) don't contaminate
        # this attempt's diff.
        run(["git", "reset", "--hard", "main"], cwd=path)
        run(["git", "clean", "-fdx"], cwd=path)
        return path
    result = run(["git", "worktree", "add", "-b", branch, path, "main"], cwd=repo_root)
    if result.returncode != 0:
        # main might not exist yet on first run
        run(["git", "branch", "-M", "main"], cwd=repo_root)
        result = run(["git", "worktree", "add", "-b", branch, path, "main"], cwd=repo_root)
        if result.returncode != 0:
            raise RuntimeError(f"worktree add failed: {result.stderr}")
    return path


def commit_worker_changes(worktree_path: str, message: str) -> str:
    """Stages and commits everything in the worktree, then VERIFIES the commit
    exists. A commit that silently fails would make the branch identical to
    main — empty diff, vacuously clean scope check, no-op merge — so failure
    here is a hard error, not a warning. Returns the commit sha."""
    run(["git", "add", "-A"], cwd=worktree_path)
    result = run(["git", "commit", "-m", message], cwd=worktree_path)
    if result.returncode != 0 and "nothing to commit" not in (result.stdout + result.stderr):
        raise RuntimeError(f"commit failed in {worktree_path}: {result.stderr or result.stdout}")
    head = run(["git", "rev-parse", "HEAD"], cwd=worktree_path)
    if head.returncode != 0:
        raise RuntimeError(f"{worktree_path} is not a git worktree: {head.stderr}")
    return head.stdout.strip()


def changed_files(repo_root: str, branch: str, base: str = "main") -> list[str]:
    result = run(["git", "diff", "--name-only", f"{base}...{branch}"], cwd=repo_root)
    return [f for f in result.stdout.strip().splitlines() if f]


def _norm(path: str) -> str:
    """Canonical repo-relative form. MUST match guards._norm so the pre-flight
    validator and this runtime check agree on what a path is: a plan entry like
    './pkg/mod.py' validates fine there but git diff reports 'pkg/mod.py' here,
    so comparing raw strings would flag a worker's OWN assigned file as a scope
    violation and auto-reject correct work."""
    return os.path.normpath(path.strip()).replace("\\", "/")


def check_scope_violation(repo_root: str, branch: str, allowed_scope: list[str]) -> list[str]:
    """Returns list of files touched OUTSIDE the allowed scope. Empty list = clean.
    Both sides are normalized so './pkg/x', 'pkg/x/' and 'pkg/x' compare equal."""
    allowed = [_norm(a) for a in allowed_scope]
    violations = []
    for f in changed_files(repo_root, branch):
        nf = _norm(f)
        if not any(nf == a or nf.startswith(a.rstrip("/") + "/") for a in allowed):
            violations.append(f)
    return violations


def merge_branch_to_main(repo_root: str, branch: str) -> subprocess.CompletedProcess:
    """Sequential merge — call this one branch at a time, never in parallel."""
    run(["git", "checkout", "main"], cwd=repo_root)
    return run(["git", "merge", "--no-ff", branch, "-m", f"merge: {branch}"], cwd=repo_root)


def get_diff_text(repo_root: str, branch: str, base: str = "main") -> str:
    result = run(["git", "diff", f"{base}...{branch}"], cwd=repo_root)
    return result.stdout