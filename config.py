"""
Provider configuration — every credential comes from environment variables,
and NOTHING here hardcodes Anthropic's endpoint. Each role resolves to a
(api_key, base_url, model) triple, so any Anthropic-SDK-compatible provider
(Anthropic itself, a proxy, OpenRouter-style gateways, local servers) works
per-role.

Fixed roles read exact variables:
    SENIOR_API_KEY    SENIOR_BASE_URL    SENIOR_MODEL
    REVIEWER_API_KEY  REVIEWER_BASE_URL  REVIEWER_MODEL

Workers draw from a POOL of numbered slots — define one slot per key you
want balanced (we run four):
    WORKER_1_API_KEY  WORKER_1_BASE_URL  WORKER_1_MODEL
    WORKER_2_API_KEY  ...
    WORKER_3_API_KEY  ...
    WORKER_4_API_KEY  ...
(Legacy WORKER_A_* / WORKER_B_* prefixes are picked up too.)

Fallbacks for anything unset: ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL /
ORCH_MODEL. An unset *_BASE_URL means "let the SDK use its default", an
unset key with no fallback is a hard startup error — never an empty string
silently sent to a server.
"""
import os
import re
from dataclasses import dataclass


def _load_dotenv() -> None:
    """Minimal .env loader — KEY=VALUE lines, surrounding quotes stripped,
    blank lines and # comments ignored. Values already present in the real
    environment always win (setdefault), so shell exports override the file.
    Looks for .env next to this file, then in the current directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, ".env"), ".env"):
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                os.environ.setdefault(key.strip(), value)
        break


_load_dotenv()

DEFAULT_MODEL = os.environ.get("ORCH_MODEL", "claude-sonnet-4-6")

REPO_ROOT = os.environ.get("ORCH_REPO_ROOT", "./project_repo")
WORKTREE_ROOT = os.environ.get("ORCH_WORKTREE_ROOT", "./worktrees")

MAX_REASSIGN_ATTEMPTS = 3   # hard cap so a broken worker doesn't loop forever
MAX_WORKERS = int(os.environ.get("ORCH_MAX_WORKERS", "8"))  # upper bound on plan size
# How many workers run at once. Default 4 matches the typical key pool; set 1
# to force fully sequential, or higher to fan out more. Each concurrent worker
# uses its own worktree + key slot, so real concurrency needs enough key slots.
PARALLELISM = int(os.environ.get("ORCH_PARALLELISM", "4"))
# Escalation (#4): the model a worker's FINAL attempt is retried on, after it
# has already failed with its normal model. Same provider/key, stronger brain.
# Empty/unset -> no escalation (final attempt uses the worker's normal model).
ESCALATION_MODEL = os.environ.get("ORCH_ESCALATION_MODEL", "").strip() or None


@dataclass(frozen=True)
class Provider:
    name: str            # env prefix this was loaded from (safe to print)
    api_key: str         # NEVER printed or logged
    base_url: str | None  # None -> SDK default endpoint
    model: str


def _load(prefix: str) -> Provider | None:
    key = os.environ.get(f"{prefix}_API_KEY")
    if not key:
        return None
    return Provider(
        name=prefix.lower(),
        api_key=key,
        base_url=os.environ.get(f"{prefix}_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL") or None,
        model=os.environ.get(f"{prefix}_MODEL", DEFAULT_MODEL),
    )


def _fallback(role: str) -> Provider:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            f"No API key for role '{role}': set {role.upper()}_API_KEY "
            f"(and optionally {role.upper()}_BASE_URL) or ANTHROPIC_API_KEY."
        )
    return Provider(
        name=f"{role}-fallback",
        api_key=key,
        base_url=os.environ.get("ANTHROPIC_BASE_URL") or None,
        model=DEFAULT_MODEL,
    )


def worker_pool() -> list[Provider]:
    """All WORKER_*_API_KEY slots found in the environment, in sorted order
    so key routing is deterministic run-to-run."""
    prefixes = sorted(
        m.group(1)
        for name in os.environ
        if (m := re.match(r"^(WORKER_[A-Z0-9]+)_API_KEY$", name))
    )
    return [p for p in (_load(pref) for pref in prefixes) if p]


# owner name from the Senior's plan -> Provider. Populated once per run by
# assign_worker_providers(); provider_for() refuses to guess before that.
_ASSIGNMENTS: dict[str, Provider] = {}


def assign_worker_providers(owners: list[str]) -> dict[str, Provider]:
    """Round-robins the worker key pool across the plan's owners, in plan
    order. With 4 slots and N subtasks, keys balance as evenly as possible.
    Must be called after plan validation and before any worker call."""
    pool = worker_pool()
    if not pool:
        pool = [_fallback("worker")]
    _ASSIGNMENTS.clear()
    for i, owner in enumerate(dict.fromkeys(owners)):  # dedupe, keep plan order
        _ASSIGNMENTS[owner] = pool[i % len(pool)]
    return dict(_ASSIGNMENTS)


def provider_for(role: str) -> Provider:
    if role == "senior":
        return _load("SENIOR") or _fallback("senior")
    if role == "reviewer":
        return _load("REVIEWER") or _fallback("reviewer")
    if role in _ASSIGNMENTS:
        return _ASSIGNMENTS[role]
    raise KeyError(
        f"No provider assigned for worker '{role}' — assign_worker_providers() "
        f"must run (after plan validation) before any worker is called."
    )
