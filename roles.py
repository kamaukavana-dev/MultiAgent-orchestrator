"""
Senior / Worker / Reviewer — each is just a system prompt + a Provider
(api_key + base_url + model). The strictness lives in orchestrator.py's
control flow and guards.py's checks, NOT in how sternly these prompts are
worded. Wording the Reviewer's prompt as "be very strict" does nothing on
its own — what makes it strict is that orchestrator.py only trusts the
parsed test output, never the model's prose.

Clients are built per-role from config.provider_for(): the api_key AND the
base_url both come from environment variables, so any Anthropic-SDK-
compatible provider works. Nothing here hardcodes Anthropic's endpoint —
when a provider has no base_url configured, the argument is omitted and
the SDK falls back to its own default.
"""
import json
import time
import anthropic
from config import provider_for

SENIOR_SYSTEM = """You are the Senior Engineer. You do not write code.
You decompose a goal into between 1 and {max_workers} subtasks with STRICTLY
NON-OVERLAPPING file scopes: no two subtasks may list the same file, and no
subtask's scope entry may be a parent directory of another subtask's entry.
Use as many subtasks as the goal genuinely needs — no more.
Each subtask's "owner" must be a short slug like "worker_1", "worker_2", ...
Respond ONLY with JSON, no prose, no markdown fences:
{{
  "subtasks": [
    {{"id": "task_1", "owner": "worker_1", "description": "...", "file_scope": ["path/to/file.py"]}},
    {{"id": "task_2", "owner": "worker_2", "description": "...", "file_scope": ["path/to/other.py"]}}
  ]
}}
"""

WORKER_SYSTEM = """You are a Worker Engineer. You will be given ONE subtask and an
exact list of files you are allowed to create or modify. You must NOT touch any
file outside that list. Write complete, working file contents, INCLUDING real
tests inside your allowed files — submissions whose test run executes zero
tests are automatically rejected by the orchestrator.
Respond ONLY with JSON, no prose, no markdown fences:
{
  "files": {"path/to/file.py": "<full file content>"},
  "notes": "brief note on what you did"
}
"""

REVIEWER_SYSTEM = """You are the Reviewer. You will be given a diff and REAL test
output (exit code + stdout/stderr) that was captured by the orchestrator, not
self-reported by the worker. You do not re-run anything and you do not trust
claims of success that aren't backed by the test output shown to you.
Respond ONLY with JSON, no prose, no markdown fences:
{"verdict": "approve" or "reject", "reason": "..."}
"""


API_MAX_RETRIES = 5
API_TIMEOUT_S = 300.0


def _call(role_key: str, system: str, user_content: str) -> str:
    provider = provider_for(role_key)
    # base_url is passed dynamically — only when configured, never hardcoded.
    client_kwargs = {"api_key": provider.api_key, "timeout": API_TIMEOUT_S, "max_retries": 0}
    if provider.base_url:
        client_kwargs["base_url"] = provider.base_url
    client = anthropic.Anthropic(**client_kwargs)

    # Retry transient failures (5xx, 429, timeouts, connection drops) with
    # exponential backoff. Auth/permission/bad-request errors fail fast.
    last_exc: Exception | None = None
    for attempt in range(API_MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=provider.model,
                max_tokens=4000,
                system=system,
                messages=[{"role": "user", "content": user_content}],
            )
            return "".join(b.text for b in resp.content if b.type == "text")
        except (anthropic.InternalServerError, anthropic.RateLimitError,
                anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            last_exc = e
            delay = min(2 ** attempt * 2, 30)
            print(f"  [api] transient error for {role_key} ({type(e).__name__}), "
                  f"retry {attempt + 1}/{API_MAX_RETRIES} in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"API call for '{role_key}' failed after {API_MAX_RETRIES} retries") from last_exc


def _parse_json(raw: str) -> dict:
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned)


def senior_decompose(goal: str, max_workers: int) -> dict:
    system = SENIOR_SYSTEM.format(max_workers=max_workers)
    raw = _call("senior", system, f"Goal: {goal}")
    return _parse_json(raw)


def worker_implement(owner: str, description: str, file_scope: list[str], rejection_reason: str = "") -> dict:
    prompt = f"Subtask: {description}\nAllowed files ONLY: {file_scope}"
    if rejection_reason:
        prompt += f"\n\nPREVIOUS ATTEMPT WAS REJECTED. Reason: {rejection_reason}\nFix this and resubmit."
    raw = _call(owner, WORKER_SYSTEM, prompt)
    return _parse_json(raw)


def reviewer_verdict(diff_text: str, test_exit_code: int, test_output: str, scope_violations: list[str]) -> dict:
    if scope_violations:
        # Don't even bother calling the model — this is an automatic, non-negotiable reject.
        return {"verdict": "reject", "reason": f"Scope violation, touched files outside assignment: {scope_violations}"}
    prompt = (
        f"Diff:\n{diff_text}\n\n"
        f"Test exit code (0 = pass): {test_exit_code}\n"
        f"Test output:\n{test_output}"
    )
    raw = _call("reviewer", REVIEWER_SYSTEM, prompt)
    return _parse_json(raw)
