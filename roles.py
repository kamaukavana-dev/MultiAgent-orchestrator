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
You will be shown the repo's REAL file tree and current file contents.
You decompose a goal into between 1 and {max_workers} subtasks with STRICTLY
NON-OVERLAPPING file scopes: no two subtasks may list the same file, and no
subtask's scope entry may be a parent directory of another subtask's entry.
Use as many subtasks as the goal genuinely needs — no more.

Scope paths MUST be real paths from the shown tree (to modify existing files)
or new paths consistent with the repo's actual layout and package/module
naming. NEVER invent a directory structure that contradicts the tree.

Each subtask may also list "context_files": existing repo files the worker
should SEE read-only (interfaces it must match, conventions to follow) but is
NOT allowed to modify. Context files may overlap freely between subtasks —
only file_scope must be exclusive.

CRITICAL: every subtask must be INDEPENDENTLY BUILDABLE AND TESTABLE. Each
worker's branch starts from current main — existing repo files are present,
but files created by OTHER subtasks in this plan will NOT exist yet, and the
FULL build + test suite (not just the new tests) runs on that branch. So a
subtask's files must COMPILE and TEST GREEN against only current main + its
own files.

This has HARD consequences that OVERRIDE how the goal is phrased — obey them
even if the user asked to split things differently:
  - A source file and the tests that exercise it MUST be in the SAME subtask.
    A test cannot compile without the class it tests, so splitting them makes
    a branch that cannot build. NEVER put a class in one subtask and its test
    in another.
  - Everything a subtask needs to COMPILE must be in that subtask or already
    on main. If new code needs a new dependency or build-file change (e.g. a
    new library in pom.xml / build.gradle / package.json / requirements), that
    build-file edit MUST be in the SAME subtask's file_scope. NEVER leave a
    subtask depending on a dependency another subtask is supposed to add.
  - If a shared build file (pom.xml, package.json, ...) would need edits from
    two different features, either put all its edits in ONE subtask that owns
    that file, or fold the features into a single subtask. The build file can
    only be owned by one subtask (scopes are exclusive).
  - A worker must NEVER silently drop a required annotation, dependency, or
    behavior to make a branch compile. If satisfying the goal needs something
    outside its scope, that is a PLANNING error — you must widen the scope so
    the subtask is self-sufficient, not shrink the implementation.

When in doubt, prefer FEWER, SELF-SUFFICIENT subtasks over many fragmented
ones. A single subtask owning [Feature.java, FeatureTest.java, pom.xml] that
builds green beats three that don't.

CRITICAL: when subtasks share a NEW interface (one imports the other after the
final merge), your descriptions must PIN THE EXACT CONTRACT — module names,
function names, signatures, exception types — identically in BOTH subtasks'
descriptions, so independently-working workers cannot drift apart. After all
branches merge, the combined test suite is run as a final integration gate.

Each subtask's "owner" must be a short slug like "worker_1", "worker_2", ...
Respond ONLY with JSON, no prose, no markdown fences:
{{
  "subtasks": [
    {{"id": "task_1", "owner": "worker_1", "description": "...",
      "file_scope": ["real/path/module.py", "real/path/test_module.py"],
      "context_files": ["real/path/existing_interface.py"]}}
  ]
}}
"""

WORKER_SYSTEM = """You are a Worker Engineer. You will be given ONE subtask, the
repo's file tree, the CURRENT contents of the files you own, and possibly some
read-only context files. You must NOT touch any file outside your assigned list.

You have NO tools and NO filesystem access beyond what is shown in your prompt.
Do not announce plans to explore or read anything — everything you get to see
is already in the prompt. The ONLY thing you can do is return complete file
contents in JSON.

For files that already exist, you are MODIFYING them: return the ENTIRE new
file content (not a diff, not a fragment), preserving everything that should
survive your change. Match the conventions, package names, and interfaces
visible in the provided tree and context files exactly.

Your branch is tested in ISOLATION: current main plus only YOUR files. Write
complete, working content, INCLUDING real tests inside your allowed files —
submissions whose test run executes zero tests are automatically rejected.
Do not import anything that is not in the shown tree, your own files, or the
standard library / declared project dependencies.

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
API_MAX_TOKENS = 16000  # workers return whole files inline — 4k truncated real replies mid-JSON


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
                max_tokens=API_MAX_TOKENS,
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


def _parse_json(raw: str, required_key: str | None = None) -> dict:
    """Extracts the JSON object from a model reply. Handles clean JSON,
    ```json fences, and prose-wrapped JSON (some proxied models preface the
    object with text despite the no-prose instruction) by scanning for the
    first parseable {...} in the reply. When required_key is given, only an
    object containing that key counts — this skips incidental JSON fragments
    (e.g. tool-call markup a proxy's injected system prompt provokes)."""
    def ok(obj) -> bool:
        return isinstance(obj, dict) and (required_key is None or required_key in obj)

    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        obj = json.loads(cleaned)
        if ok(obj):
            return obj
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    idx = cleaned.find("{")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(cleaned, idx)
            if ok(obj):
                return obj
        except json.JSONDecodeError:
            pass
        idx = cleaned.find("{", idx + 1)
    raise ValueError(f"no JSON object{f' with key {required_key!r}' if required_key else ''} "
                     f"found in model reply: {cleaned[:200]!r}")


JSON_PARSE_RETRIES = 3


def _call_json(role_key: str, system: str, user_content: str, required_key: str,
               normalize=None) -> dict:
    """_call + _parse_json with a corrective retry loop. Some providers/proxies
    inject their own agent-style system prompts, making the model reply with
    prose or tool-call markup instead of the demanded JSON. On a parse failure
    we re-ask, quoting the broken reply and restating the JSON-only contract.
    required_key anchors parsing to the role's expected schema so stray JSON
    fragments in a prose reply can't be mistaken for the answer. `normalize`
    (optional) validates/coerces the parsed object and raises ValueError on a
    shape mismatch, which also triggers a corrective retry."""
    prompt = user_content
    last_err = ""
    for attempt in range(JSON_PARSE_RETRIES):
        raw = _call(role_key, system, prompt)
        try:
            obj = _parse_json(raw, required_key=required_key)
            return normalize(obj) if normalize else obj
        except (ValueError, json.JSONDecodeError) as e:
            last_err = str(e)
            print(f"  [parse] {role_key} reply was not valid JSON "
                  f"(retry {attempt + 1}/{JSON_PARSE_RETRIES}): {last_err[:120]}")
            prompt = (
                f"{user_content}\n\n"
                f"YOUR PREVIOUS REPLY WAS REJECTED — it was not a single valid JSON object "
                f"in the required schema (key {required_key!r}). Problem: {last_err[:300]}\n"
                f"Do NOT write prose, markdown, or tool/function-call markup. Do NOT try to "
                f"create files yourself — you have no tools. Reply with ONLY the JSON object "
                f"in EXACTLY the schema described in your instructions."
            )
    raise RuntimeError(f"'{role_key}' never produced valid JSON after "
                       f"{JSON_PARSE_RETRIES} attempts: {last_err}")


def _normalize_worker(obj: dict) -> dict:
    """Coerces the worker's 'files' into {path: content}. Accepts the demanded
    dict shape, plus the common deviation of a list of {path, content} objects.
    Anything else raises ValueError so _call_json retries with feedback."""
    files = obj.get("files")
    if isinstance(files, list):
        coerced = {}
        for entry in files:
            if (isinstance(entry, dict)
                    and isinstance(entry.get("path"), str)
                    and isinstance(entry.get("content"), str)):
                coerced[entry["path"]] = entry["content"]
            else:
                raise ValueError("'files' must be an object mapping path -> full file content")
        files = coerced
    if (not isinstance(files, dict) or not files
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in files.items())):
        raise ValueError("'files' must be a non-empty object mapping path -> full file content (strings)")
    return {"files": files, "notes": str(obj.get("notes", ""))}


def senior_decompose(goal: str, max_workers: int, repo_snapshot: str = "",
                     plan_feedback: str = "") -> dict:
    system = SENIOR_SYSTEM.format(max_workers=max_workers)
    prompt = f"Goal: {goal}"
    if repo_snapshot:
        prompt += f"\n\n=== CURRENT REPOSITORY ===\n{repo_snapshot}"
    if plan_feedback:
        prompt += (
            f"\n\nYOUR PREVIOUS PLAN WAS REJECTED by pre-flight validation:\n{plan_feedback}\n"
            f"Produce a corrected plan. Every subtask needs: 'id' (slug), 'owner' (slug like "
            f"'worker_1'), 'description' (non-empty), 'file_scope' (non-empty list of REAL "
            f"repo-relative paths, no overlap with any other subtask). Use paths that actually "
            f"fit the repo tree shown above."
        )
    return _call_json("senior", system, prompt, required_key="subtasks")


def worker_implement(owner: str, description: str, file_scope: list[str],
                     context: str = "", rejection_reason: str = "") -> dict:
    prompt = f"Subtask: {description}\nFiles you may create or modify (ONLY these): {file_scope}"
    if context:
        prompt += f"\n\n=== REPOSITORY CONTEXT ===\n{context}"
    if rejection_reason:
        prompt += f"\n\nPREVIOUS ATTEMPT WAS REJECTED. Reason: {rejection_reason}\nFix this and resubmit."
    return _call_json(owner, WORKER_SYSTEM, prompt, required_key="files",
                      normalize=_normalize_worker)


def reviewer_verdict(diff_text: str, test_exit_code: int, test_output: str, scope_violations: list[str]) -> dict:
    if scope_violations:
        # Don't even bother calling the model — this is an automatic, non-negotiable reject.
        return {"verdict": "reject", "reason": f"Scope violation, touched files outside assignment: {scope_violations}"}
    prompt = (
        f"Diff:\n{diff_text}\n\n"
        f"Test exit code (0 = pass): {test_exit_code}\n"
        f"Test output:\n{test_output}"
    )
    return _call_json("reviewer", REVIEWER_SYSTEM, prompt, required_key="verdict")
