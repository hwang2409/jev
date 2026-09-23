# Jev Tool Router (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a standalone tool router that asks Jev (TypeSafe System One) to pick the next tool for an agent's described step.

**Architecture:** One Jev `systemone` call per hop carrying three parallel questions: a Choice over a 15-tool catalog, a `needs_tool` Noul gate, and a `step_clarity` Noul gate. A hand-labeled 60-case evalset and an eval runner measure top-1/top-3 accuracy, confidence calibration, and gate separation.

**Tech Stack:** Python 3.11+, `requests`, `pytest`. Nothing else.

**Spec:** `docs/superpowers/specs/2026-09-17-jev-tool-router-design.md`

## Global Constraints

- Dependencies: stdlib + `requests` (runtime), `pytest` (tests). No frameworks, no packaging scaffold.
- Auth: a Vercel AI Gateway key (set in Henry's `~/.zshrc`; tests must NOT hit the network).
- API: `POST https://api.typesafe.ai/v1/systemone`, model `"jev-latest"`, retry 429/529 3 attempts with exponential backoff.
- All files live at the repo root (`~/me/fun/jev`) except docs. `results/` is gitignored.
- Commit style: Tim Pope seven rules, imperative subject, no emojis.
- Run tests via the project venv: `.venv/bin/pytest`.

---

### Task 1: Tool catalog

**Files:**
- Create: `catalog.py`
- Test: `tests/test_catalog.py`

**Interfaces:**
- Produces: `CATALOG: dict[str, str]` — 15 entries, tool name -> one-line description. Imported by `router.py` (Task 2) and referenced by the evalset (Task 4).

- [ ] **Step 1: Create the venv**

```bash
cd ~/me/fun/jev && python3 -m venv .venv && .venv/bin/pip install -q requests pytest
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_catalog.py
from catalog import CATALOG


def test_catalog_has_15_unique_tools():
    assert len(CATALOG) == 15
    assert len(set(CATALOG)) == 15


def test_descriptions_are_short_and_nonempty():
    for name, desc in CATALOG.items():
        assert name.strip() == name and name
        assert 10 <= len(desc) <= 120, f"{name}: bad description length"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_catalog.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'catalog'`
(If `tests/` import fails to find root modules, add an empty `conftest.py` at repo root — pytest rootdir insertion handles the path.)

- [ ] **Step 4: Write the catalog**

```python
# catalog.py
"""Phase-1 tool catalog: a realistic Claude Code-style toolset.

Descriptions are the routing criteria Jev sees. Deliberately overlapping
pairs (Read/Grep/Glob, Bash vs dedicated tools, WebFetch/WebSearch,
Grep vs LSP-references) keep routing non-trivial.
"""

CATALOG: dict[str, str] = {
    "Read": "Read the contents of a specific known file by path",
    "Write": "Create a new file or fully replace a file's contents",
    "Edit": "Make an exact in-place string replacement inside an existing file",
    "Bash": "Run a shell command: git, tests, linters, installs, scripts",
    "Grep": "Search file CONTENTS for a regex/text pattern across the codebase",
    "Glob": "Find files by NAME/path pattern, e.g. tests/**/*_test.py",
    "ListDir": "List the files and subdirectories of one directory",
    "WebFetch": "Fetch a specific known URL and extract information from it",
    "WebSearch": "Search the web when no specific URL is known",
    "Agent": "Spawn a subagent for a large independent subtask or parallel work",
    "TodoWrite": "Create or update the tracked todo/plan list for the task",
    "NotebookEdit": "Edit, insert, or delete a cell in a Jupyter notebook",
    "AskUserQuestion": "Ask the user a clarifying question and wait for the answer",
    "TaskStop": "Stop or kill a running background task/shell by id",
    "LSP": "Code intelligence: go-to-definition, find-references, symbol types",
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_catalog.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add catalog.py tests/ conftest.py && git commit -m "Add phase-1 tool catalog"
```

(`conftest.py` only exists if Step 3 needed it; `git add` a missing path errors, so create an empty one regardless — it is standard pytest furniture.)

---

### Task 2: Request builder and response parser

**Files:**
- Create: `router.py`
- Test: `tests/test_router.py`

**Interfaces:**
- Consumes: `CATALOG` from Task 1.
- Produces:
  - `RouteResult` dataclass: fields `tool: str`, `probabilities: dict[str, float]`, `confidence: float`, `needs_tool: float`, `step_clarity: float`, `usage: dict[str, int]`.
  - `build_request(task: str, step: str, history: list[str], catalog: dict[str, str]) -> dict`
  - `parse_response(data: dict) -> RouteResult`
  - Module constants `API_URL`, `MODEL`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_router.py
from router import RouteResult, build_request, parse_response

CATALOG_STUB = {"Read": "read a file", "Bash": "run a command"}

RESPONSE_STUB = {
    "model": "jev-1.13.0",
    "answers": {
        "tool": {
            "type": "choice",
            "choice": "Read",
            "confidence": 0.9,
            "probabilities": {"Read": 0.95, "Bash": 0.05},
        },
        "needs_tool": {"type": "noul", "noul": 0.97},
        "step_clarity": {"type": "noul", "noul": 0.88},
    },
    "usage": {"input_tokens": 400, "output_tokens": 60},
}


def test_build_request_shape():
    body = build_request("fix bug", "read config.py", ["opened repo"], CATALOG_STUB)
    assert body["model"] == "jev-latest"
    assert body["state"] == {
        "task": "fix bug",
        "current_step": "read config.py",
        "recent_steps": ["opened repo"],
    }
    q = body["questions"]
    assert q["tool"]["type"] == "choice"
    assert q["tool"]["criteria"] == CATALOG_STUB
    assert q["needs_tool"]["type"] == "noul"
    assert q["step_clarity"]["type"] == "noul"


def test_parse_response():
    r = parse_response(RESPONSE_STUB)
    assert isinstance(r, RouteResult)
    assert r.tool == "Read"
    assert r.probabilities == {"Read": 0.95, "Bash": 0.05}
    assert r.confidence == 0.9
    assert r.needs_tool == 0.97
    assert r.step_clarity == 0.88
    assert r.usage == {"input_tokens": 400, "output_tokens": 60}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_router.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'router'`

- [ ] **Step 3: Write the implementation**

```python
# router.py
"""Jev-backed tool router: one systemone call routes an agent step to a tool."""

import os
import time
from dataclasses import dataclass

import requests

from catalog import CATALOG

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"


@dataclass
class RouteResult:
    tool: str
    probabilities: dict[str, float]
    confidence: float
    needs_tool: float
    step_clarity: float
    usage: dict[str, int]


def build_request(
    task: str, step: str, history: list[str], catalog: dict[str, str]
) -> dict:
    return {
        "state": {
            "task": task,
            "current_step": step,
            "recent_steps": list(history),
        },
        "model": MODEL,
        "questions": {
            "tool": {
                "type": "choice",
                "instructions": (
                    "An agent is working on the task and describes its current "
                    "step. Which single tool should it call to accomplish this "
                    "step?"
                ),
                "criteria": catalog,
            },
            "needs_tool": {
                "type": "noul",
                "instructions": (
                    "Does the current step require calling a tool, rather than "
                    "the agent answering or reasoning directly from what it "
                    "already knows?"
                ),
            },
            "step_clarity": {
                "type": "noul",
                "instructions": (
                    "Is the current step description specific enough to route "
                    "to a single tool with confidence?"
                ),
            },
        },
    }


def parse_response(data: dict) -> RouteResult:
    answers = data["answers"]
    tool = answers["tool"]
    return RouteResult(
        tool=tool["choice"],
        probabilities=tool["probabilities"],
        confidence=tool["confidence"],
        needs_tool=answers["needs_tool"]["noul"],
        step_clarity=answers["step_clarity"]["noul"],
        usage=data["usage"],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_router.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add router.py tests/test_router.py && git commit -m "Add router request builder and response parser"
```

---

### Task 3: Live route() with retry

**Files:**
- Modify: `router.py` (append `route()`)
- Modify: `tests/test_router.py` (append retry tests)

**Interfaces:**
- Consumes: `build_request`, `parse_response`, `CATALOG` from Task 2.
- Produces: `route(task: str, step: str, history: list[str] | None = None, catalog: dict[str, str] | None = None, session=None, api_key: str | None = None) -> RouteResult`. Used by `run_eval.py` (Task 5).

- [ ] **Step 1: Write the failing tests (fake session, no network)**

Append to `tests/test_router.py`:

```python
import requests

import router
from router import route


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self.responses.pop(0)


def test_route_success(monkeypatch):
    sess = FakeSession([FakeResponse(200, RESPONSE_STUB)])
    r = route("t", "s", session=sess, api_key="k", catalog=CATALOG_STUB)
    assert r.tool == "Read"
    assert sess.calls[0]["headers"]["Authorization"] == "Bearer k"
    assert sess.calls[0]["json"]["questions"]["tool"]["criteria"] == CATALOG_STUB


def test_route_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(router.time, "sleep", lambda s: None)
    sess = FakeSession([FakeResponse(429), FakeResponse(200, RESPONSE_STUB)])
    r = route("t", "s", session=sess, api_key="k", catalog=CATALOG_STUB)
    assert r.tool == "Read"
    assert len(sess.calls) == 2


def test_route_gives_up_after_three_attempts(monkeypatch):
    monkeypatch.setattr(router.time, "sleep", lambda s: None)
    sess = FakeSession([FakeResponse(529)] * 3)
    try:
        route("t", "s", session=sess, api_key="k", catalog=CATALOG_STUB)
        raise AssertionError("expected HTTPError")
    except requests.HTTPError:
        pass
    assert len(sess.calls) == 3


def test_route_does_not_retry_client_errors(monkeypatch):
    sess = FakeSession([FakeResponse(422)])
    try:
        route("t", "s", session=sess, api_key="k", catalog=CATALOG_STUB)
        raise AssertionError("expected HTTPError")
    except requests.HTTPError:
        pass
    assert len(sess.calls) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_router.py -v`
Expected: 4 new tests FAIL with `ImportError: cannot import name 'route'`

- [ ] **Step 3: Implement route()**

Append to `router.py`:

```python
def route(
    task: str,
    step: str,
    history: list[str] | None = None,
    catalog: dict[str, str] | None = None,
    session=None,
    api_key: str | None = None,
) -> RouteResult:
    body = build_request(task, step, history or [], catalog or CATALOG)
    key = api_key or os.environ["VERCEL_AI_GATEWAY"]
    sess = session or requests.Session()
    headers = {"Authorization": f"Bearer {key}"}
    delay = 1.0
    for attempt in range(3):
        resp = sess.post(API_URL, json=body, headers=headers, timeout=60)
        if resp.status_code in (429, 529) and attempt < 2:
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        return parse_response(resp.json())
    raise AssertionError("unreachable")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_router.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add router.py tests/test_router.py && git commit -m "Add live route call with backoff retry"
```

---

### Task 4: Evalset

**Files:**
- Create: `evalset.jsonl`
- Test: `tests/test_evalset.py`

**Interfaces:**
- Produces: `evalset.jsonl` — one JSON object per line with keys `id`, `task`, `step`, `history` (list), `expected_tool` (string or null), `expected_needs_tool` (bool), `vague` (bool). Consumed by `run_eval.py` (Task 5).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evalset.py
import json
from collections import Counter
from pathlib import Path

from catalog import CATALOG


def load():
    lines = Path("evalset.jsonl").read_text().strip().splitlines()
    return [json.loads(line) for line in lines]


def test_size_and_schema():
    cases = load()
    assert len(cases) == 60
    ids = [c["id"] for c in cases]
    assert len(set(ids)) == 60
    for c in cases:
        assert set(c) == {"id", "task", "step", "history",
                          "expected_tool", "expected_needs_tool", "vague"}
        assert isinstance(c["history"], list)
        if c["expected_tool"] is not None:
            assert c["expected_tool"] in CATALOG


def test_composition():
    cases = load()
    clear = [c for c in cases if not c["vague"] and c["expected_needs_tool"]]
    no_tool = [c for c in cases if not c["expected_needs_tool"]]
    vague = [c for c in cases if c["vague"]]
    assert len(clear) == 46 and len(no_tool) == 8 and len(vague) == 6
    coverage = Counter(c["expected_tool"] for c in clear)
    for tool in CATALOG:
        assert coverage[tool] >= 2, f"{tool} covered by <2 cases"
    for c in no_tool + vague:
        assert c["expected_tool"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_evalset.py -v`
Expected: FAIL with `FileNotFoundError: evalset.jsonl`

- [ ] **Step 3: Write evalset.jsonl (exactly these 60 lines)**

```json
{"id": "read-1", "task": "Fix the login bug in the auth service", "step": "Look at the contents of src/auth/login.py to understand the current flow", "history": [], "expected_tool": "Read", "expected_needs_tool": true, "vague": false}
{"id": "read-2", "task": "Review a teammate's pull request", "step": "Open tests/test_api.py and read the newly added test", "history": [], "expected_tool": "Read", "expected_needs_tool": true, "vague": false}
{"id": "read-3", "task": "Debug last night's crash", "step": "Inspect the stack trace saved at /tmp/crash.log", "history": [], "expected_tool": "Read", "expected_needs_tool": true, "vague": false}
{"id": "read-4", "task": "Verify the app configuration", "step": "Check whether config.py sets DEBUG to True", "history": [], "expected_tool": "Read", "expected_needs_tool": true, "vague": false}
{"id": "write-1", "task": "Automate nightly backups", "step": "Create a new file scripts/backup.sh containing the backup logic", "history": [], "expected_tool": "Write", "expected_needs_tool": true, "vague": false}
{"id": "write-2", "task": "Set up continuous integration", "step": "Create .github/workflows/ci.yml from scratch", "history": [], "expected_tool": "Write", "expected_needs_tool": true, "vague": false}
{"id": "write-3", "task": "Document the project", "step": "Write a brand new README.md for the repository", "history": [], "expected_tool": "Write", "expected_needs_tool": true, "vague": false}
{"id": "write-4", "task": "Refresh generated code", "step": "Replace the entire contents of constants.py with the newly generated output", "history": [], "expected_tool": "Write", "expected_needs_tool": true, "vague": false}
{"id": "edit-1", "task": "Tune retry behavior", "step": "Change the constant MAX_RETRIES from 3 to 5 in config.py", "history": [], "expected_tool": "Edit", "expected_needs_tool": true, "vague": false}
{"id": "edit-2", "task": "Finish the rename refactor", "step": "Replace the call to old_name() with new_name() in utils.py", "history": [], "expected_tool": "Edit", "expected_needs_tool": true, "vague": false}
{"id": "edit-3", "task": "Fix the pagination bug", "step": "Correct the off-by-one loop bound on line 88 of parser.py", "history": [], "expected_tool": "Edit", "expected_needs_tool": true, "vague": false}
{"id": "bash-1", "task": "Ship the bug fix", "step": "Run the test suite with pytest to confirm nothing broke", "history": [], "expected_tool": "Bash", "expected_needs_tool": true, "vague": false}
{"id": "bash-2", "task": "Prepare the release", "step": "Create a git commit with the staged changes", "history": [], "expected_tool": "Bash", "expected_needs_tool": true, "vague": false}
{"id": "bash-3", "task": "Set up the project locally", "step": "Install the Python dependencies with pip", "history": [], "expected_tool": "Bash", "expected_needs_tool": true, "vague": false}
{"id": "bash-4", "task": "Clean up code style", "step": "Run the ruff linter over the whole repository", "history": [], "expected_tool": "Bash", "expected_needs_tool": true, "vague": false}
{"id": "grep-1", "task": "Remove a deprecated API", "step": "Find every file that still calls fetch_legacy() across the codebase", "history": [], "expected_tool": "Grep", "expected_needs_tool": true, "vague": false}
{"id": "grep-2", "task": "Audit logging practices", "step": "Search the repo for print( calls used instead of the logger", "history": [], "expected_tool": "Grep", "expected_needs_tool": true, "vague": false}
{"id": "grep-3", "task": "Trace configuration usage", "step": "Find where the environment variable DATABASE_URL is referenced", "history": [], "expected_tool": "Grep", "expected_needs_tool": true, "vague": false}
{"id": "grep-4", "task": "Understand the data model", "step": "Find which file contains the text 'class User' in its contents", "history": [], "expected_tool": "Grep", "expected_needs_tool": true, "vague": false}
{"id": "glob-1", "task": "Migrate the test layout", "step": "List all files matching tests/**/*_test.py", "history": [], "expected_tool": "Glob", "expected_needs_tool": true, "vague": false}
{"id": "glob-2", "task": "Audit static assets", "step": "Find every .png file under the static/ directory by filename", "history": [], "expected_tool": "Glob", "expected_needs_tool": true, "vague": false}
{"id": "glob-3", "task": "Consolidate test fixtures", "step": "Locate all files named conftest.py anywhere in the repo", "history": [], "expected_tool": "Glob", "expected_needs_tool": true, "vague": false}
{"id": "listdir-1", "task": "Get oriented in an unfamiliar repo", "step": "See what files and folders sit at the repository root", "history": [], "expected_tool": "ListDir", "expected_needs_tool": true, "vague": false}
{"id": "listdir-2", "task": "Explore the request handlers", "step": "Show what is inside the src/handlers directory", "history": [], "expected_tool": "ListDir", "expected_needs_tool": true, "vague": false}
{"id": "webfetch-1", "task": "Integrate Stripe webhooks", "step": "Read the docs page at https://docs.stripe.com/webhooks", "history": [], "expected_tool": "WebFetch", "expected_needs_tool": true, "vague": false}
{"id": "webfetch-2", "task": "Upgrade a dependency safely", "step": "Open the changelog at the GitHub releases URL the user provided", "history": [], "expected_tool": "WebFetch", "expected_needs_tool": true, "vague": false}
{"id": "webfetch-3", "task": "Summarize an article", "step": "Get the content of the blog post at https://example.com/post", "history": [], "expected_tool": "WebFetch", "expected_needs_tool": true, "vague": false}
{"id": "websearch-1", "task": "Fix an obscure runtime error", "step": "Search online for solutions to the error EADDRINUSE address already in use", "history": [], "expected_tool": "WebSearch", "expected_needs_tool": true, "vague": false}
{"id": "websearch-2", "task": "Choose a PDF library", "step": "Find the currently recommended Python libraries for PDF parsing", "history": [], "expected_tool": "WebSearch", "expected_needs_tool": true, "vague": false}
{"id": "websearch-3", "task": "Plan a version upgrade", "step": "Look up when Python 3.13 reaches end of life; no URL known", "history": [], "expected_tool": "WebSearch", "expected_needs_tool": true, "vague": false}
{"id": "websearch-4", "task": "Add middleware correctly", "step": "Find the official FastAPI middleware documentation page, URL unknown", "history": [], "expected_tool": "WebSearch", "expected_needs_tool": true, "vague": false}
{"id": "agent-1", "task": "Refactor the whole payments area", "step": "Fan out an independent subagent to migrate the payments module in parallel", "history": [], "expected_tool": "Agent", "expected_needs_tool": true, "vague": false}
{"id": "agent-2", "task": "Security review before launch", "step": "Delegate a broad codebase-wide security sweep to a separate agent", "history": [], "expected_tool": "Agent", "expected_needs_tool": true, "vague": false}
{"id": "agent-3", "task": "Keep momentum while debugging", "step": "Spin up a subagent to investigate the flaky test in parallel with my work", "history": [], "expected_tool": "Agent", "expected_needs_tool": true, "vague": false}
{"id": "todo-1", "task": "Implement a large multi-part feature", "step": "Break the remaining work into a tracked todo list", "history": [], "expected_tool": "TodoWrite", "expected_needs_tool": true, "vague": false}
{"id": "todo-2", "task": "Track migration progress", "step": "Mark the database migration item as completed in my plan list", "history": [], "expected_tool": "TodoWrite", "expected_needs_tool": true, "vague": false}
{"id": "notebook-1", "task": "Fix the analysis notebook", "step": "Update the plotting code in cell 4 of analysis.ipynb", "history": [], "expected_tool": "NotebookEdit", "expected_needs_tool": true, "vague": false}
{"id": "notebook-2", "task": "Clean up the exploration notebook", "step": "Delete the broken cell from exploration.ipynb", "history": [], "expected_tool": "NotebookEdit", "expected_needs_tool": true, "vague": false}
{"id": "ask-1", "task": "Deploy the new version", "step": "The user must choose between blue-green and rolling deploy; get their decision", "history": [], "expected_tool": "AskUserQuestion", "expected_needs_tool": true, "vague": false}
{"id": "ask-2", "task": "Clean up the repository", "step": "Ask whether cleaning up includes deleting the old release branch", "history": [], "expected_tool": "AskUserQuestion", "expected_needs_tool": true, "vague": false}
{"id": "ask-3", "task": "Add authentication", "step": "Confirm with the user which OAuth provider they want to support", "history": [], "expected_tool": "AskUserQuestion", "expected_needs_tool": true, "vague": false}
{"id": "taskstop-1", "task": "Recover from a runaway process", "step": "Stop the background dev-server task that is stuck in a loop", "history": [], "expected_tool": "TaskStop", "expected_needs_tool": true, "vague": false}
{"id": "taskstop-2", "task": "Wrap up the session", "step": "Kill the long-running watch task with id bg-42", "history": [], "expected_tool": "TaskStop", "expected_needs_tool": true, "vague": false}
{"id": "lsp-1", "task": "Rename a core function safely", "step": "Find all code references to the function compute_totals before renaming it", "history": [], "expected_tool": "LSP", "expected_needs_tool": true, "vague": false}
{"id": "lsp-2", "task": "Understand unfamiliar code", "step": "Jump to the definition of the Config class used on this line", "history": [], "expected_tool": "LSP", "expected_needs_tool": true, "vague": false}
{"id": "lsp-3", "task": "Fix a type error", "step": "Get the inferred type of the variable result at line 33", "history": [], "expected_tool": "LSP", "expected_needs_tool": true, "vague": false}
{"id": "notool-1", "task": "Investigate the outage", "step": "Summarize the findings from the log files I already read for the user", "history": ["read the outage logs", "grepped for error spikes"], "expected_tool": null, "expected_needs_tool": false, "vague": false}
{"id": "notool-2", "task": "Explain a concurrency bug", "step": "Explain what a race condition is in my final answer", "history": [], "expected_tool": null, "expected_needs_tool": false, "vague": false}
{"id": "notool-3", "task": "Pick an architecture", "step": "Decide which of the two approaches I already analyzed is better and state it", "history": ["compared approach A and B"], "expected_tool": null, "expected_needs_tool": false, "vague": false}
{"id": "notool-4", "task": "Open the pull request", "step": "Compose the PR description text in my reply", "history": ["pushed the branch"], "expected_tool": null, "expected_needs_tool": false, "vague": false}
{"id": "notool-5", "task": "Finish the refactor", "step": "The work is complete; report the results to the user", "history": ["ran the full test suite, all green"], "expected_tool": null, "expected_needs_tool": false, "vague": false}
{"id": "notool-6", "task": "Answer an algorithms question", "step": "State the average-case time complexity of quicksort", "history": [], "expected_tool": null, "expected_needs_tool": false, "vague": false}
{"id": "notool-7", "task": "Align on the plan", "step": "Restate the migration plan in simpler terms for the user", "history": [], "expected_tool": null, "expected_needs_tool": false, "vague": false}
{"id": "notool-8", "task": "Wrap up the conversation", "step": "Acknowledge the user's thanks and end the turn", "history": [], "expected_tool": null, "expected_needs_tool": false, "vague": false}
{"id": "vague-1", "task": "Improve the project", "step": "Handle the file stuff", "history": [], "expected_tool": null, "expected_needs_tool": true, "vague": true}
{"id": "vague-2", "task": "Keep going", "step": "Do the next thing", "history": [], "expected_tool": null, "expected_needs_tool": true, "vague": true}
{"id": "vague-3", "task": "Deal with the bug report", "step": "Fix it", "history": [], "expected_tool": null, "expected_needs_tool": true, "vague": true}
{"id": "vague-4", "task": "Follow up on the discussion", "step": "Check the thing we talked about", "history": [], "expected_tool": null, "expected_needs_tool": true, "vague": true}
{"id": "vague-5", "task": "Finish the integration", "step": "Deal with the web part", "history": [], "expected_tool": null, "expected_needs_tool": true, "vague": true}
{"id": "vague-6", "task": "General assistance", "step": "Make progress on the task", "history": [], "expected_tool": null, "expected_needs_tool": true, "vague": true}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_evalset.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add evalset.jsonl tests/test_evalset.py && git commit -m "Add hand-labeled 60-case routing evalset"
```

---

### Task 5: Eval runner

**Files:**
- Create: `run_eval.py`
- Test: `tests/test_run_eval.py`

**Interfaces:**
- Consumes: `route`, `RouteResult` from Task 3; `evalset.jsonl` from Task 4.
- Produces:
  - `load_cases(path: str) -> list[dict]`
  - `top_k(probabilities: dict[str, float], k: int) -> list[str]`
  - `evaluate(cases: list[dict], route_fn) -> list[dict]` — each result dict is the case merged with keys `tool`, `probabilities`, `confidence`, `needs_tool`, `step_clarity`, `usage`, or `error` (string) on failure.
  - `summarize(results: list[dict]) -> dict`
  - `main()` — CLI entry: run all, print report, save `results/<YYYYmmdd-HHMMSS>.json`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_run_eval.py
from run_eval import evaluate, summarize, top_k
from router import RouteResult


def fake_route_factory(tool, confidence, needs_tool, step_clarity):
    def fake_route(task, step, history=None):
        probs = {tool: confidence, "Other": round(1 - confidence, 4)}
        return RouteResult(
            tool=tool,
            probabilities=probs,
            confidence=confidence,
            needs_tool=needs_tool,
            step_clarity=step_clarity,
            usage={"input_tokens": 100, "output_tokens": 10},
        )
    return fake_route


def case(id, expected_tool, needs=True, vague=False):
    return {"id": id, "task": "t", "step": "s", "history": [],
            "expected_tool": expected_tool, "expected_needs_tool": needs,
            "vague": vague}


def test_top_k():
    probs = {"A": 0.5, "B": 0.3, "C": 0.2}
    assert top_k(probs, 2) == ["A", "B"]


def test_summarize_metrics():
    cases_and_routes = [
        (case("c1", "Read"), fake_route_factory("Read", 0.9, 0.95, 0.9)),
        (case("c2", "Bash"), fake_route_factory("Grep", 0.4, 0.95, 0.9)),
        (case("n1", None, needs=False), fake_route_factory("Read", 0.5, 0.1, 0.9)),
        (case("v1", None, vague=True), fake_route_factory("Bash", 0.3, 0.8, 0.2)),
    ]
    results = []
    for c, fn in cases_and_routes:
        results.extend(evaluate([c], route_fn=fn))
    s = summarize(results)
    assert s["clear_cases"] == 2
    assert s["top1_accuracy"] == 0.5
    assert s["mean_confidence_correct"] == 0.9
    assert s["mean_confidence_incorrect"] == 0.4
    assert s["confusions"] == [{"id": "c2", "expected": "Bash", "chosen": "Grep"}]
    assert s["needs_tool_mean_on_tool_cases"] == 0.95
    assert s["needs_tool_mean_on_no_tool_cases"] == 0.1
    assert s["clarity_mean_on_clear"] == 0.9
    assert s["clarity_mean_on_vague"] == 0.2
    assert s["total_input_tokens"] == 400
    assert s["total_output_tokens"] == 40
    assert s["errors"] == 0


def test_evaluate_captures_errors():
    def boom(task, step, history=None):
        raise RuntimeError("api down")
    results = evaluate([case("c1", "Read")], route_fn=boom)
    assert results[0]["error"] == "api down"
    assert summarize(results)["errors"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_run_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_eval'`

- [ ] **Step 3: Write the implementation**

```python
# run_eval.py
"""Run the routing evalset through Jev and report accuracy + calibration."""

import json
import statistics
import time
from pathlib import Path

from router import route


def load_cases(path: str = "evalset.jsonl") -> list[dict]:
    lines = Path(path).read_text().strip().splitlines()
    return [json.loads(line) for line in lines]


def top_k(probabilities: dict[str, float], k: int) -> list[str]:
    return sorted(probabilities, key=probabilities.get, reverse=True)[:k]


def evaluate(cases: list[dict], route_fn=route) -> list[dict]:
    results = []
    for c in cases:
        row = dict(c)
        try:
            r = route_fn(c["task"], c["step"], history=c["history"])
            row.update(
                tool=r.tool,
                probabilities=r.probabilities,
                confidence=r.confidence,
                needs_tool=r.needs_tool,
                step_clarity=r.step_clarity,
                usage=r.usage,
            )
        except Exception as exc:  # noqa: BLE001 - eval must survive bad calls
            row["error"] = str(exc)
        results.append(row)
    return results


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r]
    clear = [r for r in ok if not r["vague"] and r["expected_needs_tool"]
             and r["expected_tool"] is not None]
    correct = [r for r in clear if r["tool"] == r["expected_tool"]]
    incorrect = [r for r in clear if r["tool"] != r["expected_tool"]]
    in_top3 = [r for r in clear if r["expected_tool"] in top_k(r["probabilities"], 3)]
    no_tool = [r for r in ok if not r["expected_needs_tool"]]
    tool_cases = [r for r in ok if r["expected_needs_tool"] and not r["vague"]]
    vague = [r for r in ok if r["vague"]]
    return {
        "cases": len(results),
        "errors": len(results) - len(ok),
        "clear_cases": len(clear),
        "top1_accuracy": round(len(correct) / len(clear), 4) if clear else None,
        "top3_accuracy": round(len(in_top3) / len(clear), 4) if clear else None,
        "confusions": [
            {"id": r["id"], "expected": r["expected_tool"], "chosen": r["tool"]}
            for r in incorrect
        ],
        "mean_confidence_correct": _mean([r["confidence"] for r in correct]),
        "mean_confidence_incorrect": _mean([r["confidence"] for r in incorrect]),
        "needs_tool_mean_on_tool_cases": _mean([r["needs_tool"] for r in tool_cases]),
        "needs_tool_mean_on_no_tool_cases": _mean([r["needs_tool"] for r in no_tool]),
        "clarity_mean_on_clear": _mean([r["step_clarity"] for r in clear]),
        "clarity_mean_on_vague": _mean([r["step_clarity"] for r in vague]),
        "total_input_tokens": sum(r["usage"]["input_tokens"] for r in ok),
        "total_output_tokens": sum(r["usage"]["output_tokens"] for r in ok),
    }


def format_report(summary: dict) -> str:
    lines = ["Jev tool-router eval", "=" * 40]
    for key, value in summary.items():
        if key == "confusions":
            lines.append(f"confusions ({len(value)}):")
            for c in value:
                lines.append(f"  {c['id']}: expected {c['expected']}, chose {c['chosen']}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def main() -> None:
    cases = load_cases()
    results = evaluate(cases)
    summary = summarize(results)
    print(format_report(summary))
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = out_dir / f"{stamp}.json"
    out.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest -v`
Expected: all tests pass (catalog 2, router 6, evalset 2, run_eval 3)

- [ ] **Step 5: Commit**

```bash
git add run_eval.py tests/test_run_eval.py && git commit -m "Add eval runner with accuracy and calibration report"
```

---

### Task 6: Live eval run (the experiment)

**Files:**
- No new source files. Produces `results/<timestamp>.json` (gitignored) and findings.

**Interfaces:**
- Consumes: everything above plus a Vercel AI Gateway key from the environment.

- [ ] **Step 1: Smoke-test one live call**

```bash
cd ~/me/fun/jev && export VERCEL_AI_GATEWAY=$(grep VERCEL_AI_GATEWAY ~/.zshrc | sed 's/^export VERCEL_AI_GATEWAY=//') && .venv/bin/python -c "
from router import route
r = route('Fix the login bug', 'Look at the contents of src/auth/login.py')
print(r)"
```

Expected: a `RouteResult` with `tool='Read'` and sensible probabilities.

- [ ] **Step 2: Run the full evalset**

```bash
.venv/bin/python run_eval.py
```

Expected: report prints; 0 errors; `results/<timestamp>.json` written. ~60 calls at ~500 input tokens each.

- [ ] **Step 3: Judge against the spec's success criteria**

1. top1_accuracy >= 0.90 on clear steps.
2. mean_confidence_incorrect meaningfully below mean_confidence_correct.
3. clarity_mean_on_vague meaningfully below clarity_mean_on_clear, and needs_tool separates no-tool cases from tool cases.

Record pass/fail per criterion and the confusion list verbatim in the findings report.

- [ ] **Step 4: Report findings**

Report the three criteria verdicts, the confusion pairs, and token cost. If criterion 2 fails, flag that the harness idea needs a rethink before phase 2 (per spec).
