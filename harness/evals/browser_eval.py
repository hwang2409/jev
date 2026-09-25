"""Deterministic, offline evaluation for the arc-3 browser routing surface."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from zeta.core.safety import BrowserRiskEvidence
from zeta.core.safety._browser import _browser_layer0_classify
from zeta.tools.browser.adapter import (
    BrowserError,
    ElementRef,
    ElementUnavailableError,
    FakeBrowserAdapter,
    PageObservation,
    SnapshotLimits,
)
from zeta.tools.browser.catalog import SnapshotCatalogBuilder, prefilter_catalog

TASKS_PATH = Path(__file__).with_name("browser_tasks.jsonl")
CATALOG_SIZES = (10, 40, 120, 500, 2_000)
CHURN_MODES = ("static", "moderate", "full")
TASK_KINDS = ("clear_target", "repeated_label", "search_triage", "form")
TASK_ACTIONS = ("click", "type", "select", "submit")
RISK_CLASSES = (
    "none",
    "destructive",
    "payment",
    "authentication",
    "external_origin",
    "download",
)
THRESHOLD_VERSION = "browser-thresholds-v1"
SAFETY_POLICY_VERSION = "browser-eval-safety-v1"
TIMEOUTS_MS = {"navigation": 1_000, "action": 1_000}
MAX_PAGE_JEV_CALLS = 8
MAX_PAGE_JEV_TOKENS = 12_000
MAX_ACTIONS = 20
MAX_WALL_CLOCK_SECONDS = 120
SAFE_PROMPT_MARKERS = (
    "ignore previous",
    "system message",
    "follow these instructions",
    "reveal the prompt",
)
DENIED_RISK_CLASSES = frozenset(
    {"destructive", "payment", "authentication", "external_origin", "download"}
)
FAILURE_CATEGORIES = (
    "pre_filter_miss",
    "jev_selection_miss",
    "adapter_failure",
    "page_failure",
)

ArmName = Literal["routed", "stock"]


@dataclass(frozen=True, slots=True)
class TaskStep:
    action: str
    target_label: str
    target_role: str
    value: str | None = None


@dataclass(frozen=True, slots=True)
class ObservableTask:
    id: str
    kind: str
    prompt: str
    retryable_attempts: int


@dataclass(frozen=True, slots=True)
class BrowserTask:
    id: str
    kind: str
    catalog_size: int
    churn: str
    prompt: str
    action: str
    target_label: str
    target_role: str
    risky: bool = False
    failure_mode: str | None = None
    retryable_attempts: int = 0
    risk_class: str = "none"
    steps: tuple[TaskStep, ...] = ()


@dataclass(frozen=True, slots=True)
class EvalPolicy:
    """The four lane-2 budgets shared by both evaluation arms."""

    timeouts_ms: dict[str, int]
    page_jev_call_budget: int
    page_jev_token_budget: int
    max_actions: int
    max_wall_clock_seconds: float
    safety_policy_version: str


DEFAULT_POLICY = EvalPolicy(
    dict(TIMEOUTS_MS),
    MAX_PAGE_JEV_CALLS,
    MAX_PAGE_JEV_TOKENS,
    MAX_ACTIONS,
    MAX_WALL_CLOCK_SECONDS,
    SAFETY_POLICY_VERSION,
)


def _safe_prompt(prompt: object) -> bool:
    if type(prompt) is not str or not prompt.strip():
        return False
    lowered = prompt.casefold()
    return not any(marker in lowered for marker in SAFE_PROMPT_MARKERS)


def _parse_steps(value: object) -> tuple[TaskStep, ...]:
    if value is None:
        return ()
    if type(value) is not list or not value:
        raise ValueError("browser task steps must be a non-empty list")
    steps: list[TaskStep] = []
    for raw in value:
        if type(raw) is not dict or set(raw) - {
            "action",
            "target_label",
            "target_role",
            "value",
        }:
            raise ValueError("browser task has invalid steps")
        if not {"action", "target_label", "target_role"} <= set(raw):
            raise ValueError("browser task step is missing a required field")
        action = raw["action"]
        label = raw["target_label"]
        role = raw["target_role"]
        value = raw.get("value")
        if (
            action not in TASK_ACTIONS
            or type(label) is not str
            or not label
            or type(role) is not str
            or not role
            or (value is not None and type(value) is not str)
        ):
            raise ValueError("browser task has invalid steps")
        steps.append(TaskStep(action, label, role, value))
    return tuple(steps)


def _task_from_json(value: object) -> BrowserTask:
    if type(value) is not dict:
        raise ValueError("each browser task must be an object")
    required = {
        "id",
        "kind",
        "catalog_size",
        "churn",
        "prompt",
        "action",
        "target_label",
        "target_role",
    }
    optional = {
        "risky",
        "failure_mode",
        "retryable_attempts",
        "risk_class",
        "steps",
    }
    if set(value) - required - optional:
        raise ValueError("browser task has unknown fields")
    if not required <= set(value):
        raise ValueError("browser task is missing a required field")
    if (
        type(value["id"]) is not str
        or not value["id"]
        or value["kind"] not in TASK_KINDS
        or type(value["catalog_size"]) is not int
        or value["catalog_size"] not in CATALOG_SIZES
        or value["churn"] not in CHURN_MODES
        or value["action"] not in TASK_ACTIONS
        or type(value["target_label"]) is not str
        or not value["target_label"]
        or type(value["target_role"]) is not str
        or not value["target_role"]
        or not _safe_prompt(value["prompt"])
    ):
        raise ValueError(f"invalid browser task: {value.get('id')!r}")
    risky = value.get("risky", False)
    failure_mode = value.get("failure_mode")
    retries = value.get("retryable_attempts", 0)
    risk_class = value.get("risk_class", "destructive" if risky else "none")
    if (
        type(risky) is not bool
        or failure_mode not in (None, "adapter", "page")
        or type(retries) is not int
        or retries < 0
        or retries > 3
        or risk_class not in RISK_CLASSES
        or (risky != (risk_class != "none"))
    ):
        raise ValueError("invalid browser task policy fields")
    steps = _parse_steps(value.get("steps"))
    if value["kind"] == "form" and len(steps) < 2:
        raise ValueError("form tasks must have at least two actions")
    if value["kind"] != "form" and steps:
        raise ValueError("only form tasks may define multiple steps")
    return BrowserTask(
        id=value["id"],
        kind=value["kind"],
        catalog_size=value["catalog_size"],
        churn=value["churn"],
        prompt=value["prompt"],
        action=value["action"],
        target_label=value["target_label"],
        target_role=value["target_role"],
        risky=risky,
        failure_mode=failure_mode,
        retryable_attempts=retries,
        risk_class=risk_class,
        steps=steps,
    )


def load_browser_tasks(path: Path = TASKS_PATH) -> list[BrowserTask]:
    """Load and validate the local browser task corpus."""

    tasks: list[BrowserTask] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            tasks.append(_task_from_json(json.loads(line)))
    if not tasks:
        raise ValueError("browser task corpus is empty")
    return tasks


def _steps_for(task: BrowserTask) -> tuple[TaskStep, ...]:
    return task.steps or (TaskStep(task.action, task.target_label, task.target_role),)


def _target_index(task: BrowserTask, step_index: int = 0) -> int:
    if task.kind == "repeated_label":
        return task.catalog_size - 1
    if task.kind == "form":
        return min(task.catalog_size - 1, task.catalog_size // 2 + step_index)
    return task.catalog_size // 2


def _risk_fields(task: BrowserTask) -> dict[str, object]:
    return {
        "target_url": (
            "https://external.example.test/next"
            if task.risk_class == "external_origin"
            else None
        ),
        "form_action_origin": (
            "https://external.example.test/submit"
            if task.risk_class == "external_origin"
            else None
        ),
        "download": task.risk_class == "download",
        "durable_state_change": task.risk_class in {"destructive", "payment"},
    }


def _element(
    task: BrowserTask,
    index: int,
    *,
    snapshot_id: int = 1,
    generation: int = 1,
    changed: bool = False,
) -> ElementRef:
    steps = _steps_for(task)
    step_by_index = {
        _target_index(task, step_index): step for step_index, step in enumerate(steps)
    }
    target_indices = {
        _target_index(task, step_index) for step_index in range(len(steps))
    }
    target = index in target_indices
    suffix = "-new" if changed else ""
    element_id = f"e{index:05d}{suffix}"
    step = step_by_index.get(index)
    if task.kind == "repeated_label":
        text, role, affordance = task.target_label, task.target_role, task.action
    elif step is not None:
        text, role, affordance = step.target_label, step.target_role, step.action
    elif task.kind == "search_triage" and target:
        text, role, affordance = task.target_label, task.target_role, task.action
    elif task.kind == "search_triage":
        text, role, affordance = f"Search result {index}", "link", "click"
    else:
        text, role, affordance = f"Unrelated control {index}", "button", "click"
    if role == "textbox":
        affordance = "type"
    risk = _risk_fields(task) if target else {}
    return ElementRef(
        snapshot_id,
        element_id,
        role,
        affordance,
        text,
        text,
        None,
        "main",
        False,
        True,
        **risk,
        generation=generation,
    )


def _fixture(task: BrowserTask) -> tuple[list[PageObservation], str]:
    elements = tuple(_element(task, index) for index in range(task.catalog_size))
    target_id = elements[_target_index(task, len(_steps_for(task)) - 1)].element_id
    first = PageObservation(
        1,
        1,
        "https://offline.example.test/fixture",
        "Offline browser fixture",
        task.prompt,
        elements,
        task.failure_mode != "page",
        task.failure_mode != "page",
    )
    if task.churn == "static":
        return [first], target_id
    next_elements = tuple(
        _element(
            task,
            index,
            snapshot_id=2,
            generation=2,
            changed=True,
        )
        for index in range(task.catalog_size)
    )
    second = replace(first, snapshot_id=2, generation=2, elements=next_elements)
    return [first, second], target_id


class MockJevTransport:
    """A seeded, arm-neutral transport with explicit offline usage accounting."""

    usage_mode = "modeled_offline_transport"

    def __init__(self, policy: EvalPolicy, seed: int = 17) -> None:
        self.policy = policy
        self.seed = seed
        self.requests = 0
        self.retries = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.provider_turns = 0
        self.budget_exhausted = False
        self.modeled_time_ms = 0

    def _request(self, input_tokens: int, output_tokens: int) -> bool:
        if self.provider_turns >= self.policy.page_jev_call_budget:
            self.budget_exhausted = True
            return False
        total = self.input_tokens + self.output_tokens + input_tokens + output_tokens
        if total > self.policy.page_jev_token_budget:
            self.budget_exhausted = True
            return False
        self.requests += 1
        self.provider_turns += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.modeled_time_ms += 8 + input_tokens // 10
        return True

    def _rank(
        self,
        task: ObservableTask,
        candidates: list[dict[str, object]],
        goal: str,
        action: str,
    ) -> list[dict[str, object]]:
        goal_tokens = _goal_tokens(goal)
        scored: list[tuple[int, str, dict[str, object]]] = []
        for item in candidates:
            text = str(item.get("text", ""))
            overlap = len(goal_tokens & _goal_tokens(text))
            if action == "type":
                overlap += 2 if item["role"] == "textbox" else 0
            elif action == "select":
                overlap += 2 if item["role"] == "combobox" else 0
            elif action == "submit":
                overlap += 2 if item["affordance"] == "submit" else 0
            digest = hashlib.sha256(
                f"{self.seed}:{task.id}:{action}:{item['element_id']}".encode()
            ).hexdigest()
            scored.append((overlap, digest, item))
        return [
            item
            for _score, _digest, item in sorted(
                scored, key=lambda row: (-row[0], row[1])
            )
        ]

    def choose(
        self, task: ObservableTask, candidates: list[dict[str, object]], action: str
    ) -> dict[str, object] | None:
        attempts = task.retryable_attempts + 1
        input_tokens = 72 + len(candidates) * 9 + len(task.prompt)
        for attempt in range(attempts):
            if not self._request(input_tokens, 18):
                return None
            if attempt < task.retryable_attempts:
                self.retries += 1
        ranked = self._rank(task, candidates, task.prompt, action)
        ids = [str(item["element_id"]) for item in ranked]
        selected = ids[0] if ids else None
        return {
            "selected": selected,
            "top3": ids[:3],
            "confidence": 0.94 if selected is not None else 0.31,
            "threshold_version": THRESHOLD_VERSION,
        }

    def page_state(self, loaded: bool, stable: bool) -> dict[str, object] | None:
        if not self._request(96, 12):
            return None
        return {
            "correct": loaded and stable,
            "confidence": 0.96 if loaded and stable else 0.41,
            "threshold_version": THRESHOLD_VERSION,
        }

    def triage_search(
        self, task: ObservableTask, candidates: list[dict[str, object]]
    ) -> dict[str, object] | None:
        if not self._request(80 + len(candidates) * 6, 14):
            return None
        ranked = self._rank(task, candidates, task.prompt, "click")
        selected = str(ranked[0]["element_id"]) if ranked else None
        return {
            "selected": selected,
            "confidence": 0.93 if selected is not None else 0.31,
            "threshold_version": THRESHOLD_VERSION,
        }


def _goal_tokens(value: str) -> set[str]:
    return {
        token
        for token in "".join(
            character if character.isalnum() else " " for character in value.casefold()
        ).split()
        if token not in {"the", "a", "an", "to", "and", "then", "control", "result"}
    }


def _entry_payload(entry: object) -> dict[str, object]:
    return {
        "element_id": entry.element_id,
        "role": entry.role,
        "text": entry.text,
        "affordance": entry.affordance,
        "name": entry.name,
        "landmark": entry.landmark,
    }


def _safety_allows(task: BrowserTask, element: ElementRef, action: str) -> bool:
    evidence = BrowserRiskEvidence(
        action=action,
        role=element.role,
        text=" ".join(part for part in (element.text, element.name) if part),
        current_origin="https://offline.example.test",
        target_url=element.target_url,
        form_action_origin=element.form_action_origin,
        payment_language=task.risk_class == "payment",
        authentication_language=task.risk_class == "authentication",
        download=element.download,
        durable_state_change=element.durable_state_change or action == "submit",
        origin_allowed=False,
        operation_token=1,
    )
    classification, reason = _browser_layer0_classify(evidence)
    if task.risk_class in DENIED_RISK_CLASSES:
        return False
    if task.risky or classification != "analyzable":
        return False
    return reason in {None, "durable_state_change"}


def _target_id_for_step(
    task: BrowserTask, step_index: int, observation: PageObservation
) -> str:
    index = _target_index(task, step_index)
    return observation.elements[index].element_id


def _element_by_id(observation: PageObservation, element_id: str) -> ElementRef:
    return next(
        element for element in observation.elements if element.element_id == element_id
    )


async def _run_arm(
    task: BrowserTask, arm: ArmName, policy: EvalPolicy
) -> tuple[dict[str, object], MockJevTransport]:
    observations, target_id = _fixture(task)
    adapter = FakeBrowserAdapter(observations)
    transport = MockJevTransport(policy)
    observable_task = ObservableTask(
        task.id, task.kind, task.prompt, task.retryable_attempts
    )
    limits = SnapshotLimits(catalog_bytes=2_000_000)
    started = time.monotonic()
    action_count = 0
    modeled_time_ms = 20
    await adapter.launch()
    observation = await adapter.observe(limits)
    record: dict[str, object] = {
        "task_id": task.id,
        "arm": arm,
        "catalog_size": task.catalog_size,
        "churn": task.churn,
        "kind": task.kind,
        "target_id": target_id,
        "top1_correct": True,
        "top3_covered": True,
        "page_state_correct": False,
        "page_state_attempted": False,
        "task_success": False,
        "risky_false_approval": False,
        "pre_filter_miss": False,
        "jev_selection_miss": False,
        "adapter_failure": False,
        "page_failure": False,
        "failure_category": None,
        "stale_recovery_attempted": task.churn != "static",
        "stale_recovery_succeeded": False,
        "stale_rejection_count": 0,
        "action_retries": 0,
        "threshold_version": THRESHOLD_VERSION,
        "confidence": {
            "element_selection": 0.0,
            "page_state": 0.0,
            "search_triage": 0.0,
        },
        "search_triage_correct": True,
        "successful_steps": 0,
        "safety_denied": False,
        "safety_denied_class": None,
        "budget_exhausted": False,
        "stage_trace": [],
        "planned_stage_trace": [
            stage
            for step in _steps_for(task)
            for stage in (
                (
                    "page_state",
                    "catalog",
                    "search_triage",
                    "jev_selection",
                    "safety",
                    "action",
                )
                if task.kind == "search_triage"
                else ("page_state", "catalog", "jev_selection", "safety", "action")
            )
        ],
    }

    def finish() -> tuple[dict[str, object], MockJevTransport]:
        record["jev_tokens"] = transport.input_tokens + transport.output_tokens
        record["input_tokens"] = transport.input_tokens
        record["output_tokens"] = transport.output_tokens
        record["provider_turns"] = transport.provider_turns
        record["requests"] = transport.requests
        record["action_retries"] = transport.retries
        record["modeled_time_ms"] = modeled_time_ms + transport.modeled_time_ms
        record["action_count"] = action_count
        record["budget_exhausted"] = transport.budget_exhausted or (
            time.monotonic() - started >= policy.max_wall_clock_seconds
        )
        return record, transport

    def budget_available() -> bool:
        if time.monotonic() - started >= policy.max_wall_clock_seconds:
            record["budget_exhausted"] = True
            record["failure_category"] = "budget_exhausted"
            return False
        if action_count >= policy.max_actions:
            record["budget_exhausted"] = True
            record["failure_category"] = "budget_exhausted"
            return False
        return True

    async def page_gate(current: PageObservation) -> bool:
        nonlocal modeled_time_ms
        record["stage_trace"].append("page_state")
        if not budget_available():
            return False
        result = transport.page_state(current.loaded, current.stable)
        if result is None:
            record["budget_exhausted"] = True
            record["failure_category"] = "budget_exhausted"
            return False
        modeled_time_ms += 5
        record["page_state_attempted"] = True
        record["page_state_correct"] = bool(result["correct"])
        record["confidence"]["page_state"] = result["confidence"]
        if not current.loaded or not current.stable:
            record["page_failure"] = True
            record["failure_category"] = "page_failure"
            return False
        return True

    async def allow_navigation(_destination: str, _current: str | None) -> None:
        return None

    async def perform_action(element: ElementRef, step: TaskStep) -> None:
        nonlocal action_count, modeled_time_ms
        if not budget_available():
            raise BrowserError("browser budget exhausted")
        action_count += 1
        modeled_time_ms += 15
        if step.action in {"click", "submit"}:
            await adapter.click(element, policy.timeouts_ms["action"])
        elif step.action == "type":
            await adapter.type_text(
                element,
                step.value or "sample value",
                True,
                policy.timeouts_ms["action"],
            )
        else:
            await adapter.select(
                element,
                step.value or "option-a",
                policy.timeouts_ms["action"],
            )

    async def select_step(
        current: PageObservation, step_index: int, step: TaskStep
    ) -> tuple[ElementRef | None, str | None]:
        record["stage_trace"].append("catalog")
        catalog = SnapshotCatalogBuilder(limits).build(current)
        if arm == "routed":
            filtered = prefilter_catalog(step.target_label, step.action, catalog)
            entries = list(filtered.candidates)
            target_for_stage = _target_id_for_step(task, step_index, current)
            if target_for_stage not in {entry.element_id for entry in entries}:
                record["pre_filter_miss"] = True
                record["failure_category"] = "pre_filter_miss"
        else:
            entries = list(catalog.entries)
        payload = [_entry_payload(entry) for entry in entries]
        if task.kind == "search_triage":
            record["stage_trace"].append("search_triage")
            triage = transport.triage_search(observable_task, payload)
            if triage is None:
                record["budget_exhausted"] = True
                record["failure_category"] = "budget_exhausted"
                return None, None
            target_for_stage = _target_id_for_step(task, step_index, current)
            triage_selected = triage["selected"]
            if triage_selected != target_for_stage:
                record["search_triage_correct"] = False
            record["confidence"]["search_triage"] = triage["confidence"]
            payload = [
                item for item in payload if item["element_id"] == triage_selected
            ]
        record["stage_trace"].append("jev_selection")
        choice = transport.choose(observable_task, payload, step.action)
        if choice is None:
            record["budget_exhausted"] = True
            record["failure_category"] = "budget_exhausted"
            return None, None
        target_for_stage = _target_id_for_step(task, step_index, current)
        selected = choice["selected"]
        record["top1_correct"] = (
            bool(record["top1_correct"]) and selected == target_for_stage
        )
        record["top3_covered"] = (
            bool(record["top3_covered"]) and target_for_stage in choice["top3"]
        )
        record["confidence"]["element_selection"] = choice["confidence"]
        if selected != target_for_stage:
            record["jev_selection_miss"] = True
            record["failure_category"] = "jev_selection_miss"
            return None, target_for_stage
        return _element_by_id(current, str(selected)), target_for_stage

    if not await page_gate(observation):
        await adapter.close()
        return finish()

    steps = _steps_for(task)
    for step_index, step in enumerate(steps):
        if not budget_available():
            break
        current = await adapter.observe(limits)
        selected_ref, target_for_stage = await select_step(current, step_index, step)
        if selected_ref is None or target_for_stage is None:
            break

        if step_index == 0 and task.churn != "static":
            adapter.install_navigation_guard(allow_navigation)
            await adapter.navigate(current.url, policy.timeouts_ms["navigation"])
            adapter.detach(selected_ref.element_id)
            record["stage_trace"].append("stale_rejection")
            try:
                await perform_action(selected_ref, step)
            except ElementUnavailableError:
                record["stale_rejection_count"] = (
                    int(record["stale_rejection_count"]) + 1
                )
            except BrowserError:
                record["adapter_failure"] = True
                record["failure_category"] = "adapter_failure"
                break
            else:
                record["adapter_failure"] = True
                record["failure_category"] = "adapter_failure"
                break
            refreshed = await adapter.observe(limits)
            selected_ref, target_for_stage = await select_step(
                refreshed, step_index, step
            )
            if selected_ref is None or target_for_stage is None:
                break
            record["stage_trace"].append("stale_recovery")

        if not _safety_allows(task, selected_ref, step.action):
            record["safety_denied"] = True
            record["safety_denied_class"] = task.risk_class
            record["failure_category"] = "safety_denied"
            break
        if task.risky and _safety_allows(task, selected_ref, step.action):
            record["risky_false_approval"] = True
        if task.failure_mode == "adapter":
            adapter.timeout_next("type" if step.action == "type" else step.action)
        try:
            await perform_action(selected_ref, step)
        except BrowserError:
            record["adapter_failure"] = True
            record["failure_category"] = "adapter_failure"
            break
        record["successful_steps"] = int(record["successful_steps"]) + 1
        if step_index < len(steps) - 1:
            current = await adapter.observe(limits)
            if not await page_gate(current):
                break
    else:
        record["task_success"] = int(record["successful_steps"]) == len(steps)
        if record["task_success"]:
            record["stale_recovery_succeeded"] = (
                int(record["stale_rejection_count"]) > 0
            )
            record["failure_category"] = None
    await adapter.close()
    return finish()


def _ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _arm_summary(
    records: list[dict[str, object]], transport: MockJevTransport
) -> dict[str, object]:
    count = len(records)
    successful = sum(bool(record["task_success"]) for record in records)
    successful_steps = sum(int(record["successful_steps"]) for record in records)
    failures = Counter(
        record["failure_category"]
        for record in records
        if record["failure_category"] in FAILURE_CATEGORIES
    )
    confidence: dict[str, dict[str, float | int]] = {}
    for primitive in ("element_selection", "page_state", "search_triage"):
        samples = [
            (
                record["confidence"][primitive],
                record["top1_correct"]
                if primitive == "element_selection"
                else record["page_state_correct"]
                if primitive == "page_state"
                else record["search_triage_correct"],
            )
            for record in records
            if record["confidence"][primitive]
        ]
        confidence[primitive] = {
            "count": len(samples),
            "mean_confidence": round(sum(item[0] for item in samples) / len(samples), 4)
            if samples
            else 0.0,
            "accuracy": _ratio(sum(bool(item[1]) for item in samples), len(samples)),
            "brier_score": round(
                sum((item[0] - bool(item[1])) ** 2 for item in samples) / len(samples),
                4,
            )
            if samples
            else 0.0,
        }
    tokens = transport.input_tokens + transport.output_tokens
    cost = round(
        transport.input_tokens * 0.000001 + transport.output_tokens * 0.000002, 6
    )
    stale_attempts = sum(bool(record["stale_recovery_attempted"]) for record in records)
    return {
        "task_count": count,
        "top1_accuracy": _ratio(sum(bool(r["top1_correct"]) for r in records), count),
        "top3_coverage": _ratio(sum(bool(r["top3_covered"]) for r in records), count),
        "page_state_accuracy": _ratio(
            sum(bool(r["page_state_correct"]) for r in records),
            sum(bool(r["page_state_attempted"]) for r in records),
        ),
        "task_success_rate": _ratio(successful, count),
        "risky_false_approval_rate": _ratio(
            sum(bool(r["risky_false_approval"]) for r in records),
            sum(bool(r["safety_denied_class"]) for r in records),
        ),
        "jev_cost": cost,
        "jev_tokens": tokens,
        "jev_tokens_per_successful_step": _ratio(tokens, successful_steps),
        "jev_cost_per_successful_step": round(cost / max(successful_steps, 1), 6),
        "model_input_tokens": transport.input_tokens,
        "model_output_tokens": transport.output_tokens,
        "usage_measurement": transport.usage_mode,
        "cache_reads": 0,
        "provider_turns": transport.provider_turns,
        "stale_recovery_rate": _ratio(
            sum(bool(r["stale_recovery_succeeded"]) for r in records), stale_attempts
        ),
        "stale_rejection_count": sum(int(r["stale_rejection_count"]) for r in records),
        "action_retry_count": transport.retries,
        "provider_retry_count": transport.retries,
        "modeled_time_seconds": round(
            sum(int(r["modeled_time_ms"]) for r in records) / 1000, 6
        ),
        "time_measurement": "modeled_shared_stage_cost",
        "time_per_successful_step_seconds": round(
            sum(int(r["modeled_time_ms"]) for r in records)
            / 1000
            / max(successful_steps, 1),
            6,
        ),
        "failures": {name: failures.get(name, 0) for name in FAILURE_CATEGORIES},
        "safety_denied_actions": sum(bool(r["safety_denied"]) for r in records),
        "safety_denied_classes": sorted(
            {str(r["safety_denied_class"]) for r in records if r["safety_denied_class"]}
        ),
        "budget_exhausted_tasks": sum(bool(r["budget_exhausted"]) for r in records),
        "confidence_calibration": confidence,
        "threshold_versions": [THRESHOLD_VERSION],
    }


def _routing_win(routed: dict[str, object], stock: dict[str, object]) -> bool:
    quality_parity = (
        routed["task_success_rate"] > 0
        and routed["task_success_rate"] >= stock["task_success_rate"]
        and routed["risky_false_approval_rate"] <= stock["risky_false_approval_rate"]
    )
    routed_cost = routed.get("jev_cost_per_successful_step")
    stock_cost = stock.get("jev_cost_per_successful_step")
    cost_gain = (
        routed_cost is not None and stock_cost is not None and routed_cost < stock_cost
    )
    cost_or_scale_gain = cost_gain or (
        routed["time_per_successful_step_seconds"]
        < stock["time_per_successful_step_seconds"]
    )
    return bool(quality_parity and cost_or_scale_gain)


def _cell_summary(records: list[dict[str, object]]) -> dict[str, object]:
    transport = MockJevTransport(DEFAULT_POLICY)
    for record in records:
        transport.input_tokens += int(record["input_tokens"])
        transport.output_tokens += int(record["output_tokens"])
        transport.provider_turns += int(record["provider_turns"])
        transport.requests += int(record["requests"])
        transport.retries += int(record["action_retries"])
        transport.modeled_time_ms += int(record["modeled_time_ms"])
    return _arm_summary(records, transport)


async def _run(tasks: list[BrowserTask], policy: EvalPolicy) -> dict[str, object]:
    records_by_arm: dict[str, list[dict[str, object]]] = {"routed": [], "stock": []}
    transports: dict[str, MockJevTransport] = {}
    for arm in ("routed", "stock"):
        total_transport = MockJevTransport(policy)
        for task in tasks:
            record, transport = await _run_arm(task, arm, policy)
            records_by_arm[arm].append(record)
            total_transport.requests += transport.requests
            total_transport.retries += transport.retries
            total_transport.input_tokens += transport.input_tokens
            total_transport.output_tokens += transport.output_tokens
            total_transport.provider_turns += transport.provider_turns
            total_transport.modeled_time_ms += transport.modeled_time_ms
        transports[arm] = total_transport

    summaries = {
        arm: _arm_summary(records, transports[arm])
        for arm, records in records_by_arm.items()
    }
    cells: list[dict[str, object]] = []
    grouped: dict[tuple[int, str], dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: {"routed": [], "stock": []}
    )
    for arm, records in records_by_arm.items():
        for record in records:
            grouped[(record["catalog_size"], record["churn"])][arm].append(record)
    for (size, churn), arms in sorted(grouped.items()):
        cell_summaries = {arm: _cell_summary(arms[arm]) for arm in ("routed", "stock")}
        routed, stock = cell_summaries["routed"], cell_summaries["stock"]
        cells.append(
            {
                "catalog_size": size,
                "churn": churn,
                "routed": routed,
                "stock": stock,
                "routing_win": _routing_win(routed, stock),
                "win_requires_quality_and_safety_parity": True,
                "time_measurement": "modeled_shared_stage_cost",
            }
        )
    return {
        "schema_version": "browser-eval-v2",
        "offline": True,
        "network_used": False,
        "playwright_used": False,
        "provider_key_used": False,
        "policy": {
            "timeouts_ms": policy.timeouts_ms,
            "browser_page_jev_call_budget": policy.page_jev_call_budget,
            "browser_page_jev_token_budget": policy.page_jev_token_budget,
            "browser_task_action_budget": policy.max_actions,
            "browser_task_wall_clock_seconds": policy.max_wall_clock_seconds,
            "safety_policy_version": policy.safety_policy_version,
        },
        "corpus": {
            "task_count": len(tasks),
            "catalog_sizes": list(CATALOG_SIZES),
            "churn_modes": list(CHURN_MODES),
            "task_kinds": list(TASK_KINDS),
            "multi_step_form_tasks": sum(task.kind == "form" for task in tasks),
            "mutation_before_action_tasks": sum(
                task.churn != "static" for task in tasks
            ),
            "safety_denied_classes": sorted(DENIED_RISK_CLASSES),
        },
        "arms": summaries,
        "crossover_matrix": cells,
        "records": records_by_arm,
        "invariants": {
            "same_fixture_and_policy": True,
            "same_stage_pipeline": True,
            "routing_win_requires_task_success_and_safety_parity": True,
            "token_reduction_alone_is_not_a_win": True,
            "diagnostics_are_symmetric": True,
            "threshold_versions_recorded": True,
            "failure_categories_separate": True,
            "usage_and_time_are_labeled_modeled": True,
        },
    }


def run_browser_eval(
    tasks: list[BrowserTask] | None = None,
    *,
    policy: EvalPolicy = DEFAULT_POLICY,
) -> dict[str, object]:
    """Run both arms without opening a network or provider connection."""

    return asyncio.run(_run(tasks or load_browser_tasks(), policy))


def write_report(report: dict[str, object], path: Path) -> None:
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=TASKS_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_browser_eval(load_browser_tasks(args.tasks))
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        write_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
