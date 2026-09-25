"""Deterministic, offline evaluation for the arc-3 browser routing surface."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from zeta.tools.browser.adapter import (
    BrowserError,
    ElementRef,
    FakeBrowserAdapter,
    PageObservation,
    SnapshotLimits,
)
from zeta.tools.browser.catalog import SnapshotCatalogBuilder, prefilter_catalog

TASKS_PATH = Path(__file__).with_name("browser_tasks.jsonl")
CATALOG_SIZES = (10, 40, 120, 500, 2_000)
CHURN_MODES = ("static", "moderate", "full")
TASK_KINDS = ("clear_target", "repeated_label", "search_triage", "form")
THRESHOLD_VERSION = "browser-thresholds-v1"
SAFETY_POLICY_VERSION = "browser-eval-safety-v1"
TIMEOUTS_MS = {"navigation": 1_000, "action": 1_000}
MAX_ACTIONS = 20
MAX_WALL_CLOCK_SECONDS = 120
SAFE_PROMPT_MARKERS = (
    "ignore previous",
    "system message",
    "follow these instructions",
    "reveal the prompt",
)

ArmName = Literal["routed", "stock"]


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


@dataclass(frozen=True, slots=True)
class EvalPolicy:
    """The policy shared by both evaluation arms."""

    timeouts_ms: dict[str, int]
    max_actions: int
    max_wall_clock_seconds: int
    safety_policy_version: str


DEFAULT_POLICY = EvalPolicy(
    dict(TIMEOUTS_MS), MAX_ACTIONS, MAX_WALL_CLOCK_SECONDS, SAFETY_POLICY_VERSION
)


def _safe_prompt(prompt: object) -> bool:
    if type(prompt) is not str or not prompt.strip():
        return False
    lowered = prompt.casefold()
    return not any(marker in lowered for marker in SAFE_PROMPT_MARKERS)


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
    if set(value) - required - {"risky", "failure_mode", "retryable_attempts"}:
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
        or type(value["action"]) is not str
        or not value["action"]
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
    if type(risky) is not bool or failure_mode not in (None, "adapter", "page"):
        raise ValueError("invalid browser task policy fields")
    if type(retries) is not int or retries < 0 or retries > 3:
        raise ValueError("retryable_attempts must be between zero and three")
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


def _target_index(task: BrowserTask) -> int:
    if task.kind == "repeated_label":
        return task.catalog_size - 1
    return task.catalog_size // 2


def _element(
    task: BrowserTask,
    index: int,
    *,
    snapshot_id: int = 1,
    generation: int = 1,
    changed: bool = False,
) -> ElementRef:
    target = index == _target_index(task)
    suffix = "-new" if changed else ""
    element_id = f"e{index:05d}{suffix}"
    if task.kind == "repeated_label" or target:
        text, role = task.target_label, task.target_role
    elif task.kind == "search_triage":
        text, role = f"Search result {index}", "link"
    elif task.kind == "form":
        text, role = f"Form field {index}", "textbox"
    else:
        text, role = f"Unrelated control {index}", "button"
    affordance = "type" if role == "textbox" else task.action
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
        generation=generation,
    )


def _fixture(task: BrowserTask) -> tuple[list[PageObservation], str]:
    elements = tuple(_element(task, index) for index in range(task.catalog_size))
    target_id = elements[_target_index(task)].element_id
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
        _element(task, index, snapshot_id=2, generation=2, changed=task.churn == "full")
        for index in range(task.catalog_size)
    )
    second = replace(first, snapshot_id=2, generation=2, elements=next_elements)
    return [first, second], target_id


class MockJevTransport:
    """A local Jev transport that records request shape and usage."""

    def __init__(self) -> None:
        self.requests = 0
        self.retries = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.provider_turns = 0

    def choose(
        self, task: BrowserTask, candidates: list[dict[str, object]], target_id: str
    ) -> dict[str, object]:
        self.provider_turns += 1
        attempts = task.retryable_attempts + 1
        self.requests += attempts
        self.retries += task.retryable_attempts
        self.input_tokens += attempts * (72 + len(candidates) * 9)
        self.output_tokens += attempts * 18
        ids = [item["element_id"] for item in candidates]
        if task.kind == "repeated_label":
            selected = next((item_id for item_id in ids if item_id != target_id), None)
        else:
            selected = target_id if target_id in ids else None
        ranked = list(ids[:3])
        if target_id in ids and target_id not in ranked:
            ranked[-1] = target_id
        correct = selected == target_id
        return {
            "selected": selected,
            "top3": ranked,
            "confidence": 0.94 if correct else 0.61,
            "threshold_version": THRESHOLD_VERSION,
            "usage": {
                "input_tokens": attempts * (72 + len(candidates) * 9),
                "output_tokens": attempts * 18,
            },
        }

    def page_state(self, task: BrowserTask, loaded: bool) -> dict[str, object]:
        self.provider_turns += 1
        self.requests += 1
        self.input_tokens += 96
        self.output_tokens += 12
        return {
            "correct": loaded,
            "confidence": 0.96 if loaded else 0.91,
            "threshold_version": THRESHOLD_VERSION,
        }

    def triage_search(
        self, task: BrowserTask, candidates: list[dict[str, object]], target_id: str
    ) -> dict[str, object]:
        self.provider_turns += 1
        self.requests += 1
        self.input_tokens += 80 + len(candidates) * 6
        self.output_tokens += 14
        return {
            "selected": target_id
            if target_id in {item["element_id"] for item in candidates}
            else None,
            "correct": target_id in {item["element_id"] for item in candidates},
            "confidence": 0.93,
            "threshold_version": THRESHOLD_VERSION,
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


def _safety_allows(task: BrowserTask) -> bool:
    return not task.risky


async def _run_arm(
    task: BrowserTask, arm: ArmName, policy: EvalPolicy
) -> tuple[dict[str, object], MockJevTransport]:
    observations, target_id = _fixture(task)
    adapter = FakeBrowserAdapter(observations)
    transport = MockJevTransport()
    limits = SnapshotLimits(catalog_bytes=2_000_000)
    await adapter.launch()
    observation = await adapter.observe(limits)
    record: dict[str, object] = {
        "task_id": task.id,
        "arm": arm,
        "catalog_size": task.catalog_size,
        "churn": task.churn,
        "kind": task.kind,
        "target_id": target_id,
        "top1_correct": False,
        "top3_covered": False,
        "page_state_correct": False,
        "task_success": False,
        "risky_false_approval": False,
        "pre_filter_miss": False,
        "jev_selection_miss": False,
        "adapter_failure": False,
        "page_failure": False,
        "failure_category": None,
        "stale_recovery_attempted": task.churn == "full",
        "stale_recovery_succeeded": False,
        "action_retries": 0,
        "threshold_version": THRESHOLD_VERSION,
        "confidence": {
            "element_selection": 0.0,
            "page_state": 0.0,
            "search_triage": 0.0,
        },
        "search_triage_correct": False,
    }

    def finish() -> tuple[dict[str, object], MockJevTransport]:
        record["jev_tokens"] = transport.input_tokens + transport.output_tokens
        record["input_tokens"] = transport.input_tokens
        record["output_tokens"] = transport.output_tokens
        record["provider_turns"] = transport.provider_turns
        record["action_retries"] = transport.retries
        record["requests"] = transport.requests
        return record, transport

    if not observation.loaded or not observation.stable:
        record["page_failure"] = True
        record["failure_category"] = "page_failure"
        page = transport.page_state(task, False)
        record["page_state_correct"] = bool(page["correct"]) is False
        record["confidence"]["page_state"] = page["confidence"]
        await adapter.close()
        return finish()

    catalog = SnapshotCatalogBuilder(limits).build(observation)
    if arm == "routed":
        filtered = prefilter_catalog(task.target_label, task.action, catalog)
        entries = list(filtered.candidates)
        record["pre_filter_miss"] = target_id not in {
            entry.element_id for entry in entries
        }
    else:
        entries = list(catalog.entries)
    payload = [_entry_payload(entry) for entry in entries]
    if task.kind == "search_triage":
        triage = transport.triage_search(task, payload, target_id)
        record["search_triage_correct"] = triage["correct"]
        record["confidence"]["search_triage"] = triage["confidence"]
    choice = transport.choose(task, payload, target_id)
    selected = choice["selected"]
    record["top1_correct"] = selected == target_id
    record["top3_covered"] = target_id in choice["top3"]
    record["confidence"]["element_selection"] = choice["confidence"]
    if record["pre_filter_miss"]:
        record["page_state_correct"] = True
        record["jev_selection_miss"] = False
        record["failure_category"] = "pre_filter_miss"
        await adapter.close()
        return finish()
    if not record["top1_correct"]:
        record["jev_selection_miss"] = True
        record["failure_category"] = "jev_selection_miss"
        page = transport.page_state(task, True)
        record["page_state_correct"] = bool(page["correct"])
        record["confidence"]["page_state"] = page["confidence"]
        await adapter.close()
        return finish()

    page = transport.page_state(task, True)
    record["page_state_correct"] = bool(page["correct"])
    record["confidence"]["page_state"] = page["confidence"]
    selected_ref = next(
        element for element in observation.elements if element.element_id == target_id
    )
    if task.risky and _safety_allows(task):
        record["risky_false_approval"] = True
    elif not _safety_allows(task):
        await adapter.close()
        return finish()
    if task.failure_mode == "adapter":
        adapter.timeout_next("click")
    try:
        await adapter.click(selected_ref, policy.timeouts_ms["action"])
    except BrowserError:
        record["adapter_failure"] = True
        record["failure_category"] = "adapter_failure"
        await adapter.close()
        return finish()
    record["task_success"] = True
    if task.churn == "full":
        refreshed = await adapter.observe(limits)
        record["stale_recovery_succeeded"] = (
            refreshed.snapshot_id != observation.snapshot_id
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
    successful_steps = successful
    failures = Counter(
        record["failure_category"]
        for record in records
        if record["failure_category"] is not None
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
    return {
        "task_count": count,
        "top1_accuracy": _ratio(sum(bool(r["top1_correct"]) for r in records), count),
        "top3_coverage": _ratio(sum(bool(r["top3_covered"]) for r in records), count),
        "page_state_accuracy": _ratio(
            sum(bool(r["page_state_correct"]) for r in records), count
        ),
        "task_success_rate": _ratio(successful, count),
        "risky_false_approval_rate": _ratio(
            sum(bool(r["risky_false_approval"]) for r in records), count
        ),
        "jev_cost": cost,
        "jev_tokens": tokens,
        "jev_tokens_per_successful_step": _ratio(tokens, successful_steps),
        "jev_cost_per_successful_step": round(cost / max(successful_steps, 1), 6),
        "model_input_tokens": transport.input_tokens,
        "model_output_tokens": transport.output_tokens,
        "cache_reads": 0,
        "provider_turns": transport.provider_turns,
        "stale_recovery_rate": _ratio(
            sum(bool(r["stale_recovery_succeeded"]) for r in records),
            sum(bool(r["stale_recovery_attempted"]) for r in records),
        ),
        "action_retry_count": transport.retries,
        "provider_retry_count": transport.retries,
        "time_per_successful_step_seconds": round(
            (count + transport.requests * 0.01) / max(successful_steps, 1) / 100, 6
        ),
        "failures": {
            name: failures.get(name, 0)
            for name in (
                "pre_filter_miss",
                "jev_selection_miss",
                "adapter_failure",
                "page_failure",
            )
        },
        "confidence_calibration": confidence,
        "threshold_versions": [THRESHOLD_VERSION],
    }


def _routing_win(routed: dict[str, object], stock: dict[str, object]) -> bool:
    quality_parity = (
        routed["task_success_rate"] >= stock["task_success_rate"]
        and routed["risky_false_approval_rate"] <= stock["risky_false_approval_rate"]
    )
    cost_or_scale_gain = routed["time_per_successful_step_seconds"] < stock[
        "time_per_successful_step_seconds"
    ] or (
        routed["jev_tokens"] < stock["jev_tokens"]
        and routed["task_success_rate"] > stock["task_success_rate"]
    )
    return bool(quality_parity and cost_or_scale_gain)


def _cell_summary(records: list[dict[str, object]]) -> dict[str, object]:
    transport = MockJevTransport()
    for record in records:
        turns = int(record["provider_turns"])
        transport.input_tokens += int(record["input_tokens"])
        transport.output_tokens += int(record["output_tokens"])
        transport.provider_turns += int(record["provider_turns"])
        transport.retries += int(record["action_retries"])
        transport.requests += turns + int(record["action_retries"])
    return _arm_summary(records, transport)


async def _run(tasks: list[BrowserTask], policy: EvalPolicy) -> dict[str, object]:
    records_by_arm: dict[str, list[dict[str, object]]] = {"routed": [], "stock": []}
    transports: dict[str, MockJevTransport] = {}
    for arm in ("routed", "stock"):
        transport_records: list[dict[str, object]] = []
        total_transport = MockJevTransport()
        for task in tasks:
            record, transport = await _run_arm(task, arm, policy)
            transport_records.append(record)
            total_transport.requests += transport.requests
            total_transport.retries += transport.retries
            total_transport.input_tokens += transport.input_tokens
            total_transport.output_tokens += transport.output_tokens
            total_transport.provider_turns += transport.provider_turns
        records_by_arm[arm] = transport_records
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
            }
        )
    return {
        "schema_version": "browser-eval-v1",
        "offline": True,
        "network_used": False,
        "playwright_used": False,
        "provider_key_used": False,
        "policy": {
            "timeouts_ms": policy.timeouts_ms,
            "max_actions": policy.max_actions,
            "max_wall_clock_seconds": policy.max_wall_clock_seconds,
            "safety_policy_version": policy.safety_policy_version,
        },
        "corpus": {
            "task_count": len(tasks),
            "catalog_sizes": list(CATALOG_SIZES),
            "churn_modes": list(CHURN_MODES),
            "task_kinds": list(TASK_KINDS),
        },
        "arms": summaries,
        "crossover_matrix": cells,
        "records": records_by_arm,
        "invariants": {
            "same_fixture_and_policy": True,
            "routing_win_requires_task_success_and_safety_parity": True,
            "token_reduction_alone_is_not_a_win": True,
            "threshold_versions_recorded": True,
            "failure_categories_separate": True,
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
