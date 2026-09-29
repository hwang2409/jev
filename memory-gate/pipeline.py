"""Production-fidelity memory retrieval, relevance, and packet pipeline.

The routing implementation is intentionally kept as the authority.  Constants,
hashing, request construction, parsing, and rendering primitives are imported
from harness; the small orchestration below mirrors the inline production
sequence in ``runtime/loop/routing.py:413-460`` and ``:535-580``.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
for _path in (_ROOT / "harness" / "src", _ROOT / "jm", Path("/tmp/pausanias") / "src"):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from zeta.core.context import MEMORY_INJECTION_PREFIX
from zeta.providers.jev import (
    _parse_memory_relevance,
    build_memory_relevance_request,
    runtime_preset,
)
from zeta.runtime.loop import (
    MEMORY_INJECTION_EXCERPT_CHARS,
    MEMORY_INJECTION_TOP_K,
    MEMORY_INJECTION_TOTAL_CHARS,
    _memory_content_hash,
)
from zeta.runtime.loop.routing import RoutingMixin

# Public aliases make the provenance of these values explicit to callers.
EXCERPT_CHARS = MEMORY_INJECTION_EXCERPT_CHARS
TOP_K = MEMORY_INJECTION_TOP_K
TOTAL_CHARS = MEMORY_INJECTION_TOTAL_CHARS


def content_hash(value: str) -> str:
    """Return the production whitespace-normalized, untruncated hash."""

    return _memory_content_hash(value)


def _section_date(candidate: Mapping[str, object]) -> datetime | None:
    """Same parser as ``routing.py:296-307`` (called directly from production)."""

    return RoutingMixin._memory_section_date(candidate)


def _superseded(raw: Sequence[Mapping[str, object]]) -> dict[str, str]:
    """Extracted from ``routing.py:309-340`` for standalone eval use."""

    by_path: dict[str, list[Mapping[str, object]]] = {}
    for candidate in raw:
        path = candidate.get("path")
        if isinstance(path, str):
            by_path.setdefault(str(Path(path).expanduser().resolve()), []).append(candidate)
    skipped: dict[str, str] = {}
    for path, candidates in by_path.items():
        dates = [date for candidate in candidates if (date := _section_date(candidate)) is not None]
        newest = max(dates) if dates else None
        for candidate in candidates:
            candidate_id = candidate.get("id")
            if not isinstance(candidate_id, str):
                continue
            date = _section_date(candidate)
            if newest is not None and (date is None or date < newest):
                skipped[candidate_id] = "superseded"
    return skipped


def _active_paths(case: Mapping[str, object]) -> set[str]:
    values = case.get("actively_modified_paths", case.get("actively_modified", []))
    if isinstance(values, str):
        values = [values]
    return {
        str(Path(value).expanduser().resolve())
        for value in values
        if isinstance(value, str)
    } if isinstance(values, Sequence) else set()


def _known_hashes(case: Mapping[str, object]) -> set[str]:
    values = case.get("known_content_hashes", [])
    result = {value for value in values if isinstance(value, str)} if isinstance(values, Sequence) else set()
    known = case.get("known_candidates", [])
    if isinstance(known, Sequence):
        result.update(
            value["content_hash"] for value in known
            if isinstance(value, Mapping) and isinstance(value.get("content_hash"), str)
        )
    return result


def prepare_candidates(
    case: Mapping[str, object], retrieved: Sequence[object]
) -> list[dict[str, object]]:
    """Apply fresh-session production routing, returning at most ``TOP_K`` items.

    Candidate IDs are allocated only after the same invalid-item discard as
    production (``routing.py:415-433``), and the cap is applied before scoring
    (``routing.py:444-460``).
    """

    raw: list[dict[str, object]] = []
    for item in retrieved:
        if not isinstance(item, Mapping):
            continue
        excerpt = item.get("excerpt")
        if not isinstance(excerpt, str) or not excerpt:
            continue
        digest = content_hash(excerpt)
        key = RoutingMixin._memory_key({**item, "content_hash": digest})
        if key is None:
            continue
        raw.append({
            "id": f"candidate-{len(raw)}",
            "path": key[0],
            "heading": list(key[1]),
            "excerpt": excerpt[:EXCERPT_CHARS],
            "content_hash": digest,
        })

    skipped = _superseded(raw)
    active = _active_paths(case)
    for item in raw:
        if str(Path(str(item["path"])).expanduser().resolve()) in active:
            skipped[str(item["id"])] = "actively_modified"
    known_hashes = _known_hashes(case)
    selected: list[dict[str, object]] = []
    seen_hashes = set(known_hashes)
    for item in raw:
        if len(selected) >= TOP_K:
            break
        candidate_id = str(item["id"])
        if candidate_id in skipped:
            continue
        digest = str(item["content_hash"])
        if digest in seen_hashes:
            continue
        selected.append(item)
        seen_hashes.add(digest)
    return selected


def build_request(query: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Use the production adapter's canonical request builder."""

    return build_memory_relevance_request(query, candidates)


def formation_state(request: Mapping[str, Any]) -> Any:
    """Form the standalone-path ``jm.client.State`` exactly as ``jev.py:1052``."""

    from jm.client import State

    return State("harness", json.dumps(request["state"], ensure_ascii=False, sort_keys=True))


class AdapterError(RuntimeError):
    """The complete production judgment path could not produce valid coverage."""

    def __init__(self, message: str, *, http_status: int | None = None, gate: str | None = None):
        super().__init__(message)
        self.http_status = http_status
        self.gate = gate


def _decode_records(records: Sequence[Any], *, gate: str | None = None,
                    question_ids: Sequence[str] = ()) -> Any:
    """Decode terminal records in the exact production order."""
    payloads = [record.to_dict() for record in records]
    coverage = next((payload for payload in payloads if payload.get("record_type") == "coverage"), None)
    if coverage is None:
        raise AdapterError("Jev judgment did not produce terminal coverage", gate=gate)

    result = next((payload for payload in payloads if payload.get("record_type") == "result"), None)
    if result is None:
        error = next((payload for payload in payloads if payload.get("record_type") == "error"), None)
        error_message = (
            error.get("error", {}).get("message", "Jev judgment failed")
            if error is not None and isinstance(error.get("error"), Mapping)
            else ("Jev judgment failed" if error is None
                  else str(error.get("error") or "Jev judgment failed"))
        )
        # Production specializations (browser/search) — none apply for
        # memory relevance questions, but mirror the exact branching so
        # the eval path is a semantic copy of the production block.
        question_id_set = set(question_ids)
        if error_message == "malformed answer" or error is None:
            if "element_id" in question_id_set:
                error_message = "invalid Jev browser choice response"
            elif "page_loaded_and_stable" in question_id_set:
                error_message = "invalid Jev browser page-state response"
            elif any(qid.startswith("result-") for qid in question_id_set):
                error_message = "invalid Jev search result score response"
        elif any(qid.startswith("result-") for qid in question_id_set):
            error_message = "invalid Jev search result score response"
        error_detail = error.get("error") if error is not None else {}
        detail = str(error_detail) if error is not None else error_message
        raise AdapterError(
            error_message if error_message != "Jev judgment failed" else detail,
            http_status=(error_detail.get("http_status")
                         if isinstance(error_detail, Mapping)
                         and isinstance(error_detail.get("http_status"), int) else None),
            gate=gate,
        )

    if coverage.get("coverage") != "complete":
        raise AdapterError("Jev judgment returned partial coverage", gate=gate)
    # The caller consumes the original record object, not this payload.
    return next(record for record in records if record.to_dict().get("record_type") == "result")


def evaluate_production(
    request: Mapping[str, Any], client: Any = None, *, cache_store: Any = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Run the production ``runtime_preset``/``State``/``judge`` path.

    ``client`` is the sole seam: it is either a JevClient-like ``evaluate``
    object or a callable accepting ``(state, questions, model)``.  Everything
    around that boundary is the same synchronous runner used by production,
    including formation, parsing, terminal coverage, and the jm cache.
    """
    from jm.client import FormationReport, judge

    questions = request["questions"]
    state = formation_state(request)
    preset = runtime_preset(questions)
    if model is not None and model != preset.model:
        from dataclasses import replace
        preset = replace(preset, data={**preset.data, "model": model})
    if client is None:
        judge_fn = None
    elif hasattr(client, "evaluate"):
        def judge_fn(call_state: Any, call_questions: Any, configured_model: str) -> Any:
            response = client.evaluate(call_state, call_questions, model=configured_model)
            if isinstance(response, Mapping):
                from dataclasses import replace

                from jm.answers import parse_judge_response
                answers = response.get("answers", {})
                answers = {
                    key: ({"type": "noul", **value} if isinstance(value, Mapping) and "type" not in value else value)
                    for key, value in answers.items()
                } if isinstance(answers, Mapping) else answers
                parsed = parse_judge_response({"answers": answers}, call_questions)
                served = response.get("served_model")
                return replace(parsed, served_model=served, usage=response.get("usage"))

            return response
    else:
        # The legacy one-argument adapter is selected before invocation.  Never
        # use a TypeError from inside a transport as protocol negotiation.
        try:
            parameters = inspect.signature(client).parameters
            legacy_callable = len(parameters) == 1
        except (TypeError, ValueError):
            legacy_callable = False

        def judge_fn(call_state: Any, call_questions: Any, configured_model: str) -> Any:
            response = client(request) if legacy_callable else client(call_state, call_questions, configured_model)
            if isinstance(response, Mapping):
                from jm.answers import parse_judge_response
                payload = response
                answers = payload.get("answers")
                if isinstance(answers, Mapping):
                    answers = {
                        key: ({"type": "noul", **value} if isinstance(value, Mapping) and "type" not in value else value)
                        for key, value in answers.items()
                    }
                    payload = {**payload, "answers": answers}
                from dataclasses import replace
                parsed = parse_judge_response(payload, call_questions)
                return replace(parsed, served_model=payload.get("served_model"), usage=payload.get("usage"))
            return response
    records = list(judge(
        preset, (state,), formation_report=FormationReport(),
        cache_store=cache_store, judge_fn=judge_fn,
    ))
    result = _decode_records(records, question_ids=tuple(questions))
    payload = result.to_dict()
    return {
        "answers": payload["answers"],
        "usage": result.usage,
        "served_model": (
            payload.get("meta", {}).get("served_model")
            if payload.get("meta", {}).get("served_model") not in {None, "unknown"}
            else None
        ),
        "configured_model": payload.get("meta", {}).get("model"),
    }


class ScoreValidationError(ValueError):
    """A score artifact cannot safely drive memory gating."""


def validate_scores(
    candidates: Sequence[Mapping[str, object]], scores: Mapping[str, object] | None,
) -> dict[str, float]:
    """Require exactly one finite, in-range score for every candidate."""

    expected = {
        str(candidate.get("id", candidate.get("candidate_id", f"candidate-{i}")))
        for i, candidate in enumerate(candidates)
    }
    if scores is None or set(scores) != expected or len(expected) != len(candidates):
        raise ScoreValidationError("incomplete or non-one-to-one score coverage")
    validated: dict[str, float] = {}
    for identifier, score in scores.items():
        if (
            not isinstance(identifier, str)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ScoreValidationError(f"invalid score for {identifier!r}")
        validated[identifier] = float(score)
    return validated


def parse_scores(response: Mapping[str, Any], candidates: list[dict[str, Any]]) -> dict[str, float]:
    """Parse through production and require valid complete candidate coverage."""

    answers = response.get("answers")
    if not isinstance(answers, Mapping):
        raise TypeError("missing adapter answers")
    return validate_scores(candidates, _parse_memory_relevance(dict(answers), candidates))


def render_block(candidate: Mapping[str, object]) -> str:
    """Exact text assembled by ``routing.py:543-568``."""

    heading_value = candidate.get("heading", [])
    heading = " > ".join(heading_value) if isinstance(heading_value, list) else "(document)"
    heading = heading or "(document)"
    excerpt = candidate.get("excerpt", candidate.get("presented_excerpt", ""))
    return f"{MEMORY_INJECTION_PREFIX}\npath: {candidate['path']}\nheading: {heading}\n{excerpt}"


def select_blocks(
    candidates: Sequence[Mapping[str, object]],
    scores: Mapping[str, float] | None,
    tau: float,
    *,
    no_gate: bool = False,
) -> list[Mapping[str, object]]:
    """Validate scores, then preserve order and apply the strict gate."""

    validated_scores = validate_scores(candidates, scores)
    selected: list[Mapping[str, object]] = []
    total = 0
    for candidate in candidates:
        identifier = str(candidate.get("id", candidate.get("candidate_id")))
        score = validated_scores[identifier]
        if not no_gate and (score is None or not score > tau):
            continue
        block = render_block(candidate)
        if total + len(block) > TOTAL_CHARS:
            break
        selected.append(candidate)
        total += len(block)
    return selected


# Compatibility names used by the existing offline runner/tests.
def production_adapter() -> tuple[Any, Any]:
    return build_memory_relevance_request, _parse_memory_relevance


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


production_request = build_request
parse_production_scores = parse_scores
