"""Production-fidelity memory retrieval, relevance, and packet pipeline.

The routing implementation is intentionally kept as the authority.  Constants,
hashing, request construction, parsing, and rendering primitives are imported
from harness; the small orchestration below mirrors the inline production
sequence in ``runtime/loop/routing.py:413-460`` and ``:535-580``.
"""
from __future__ import annotations

import hashlib
import json
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
    case: Mapping[str, object], retrieved: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    """Apply fresh-session production routing, returning at most ``TOP_K`` items.

    Candidate IDs are allocated only after the same invalid-item discard as
    production (``routing.py:415-433``), and the cap is applied before scoring
    (``routing.py:444-460``).
    """

    raw: list[dict[str, object]] = []
    for item in retrieved:
        excerpt, path, heading = item.get("excerpt"), item.get("path"), item.get("heading", [])
        if not isinstance(excerpt, str) or not excerpt or not isinstance(path, str):
            continue
        if isinstance(heading, str):
            heading = [heading]
        if not isinstance(heading, list) or not all(isinstance(part, str) for part in heading):
            continue
        digest = content_hash(excerpt)
        raw.append({
            "id": f"candidate-{len(raw)}",
            "path": path,
            "heading": list(heading),
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


def parse_scores(response: Mapping[str, Any], candidates: list[dict[str, Any]]) -> dict[str, float]:
    """Parse through production and require complete candidate coverage."""

    answers = response.get("answers")
    if not isinstance(answers, Mapping):
        raise TypeError("missing adapter answers")
    scores = _parse_memory_relevance(dict(answers), candidates)
    expected = {str(candidate.get("id", f"candidate-{i}")) for i, candidate in enumerate(candidates)}
    if set(scores) != expected:
        raise ValueError("partial score coverage")
    return scores


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
    """Preserve retrieval order; strict gate and full-block first-over-budget break."""

    selected: list[Mapping[str, object]] = []
    total = 0
    for candidate in candidates:
        identifier = str(candidate.get("id", candidate.get("candidate_id")))
        score = scores.get(identifier) if scores is not None else None
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
