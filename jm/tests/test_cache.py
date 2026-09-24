from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_v4_cache import QUESTIONS, _answer

from jm.answers import JudgeResponse
from jm.cache import CacheStore
from jm.client import build_canonical_request


def _seed(store: CacheStore):
    request = build_canonical_request(
        {"focus": "focus", "context": {"query": "q"}}, QUESTIONS
    )
    return store.publish(
        request.payload["state"],
        _answer(),
        battery=QUESTIONS,
        preset="p",
        preset_version="1",
        configured_model="typesafe-ai/jev",
        transport_identity=request.transport_identity,
        state_ref="ref",
    )


def test_malformed_and_partial_v3_entries_are_cache_misses(tmp_path: Path) -> None:
    store = CacheStore(tmp_path)
    entry = _seed(store)
    path = store.path_for(entry.cache_key)
    payload = json.loads(path.read_text())
    del payload["response"]
    path.write_text(json.dumps(payload))
    assert store.get(entry.cache_key) is None


def test_unknown_entry_fields_are_rejected(tmp_path: Path) -> None:
    store = CacheStore(tmp_path)
    entry = _seed(store)
    path = store.path_for(entry.cache_key)
    payload = json.loads(path.read_text())
    payload["wire_request"] = {}
    path.write_text(json.dumps(payload))
    assert store.get(entry.cache_key) is None


@pytest.mark.parametrize("state_ref", [None, 3, ""])
def test_unnamed_state_ref_cells_are_not_accepted(
    tmp_path: Path, state_ref: object
) -> None:
    store = CacheStore(tmp_path)
    entry = _seed(store)
    path = store.path_for(entry.cache_key)
    payload = json.loads(path.read_text())
    payload["wire_state"]["context"]["state_ref"] = state_ref
    path.write_text(json.dumps(payload))
    assert store.get(entry.cache_key) is None


def test_publish_requires_complete_answers(tmp_path: Path) -> None:
    store = CacheStore(tmp_path)
    request = build_canonical_request(
        {"focus": "focus", "context": {"query": "q"}}, QUESTIONS
    )
    with pytest.raises(ValueError, match="complete"):
        store.publish(
            request.payload["state"],
            JudgeResponse({}, ("match",)),
            battery=QUESTIONS,
            preset="p",
            preset_version="1",
            configured_model="typesafe-ai/jev",
            transport_identity=request.transport_identity,
            state_ref="ref",
        )
