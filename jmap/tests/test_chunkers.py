from __future__ import annotations

import json

import pytest

from jmap.chunkers import (
    DEFAULT_CONTEXT_FIELD_BYTES,
    DEFAULT_FOCUS_BYTES,
    DEFAULT_STATE_BYTES,
    StateLimitError,
    chunk_file,
    chunk_hunk,
    chunk_line,
    chunk_para,
    chunk_record,
    decode_stdin,
)
from jmap.runner import StateLimits


def test_default_byte_limits_are_pinned() -> None:
    limits = StateLimits()
    assert limits.focus_bytes == DEFAULT_FOCUS_BYTES == 16_384
    assert limits.context_field_bytes == DEFAULT_CONTEXT_FIELD_BYTES == 4_096
    assert limits.state_bytes == DEFAULT_STATE_BYTES == 32_768


def test_chunkers_produce_stable_refs() -> None:
    assert chunk_line("one\ntwo\n", source="notes.md")[1].state_ref == "notes.md#L2"
    assert (
        chunk_para("# Intro\n\none\n\ntwo\n", source="notes.md")[0].state_ref
        == "notes.md#P1"
    )
    diff = (
        "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
        "@@ -1,2 +1,2 @@\n-old\n+new\n"
    )
    assert chunk_hunk(diff)[0].state_ref == "app.py@@-1,2+1,2"
    assert chunk_file("app.py", "print('ok')")[0].state_ref == "app.py"
    assert chunk_record('{"id":"evt-7","kind":"payment"}\n')[0].state_ref == "evt-7"


def test_record_requires_selected_stable_identity() -> None:
    with pytest.raises(ValueError, match="id"):
        chunk_record('{"kind":"payment"}\n')
    with pytest.raises(ValueError, match="event_id"):
        chunk_record('{"id":"evt-7"}\n', state_ref_field="event_id")


def test_invalid_stdin_bytes_use_replacement_characters() -> None:
    assert decode_stdin(b"ok\xff\n") == "ok\ufffd\n"


def test_string_limits_use_utf8_bytes() -> None:
    limits = StateLimits(focus_bytes=1)
    with pytest.raises(StateLimitError):
        chunk_line("é", limits=limits)
    assert len("é".encode()) == 2


def test_identity_and_query_fields_are_rejected_when_oversized() -> None:
    limits = StateLimits(context_field_bytes=3)
    with pytest.raises(StateLimitError, match="state_ref"):
        chunk_line("ok\n", source="long", limits=limits)
    with pytest.raises(StateLimitError, match="query"):
        chunk_para(
            "ok\n",
            query="long query value",
            limits=StateLimits(context_field_bytes=10),
        )


def test_supported_oversized_focus_is_split_into_explicit_subunits() -> None:
    states = chunk_para("abcdefghij", limits=StateLimits(focus_bytes=4))
    assert [state.state_ref for state in states] == [
        "stdin#P1/1",
        "stdin#P1/2",
        "stdin#P1/3",
    ]
    assert "subunit" in states[0].context
    assert all(len(state.focus.encode("utf-8")) <= 4 for state in states)


def test_files_reject_oversized_focus_instead_of_truncating() -> None:
    with pytest.raises(StateLimitError, match="focus"):
        chunk_file("app.py", "abcdefghij", limits=StateLimits(focus_bytes=4))


def test_context_and_complete_state_limits_are_enforced() -> None:
    with pytest.raises(StateLimitError, match="surrounding"):
        chunk_line(
            "one\n" + "x" * 20,
            source="s",
            limits=StateLimits(context_field_bytes=10),
        )
    with pytest.raises(StateLimitError, match="state"):
        chunk_record(
            json.dumps({"id": "x", "payload": "abcdefgh"}) + "\n",
            limits=StateLimits(context_field_bytes=100, state_bytes=20),
        )
