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
    chunk_input,
    chunk_line,
    chunk_para,
    chunk_record,
    decode_stdin,
)
from jmap.runner import State, StateLimits, validate_state


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


def test_hunks_keep_file_ownership_across_files_and_deleted_files() -> None:
    diff = """\
diff --git a/first.py b/first.py
--- a/first.py
+++ b/first.py
@@ -1 +1 @@
-old
+new
diff --git a/second.py b/second.py
--- a/second.py
+++ b/second.py
@@ -2 +2 @@
-old
+new
"""
    states = chunk_hunk(diff)
    assert [state.context["file"] for state in states] == ["first.py", "second.py"]
    assert "second.py" not in states[0].focus

    deleted = """\
diff --git a/deleted.py b/deleted.py
--- a/deleted.py
+++ /dev/null
@@ -1 +0,0 @@
-gone
"""
    assert chunk_hunk(deleted)[0].state_ref == "deleted.py@@-1+0,0"


def test_record_requires_selected_stable_identity() -> None:
    with pytest.raises(ValueError, match="id"):
        chunk_record('{"kind":"payment"}\n')
    with pytest.raises(ValueError, match="event_id"):
        chunk_record('{"id":"evt-7"}\n', state_ref_field="event_id")


def test_record_metadata_is_opt_in_and_selected() -> None:
    state = chunk_record(
        '{"id":"evt-7","kind":"payment","secret":"do not include"}\n',
        metadata_fields=["kind"],
    )[0]
    assert state.context["metadata"] == {"kind": "payment"}

    default_state = chunk_record('{"id":"evt-8","kind":"payment"}\n')[0]
    assert default_state.context["metadata"] == {}


def test_unselected_record_payload_does_not_affect_state_admission() -> None:
    record = {"id": "evt-9", "selected": "ok", "unselected": "x" * 1000}
    result = chunk_input(
        "record",
        json.dumps(record),
        metadata_fields=["selected"],
        limits=StateLimits(
            focus_bytes=2_000, context_field_bytes=100, state_bytes=1_200
        ),
    )
    assert result.discovered == result.judged == 1
    assert result.rejections == ()
    assert result.admitted[0].context["metadata"] == {"selected": "ok"}


@pytest.mark.parametrize("value", [None, {"nested": "value"}, ["list"]])
def test_record_rejects_non_scalar_identity_per_record(value: object) -> None:
    result = chunk_input("record", json.dumps({"id": value, "kind": "x"}))
    assert result.admitted == ()
    assert len(result.rejections) == 1
    assert result.rejections[0].reason == "input_error"
    assert result.rejections[0].state_ref is None
    assert result.discovered == result.judged + result.skipped_count == 0


def test_record_rejects_duplicate_stable_refs() -> None:
    result = chunk_input(
        "record",
        '{"id":1,"value":"first"}\n{"id":"1","value":"second"}\n',
    )
    assert [state.state_ref for state in result.admitted] == ["1"]
    assert len(result.rejections) == 1
    assert result.rejections[0].state_ref == "1"
    assert result.rejections[0].reason == "input_error"


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


def test_chunk_input_rejects_one_oversized_state_and_continues() -> None:
    records = "\n".join(
        [
            json.dumps({"path": "ok.txt", "content": "ok"}),
            json.dumps({"path": "large.txt", "content": "large"}),
        ]
    )
    result = chunk_input(
        "file",
        records,
        limits=StateLimits(focus_bytes=4, context_field_bytes=100),
    )
    assert [state.state_ref for state in result.admitted] == ["ok.txt"]
    assert result.rejections[0].state_ref == "large.txt"
    assert result.rejections[0].reason == "context_limit"
    assert result.discovered == result.judged + result.skipped_count == 2


def test_chunk_input_applies_scan_cap_after_complete_discovery() -> None:
    result = chunk_input("line", "one\ntwo\nthree", max_chunks=1)
    assert result.discovered == 3
    assert [state.state_ref for state in result.admitted] == ["stdin#L1"]
    assert result.skipped_count == 2


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


@pytest.mark.parametrize("field", ["state_ref", "focus", "query"])
def test_byte_limit_boundaries_accept_exact_and_reject_plus_one(field: str) -> None:
    limit = 8
    value_factories = {
        "state_ref": lambda size: "r" * size,
        "focus": lambda size: "f" * size,
        "query": lambda size: "q" * size,
    }
    make_value = value_factories[field]
    if field == "state_ref":
        source = make_value(limit - len("#L1"))
        exact = chunk_line(
            "ok\n", source=source, limits=StateLimits(context_field_bytes=limit)
        )
        assert exact[0].state_ref == f"{source}#L1"
        with pytest.raises(StateLimitError, match="state_ref"):
            chunk_line(
                "ok\n",
                source=make_value(limit - len("#L1") + 1),
                limits=StateLimits(context_field_bytes=limit),
            )
    elif field == "focus":
        exact = chunk_line(make_value(limit), limits=StateLimits(focus_bytes=limit))
        assert exact[0].focus == make_value(limit)
        with pytest.raises(StateLimitError, match="focus"):
            chunk_file(
                "exact.txt",
                make_value(limit + 1),
                limits=StateLimits(focus_bytes=limit),
            )
    else:
        exact = chunk_line(
            "ok\n",
            query=make_value(limit),
            limits=StateLimits(context_field_bytes=limit),
        )
        assert exact[0].context["query"] == make_value(limit)
        with pytest.raises(StateLimitError, match="query"):
            chunk_line(
                "ok\n",
                query=make_value(limit + 1),
                limits=StateLimits(context_field_bytes=limit),
            )


def test_structured_context_uses_canonical_json_bytes_at_boundary() -> None:
    value = {"b": 2, "a": 1}
    exact = len(b'{"a":1,"b":2}')
    state = State("ref", "ok", {"structured": value})
    validate_state(state, StateLimits(context_field_bytes=exact))
    assert (
        len(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()) == exact
    )

    with pytest.raises(StateLimitError, match="structured"):
        validate_state(
            state,
            StateLimits(context_field_bytes=exact - 1),
        )


def test_combined_state_limit_accepts_exact_and_rejects_plus_one() -> None:
    state = State("ref", "focus", {"context": "value"})
    payload_bytes = len(
        json.dumps(state.payload, sort_keys=True, separators=(",", ":")).encode()
    )
    validate_state(state, StateLimits(state_bytes=payload_bytes))
    with pytest.raises(StateLimitError, match="state"):
        validate_state(state, StateLimits(state_bytes=payload_bytes - 1))


def test_hostile_inputs_remain_finite_and_decodable() -> None:
    states = chunk_input("line", b"ok\xff\n", limits=StateLimits(focus_bytes=10))
    assert states.admitted[0].focus == "ok\ufffd"
    huge = chunk_input("line", "x" * 10_000, limits=StateLimits(focus_bytes=100))
    assert len(huge.admitted) == 100
    assert chunk_file("empty.txt", b"")[0].focus == ""
