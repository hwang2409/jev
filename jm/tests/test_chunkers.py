from __future__ import annotations

import json

import pytest

from jm.chunkers import (
    DEFAULT_CONTEXT_FIELD_BYTES,
    DEFAULT_FOCUS_BYTES,
    DEFAULT_STATE_BYTES,
    SURROUNDING_TRUNCATION_MARKER,
    StateLimitError,
    chunk_file,
    chunk_hunk,
    chunk_input,
    chunk_line,
    chunk_para,
    chunk_record,
    decode_stdin,
)
from jm.runner import State, StateLimits, validate_state


def test_default_byte_limits_are_pinned() -> None:
    limits = StateLimits()
    assert limits.focus_bytes == DEFAULT_FOCUS_BYTES == 16_384
    assert limits.context_field_bytes == DEFAULT_CONTEXT_FIELD_BYTES == 4_096
    assert limits.state_bytes == DEFAULT_STATE_BYTES == 32_768


def test_chunkers_produce_stable_refs() -> None:
    assert chunk_line("one\ntwo\n", source="notes.md")[1].state_ref == "notes.md#L2"
    assert (
        chunk_para("# Intro\n\none\n\ntwo\n", source="notes.md")[0].state_ref
        == "notes.md#P2"
    )
    diff = (
        "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    assert chunk_hunk(diff)[0].state_ref == "app.py@@-1+1"
    assert chunk_file("app.py", "print('ok')")[0].state_ref == "app.py"
    assert chunk_file("a/app.py", "print('ok')")[0].state_ref == "a/app.py"
    git_diff = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n"
    assert chunk_hunk(git_diff)[0].state_ref == "app.py@@-1+1"
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


def test_hunk_body_markers_are_not_file_headers() -> None:
    diff = """\
diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1,2 +1,3 @@
-old
+new
++ literal added
+++ literal added
--- literal removed
"""
    states = chunk_hunk(diff)
    assert len(states) == 1
    assert "++ literal added" in states[0].focus
    assert "+++ literal added" in states[0].focus
    assert "--- literal removed" in states[0].focus


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


def test_record_reports_duplicate_stable_refs_to_the_judge() -> None:
    result = chunk_input(
        "record",
        '{"id":1,"value":"first"}\n{"id":"1","value":"second"}\n',
    )
    assert [state.state_ref for state in result.admitted] == ["1", "1"]


def test_record_jsonl_rejects_bad_lines_and_continues() -> None:
    records = '{"id":"é"}\nnot json\n{"id":"last"}\n'
    result = chunk_input("record", records)
    assert [state.state_ref for state in result.admitted] == ["é", "last"]
    assert len(result.rejections) == 1
    assert result.rejections[0].reason == "input_error"
    assert result.rejections[0].source_ref == (
        f"stdin:byte={len('{\"id\":\"é\"}\n'.encode())},line=2"
    )


def test_record_jsonl_errors_count_blank_lines_and_bytes() -> None:
    records = " \n\n{\"kind\":\"missing-id\"}\n{\"id\":\"last\"}\n"
    result = chunk_input("record", records)
    assert [state.state_ref for state in result.admitted] == ["last"]
    assert len(result.rejections) == 1
    assert result.rejections[0].message.endswith("'id'")
    assert result.rejections[0].source_ref == (
        f"stdin:byte={len(b' \n\n')},line=3"
    )


def test_file_jsonl_rejects_bad_lines_and_continues() -> None:
    records = (
        '{"path":"first.txt","content":"one"}\n'
        "not json\n"
        '{"path":"last.txt","content":"last"}\n'
    )
    result = chunk_input("file", records)
    assert [state.state_ref for state in result.admitted] == ["first.txt", "last.txt"]
    assert len(result.rejections) == 1
    assert result.rejections[0].source_ref == (
        f"stdin:byte={len(b'{\"path\":\"first.txt\",\"content\":\"one\"}\n')},line=2"
    )


def test_file_jsonl_rejects_wrong_field_types_and_continues() -> None:
    records = (
        '{"path":"first.txt","content":"one"}\n'
        '{"path":[],"content":{}}\n'
        '{"path":"last.txt","content":"last"}\n'
    )
    result = chunk_input("file", records)
    assert [state.state_ref for state in result.admitted] == [
        "first.txt",
        "last.txt",
    ]
    assert len(result.rejections) == 1
    assert result.rejections[0].reason == "input_error"
    assert result.rejections[0].source_ref == (
        f"stdin:byte={len(b'{\"path\":\"first.txt\",\"content\":\"one\"}\n')},line=2"
    )
    assert result.judged == 2


def test_hunk_paths_preserve_spaces_decode_quotes_and_strip_timestamps() -> None:
    space_diff = "--- a/src/my file.py\n+++ b/src/my file.py\n@@ -1 +1 @@\n-old\n+new\n"
    quoted_diff = (
        '--- "a/src/quote\\"file.py"\n'
        '+++ "b/src/quote\\"file.py"\n'
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    timestamp_diff = (
        "--- a/src/timed.py\t2026-09-22 12:00:00\n"
        "+++ b/src/timed.py\t2026-09-22 12:01:00\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )

    assert chunk_hunk(space_diff)[0].state_ref == "src/my file.py@@-1+1"
    assert chunk_hunk(quoted_diff)[0].state_ref == 'src/quote"file.py@@-1+1'
    assert chunk_hunk(timestamp_diff)[0].state_ref == "src/timed.py@@-1+1"


def test_invalid_stdin_bytes_use_replacement_characters() -> None:
    assert decode_stdin(b"ok\xff\n") == "ok\ufffd\n"


def test_state_jsonl_rejects_invalid_utf8_without_replacement_text() -> None:
    rejections = []
    result = chunk_input(
        "state",
        b'{"state_ref":"first","focus":"ok","context":{}}\n'
        b'{"state_ref":"bad","focus":"bad\xff","context":{}}\n'
        b'{"state_ref":"last","focus":"ok","context":{}}\n',
    )

    assert [state.state_ref for state in result.formed] == ["first", "last"]
    rejections = result.rejections
    assert len(rejections) == 1
    assert rejections[0].reason == "input_error"
    assert "not valid UTF-8" in rejections[0].message
    assert "\ufffd" not in rejections[0].message
    assert all("\ufffd" not in state.focus for state in result.formed)


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
    states = chunk_line("abcdefghij\nok", limits=StateLimits(focus_bytes=4))
    assert [state.state_ref for state in states] == [
        "stdin#L1/1",
        "stdin#L1/2",
        "stdin#L1/3",
        "stdin#L2",
    ]
    assert "".join(state.focus for state in states[:3]) == "abcdefghij"
    assert states[3].focus == "ok"
    assert [state.context["subunit"] for state in states[:3]] == ["1/3", "2/3", "3/3"]
    assert all(len(state.focus.encode("utf-8")) <= 4 for state in states)


def test_empty_and_unadmittable_inputs_remain_finite() -> None:
    empty = chunk_input("line", "")
    assert empty.admitted == ()
    assert empty.discovered == 0

    exceeds_all_limits = chunk_input(
        "line",
        "abcdefghij",
        limits=StateLimits(focus_bytes=4, state_bytes=1),
    )
    assert exceeds_all_limits.admitted == ()
    assert exceeds_all_limits.discovered == 3
    assert exceeds_all_limits.judged == 0
    assert len(exceeds_all_limits.rejections) == 3


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
    state = chunk_line(
        "one\n" + "x" * 100,
        source="s",
        limits=StateLimits(context_field_bytes=36),
    )[0]
    assert any(
        SURROUNDING_TRUNCATION_MARKER in item
        for item in state.context["surrounding"]
    )
    assert (
        len(
            json.dumps(state.context["surrounding"], separators=(",", ":")).encode()
        )
        == 36
    )
    with pytest.raises(StateLimitError, match="state"):
        chunk_record(
            json.dumps({"id": "x", "payload": "abcdefgh"}) + "\n",
            limits=StateLimits(context_field_bytes=100, state_bytes=20),
        )


def test_surrounding_truncation_is_deterministic_and_uses_the_exact_limit() -> None:
    limits = StateLimits(context_field_bytes=64)
    first = chunk_para("small\n\n" + "x" * 500, limits=limits)[0]
    second = chunk_para("small\n\n" + "x" * 500, limits=limits)[0]

    assert first.context["surrounding"] == second.context["surrounding"]
    assert any(
        SURROUNDING_TRUNCATION_MARKER in item
        for item in first.context["surrounding"]
    )
    assert (
        len(
            json.dumps(first.context["surrounding"], separators=(",", ":")).encode()
        )
        == 64
    )


def test_heading_only_paragraphs_are_not_formed() -> None:
    states = chunk_para("# Intro\n\nbody\n\n## Details\n\nmore\n")

    assert [state.state_ref for state in states] == ["stdin#P2", "stdin#P4"]
    assert states[0].context["heading"] == "# Intro"
    assert states[1].context["heading"] == "## Details"


def test_heading_only_paragraphs_are_typed_skips_with_exact_coverage() -> None:
    result = chunk_input("para", "# Intro\n\nbody\n\n## Details\n\nmore\n")

    assert result.discovered == 4
    assert result.judged == 2
    assert result.discovered == result.judged + result.skipped_count
    assert [rejection.reason for rejection in result.rejections] == [
        "heading_only",
        "heading_only",
    ]
    assert [rejection.state_ref for rejection in result.rejections] == [
        "stdin#P1",
        "stdin#P3",
    ]


def test_hunk_counts_allow_header_like_body_content() -> None:
    diff = (
        "--- a/one.py\n+++ b/one.py\n@@ -1,2 +1,2 @@\n"
        "--- a/literal\n+++ b/literal\n keep\n"
    )

    result = chunk_input("hunk", diff)

    assert result.rejections == ()
    assert "--- a/literal" in result.formed[0].focus
    assert "+++ b/literal" in result.formed[0].focus


def test_hunk_counts_reject_unfinished_and_excess_body_lines() -> None:
    unfinished = chunk_input(
        "hunk",
        "--- a/x\n+++ b/x\n@@ -1,2 +1,2 @@\n-old\n+new\n",
    )
    excess = chunk_input(
        "hunk",
        "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n+extra\n",
    )

    assert unfinished.rejections[0].reason == "input_error"
    assert "declared counts" in unfinished.rejections[0].message
    assert excess.rejections[0].reason == "input_error"
    assert "extra" in excess.rejections[0].message


def test_hunk_invalid_utf8_is_not_repaired_or_judged() -> None:
    result = chunk_input(
        "hunk",
        b"--- a/x\n+++ b/x\n@@ -1,1 +1,1 @@\n-old\n+ne\xffw\n",
    )

    assert result.rejections[0].reason == "input_error"
    assert "byte=40" in result.rejections[0].source_ref
    assert "\ufffd" not in "".join(state.focus for state in result.formed)


def test_surrounding_marker_requires_a_large_enough_context_limit() -> None:
    with pytest.raises(ValueError, match="truncation marker"):
        chunk_line(
            "one\n" + "x" * 100,
            limits=StateLimits(context_field_bytes=8),
        )


def test_invalid_hunk_body_is_an_input_error_and_does_not_swallow_next_hunk() -> None:
    diff = (
        "--- a/one.py\n+++ b/one.py\n@@ -1,2 +1,2 @@\n-old\n"
        "not-a-body\n"
        "diff --git a/two.py b/two.py\n--- a/two.py\n+++ b/two.py\n"
        "@@ -1,1 +1,1 @@\n-old\n+new\n"
    )

    result = chunk_input("hunk", diff)

    assert [state.context["file"] for state in result.formed] == ["one.py", "two.py"]
    assert result.rejections[0].reason == "input_error"
    assert result.rejections[0].source_ref == "stdin:line=5"
    assert result.discovered == result.judged == 2


def test_wrong_hunk_counts_reject_a_following_file_header() -> None:
    diff = (
        "--- a/one.py\n+++ b/one.py\n@@ -1,2 +1,2 @@\n-old\n+new\n"
        "diff --git a/two.py b/two.py\n--- a/two.py\n+++ b/two.py\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )

    result = chunk_input("hunk", diff)

    assert [state.context["file"] for state in result.formed] == [
        "one.py",
        "two.py",
    ]
    assert result.rejections[0].reason == "input_error"
    assert result.discovered == result.judged == 2


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
