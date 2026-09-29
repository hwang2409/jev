"""Tests for the discard-guard safety controls.

Incident scenario: retrieved items with heading as a string (locomo checkpoint
export shape) instead of a list caused prepare_candidates to discard every item
as invalid.  All 446 cases became zero-candidate markers, safety_result froze
an authoritative pass with zero batteries scored — vacuously true.

Fix 1: score_cases must detect shape-mismatch (raw >=1 item, 0 valid
        candidates) and record discarded_all: true; safety_result must
        refuse the run when ANY such case exists.
Fix 2: safety_result/freeze must refuse when zero batteries were scored.
Fix 3: run_repeatability must name the marker/zero-candidate cases.
"""
from __future__ import annotations

import pytest
import run


def _locomo_fixture_with_bad_shape(
    n_bad: int = 446,
) -> list[dict]:
    """Build a locomo-shaped fixture where retrieved items have invalid shape.

    This reproduces the EXACT real incident shape: checkpoint export records
    carry ``source_path`` (not ``path``) and a STRING heading.  Production
    normalization accepts string headings (routing.py:346-351) but requires a
    string ``path``; its absence makes ``_memory_key`` return ``None`` and
    every item is discarded.  The per-item discard is production-faithful;
    the fix detects that ALL items were discarded.
    """
    conversations: list[dict] = []
    total = 0
    for i in range(10):
        if i < 9:
            n_q = 45
        else:
            n_q = 446 - total
        questions = []
        for j in range(n_q):
            q_index = total + j
            if q_index < n_bad:
                # EXACT incident shape: source_path instead of path, string
                # heading.  String heading alone is fine in production; the
                # missing ``path`` is what discards the item.
                retrieved = [
                    {
                        "excerpt": f"Some memory content for q{j} conv{i}",
                        "source_path": f"conversation-{i:02d}--session-{j}.md",
                        "heading": f"turn D{j}:1 | speaker: test",  # string
                        "project": f"locomo-{i}",
                        "rank": 1,
                    }
                ]
            else:
                # GOOD shape: heading is a list
                retrieved = [
                    {
                        "excerpt": f"Some memory content for q{j} conv{i}",
                        "path": f"memories/conv{i}/doc{j}.md",
                        "heading": [f"Section {j}"],  # list — correct
                    }
                ]
            q: dict = {
                "question": f"Question {j} of conversation {i}",
                "category": 5,
                "retrieved": retrieved,
                "retrieval_provenance": {
                    "pipeline": "pausanias",
                    "pipeline_revision": "pinned",
                    "search_config": {"mode": "production"},
                },
            }
            questions.append(q)
        conversations.append({"conversation_id": f"conv-{i}", "qa": questions})
        total += n_q
    assert total == 446
    return conversations


def _locomo_fixture_genuine_empty(n_conversations: int = 10) -> list[dict]:
    """Build a locomo-shaped fixture where all cases have genuine empty retrieval.

    Every case has retrieved=[] with valid retrieval_provenance, simulating
    a genuine zero-retrieval result from the pausanias pipeline.
    """
    conversations: list[dict] = []
    total = 0
    for i in range(n_conversations):
        if i < 9:
            n_q = 45
        else:
            n_q = 446 - total
        questions = []
        for j in range(n_q):
            q: dict = {
                "question": f"Question {j} of conversation {i}",
                "category": 5,
                "retrieved": [],  # genuinely empty
                "retrieval_provenance": {
                    "pipeline": "pausanias",
                    "pipeline_revision": "pinned",
                    "search_config": {"mode": "production"},
                },
            }
            questions.append(q)
        conversations.append({"conversation_id": f"conv-{i}", "qa": questions})
        total += n_q
    assert total == 446
    return conversations


def _build_fake_response(n_candidates: int, scores: list[float] | None = None):
    if scores is None:
        scores = [0.7] * n_candidates
    return {
        "answers": {
            f"memory_relevance_{i}": {"noul": scores[i]}
            for i in range(n_candidates)
        },
        "configured_model": "test-model",
        "served_model": "test-served",
    }


# ---------------------------------------------------------------------------
# Fix 1: shape-mismatch detection -> refusal
# ---------------------------------------------------------------------------

class TestShapeMismatchRefusal:
    """String-heading retrieved items must cause a refusal naming the case
    and the shape problem, NOT a vacuous pass."""

    def test_string_heading_causes_shape_mismatch_refusal(self, tmp_path):
        """score_cases marks discarded_all on marker rows; safety_result
        refuses the run, naming the case IDs and the offending item shape."""
        locomo_data = _locomo_fixture_with_bad_shape(n_bad=446)
        cases = run.safety_cases(locomo_data)

        # score_cases should mark shape-mismatch cases with discarded_all
        scores = run.score_cases(
            cases, lambda _req: _build_fake_response(0),
            model="test-model", cache_dir=tmp_path / "cache",
        )

        # Verify marker rows carry discarded_all: true
        markers = [r for r in scores if r.get("candidates") == []]
        assert len(markers) == 446
        assert all(r.get("discarded_all") is True for r in markers)

        # safety_result must refuse
        run_dir = tmp_path / "safety"
        run_dir.mkdir()
        expected_ids = {str(c["case_id"]) for c in cases}
        with pytest.raises(ValueError, match="shape.mismatch|discarded.*all"):
            run.safety_result(
                scores, 0.6,
                witness_commit="abc123",
                output=run_dir / "safety.json",
                scores_output=run_dir / "safety-scores.jsonl",
                expected_case_ids=expected_ids,
            )

    def test_genuine_empty_retrieval_still_accepted(self, tmp_path):
        """Cases with genuinely empty retrieval (retrieved=[]) must still
        pass through safety_cases and score_cases without error."""
        locomo_data = _locomo_fixture_genuine_empty()
        cases = run.safety_cases(locomo_data)

        scores = run.score_cases(
            cases, lambda _req: _build_fake_response(0),
            model="test-model", cache_dir=tmp_path / "cache",
        )

        # Genuine empties should NOT have discarded_all
        markers = [r for r in scores if r.get("candidates") == []]
        assert len(markers) == 446
        assert all(r.get("discarded_all") is not True for r in markers)

    def test_shape_mismatch_error_names_case_and_keys(self, tmp_path):
        """The refusal error message must name case IDs and the offending
        item's keys/types so the operator can diagnose."""
        # Use just a few bad cases in an otherwise valid dataset
        locomo_data = _locomo_fixture_with_bad_shape(n_bad=446)
        cases = run.safety_cases(locomo_data)

        scores = run.score_cases(
            cases, lambda _req: _build_fake_response(0),
            model="test-model", cache_dir=tmp_path / "cache",
        )

        run_dir = tmp_path / "safety"
        run_dir.mkdir()
        expected_ids = {str(c["case_id"]) for c in cases}
        with pytest.raises(ValueError, match="locomo-0-0") as exc_info:
            run.safety_result(
                scores, 0.6,
                witness_commit="abc123",
                output=run_dir / "safety.json",
                scores_output=run_dir / "safety-scores.jsonl",
                expected_case_ids=expected_ids,
            )
        # Error should mention shape/type info
        msg = str(exc_info.value)
        assert "heading" in msg or "str" in msg or "keys" in msg


# ---------------------------------------------------------------------------
# Fix 2: zero-batteries freeze refusal
# ---------------------------------------------------------------------------

class TestZeroBatteriesRefusal:
    """An authoritative safety pass with zero batteries scored is
    definitionally invalid. Even when fix 1 is bypassed (all genuine
    empties), safety_result must refuse to freeze."""

    def test_all_genuine_empties_refuses_freeze(self, tmp_path):
        """All 446 cases genuine-empty -> zero batteries -> refuse freeze.
        This bypasses fix 1 (no discarded_all) but fix 2 still catches it."""
        locomo_data = _locomo_fixture_genuine_empty()
        cases = run.safety_cases(locomo_data)

        scores = run.score_cases(
            cases, lambda _req: _build_fake_response(0),
            model="test-model", cache_dir=tmp_path / "cache",
        )

        run_dir = tmp_path / "safety"
        run_dir.mkdir()
        expected_ids = {str(c["case_id"]) for c in cases}
        with pytest.raises(ValueError, match="zero.batter|no.*scored.candidates"):
            run.safety_result(
                scores, 0.6,
                witness_commit="abc123",
                output=run_dir / "safety.json",
                scores_output=run_dir / "safety-scores.jsonl",
                expected_case_ids=expected_ids,
            )


# ---------------------------------------------------------------------------
# Fix 3: repeatability marker-row naming
# ---------------------------------------------------------------------------

class TestRepeatabilityMarkerNaming:
    """run_repeatability must name the marker/zero-candidate cases in
    its error message, not give a generic identity error."""

    def test_repeatability_names_zero_candidate_cases(self, tmp_path):
        """When marker rows sneak in (all candidates discarded),
        run_repeatability must name the affected case IDs."""
        # Build cases with the correct stratum structure but string headings
        cases = []
        for stratum in ("abstain", "verbatim", "paraphrase"):
            for i in range(4):
                cases.append({
                    "case_id": f"rep-{stratum}-{i}",
                    "query": f"Query {stratum} {i}",
                    "category": stratum,
                    "retrieved": [
                        {
                            "excerpt": f"Some content for {stratum} {i}",
                            "path": f"memories/{stratum}/doc{i}.md",
                            "heading": {"text": "Bad heading"},  # dict — will be discarded
                        }
                    ],
                    "retrieval_provenance": {
                        "pipeline": "pausanias",
                        "pipeline_revision": "pinned",
                        "search_config": {"mode": "production"},
                    },
                })

        def fake_client(_req):
            return _build_fake_response(0)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        with pytest.raises(ValueError, match="rep-") as exc_info:
            run.run_repeatability(
                cases, fake_client, model="test-model",
                cache_dir=cache_dir, tau=0.6,
            )
        msg = str(exc_info.value)
        # Must name the case IDs, not just give a generic identity error
        assert "zero.candidate" in msg.lower() or "marker" in msg.lower() or "discard" in msg.lower()
