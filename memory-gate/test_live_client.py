"""Tests for the live-scoring client construction path.

All tests run offline: the autouse ``refuse_sockets`` fixture in conftest.py
guarantees no real network I/O.  These tests exercise the REAL
``evaluate_production`` adapter with a fake CLIENT (valid production-shaped
responses) and a spy-wrapped CacheStore, verifying both scoring success and
cache interaction.  Three CLI-dispatch tests drive ``run.main`` without
``--responses`` to prove each live lane (calibration, safety, generic) threads
cache_store correctly.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

spec = importlib.util.spec_from_file_location("memory_gate_run", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(run)

import pipeline  # must follow run.py bootstrap

FIXTURE = HERE / "runs" / "fixture-dev"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SpyCacheStore:
    """Wraps a real ``jm.cache.CacheStore`` to record get/publish calls.

    Uses a real CacheStore in a tmpdir so that the jm cache machinery works,
    while letting tests assert on interaction counts.
    """

    def __init__(self, tmp_path: Path) -> None:
        import os
        os.environ.setdefault("JM_CACHE_DIR", str(tmp_path / "jm-cache"))
        from jm.cache import CacheStore
        self._real = CacheStore(root=tmp_path / "jm-cache")
        self.get_calls: list[str] = []
        self.publish_calls: list[tuple[Any, ...]] = []

    def get(self, key: str, questions: Any = None) -> Any:
        self.get_calls.append(key)
        return self._real.get(key, questions)

    def publish(self, *args: Any, **kwargs: Any) -> Any:
        self.publish_calls.append((args, kwargs))
        return self._real.publish(*args, **kwargs)

    # Forward any other attribute access to the real store.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class FakeClient:
    """A client with an ``.evaluate`` method returning valid production responses.

    Returns a Mapping with ``answers``, ``served_model``, and ``usage`` that
    the real ``evaluate_production`` adapter can parse through
    ``parse_judge_response``.
    """

    def __init__(self, *, served_model: str = "test-served") -> None:
        self.calls: list[tuple[Any, ...]] = []
        self._served_model = served_model

    def evaluate(self, state: Any, questions: Any, *, model: str = "") -> Any:
        self.calls.append((state, questions, model))
        # Build valid noul answers for every question ID in the battery.
        answers: dict[str, dict[str, Any]] = {}
        if isinstance(questions, dict):
            for qid in questions:
                answers[qid] = {"noul": 0.85}
        return {
            "answers": answers,
            "served_model": self._served_model,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }


def _git(cwd: Path, *args: str, check: bool = True):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=check, capture_output=True, text=True,
    )


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
# (a) Missing env var → distinct error before any scoring
# ---------------------------------------------------------------------------

class TestMissingEnvVar:

    def test_missing_env_var_raises_live_client_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """build_live_client must fail with LiveClientError naming the env var
        when VERCEL_AI_GATEWAY (and fallbacks) are all unset."""
        monkeypatch.delenv("VERCEL_AI_GATEWAY", raising=False)
        monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
        monkeypatch.delenv("VERCEL_JEV_KEY", raising=False)
        monkeypatch.setattr(Path, "read_text", lambda *a, **kw: (_ for _ in ()).throw(OSError("no zshrc")))

        with pytest.raises(run.LiveClientError, match="VERCEL_AI_GATEWAY"):
            run.build_live_client()

    def test_error_is_distinct_from_system_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VERCEL_AI_GATEWAY", raising=False)
        monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
        monkeypatch.delenv("VERCEL_JEV_KEY", raising=False)
        monkeypatch.setattr(Path, "read_text", lambda *a, **kw: (_ for _ in ()).throw(OSError("no zshrc")))

        with pytest.raises(run.LiveClientError):
            run.build_live_client()
        assert not issubclass(run.LiveClientError, SystemExit)


# ---------------------------------------------------------------------------
# (b) Real evaluate_production with fake client — scoring succeeds and
#     CacheStore sees interaction
# ---------------------------------------------------------------------------

class TestRealAdapterWithFakeClient:

    def test_scoring_succeeds_through_real_adapter(self, tmp_path: Path) -> None:
        """A fake client producing valid production-shaped responses must
        score successfully through the real evaluate_production path."""
        fake = FakeClient()
        spy = SpyCacheStore(tmp_path)

        candidates = [
            {"id": "c0", "path": "p", "heading": [], "excerpt": "test content"},
        ]
        request = pipeline.build_request("test query", candidates)

        result = run._client_response(
            fake, request, model="typesafe-ai/jev", cache_store=spy,
        )

        assert isinstance(result, dict)
        assert "answers" in result
        assert len(fake.calls) == 1, "client.evaluate must be called once"

    def test_cache_store_receives_puts_and_gets(self, tmp_path: Path) -> None:
        """The CacheStore must see get (miss) and publish (store) calls when
        scoring through the real adapter."""
        fake = FakeClient()
        spy = SpyCacheStore(tmp_path)

        candidates = [
            {"id": "c0", "path": "p", "heading": [], "excerpt": "test content"},
        ]
        request = pipeline.build_request("test query", candidates)

        run._client_response(
            fake, request, model="typesafe-ai/jev", cache_store=spy,
        )

        assert len(spy.get_calls) >= 1, "CacheStore.get must be called (cache lookup)"
        assert len(spy.publish_calls) >= 1, "CacheStore.publish must be called (cache store)"

    def test_cache_replay_on_second_call(self, tmp_path: Path) -> None:
        """A second call with the same request must replay from cache (no
        additional client.evaluate call)."""
        fake = FakeClient()
        spy = SpyCacheStore(tmp_path)

        candidates = [
            {"id": "c0", "path": "p", "heading": [], "excerpt": "test content"},
        ]
        request = pipeline.build_request("test query", candidates)

        run._client_response(fake, request, model="typesafe-ai/jev", cache_store=spy)
        assert len(fake.calls) == 1

        # Second call — should replay from cache
        run._client_response(fake, request, model="typesafe-ai/jev", cache_store=spy)
        assert len(fake.calls) == 1, "client must NOT be called on cache hit"
        assert len(spy.get_calls) == 2, "CacheStore.get must be called twice"

    def test_score_cases_threads_cache_store(self, tmp_path: Path) -> None:
        """score_cases with cache_store= must thread it to _client_response,
        which must thread it to evaluate_production."""
        fake = FakeClient()
        spy = SpyCacheStore(tmp_path)

        cases = [
            {"case_id": "test-1", "query": "where?", "scope": {"project": "test"},
             "retrieved": [{"excerpt": "memory content", "path": "m.md", "heading": []}]},
        ]
        rows = run.score_cases(
            cases, fake, model="typesafe-ai/jev",
            cache_store=spy,
        )

        assert rows, "must produce score rows"
        assert rows[0].get("coverage") is True
        assert len(spy.get_calls) >= 1
        assert len(spy.publish_calls) >= 1


# ---------------------------------------------------------------------------
# (c) CacheStore construction via build_live_client
# ---------------------------------------------------------------------------

class TestCacheStoreWiring:

    def test_cache_store_is_constructed(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("VERCEL_AI_GATEWAY", "test-key-for-cache")
        monkeypatch.setenv("JM_CACHE_DIR", str(tmp_path / "jm-cache"))

        fake = FakeClient()
        _client, cache_store = run.build_live_client(_client_factory=lambda: fake)

        from jm.cache import CacheStore
        assert isinstance(cache_store, CacheStore)

    def test_factory_is_called_and_client_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VERCEL_AI_GATEWAY", "test-key-for-wiring")

        fake = FakeClient()
        factory = MagicMock(return_value=fake)

        client, _cache_store = run.build_live_client(_client_factory=factory)

        factory.assert_called_once()
        assert client is fake


# ---------------------------------------------------------------------------
# (d) CLI dispatch tests — each lane threads cache_store when --responses
#     is absent.  Monkeypatch build_live_client to inject FakeClient +
#     SpyCacheStore; verify the store was USED.
# ---------------------------------------------------------------------------

class TestCLIDispatchCalibrationLive:
    """Calibration lane without --responses must thread cache_store."""

    def test_calibration_live_uses_cache_store(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Set up candidates.jsonl and labels.jsonl (required by calibration)
        for name in ("candidates.jsonl", "labels.jsonl"):
            shutil.copy(FIXTURE / name, run_dir / name)

        fake = FakeClient()
        spy = SpyCacheStore(tmp_path / "cache")

        monkeypatch.setattr(run, "build_live_client", lambda **kw: (fake, spy))

        exit_code = run.main([
            "score", "--lane", "calibration",
            "--run", str(run_dir),
            "--model", "typesafe-ai/jev",
        ])

        assert exit_code == 0
        assert (run_dir / "scores.jsonl").exists()
        assert len(spy.get_calls) >= 1, "CacheStore.get must be called in calibration live lane"
        assert len(spy.publish_calls) >= 1, "CacheStore.publish must be called in calibration live lane"
        assert len(fake.calls) >= 1, "Client.evaluate must be called"


def _locomo_fixture() -> list[dict]:
    """Build a locomo-shaped fixture with 10 conversations / 446 questions."""
    conversations = []
    total = 0
    for i in range(10):
        n_q = 45 if i < 9 else 446 - total
        questions = []
        for j in range(n_q):
            questions.append({
                "question": f"Question {j} of conversation {i}",
                "category": 5,
                "retrieved": [
                    {"excerpt": f"Retrieved text for q{j} conv{i}",
                     "path": f"memories/conv{i}/doc{j}.md",
                     "heading": [f"Section {j}"],
                     "retrieval_provenance": {
                         "pipeline": "pausanias",
                         "pipeline_revision": "pinned",
                         "search_config": {"mode": "production"},
                     }},
                ],
            })
        conversations.append({"conversation_id": f"conv-{i}", "qa": questions})
        total += n_q
    assert total == 446
    return conversations


class TestCLIDispatchSafetyLive:
    """Safety lane without --responses must thread cache_store."""

    def test_safety_live_uses_cache_store(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # Set up git repo with lock (required by safety lane)
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "-b", "main")
        _git(repo, "config", "user.email", "test@example.com")
        _git(repo, "config", "user.name", "test")
        bare = tmp_path / "remote.git"
        bare.mkdir()
        _git(bare, "init", "--bare")
        _git(repo, "remote", "add", "origin", str(bare))

        run_dir = repo / "memory-gate" / "runs" / "20250101-test-model"
        run_dir.mkdir(parents=True)

        # Copy calibration artifacts and create lock
        import lock as lock_module
        for name in ("candidates.jsonl", "scores.jsonl", "labels.jsonl"):
            shutil.copy(FIXTURE / name, run_dir / name)
        (run_dir / "report.md").write_text("# Test report\n\nSynthetic calibration.\n")
        lock_module.write_lock(
            run_dir / "LOCK.json",
            [run_dir / n for n in ("candidates.jsonl", "scores.jsonl", "labels.jsonl", "report.md")],
            0.55,
        )

        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "calibration lock")
        witness = _git(repo, "rev-parse", "HEAD").stdout.strip()
        _git(repo, "push", "origin", "main")

        # Write locomo cases
        cases_file = tmp_path / "locomo-safety.json"
        cases_file.write_text(json.dumps(_locomo_fixture()))

        fake = FakeClient()
        spy = SpyCacheStore(tmp_path / "cache")

        monkeypatch.setattr(run, "build_live_client", lambda **kw: (fake, spy))

        exit_code = run.main([
            "score", "--lane", "safety",
            "--run", str(run_dir),
            "--cases", str(cases_file),
            "--model", "typesafe-ai/jev",
            "--witness", witness,
        ])

        assert exit_code == 0
        assert (run_dir / "safety.json").exists()
        assert len(spy.get_calls) >= 1, "CacheStore.get must be called in safety live lane"
        assert len(spy.publish_calls) >= 1, "CacheStore.publish must be called in safety live lane"
        assert len(fake.calls) >= 1, "Client.evaluate must be called"


class TestCLIDispatchGenericLive:
    """Generic scoring lane without --responses must thread cache_store."""

    def test_generic_live_uses_cache_store(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        cases = [
            {"case_id": "gen-1", "query": "where is the cache?",
             "scope": {"project": "test"},
             "retrieved": [{"excerpt": "The cache is in SQLite.", "path": "docs/cache.md", "heading": ["Storage"]}]},
        ]
        cases_file = tmp_path / "cases.json"
        cases_file.write_text(json.dumps(cases))

        fake = FakeClient()
        spy = SpyCacheStore(tmp_path / "cache")

        monkeypatch.setattr(run, "build_live_client", lambda **kw: (fake, spy))

        exit_code = run.main([
            "score",
            "--lane", "generic",
            "--run", str(run_dir),
            "--cases", str(cases_file),
            "--model", "typesafe-ai/jev",
        ])

        assert exit_code == 0
        assert len(spy.get_calls) >= 1, "CacheStore.get must be called in generic live lane"
        assert len(spy.publish_calls) >= 1, "CacheStore.publish must be called in generic live lane"
        assert len(fake.calls) >= 1, "Client.evaluate must be called"


# ---------------------------------------------------------------------------
# Default model documentation
# ---------------------------------------------------------------------------

class TestDefaultModel:
    """The default model is ``typesafe-ai/jev`` — the gateway model id."""

    def test_default_matches_gateway_model(self) -> None:
        from jm._transport import _GATEWAY_MODEL
        assert run._LIVE_DEFAULT_MODEL == _GATEWAY_MODEL

    def test_default_matches_argparse(self) -> None:
        assert run._LIVE_DEFAULT_MODEL == "typesafe-ai/jev"


# ---------------------------------------------------------------------------
# (e) Fixture compatibility — committed fixture responses parse identically
#     through the production adapter (evaluate_production) and through direct
#     parse_scores.  This locks the claim: "fixture path now routes through
#     the production adapter; verified compatible."
# ---------------------------------------------------------------------------

class TestFixtureCompatibility:
    """The committed fixture responses in runs/fixture-dev and the canonical
    fixtures in fixtures/ must produce identical scores whether parsed via
    the legacy direct path (parse_scores on the raw dict) or the new unified
    path (evaluate_production with a callable returning the dict)."""

    def test_canonical_fixture_through_adapter_matches_direct(self) -> None:
        """fixtures/memory-response.json parsed through evaluate_production
        matches pipeline.parse_scores on the same dict."""
        fixture_response = json.loads(
            (HERE / "fixtures" / "memory-response.json").read_text()
        )
        candidates = [
            {"id": "candidate-0", "path": "p", "heading": [], "excerpt": "test content"},
        ]
        request = pipeline.build_request("test query", candidates)

        # Direct path: parse_scores on the raw fixture dict
        direct_scores = pipeline.parse_scores(fixture_response, candidates)

        # Adapter path: evaluate_production with a callable returning the fixture
        def fixture_client(req):
            return fixture_response

        adapter_response = pipeline.evaluate_production(
            request, fixture_client, model="typesafe-ai/jev",
        )
        adapter_scores = pipeline.parse_scores(adapter_response, candidates)

        assert direct_scores == adapter_scores, (
            "fixture response must produce identical scores through both paths"
        )

    def test_fixture_dev_scores_reproducible_through_adapter(self) -> None:
        """Each case group in runs/fixture-dev/candidates.jsonl, when scored
        through evaluate_production with the matching response shape, produces
        the same score values as runs/fixture-dev/scores.jsonl."""
        candidates_rows = run.read_jsonl(FIXTURE / "candidates.jsonl")
        scores_rows = run.read_jsonl(FIXTURE / "scores.jsonl")

        # Build expected scores from the committed scores.jsonl
        expected: dict[str, float] = {}
        for row in scores_rows:
            if row.get("coverage") and row.get("score") is not None:
                expected[row["candidate_id"]] = row["score"]

        # Group candidates by case_id
        cases_by_id: dict[str, list[dict]] = {}
        for row in candidates_rows:
            cases_by_id.setdefault(row["case_id"], []).append(row)

        # Score each group through the adapter with a fixture-shaped response
        actual: dict[str, float] = {}
        for group in cases_by_id.values():
            n = len(group)
            # Look up committed scores for this group
            case_scores = [
                expected.get(row["candidate_id"], 0.5)
                for row in group
            ]
            # Build a fixture response matching the committed scores
            fixture_resp = {
                "answers": {
                    f"memory_relevance_{i}": {"noul": case_scores[i]}
                    for i in range(n)
                },
            }
            request_candidates = [
                {"id": row["candidate_id"], "path": row["path"],
                 "heading": row["heading"], "excerpt": row["presented_excerpt"]}
                for row in group
            ]
            request = pipeline.build_request(group[0]["query"], request_candidates)

            def make_client(resp):
                return lambda req: resp

            adapter_response = pipeline.evaluate_production(
                request, make_client(fixture_resp), model="typesafe-ai/jev",
            )
            parsed = pipeline.parse_scores(adapter_response, request_candidates)
            for cid, score in parsed.items():
                actual[cid] = score

        # Verify every committed score is reproduced
        for cid, exp_score in expected.items():
            assert cid in actual, f"missing score for {cid}"
            assert abs(actual[cid] - exp_score) < 1e-9, (
                f"score mismatch for {cid}: {actual[cid]} != {exp_score}"
            )
