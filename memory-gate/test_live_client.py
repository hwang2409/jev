"""Tests for the live-scoring client construction path.

All tests run offline: the autouse ``refuse_sockets`` fixture in conftest.py
guarantees no real network I/O.  The REAL ``JevClient`` construction (which
opens an HTTP transport) is exercised only at real-run time — these tests
verify the *wiring* through a fake client factory injected via the
``_client_factory`` seam on ``build_live_client``.
"""
from __future__ import annotations

import importlib.util
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

import pipeline  # noqa: I001  (must follow run.py bootstrap)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeCacheStore:
    """Minimal stand-in for ``jm.cache.CacheStore`` that records puts/gets."""

    def __init__(self) -> None:
        self.puts: list[tuple[str, Any]] = []
        self.gets: list[str] = []
        self._store: dict[str, Any] = {}

    def get(self, key: str, questions: Any = None) -> Any:
        self.gets.append(key)
        return self._store.get(key)

    def publish(self, *args: Any, **kwargs: Any) -> Any:
        # Record that publish was called; return a minimal entry.
        self.puts.append((args, kwargs))
        return None


class FakeClient:
    """A client with an ``.evaluate`` method that returns canned responses."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def evaluate(self, state: Any, questions: Any, *, model: str = "") -> Any:
        self.calls.append((state, questions, model))
        # Return a minimal valid JudgeResponse-like mapping.
        # The actual parsing through evaluate_production will fail because
        # this is not a real JudgeResponse — but the test is checking wiring,
        # not full scoring.
        from jm.answers import JudgeResponse, NoulAnswer
        answers = {}
        for qid in questions:
            answers[qid] = NoulAnswer(True, 0.9)
        return JudgeResponse(
            answers=answers,
            complete=True,
            served_model="typesafe-ai/jev",
        )


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
        # Also ensure the ~/.zshrc fallback doesn't leak a key.
        monkeypatch.setattr(Path, "read_text", lambda *a, **kw: (_ for _ in ()).throw(OSError("no zshrc")))

        with pytest.raises(run.LiveClientError, match="VERCEL_AI_GATEWAY"):
            run.build_live_client()

    def test_error_is_distinct_from_system_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The error type must NOT be SystemExit so callers can distinguish
        'env not set' from 'offline score requires --responses'."""
        monkeypatch.delenv("VERCEL_AI_GATEWAY", raising=False)
        monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
        monkeypatch.delenv("VERCEL_JEV_KEY", raising=False)
        monkeypatch.setattr(Path, "read_text", lambda *a, **kw: (_ for _ in ()).throw(OSError("no zshrc")))

        with pytest.raises(run.LiveClientError):
            run.build_live_client()
        # LiveClientError is a RuntimeError, not SystemExit.
        assert not issubclass(run.LiveClientError, SystemExit)


# ---------------------------------------------------------------------------
# (b) Live path constructs via factory and routes through evaluate_production
# ---------------------------------------------------------------------------

class TestLivePathWiring:

    def test_factory_is_called_and_client_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When _client_factory is provided, build_live_client calls it instead
        of constructing a real JevClient."""
        monkeypatch.setenv("VERCEL_AI_GATEWAY", "test-key-for-wiring")

        fake = FakeClient()
        factory = MagicMock(return_value=fake)

        client, _cache_store = run.build_live_client(_client_factory=factory)

        factory.assert_called_once()
        assert client is fake

    def test_live_client_routes_through_evaluate_production(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The live path must invoke pipeline.evaluate_production (the same
        adapter function used by fixtures), not a parallel path."""
        monkeypatch.setenv("VERCEL_AI_GATEWAY", "test-key-for-wiring")

        fake = FakeClient()
        client, cache_store = run.build_live_client(_client_factory=lambda: fake)

        # Patch evaluate_production to record whether it was called.
        # Return a valid response shape rather than hitting the real judge path
        # (which would require network).
        calls: list[tuple[Any, ...]] = []

        def tracking_evaluate(request: Any, client: Any = None, **kwargs: Any) -> Any:
            calls.append((request, client, kwargs))
            # Minimal valid response with answers for every question.
            answers = {
                qid: 0.8 for qid in request["questions"]
            }
            return {
                "answers": answers,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "served_model": "typesafe-ai/jev",
                "configured_model": "typesafe-ai/jev",
            }

        monkeypatch.setattr(pipeline, "evaluate_production", tracking_evaluate)

        # Build a minimal request.
        candidates = [
            {"id": "c0", "path": "p", "heading": [], "excerpt": "test content"},
        ]
        request = pipeline.build_request("test query", candidates)

        # Call through _client_response — the same function used by both lanes.
        result = run._client_response(client, request, model="typesafe-ai/jev",
                                      cache_store=cache_store)

        assert len(calls) == 1, "evaluate_production must be called exactly once"
        assert calls[0][1] is client, "the fake client must be passed through"
        assert isinstance(result, dict)
        assert "answers" in result


# ---------------------------------------------------------------------------
# (c) CacheStore is wired — fake store sees interaction
# ---------------------------------------------------------------------------

class TestCacheStoreWiring:

    def test_cache_store_is_constructed(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """build_live_client returns a CacheStore (from jm.cache)."""
        monkeypatch.setenv("VERCEL_AI_GATEWAY", "test-key-for-cache")
        monkeypatch.setenv("JM_CACHE_DIR", str(tmp_path / "jm-cache"))

        fake = FakeClient()
        _client, cache_store = run.build_live_client(_client_factory=lambda: fake)

        from jm.cache import CacheStore
        assert isinstance(cache_store, CacheStore)

    def test_cache_store_passed_to_evaluate_production(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The cache_store from build_live_client reaches evaluate_production's
        cache_store parameter through _client_response."""
        monkeypatch.setenv("VERCEL_AI_GATEWAY", "test-key-for-cache")
        monkeypatch.setenv("JM_CACHE_DIR", str(tmp_path / "jm-cache"))

        fake = FakeClient()
        client, cache_store = run.build_live_client(_client_factory=lambda: fake)

        received_cache: list[Any] = []

        def spy_evaluate(request: Any, client: Any = None, **kwargs: Any) -> Any:
            received_cache.append(kwargs.get("cache_store"))
            answers = {qid: 0.8 for qid in request["questions"]}
            return {
                "answers": answers,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "served_model": "typesafe-ai/jev",
                "configured_model": "typesafe-ai/jev",
            }

        monkeypatch.setattr(pipeline, "evaluate_production", spy_evaluate)

        candidates = [
            {"id": "c0", "path": "p", "heading": [], "excerpt": "test content"},
        ]
        request = pipeline.build_request("test query", candidates)
        run._client_response(client, request, model="typesafe-ai/jev",
                             cache_store=cache_store)

        assert len(received_cache) == 1
        assert received_cache[0] is cache_store


# ---------------------------------------------------------------------------
# Default model documentation
# ---------------------------------------------------------------------------

class TestDefaultModel:
    """The default model is ``typesafe-ai/jev`` — the gateway model id from:
    - ``jm/jm/_transport.py:8``  (_GATEWAY_MODEL)
    - ``jm/jm/client.py:413``   (runtime_preset default)
    - ``memory-gate/run.py``     (--model default)
    """

    def test_default_matches_gateway_model(self) -> None:
        from jm._transport import _GATEWAY_MODEL
        assert run._LIVE_DEFAULT_MODEL == _GATEWAY_MODEL

    def test_default_matches_argparse(self) -> None:
        assert run._LIVE_DEFAULT_MODEL == "typesafe-ai/jev"
