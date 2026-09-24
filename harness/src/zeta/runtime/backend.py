"""Provider-neutral backend selection for frontend composition."""

from __future__ import annotations

from pathlib import Path

from ..protocol.types import CompletionBackend
from ..providers.factory import build_backend as build_network_backend
from .fake_backend import FakeInteractiveBackend


def build_backend(
    provider: str,
    model: str | None,
    *,
    home: str | Path | None = None,
    stall_seconds: float | None = None,
    stall_retries: int | None = None,
) -> tuple[CompletionBackend, str]:
    """Build the selected provider without loading fake credentials."""

    if provider == "fake":
        selected_model = model or "offline"
        return FakeInteractiveBackend(model=selected_model), selected_model
    return build_network_backend(
        provider,
        model,
        home=home,
        stall_seconds=stall_seconds,
        stall_retries=stall_retries,
    )


__all__ = ["build_backend"]
