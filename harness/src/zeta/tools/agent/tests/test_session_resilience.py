import os


from pathlib import Path


import pytest


from zeta.core.session import SessionError, SessionManager


from zeta.core.store import ConversationIntegrityError, ConversationStore


@pytest.mark.parametrize("surface", ["tool", "card"])
def test_child_lifecycle_readers_reject_decoder_surviving_depth(
    tmp_path: Path, surface: str
) -> None:
    from zeta.tools.agent import _read_agent_lifecycle
    from zeta.tui.agent_card import _read_lifecycle

    path = tmp_path / "agent_lifecycle.json"
    path.write_text('{"extra":' + '{"nested":' * 500 + "0" + "}" * 501)
    reader = _read_agent_lifecycle if surface == "tool" else _read_lifecycle
    assert reader(str(tmp_path)) == {}
