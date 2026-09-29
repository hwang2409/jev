"""Suite-wide offline guarantee for memory-gate tests."""
import socket

import pytest


@pytest.fixture(autouse=True)
def refuse_sockets(monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("memory-gate tests must not open sockets")

    monkeypatch.setattr(socket, "socket", refused)
