"""Serve deterministic, non-production browser smoke pages."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Self
from urllib.parse import parse_qs, urlsplit

_ROOT = Path(__file__).with_name("index.html")


def _layout(title: str, body: str) -> bytes:
    navigation = """
    <nav aria-label="fixture navigation">
      <a href="/">Home</a>
      <a href="/search">Search</a>
      <a href="/form">Form</a>
      <a href="/stale">Stale state</a>
      <a href="/low-confidence">Low confidence</a>
    </nav>
    """
    document = "<!doctype html><html><head><meta charset='utf-8'>"
    document += f"<title>{escape(title)}</title></head><body>{navigation}"
    document += f"<main>{body}</main></body></html>"
    return document.encode("utf-8")


def _search_page() -> bytes:
    body = """
    <h1>Search fixture</h1>
    <label for="filter">Filter results</label>
    <input id="filter" name="filter" type="text">
    <button type="button" id="apply-filter" onclick="filterResults()">Apply filter</button>
    <section aria-label="fixture search results">
      <article data-search-result data-result-id="local-guide">
        <h2>Local browser guide</h2>
        <p>Deterministic guidance for the local browser fixture.</p>
        <a href="/form">Open the harmless form</a>
      </article>
      <article data-search-result data-result-id="state-guide">
        <h2>State transition guide</h2>
        <p>Deterministic guidance for stale state recovery.</p>
        <a href="/stale">Open stale state</a>
      </article>
      <article data-search-result data-result-id="unrelated">
        <h2>Unrelated fixture note</h2>
        <p>This result is intentionally low relevance.</p>
        <a href="/low-confidence">Open controls</a>
      </article>
    </section>
    <script>
      function filterResults() {
        const query = document.getElementById('filter').value.toLowerCase();
        for (const result of document.querySelectorAll('[data-search-result]')) {
          result.hidden = query !== '' && !result.innerText.toLowerCase().includes(query);
        }
      }
    </script>
    """
    return _layout("Search fixture", body)


def _form_page() -> bytes:
    body = """
    <h1>Harmless form fixture</h1>
    <form action="/submitted" method="get">
      <label for="smoke-text">Smoke text</label>
      <input id="smoke-text" name="smoke-text" type="text">
      <label for="smoke-choice">Smoke choice</label>
      <select id="smoke-choice" name="smoke-choice">
        <option value="red">Red</option>
        <option value="green">Green</option>
        <option value="blue">Blue</option>
      </select>
      <button id="submit-form" type="submit">Submit harmless form</button>
    </form>
    """
    return _layout("Harmless form fixture", body)


def _submitted_page(query: dict[str, list[str]]) -> bytes:
    text = escape(query.get("smoke-text", [""])[0])
    choice = escape(query.get("smoke-choice", [""])[0])
    body = (
        "<h1>Form submitted</h1>"
        f"<p id='submitted'>Received text {text}; choice {choice}.</p>"
    )
    return _layout("Submitted fixture", body)


def _stale_page() -> bytes:
    body = """
    <h1>Stale state fixture</h1>
    <button id="replace-state" type="button" onclick="replaceState()">Replace stale state</button>
    <button id="stale-target" type="button">Stale target</button>
    <p id="state-status">Initial state</p>
    <script>
      function replaceState() {
        document.getElementById('state-status').textContent = 'Replaced state';
        document.getElementById('stale-target').replaceWith(
          Object.assign(document.createElement('button'), {
            id: 'fresh-target',
            type: 'button',
            textContent: 'Fresh target'
          })
        );
      }
    </script>
    """
    return _layout("Stale state fixture", body)


def _low_confidence_page() -> bytes:
    body = """
    <h1>Low confidence fixture</h1>
    <p>These controls are deliberately similar.</p>
    <button type="button">Continue with local fixture</button>
    <button type="button">Continue with local fixture</button>
    <button type="button">Continue with local fixture</button>
    """
    return _layout("Low confidence fixture", body)


def _page(path: str, query: dict[str, list[str]]) -> tuple[int, bytes]:
    if path == "/":
        return 200, _ROOT.read_bytes()
    if path == "/search":
        return 200, _search_page()
    if path == "/form":
        return 200, _form_page()
    if path == "/submitted":
        return 200, _submitted_page(query)
    if path == "/stale":
        return 200, _stale_page()
    if path == "/low-confidence":
        return 200, _low_confidence_page()
    return 404, _layout("Not found", "<h1>Not found</h1>")


class _FixtureHandler(BaseHTTPRequestHandler):
    server_version = "JevBrowserFixture/1"

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        status, body = _page(parsed.path, parse_qs(parsed.query))
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


@dataclass(slots=True)
class FixtureServer(AbstractContextManager["FixtureServer"]):
    """Own one ephemeral localhost fixture server and its thread."""

    host: str = "127.0.0.1"
    _server: ThreadingHTTPServer | None = None
    _thread: Thread | None = None

    def start(self) -> FixtureServer:
        if self._server is not None:
            return self
        self._server = ThreadingHTTPServer((self.host, 0), _FixtureHandler)
        self._thread = Thread(
            target=self._server.serve_forever,
            name="jev-browser-fixture",
            daemon=True,
        )
        self._thread.start()
        return self

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("fixture server is not running")
        return int(self._server.server_address[1])

    @property
    def origin(self) -> str:
        return f"http://{self.host}:{self.port}"

    def url(self, path: str = "/") -> str:
        if not path.startswith("/"):
            raise ValueError("fixture path must start with '/'")
        return f"{self.origin}{path}"

    def close(self) -> None:
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=2)

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ["FixtureServer"]
