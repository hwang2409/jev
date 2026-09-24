"""Transcript widgets, presentation, and search helpers for the TUI."""

from . import transcript as _transcript
from . import transcript_presenter as _transcript_presenter
from . import transcript_search as _transcript_search

# Keep the package surface equivalent to the former flat modules. This also
# preserves private names used by the presenter without duplicating them.
for _module in (_transcript, _transcript_search, _transcript_presenter):
    globals().update(
        {
            name: value
            for name, value in vars(_module).items()
            if not name.startswith("__")
        }
    )

transcript = _transcript
transcript_presenter = _transcript_presenter
transcript_search = _transcript_search

del _module, _transcript, _transcript_presenter, _transcript_search
