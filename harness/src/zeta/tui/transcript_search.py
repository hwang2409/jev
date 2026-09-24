"""Compatibility alias for the transcript search module."""

import sys

from .transcript import transcript_search as _module

sys.modules[__name__] = _module
