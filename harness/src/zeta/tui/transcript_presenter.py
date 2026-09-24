"""Compatibility alias for the transcript presenter module."""

import sys

from .transcript import transcript_presenter as _module

sys.modules[__name__] = _module
