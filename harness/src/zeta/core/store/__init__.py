"""Public session persistence API."""

import os  # noqa: F401 - preserve patch targets at the legacy import path
import uuid  # noqa: F401 - preserve patch targets at the legacy import path

from ._pending import (
    MAX_PENDING_PROMPT_TEXT,
    PendingPromptCommitTimeoutError,
    PendingPromptQueue,
    PendingPromptsClosedError,
)
from ._store import (
    MAX_AGENT_NOTIFICATION_TEXT,
    SCHEMA,
    ConversationEntry,
    ConversationIntegrityError,
    ConversationStore,
)

__all__ = [
    "MAX_AGENT_NOTIFICATION_TEXT",
    "MAX_PENDING_PROMPT_TEXT",
    "SCHEMA",
    "ConversationEntry",
    "ConversationIntegrityError",
    "ConversationStore",
    "PendingPromptCommitTimeoutError",
    "PendingPromptQueue",
    "PendingPromptsClosedError",
]
