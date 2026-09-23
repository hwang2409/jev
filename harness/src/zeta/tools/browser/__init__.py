"""Browser foundation and page-state helpers."""

from .gates import (
    PAGE_STATE_RECOVERY_ATTEMPT_CAP,
    PageStateDecision,
    conservative_provider_error_decision,
    evaluate_page_state,
    evaluate_page_state_with_provider,
)

__all__ = [
    "PAGE_STATE_RECOVERY_ATTEMPT_CAP",
    "PageStateDecision",
    "conservative_provider_error_decision",
    "evaluate_page_state",
    "evaluate_page_state_with_provider",
]
