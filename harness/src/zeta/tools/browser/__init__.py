"""Browser foundation and page-state helpers."""

from .gates import (
    PageStateDecision,
    conservative_provider_error_decision,
    evaluate_page_state,
    evaluate_page_state_with_provider,
)

__all__ = [
    "PageStateDecision",
    "conservative_provider_error_decision",
    "evaluate_page_state",
    "evaluate_page_state_with_provider",
]
