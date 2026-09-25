"""Shared confidence settings for provider-backed routing."""

import hashlib
import json

ROUTE_TOPK_CONFIDENCE = 0.8
"""Choice cutoff for exposing one routed tool."""
# TODO: calibrate this choice threshold with route confidence data.

BROWSER_ELEMENT_TOP1_CONFIDENCE = 0.8
"""Confidence required to select one browser element."""
# TODO: calibrate this choice threshold with browser element data.

BROWSER_ELEMENT_TOPN = 3
"""Number of element candidates exposed below the choice cutoff."""

SEARCH_RESULT_RELEVANCE_THRESHOLD = 0.7
"""Minimum score for accepting one search result."""
# TODO: calibrate this score threshold with search acceptance data.

SEARCH_RESULT_TIE_MARGIN = 0.1
"""Maximum score gap treated as a search-result tie."""
# TODO: calibrate this tie margin with search ambiguity data.

SEARCH_RESULT_RELEVANCE_FLOOR = 0.4
"""Minimum score for a search result to be relevant evidence."""
# TODO: calibrate this score floor with search relevance data.

SEARCH_RESULT_CALL_CONFIDENCE_THRESHOLD = 0.8
"""Minimum Jev confidence for automatic search-result ranking."""
# TODO: calibrate this call-confidence threshold with search ranking data.

BROWSER_THRESHOLD_VERSION = "browser-thresholds-v1"
"""Telemetry version for the independent browser routing thresholds."""


def browser_threshold_version(
    *,
    element_top1_confidence: float,
    element_topn: int,
    search_relevance_threshold: float,
    search_tie_margin: float,
    search_relevance_floor: float,
    search_call_confidence_threshold: float,
) -> str:
    """Return a stable telemetry version for resolved browser thresholds."""

    settings = {
        "element_top1_confidence": element_top1_confidence,
        "element_topn": element_topn,
        "search_relevance_threshold": search_relevance_threshold,
        "search_tie_margin": search_tie_margin,
        "search_relevance_floor": search_relevance_floor,
        "search_call_confidence_threshold": search_call_confidence_threshold,
    }
    defaults = {
        "element_top1_confidence": BROWSER_ELEMENT_TOP1_CONFIDENCE,
        "element_topn": BROWSER_ELEMENT_TOPN,
        "search_relevance_threshold": SEARCH_RESULT_RELEVANCE_THRESHOLD,
        "search_tie_margin": SEARCH_RESULT_TIE_MARGIN,
        "search_relevance_floor": SEARCH_RESULT_RELEVANCE_FLOOR,
        "search_call_confidence_threshold": SEARCH_RESULT_CALL_CONFIDENCE_THRESHOLD,
    }
    if settings == defaults:
        return BROWSER_THRESHOLD_VERSION
    serialized = json.dumps(settings, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
    return f"browser-thresholds-{digest}"
