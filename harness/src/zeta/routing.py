"""Shared confidence settings for provider-backed routing."""

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
