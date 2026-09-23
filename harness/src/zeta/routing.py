"""Shared confidence settings for provider-backed routing."""

# threshold evidence compares derived confidence with the true Jev baseline
# in router/results/20260917-202530.json; 30 cases, none near the 0.8 gate.

ROUTE_TOPK_CONFIDENCE = 0.8
"""Choice cutoff shared by tool and browser-element routing."""

BROWSER_ELEMENT_TOP1_CONFIDENCE = ROUTE_TOPK_CONFIDENCE
"""Confidence required to select one browser element."""

BROWSER_ELEMENT_TOPN = 3
"""Number of element candidates exposed below the choice cutoff."""

SEARCH_RESULT_RELEVANCE_THRESHOLD = 0.7
"""Minimum score for accepting one search result."""

SEARCH_RESULT_TIE_MARGIN = 0.1
"""Maximum score gap treated as a search-result tie."""

SEARCH_RESULT_RELEVANCE_FLOOR = 0.4
"""Minimum score for a search result to be relevant evidence."""

SEARCH_RESULT_CALL_CONFIDENCE_THRESHOLD = 0.8
"""Minimum Jev confidence for automatic search-result ranking."""
