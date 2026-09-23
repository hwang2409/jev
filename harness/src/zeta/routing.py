"""Shared confidence settings for provider-backed routing."""

ROUTE_TOPK_CONFIDENCE = 0.8
"""Choice cutoff shared by tool and browser-element routing."""

BROWSER_ELEMENT_TOP1_CONFIDENCE = ROUTE_TOPK_CONFIDENCE
"""Confidence required to select one browser element."""

BROWSER_ELEMENT_TOPN = 3
"""Number of element candidates exposed below the choice cutoff."""
