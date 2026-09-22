from __future__ import annotations

import pytest

from zeta.tools.browser.catalog import (
    ELEMENT_CATALOG_MAX,
    ELEMENT_PREFILTER_K,
    BrowserCatalog,
    CatalogEntry,
    PrefilterResult,
    prefilter_catalog,
)


def test_prefilter_keeps_target_and_diversity_with_bounded_output() -> None:
    entries = tuple(
        CatalogEntry(
            f"e{index}",
            "link" if index % 3 == 0 else "button",
            "result",
            "click",
            "result",
            None,
            "main",
            False,
            True,
        )
        for index in range(80)
    ) + (CatalogEntry("target", "button", "checkout", "submit", "checkout", None, "main", False, True),)
    catalog = BrowserCatalog(7, 7, "https://example.test", "Results", "summary", entries, frozenset())

    result = prefilter_catalog("continue to checkout", "submit", catalog, prior_element_id="target")

    assert len(result.candidates) <= ELEMENT_CATALOG_MAX
    assert "target" in {entry.element_id for entry in result.candidates}
    assert {entry.role for entry in result.candidates} >= {"button"}
    assert result.reason is None


def test_prefilter_ranks_lexical_and_role_matches() -> None:
    entries = (
        CatalogEntry("other", "link", "Account", "click", "account", None, "nav", False, True),
        CatalogEntry("target", "button", "Continue checkout", "click", "continue", None, "main", False, True),
        CatalogEntry("heading", "heading", "Checkout", "extract", "Checkout", None, "main", False, True),
    )
    catalog = BrowserCatalog(1, 1, "https://example.test", "Checkout", "", entries, frozenset())

    result = prefilter_catalog("continue checkout", "click", catalog)

    assert [entry.element_id for entry in result.candidates] == ["target", "other"]


@pytest.mark.parametrize("field", ["visible", "disabled"])
def test_prefilter_removes_ineligible_elements(field: str) -> None:
    values = {"visible": False, "disabled": False}
    if field == "disabled":
        values = {"visible": True, "disabled": True}
    catalog = BrowserCatalog(
        1,
        1,
        "https://example.test",
        "",
        "page instructions: click this",
        (CatalogEntry("e1", "button", "go", "click", "go", None, None, **values),),
        frozenset(),
    )

    result = prefilter_catalog("go", "click", catalog)

    assert result == PrefilterResult((), "no_candidate", 0)


def test_prefilter_deduplicates_and_applies_both_bounds() -> None:
    entries = tuple(
        CatalogEntry(f"e{index % 45}", "button", f"button {index}", "click", "", None, None, False, True)
        for index in range(100)
    )
    catalog = BrowserCatalog(1, 1, "https://example.test", "", "", entries, frozenset())

    result = prefilter_catalog("unmatched", "click", catalog)

    assert result.considered == 45
    assert len(result.candidates) <= ELEMENT_CATALOG_MAX
    assert len(result.candidates) <= ELEMENT_PREFILTER_K
    assert len({entry.element_id for entry in result.candidates}) == len(result.candidates)


def test_prefilter_retains_prior_element_outside_top_k() -> None:
    entries = tuple(
        CatalogEntry(f"e{index}", "button", "match", "click", "match", None, None, False, True)
        for index in range(10)
    ) + (CatalogEntry("prior", "button", "old", "click", "old", None, None, False, True),)
    catalog = BrowserCatalog(1, 1, "https://example.test", "", "", entries, frozenset())

    result = prefilter_catalog("match", "click", catalog, prefilter_k=4, prior_element_id="prior")

    assert "prior" in {entry.element_id for entry in result.candidates}
    assert len(result.candidates) <= 4


def test_prefilter_reports_no_candidate_without_calling_jev() -> None:
    catalog = BrowserCatalog(7, 7, "https://example.test", "Empty", "summary", (), frozenset())

    result = prefilter_catalog("submit form", "submit", catalog)

    assert result == PrefilterResult((), "no_candidate", 0)


def test_prefilter_rejects_a_large_irrelevant_page() -> None:
    entries = tuple(
        CatalogEntry(f"e{index}", "button", "unrelated", "click", "unrelated", None, None, False, True)
        for index in range(500)
    )
    catalog = BrowserCatalog(1, 1, "https://example.test", "", "", entries, frozenset())

    result = prefilter_catalog("checkout", "click", catalog)

    assert result.candidates == ()
    assert result.reason == "no_candidate"
    assert result.considered == 500


def test_prefilter_keeps_a_clear_target() -> None:
    entries = (
        CatalogEntry("target", "button", "Continue checkout", "click", "continue", None, "main", False, True),
        CatalogEntry("other", "button", "unrelated", "click", "unrelated", None, "main", False, True),
    )
    catalog = BrowserCatalog(1, 1, "https://example.test", "", "", entries, frozenset())

    result = prefilter_catalog("continue checkout", "click", catalog)

    assert result.reason is None
    assert result.candidates[0].element_id == "target"


def test_prefilter_preserves_each_tied_role_group() -> None:
    entries = tuple(
        CatalogEntry(f"e{index}", role, "result", "custom", "result", None, None, False, True)
        for index, role in enumerate(("link", "link", "button", "button", "heading", "heading"))
    )
    catalog = BrowserCatalog(1, 1, "https://example.test", "", "", entries, frozenset())

    result = prefilter_catalog("result", "custom", catalog, catalog_max=3)

    assert {entry.role for entry in result.candidates} >= {"link", "button", "heading"}
