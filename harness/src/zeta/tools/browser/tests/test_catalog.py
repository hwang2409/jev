from __future__ import annotations

from zeta.tools.browser.adapter import ElementRef, PageObservation, SnapshotLimits
from zeta.tools.browser.catalog import SnapshotCatalogBuilder, _serialized_size


def test_builder_normalizes_text_caps_catalog_and_stale_ids() -> None:
    first_element = ElementRef(
        1,
        "e1",
        "button",
        "click",
        "  Continue\n now  ",
        "Continue",
        None,
        "main",
        False,
        True,
    )
    hidden = ElementRef(
        1,
        "e2",
        "link",
        "click",
        "hidden",
        "hidden",
        None,
        "nav",
        False,
        False,
    )
    first = PageObservation(
        1,
        3,
        "https://example.test",
        "Title",
        "  page\n text  ",
        (first_element, hidden),
        True,
        True,
    )
    builder = SnapshotCatalogBuilder(
        SnapshotLimits(page_text_bytes=5, element_text_bytes=8, catalog_bytes=500)
    )

    catalog = builder.build(first)

    assert catalog.entries[0].text == "Continue"
    assert catalog.entries[0].affordance == "click"
    assert catalog.summary == "page"
    assert catalog.snapshot_id == 1
    assert builder.is_current(1, "e1") is True
    second = PageObservation(2, 4, first.url, "Title", "new", (), True, True)
    next_catalog = builder.build(second)
    assert next_catalog.invalidated_element_ids == frozenset({"e1", "e2"})
    assert builder.is_current(1, "e1") is False


def test_builder_prefers_accessible_name_and_bounds_large_inputs() -> None:
    element = ElementRef(
        5,
        "e1",
        "button",
        "click",
        "visible text",
        "  accessible\n name  ",
        "value " * 100,
        "main " * 100,
        False,
        True,
    )
    observation = PageObservation(
        5,
        8,
        "https://example.test",
        "  page title  ",
        "x" * 2_000,
        (element,),
        True,
        True,
    )
    builder = SnapshotCatalogBuilder(
        SnapshotLimits(page_text_bytes=37, element_text_bytes=12, catalog_bytes=500)
    )

    catalog = builder.build(observation)

    assert catalog.summary == "x" * 37
    assert catalog.entries[0].text == "accessible n"
    assert len(catalog.entries[0].value_hint or "") <= 12
    assert len(catalog.entries[0].landmark or "") <= 12
    assert len(catalog.entries) <= 1


def test_builder_invalidates_all_ids_when_generation_changes() -> None:
    first = PageObservation(
        10,
        1,
        "https://example.test",
        "One",
        "",
        (ElementRef(10, "e1", "button", "click", "one", "one", None, None, False, True),),
        True,
        True,
    )
    second = PageObservation(
        11,
        2,
        "https://example.test",
        "Two",
        "",
        (ElementRef(11, "e1", "button", "click", "two", "two", None, None, False, True),),
        True,
        True,
    )
    builder = SnapshotCatalogBuilder(SnapshotLimits())

    builder.build(first)
    catalog = builder.build(second)

    assert catalog.invalidated_element_ids == frozenset({"e1"})
    assert builder.is_current(10, "e1") is False
    assert builder.is_current(11, "e1") is True


def test_builder_assigns_monotonic_ids_and_bumps_generation_for_url_changes() -> None:
    first = PageObservation(
        10,
        7,
        "https://example.test/one",
        "One",
        "",
        (ElementRef(10, "e1", "button", "click", "one", "one", None, None, False, True),),
        True,
        True,
    )
    second = PageObservation(
        9,
        7,
        "https://example.test/two",
        "Two",
        "",
        (),
        True,
        True,
    )
    builder = SnapshotCatalogBuilder(SnapshotLimits())

    first_catalog = builder.build(first)
    second_catalog = builder.build(second)

    assert second_catalog.snapshot_id > first_catalog.snapshot_id
    assert second_catalog.generation == first_catalog.generation + 1
    assert second_catalog.invalidated_element_ids == frozenset({"e1"})


def test_builder_catalog_byte_cap_includes_invalidated_ids() -> None:
    first = PageObservation(
        1,
        1,
        "https://example.test/one",
        "One",
        "",
        tuple(
            ElementRef(1, f"e{index}", "button", "click", "item", "item", None, None, False, True)
            for index in range(6)
        ),
        True,
        True,
    )
    builder = SnapshotCatalogBuilder(
        SnapshotLimits(catalog_bytes=390)
    )
    builder.build(first)
    second = builder.build(
        PageObservation(2, 1, "https://example.test/two", "Two", "", (), True, True)
    )

    assert _serialized_size(
        second.snapshot_id,
        second.generation,
        second.url,
        second.title,
        second.summary,
        second.entries,
        second.invalidated_element_ids,
    ) <= 390


def test_builder_fits_two_thousand_elements_within_time_and_bounds() -> None:
    import time

    observation = PageObservation(
        1,
        1,
        "https://example.test",
        "Results",
        "",
        tuple(
            ElementRef(1, f"e{index}", "button", "click", f"item {index}", f"item {index}", None, None, False, True)
            for index in range(2_000)
        ),
        True,
        True,
    )
    builder = SnapshotCatalogBuilder(SnapshotLimits())

    started = time.perf_counter()
    catalog = builder.build(observation)
    elapsed = time.perf_counter() - started

    assert elapsed < 2
    assert len(catalog.entries) <= 2_000
    assert _serialized_size(
        catalog.snapshot_id,
        catalog.generation,
        catalog.url,
        catalog.title,
        catalog.summary,
        catalog.entries,
        catalog.invalidated_element_ids,
    ) <= builder.limits.catalog_bytes
