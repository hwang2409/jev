from catalog import CATALOG


def test_catalog_has_15_unique_tools():
    assert len(CATALOG) == 15
    assert len(set(CATALOG)) == 15


def test_descriptions_are_short_and_nonempty():
    for name, desc in CATALOG.items():
        assert name.strip() == name and name
        assert 10 <= len(desc) <= 120, f"{name}: bad description length"
