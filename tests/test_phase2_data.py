import json
import re
from collections import Counter
from pathlib import Path

from catalogs import CATALOG_120, SUBSETS


DOMAINS = {
    "files",
    "shell",
    "web",
    "calendar",
    "email",
    "crm",
    "deploy",
    "data",
    "chat",
    "payments",
}
SCHEMA = {
    "id",
    "task",
    "step",
    "history",
    "expected_tool",
    "expected_needs_tool",
    "vague",
}


def load_evalset(name):
    lines = Path(name).read_text().strip().splitlines()
    return [json.loads(line) for line in lines]


def test_catalog_has_120_tools_in_ten_domains():
    assert len(CATALOG_120) == 120
    assert len(set(CATALOG_120)) == 120
    counts = Counter(name.split("_", 1)[0] for name in CATALOG_120)
    assert set(counts) == DOMAINS
    assert set(counts.values()) == {12}


def test_catalog_names_and_descriptions_are_concrete():
    name_pattern = re.compile(r"^[a-z]+_[a-z]+_[a-z]+$")
    for name, description in CATALOG_120.items():
        assert name_pattern.fullmatch(name)
        assert name.strip() == name
        assert 10 <= len(description) <= 120, name
        assert "  " not in description


def test_subsets_are_strictly_nested_and_complete():
    assert list(SUBSETS) == [15, 30, 60, 120]
    assert len(SUBSETS[15]) == 15
    assert len(SUBSETS[30]) == 30
    assert len(SUBSETS[60]) == 60
    assert SUBSETS[120] == CATALOG_120
    assert set(SUBSETS[15]) < set(SUBSETS[30])
    assert set(SUBSETS[30]) < set(SUBSETS[60])
    assert set(SUBSETS[60]) < set(SUBSETS[120])
    assert {name.split("_", 1)[0] for name in SUBSETS[15]} == DOMAINS


def test_subset_names_are_valid_catalog_names():
    for subset in SUBSETS.values():
        assert set(subset) <= set(CATALOG_120)
        assert all(subset[name] == CATALOG_120[name] for name in subset)
