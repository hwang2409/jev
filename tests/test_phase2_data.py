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


def test_curve_evalset_has_schema_count_and_subset_coverage():
    cases = load_evalset("evalset_curve.jsonl")
    assert len(cases) == 40
    assert len({case["id"] for case in cases}) == 40
    counts = Counter(case["expected_tool"] for case in cases)
    assert set(counts) <= set(SUBSETS[15])
    assert all(counts[tool] >= 2 for tool in SUBSETS[15])
    for case in cases:
        assert set(case) == SCHEMA
        assert case["expected_needs_tool"] is True
        assert case["vague"] is False
        assert isinstance(case["history"], list)


def test_full_evalset_has_exact_coverage_and_hard_cases():
    cases = load_evalset("evalset_full.jsonl")
    assert len(cases) == 140
    assert len({case["id"] for case in cases}) == 140
    for case in cases:
        assert set(case) == SCHEMA
        assert case["expected_tool"] in CATALOG_120
        assert case["expected_needs_tool"] is True
        assert case["vague"] is False
        assert isinstance(case["history"], list)

    coverage = [case for case in cases if case["id"].startswith("cov-")]
    hard = [case for case in cases if case["id"].startswith("hard-")]
    assert len(coverage) == 120
    assert Counter(case["expected_tool"] for case in coverage) == Counter(
        {tool: 1 for tool in CATALOG_120}
    )
    assert {case["id"] for case in coverage} == {
        f"cov-{tool}" for tool in CATALOG_120
    }
    assert len(hard) == 20
    assert {case["id"] for case in hard} == {f"hard-{i}" for i in range(1, 21)}


def test_hard_steps_do_not_copy_expected_description_phrases():
    cases = load_evalset("evalset_full.jsonl")
    hard = [case for case in cases if case["id"].startswith("hard-")]
    for case in hard:
        description_words = re.findall(
            r"[a-z0-9]+", CATALOG_120[case["expected_tool"]].lower()
        )
        step_words = re.findall(r"[a-z0-9]+", case["step"].lower())
        phrases = {
            " ".join(description_words[index:index + 4])
            for index in range(len(description_words) - 3)
        }
        assert not any(
            " ".join(step_words[index:index + 4]) in phrases
            for index in range(len(step_words) - 3)
        ), case["id"]
