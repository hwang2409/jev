import json
from collections import Counter
from pathlib import Path

from catalog import CATALOG


def load():
    lines = (
        Path(__file__).parents[1] / "evalset.jsonl"
    ).read_text().strip().splitlines()
    return [json.loads(line) for line in lines]


def test_size_and_schema():
    cases = load()
    assert len(cases) == 60
    ids = [c["id"] for c in cases]
    assert len(set(ids)) == 60
    for c in cases:
        assert set(c) == {"id", "task", "step", "history",
                          "expected_tool", "expected_needs_tool", "vague"}
        assert isinstance(c["history"], list)
        if c["expected_tool"] is not None:
            assert c["expected_tool"] in CATALOG


def test_composition():
    cases = load()
    clear = [c for c in cases if not c["vague"] and c["expected_needs_tool"]]
    no_tool = [c for c in cases if not c["expected_needs_tool"]]
    vague = [c for c in cases if c["vague"]]
    assert len(clear) == 46 and len(no_tool) == 8 and len(vague) == 6
    coverage = Counter(c["expected_tool"] for c in clear)
    for tool in CATALOG:
        assert coverage[tool] >= 2, f"{tool} covered by <2 cases"
    for c in no_tool + vague:
        assert c["expected_tool"] is None
