import json
from pathlib import Path

from catalogs import CATALOG_120


SCENARIOS_PATH = Path(__file__).parents[1] / "scenarios.jsonl"


def load_scenarios():
    return [
        json.loads(line)
        for line in SCENARIOS_PATH.read_text().splitlines()
        if line.strip()
    ]


def test_phase3_has_ten_valid_chained_scenarios():
    scenarios = load_scenarios()

    assert len(scenarios) == 10
    assert len({scenario["id"] for scenario in scenarios}) == 10
    domains = set()
    cross_domain = 0
    for scenario in scenarios:
        expected = scenario["expected_tools"]
        assert 3 <= len(expected) <= 5
        assert len(expected) == len(set(expected))
        assert all(tool in CATALOG_120 for tool in expected)
        assert set(scenario["results"]) == set(expected)
        assert 1 <= len(scenario["answer_keys"]) <= 3
        assert all(
            any(key.lower() in result.lower() for result in scenario["results"].values())
            for key in scenario["answer_keys"]
        )
        assert not any(tool in scenario["task"] for tool in CATALOG_120)
        scenario_domains = {tool.split("_", 1)[0] for tool in expected}
        domains.update(scenario_domains)
        cross_domain += len(scenario_domains) > 1

    assert len(domains) >= 6
    assert cross_domain >= 3


def test_each_result_contains_a_follow_up_signal():
    for scenario in load_scenarios():
        values = list(scenario["results"].values())
        assert all(value.strip() for value in values)
        assert all(
            any(token.lower() in values[index + 1].lower() for token in value.split() if len(token) > 3)
            for index, value in enumerate(values[:-1])
        )
