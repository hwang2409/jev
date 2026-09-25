from __future__ import annotations

import json
from pathlib import Path

import pytest

from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools.browser import register
from zeta.tools.browser.adapter import FakeBrowserAdapter
from zeta.tools.browser.tests.test_tools import _observation, _registry, _structured


@pytest.mark.asyncio
async def test_browser_registers_stable_schemas_and_starts_lazily(
    tmp_path: Path,
) -> None:
    adapter = FakeBrowserAdapter([_observation()])
    registry = _registry(tmp_path, adapter)

    assert {
        "browser_navigate",
        "browser_state",
        "browser_click",
        "browser_type",
        "browser_select",
        "browser_extract",
        "browser_submit",
    } <= registry.registered_names
    assert adapter.navigations == []
    result = await registry.execute(ToolCall("state", "browser_state", {}))

    assert result["isError"] is False
    assert _structured(result)["snapshot_id"] == 1
    assert adapter.navigations == []


def test_browser_schemas_match_the_complete_spec() -> None:
    registry = ToolRegistry(
        Path("."),
        register_builtin=False,
        browser_enabled=True,
        skill_catalog=SkillCatalog.empty(),
    )
    registry.browser_adapter_factory = lambda: FakeBrowserAdapter([_observation()])
    register(registry)

    expected = {
        "browser_navigate": '{"description":"Open an allowed URL in the session page.","name":"browser_navigate","parameters":{"additionalProperties":false,"properties":{"url":{"minLength":1,"type":"string"}},"required":["url"],"type":"object"}}',
        "browser_state": '{"description":"Return the current bounded page snapshot and element catalog.","name":"browser_state","parameters":{"additionalProperties":false,"properties":{},"type":"object"}}',
        "browser_click": '{"description":"Click one catalog element by stable snapshot id.","name":"browser_click","parameters":{"additionalProperties":false,"properties":{"affordance":{"minLength":1,"type":"string"},"element_id":{"minLength":1,"type":"string"},"role":{"minLength":1,"type":"string"},"snapshot_id":{"minimum":1,"type":"integer"}},"required":["snapshot_id","element_id","role","affordance"],"type":"object"}}',
        "browser_type": '{"description":"Replace or append text in one input by snapshot id.","name":"browser_type","parameters":{"additionalProperties":false,"properties":{"affordance":{"minLength":1,"type":"string"},"element_id":{"minLength":1,"type":"string"},"replace":{"type":"boolean"},"role":{"minLength":1,"type":"string"},"snapshot_id":{"minimum":1,"type":"integer"},"text":{"type":"string"}},"required":["snapshot_id","element_id","role","affordance","text","replace"],"type":"object"}}',
        "browser_select": '{"description":"Select one option in a select control by snapshot id and value.","name":"browser_select","parameters":{"additionalProperties":false,"properties":{"affordance":{"minLength":1,"type":"string"},"element_id":{"minLength":1,"type":"string"},"role":{"minLength":1,"type":"string"},"snapshot_id":{"minimum":1,"type":"integer"},"value":{"type":"string"}},"required":["snapshot_id","element_id","role","affordance","value"],"type":"object"}}',
        "browser_submit": '{"description":"Submit a form or click the identified submit control after safety approval.","name":"browser_submit","parameters":{"additionalProperties":false,"properties":{"affordance":{"minLength":1,"type":"string"},"element_id":{"minLength":1,"type":"string"},"role":{"minLength":1,"type":"string"},"snapshot_id":{"minimum":1,"type":"integer"}},"required":["snapshot_id","element_id","role","affordance"],"type":"object"}}',
        "browser_extract": '{"description":"Return bounded text or selected attributes from one element or the page.","name":"browser_extract","parameters":{"additionalProperties":false,"properties":{"attributes":{"items":{"type":"string"},"type":"array"},"element_id":{"minLength":1,"type":["string","null"]},"limit":{"minimum":1,"type":"integer"},"snapshot_id":{"minimum":1,"type":["integer","null"]}},"type":"object"}}',
    }

    actual = {
        schema["name"]: json.dumps(schema, sort_keys=True, separators=(",", ":"))
        for schema in registry.schemas
    }
    assert actual == expected
