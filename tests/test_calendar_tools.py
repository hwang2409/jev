from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

import zeta.tools.calendar as calendar_module
from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.store import ConversationStore
from zeta.skills import SkillCatalog
from zeta.tools.registry import ToolRegistry
from zeta.types import ToolCall


class FakeCalendarAdapter:
    def __init__(
        self,
        events: list[calendar_module.CalendarEvent] | None = None,
        *,
        status: str = calendar_module.AUTHORIZED,
        request_result: bool = True,
    ) -> None:
        self.events = events or []
        self.status = status
        self.request_result = request_result
        self.request_count = 0
        self.fetched: tuple[datetime, datetime] | None = None
        self.saved: dict[str, object] | None = None

    def authorization_status(self) -> str:
        return self.status

    def request_access(self) -> bool:
        self.request_count += 1
        if self.request_result:
            self.status = calendar_module.AUTHORIZED
        return self.request_result

    def fetch_events(self, start: datetime, end: datetime) -> list[calendar_module.CalendarEvent]:
        self.fetched = (start, end)
        return self.events

    def save_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        calendar: str | None,
        notes: str | None,
    ) -> calendar_module.CalendarEvent:
        self.saved = {
            "title": title,
            "start": start,
            "end": end,
            "calendar": calendar,
            "notes": notes,
        }
        return calendar_module.CalendarEvent(
            title=title,
            start=start,
            end=end,
            calendar_name=calendar or "Default",
            location=None,
            all_day=False,
        )


def event(
    title: str,
    start: str,
    end: str,
    calendar: str,
    *,
    location: str | None = None,
    all_day: bool = False,
) -> calendar_module.CalendarEvent:
    return calendar_module.CalendarEvent(
        title=title,
        start=calendar_module.parse_iso8601(start),
        end=calendar_module.parse_iso8601(end),
        calendar_name=calendar,
        location=location,
        all_day=all_day,
    )


def structured(result: dict[str, object]) -> dict[str, object]:
    value = result["structuredContent"]
    assert isinstance(value, dict)
    return value


@pytest.mark.asyncio
async def test_events_lists_all_day_cross_midnight_and_named_calendar() -> None:
    adapter = FakeCalendarAdapter(
        [
            event(
                "all day",
                "2026-01-02T00:00:00",
                "2026-01-03T00:00:00",
                "work",
                all_day=True,
            ),
            event(
                "overnight",
                "2026-01-02T23:00:00",
                "2026-01-03T01:00:00",
                "personal",
                location="home",
            ),
        ]
    )

    result = await calendar_module._calendar_events(
        None,
        {
            "start": "2026-01-02T00:00:00",
            "end": "2026-01-04T00:00:00",
            "calendar": "personal",
        },
        adapter=adapter,
    )

    assert result["isError"] is False
    events = structured(result)["events"]
    assert events == [
        {
            "title": "overnight",
            "start": "2026-01-02T23:00:00",
            "end": "2026-01-03T01:00:00",
            "calendar": "personal",
            "location": "home",
            "all_day": False,
        }
    ]

    adapter = FakeCalendarAdapter(
        [
            event(
                "all day",
                "2026-01-02T00:00:00",
                "2026-01-03T00:00:00",
                "work",
                all_day=True,
            ),
        ]
    )
    result = await calendar_module._calendar_events(
        None,
        {
            "start": "2026-01-02T00:00:00",
            "end": "2026-01-04T00:00:00",
            "calendar": None,
        },
        adapter=adapter,
    )
    assert structured(result)["events"][0]["all_day"] is True


@pytest.mark.asyncio
async def test_events_cap_window_and_parse_iso_errors_without_adapter_access() -> None:
    adapter = FakeCalendarAdapter()

    too_wide = await calendar_module._calendar_events(
        None,
        {
            "start": "2026-01-01T00:00:00Z",
            "end": "2026-04-04T00:00:01Z",
            "calendar": None,
        },
        adapter=adapter,
    )
    invalid = await calendar_module._calendar_events(
        None,
        {"start": "not-a-date", "end": "2026-01-02T00:00:00", "calendar": None},
        adapter=adapter,
    )

    assert too_wide["isError"] is True
    assert "92 days" in str(too_wide["content"])
    assert invalid["isError"] is True
    assert "ISO-8601" in str(invalid["content"])
    assert adapter.fetched is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "request_result", "message"),
    [
        (calendar_module.DENIED, True, "denied"),
        (calendar_module.UNDETERMINED, False, "denied"),
        (calendar_module.RESTRICTED, True, "restricted"),
    ],
)
async def test_events_reports_authorization_failures(
    status: str, request_result: bool, message: str
) -> None:
    adapter = FakeCalendarAdapter(
        status=status,
        request_result=request_result,
    )

    result = await calendar_module._calendar_events(
        None,
        {
            "start": "2026-01-01T00:00:00",
            "end": "2026-01-02T00:00:00",
            "calendar": None,
        },
        adapter=adapter,
    )

    assert result["isError"] is True
    assert message in str(result["content"]).lower()
    assert adapter.fetched is None
    assert adapter.request_count == (1 if status == calendar_module.UNDETERMINED else 0)


@pytest.mark.asyncio
async def test_create_passes_notes_and_calendar_to_adapter() -> None:
    adapter = FakeCalendarAdapter()
    result = await calendar_module._calendar_create(
        None,
        {
            "title": "Planning",
            "start": "2026-01-02T23:00:00",
            "end": "2026-01-03T01:00:00",
            "calendar": "work",
            "notes": "bring the draft",
        },
        adapter=adapter,
    )

    assert result["isError"] is False
    assert adapter.saved == {
        "title": "Planning",
        "start": calendar_module.parse_iso8601("2026-01-02T23:00:00"),
        "end": calendar_module.parse_iso8601("2026-01-03T01:00:00"),
        "calendar": "work",
        "notes": "bring the draft",
    }


def test_create_is_approval_required_and_catalog_has_neutral_boundaries(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
    )
    calendar_module.register(registry)

    create = registry.definitions_by_name["calendar_create"]
    assert create.requires_approval is True
    assert create.approval_subject == "calendar"

    catalog = calendar_module.catalog_criteria()
    assert set(catalog) == {"calendar_events", "calendar_create"}
    assert all(set(entry) == {"what", "not_for", "examples"} for entry in catalog.values())
    assert "todo" in catalog["calendar_events"]["not_for"]
    assert "automation" in catalog["calendar_create"]["not_for"]
    assert "memory" in catalog["calendar_events"]["not_for"]
    assert all("calendar" in example.lower() for entry in catalog.values() for example in entry["examples"])


@pytest.mark.asyncio
async def test_create_uses_registry_approval_gate(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.DENY)
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
        approval_policy=policy,
        approval_store=store,
    )
    calendar_module.register(registry)
    adapter = FakeCalendarAdapter()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(calendar_module, "_event_store_adapter", lambda: adapter)
    try:
        result = await registry.execute(
            ToolCall(
                "create-1",
                "calendar_create",
                {
                    "title": "Planning",
                    "start": "2026-01-02T23:00:00",
                    "end": "2026-01-03T01:00:00",
                    "calendar": "work",
                    "notes": None,
                },
            )
        )
    finally:
        monkeypatch.undo()

    assert result["isError"] is True
    assert "denied" in str(result["content"]).lower()
    assert adapter.saved is None
