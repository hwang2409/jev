from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
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


class FakeNSDate:
    @staticmethod
    def dateWithTimeIntervalSince1970_(timestamp: float) -> FakeNSDate:
        return FakeNSDate(timestamp)

    def __init__(self, timestamp: float) -> None:
        self.timestamp = timestamp

    def timeIntervalSince1970(self) -> float:
        return self.timestamp


class FakeEventKitEvent:
    def __init__(
        self,
        *,
        title: str,
        start: datetime,
        end: datetime,
        calendar: str,
        notes: str | None,
        timezone_value: object | None,
        all_day: bool = False,
    ) -> None:
        self._title = title
        self._start = FakeNSDate(start.timestamp())
        self._end = FakeNSDate(end.timestamp())
        self._calendar = calendar
        self._notes = notes
        self._timezone = timezone_value
        self._all_day = all_day

    def title(self) -> str:
        return self._title

    def startDate(self) -> FakeNSDate:
        return self._start

    def endDate(self) -> FakeNSDate:
        return self._end

    def calendar(self) -> object:
        return type("Calendar", (), {"title": lambda _self: self._calendar})()

    def location(self) -> None:
        return None

    def notes(self) -> str | None:
        return self._notes

    def timeZone(self) -> object | None:
        return self._timezone

    def isAllDay(self) -> bool:
        return self._all_day


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
            "notes": None,
            "floating": False,
            "time_zone": None,
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
@pytest.mark.parametrize(
    ("tool_name", "arguments", "adapter_field"),
    [
        (
            "calendar_events",
            {
                "start": "2026-01-01T00:00:00",
                "end": "2026-01-02T00:00:00",
                "calendar": None,
                "extra": True,
            },
            "fetched",
        ),
        (
            "calendar_events",
            {
                "start": "2026-01-01T00:00:00",
                "end": "2026-01-02T00:00:00",
            },
            "fetched",
        ),
        (
            "calendar_create",
            {
                "title": "Planning",
                "start": "2026-01-01T00:00:00",
                "end": "2026-01-02T00:00:00",
                "calendar": None,
                "notes": 42,
            },
            "saved",
        ),
    ],
)
async def test_registry_rejects_invalid_arguments_before_approval_or_adapter(
    tmp_path: Path,
    tool_name: str,
    arguments: dict[str, object],
    adapter_field: str,
) -> None:
    store = ConversationStore(tmp_path)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ASK)
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
        result = await registry.execute(ToolCall("invalid-1", tool_name, arguments))
    finally:
        monkeypatch.undo()

    assert result["isError"] is True
    assert structured(result)["error"]["kind"] == "invalid_arguments"
    assert getattr(adapter, adapter_field) is None
    assert policy.pending_requests() == []


def test_eventkit_conversion_preserves_notes_zones_and_floating_dates() -> None:
    zone = UTC
    zoned = FakeEventKitEvent(
        title="zoned",
        start=datetime(2026, 1, 2, 9, tzinfo=zone),
        end=datetime(2026, 1, 2, 10, tzinfo=zone),
        calendar="work",
        notes="agenda",
        timezone_value=zone,
    )
    floating = FakeEventKitEvent(
        title="floating",
        start=datetime(2026, 1, 2, 9),  # noqa: DTZ001 - fake floating EventKit data
        end=datetime(2026, 1, 2, 10),  # noqa: DTZ001 - fake floating EventKit data
        calendar="work",
        notes=None,
        timezone_value=None,
    )
    all_day = FakeEventKitEvent(
        title="all day",
        start=datetime(2026, 1, 2, tzinfo=zone),
        end=datetime(2026, 1, 3, tzinfo=zone),
        calendar="work",
        notes="holiday",
        timezone_value=zone,
        all_day=True,
    )

    zoned_event = calendar_module.EventStoreAdapter._event_from_ek(zoned)
    floating_event = calendar_module.EventStoreAdapter._event_from_ek(floating)
    all_day_event = calendar_module.EventStoreAdapter._event_from_ek(all_day)

    assert zoned_event.notes == "agenda"
    assert zoned_event.start.tzinfo is zone
    assert zoned_event.floating is False
    assert zoned_event.time_zone == "UTC"
    assert floating_event.start.tzinfo is None
    assert floating_event.floating is True
    assert all_day_event.all_day is True
    assert all_day_event.start.tzinfo is None
    assert calendar_module._event_dict(all_day_event)["start"] == "2026-01-02"
    assert calendar_module._event_dict(all_day_event)["end"] == "2026-01-03"


def test_eventkit_floating_conversion_preserves_wall_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_tz = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    try:
        floating = FakeEventKitEvent(
            title="floating",
            start=datetime(2026, 1, 2, 9),  # noqa: DTZ001 - fake floating EventKit data
            end=datetime(2026, 1, 2, 10),  # noqa: DTZ001 - fake floating EventKit data
            calendar="work",
            notes=None,
            timezone_value=None,
        )

        floating_event = calendar_module.EventStoreAdapter._event_from_ek(floating)

        assert floating_event.start == datetime(2026, 1, 2, 9)
        assert floating_event.end == datetime(2026, 1, 2, 10)
    finally:
        if original_tz is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", original_tz)
        time.tzset()


def test_eventkit_date_conversion_uses_process_timezone_for_naive_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = object.__new__(calendar_module.EventStoreAdapter)
    adapter._nsdate = FakeNSDate
    original_tz = os.environ.get("TZ")
    try:
        for zone, expected in (
            ("UTC", 1767261600.0),
            ("America/New_York", 1767279600.0),
        ):
            monkeypatch.setenv("TZ", zone)
            time.tzset()
            assert adapter._date(datetime(2026, 1, 1, 10)).timestamp == expected  # noqa: DTZ001 - test naive input

        monkeypatch.setenv("TZ", "America/New_York")
        time.tzset()
        assert adapter._date(datetime.fromisoformat("2026-01-01T10:00:00+02:00")).timestamp == 1767254400.0
    finally:
        if original_tz is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", original_tz)
        time.tzset()


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
async def test_events_normalize_naive_and_floating_datetimes_for_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_tz = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    try:
        zoned_adapter = FakeCalendarAdapter(
            [
                event(
                    "zoned",
                    "2026-01-02T14:00:00+00:00",
                    "2026-01-02T15:00:00+00:00",
                    "work",
                )
            ]
        )
        zoned_result = await calendar_module._calendar_events(
            None,
            {
                "start": "2026-01-02T08:00:00",
                "end": "2026-01-02T10:00:00",
                "calendar": None,
            },
            adapter=zoned_adapter,
        )

        floating_adapter = FakeCalendarAdapter(
            [event("floating", "2026-01-02T09:00:00", "2026-01-02T10:00:00", "work")]
        )
        floating_result = await calendar_module._calendar_events(
            None,
            {
                "start": "2026-01-02T14:00:00+00:00",
                "end": "2026-01-02T15:00:00+00:00",
                "calendar": None,
            },
            adapter=floating_adapter,
        )

        assert len(structured(zoned_result)["events"]) == 1
        assert len(structured(floating_result)["events"]) == 1
    finally:
        if original_tz is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", original_tz)
        time.tzset()


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
    assert catalog["calendar_events"]["not_for"] == (
        "Creating events (use calendar_create), Zeta task lists (use todo), "
        "scheduled agent actions (use automation), or memory and search tools."
    )
    assert catalog["calendar_create"]["not_for"] == (
        "Reading events (use calendar_events), Zeta task lists (use todo), "
        "scheduled agent actions (use automation), or memory and search tools."
    )
    assert all(len(str(entry["not_for"])) <= 150 for entry in catalog.values())
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
    assert result["isCanceled"] is True
    assert result["content"][0]["text"] == "tool execution canceled"
    assert structured(result)["error"]["kind"] == "canceled"
    assert adapter.saved is None


@pytest.mark.asyncio
async def test_fake_adapter_loads_seed_appends_events_and_writes_created_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = tmp_path / "calendar.json"
    seed.write_text(
        json.dumps(
            [
                {
                    "title": "Seeded event",
                    "start": "2026-09-19T09:00:00",
                    "end": "2026-09-19T10:00:00",
                    "calendar": "work",
                    "all_day": False,
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ZETA_CALENDAR_ADAPTER", f"fake:{seed}")

    listed = await calendar_module._calendar_events(
        None,
        {
            "start": "2026-09-19T00:00:00",
            "end": "2026-09-20T00:00:00",
            "calendar": None,
        },
    )
    assert [item["title"] for item in structured(listed)["events"]] == [
        "Seeded event"
    ]

    created = await calendar_module._calendar_create(
        None,
        {
            "title": "Created event",
            "start": "2026-09-19T11:00:00",
            "end": "2026-09-19T12:00:00",
            "calendar": "work",
            "notes": None,
        },
    )
    assert created["isError"] is False
    listed_again = await calendar_module._calendar_events(
        None,
        {
            "start": "2026-09-19T00:00:00",
            "end": "2026-09-20T00:00:00",
            "calendar": None,
        },
    )
    assert [item["title"] for item in structured(listed_again)["events"]] == [
        "Seeded event",
        "Created event",
    ]
    assert json.loads(Path(f"{seed}.out").read_text(encoding="utf-8")) == [
        {
            "all_day": False,
            "calendar": "work",
            "floating": False,
            "location": None,
            "notes": None,
            "start": "2026-09-19T11:00:00",
            "end": "2026-09-19T12:00:00",
            "time_zone": None,
            "title": "Created event",
        }
    ]


def test_fake_adapter_is_disabled_when_environment_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SentinelAdapter:
        pass

    monkeypatch.delenv("ZETA_CALENDAR_ADAPTER", raising=False)
    monkeypatch.setattr(calendar_module, "EventStoreAdapter", SentinelAdapter)
    assert isinstance(calendar_module._event_store_adapter(), SentinelAdapter)
