"""Apple Calendar tools backed by EventKit."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from ..types import StructuredToolResult
from .registry import ToolRegistry, _error_result, _success_result, text_block

UNDETERMINED = "undetermined"
DENIED = "denied"
RESTRICTED = "restricted"
AUTHORIZED = "authorized"
MAX_WINDOW = timedelta(days=92)


class CalendarError(ValueError):
    """A user-facing Calendar tool error."""


@dataclass(frozen=True, slots=True)
class CalendarEvent:
    title: str
    start: datetime
    end: datetime
    calendar_name: str
    location: str | None
    all_day: bool


class CalendarAdapter(Protocol):
    def authorization_status(self) -> str: ...

    def request_access(self) -> bool: ...

    def fetch_events(self, start: datetime, end: datetime) -> list[CalendarEvent]: ...

    def save_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        calendar: str | None,
        notes: str | None,
    ) -> CalendarEvent: ...


def _load_eventkit() -> tuple[Any, Any]:
    try:
        import EventKit
        from Foundation import NSDate
    except ImportError as exc:
        raise CalendarError(
            "Apple Calendar tools require pyobjc-framework-EventKit on macOS"
        ) from exc
    return EventKit, NSDate


class EventStoreAdapter:
    """Small EventKit wrapper that keeps framework types out of tool logic."""

    def __init__(self) -> None:
        self._eventkit, self._nsdate = _load_eventkit()
        self._store = self._eventkit.EKEventStore.alloc().init()

    def authorization_status(self) -> str:
        status = self._eventkit.EKEventStore.authorizationStatusForEntityType_(
            self._eventkit.EKEntityTypeEvent
        )
        names = (
            ("EKAuthorizationStatusNotDetermined", UNDETERMINED),
            ("EKAuthorizationStatusDenied", DENIED),
            ("EKAuthorizationStatusRestricted", RESTRICTED),
            ("EKAuthorizationStatusAuthorized", AUTHORIZED),
            ("EKAuthorizationStatusFullAccess", AUTHORIZED),
        )
        for constant_name, result in names:
            if status == getattr(self._eventkit, constant_name, object()):
                return result
        return DENIED

    def request_access(self) -> bool:
        completed = threading.Event()
        result = {"granted": False}

        def completion(granted: bool, _error: object) -> None:
            result["granted"] = bool(granted)
            completed.set()

        full_access = getattr(
            self._store, "requestFullAccessToEventsWithCompletion_", None
        )
        if full_access is not None:
            full_access(completion)
        else:
            self._store.requestAccessToEntityType_completion_(
                self._eventkit.EKEntityTypeEvent,
                completion,
            )
        completed.wait()
        return result["granted"]

    def fetch_events(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        predicate = self._store.predicateForEventsWithStartDate_endDate_calendars_(
            self._date(start), self._date(end), None
        )
        return [self._event_from_ek(event) for event in self._store.eventsMatchingPredicate_(predicate)]

    def save_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        calendar: str | None,
        notes: str | None,
    ) -> CalendarEvent:
        calendars = self._store.calendarsForEntityType_(self._eventkit.EKEntityTypeEvent)
        selected = None
        if calendar is None:
            selected = self._store.defaultCalendarForNewEvents()
        else:
            selected = next(
                (candidate for candidate in calendars if str(candidate.title()) == calendar),
                None,
            )
            if selected is None:
                raise CalendarError(f"calendar not found: {calendar}")

        if selected is None:
            raise CalendarError("no default calendar is available")
        event = self._eventkit.EKEvent.eventWithEventStore_(self._store)
        event.setTitle_(title)
        event.setStartDate_(self._date(start))
        event.setEndDate_(self._date(end))
        event.setCalendar_(selected)
        if notes is not None:
            event.setNotes_(notes)
        saved = self._store.saveEvent_span_error_(
            event,
            self._eventkit.EKSpanThisEvent,
            None,
        )
        if isinstance(saved, tuple):
            did_save, error = saved
            if not did_save:
                raise CalendarError(str(error or "EventKit could not save the event"))
        elif saved is False:
            raise CalendarError("EventKit could not save the event")
        return self._event_from_ek(event)

    def _date(self, value: datetime) -> Any:
        return self._nsdate.dateWithTimeIntervalSince1970_(value.timestamp())

    @staticmethod
    def _event_from_ek(event: Any) -> CalendarEvent:
        def date_value(value: Any) -> datetime:
            if isinstance(value, datetime):
                return value
            return datetime.fromtimestamp(value.timeIntervalSince1970(), UTC)

        calendar = event.calendar()
        location = event.location()
        return CalendarEvent(
            title=str(event.title() or ""),
            start=date_value(event.startDate()),
            end=date_value(event.endDate()),
            calendar_name=str(calendar.title() if calendar is not None else ""),
            location=str(location) if location else None,
            all_day=bool(event.isAllDay()),
        )


def parse_iso8601(value: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise CalendarError("calendar dates must be ISO-8601 strings")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise CalendarError(f"invalid ISO-8601 calendar date: {value!r}") from exc


def _parse_window(start_value: str, end_value: str) -> tuple[datetime, datetime]:
    start = parse_iso8601(start_value)
    end = parse_iso8601(end_value)
    if (start.tzinfo is None) != (end.tzinfo is None):
        raise CalendarError("start and end must both include a timezone or both omit it")
    if end <= start:
        raise CalendarError("calendar end must be after start")
    if end - start > MAX_WINDOW:
        raise CalendarError("calendar window cannot exceed 92 days")
    return start, end


def _event_store_adapter() -> EventStoreAdapter:
    return EventStoreAdapter()


def _ensure_authorized(adapter: CalendarAdapter) -> None:
    status = adapter.authorization_status()
    if status == UNDETERMINED:
        if not adapter.request_access():
            raise CalendarError(
                "calendar access was denied; allow Calendar access for the python process"
            )
        status = adapter.authorization_status()
        if status not in {AUTHORIZED, "full_access"}:
            raise CalendarError(
                "calendar access was not granted; allow Calendar access for the python process"
            )
    if status == DENIED:
        raise CalendarError(
            "calendar access is denied; allow Calendar access for the python process"
        )
    if status == RESTRICTED:
        raise CalendarError("calendar access is restricted by macOS")
    if status not in {AUTHORIZED, "full_access"}:
        raise CalendarError(f"calendar access is unavailable: {status}")


def _event_dict(event: CalendarEvent) -> dict[str, object]:
    value = asdict(event)
    value["start"] = event.start.isoformat()
    value["end"] = event.end.isoformat()
    value["calendar"] = value.pop("calendar_name")
    return value


def _failure(exc: Exception) -> StructuredToolResult:
    return _error_result(str(exc), kind="error")


async def _calendar_events(
    _registry: ToolRegistry | None,
    arguments: dict[str, object],
    *,
    adapter: CalendarAdapter | None = None,
) -> StructuredToolResult:
    try:
        start, end = _parse_window(arguments["start"], arguments["end"])
        calendar = arguments.get("calendar")
        if calendar is not None and not isinstance(calendar, str):
            raise CalendarError("calendar must be a string or null")
        active_adapter = adapter if adapter is not None else _event_store_adapter()
        _ensure_authorized(active_adapter)
        events = [
            event
            for event in active_adapter.fetch_events(start, end)
            if event.end > start
            and event.start < end
            and (calendar is None or event.calendar_name == calendar)
        ]
        payload = {"events": [_event_dict(event) for event in events]}
        return _success_result(
            text_block(json.dumps(payload, sort_keys=True)),
            structured_content=payload,
        )
    except (CalendarError, KeyError, TypeError) as exc:
        return _failure(exc)


async def _calendar_create(
    _registry: ToolRegistry | None,
    arguments: dict[str, object],
    *,
    adapter: CalendarAdapter | None = None,
) -> StructuredToolResult:
    try:
        title = arguments["title"]
        if not isinstance(title, str) or not title:
            raise CalendarError("title must be a non-empty string")
        start, end = _parse_window(arguments["start"], arguments["end"])
        calendar = arguments.get("calendar")
        notes = arguments.get("notes")
        if calendar is not None and not isinstance(calendar, str):
            raise CalendarError("calendar must be a string or null")
        if notes is not None and not isinstance(notes, str):
            raise CalendarError("notes must be a string or null")
        active_adapter = adapter if adapter is not None else _event_store_adapter()
        _ensure_authorized(active_adapter)
        event = active_adapter.save_event(title, start, end, calendar, notes)
        payload = {"event": _event_dict(event)}
        return _success_result(
            text_block(json.dumps(payload, sort_keys=True)),
            structured_content=payload,
        )
    except (CalendarError, KeyError, TypeError) as exc:
        return _failure(exc)


def catalog_criteria() -> dict[str, dict[str, object]]:
    """Return neutral router boundaries for the Calendar tools."""

    return {
        "calendar_events": {
            "what": "Read Apple Calendar events in a bounded time window.",
            "not_for": "Zeta task lists (use todo), scheduled agent actions (use automation), or memory and search tools.",
            "examples": [
                "List Apple Calendar events for tomorrow.",
                "Show work calendar events during the next week.",
            ],
        },
        "calendar_create": {
            "what": "Create one Apple Calendar event after approval.",
            "not_for": "Zeta task lists (use todo), scheduled agent actions (use automation), or memory and search tools.",
            "examples": [
                "Create an Apple Calendar event for the design review.",
                "Add a meeting to the work calendar with notes.",
            ],
        },
    }


def register(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "calendar_events",
        _calendar_events,
        requires_approval=False,
        validate_arguments=False,
        description=(
            "Read Apple Calendar events in an ISO-8601 time window of at most 92 days. "
            "Optionally filter by an exact calendar name."
        ),
        parameters={
            "type": "object",
            "properties": {
                "start": {"type": "string", "minLength": 1},
                "end": {"type": "string", "minLength": 1},
                "calendar": {"type": ["string", "null"]},
            },
            "required": ["start", "end", "calendar"],
            "additionalProperties": False,
        },
    )
    registry.register_session_tool(
        "calendar_create",
        _calendar_create,
        approval_subject="calendar",
        validate_arguments=False,
        description=(
            "Create an Apple Calendar event. This changes external calendar data "
            "and requires approval."
        ),
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "minLength": 1},
                "start": {"type": "string", "minLength": 1},
                "end": {"type": "string", "minLength": 1},
                "calendar": {"type": ["string", "null"]},
                "notes": {"type": ["string", "null"]},
            },
            "required": ["title", "start", "end", "calendar", "notes"],
            "additionalProperties": False,
        },
    )
