"""Apple Calendar tools backed by EventKit."""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ...types import StructuredToolResult
from ..registry import ToolRegistry, _error_result, _success_result, text_block

UNDETERMINED = "undetermined"
DENIED = "denied"
RESTRICTED = "restricted"
AUTHORIZED = "authorized"
MAX_WINDOW = timedelta(days=92)


class CalendarError(ValueError):
    """A user-facing Calendar tool error."""


class CalendarArgumentError(CalendarError):
    """A Calendar tool argument does not match its declared schema."""


@dataclass(frozen=True, slots=True)
class CalendarEvent:
    title: str
    start: datetime
    end: datetime
    calendar_name: str
    location: str | None
    all_day: bool
    notes: str | None = None
    floating: bool = False
    time_zone: str | None = None


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
        """Convert dates; naive ISO-8601 inputs use the process local time zone."""

        if value.tzinfo is None:
            timestamp = time.mktime(value.timetuple()) + value.microsecond / 1_000_000
        else:
            timestamp = value.timestamp()
        return self._nsdate.dateWithTimeIntervalSince1970_(timestamp)

    @staticmethod
    def _event_from_ek(event: Any) -> CalendarEvent:
        def event_timezone() -> tuple[tzinfo | None, str | None]:
            timezone_method = getattr(event, "timeZone", None)
            raw_timezone = timezone_method() if callable(timezone_method) else None
            if raw_timezone is None:
                return None, None
            if isinstance(raw_timezone, tzinfo):
                return raw_timezone, getattr(raw_timezone, "key", None) or raw_timezone.tzname(None)
            name_method = getattr(raw_timezone, "name", None)
            name = name_method() if callable(name_method) else None
            if isinstance(name, str) and name:
                try:
                    return ZoneInfo(name), name
                except ZoneInfoNotFoundError:
                    pass
            seconds_method = getattr(raw_timezone, "secondsFromGMTForDate_", None)
            if callable(seconds_method):
                seconds = seconds_method(event.startDate())
                if isinstance(seconds, int):
                    offset = timedelta(seconds=seconds)
                    return timezone(offset), str(raw_timezone)
            return UTC, str(raw_timezone)

        event_tz, time_zone = event_timezone()

        def date_value(value: Any) -> datetime:
            timestamp = (
                value.timestamp()
                if isinstance(value, datetime)
                else value.timeIntervalSince1970()
            )
            if event_tz is None:
                return datetime.fromtimestamp(timestamp)
            return datetime.fromtimestamp(timestamp, event_tz)

        calendar = event.calendar()
        location = event.location()
        all_day = bool(event.isAllDay())
        start = date_value(event.startDate())
        end = date_value(event.endDate())
        if all_day:
            start = start.replace(tzinfo=None)
            end = end.replace(tzinfo=None)
        notes_method = getattr(event, "notes", None)
        notes_value = notes_method() if callable(notes_method) else None
        return CalendarEvent(
            title=str(event.title() or ""),
            start=start,
            end=end,
            calendar_name=str(calendar.title() if calendar is not None else ""),
            location=str(location) if location else None,
            all_day=all_day,
            notes=str(notes_value) if notes_value is not None else None,
            floating=event_tz is None and not all_day,
            time_zone=None if event_tz is None else time_zone,
        )


def parse_iso8601(value: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise CalendarArgumentError("calendar dates must be ISO-8601 strings")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise CalendarArgumentError(f"invalid ISO-8601 calendar date: {value!r}") from exc


def _parse_window(start_value: str, end_value: str) -> tuple[datetime, datetime]:
    start = parse_iso8601(start_value)
    end = parse_iso8601(end_value)
    if (start.tzinfo is None) != (end.tzinfo is None):
        raise CalendarArgumentError("start and end must both include a timezone or both omit it")
    if end <= start:
        raise CalendarArgumentError("calendar end must be after start")
    if end - start > MAX_WINDOW:
        raise CalendarArgumentError("calendar window cannot exceed 92 days")
    return start, end


class FakeCalendarAdapter:
    """Deterministic calendar adapter used by offline evaluations."""

    def __init__(self, seed_path: Path) -> None:
        self.seed_path = seed_path
        try:
            payload = json.loads(seed_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CalendarError(f"could not load fake calendar seed: {seed_path}") from exc
        raw_events = payload.get("events") if isinstance(payload, dict) else payload
        if not isinstance(raw_events, list):
            raise CalendarError("fake calendar seed must contain an events list")
        self.events = [_fake_event(raw_event) for raw_event in raw_events]
        self.created_log: list[dict[str, object]] = []
        atexit.register(self._write_created_log)

    def authorization_status(self) -> str:
        return AUTHORIZED

    def request_access(self) -> bool:
        return True

    def fetch_events(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        return list(self.events)

    def save_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        calendar: str | None,
        notes: str | None,
    ) -> CalendarEvent:
        event = CalendarEvent(
            title=title,
            start=start,
            end=end,
            calendar_name=calendar or "default",
            location=None,
            all_day=False,
            notes=notes,
        )
        self.events.append(event)
        self.created_log.append(_event_dict(event))
        self._write_created_log()
        return event

    def _write_created_log(self) -> None:
        output_path = Path(f"{self.seed_path}.out")
        output_path.write_text(
            json.dumps(self.created_log, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _fake_event(raw_event: object) -> CalendarEvent:
    if not isinstance(raw_event, dict):
        raise CalendarError("fake calendar events must be objects")
    required = {"title", "start", "end", "calendar", "all_day"}
    if set(raw_event) != required:
        raise CalendarError(
            "fake calendar events require title, start, end, calendar, and all_day"
        )
    if any(not isinstance(raw_event[key], str) for key in ("title", "start", "end", "calendar")):
        raise CalendarError("fake calendar event text fields must be strings")
    if type(raw_event["all_day"]) is not bool:
        raise CalendarError("fake calendar event all_day must be a boolean")
    return CalendarEvent(
        title=raw_event["title"],
        start=parse_iso8601(raw_event["start"]),
        end=parse_iso8601(raw_event["end"]),
        calendar_name=raw_event["calendar"],
        location=None,
        all_day=raw_event["all_day"],
    )


_fake_adapters: dict[Path, FakeCalendarAdapter] = {}


def _event_store_adapter() -> CalendarAdapter:
    configured = os.environ.get("ZETA_CALENDAR_ADAPTER")
    if configured is not None:
        if not configured.startswith("fake:"):
            raise CalendarError(
                "ZETA_CALENDAR_ADAPTER must be unset or use fake:<path.json>"
            )
        raw_seed_path = configured.removeprefix("fake:")
        if not raw_seed_path:
            raise CalendarError("fake calendar adapter requires a seed file path")
        seed_path = Path(raw_seed_path)
        adapter = _fake_adapters.get(seed_path)
        if adapter is None:
            adapter = FakeCalendarAdapter(seed_path)
            _fake_adapters[seed_path] = adapter
        return adapter
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
    if event.all_day:
        value["start"] = event.start.date().isoformat()
        value["end"] = event.end.date().isoformat()
    else:
        value["start"] = event.start.isoformat()
        value["end"] = event.end.isoformat()
    value["calendar"] = value.pop("calendar_name")
    return value


def _local_aware(value: datetime) -> datetime:
    """Treat naive datetimes as local time and convert all values to local time."""

    return value.astimezone()


def _failure(exc: Exception, *, kind: str = "error") -> StructuredToolResult:
    return _error_result(str(exc), kind=kind)


def _validate_tool_arguments(
    arguments: object,
    *,
    required: set[str],
) -> dict[str, object]:
    if type(arguments) is not dict:
        raise CalendarArgumentError("arguments must be an object")
    missing = sorted(required - set(arguments))
    if missing:
        raise CalendarArgumentError(f"missing required arguments: {', '.join(missing)}")
    extra = sorted(set(arguments) - required)
    if extra:
        raise CalendarArgumentError(f"unexpected arguments: {', '.join(extra)}")
    return arguments


def _validate_events_arguments(arguments: object) -> dict[str, object]:
    values = _validate_tool_arguments(
        arguments,
        required={"start", "end", "calendar"},
    )
    if type(values["start"]) is not str or type(values["end"]) is not str:
        raise CalendarArgumentError("start and end must be ISO-8601 strings")
    if values["calendar"] is not None and type(values["calendar"]) is not str:
        raise CalendarArgumentError("calendar must be a string or null")
    return values


def _validate_create_arguments(arguments: object) -> dict[str, object]:
    values = _validate_tool_arguments(
        arguments,
        required={"title", "start", "end", "calendar", "notes"},
    )
    if type(values["title"]) is not str or not values["title"]:
        raise CalendarArgumentError("title must be a non-empty string")
    if type(values["start"]) is not str or type(values["end"]) is not str:
        raise CalendarArgumentError("start and end must be ISO-8601 strings")
    if values["calendar"] is not None and type(values["calendar"]) is not str:
        raise CalendarArgumentError("calendar must be a string or null")
    if values["notes"] is not None and type(values["notes"]) is not str:
        raise CalendarArgumentError("notes must be a string or null")
    return values


async def _calendar_events(
    _registry: ToolRegistry | None,
    arguments: dict[str, object],
    *,
    adapter: CalendarAdapter | None = None,
) -> StructuredToolResult:
    try:
        values = _validate_events_arguments(arguments)
        start, end = _parse_window(values["start"], values["end"])
        calendar = values["calendar"]
        active_adapter = adapter if adapter is not None else _event_store_adapter()
        _ensure_authorized(active_adapter)
        local_start = _local_aware(start)
        local_end = _local_aware(end)
        events = [
            event
            for event in active_adapter.fetch_events(start, end)
            if _local_aware(event.end) > local_start
            and _local_aware(event.start) < local_end
            and (calendar is None or event.calendar_name == calendar)
        ]
        payload = {"events": [_event_dict(event) for event in events]}
        return _success_result(
            text_block(json.dumps(payload, sort_keys=True)),
            structured_content=payload,
        )
    except CalendarArgumentError as exc:
        return _failure(exc, kind="invalid_arguments")
    except (CalendarError, KeyError, TypeError) as exc:
        return _failure(exc)


async def _calendar_create(
    _registry: ToolRegistry | None,
    arguments: dict[str, object],
    *,
    adapter: CalendarAdapter | None = None,
) -> StructuredToolResult:
    try:
        values = _validate_create_arguments(arguments)
        title = values["title"]
        start, end = _parse_window(values["start"], values["end"])
        calendar = values["calendar"]
        notes = values["notes"]
        active_adapter = adapter if adapter is not None else _event_store_adapter()
        _ensure_authorized(active_adapter)
        event = active_adapter.save_event(title, start, end, calendar, notes)
        payload = {"event": _event_dict(event)}
        return _success_result(
            text_block(json.dumps(payload, sort_keys=True)),
            structured_content=payload,
        )
    except CalendarArgumentError as exc:
        return _failure(exc, kind="invalid_arguments")
    except (CalendarError, KeyError, TypeError) as exc:
        return _failure(exc)


def catalog_criteria() -> dict[str, dict[str, object]]:
    """Return neutral router boundaries for the Calendar tools."""

    return {
        "calendar_events": {
            "what": "Read Apple Calendar events in a bounded time window.",
            "not_for": "Creating events (use calendar_create), Zeta task lists (use todo), scheduled agent actions (use automation), or memory and search tools.",
            "examples": [
                "List Apple Calendar events for tomorrow.",
                "Show work calendar events during the next week.",
            ],
        },
        "calendar_create": {
            "what": "Create one Apple Calendar event after approval.",
            "not_for": "Reading events (use calendar_events), Zeta task lists (use todo), scheduled agent actions (use automation), or memory and search tools.",
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
        validate_arguments=True,
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
        validate_arguments=True,
        approval_denial_is_cancellation=True,
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
