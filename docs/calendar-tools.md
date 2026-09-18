# apple calendar tools

the calendar tools use EventKit through `pyobjc-framework-EventKit`.
they access Apple Calendar only. they do not access reminders.

## permission flow

the first live call creates an `EKEventStore` for the current python process.
if macOS reports `not determined`, the tool calls EventKit's full calendar
access request. macOS then shows a Calendar permission prompt for that python
process. The prompt can appear when `calendar_events` or `calendar_create`
runs for the first time.

the tool does not prompt again when macOS reports an existing decision.

the tool handles these states:

- `not determined`: request calendar access once, then continue only if granted.
- `authorized` or `full access`: read or create calendar events.
- `denied`: return an error that asks the user to allow Calendar access for the
  python process.
- `restricted`: return an error because macOS policy blocks access.
- `write-only`: treated as insufficient because these tools read event data.

event reads accept ISO-8601 `start` and `end` values. A window cannot exceed
92 days. A named calendar uses an exact name match.

naive ISO-8601 inputs use the process local time zone. Inputs with an explicit
offset keep that offset. All-day events return date-only `start` and `end`
values. Floating events return naive date-times with `floating` set to `true`.

## live smoke test

run this command on macOS from the repository root:

```bash
uv run python - <<'PY'
import asyncio

from zeta.tools.calendar import _calendar_events

result = asyncio.run(
    _calendar_events(
        None,
        {
            "start": "2026-01-01T00:00:00",
            "end": "2026-01-02T00:00:00",
            "calendar": None,
        },
    )
)
print(result)
PY
```

approve the macOS prompt for the python process. A denied prompt returns a
structured tool error. The command does not create or change calendar data.
