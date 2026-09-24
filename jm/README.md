# jm

formerly jmap, renamed 2026-09-23 (Java jmap conflict)

`jm` is a CLI primitive for applying typed Jev questions to finite input states.
It emits calibrated answers as JSONL so judgment can compose with shell tools.

The design spec is [2026-09-22-jmap-design.md](docs/superpowers/specs/2026-09-22-jmap-design.md).

## finite workflow

Install the package from this directory with `uv sync`, then use either
`python -m jm` or the installed `jm` script. Both entry points call the
same CLI.

```bash
export VERCEL_AI_GATEWAY='...'
printf 'the launch decision is go\n\nold notes\n' | \
  jm jgrep --query 'describes the launch decision' --filter=keep
```

The judgment commands are `run`, `jgrep`, `jfilter`, and `gate`. They read
stdin to EOF unless `--input PATH` is set. With `--by file`, one or more
positional paths are also valid finite input units.

```bash
find notes -type f -print0 | xargs -0 jm jgrep \
  --query 'describes the launch decision' --by file --format=pretty

cat events.jsonl | jm jfilter 'describes a failed payment' \
  --by record --state-ref event_id

git diff --no-ext-diff --unified=40 | \
  jm gate --preset diff-risk-heat.yml --by hunk \
  --policy 'any(change_scope.score >= 2 or security_boundary_change.noul >= 0.75)'
```

Judgment stdout is JSONL only. Each invocation ends with one `coverage` record.
`--format=pretty` writes selected successful results to stderr; it does not
change stdout. Errors, partial results, skipped states, and coverage remain on
stdout. Human warnings also use stderr.

For a file-path invocation, jm normalizes each path before judging it. The
equivalent JSONL input is `{"path":"...","content":"..."}`. A path list
requires `--by file`; it cannot be combined with `--input`.

### transcript

This is a complete, finite `jgrep` transcript. The answer value is illustrative;
the record shape and terminal coverage line are stable.

```text
$ printf 'the launch decision is go\n' | jm jgrep --query 'describes the launch decision'
{"answers":{"matches_query":{"noul":0.93,"type":"noul"}},"meta":{"cache":"miss","chunker":"para","model":"typesafe-ai/jev","preset":"jgrep","preset_version":"1"},"record_type":"result","state_ref":"stdin#P1"}
{"coverage":"complete","coverage_counts":{"discovered":1,"emitted":1,"failed":0,"judged":1,"skipped":0},"coverage_reasons":[],"meta":{"cache":"not_applicable","chunker":"para","model":"typesafe-ai/jev","preset":"jgrep","preset_version":"1"},"record_type":"coverage"}
```

If an input limit or scan cap prevents a formed state from being judged, jm
emits a skip-summary error, warns on stderr, emits partial coverage, and exits
with `2`. `jgrep` defaults to a 512-paragraph cap. Set `--max-chunks` for a
deterministic smaller or larger cap.

```text
$ printf 'one\n\ntwo\n\nthree\n' | jm jgrep --query 'mentions one' --max-chunks 1 >/tmp/jm.jsonl
jm: warning: results are partial; coverage reasons: scan_cap
$ echo $?
2
```

### presets and cache

Use preset commands to inspect reviewable YAML without invoking judgment:

```bash
jm preset list
jm preset show jgrep
jm preset validate ./review-preset.yml
```

Complete answers use the content-addressed cache at
`${JM_CACHE_DIR:-~/.cache/jm}/answers/`. Cache keys include the model,
preset version, resolved chunking and limits, question battery, and state.

```bash
jm cache export --preset jgrep > jgrep-training.jsonl
jm cache clear --preset jgrep
```

Export writes one answer triple per line and no coverage record. Clear writes a
small status object. Neither command calls the Jev API.

### offline eval and live smoke

Run the offline eval harness with the same CLI-facing paths and a deterministic
injected judge:

```bash
cd jm && uv run pytest -q tests/test_eval.py tests/test_answers.py \
  tests/test_cache.py tests/test_chunkers.py tests/test_gates.py \
  tests/test_presets.py tests/test_runner.py
```

The live smoke is manual and is not part of pytest or CI. It skips before
creating a client when a Vercel AI Gateway key is absent. When a key is available, run:

```bash
cd jm && VERCEL_AI_GATEWAY="$VERCEL_AI_GATEWAY" \
  uv run python scripts/live_api_smoke.py
```

The smoke sends one small three-question battery. It prints latency, model,
usage, and typed answers.

### client request observer

`JevClient.set_request_observer` accepts a synchronous zero-argument callback.
jm calls it directly before every transport attempt, including retries.
`evaluate_async` does not await it, so the callback runs synchronously in the
async path. If it raises, the exception propagates unchanged and jm stops
before sending that attempt. Pass `None` to remove the observer.

```python
client.set_request_observer(on_request)
client.set_request_observer(None)
```

### exit codes

| code | meaning |
| ---: | --- |
| 0 | complete coverage and no active gate failure |
| 1 | a gate failure policy evaluated true |
| 2 | operational failure, incomplete coverage, or fail-closed gate |
| 64 | command usage, preset validation, or policy syntax error |

Wait for the terminal coverage record before treating a judgment stream as
complete. A usage error occurs before execution and emits no coverage record.

v1 is a finite workflow tool, not a CI security boundary. Input can contain
prompt injection, and `diff-risk-heat` is triage evidence. Keep tests, review,
permissions, and other security controls in place.
