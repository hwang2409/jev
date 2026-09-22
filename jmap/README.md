# jmap

`jmap` is a CLI primitive for applying typed Jev questions to finite input states.
It emits calibrated answers as JSONL so judgment can compose with shell tools.

The design spec is [2026-09-22-jmap-design.md](docs/superpowers/specs/2026-09-22-jmap-design.md).

## finite workflow

Install the package from this directory with `uv sync`, then use either
`python -m jmap` or the installed `jmap` script. Both entry points call the
same CLI.

```bash
export JEV_API_KEY='...'
printf 'the launch decision is go\n\nold notes\n' | \
  jmap jgrep 'describes the launch decision' --filter=keep
```

The judgment commands are `run`, `jgrep`, `jfilter`, and `gate`. They read
stdin to EOF unless `--input PATH` is set. With `--by file`, one or more
positional paths are also valid finite input units.

```bash
find notes -type f -print0 | xargs -0 jmap jgrep \
  'describes the launch decision' --by file --format=pretty

cat events.jsonl | jmap jfilter 'describes a failed payment' \
  --by record --state-ref event_id

git diff --no-ext-diff --unified=40 | \
  jmap gate --preset diff-risk-heat.yml --by hunk \
  --policy 'any(change_scope.score >= 2 or security_boundary_change.noul >= 0.75)'
```

Judgment stdout is JSONL only. Each invocation ends with one `coverage` record.
`--format=pretty` writes selected successful results to stderr; it does not
change stdout. Errors, partial results, skipped states, and coverage remain on
stdout. Human warnings also use stderr.

For a file-path invocation, jmap normalizes each path before judging it. The
equivalent JSONL input is `{"path":"...","content":"..."}`. A path list
requires `--by file`; it cannot be combined with `--input`.

### transcript

This is a complete, finite `jgrep` transcript. The answer value is illustrative;
the record shape and terminal coverage line are stable.

```text
$ printf 'the launch decision is go\n' | jmap jgrep 'describes the launch decision'
{"answers":{"matches_query":{"noul":0.93,"type":"noul"}},"meta":{"cache":"miss","chunker":"para","model":"jev-1.13.0","preset":"jgrep","preset_version":"1"},"record_type":"result","state_ref":"stdin#P1"}
{"coverage":"complete","coverage_counts":{"discovered":1,"emitted":1,"failed":0,"judged":1,"skipped":0},"coverage_reasons":[],"meta":{"cache":"not_applicable","chunker":"para","model":"jev-1.13.0","preset":"jgrep","preset_version":"1"},"record_type":"coverage"}
```

If an input limit or scan cap prevents a formed state from being judged, jmap
emits a skip-summary error, warns on stderr, emits partial coverage, and exits
with `2`. `jgrep` defaults to a 512-paragraph cap. Set `--max-chunks` for a
deterministic smaller or larger cap.

```text
$ printf 'one\n\ntwo\n\nthree\n' | jmap jgrep 'mentions one' --max-chunks 1 >/tmp/jmap.jsonl
jmap: warning: results are partial; coverage reasons: scan_cap
$ echo $?
2
```

### presets and cache

Use preset commands to inspect reviewable YAML without invoking judgment:

```bash
jmap preset list
jmap preset show jgrep
jmap preset validate ./review-preset.yml
```

Complete answers use the content-addressed cache at
`${JMAP_CACHE_DIR:-~/.cache/jmap}/answers/`. Cache keys include the model,
preset version, resolved chunking and limits, question battery, and state.

```bash
jmap cache export --preset jgrep > jgrep-training.jsonl
jmap cache clear --preset jgrep
```

Export writes one answer triple per line and no coverage record. Clear writes a
small status object. Neither command calls the Jev API.

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
