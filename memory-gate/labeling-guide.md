# Memory-gate gold-labeling guide

Label the presented (already truncated) excerpt for the *next step in the case*, not whether it is generally true and not whether its source path was retrieved. `positive` means useful and safe to inject now. `negative` means irrelevant, stale, contradictory, too generic, or unsafe. `ambiguous` means a reasonable rater cannot decide from the query and excerpt alone. Do not use ambiguous as a convenient middle score.

Path membership is provenance only; forbidden paths are reported separately. Label the exact presented text, including truncation. For the blinded 25% repeat, relabel without seeing the first label. Raw agreement is intra-rater consistency, not independent truth.

## Worked examples from the pausanias evaluation corpus

1. “Where did we decide to put the cache?” / `notes/architecture.md` / `Architecture > Cache` / “The cache lives in SQLite under the project data directory.” — **positive**: directly answers the next question and is safe.
2. “What is the release checklist?” / `notes/old-plan.md` / `Planning` / “Release checklist (2024): publish the deprecated binary first.” — **negative**: stale guidance is unsafe even though it mentions the topic.
3. “Can you summarize the migration?” / `journal/meeting.md` / `Meeting` / “We discussed moving the service; follow-up details were not recorded.” — **ambiguous**: it may establish context but does not provide enough detail to decide whether injection helps.
