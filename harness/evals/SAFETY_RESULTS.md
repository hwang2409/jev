# safety-tier acceptance results

the offline acceptance eval runs the layer-0 classifier and the full safety-tier
decision path. it replaces `zeta.providers.jev.safety_score` with a valid,
most-benign `SafetyScoreResult`. dangerous rows must still stop at layer 0.

## offline result

| metric | result |
| --- | ---: |
| corpus rows | 160 |
| dangerous rows | 136 |
| safety recall | 1.0000 (136/136) |
| layer-0 exact | 1.0000 (160/160) |
| benign auto-approve rate | 1.0000 (24/24) |
| false-escalate rate | 0.0000 (0/24) |

### per-category layer-0 confusion

| category | rows | expected -> actual |
| --- | ---: | --- |
| benign | 20 | analyzable -> analyzable: 20 |
| compound_and_substitution | 11 | deny -> deny: 4; escalate -> escalate: 7 |
| credential_stores | 21 | analyzable -> analyzable: 1; deny -> deny: 20 |
| destructive_system_paths | 17 | deny -> deny: 12; escalate -> escalate: 5 |
| nested_interpreters | 18 | escalate -> escalate: 18 |
| novel_wrappers | 12 | deny -> deny: 7; escalate -> escalate: 5 |
| persistence_network | 26 | analyzable -> analyzable: 1; deny -> deny: 14; escalate -> escalate: 11 |
| pipe_to_shell | 10 | deny -> deny: 10 |
| privilege_escalation | 10 | deny -> deny: 10 |
| runner_wrappers | 15 | analyzable -> analyzable: 2; escalate -> escalate: 13 |

### corpus composition

| source | rows |
| --- | ---: |
| benign | 20 |
| novel | 12 |
| round1 | 16 |
| round2 | 15 |
| round3 | 39 |
| round4 | 15 |
| round5 | 17 |
| round6 | 26 |

| expected class | rows |
| --- | ---: |
| analyzable | 24 |
| deny | 77 |
| escalate | 59 |

## what this covers

- deterministic layer-0 coverage for privilege escalation, credentials,
  shell pipelines, interpreters, wrappers, destructive paths, persistence,
  network exfiltration, compound syntax, and benign commands;
- the full offline decision path with a valid benign Jev result;
- fail-closed protection for every deny and escalate row under that benign
  result;
- a live Jev smoke path for six representative commands when `--live` and
  `JEV_API_KEY` are both present. it also forces a client error and checks
  that the tier returns `ask` or `deny`.

## what this does not cover

it does not run the full agent-loop live smoke. that path needs a real
`ANTHROPIC_API_KEY` and remains pending Henry. the default eval never calls
the network and never executes a corpus command.

the corpus is a regression guard, not an independent adversarial dataset. its
rows are derived from the JEV-54 safety-tier test suite and the six review-round
verdicts (`source` values `round1`..`round6`), so the categories map 1:1 to the
shell shapes layer 0 already handles. recall = 1.0 confirms the merged classifier
still covers every known shape; it is not a discovery of new coverage.
