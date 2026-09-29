# RETRACTION — safety.json in this run directory is INVALID

The frozen safety artifact (accepts: true, 0/446) is VACUOUS and must not
authorize anything. Root cause: the operational merge adapter attached locomo
checkpoint retrieval records whose "heading" field is a string; production
candidate normalization (correctly) discards such items, so all 446 cases
collapsed to zero-candidate markers with retrieval_provenance present —
no Jev battery was ever scored. Discovered by the repeatability probe's
model-identity refusal (marker rows carry no identities).

Calibration artifacts and LOCK (tau=0.58) in this directory remain valid —
the defect was confined to the safety lane input. Safety is re-run in a
fresh run directory with corrected input shapes and a new witness.
Structural hardening (mass-discard guard) tracked in jev repo.
