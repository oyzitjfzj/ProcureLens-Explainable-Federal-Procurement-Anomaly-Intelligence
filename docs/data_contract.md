# ProcureLens Data Contract

## Scope

This document describes the semantic contract between source ingestion and analytical layers. It is intentionally stricter than a list of CSV columns: each field has a meaning that downstream features and models must preserve.

ProcureLens currently targets USAspending prime-award contract transaction data, but the canonical transaction object is source-neutral.

## Canonical identity

Every `ProcurementTransaction` requires:

- `transaction_id` — stable identity for one source transaction;
- `award_id` — stable identity for the containing award;
- `SourceRecordRef` lineage — source name, source transaction identity, retrieval timestamp, optional source schema, and optional raw-record SHA-256.

`piid`, modification number, parent award identity, and award type remain separate fields. ProcureLens does not infer one identifier from another when the source did not provide it.

## Money semantics

### `action_obligation`

Transaction-level change in obligation. It is additive across distinct transactions and may be:

- positive — additional obligation;
- zero — no obligation change on that action;
- negative — deobligation.

Negative values are valid data and are never made positive merely to fit a model.

### `award_total_obligation`

Award-level cumulative/summary obligation as reported for the transaction context. It is **not** treated as an additive transaction flow and is never blindly summed across modifications.

Whenever an analysis can use either money concept, the basis is explicit and fingerprinted.

## Dates and fiscal context

`action_date` is the date of the transaction action. Peer-group logic may map it to federal fiscal year or another explicit time scope.

For competition and award-change analyses, a later modification date must not silently replace the original award-formation context. Indexed base-award evidence is reused where the analytical question is about formation conditions.

## Vendor identity

ProcureLens resolves two independent scopes:

- **entity** — the reported recipient itself;
- **ultimate parent** — the reported parent recipient.

Within a scope, identity resolution follows explicit source evidence:

1. UEI when present;
2. legacy identifier when present;
3. conservative normalized reported name as a weaker fallback.

Parent absence does not fall back to entity identity. Name fallback uses conservative whitespace/case normalization rather than fuzzy entity resolution. The identity method remains visible in evidence.

## Award lifecycle

The reported modification number is the authoritative lifecycle evidence currently used by ProcureLens:

- numeric zero spellings such as `0` or `000` → observed base/new-award action;
- nonzero reported modification values → modification;
- missing/blank modification number → lifecycle unknown.

Lifecycle is not guessed from action amount, date ordering, PIID shape, or transaction position.

This distinction is essential for vendor-frequency and competition reference populations: repeated modifications must not be counted as repeated new awards.

## Agency and category context

Peer construction can use:

- top-level awarding agency;
- awarding subtier agency;
- exact PSC;
- NAICS at 6-, 4-, or 2-digit resolution;
- explicit time scope;
- award type where configured.

Missing agency/category components make a candidate peer level unavailable. ProcureLens does not silently substitute a category-free global fallback in the default federal-contract context plans.

## Competition fields

Competition evidence stays decomposed.

### Extent competed

Represents the reported competition process/status. ProcureLens normalizes recognized descriptions into explicit categories such as full and open competition, full and open after exclusion, competed under simplified acquisition, and reported noncompetitive states.

### Number of offers received

Represents an observed outcome, not the process itself:

- 0 reported offers;
- 1 reported offer;
- multiple reported offers;
- missing/unknown.

A single offer does **not** automatically convert a reported full-and-open process into a noncompetitive process.

### Solicitation procedure

Kept separately from extent competed. Recognized procedures include simplified acquisition, only-one-source solicitation, negotiated proposal/quote, sealed bid, and other explicitly mapped source descriptions.

### Other-than-full-and-open authority

Reported authorities/reasons are retained as evidence. Their presence does not by itself mean wrongdoing; their absence when a noncompetitive extent is reported can be surfaced as missing evidence.

### Cross-field conflicts

If reported fields point in materially different directions, ProcureLens records an evidence conflict instead of choosing one field and discarding the other.

## Observed winners are not the full competitor market

USAspending award transactions identify winning recipients. They do not necessarily reveal every bidder or every economically capable supplier.

Vendor-market metrics therefore use language such as **observed winning vendors**. They must not be presented as a complete count of potential competitors.

## Modification/change activity

Award-change aggregation distinguishes:

- base-award action count;
- modification action count;
- lifecycle-unknown action count;
- distinct reported modification numbers;
- repeated transactions on one modification number;
- positive modification obligation;
- deobligation magnitude;
- zero-dollar modification count;
- net modification activity;
- gross absolute modification activity;
- observation/follow-up time.

These are descriptive facts. A modification is not labeled unjustified merely because it is large or frequent.

## Missingness

Missingness is part of the analytical contract.

Examples:

- missing award total → an award-total-based amount feature may be unavailable;
- zero peer MAD/IQR → the corresponding robust distance is undefined;
- missing vendor identity → vendor-frequency context may be unavailable;
- missing base-award formation evidence → some competition/change contexts may be unavailable;
- insufficient peer support → context resolution may fall back to a broader approved level or remain unavailable.

The feature layer carries an unavailable reason. It never silently emits zero for an undefined statistic.

## Reference versus scoring populations

Reference snapshots are built only from the caller-designated reference population. Scoring transactions are resolved against frozen snapshots and cannot mutate them.

If a historical target is already indexed in the reference population, the relevant analysis uses explicit leave-one-out behavior to avoid self-influence.

The same principle continues downstream: preprocessing, detectors, and calibration fit on training artifacts only and scoring rows reuse the frozen fitted state.

## Data-quality evidence is not anomaly evidence

Coverage, missing fields, schema issues, source integrity, and readiness are measured explicitly. These facts can block or qualify an analysis, but they do not automatically increase a transaction's anomaly score.

## Output semantics

Public exports may contain a client-compatible `risk_score_0_100` alias. Its documented semantics are the same as `review_priority_score`: a relative anomaly-review priority derived from calibrated ensemble position.

It is not a probability of fraud, corruption, illegality, waste, or culpability.
