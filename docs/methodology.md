# ProcureLens Methodology

## Purpose

ProcureLens is an explainable anomaly-intelligence system for United States federal procurement transactions. Its output is intended to prioritize records for human review. It does **not** determine fraud, corruption, collusion, illegality, waste, or contractor culpability.

The architecture is deliberately evidence-first. A model score is allowed to exist only after source integrity, canonical transaction semantics, missingness, peer context, feature provenance, train/score separation, detector provenance, calibration, and review-policy semantics are explicit.

## Core reasoning contract

ProcureLens separates five concepts that are often collapsed in smaller anomaly-detection prototypes:

1. **Reported fact** — what the source data actually says.
2. **Reference context** — what comparable historical observations look like.
3. **Candidate feature** — a transparent measurement that may be used by a detector.
4. **Anomaly evidence** — detector/calibration/ensemble output describing statistical unusualness.
5. **Review policy** — an explicit operational decision about which records humans will inspect.

No layer is allowed to silently reinterpret the layer before it.

## Canonical transaction semantics

Every source row is normalized into one source-neutral `ProcurementTransaction` before analytical work begins. The canonical contract keeps transaction-level obligation changes separate from award-level totals.

Important invariants include:

- `action_obligation` is a transaction-level change and may legitimately be negative.
- `award_total_obligation` is an award-level cumulative value and is never summed across transaction rows as if it were additive activity.
- source lineage is retained on every canonical transaction;
- missing source values remain missing;
- vendor entity and ultimate-parent identities are distinct analytical scopes;
- modification/base-award lifecycle evidence is explicit rather than inferred from amount or date heuristics.

## Integrity before analysis

The USAspending source layer separates control-plane download orchestration, artifact materialization, archive/table reading, schema interpretation, and canonical loading. The loader is designed to avoid partially trusted analysis populations: whole-artifact integrity and schema checks occur before a dataset is treated as analyzable.

Quality profiling and readiness gating remain distinct from anomaly scoring. Data-quality weakness can block or degrade an analysis without becoming a hidden anomaly penalty.

## Contextual evidence families

### 1. Amount context

Raw dollar size is not judged globally. ProcureLens constructs hierarchical agency/category/time peer groups and resolves a target against the first caller-approved group with sufficient support.

When the target is part of the reference population, its own observation is removed before peer statistics are computed. Candidate amount measurements include empirical positions, median/MAD distances, IQR distances, direction, magnitude, and same-direction support.

No fixed amount threshold is encoded as "suspicious."

### 2. Vendor new-award frequency

Vendor frequency is based on observed **new/base awards**, not raw transaction count. Later modifications to an old award therefore do not masquerade as repeated new wins.

Entity and ultimate-parent identity scopes remain separate. The vendor-market layer describes observed winning vendors in the available procurement data; it does not claim to represent every potential bidder in the economic market.

Candidate measurements include new-award counts, shares, equal-share lift, peer empirical position, robust distances, market concentration context, and identity coverage.

### 3. Competition evidence

Competition process and competition outcome are separate dimensions.

For example, `Full and Open Competition` with one received offer is represented as a competitive reported process **and** a single-offer outcome. One fact does not overwrite the other.

ProcureLens also preserves solicitation procedure, reported other-than-full-and-open authority, missingness, and cross-field conflicts. Contextual prevalence is estimated over comparable base awards, not repeated modifications of the same award.

### 4. Award-change behavior

Modifications are normal parts of many contracts, so "many modifications" is not a built-in wrongdoing rule.

ProcureLens separately measures positive obligations, deobligations, zero-dollar actions, net activity, gross activity, modification counts, repeated modification numbers, follow-up exposure, and ratios to usable base-award obligation evidence.

Reference comparisons are anchored to award-formation context. Exposure time is preserved because older awards have had more opportunity to accumulate changes.

## Frozen reference populations

Feature construction and model fitting both enforce a train/reference-versus-score separation.

A `FeatureBuildPlan` explicitly pins:

- peer hierarchies;
- amount basis;
- minimum peer support;
- vendor identity scope;
- vendor-market support and coverage requirements;
- competition support and coverage requirements;
- award-change peer support;
- resource budgets;
- quantile method;
- feature catalog fingerprint.

Reference transactions build immutable snapshots. Scoring transactions are resolved against those snapshots and cannot mutate them. Historical targets that already belong to a reference population use the relevant leave-one-out logic.

## Candidate feature catalog

All candidate feature definitions are registered in one catalog with globally unique names, source family, description, and source-definition fingerprint.

The catalog does **not** choose model columns. A detector configuration must explicitly name and order its selected features. This prevents newly added descriptive measurements from silently entering an existing trained model.

Missing values carry reasons. Undefined robust distances, insufficient peer support, unavailable vendor identity, or unavailable formation context are never silently replaced by zero at the feature layer.

## Preprocessing

Preprocessing is an explicit, fingerprinted contract per selected feature. Available strategies include requiring presence or fitting a caller-selected imputation strategy, optionally retaining an original-missingness indicator, and applying an explicitly selected numeric transform.

All fitted preprocessing state is learned from the training population only. Scoring rows cannot change medians, means, variances, robust quantiles, or any other fitted state.

Zero-spread scaling states fail loudly when the requested transformation would be undefined instead of silently creating artificial information.

## Detector architecture

ProcureLens currently uses two deliberately different unsupervised views.

### Isolation Forest

The Isolation Forest adapter pins estimator configuration, random seed, training-matrix fingerprint, scikit-learn version, float-conversion diagnostics, and fitted-model/tree structure fingerprints.

`contamination` is not used as a hidden fraud or review-rate assumption. Review selection is handled later by a separate policy layer.

### Frozen empirical-tail detector

The second detector is a ProcureLens empirical-tail implementation inspired by empirical-distribution outlier methods. Per-feature training distributions are frozen; scoring data never refits the ECDF reference.

It preserves left-tail, right-tail, and skew-directed evidence and combines them into an additive tail score. Because a finite empirical sample cannot resolve arbitrarily small probabilities without additional distribution assumptions, its tail resolution is explicitly finite.

This detector is intentionally not presented as an exact reimplementation of a third-party ECOD/COPOD package.

## Score calibration

Raw detector scores are not directly averaged because detector score scales and directions differ.

Calibration is fitted only on training detector scores and preserves:

- tie-aware empirical lower/mid/upper position;
- robust median/MAD distance where defined;
- robust IQR distance where defined;
- detector/model/training/scoring provenance.

The empirical position is a relative anomaly position, **not** a probability of fraud.

## Ensemble and disagreement

Only calibrated detector outputs are combined. The ensemble method is explicit: weighted mean, maximum, or median. Weighted combinations require caller-supplied weights that sum exactly to one.

The ensemble retains the calibration interval and the spread between detector midpoint positions. Disagreement is evidence; it is not silently converted into a penalty or bonus.

## Run-to-run stability

Randomized detectors can produce different rankings across runs. When enabled, ProcureLens evaluates multiple explicitly named runs over the exact same row population.

Stability evidence includes tie-aware pairwise rank agreement and row-level ranges/median absolute deviation across run positions. No built-in threshold declares a run "stable enough."

## Review priority score

The client-compatible 0–100 score is a transparent linear presentation of the primary ensemble's calibrated midpoint position:

`review_priority_score = 100 × ensemble_midpoint_fraction`

It is **not** a 0–100 probability of fraud or wrongdoing. Lower and upper calibrated bounds, detector disagreement, evidence completeness, and stability remain separate fields.

## Review selection policy

Flagging is operational policy, not detector truth.

Supported policies include:

- explicit minimum review-priority score;
- explicit top-N review budget;
- explicit top-N boundary-tie handling.

No default `80+ = suspicious` rule exists. A top-N review budget does not redefine model scores.

## Explanations

Explanations are faithful evidence summaries. They can surface selected catalog-backed feature facts, review score/rank, selection reason, detector disagreement, stability evidence, and missing evidence.

ProcureLens does not currently claim that an individual feature "caused" a nonlinear unsupervised ensemble score. Without a validated attribution method, that statement would overstate what the model actually computed.

## Evaluation

When external labels exist, labeled evaluation keeps positive, negative, and unknown labels separate. Unknown/unreviewed procurement records are not silently treated as negatives.

Current labeled metrics include tie-aware ranking metrics and review-queue measurements. In unlabeled settings, stability, disagreement, data coverage, and deterministic synthetic integration are diagnostics rather than substitutes for ground truth.

## Export and publication

Validated review records can be serialized deterministically to JSON and CSV. CSV has an explicit programmatic mode and a spreadsheet-oriented safe-text mode; text sanitization does not alter valid negative numeric obligations.

A complete analysis run can be published as one staged directory bundle containing:

- deterministic run manifest;
- exact JSON/CSV payloads;
- publication metadata with hashes and byte counts.

The bundle is renamed into place only after payload hashes are verified and filesystem buffers are flushed. Existing bundles are not silently overwritten.

## Provenance

A deterministic run manifest links immutable input artifacts, stage executions, configuration fingerprints, generated artifacts, software/runtime identifiers, and final exports.

Two hashes are intentionally distinct:

- **recipe hash** — identifies inputs, stage implementations/configuration, environment, and requested output shapes while excluding generated output hashes;
- **evidence hash** — includes the generated artifact hashes as well.

This allows otherwise identical recipes with different outputs to expose nondeterminism rather than hide it.

## Validation status

The repository includes controlled contract tests, real cross-module feature integration tests, and a real synthetic full-pipeline integration test that executes feature construction, preprocessing, both detectors, calibration, ensemble, stability, review scoring/selection, explanations, exports, manifest creation, and atomic publication.

These tests validate software behavior on deterministic synthetic procurement records. They are **not** evidence that every USAspending data slice is free of source/schema surprises, nor are they a validation of fraud-detection accuracy.

A live USAspending end-to-end run is intentionally treated as a separate validation step and should not be claimed until it has actually been executed on a chosen real-data population.

## Responsible interpretation

ProcureLens should be used to prioritize human attention, compare statistical evidence, and make analytical reasoning auditable. High review priority can result from legitimate procurement structure, rare but lawful contracting circumstances, incomplete source data, or true unusual behavior. Human review and source-document context remain essential.
