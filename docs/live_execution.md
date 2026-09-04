# Running ProcureLens from live USAspending data

This document describes the supported **source-to-analysis** Python API for a bounded real-USAspending run.

It does not define an analytical population for you. The filters, quality requirements, peer-support rules, model inputs, detector configuration, ensemble policy, and review policy remain explicit analysis decisions.

ProcureLens does not automatically pick a time range, agency, category, or row count merely to obtain interesting anomalies.

## Supported live bridge

The live orchestration boundary is:

```python
from procurelens.pipeline.usaspending_live import (
    LiveUSAspendingPlan,
    prepare_live_usaspending_dataset,
    run_live_usaspending_analysis,
)
```

The source-aware publication boundary is:

```python
from procurelens.runtime.publication import publish_live_usaspending_run
```

The orchestration deliberately reuses the existing source components rather than bypassing them:

```text
explicit filters
    ↓
USAspending count-only preflight
    ↓
asynchronous download request + status polling
    ↓
verified ZIP materialization
    ↓
explicit schema plan + complete canonical loading
    ↓
quality profile + caller-supplied readiness gate
    ↓
frozen-reference ProcureLens analysis
    ↓
source-aware atomic publication
```

## Why the count preflight is mandatory

Before creating a download job, `prepare_live_usaspending_dataset(...)` calls the USAspending transaction-count endpoint for the exact filter payload.

If USAspending reports that the requested population exceeds its current download limit, ProcureLens raises `USAspendingPopulationLimitError` **before creating the download job**.

The runner does not silently apply a smaller `limit` and pretend that the truncated rows represent the requested population.

If a population is too large, narrow or deliberately partition the analytical question and treat the resulting plan as a different recipe.

## Build the plan explicitly

A live plan binds four different kinds of policy:

1. source population filters;
2. data-quality readiness requirements;
3. contextual feature/reference policy;
4. model/review/export policy.

Example shape:

```python
from procurelens.pipeline.usaspending_live import LiveUSAspendingPlan

live_plan = LiveUSAspendingPlan(
    name="bounded-federal-contract-review",
    description="A deliberately bounded real-USAspending validation population.",
    filters=explicit_filters,
    quality_gate=quality_gate_spec,
    feature_plan=feature_build_plan,
    model_plan=model_review_plan,
)
```

`explicit_filters`, `quality_gate_spec`, `feature_build_plan`, and `model_review_plan` are intentionally not invented by this example.

The plan deep-freezes the filter payload and fingerprints its effective policy. Mutating the original Python dictionary later does not silently mutate the stored run plan.

## Optional operational controls

The live plan can also explicitly set:

- USAspending download columns;
- download format (`csv`, `tsv`, or `pstxt`);
- asynchronous polling policy;
- exact archive members to load;
- explicit download-schema profile name;
- whether a `DEGRADED` quality-gate result is permitted to continue.

These are operational/source-readiness choices. They do not change anomaly strength by hidden policy.

## Prepare without modeling first

For the first real-data check, it is often useful to stop after source preparation:

```python
prepared = prepare_live_usaspending_dataset(
    live_plan,
    "work/live-validation",
)

print("rows:", prepared.transaction_count)
print("artifact SHA-256:", prepared.artifact.sha256_hex)
print("quality status:", prepared.quality_gate.status.value)
print("prepared evidence:", prepared.evidence_sha256)
```

A prepared dataset preserves evidence for:

- count preflight and request fingerprint;
- download job and terminal status;
- exact verified artifact bytes;
- schema/load plan;
- complete loader report;
- quality profile;
- quality-gate report;
- canonical transaction-population fingerprint.

The serialized preparation evidence stores the population count and fingerprint, not a second raw copy of every canonical transaction.

## Quality-gate behavior is fail-closed

A `BLOCKED` gate never reaches the model runner.

A `DEGRADED` gate reaches the model runner only if the live plan explicitly sets:

```python
allow_degraded_quality=True
```

This opt-in does not change the failed warning evidence. It only records that the caller deliberately accepted a degraded-but-not-blocked dataset for that run.

A `READY` gate proceeds normally.

## Run the full analysis

After the bounded source preparation is acceptable:

```python
live_run = run_live_usaspending_analysis(
    live_plan,
    "work/live-validation",
)
```

For this first bounded historical validation, the prepared live population is supplied as both the reference/training and scoring population. Existing ProcureLens leave-one-out contracts prevent an indexed target from simply becoming its own peer where those contextual families require exclusion.

The underlying source artifact SHA-256 is passed into the analysis manifest as `source_revision`.

For later forward-looking scoring, preserve a genuinely frozen historical/reference state rather than appending new rows immediately before scoring them.

## Publish the live run with its source evidence

Do not discard the acquisition/loading/quality chain when publishing a live analysis.

Use:

```python
from procurelens.runtime.publication import publish_live_usaspending_run

receipt = publish_live_usaspending_run(
    live_run,
    "runs",
    bundle_name="bounded-federal-contract-review",
)
```

A source-aware live bundle contains:

```text
<bundle>/
├── manifest.json
├── publication.json
├── live_run.json
├── source/
│   ├── live_plan.json
│   └── prepared_dataset.json
└── exports/
    ├── export_00.json
    └── export_01.csv
```

The exact export count and extensions depend on the explicit `ModelReviewPlan`.

`publication.json` distinguishes the top-level live-run evidence fingerprint from the underlying analysis-run evidence fingerprint.

The publication API stages the complete bundle on the destination filesystem, validates payload hashes, flushes file and directory buffers, and atomically renames the completed directory into place. Existing bundles are not silently overwritten.

## What counts as a successful first real run

A first live run is technically acceptable only after all of the following are true:

- committed tests pass in the execution environment;
- the exact filter payload and count preflight are preserved;
- the requested population is within the server-supported download limit;
- the download job reaches `finished`;
- the complete ZIP is verified and fingerprinted;
- schema planning and canonical loading fully complete;
- duplicate/conflict/quarantine outcomes are reviewed;
- the explicit quality gate allows analysis;
- feature construction completes under the chosen support policies;
- preprocessing and detector/calibration fitting use the intended training/reference state only;
- JSON/CSV exports contain the intended scoring population;
- source-aware publication succeeds;
- a deterministic rerun is checked before scaling toward a larger analytical population.

This validates the source/software path. It does **not** prove fraud-detection accuracy.

## Current USAspending control-plane endpoints

ProcureLens uses the USAspending V2 API control plane currently documented at:

- <https://api.usaspending.gov/docs/endpoints>
- <https://api.usaspending.gov/docs/>

The relevant documented flow is the transaction count preflight, asynchronous filtered download creation, and returned download-status polling endpoint. ProcureLens keeps large artifact transfer separate from those control-plane requests.

## Relationship to the broader validation protocol

This document describes the concrete live orchestration API. The complete acceptance ladder, scale-up rules, deterministic-rerun expectations, and real-world labeled-evaluation boundaries remain documented in [`live_validation.md`](live_validation.md).
