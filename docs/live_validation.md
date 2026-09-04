# Live USAspending validation protocol

This document defines the evidence required before ProcureLens can claim that a real USAspending analysis has completed successfully.

It is a validation protocol, not a source of anomaly thresholds. Population size, peer-support requirements, feature selection, preprocessing, detector seeds, ensemble policy, and review budget remain explicit run configuration.

ProcureLens continues to make the following distinction:

> A high review-priority score means that a record is statistically unusual relative to the fitted analysis. It is not a probability of fraud, illegality, collusion, waste, or misconduct.

## Why a separate live-data protocol exists

Synthetic tests prove software contracts and cross-module behavior. They cannot prove that a current USAspending download has the same field coverage, schema behavior, transaction mix, recipient identity coverage, competition reporting, modification history, or operational size characteristics as the synthetic fixtures.

A live validation therefore needs to preserve four different kinds of evidence:

1. **source acquisition evidence** — what population was requested and what USAspending returned;
2. **artifact integrity evidence** — which exact bytes were analyzed;
3. **data-readiness evidence** — whether the canonical population supports the intended analysis;
4. **analysis evidence** — which exact plans, fitted state, detector artifacts, scores, explanations, and exports were produced.

Do not collapse these into a single “run succeeded” boolean.

## Official USAspending interfaces used by the source layer

The current USAspending V2 API documents public endpoints without an authorization requirement. ProcureLens uses the API as a control plane and keeps large artifact transfer separate.

Relevant official endpoints include:

- `POST /api/v2/download/count/` — preflight the number of transactions for a filter set;
- `POST /api/v2/download/search/` — create an asynchronous filtered download job;
- the returned status URL — poll the asynchronous job until it finishes or fails.

ProcureLens wraps these behaviors in `procurelens.sources.usaspending.client.USAspendingClient` and records deterministic request fingerprints. Large ZIP/CSV transfer, hashing, resume behavior, archive inspection, schema mapping, and canonical loading are separate source-layer responsibilities.

Official API documentation:

- <https://api.usaspending.gov/docs/endpoints>
- <https://api.usaspending.gov/docs/>

## Validation ladder

A real-data validation should advance through the following stages. Do not skip a failed stage merely because a later stage can technically be forced to run.

### Stage 0 — repository baseline

Before network acquisition:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The committed suite must pass in the intended execution environment.

The source-to-analysis integration test is especially relevant:

```bash
python -m pytest tests/test_usaspending_to_analysis_integration.py -q
```

That test proves the installed package can take a verified USAspending-shaped ZIP through the real loader, quality layer, feature pipeline, detector stack, review layer, and exports without network dependence.

### Stage 1 — count-only live preflight

Define the analytical population before looking at anomaly results. The filter should be justified by the analytical question, not chosen because it creates interesting outliers.

Use:

```python
from procurelens.sources.usaspending.client import USAspendingClient

client = USAspendingClient()
count = client.count_transactions(filters, spending_level="transactions")
```

Record at minimum:

- the exact filter object;
- `count.request_fingerprint.sha256_hex`;
- `calculated_transaction_count`;
- the server-reported transaction/download limits;
- `transaction_rows_gt_limit` and `rows_gt_limit`;
- any server messages.

If the requested population exceeds the current server limit, narrow or partition the analytical population deliberately. Do not truncate silently and call the result the original population.

### Stage 2 — bounded live acquisition

For the first live execution, prefer a deliberately bounded population that is large enough to exercise peer groups and detector behavior but small enough that failures are cheap to diagnose.

The original project brief targeted roughly 10,000–25,000 contract transactions for the intended analysis. That range is an operational target, not a universal statistical requirement. A smaller smoke population can be used first; the intended target-scale run should follow only after the smoke run is clean.

Start the download with an explicit request and preserve its fingerprint:

```python
job = client.start_search_download(
    filters,
    spending_levels=("transactions",),
    file_format="csv",
)
status = client.wait_for_download(job)
```

Required checks:

- terminal status must be `finished`;
- the job request fingerprint must be preserved;
- returned row/size metadata should be recorded when present;
- the artifact URL must be the one associated with the completed job;
- the complete artifact must be materialized before analysis begins.

Do not treat a partially transferred ZIP as a usable dataset.

### Stage 3 — artifact integrity and archive scan

Materialize the download through the artifact layer so the resulting `ArtifactReceipt` records the exact bytes and archive structure.

Preserve:

- final file name and path;
- byte size;
- SHA-256;
- source/final URL;
- request fingerprint;
- ETag/Last-Modified when supplied;
- archive member list;
- compressed/uncompressed sizes;
- retrieval timestamp.

The reader should verify the receipt before parsing. A same-size file whose bytes have changed must not be accepted merely because the path and size still look correct.

### Stage 4 — explicit schema and canonical loading

Use the USAspending reader/schema/loader path rather than ad-hoc `csv.DictReader` field guessing.

After loading, preserve the complete loader report. Review at minimum:

- rows seen;
- canonical transactions emitted;
- exact duplicates dropped;
- conflicting duplicate count;
- quarantined row count;
- schema/profile name;
- additive/unrecognized schema headers;
- completion state.

A loader report with conflicts or quarantines is not automatically unusable, but it requires an explicit decision before modeling. Do not silently convert malformed rows into apparently valid zero-valued evidence.

### Stage 5 — quality profile and readiness gate

Profile the canonical population before fitting feature references or models:

```python
from procurelens.quality.profile import profile_transactions

profile = profile_transactions(transactions)
```

Then evaluate an explicit `QualityGateSpec` for the intended analysis.

Useful readiness dimensions include:

- population size;
- vendor-identity coverage;
- procurement-category coverage;
- awarding-agency coverage;
- competition extent/offers coverage;
- solicitation-procedure coverage;
- action-obligation coverage;
- source/schema cardinality;
- missing-analysis-context share.

The gate thresholds are part of the analysis plan. ProcureLens deliberately does not provide a hidden universal rule such as “80% coverage is good enough.”

Interpretation:

- `READY` — every configured requirement passed;
- `DEGRADED` — only configured warnings failed; proceed only with those limitations visible;
- `BLOCKED` — a blocking requirement failed; do not run that analysis configuration unchanged.

### Stage 6 — freeze reference and scoring roles

Before feature construction, decide whether the run is historical or forward-looking.

#### Historical validation

The same bounded population may be used for reference/training and scoring. ProcureLens uses designed leave-one-out behavior where applicable so an indexed target is not simply its own peer.

#### Forward-looking validation

Historical/reference rows and new scoring rows must remain separate:

```text
historical/reference population  ──► frozen references + fitted state
new scoring population           ──► score against frozen state
```

Do not append new rows to the reference population immediately before scoring them. That changes the baseline using the data being evaluated.

### Stage 7 — feature availability audit

Build the frozen feature reference bundle and candidate feature rows before fitting the detector stack.

Review:

- selected peer levels and fallback frequency;
- peer counts/support;
- vendor identity coverage;
- competition process/offers/procedure coverage;
- award-change observation coverage;
- missing candidate features and their explicit reasons;
- feature catalog and definition fingerprints.

Do not solve a missingness problem by replacing unavailable evidence with zero unless the preprocessing plan explicitly and defensibly defines that behavior.

### Stage 8 — model/review run

Run the explicit `FeatureBuildPlan` and `ModelReviewPlan` through:

```python
from procurelens.pipeline.run import run_procurelens_analysis

run = run_procurelens_analysis(
    reference_transactions=reference_transactions,
    scoring_transactions=scoring_transactions,
    feature_plan=feature_plan,
    model_plan=model_plan,
    run_name="bounded-live-validation",
    source_revision=artifact_receipt.sha256_hex,
)
```

Check that:

- preprocessing was fitted only on training/reference feature rows;
- detector models were fitted only on the training matrix;
- score calibrations were fitted only on training scores;
- scoring rows did not mutate fitted state;
- all detector members scored the exact intended row population;
- stability evidence covers the exact configured repeated runs;
- review policy is the explicitly configured policy;
- score semantics remain review-priority semantics, not probability semantics.

### Stage 9 — deterministic rerun

Before scaling up, repeat the bounded run with the exact same:

- source artifact;
- software environment;
- feature plan;
- model plan;
- detector seeds;
- reference/scoring population roles.

The manifest recipe identity should remain stable. With the same deterministic inputs and runtime, the generated evidence should also remain stable. If the recipe is unchanged but generated artifact hashes differ, investigate nondeterminism before scaling the run.

### Stage 10 — publish and preserve

Publish the successful run atomically:

```python
from procurelens.runtime.publication import publish_analysis_run

receipt = publish_analysis_run(
    run,
    "runs",
    bundle_name="bounded-live-validation",
)
```

Preserve together:

- original downloaded artifact;
- `ArtifactReceipt` evidence;
- filter/request fingerprint;
- loader report;
- quality profile;
- quality-gate report;
- exact feature/model plans;
- `manifest.json`;
- `publication.json`;
- JSON export;
- CSV export;
- environment/package versions;
- Git commit SHA.

A later reviewer should be able to determine exactly what data and software produced a reported anomaly without reconstructing undocumented state from memory.

## First live-run acceptance checklist

A bounded live validation is considered technically successful only when all applicable items below are satisfied:

- [ ] committed repository tests pass in the run environment;
- [ ] preflight count and request fingerprint are preserved;
- [ ] download job reaches `finished`;
- [ ] complete artifact SHA-256 is preserved;
- [ ] archive verification succeeds before parsing;
- [ ] loader finishes completely;
- [ ] duplicate/conflict/quarantine outcomes are explicitly reviewed;
- [ ] quality gate is not `BLOCKED`;
- [ ] feature construction completes without hidden fallback policy;
- [ ] selected model inputs satisfy the explicit preprocessing contract;
- [ ] both detector families produce valid score batches;
- [ ] calibration, ensemble, and stability evidence are internally consistent;
- [ ] review scores stay within documented 0–100 semantics;
- [ ] JSON/CSV exports contain the expected scoring population;
- [ ] manifest links the intended source revision and generated artifacts;
- [ ] deterministic rerun is consistent before scale-up.

Passing this checklist validates the software/data path. It does **not** validate fraud-detection accuracy.

## Scale-up sequence

After a clean bounded run:

1. increase toward the intended 10,000–25,000 transaction analysis range when the analytical population supports it;
2. re-run the count preflight rather than assuming previous server limits or row counts;
3. observe memory/runtime behavior and configured resource budgets;
4. inspect whether peer-group support and feature availability improve or degrade with the new population;
5. preserve the new artifact and plans as a distinct run rather than overwriting the bounded validation;
6. compare stability, detector disagreement, review-queue composition, and explanation quality across runs.

If scale-up requires changing filters, feature plans, model plans, or review policy, treat it as a new recipe. Do not compare evidence hashes as though only population size changed.

## Real-world analytical validation

Software validation and real-world analytical validation are separate.

To evaluate whether ProcureLens helps investigators prioritize meaningful cases, use externally reviewed labels or adjudicated outcomes when available. Unknown/unreviewed records must remain unknown; do not convert them into negatives merely to obtain a convenient accuracy number.

Appropriate later evaluation can include:

- AUROC and average precision over genuinely labeled rows;
- precision/recall at the actual human-review budget;
- stability under reasonable detector seeds/configurations;
- analyst assessment of explanation usefulness;
- false-positive analysis by agency/category/vendor context;
- investigation of systematic missingness or reporting bias.

No live validation should be described as evidence that ProcureLens “detects fraud” unless a separate, defensible labeled evaluation supports such a claim—and even then, the system should remain a human-review prioritization tool rather than an automatic accusation mechanism.
