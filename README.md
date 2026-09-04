# ProcureLens — Explainable Federal Procurement Anomaly Intelligence

[![CI](https://github.com/oyzitjfzj/ProcureLens-Explainable-Federal-Procurement-Anomaly-Intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/oyzitjfzj/ProcureLens-Explainable-Federal-Procurement-Anomaly-Intelligence/actions/workflows/ci.yml)

ProcureLens is a research-driven Python system for finding statistically unusual patterns in United States federal procurement transactions while preserving enough context, provenance, missingness, uncertainty, and model evidence for a human reviewer to understand what happened.

It is deliberately **not** a fraud detector. A high ProcureLens score means **higher anomaly-review priority relative to the fitted reference population**. It does not mean a vendor committed fraud, a procurement was unlawful, or a numerical score is a probability of wrongdoing.

## Why this project exists

A minimal procurement anomaly prototype can load award rows, fit an Isolation Forest, flag a percentage of records, and print a few reasons. That is easy to build and easy to misuse.

ProcureLens takes the harder path:

- source integrity before analysis;
- a source-neutral canonical transaction contract;
- explicit distinction between transaction obligations and award totals;
- leave-one-out, hierarchical peer context instead of global dollar thresholds;
- vendor **new-award** frequency instead of raw transaction frequency;
- competition process separated from offer outcome;
- modification activity separated from unsupported claims about justification;
- explicit missingness instead of invented zeros;
- frozen reference/training populations;
- train-only preprocessing, detector fitting, and score calibration;
- heterogeneous unsupervised detectors rather than one model treated as truth;
- run-to-run stability and detector disagreement retained as evidence;
- review policy kept separate from anomaly strength;
- explanations that report evidence without inventing causal feature attribution;
- deterministic JSON/CSV exports, provenance manifests, and atomic run publication.

## Analytical pipeline

```text
USAspending artifact
      │
      ▼
verified download / schema / loader
      │
      ▼
canonical ProcurementTransaction
      │
      ├── data-quality profile + readiness gate
      │
      ▼
frozen contextual reference populations
      │
      ├── amount context
      ├── observed vendor new-award frequency
      ├── competition process/outcome context
      └── award-change behavior
      │
      ▼
explicit candidate-feature catalog
      │
      ▼
explicit feature selection + train-only preprocessing
      │
      ▼
┌───────────────────────┬────────────────────────┐
│ Isolation Forest      │ frozen empirical-tail  │
└───────────┬───────────┴────────────┬───────────┘
            │ train-fitted calibration│
            └────────────┬────────────┘
                         ▼
                 calibrated ensemble
                         │
              stability + disagreement
                         │
                         ▼
              0–100 review priority
                         │
             explicit review policy
                         │
                         ▼
          faithful explanation + export
                         │
                         ▼
           manifest + atomic run bundle
```

## Signal families

### Contextual award/transaction amount

ProcureLens does not compare every federal contract to one global dollar distribution. It resolves transactions through hierarchical agency/category/time peer groups, excludes the indexed target from its own reference sample, and emits robust empirical-position, MAD, IQR, magnitude, direction, and support evidence.

### Vendor new-award frequency

Repeated modifications of one award do not count as repeated wins. Vendor context is built from observed base/new-award actions, with entity and ultimate-parent scopes kept separate. The system measures observed winner shares, peer positions, concentration context, and identity coverage without pretending the data contains every potential bidder in the market.

### Competition evidence

A competitive process can result in one offer. ProcureLens therefore keeps reported process/status, solicitation procedure, number of offers, other-than-full-and-open authority, missing evidence, and field conflicts separate. Contextual prevalence is calculated over comparable base awards rather than duplicated modification rows.

### Award-change behavior

Modifications can be ordinary and lawful. ProcureLens records counts, positive changes, deobligations, zero-dollar actions, gross/net activity, repeated modification numbers, follow-up exposure, and base-obligation-normalized measurements. It does not infer that a modification was unjustified from these fields alone.

## Modeling architecture

### Explicit feature contract

Every candidate feature has a globally unique name, source family, definition description, and definition fingerprint. A detector plan must explicitly select and order columns; adding a new descriptive feature cannot silently change an existing model input.

### Preprocessing

Missing-value handling, missingness indicators, and numeric transforms are explicit per selected feature. Fitted state is learned from training rows only and reused unchanged for scoring rows.

### Two different detector views

ProcureLens currently combines:

1. **scikit-learn Isolation Forest** with explicit configuration, seeded reproducibility, input-conversion diagnostics, and model/tree fingerprints;
2. **ProcureLens frozen empirical-tail detector**, a training-ECDF-based tail method whose scoring distribution is never refitted on inference/test rows.

The empirical-tail implementation is paper-inspired and intentionally not advertised as an exact clone of a third-party ECOD/COPOD implementation.

### Calibration before combination

Raw detector scores have different scales and directions, so they are not directly averaged. Calibration is fitted on training scores and retains tie-aware empirical position plus robust tail-distance evidence. The ensemble then combines calibrated positions with an explicit weighted-mean, maximum, or median specification.

### Stability and uncertainty

Named repeated detector runs can be compared over the exact same scoring population. ProcureLens preserves pairwise rank agreement, per-row position span, median absolute deviation, calibration interval, and detector disagreement. It does not hide them inside one confidence number.

## What the 0–100 score means

The client-compatible score is a presentation layer:

```text
review_priority_score = 100 × calibrated_primary_ensemble_midpoint
```

So a score of `83` means the row is at a relatively high calibrated anomaly position for that fitted analysis. It does **not** mean `83% probability of fraud`.

The CSV compatibility field `risk_score_0_100` carries the same documented review-priority semantics.

## Review selection

Flagging is explicit operational policy. A run can specify either:

- a minimum review-priority score; or
- a top-N human review budget.

Top-N boundary ties require an explicit policy: include all boundary ties, exclude them, or use a deterministic identity tie-break to satisfy an exact budget. There is no hidden `score >= 80` rule.

## Explanations

Explanations bind the exact feature row, ensemble/review evidence, score artifact, review-selection decision, and selected catalog facts.

They describe **observed evidence**, not unsupported causal attribution. ProcureLens does not currently claim that a particular feature "caused 37% of the anomaly score" in a heterogeneous nonlinear unsupervised ensemble.

## Deterministic outputs

The export layer supports:

- deterministic JSON;
- deterministic CSV;
- explicit spreadsheet-safe text mode for CSV;
- provenance hashes on records and payloads;
- client-facing review score, flag, rank interval, reasons, source lineage, and evidence facts.

A complete run can be atomically published as a directory bundle:

```text
<bundle>/
├── manifest.json
├── publication.json
└── exports/
    ├── export_00.json
    └── export_01.csv
```

Existing bundles are not silently overwritten.

## Provenance

Every top-level run produces a deterministic provenance graph describing inputs, exact plans, stage implementations, generated artifacts, runtime/software identifiers, and final exports.

Two hashes serve different purposes:

- **recipe SHA-256** — input snapshot + environment + stage/configuration recipe, excluding generated-output hashes;
- **evidence SHA-256** — complete observed run including generated artifact hashes.

If the same recipe unexpectedly generates different artifacts, the difference remains visible.

## Start here — first-time setup and a complete local check

This section is the practical path for somebody who has just found the repository and wants to get it onto a computer, verify the installation, understand the supported entry point, and reach a known-good full-pipeline run before touching real federal data.

### 1. Prerequisites

You need:

- Git;
- Python 3.11 or newer;
- a normal CPython environment in which NumPy and scikit-learn can be installed;
- enough memory for the reference population and detector configuration you choose.

The CI workflow currently exercises Python 3.11, 3.12, 3.13, and 3.14 on Ubuntu. Other platforms may work, but a platform on which the compiled NumPy/scikit-learn stack cannot be installed is not a supported substitute for a successful test run.

### 2. Clone ProcureLens onto your computer

```bash
git clone https://github.com/oyzitjfzj/ProcureLens-Explainable-Federal-Procurement-Anomaly-Intelligence.git
cd ProcureLens-Explainable-Federal-Procurement-Anomaly-Intelligence
```

If you already cloned the project earlier and only want the latest verified `main` branch:

```bash
git switch main
git pull --ff-only origin main
```

`--ff-only` is intentional: it refuses to silently create a merge commit when your local history has diverged.

Before installing anything, confirm your Python version:

```bash
python --version
```

On systems where the command is named `python3` instead:

```bash
python3 --version
```

On Windows, the Python launcher can also be checked with:

```powershell
py --version
```

The reported version must be at least Python 3.11.

### 3. Create an isolated virtual environment

Linux, macOS, and most shell environments:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If your system already maps `python` to the correct Python 3.11+ interpreter, this is equivalent:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
```

After activation, `python --version` should still report Python 3.11 or newer.

### 4. Install the package and test dependencies

From the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

The editable install means source-code changes in this checkout are picked up without repeatedly copying the package into `site-packages`.

Verify that the important runtime imports work:

```bash
python -c "import numpy, sklearn, procurelens.pipeline.run; print('ProcureLens import: OK'); print('NumPy:', numpy.__version__); print('scikit-learn:', sklearn.__version__)"
```

If that command fails, fix the Python/dependency environment before attempting a real analysis.

### 5. Run the tests before real data

Fast contract check:

```bash
python -m pytest tests/test_pipeline_run_contract.py -q
```

Cross-module feature/reference integration:

```bash
python -m pytest tests/test_feature_pipeline_integration.py -q
```

Full synthetic ProcureLens pipeline, including real preprocessing, both detector families, calibration, ensemble, stability, review scoring/selection, explanation, exports, provenance, and publication:

```bash
python -m pytest tests/test_full_pipeline_integration.py -q
```

Then run the complete committed suite:

```bash
python -m pytest
```

These tests do not require live USAspending network access. They are the safest first proof that the code, package, dependencies, and cross-module contracts work together in your environment.

### 6. Understand the current entry point

The supported high-level interface is currently the **Python API**. The repository does not advertise a fake or incomplete `procurelens run ...` CLI command.

The normal flow is:

```text
validated source data
      ↓
ProcurementTransaction objects
      ↓
explicit FeatureBuildPlan
      ↓
explicit ModelReviewPlan
      ↓
run_procurelens_analysis(...)
      ↓
review evidence + score + explanations + serialized exports
      ↓
publish_analysis_run(...)
      ↓
atomic run directory
```

A fully executable configuration is intentionally kept in `tests/test_full_pipeline_integration.py`. That test is the best starting template because it is continuously checked against the real code instead of being a stale pseudo-example in documentation.

### 7. Historical analysis versus future scoring

For a first historical analysis, the same frozen transaction population can be used as the reference/training and scoring population. The contextual modules use their designed leave-one-out behavior so an indexed target is not simply treated as its own peer.

For a forward-looking or production-style run, keep the roles separate:

```text
historical/reference transactions  ──► build and freeze references/models
new scoring transactions           ──► score against the frozen state
```

Do **not** append new scoring rows to the reference population immediately before scoring them. Doing that changes the baseline using the data that is being evaluated and weakens the train/reference separation ProcureLens is designed to preserve.

### 8. Run and publish through the Python API

At the highest level:

```python
from procurelens.pipeline.run import run_procurelens_analysis
from procurelens.runtime.publication import publish_analysis_run

run = run_procurelens_analysis(
    reference_transactions=reference_transactions,
    scoring_transactions=scoring_transactions,
    feature_plan=feature_plan,
    model_plan=model_plan,
    run_name="federal-contract-review",
    source_revision="your-source-snapshot-id",
)

receipt = publish_analysis_run(
    run,
    "runs",
    bundle_name="federal-contract-review",
)

print("run evidence:", run.evidence_sha256)
print("published:", receipt)
```

`reference_transactions` and `scoring_transactions` are canonical `ProcurementTransaction` objects. `feature_plan` and `model_plan` are explicit configuration artifacts; they are not silently invented by the runner.

### 9. What to inspect after a published run

A successful publication creates a new bundle rather than silently replacing an existing one. Inspect:

- `manifest.json` — the provenance graph and exact run recipe/evidence chain;
- `publication.json` — publication receipt/integrity information;
- `exports/export_00.json` — deterministic machine-readable records when JSON export is configured;
- `exports/export_01.csv` — review-friendly tabular output when CSV export is configured.

Important output concepts include the review-priority score, flag/selection decision, rank interval, detector disagreement, feature completeness, stability evidence when available, explanation facts, source lineage, and evidence fingerprints.

### 10. Updating an existing checkout safely

From the repository root:

```bash
git status
git switch main
git pull --ff-only origin main
python -m pip install -e ".[test]"
python -m pytest
```

If `git status` shows local changes, understand or save those changes before pulling. Do not use destructive reset commands merely to make Git stop complaining.

### 11. Common setup problems

**`python` or `python3` is not found** — install a supported CPython version and reopen the shell.

**Python is older than 3.11** — create the virtual environment with a newer interpreter; do not try to work around the package requirement by editing `pyproject.toml`.

**`ModuleNotFoundError: procurelens`** — make sure you are in the repository root, the virtual environment is active, and `python -m pip install -e ".[test]"` completed successfully.

**NumPy or scikit-learn installation fails** — use a platform/interpreter combination for which those packages provide a working build. An Android/Termux checkout can still be useful for reading/editing Git files, but full model execution depends on a working compiled scientific-Python stack in that environment.

**A full integration test fails** — do not continue to real data and assume the failure is harmless. Capture the Python version, package versions, Git commit, and failing traceback first.

Useful diagnostics:

```bash
python --version
python -c "import numpy, sklearn; print('NumPy', numpy.__version__); print('scikit-learn', sklearn.__version__)"
git rev-parse HEAD
python -m pytest -q
```

**A published bundle name already exists** — choose a different bundle name or intentionally archive/remove the old bundle outside ProcureLens. Existing result bundles are not silently overwritten.

**A feature is missing** — missing evidence is often an intentional first-class state, not automatically a software failure. Check the exported reason/provenance before replacing it with zero or another invented value.

## Real USAspending data — recommended operating sequence

ProcureLens is deliberately not designed as “drop any CSV into Isolation Forest.” A real run should preserve the same source, quality, context, and train/reference boundaries exercised by the internal architecture.

A recommended sequence is:

1. **Define the analytical population deliberately.** Choose the time window, award types, agencies/categories, and other filters appropriate to the question you want to study.
2. **Acquire the USAspending artifact.** Use the source/control-plane and artifact components, or another controlled process that preserves the exact downloaded artifact and source revision.
3. **Materialize the full artifact before analysis.** Do not model a partially downloaded ZIP/CSV.
4. **Read through the explicit USAspending schema layer.** Avoid ad-hoc fuzzy column mapping or silently repurposing similarly named fields.
5. **Load canonical `ProcurementTransaction` objects.** This is where source-specific data becomes the source-neutral domain contract used by the analytical system.
6. **Inspect quality/readiness evidence.** Missing competition fields, identity coverage, signed obligations, dates, schema/source information, and other coverage facts should be visible before modeling.
7. **Freeze the reference population.** Amount, vendor, competition, and award-change context must be built from the intended reference data, not silently updated with the scoring rows.
8. **Build candidate features.** Each signal family preserves its own support, missingness, and evidence provenance.
9. **Fit preprocessing, detectors, and calibration on training/reference rows only.** Scoring rows reuse fitted state.
10. **Review outputs as anomaly evidence.** Treat the 0–100 value as review priority, not fraud probability.
11. **Publish the run bundle.** Preserve the manifest, publication receipt, JSON/CSV exports, and source revision so the run can be audited later.

For the first live validation, start with a deliberately bounded USAspending population rather than immediately attempting the entire federal universe. Confirm row counts, data-quality coverage, feature availability, runtime/memory behavior, output integrity, and investigator-readable explanations before scaling the population upward.

## Installation

ProcureLens requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Runtime dependencies are intentionally small: NumPy and scikit-learn. Test tooling is optional.

## Running the tests

```bash
python -m pytest
```

The test suite includes:

- top-level orchestration contract tests;
- real cross-module feature/reference integration;
- a real synthetic full-pipeline run through preprocessing, both detectors, calibration, ensemble, stability, review scoring/selection, explanation, export, provenance manifest, and atomic filesystem publication.

GitHub Actions runs the package and test suite across the supported CI Python matrix.

## Python API

The highest-level analysis entry point is:

```python
from procurelens.pipeline.run import run_procurelens_analysis

run = run_procurelens_analysis(
    reference_transactions=reference_transactions,
    scoring_transactions=scoring_transactions,
    feature_plan=feature_plan,
    model_plan=model_plan,
    run_name="federal-contract-review",
    source_revision="your-source-snapshot-id",
)
```

The two plans are intentionally explicit. ProcureLens does not hide peer support, identity scope, feature selection, preprocessing, detector seeds, ensemble weights, review threshold/budget, explanation fields, or export format behind undocumented defaults.

For a fully executable configuration pattern, see `tests/test_full_pipeline_integration.py`. It uses synthetic records so the complete pipeline can be validated without network access.

To publish a completed run:

```python
from procurelens.runtime.publication import publish_analysis_run

receipt = publish_analysis_run(
    run,
    "runs",
    bundle_name="federal-contract-review",
)
```

## USAspending ingestion

The `procurelens.sources.usaspending` package provides separate components for:

- request/count/download control flow;
- safe artifact materialization and resume behavior;
- ZIP/tabular reading and provenance;
- explicit USAspending download-schema mapping;
- canonical transaction loading with duplicate/conflict handling.

This keeps source-specific column knowledge out of the analytical domain model.

A real production analysis should choose its USAspending query/population deliberately, fully materialize and validate the artifact, build a frozen reference population, and only then run the analysis pipeline.

## Repository layout

```text
src/procurelens/
├── domain/          # canonical transaction and lineage contract
├── sources/         # USAspending acquisition/reading/loading
├── quality/         # data profile and analysis-readiness gates
├── statistics/      # robust distribution primitives
├── features/        # peer contexts and candidate evidence families
├── model/           # feature selection, preprocessing, matrices
├── detectors/       # detector, calibration, ensemble, stability
├── review/          # review evidence, score, policy, explanation
├── evaluation/      # labeled evaluation with unknown-label handling
├── export/          # typed export records and serialization
├── runtime/         # provenance manifest and atomic publication
└── pipeline/        # explicit feature/model/top-level orchestration

tests/               # contract and integration tests
docs/                # methodology and data semantics
```

## Design principles

- **Evidence before verdict.**
- **Context before threshold.**
- **Training/reference data never silently absorbs scoring data.**
- **Missing is not zero.**
- **A modification is not inherently suspicious.**
- **One offer is not automatically a noncompetitive process.**
- **Observed winners are not the full universe of potential competitors.**
- **Review budget is not model truth.**
- **Anomaly priority is not fraud probability.**
- **Every important transformation should be reproducible and attributable to an explicit artifact.**

## Validation status

**Implemented and CI-tested:** canonical contracts, USAspending source components, quality/readiness layers, all contextual feature families, model input/preprocessing, Isolation Forest, frozen empirical-tail detection, calibration, ensemble, stability, review scoring/policy/explanation, labeled evaluation, deterministic JSON/CSV, provenance manifests, top-level orchestration, and atomic run publication.

**Validated in repository tests:** synthetic cross-module and synthetic full-pipeline execution with real ProcureLens implementations.

**Not yet claimed:** a completed live USAspending end-to-end analysis or real-world fraud-detection accuracy. Those require a deliberately chosen real dataset and external review/ground truth. The software is designed so that validation can be performed without changing score semantics or silently weakening provenance.

## Documentation

- [`docs/methodology.md`](docs/methodology.md) — analytical/modeling methodology and interpretation.
- [`docs/data_contract.md`](docs/data_contract.md) — canonical procurement semantics and missingness rules.

## Responsible use

ProcureLens should support human investigation, audit prioritization, and analytical transparency. It should not be used as the sole basis for accusing, penalizing, excluding, or publicly labeling a vendor or official. Procurement context, source documents, legal authorities, and human review remain necessary.
