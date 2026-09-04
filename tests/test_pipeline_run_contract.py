from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import pytest

import procurelens.pipeline.run as run_module


@dataclass(frozen=True)
class _Transaction:
    transaction_id: str
    award_id: str
    payload: int


class _FeaturePlan:
    def __init__(self, catalog_sha: str = "a" * 64) -> None:
        self.feature_catalog_sha256 = catalog_sha
        self.name = "feature-plan"
        self.sha256_hex = "1" * 64


class _SelectionSpec:
    def __init__(self, catalog_sha: str) -> None:
        self.catalog_sha256 = catalog_sha


class _FeatureSelection:
    def __init__(self, catalog_sha: str) -> None:
        self.spec = _SelectionSpec(catalog_sha)


class _ModelPlan:
    def __init__(self, catalog_sha: str = "a" * 64) -> None:
        self.feature_selection = _FeatureSelection(catalog_sha)
        self.name = "model-plan"
        self.sha256_hex = "2" * 64


@dataclass(frozen=True)
class _FeatureRow:
    transaction_id: str
    award_id: str
    evidence_sha256: str


class _References:
    def __init__(self, plan: _FeaturePlan, count: int) -> None:
        self.plan = plan
        self.reference_population_sha256 = "3" * 64
        self.reference_transaction_count = count
        self.evidence_sha256 = "4" * 64

    def as_dict(self) -> dict[str, str]:
        return {"evidence_sha256": self.evidence_sha256}


class _FeatureBatch:
    def __init__(
        self,
        plan_sha256: str,
        reference_bundle_sha256: str,
        target_population_sha256: str,
        rows: tuple[_FeatureRow, ...],
        catalog_sha256: str,
    ) -> None:
        self.plan_sha256 = plan_sha256
        self.reference_bundle_sha256 = reference_bundle_sha256
        self.target_population_sha256 = target_population_sha256
        self.rows = rows
        self.feature_catalog_sha256 = catalog_sha256
        self.evidence_sha256 = _digest(
            {
                "target_population_sha256": target_population_sha256,
                "rows": [row.evidence_sha256 for row in rows],
            }
        )

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def as_dict(self) -> dict[str, object]:
        return {
            "row_count": self.row_count,
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True)
class _Format:
    value: str


@dataclass(frozen=True)
class _Export:
    format: _Format
    payload_sha256: str
    media_type: str
    row_count: int
    serialization_spec_sha256: str
    byte_count: int


@dataclass(frozen=True)
class _ReviewScores:
    row_count: int


@dataclass(frozen=True)
class _ReviewSelection:
    selected_count: int


class _ModelReviewRun:
    def __init__(
        self,
        plan: _ModelPlan,
        training_sha256: str,
        scoring_sha256: str,
        row_count: int,
    ) -> None:
        self.plan = plan
        self.training_feature_population_sha256 = training_sha256
        self.scoring_feature_population_sha256 = scoring_sha256
        self.evidence_sha256 = "5" * 64
        self.review_scores = _ReviewScores(row_count)
        self.review_selection = _ReviewSelection(1)
        self.serialized_exports = (
            _Export(
                _Format("json"),
                "6" * 64,
                "application/json",
                row_count,
                "7" * 64,
                10,
            ),
            _Export(
                _Format("csv"),
                "8" * 64,
                "text/csv",
                row_count,
                "9" * 64,
                20,
            ),
        )


def _feature_population_digest(rows: tuple[_FeatureRow, ...]) -> str:
    return _digest(
        [
            {
                "transaction_id": row.transaction_id,
                "award_id": row.award_id,
                "evidence_sha256": row.evidence_sha256,
            }
            for row in rows
        ]
    )


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple[str, ...]]]:
    calls: list[tuple[str, tuple[str, ...]]] = []

    monkeypatch.setattr(run_module, "ProcurementTransaction", _Transaction)
    monkeypatch.setattr(run_module, "FeatureBuildPlan", _FeaturePlan)
    monkeypatch.setattr(run_module, "ModelReviewPlan", _ModelPlan)
    monkeypatch.setattr(run_module, "FrozenFeatureReferences", _References)
    monkeypatch.setattr(run_module, "CandidateFeatureBatch", _FeatureBatch)
    monkeypatch.setattr(run_module, "ModelReviewRun", _ModelReviewRun)

    def build_references(
        transactions: tuple[_Transaction, ...],
        plan: _FeaturePlan,
    ) -> _References:
        population = tuple(item.transaction_id for item in transactions)
        calls.append(("references", population))
        return _References(plan, len(population))

    def build_features(
        transactions: tuple[_Transaction, ...],
        references: _References,
    ) -> _FeatureBatch:
        population = tuple(item.transaction_id for item in transactions)
        calls.append(("features", population))
        target_sha = _digest(
            [
                (item.transaction_id, item.award_id, item.payload)
                for item in transactions
            ]
        )
        rows = tuple(
            _FeatureRow(
                item.transaction_id,
                item.award_id,
                _digest(
                    (item.transaction_id, item.award_id, item.payload)
                ),
            )
            for item in transactions
        )
        return _FeatureBatch(
            references.plan.sha256_hex,
            references.evidence_sha256,
            target_sha,
            rows,
            references.plan.feature_catalog_sha256,
        )

    def model_review(
        *,
        training_feature_rows: tuple[_FeatureRow, ...],
        scoring_feature_rows: tuple[_FeatureRow, ...],
        scoring_transactions: tuple[_Transaction, ...],
        plan: _ModelPlan,
    ) -> _ModelReviewRun:
        calls.append(
            (
                "model",
                tuple(item.transaction_id for item in scoring_transactions),
            )
        )
        return _ModelReviewRun(
            plan,
            _feature_population_digest(tuple(training_feature_rows)),
            _feature_population_digest(tuple(scoring_feature_rows)),
            len(tuple(scoring_feature_rows)),
        )

    monkeypatch.setattr(run_module, "build_feature_references", build_references)
    monkeypatch.setattr(run_module, "build_candidate_features", build_features)
    monkeypatch.setattr(run_module, "run_model_review", model_review)
    return calls


def test_same_population_reuses_feature_batch_and_builds_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fakes(monkeypatch)
    feature_plan = _FeaturePlan()
    model_plan = _ModelPlan()
    population = (
        _Transaction("t1", "a1", 10),
        _Transaction("t2", "a2", 20),
    )

    result = run_module.run_procurelens_analysis(
        reference_transactions=population,
        scoring_transactions=population,
        feature_plan=feature_plan,
        model_plan=model_plan,
        run_name="same-population",
        source_revision="abc123",
        software_components={"scikit-learn": "1.9.0"},
    )

    assert [item for item in calls if item[0] == "features"] == [
        ("features", ("t1", "t2"))
    ]
    assert result.training_features is result.scoring_features
    assert result.manifest.final_artifact_ids == (
        "model_review_run",
        "export_00_json",
        "export_01_csv",
    )
    assert (
        result.manifest.artifact_index["export_00_json"].sha256_hex
        == "6" * 64
    )
    assert (
        result.manifest.artifact_index["export_01_csv"].sha256_hex
        == "8" * 64
    )


def test_separate_scoring_population_gets_separate_feature_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fakes(monkeypatch)
    reference = (
        _Transaction("r1", "a1", 1),
        _Transaction("r2", "a2", 2),
    )
    scoring = (
        _Transaction("s2", "b2", 20),
        _Transaction("s1", "b1", 10),
    )

    result = run_module.run_procurelens_analysis(
        reference_transactions=reference,
        scoring_transactions=scoring,
        feature_plan=_FeaturePlan(),
        model_plan=_ModelPlan(),
        run_name="separate-population",
    )

    assert [item for item in calls if item[0] == "features"] == [
        ("features", ("r1", "r2")),
        ("features", ("s2", "s1")),
    ]
    assert tuple(row.transaction_id for row in result.scoring_features.rows) == (
        "s2",
        "s1",
    )


def test_catalog_mismatch_fails_before_any_pipeline_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fakes(monkeypatch)
    population = (_Transaction("t1", "a1", 1),)

    with pytest.raises(run_module.AnalysisRunError, match="catalog"):
        run_module.run_procurelens_analysis(
            reference_transactions=population,
            scoring_transactions=population,
            feature_plan=_FeaturePlan("a" * 64),
            model_plan=_ModelPlan("b" * 64),
            run_name="bad-catalog",
        )

    assert calls == []


def test_duplicate_transaction_ids_fail_before_reference_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fakes(monkeypatch)
    duplicate_reference = (
        _Transaction("dup", "a1", 1),
        _Transaction("dup", "a2", 2),
    )
    scoring = (_Transaction("s1", "b1", 3),)

    with pytest.raises(run_module.AnalysisRunError, match="duplicate transaction_id"):
        run_module.run_procurelens_analysis(
            reference_transactions=duplicate_reference,
            scoring_transactions=scoring,
            feature_plan=_FeaturePlan(),
            model_plan=_ModelPlan(),
            run_name="duplicates",
        )

    assert calls == []


def _digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
