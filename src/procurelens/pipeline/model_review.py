
"""End-to-end modeling, review, explanation, and export orchestration.

Consumes already-built CandidateFeatureRow evidence plus canonical scoring
transactions. Every fit operation uses the explicit training population only.
Scoring rows never update imputation/scaling state, detector models, or score
calibrations.

Each named Isolation Forest run forms a separate heterogeneous ensemble with the
single frozen empirical-tail model. Exactly one run is designated primary by the
ModelReviewPlan; additional runs contribute stability evidence but do not silently
change the primary review score.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Iterable

from procurelens.detectors.base import DetectorScoreBatch
from procurelens.detectors.calibration import (
    CalibratedScoreBatch,
    FittedScoreCalibration,
    apply_score_calibration,
    fit_score_calibration,
)
from procurelens.detectors.empirical_tail import (
    EmpiricalTailScoreBatch,
    FittedEmpiricalTailDetector,
    fit_empirical_tail_detector,
    score_empirical_tail_detector,
)
from procurelens.detectors.ensemble import (
    EnsembleMemberSpec,
    EnsembleMethod,
    EnsembleScoreBatch,
    EnsembleSpec,
    combine_calibrated_scores,
)
from procurelens.detectors.isolation_forest import (
    FittedIsolationForest,
    fit_isolation_forest,
    score_isolation_forest,
)
from procurelens.detectors.stability import (
    StabilityReport,
    analyze_stability,
    stability_run_from_ensemble,
)
from procurelens.domain.transaction import ProcurementTransaction
from procurelens.export.records import (
    ExportRecordBatch,
    build_export_records,
)
from procurelens.export.writer import (
    SerializedExport,
    serialize_export,
)
from procurelens.model.feature_row import CandidateFeatureRow
from procurelens.model.matrix import (
    PreprocessedMatrix,
    build_preprocessed_matrix,
)
from procurelens.model.preprocessor import (
    FittedPreprocessor,
    fit_preprocessor,
    transform_row,
)
from procurelens.pipeline.config import (
    IsolationForestRunPlan,
    ModelReviewPlan,
)
from procurelens.review.evidence import (
    ReviewEvidenceBatch,
    build_review_evidence_batch,
)
from procurelens.review.explanation import (
    ExplanationBatch,
    build_explanations,
)
from procurelens.review.policy import (
    ReviewSelectionBatch,
    apply_review_policy,
)
from procurelens.review.score import (
    ReviewScoreBatch,
    build_review_scores,
)


class ModelReviewPipelineError(ValueError):
    """Raised when cross-stage pipeline provenance is inconsistent."""


@dataclass(frozen=True, slots=True)
class IsolationForestRunArtifacts:
    run_plan: IsolationForestRunPlan
    fitted_model: FittedIsolationForest
    training_scores: DetectorScoreBatch
    calibration: FittedScoreCalibration
    scoring_scores: DetectorScoreBatch
    calibrated_scoring: CalibratedScoreBatch
    ensemble: EnsembleScoreBatch

    def __post_init__(self) -> None:
        if not isinstance(self.run_plan, IsolationForestRunPlan):
            raise TypeError("run_plan must be IsolationForestRunPlan")
        if not isinstance(self.fitted_model, FittedIsolationForest):
            raise TypeError("fitted_model must be FittedIsolationForest")
        if not isinstance(self.training_scores, DetectorScoreBatch):
            raise TypeError("training_scores must be DetectorScoreBatch")
        if not isinstance(self.calibration, FittedScoreCalibration):
            raise TypeError("calibration must be FittedScoreCalibration")
        if not isinstance(self.scoring_scores, DetectorScoreBatch):
            raise TypeError("scoring_scores must be DetectorScoreBatch")
        if not isinstance(self.calibrated_scoring, CalibratedScoreBatch):
            raise TypeError("calibrated_scoring must be CalibratedScoreBatch")
        if not isinstance(self.ensemble, EnsembleScoreBatch):
            raise TypeError("ensemble must be EnsembleScoreBatch")

        if self.fitted_model.config_sha256 != self.run_plan.config.sha256_hex:
            raise ModelReviewPipelineError(
                f"{self.run_plan.run_name}: fitted IF config differs from plan"
            )
        if (
            self.training_scores.fitted_model_sha256
            != self.fitted_model.fitted_model_sha256
            or self.scoring_scores.fitted_model_sha256
            != self.fitted_model.fitted_model_sha256
        ):
            raise ModelReviewPipelineError(
                f"{self.run_plan.run_name}: IF score batch model fingerprint mismatch"
            )
        if (
            self.training_scores.scoring_matrix_sha256
            != self.training_scores.training_matrix_sha256
        ):
            raise ModelReviewPipelineError(
                f"{self.run_plan.run_name}: calibration training scores "
                "were not produced on the training matrix"
            )
        if (
            self.calibrated_scoring.calibration_sha256
            != self.calibration.sha256_hex
        ):
            raise ModelReviewPipelineError(
                f"{self.run_plan.run_name}: calibrated scoring batch "
                "uses a different calibration artifact"
            )
        if self.ensemble.row_identities != self.calibrated_scoring.row_identities:
            raise ModelReviewPipelineError(
                f"{self.run_plan.run_name}: ensemble population differs "
                "from calibrated IF scoring population"
            )

    @property
    def evidence_sha256(self) -> str:
        return _digest(
            {
                "run_plan_sha256": self.run_plan.sha256_hex,
                "fitted_model_sha256": self.fitted_model.fitted_model_sha256,
                "training_scores_sha256": self.training_scores.evidence_sha256,
                "calibration_sha256": self.calibration.sha256_hex,
                "scoring_scores_sha256": self.scoring_scores.evidence_sha256,
                "calibrated_scoring_sha256":
                    self.calibrated_scoring.evidence_sha256,
                "ensemble_sha256": self.ensemble.evidence_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class ModelReviewRun:
    plan: ModelReviewPlan
    training_feature_population_sha256: str
    scoring_feature_population_sha256: str

    fitted_preprocessor: FittedPreprocessor
    training_matrix: PreprocessedMatrix
    scoring_matrix: PreprocessedMatrix

    empirical_tail_model: FittedEmpiricalTailDetector
    empirical_tail_training_scores: EmpiricalTailScoreBatch
    empirical_tail_calibration: FittedScoreCalibration
    empirical_tail_scoring_scores: EmpiricalTailScoreBatch
    empirical_tail_calibrated_scoring: CalibratedScoreBatch

    isolation_forest_runs: tuple[IsolationForestRunArtifacts, ...]
    stability: StabilityReport | None

    review_evidence: ReviewEvidenceBatch
    review_scores: ReviewScoreBatch
    review_selection: ReviewSelectionBatch
    explanations: ExplanationBatch
    export_records: ExportRecordBatch
    serialized_exports: tuple[SerializedExport, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ModelReviewPlan):
            raise TypeError("plan must be ModelReviewPlan")
        for field_name in (
            "training_feature_population_sha256",
            "scoring_feature_population_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest_hex(getattr(self, field_name), field_name),
            )

        if not isinstance(self.fitted_preprocessor, FittedPreprocessor):
            raise TypeError("fitted_preprocessor must be FittedPreprocessor")
        if (
            self.fitted_preprocessor.resolved_spec_sha256
            != self.plan.preprocessing.sha256_hex
        ):
            raise ModelReviewPipelineError(
                "fitted preprocessor differs from pipeline preprocessing plan"
            )
        for matrix_name in ("training_matrix", "scoring_matrix"):
            matrix = getattr(self, matrix_name)
            if not isinstance(matrix, PreprocessedMatrix):
                raise TypeError(f"{matrix_name} must be PreprocessedMatrix")
            if matrix.preprocessor_sha256 != self.fitted_preprocessor.sha256_hex:
                raise ModelReviewPipelineError(
                    f"{matrix_name} uses a different fitted preprocessor"
                )

        if not isinstance(
            self.empirical_tail_model, FittedEmpiricalTailDetector
        ):
            raise TypeError(
                "empirical_tail_model must be FittedEmpiricalTailDetector"
            )
        if (
            self.empirical_tail_model.training_matrix_sha256
            != self.training_matrix.evidence_sha256
        ):
            raise ModelReviewPipelineError(
                "empirical-tail model was not fitted on training matrix"
            )
        if not isinstance(
            self.empirical_tail_training_scores, EmpiricalTailScoreBatch
        ) or not isinstance(
            self.empirical_tail_scoring_scores, EmpiricalTailScoreBatch
        ):
            raise TypeError(
                "empirical-tail scores must be EmpiricalTailScoreBatch"
            )
        if not isinstance(
            self.empirical_tail_calibration, FittedScoreCalibration
        ):
            raise TypeError(
                "empirical_tail_calibration must be FittedScoreCalibration"
            )
        if not isinstance(
            self.empirical_tail_calibrated_scoring, CalibratedScoreBatch
        ):
            raise TypeError(
                "empirical_tail_calibrated_scoring must be CalibratedScoreBatch"
            )
        if (
            self.empirical_tail_training_scores.training_matrix_sha256
            != self.training_matrix.evidence_sha256
            or self.empirical_tail_training_scores.scoring_matrix_sha256
            != self.training_matrix.evidence_sha256
        ):
            raise ModelReviewPipelineError(
                "empirical-tail calibration scores must use training matrix"
            )
        if (
            self.empirical_tail_scoring_scores.scoring_matrix_sha256
            != self.scoring_matrix.evidence_sha256
        ):
            raise ModelReviewPipelineError(
                "empirical-tail scoring batch differs from scoring matrix"
            )
        if (
            self.empirical_tail_calibrated_scoring.calibration_sha256
            != self.empirical_tail_calibration.sha256_hex
        ):
            raise ModelReviewPipelineError(
                "empirical-tail calibrated scoring uses wrong calibration"
            )

        runs = tuple(self.isolation_forest_runs)
        if len(runs) != len(self.plan.isolation_forest_runs):
            raise ModelReviewPipelineError(
                "IF run artifact count differs from pipeline plan"
            )
        if tuple(item.run_plan.run_name for item in runs) != tuple(
            item.run_name for item in self.plan.isolation_forest_runs
        ):
            raise ModelReviewPipelineError(
                "IF run artifact order differs from pipeline plan"
            )
        object.__setattr__(self, "isolation_forest_runs", runs)

        if self.plan.build_stability_report:
            if not isinstance(self.stability, StabilityReport):
                raise ModelReviewPipelineError(
                    "pipeline plan requires stability report"
                )
            ensemble_hashes = {
                item.ensemble.evidence_sha256 for item in runs
            }
            stability_hashes = {
                item.source_evidence_sha256 for item in self.stability.runs
            }
            if stability_hashes != ensemble_hashes:
                raise ModelReviewPipelineError(
                    "stability report does not cover exact ensemble run artifacts"
                )
        elif self.stability is not None:
            raise ModelReviewPipelineError(
                "pipeline produced stability evidence when plan disabled it"
            )

        primary = self.primary_isolation_forest_run.ensemble
        if self.review_evidence.ensemble_batch_sha256 != primary.evidence_sha256:
            raise ModelReviewPipelineError(
                "review evidence was not built from primary ensemble"
            )
        if (
            self.review_scores.source_review_batch_sha256
            != self.review_evidence.evidence_sha256
        ):
            raise ModelReviewPipelineError(
                "review scores were not built from review evidence"
            )
        if (
            self.review_selection.source_score_batch_sha256
            != self.review_scores.evidence_sha256
        ):
            raise ModelReviewPipelineError(
                "review selection was not built from review scores"
            )
        if (
            self.explanations.review_evidence_batch_sha256
            != self.review_evidence.evidence_sha256
            or self.explanations.review_score_batch_sha256
            != self.review_scores.evidence_sha256
            or self.explanations.review_selection_batch_sha256
            != self.review_selection.evidence_sha256
        ):
            raise ModelReviewPipelineError(
                "explanation provenance differs from review artifacts"
            )
        if (
            self.export_records.explanation_batch_sha256
            != self.explanations.evidence_sha256
        ):
            raise ModelReviewPipelineError(
                "export records were not built from explanations"
            )

        exports = tuple(self.serialized_exports)
        if len(exports) != len(self.plan.exports):
            raise ModelReviewPipelineError(
                "serialized export count differs from plan"
            )
        if tuple(item.serialization_spec_sha256 for item in exports) != tuple(
            spec.sha256_hex for spec in self.plan.exports
        ):
            raise ModelReviewPipelineError(
                "serialized export spec order differs from plan"
            )
        if any(
            item.source_record_batch_sha256
            != self.export_records.evidence_sha256
            for item in exports
        ):
            raise ModelReviewPipelineError(
                "serialized export source differs from export records"
            )
        object.__setattr__(self, "serialized_exports", exports)

    @property
    def primary_isolation_forest_run(self) -> IsolationForestRunArtifacts:
        primary_name = self.plan.primary_isolation_forest_run.run_name
        return next(
            item
            for item in self.isolation_forest_runs
            if item.run_plan.run_name == primary_name
        )

    @property
    def primary_ensemble(self) -> EnsembleScoreBatch:
        return self.primary_isolation_forest_run.ensemble

    @property
    def evidence_sha256(self) -> str:
        return _digest(
            {
                "plan_sha256": self.plan.sha256_hex,
                "training_feature_population_sha256":
                    self.training_feature_population_sha256,
                "scoring_feature_population_sha256":
                    self.scoring_feature_population_sha256,
                "fitted_preprocessor_sha256":
                    self.fitted_preprocessor.sha256_hex,
                "training_matrix_sha256": self.training_matrix.evidence_sha256,
                "scoring_matrix_sha256": self.scoring_matrix.evidence_sha256,
                "empirical_tail_model_sha256":
                    self.empirical_tail_model.fitted_model_sha256,
                "empirical_tail_calibration_sha256":
                    self.empirical_tail_calibration.sha256_hex,
                "if_runs": [
                    item.evidence_sha256 for item in self.isolation_forest_runs
                ],
                "stability_sha256": (
                    None
                    if self.stability is None
                    else self.stability.evidence_sha256
                ),
                "review_evidence_sha256":
                    self.review_evidence.evidence_sha256,
                "review_scores_sha256": self.review_scores.evidence_sha256,
                "review_selection_sha256":
                    self.review_selection.evidence_sha256,
                "explanations_sha256": self.explanations.evidence_sha256,
                "export_records_sha256": self.export_records.evidence_sha256,
                "serialized_exports": [
                    item.evidence_sha256 for item in self.serialized_exports
                ],
            }
        )


def run_model_review(
    *,
    training_feature_rows: Iterable[CandidateFeatureRow],
    scoring_feature_rows: Iterable[CandidateFeatureRow],
    scoring_transactions: Iterable[ProcurementTransaction],
    plan: ModelReviewPlan,
) -> ModelReviewRun:
    """Execute the explicit train/score/review pipeline without hidden policy."""

    if not isinstance(plan, ModelReviewPlan):
        raise TypeError("plan must be ModelReviewPlan")
    training = _feature_rows(training_feature_rows, "training_feature_rows")
    scoring = _feature_rows(scoring_feature_rows, "scoring_feature_rows")
    transactions = _transactions(scoring_transactions)

    scoring_identities = tuple(
        (row.transaction_id, row.award_id) for row in scoring
    )
    transaction_identities = tuple(
        (row.transaction_id, row.award_id) for row in transactions
    )
    if set(transaction_identities) != set(scoring_identities):
        raise ModelReviewPipelineError(
            "scoring canonical transactions must exactly match scoring feature rows"
        )

    fitted_preprocessor = fit_preprocessor(training, plan.preprocessing)
    training_preprocessed = tuple(
        transform_row(row, fitted_preprocessor) for row in training
    )
    scoring_preprocessed = tuple(
        transform_row(row, fitted_preprocessor) for row in scoring
    )
    training_matrix = build_preprocessed_matrix(training_preprocessed)
    scoring_matrix = build_preprocessed_matrix(scoring_preprocessed)

    tail_model = fit_empirical_tail_detector(training_matrix)
    tail_training_scores = score_empirical_tail_detector(
        tail_model, training_matrix
    )
    tail_scoring_scores = score_empirical_tail_detector(
        tail_model, scoring_matrix
    )
    tail_calibration = fit_score_calibration(
        tail_training_scores.generic_scores,
        plan.calibration,
    )
    tail_calibrated = apply_score_calibration(
        tail_calibration,
        tail_scoring_scores.generic_scores,
    )

    if_run_artifacts: list[IsolationForestRunArtifacts] = []
    for run_plan in plan.isolation_forest_runs:
        fitted_if = fit_isolation_forest(
            training_matrix, run_plan.config
        )
        if_training_scores = score_isolation_forest(
            fitted_if, training_matrix
        )
        if_scoring_scores = score_isolation_forest(
            fitted_if, scoring_matrix
        )
        if_calibration = fit_score_calibration(
            if_training_scores, plan.calibration
        )
        if_calibrated = apply_score_calibration(
            if_calibration, if_scoring_scores
        )

        ensemble_spec = _ensemble_spec(
            plan,
            run_plan,
            if_calibrated,
            tail_calibrated,
        )
        ensemble = combine_calibrated_scores(
            ensemble_spec,
            (if_calibrated, tail_calibrated),
        )
        if_run_artifacts.append(
            IsolationForestRunArtifacts(
                run_plan=run_plan,
                fitted_model=fitted_if,
                training_scores=if_training_scores,
                calibration=if_calibration,
                scoring_scores=if_scoring_scores,
                calibrated_scoring=if_calibrated,
                ensemble=ensemble,
            )
        )

    if_runs = tuple(if_run_artifacts)
    stability = (
        analyze_stability(
            tuple(
                stability_run_from_ensemble(
                    item.run_plan.run_name, item.ensemble
                )
                for item in if_runs
            )
        )
        if plan.build_stability_report
        else None
    )

    primary_name = plan.primary_isolation_forest_run.run_name
    primary_ensemble = next(
        item.ensemble
        for item in if_runs
        if item.run_plan.run_name == primary_name
    )
    review_evidence = build_review_evidence_batch(
        scoring,
        primary_ensemble,
        stability=stability,
    )
    review_scores = build_review_scores(review_evidence)
    review_selection = apply_review_policy(
        review_scores, plan.review_policy
    )
    explanations = build_explanations(
        feature_rows=scoring,
        review_evidence=review_evidence,
        review_scores=review_scores,
        review_selection=review_selection,
        spec=plan.explanation,
    )
    export_records = build_export_records(
        transactions, explanations
    )
    serialized = tuple(
        serialize_export(export_records, spec)
        for spec in plan.exports
    )

    return ModelReviewRun(
        plan=plan,
        training_feature_population_sha256=_feature_population_digest(
            training
        ),
        scoring_feature_population_sha256=_feature_population_digest(
            scoring
        ),
        fitted_preprocessor=fitted_preprocessor,
        training_matrix=training_matrix,
        scoring_matrix=scoring_matrix,
        empirical_tail_model=tail_model,
        empirical_tail_training_scores=tail_training_scores,
        empirical_tail_calibration=tail_calibration,
        empirical_tail_scoring_scores=tail_scoring_scores,
        empirical_tail_calibrated_scoring=tail_calibrated,
        isolation_forest_runs=if_runs,
        stability=stability,
        review_evidence=review_evidence,
        review_scores=review_scores,
        review_selection=review_selection,
        explanations=explanations,
        export_records=export_records,
        serialized_exports=serialized,
    )


def _ensemble_spec(
    plan: ModelReviewPlan,
    run_plan: IsolationForestRunPlan,
    if_batch: CalibratedScoreBatch,
    tail_batch: CalibratedScoreBatch,
) -> EnsembleSpec:
    if plan.ensemble.method is EnsembleMethod.WEIGHTED_MEAN:
        if_weight = plan.ensemble.isolation_forest_weight
        tail_weight = plan.ensemble.empirical_tail_weight
    else:
        if_weight = None
        tail_weight = None

    members = (
        EnsembleMemberSpec(
            member_name=f"{run_plan.run_name}:isolation_forest",
            detector_name=if_batch.detector_name,
            calibration_sha256=if_batch.calibration_sha256,
            weight=if_weight,
        ),
        EnsembleMemberSpec(
            member_name="empirical_tail",
            detector_name=tail_batch.detector_name,
            calibration_sha256=tail_batch.calibration_sha256,
            weight=tail_weight,
        ),
    )
    return EnsembleSpec(
        name=f"{plan.name}:{run_plan.run_name}",
        description=(
            "Calibrated heterogeneous ProcureLens ensemble for named "
            f"Isolation Forest run {run_plan.run_name}."
        ),
        method=plan.ensemble.method,
        members=members,
    )


def _feature_rows(
    values: Iterable[CandidateFeatureRow],
    name: str,
) -> tuple[CandidateFeatureRow, ...]:
    if isinstance(values, (str, bytes)):
        raise ModelReviewPipelineError(
            f"{name} must be an iterable of CandidateFeatureRow"
        )
    rows = tuple(values)
    if not rows:
        raise ModelReviewPipelineError(f"{name} must not be empty")
    if any(not isinstance(row, CandidateFeatureRow) for row in rows):
        raise TypeError(f"all {name} values must be CandidateFeatureRow")
    identities = tuple(
        (row.transaction_id, row.award_id) for row in rows
    )
    if len(identities) != len(set(identities)):
        raise ModelReviewPipelineError(
            f"{name} contains duplicate row identities"
        )
    return rows


def _transactions(
    values: Iterable[ProcurementTransaction],
) -> tuple[ProcurementTransaction, ...]:
    if isinstance(values, (str, bytes)):
        raise ModelReviewPipelineError(
            "scoring_transactions must be an iterable"
        )
    rows = tuple(values)
    if not rows:
        raise ModelReviewPipelineError(
            "scoring_transactions must not be empty"
        )
    if any(not isinstance(row, ProcurementTransaction) for row in rows):
        raise TypeError(
            "all scoring_transactions must be ProcurementTransaction"
        )
    identities = tuple(
        (row.transaction_id, row.award_id) for row in rows
    )
    if len(identities) != len(set(identities)):
        raise ModelReviewPipelineError(
            "scoring_transactions contains duplicate identities"
        )
    return rows


def _feature_population_digest(
    rows: tuple[CandidateFeatureRow, ...],
) -> str:
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


def _digest_hex(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ModelReviewPipelineError(
            f"{name} must be a SHA-256 hex digest"
        )
    return digest


def _digest(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
