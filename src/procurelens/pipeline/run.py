"""Top-level reproducible ProcureLens analysis orchestration.

Connects frozen-reference feature construction to the train-safe model/review
pipeline and emits a deterministic provenance manifest for the complete run.
Reference and scoring populations remain explicit inputs; scoring data never
mutates feature references or fitted model/calibration state.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

from procurelens.domain.transaction import ProcurementTransaction
from procurelens.pipeline.config import ModelReviewPlan
from procurelens.pipeline.feature_config import FeatureBuildPlan
from procurelens.pipeline.features import (
    CandidateFeatureBatch,
    FrozenFeatureReferences,
    build_candidate_features,
    build_feature_references,
)
from procurelens.pipeline.model_review import ModelReviewRun, run_model_review
from procurelens.runtime.manifest import (
    PipelineArtifact,
    RunManifest,
    SoftwareComponent,
    StageExecution,
    capture_runtime_environment,
)


class AnalysisRunError(ValueError):
    """Raised when top-level run inputs or cross-stage provenance disagree."""


@dataclass(frozen=True, slots=True)
class ProcureLensAnalysisRun:
    """Complete feature + model/review execution with deterministic provenance."""

    run_name: str
    feature_plan: FeatureBuildPlan
    model_plan: ModelReviewPlan
    references: FrozenFeatureReferences
    training_features: CandidateFeatureBatch
    scoring_features: CandidateFeatureBatch
    model_review: ModelReviewRun
    manifest: RunManifest

    def __post_init__(self) -> None:
        name = self.run_name.strip()
        if not name:
            raise AnalysisRunError("run_name must not be blank")
        object.__setattr__(self, "run_name", name)

        if not isinstance(self.feature_plan, FeatureBuildPlan):
            raise TypeError("feature_plan must be FeatureBuildPlan")
        if not isinstance(self.model_plan, ModelReviewPlan):
            raise TypeError("model_plan must be ModelReviewPlan")
        if not isinstance(self.references, FrozenFeatureReferences):
            raise TypeError("references must be FrozenFeatureReferences")
        if not isinstance(self.training_features, CandidateFeatureBatch):
            raise TypeError("training_features must be CandidateFeatureBatch")
        if not isinstance(self.scoring_features, CandidateFeatureBatch):
            raise TypeError("scoring_features must be CandidateFeatureBatch")
        if not isinstance(self.model_review, ModelReviewRun):
            raise TypeError("model_review must be ModelReviewRun")
        if not isinstance(self.manifest, RunManifest):
            raise TypeError("manifest must be RunManifest")

        if self.references.plan.sha256_hex != self.feature_plan.sha256_hex:
            raise AnalysisRunError(
                "reference bundle was not built from feature_plan"
            )
        for label, batch in (
            ("training", self.training_features),
            ("scoring", self.scoring_features),
        ):
            if batch.plan_sha256 != self.feature_plan.sha256_hex:
                raise AnalysisRunError(
                    f"{label} feature batch plan differs from feature_plan"
                )
            if batch.reference_bundle_sha256 != self.references.evidence_sha256:
                raise AnalysisRunError(
                    f"{label} feature batch uses a different reference bundle"
                )
            if batch.feature_catalog_sha256 != self.feature_plan.feature_catalog_sha256:
                raise AnalysisRunError(
                    f"{label} feature catalog differs from feature_plan"
                )

        if self.model_review.plan.sha256_hex != self.model_plan.sha256_hex:
            raise AnalysisRunError(
                "model review artifact was not built from model_plan"
            )
        if (
            self.model_review.training_feature_population_sha256
            != _feature_population_digest(self.training_features)
        ):
            raise AnalysisRunError(
                "model review training feature population differs from batch"
            )
        if (
            self.model_review.scoring_feature_population_sha256
            != _feature_population_digest(self.scoring_features)
        ):
            raise AnalysisRunError(
                "model review scoring feature population differs from batch"
            )

        if self.manifest.run_name != self.run_name:
            raise AnalysisRunError("manifest run_name differs from analysis run")
        final_index = self.manifest.artifact_index
        model_artifact = final_index.get("model_review_run")
        if (
            model_artifact is None
            or model_artifact.sha256_hex != self.model_review.evidence_sha256
        ):
            raise AnalysisRunError(
                "manifest model_review_run artifact differs from model review"
            )
        expected_exports = {
            f"export_{index:02d}_{item.format.value}": item.payload_sha256
            for index, item in enumerate(self.model_review.serialized_exports)
        }
        for artifact_id, payload_sha in expected_exports.items():
            artifact = final_index.get(artifact_id)
            if artifact is None or artifact.sha256_hex != payload_sha:
                raise AnalysisRunError(
                    f"manifest export artifact mismatch: {artifact_id}"
                )

    @property
    def evidence_sha256(self) -> str:
        return _digest(
            {
                "run_name": self.run_name,
                "feature_plan_sha256": self.feature_plan.sha256_hex,
                "model_plan_sha256": self.model_plan.sha256_hex,
                "references_sha256": self.references.evidence_sha256,
                "training_features_sha256": self.training_features.evidence_sha256,
                "scoring_features_sha256": self.scoring_features.evidence_sha256,
                "model_review_sha256": self.model_review.evidence_sha256,
                "manifest_sha256": self.manifest.evidence_sha256,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "feature_plan_sha256": self.feature_plan.sha256_hex,
            "model_plan_sha256": self.model_plan.sha256_hex,
            "references": self.references.as_dict(),
            "training_features": self.training_features.as_dict(),
            "scoring_features": self.scoring_features.as_dict(),
            "model_review_sha256": self.model_review.evidence_sha256,
            "manifest_recipe_sha256": self.manifest.recipe_sha256,
            "manifest_evidence_sha256": self.manifest.evidence_sha256,
            "evidence_sha256": self.evidence_sha256,
        }


def run_procurelens_analysis(
    *,
    reference_transactions: Iterable[ProcurementTransaction],
    scoring_transactions: Iterable[ProcurementTransaction],
    feature_plan: FeatureBuildPlan,
    model_plan: ModelReviewPlan,
    run_name: str,
    source_revision: str | None = None,
    software_components: Mapping[str, str] | Iterable[SoftwareComponent] = (),
) -> ProcureLensAnalysisRun:
    """Execute the complete explicit analysis chain over canonical transactions."""

    if not isinstance(feature_plan, FeatureBuildPlan):
        raise TypeError("feature_plan must be FeatureBuildPlan")
    if not isinstance(model_plan, ModelReviewPlan):
        raise TypeError("model_plan must be ModelReviewPlan")
    name = run_name.strip()
    if not name:
        raise AnalysisRunError("run_name must not be blank")

    model_catalog_sha = model_plan.feature_selection.spec.catalog_sha256
    if model_catalog_sha != feature_plan.feature_catalog_sha256:
        raise AnalysisRunError(
            "model feature-selection catalog differs from feature-build catalog"
        )

    reference = _transactions(reference_transactions, "reference_transactions")
    scoring = _transactions(scoring_transactions, "scoring_transactions")

    references = build_feature_references(reference, feature_plan)
    training_features = build_candidate_features(reference, references)

    if scoring == reference:
        scoring_features = training_features
    else:
        scoring_features = build_candidate_features(scoring, references)

    model_review = run_model_review(
        training_feature_rows=training_features.rows,
        scoring_feature_rows=scoring_features.rows,
        scoring_transactions=scoring,
        plan=model_plan,
    )

    environment = capture_runtime_environment(software_components)
    manifest = _build_manifest(
        run_name=name,
        source_revision=source_revision,
        environment=environment,
        feature_plan=feature_plan,
        model_plan=model_plan,
        references=references,
        training_features=training_features,
        scoring_features=scoring_features,
        model_review=model_review,
    )

    return ProcureLensAnalysisRun(
        run_name=name,
        feature_plan=feature_plan,
        model_plan=model_plan,
        references=references,
        training_features=training_features,
        scoring_features=scoring_features,
        model_review=model_review,
        manifest=manifest,
    )


def _build_manifest(
    *,
    run_name: str,
    source_revision: str | None,
    environment: Any,
    feature_plan: FeatureBuildPlan,
    model_plan: ModelReviewPlan,
    references: FrozenFeatureReferences,
    training_features: CandidateFeatureBatch,
    scoring_features: CandidateFeatureBatch,
    model_review: ModelReviewRun,
) -> RunManifest:
    initial = (
        PipelineArtifact(
            artifact_id="reference_population",
            artifact_kind="canonical_transaction_population",
            sha256_hex=references.reference_population_sha256,
            attributes={
                "transaction_count": str(references.reference_transaction_count),
                "role": "feature_reference_and_model_training",
            },
        ),
        PipelineArtifact(
            artifact_id="scoring_population",
            artifact_kind="canonical_transaction_population_ordered",
            sha256_hex=scoring_features.target_population_sha256,
            attributes={
                "transaction_count": str(scoring_features.row_count),
                "role": "model_scoring_and_review",
            },
        ),
        PipelineArtifact(
            artifact_id="feature_plan",
            artifact_kind="feature_build_plan",
            sha256_hex=feature_plan.sha256_hex,
            media_type="application/json",
            attributes={"name": feature_plan.name},
        ),
        PipelineArtifact(
            artifact_id="model_plan",
            artifact_kind="model_review_plan",
            sha256_hex=model_plan.sha256_hex,
            media_type="application/json",
            attributes={"name": model_plan.name},
        ),
    )

    reference_artifact = PipelineArtifact(
        artifact_id="frozen_feature_references",
        artifact_kind="frozen_feature_reference_bundle",
        sha256_hex=references.evidence_sha256,
        attributes={
            "feature_catalog_sha256": feature_plan.feature_catalog_sha256,
        },
    )
    training_artifact = PipelineArtifact(
        artifact_id="training_candidate_features",
        artifact_kind="candidate_feature_batch",
        sha256_hex=training_features.evidence_sha256,
        attributes={"row_count": str(training_features.row_count)},
    )
    scoring_artifact = PipelineArtifact(
        artifact_id="scoring_candidate_features",
        artifact_kind="candidate_feature_batch",
        sha256_hex=scoring_features.evidence_sha256,
        attributes={"row_count": str(scoring_features.row_count)},
    )
    model_artifact = PipelineArtifact(
        artifact_id="model_review_run",
        artifact_kind="model_review_execution",
        sha256_hex=model_review.evidence_sha256,
        attributes={
            "review_row_count": str(model_review.review_scores.row_count),
            "flagged_row_count": str(model_review.review_selection.selected_count),
        },
    )

    export_artifacts = tuple(
        PipelineArtifact(
            artifact_id=f"export_{index:02d}_{item.format.value}",
            artifact_kind="serialized_review_export",
            sha256_hex=item.payload_sha256,
            media_type=item.media_type,
            attributes={
                "row_count": str(item.row_count),
                "serialization_spec_sha256": item.serialization_spec_sha256,
                "byte_count": str(item.byte_count),
            },
        )
        for index, item in enumerate(model_review.serialized_exports)
    )

    stages = (
        StageExecution(
            stage_id="build_feature_references",
            implementation="procurelens.pipeline.features.build_feature_references",
            input_artifact_ids=("reference_population", "feature_plan"),
            output_artifacts=(reference_artifact,),
            config_sha256=feature_plan.sha256_hex,
        ),
        StageExecution(
            stage_id="build_training_candidate_features",
            implementation="procurelens.pipeline.features.build_candidate_features",
            input_artifact_ids=(
                "reference_population",
                "feature_plan",
                "frozen_feature_references",
            ),
            output_artifacts=(training_artifact,),
            config_sha256=feature_plan.sha256_hex,
        ),
        StageExecution(
            stage_id="build_scoring_candidate_features",
            implementation="procurelens.pipeline.features.build_candidate_features",
            input_artifact_ids=(
                "scoring_population",
                "feature_plan",
                "frozen_feature_references",
            ),
            output_artifacts=(scoring_artifact,),
            config_sha256=feature_plan.sha256_hex,
        ),
        StageExecution(
            stage_id="run_model_review",
            implementation="procurelens.pipeline.model_review.run_model_review",
            input_artifact_ids=(
                "training_candidate_features",
                "scoring_candidate_features",
                "scoring_population",
                "model_plan",
            ),
            output_artifacts=(model_artifact,) + export_artifacts,
            config_sha256=model_plan.sha256_hex,
        ),
    )
    finals = ("model_review_run",) + tuple(
        artifact.artifact_id for artifact in export_artifacts
    )
    return RunManifest(
        run_name=run_name,
        source_revision=source_revision,
        environment=environment,
        initial_artifacts=initial,
        stages=stages,
        final_artifact_ids=finals,
    )


def _transactions(
    values: Iterable[ProcurementTransaction],
    name: str,
) -> tuple[ProcurementTransaction, ...]:
    if isinstance(values, (str, bytes)):
        raise AnalysisRunError(
            f"{name} must be an iterable of ProcurementTransaction"
        )
    items = tuple(values)
    if not items:
        raise AnalysisRunError(f"{name} must not be empty")
    if any(not isinstance(item, ProcurementTransaction) for item in items):
        raise TypeError(f"all {name} values must be ProcurementTransaction")
    transaction_ids = tuple(item.transaction_id for item in items)
    if len(transaction_ids) != len(set(transaction_ids)):
        raise AnalysisRunError(f"{name} contains duplicate transaction_id values")
    return items


def _feature_population_digest(batch: CandidateFeatureBatch) -> str:
    return _digest(
        [
            {
                "transaction_id": row.transaction_id,
                "award_id": row.award_id,
                "evidence_sha256": row.evidence_sha256,
            }
            for row in batch.rows
        ]
    )


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
