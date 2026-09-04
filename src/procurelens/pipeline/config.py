"""Explicit end-to-end modeling/review plans for ProcureLens.

The plan binds already-resolved feature/preprocessing contracts to named detector
runs, score calibration, heterogeneous ensemble policy, review selection,
explanation, and export serialization. It intentionally contains no hidden
hyperparameters, anomaly threshold, feature defaults, review budget, or detector
weight.

The current public architecture uses two complementary detector families:
standard Isolation Forest and the frozen empirical-tail detector. Isolation
Forest may be repeated under multiple explicit configs/seeds to measure stability;
exactly one run is designated as the primary review-score run.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from typing import Any

from procurelens.detectors.calibration import ScoreCalibrationSpec
from procurelens.detectors.ensemble import EnsembleMethod
from procurelens.detectors.isolation_forest import IsolationForestConfig
from procurelens.export.writer import ExportFormat, ExportSerializationSpec
from procurelens.model.feature_spec import ResolvedFeatureSelection
from procurelens.model.preprocessing_spec import ResolvedPreprocessingSpec
from procurelens.review.explanation import ExplanationSpec
from procurelens.review.policy import ReviewPolicySpec


class PipelineConfigError(ValueError):
    """Raised when a modeling/review run plan is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class IsolationForestRunPlan:
    """One named, reproducible Isolation Forest fit used by the run."""

    run_name: str
    config: IsolationForestConfig
    primary: bool

    def __post_init__(self) -> None:
        name = self.run_name.strip()
        if not name:
            raise PipelineConfigError("Isolation Forest run_name must not be blank")
        object.__setattr__(self, "run_name", name)
        if not isinstance(self.config, IsolationForestConfig):
            raise TypeError("config must be IsolationForestConfig")
        if not isinstance(self.primary, bool):
            raise PipelineConfigError("primary must be bool")

    @property
    def sha256_hex(self) -> str:
        return _digest(
            {
                "run_name": self.run_name,
                "config_sha256": self.config.sha256_hex,
                "primary": self.primary,
            }
        )


@dataclass(frozen=True, slots=True)
class TwoDetectorEnsemblePlan:
    """Artifact-independent policy for combining IF and empirical-tail evidence."""

    method: EnsembleMethod
    isolation_forest_weight: Decimal | None
    empirical_tail_weight: Decimal | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", EnsembleMethod(self.method))

        if self.method is EnsembleMethod.WEIGHTED_MEAN:
            left = self.isolation_forest_weight
            right = self.empirical_tail_weight
            for name, value in (
                ("isolation_forest_weight", left),
                ("empirical_tail_weight", right),
            ):
                if (
                    not isinstance(value, Decimal)
                    or not value.is_finite()
                    or value <= 0
                ):
                    raise PipelineConfigError(
                        f"{name} must be positive finite Decimal"
                    )
            assert left is not None and right is not None
            if left + right != Decimal(1):
                raise PipelineConfigError(
                    "weighted ensemble detector weights must sum exactly to Decimal(1)"
                )
        elif (
            self.isolation_forest_weight is not None
            or self.empirical_tail_weight is not None
        ):
            raise PipelineConfigError(
                "maximum/median ensemble plan must not carry unused weights"
            )

    @property
    def sha256_hex(self) -> str:
        return _digest(
            {
                "method": self.method.value,
                "isolation_forest_weight": _decimal_text(
                    self.isolation_forest_weight
                ),
                "empirical_tail_weight": _decimal_text(
                    self.empirical_tail_weight
                ),
                "member_families": [
                    "isolation_forest",
                    "empirical_tail",
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class ModelReviewPlan:
    """Complete explicit policy contract from candidate features through exports."""

    name: str
    description: str

    feature_selection: ResolvedFeatureSelection
    preprocessing: ResolvedPreprocessingSpec

    isolation_forest_runs: tuple[IsolationForestRunPlan, ...]
    calibration: ScoreCalibrationSpec
    ensemble: TwoDetectorEnsemblePlan
    build_stability_report: bool

    review_policy: ReviewPolicySpec
    explanation: ExplanationSpec
    exports: tuple[ExportSerializationSpec, ...]

    def __post_init__(self) -> None:
        for field_name in ("name", "description"):
            text = getattr(self, field_name).strip()
            if not text:
                raise PipelineConfigError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)

        if not isinstance(self.feature_selection, ResolvedFeatureSelection):
            raise TypeError(
                "feature_selection must be ResolvedFeatureSelection"
            )
        if not isinstance(self.preprocessing, ResolvedPreprocessingSpec):
            raise TypeError(
                "preprocessing must be ResolvedPreprocessingSpec"
            )
        if (
            self.preprocessing.selection.sha256_hex
            != self.feature_selection.sha256_hex
        ):
            raise PipelineConfigError(
                "preprocessing selection differs from model feature selection"
            )

        runs = tuple(self.isolation_forest_runs)
        if not runs:
            raise PipelineConfigError(
                "at least one Isolation Forest run must be configured"
            )
        if any(not isinstance(item, IsolationForestRunPlan) for item in runs):
            raise TypeError(
                "isolation_forest_runs must be IsolationForestRunPlan"
            )
        run_names = tuple(item.run_name for item in runs)
        if len(run_names) != len(set(run_names)):
            raise PipelineConfigError(
                "Isolation Forest run names must be unique"
            )
        config_hashes = tuple(item.config.sha256_hex for item in runs)
        if len(config_hashes) != len(set(config_hashes)):
            raise PipelineConfigError(
                "Isolation Forest run configs must be distinct"
            )
        primary_count = sum(item.primary for item in runs)
        if primary_count != 1:
            raise PipelineConfigError(
                "exactly one Isolation Forest run must be primary"
            )
        object.__setattr__(self, "isolation_forest_runs", runs)

        if not isinstance(self.calibration, ScoreCalibrationSpec):
            raise TypeError("calibration must be ScoreCalibrationSpec")
        if not isinstance(self.ensemble, TwoDetectorEnsemblePlan):
            raise TypeError("ensemble must be TwoDetectorEnsemblePlan")
        if not isinstance(self.build_stability_report, bool):
            raise PipelineConfigError("build_stability_report must be bool")
        if self.build_stability_report and len(runs) < 2:
            raise PipelineConfigError(
                "stability report requires at least two explicit IF runs"
            )

        if not isinstance(self.review_policy, ReviewPolicySpec):
            raise TypeError("review_policy must be ReviewPolicySpec")
        if not isinstance(self.explanation, ExplanationSpec):
            raise TypeError("explanation must be ExplanationSpec")
        if (
            self.explanation.catalog_sha256
            != self.feature_selection.spec.catalog_sha256
        ):
            raise PipelineConfigError(
                "explanation catalog differs from feature-selection catalog"
            )

        exports = tuple(self.exports)
        if not exports:
            raise PipelineConfigError(
                "public run requires export serialization specs"
            )
        if any(not isinstance(item, ExportSerializationSpec) for item in exports):
            raise TypeError("exports must be ExportSerializationSpec")
        hashes = tuple(item.sha256_hex for item in exports)
        if len(hashes) != len(set(hashes)):
            raise PipelineConfigError(
                "export serialization specs must be unique"
            )
        formats = {item.format for item in exports}
        if ExportFormat.JSON not in formats or ExportFormat.CSV not in formats:
            raise PipelineConfigError(
                "public run requires at least one JSON and one CSV export"
            )
        object.__setattr__(self, "exports", exports)

    @property
    def primary_isolation_forest_run(self) -> IsolationForestRunPlan:
        return next(item for item in self.isolation_forest_runs if item.primary)

    @property
    def sha256_hex(self) -> str:
        return _digest(self.as_dict(include_sha=False))

    def as_dict(self, *, include_sha: bool = True) -> dict[str, Any]:
        result = {
            "name": self.name,
            "description": self.description,
            "feature_selection_sha256":
                self.feature_selection.sha256_hex,
            "preprocessing_sha256": self.preprocessing.sha256_hex,
            "isolation_forest_runs": [
                {
                    "run_name": item.run_name,
                    "config_sha256": item.config.sha256_hex,
                    "primary": item.primary,
                    "run_plan_sha256": item.sha256_hex,
                }
                for item in self.isolation_forest_runs
            ],
            "calibration_sha256": self.calibration.sha256_hex,
            "ensemble_plan_sha256": self.ensemble.sha256_hex,
            "build_stability_report": self.build_stability_report,
            "review_policy_sha256": self.review_policy.sha256_hex,
            "explanation_sha256": self.explanation.sha256_hex,
            "export_spec_sha256": [
                item.sha256_hex for item in self.exports
            ],
        }
        if include_sha:
            result["sha256"] = self.sha256_hex
        return result


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


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
