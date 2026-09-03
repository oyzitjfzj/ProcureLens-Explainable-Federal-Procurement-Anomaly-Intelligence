"""Train-fitted calibration evidence for heterogeneous ProcureLens detector scores.

Maps already oriented detector anomaly scores into comparable empirical positions
while retaining robust distance from the detector's own training-score distribution.
Calibration is fitted on a score batch produced for the exact training matrix only;
new/scoring batches never alter the fitted reference distribution.

Empirical position is descriptive, not a probability of fraud or misconduct.
Robust distances preserve tail magnitude that rank/ECDF views can saturate away.
No contamination assumption, decision threshold, sigmoid prior, or risk score lives
here.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
import math
from typing import Any

from procurelens.detectors.base import (
    DetectorScoreBatch,
    ScoreOrientation,
)
from procurelens.statistics.robust import (
    QuantileMethod,
    RobustSummary,
    empirical_position_sorted,
    iqr_distance,
    modified_z,
    sorted_decimals,
    summarize_sorted,
)


class ScoreCalibrationError(ValueError):
    """Raised when detector-score calibration evidence is inconsistent."""


@dataclass(frozen=True, slots=True)
class ScoreCalibrationSpec:
    """Statistical definition used to summarize the training-score distribution."""

    name: str
    quantile_method: QuantileMethod

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ScoreCalibrationError("calibration spec name must not be blank")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self, "quantile_method", QuantileMethod(self.quantile_method)
        )

    @property
    def sha256_hex(self) -> str:
        return _digest(
            {
                "name": self.name,
                "quantile_method": self.quantile_method.value,
                "empirical_position": "strict_midpoint_weak",
                "robust_mad_normalization": "procurelens_standard_modified_z",
                "iqr_distance": "median_centered",
            }
        )


@dataclass(frozen=True, slots=True)
class FittedScoreCalibration:
    """Frozen training-score reference for one exact fitted detector."""

    spec: ScoreCalibrationSpec
    source_training_batch_sha256: str
    detector_name: str
    detector_family: str
    implementation_name: str
    implementation_version: str
    score_orientation: ScoreOrientation
    config_sha256: str
    fitted_model_sha256: str
    training_matrix_sha256: str
    preprocessor_sha256: str
    output_feature_names: tuple[str, ...]
    training_anomaly_scores: tuple[Decimal, ...]
    training_summary: RobustSummary

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ScoreCalibrationSpec):
            raise TypeError("spec must be ScoreCalibrationSpec")
        object.__setattr__(
            self,
            "source_training_batch_sha256",
            _digest_hex(
                self.source_training_batch_sha256,
                "source_training_batch_sha256",
            ),
        )
        for field_name in (
            "detector_name",
            "detector_family",
            "implementation_name",
            "implementation_version",
        ):
            text = getattr(self, field_name).strip()
            if not text:
                raise ScoreCalibrationError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)
        object.__setattr__(
            self, "score_orientation", ScoreOrientation(self.score_orientation)
        )
        for field_name in (
            "config_sha256",
            "fitted_model_sha256",
            "training_matrix_sha256",
            "preprocessor_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest_hex(getattr(self, field_name), field_name),
            )

        names = tuple(name.strip() for name in self.output_feature_names)
        if (
            not names
            or any(not name for name in names)
            or len(names) != len(set(names))
        ):
            raise ScoreCalibrationError(
                "output_feature_names must be non-empty and unique"
            )
        object.__setattr__(self, "output_feature_names", names)

        scores = tuple(self.training_anomaly_scores)
        if len(scores) < 2:
            raise ScoreCalibrationError(
                "score calibration requires at least two training scores"
            )
        if any(
            not isinstance(value, Decimal) or not value.is_finite()
            for value in scores
        ):
            raise ScoreCalibrationError(
                "training anomaly scores must be finite Decimal values"
            )
        if tuple(sorted(scores)) != scores:
            raise ScoreCalibrationError(
                "training anomaly scores must be deterministically sorted"
            )
        if len(set(scores)) < 2:
            raise ScoreCalibrationError(
                "training detector scores have no variation to calibrate"
            )
        object.__setattr__(self, "training_anomaly_scores", scores)

        if (
            not isinstance(self.training_summary, RobustSummary)
            or self.training_summary.count != len(scores)
            or self.training_summary.quantile_method is not self.spec.quantile_method
        ):
            raise ScoreCalibrationError(
                "training robust summary differs from calibration score distribution"
            )
        if (
            self.training_summary.minimum != scores[0]
            or self.training_summary.maximum != scores[-1]
        ):
            raise ScoreCalibrationError(
                "training robust summary bounds differ from calibration scores"
            )

    @property
    def training_score_count(self) -> int:
        return len(self.training_anomaly_scores)

    @property
    def sha256_hex(self) -> str:
        return _digest(
            {
                "spec_sha256": self.spec.sha256_hex,
                "source_training_batch_sha256": self.source_training_batch_sha256,
                "detector_name": self.detector_name,
                "detector_family": self.detector_family,
                "implementation_name": self.implementation_name,
                "implementation_version": self.implementation_version,
                "score_orientation": self.score_orientation.value,
                "config_sha256": self.config_sha256,
                "fitted_model_sha256": self.fitted_model_sha256,
                "training_matrix_sha256": self.training_matrix_sha256,
                "preprocessor_sha256": self.preprocessor_sha256,
                "output_feature_names": list(self.output_feature_names),
                "training_anomaly_scores": [
                    str(value) for value in self.training_anomaly_scores
                ],
                "training_summary": _summary_dict(self.training_summary),
            }
        )


@dataclass(frozen=True, slots=True)
class CalibratedDetectorScore:
    """One detector score interpreted against its frozen training-score reference."""

    transaction_id: str
    award_id: str
    source_anomaly_score: float
    empirical_lower_fraction: Decimal
    empirical_midpoint_fraction: Decimal
    empirical_upper_fraction: Decimal
    modified_z: Decimal | None
    modified_z_unavailable_reason: str | None
    iqr_distance: Decimal | None
    iqr_distance_unavailable_reason: str | None

    def __post_init__(self) -> None:
        for field_name in ("transaction_id", "award_id"):
            text = getattr(self, field_name).strip()
            if not text:
                raise ScoreCalibrationError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)
        if (
            not isinstance(self.source_anomaly_score, float)
            or not math.isfinite(self.source_anomaly_score)
        ):
            raise ScoreCalibrationError(
                "source_anomaly_score must be a finite float"
            )
        lower = self.empirical_lower_fraction
        midpoint = self.empirical_midpoint_fraction
        upper = self.empirical_upper_fraction
        for field_name, value in (
            ("empirical_lower_fraction", lower),
            ("empirical_midpoint_fraction", midpoint),
            ("empirical_upper_fraction", upper),
        ):
            _fraction(value, field_name)
        if not lower <= midpoint <= upper:
            raise ScoreCalibrationError(
                "empirical position fractions are inconsistent"
            )
        _optional_measure(
            self.modified_z,
            self.modified_z_unavailable_reason,
            "modified_z",
        )
        _optional_measure(
            self.iqr_distance,
            self.iqr_distance_unavailable_reason,
            "iqr_distance",
        )

    @property
    def identity(self) -> tuple[str, str]:
        return self.transaction_id, self.award_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "award_id": self.award_id,
            "source_anomaly_score_hex": self.source_anomaly_score.hex(),
            "empirical_lower_fraction": str(self.empirical_lower_fraction),
            "empirical_midpoint_fraction": str(self.empirical_midpoint_fraction),
            "empirical_upper_fraction": str(self.empirical_upper_fraction),
            "modified_z": _decimal_text(self.modified_z),
            "modified_z_unavailable_reason": self.modified_z_unavailable_reason,
            "iqr_distance": _decimal_text(self.iqr_distance),
            "iqr_distance_unavailable_reason": self.iqr_distance_unavailable_reason,
        }


@dataclass(frozen=True, slots=True)
class CalibratedScoreBatch:
    """Calibrated scores from one source detector batch."""

    calibration_sha256: str
    source_score_batch_sha256: str
    detector_name: str
    detector_family: str
    fitted_model_sha256: str
    training_matrix_sha256: str
    scoring_matrix_sha256: str
    preprocessor_sha256: str
    scores: tuple[CalibratedDetectorScore, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "calibration_sha256",
            "source_score_batch_sha256",
            "fitted_model_sha256",
            "training_matrix_sha256",
            "scoring_matrix_sha256",
            "preprocessor_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest_hex(getattr(self, field_name), field_name),
            )
        for field_name in ("detector_name", "detector_family"):
            text = getattr(self, field_name).strip()
            if not text:
                raise ScoreCalibrationError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)
        scores = tuple(self.scores)
        if not scores:
            raise ScoreCalibrationError(
                "calibrated score batch requires at least one score"
            )
        identities = tuple(score.identity for score in scores)
        if len(identities) != len(set(identities)):
            raise ScoreCalibrationError(
                "calibrated score batch contains duplicate row identities"
            )
        object.__setattr__(self, "scores", scores)

    @property
    def row_count(self) -> int:
        return len(self.scores)

    @property
    def row_identities(self) -> tuple[tuple[str, str], ...]:
        return tuple(score.identity for score in self.scores)

    @property
    def evidence_sha256(self) -> str:
        return _digest(
            {
                "calibration_sha256": self.calibration_sha256,
                "source_score_batch_sha256": self.source_score_batch_sha256,
                "detector_name": self.detector_name,
                "detector_family": self.detector_family,
                "fitted_model_sha256": self.fitted_model_sha256,
                "training_matrix_sha256": self.training_matrix_sha256,
                "scoring_matrix_sha256": self.scoring_matrix_sha256,
                "preprocessor_sha256": self.preprocessor_sha256,
                "scores": [score.as_dict() for score in self.scores],
            }
        )


def fit_score_calibration(
    training_batch: DetectorScoreBatch,
    spec: ScoreCalibrationSpec,
) -> FittedScoreCalibration:
    """Fit calibration only from a detector batch scored on its training matrix."""

    if not isinstance(training_batch, DetectorScoreBatch):
        raise TypeError("training_batch must be DetectorScoreBatch")
    if not isinstance(spec, ScoreCalibrationSpec):
        raise TypeError("spec must be ScoreCalibrationSpec")
    if training_batch.scoring_matrix_sha256 != training_batch.training_matrix_sha256:
        raise ScoreCalibrationError(
            "calibration must be fitted from scores on the exact training matrix"
        )
    if training_batch.row_count < 2:
        raise ScoreCalibrationError(
            "calibration requires at least two training detector scores"
        )

    ordered = sorted_decimals(
        Decimal.from_float(score.anomaly_score)
        for score in training_batch.scores
    )
    if len(set(ordered)) < 2:
        raise ScoreCalibrationError(
            "training detector scores have no variation to calibrate"
        )
    summary = summarize_sorted(ordered, method=spec.quantile_method)
    return FittedScoreCalibration(
        spec=spec,
        source_training_batch_sha256=training_batch.evidence_sha256,
        detector_name=training_batch.detector_name,
        detector_family=training_batch.detector_family,
        implementation_name=training_batch.implementation_name,
        implementation_version=training_batch.implementation_version,
        score_orientation=training_batch.score_orientation,
        config_sha256=training_batch.config_sha256,
        fitted_model_sha256=training_batch.fitted_model_sha256,
        training_matrix_sha256=training_batch.training_matrix_sha256,
        preprocessor_sha256=training_batch.preprocessor_sha256,
        output_feature_names=training_batch.output_feature_names,
        training_anomaly_scores=ordered,
        training_summary=summary,
    )


def apply_score_calibration(
    fitted: FittedScoreCalibration,
    score_batch: DetectorScoreBatch,
) -> CalibratedScoreBatch:
    """Interpret detector scores against a frozen training-score distribution."""

    if not isinstance(fitted, FittedScoreCalibration):
        raise TypeError("fitted must be FittedScoreCalibration")
    if not isinstance(score_batch, DetectorScoreBatch):
        raise TypeError("score_batch must be DetectorScoreBatch")
    _validate_compatible_batch(fitted, score_batch)

    calibrated: list[CalibratedDetectorScore] = []
    for score in score_batch.scores:
        target = Decimal.from_float(score.anomaly_score)
        position = empirical_position_sorted(
            fitted.training_anomaly_scores, target
        )
        mad_value = modified_z(target, fitted.training_summary)
        iqr_value = iqr_distance(target, fitted.training_summary)
        calibrated.append(
            CalibratedDetectorScore(
                transaction_id=score.transaction_id,
                award_id=score.award_id,
                source_anomaly_score=score.anomaly_score,
                empirical_lower_fraction=position.lower_fraction,
                empirical_midpoint_fraction=position.midpoint_fraction,
                empirical_upper_fraction=position.upper_fraction,
                modified_z=mad_value,
                modified_z_unavailable_reason=(
                    "training_score_mad_zero" if mad_value is None else None
                ),
                iqr_distance=iqr_value,
                iqr_distance_unavailable_reason=(
                    "training_score_iqr_zero" if iqr_value is None else None
                ),
            )
        )

    return CalibratedScoreBatch(
        calibration_sha256=fitted.sha256_hex,
        source_score_batch_sha256=score_batch.evidence_sha256,
        detector_name=score_batch.detector_name,
        detector_family=score_batch.detector_family,
        fitted_model_sha256=score_batch.fitted_model_sha256,
        training_matrix_sha256=score_batch.training_matrix_sha256,
        scoring_matrix_sha256=score_batch.scoring_matrix_sha256,
        preprocessor_sha256=score_batch.preprocessor_sha256,
        scores=tuple(calibrated),
    )


def _validate_compatible_batch(
    fitted: FittedScoreCalibration,
    batch: DetectorScoreBatch,
) -> None:
    checks = (
        ("detector_name", fitted.detector_name, batch.detector_name),
        ("detector_family", fitted.detector_family, batch.detector_family),
        ("implementation_name", fitted.implementation_name, batch.implementation_name),
        (
            "implementation_version",
            fitted.implementation_version,
            batch.implementation_version,
        ),
        ("score_orientation", fitted.score_orientation, batch.score_orientation),
        ("config_sha256", fitted.config_sha256, batch.config_sha256),
        ("fitted_model_sha256", fitted.fitted_model_sha256, batch.fitted_model_sha256),
        (
            "training_matrix_sha256",
            fitted.training_matrix_sha256,
            batch.training_matrix_sha256,
        ),
        ("preprocessor_sha256", fitted.preprocessor_sha256, batch.preprocessor_sha256),
        (
            "output_feature_names",
            fitted.output_feature_names,
            batch.output_feature_names,
        ),
    )
    for name, expected, observed in checks:
        if observed != expected:
            raise ScoreCalibrationError(
                f"score batch {name} differs from fitted calibration"
            )


def _optional_measure(
    value: Decimal | None,
    reason: str | None,
    name: str,
) -> None:
    if value is None:
        text = None if reason is None else reason.strip()
        if not text:
            raise ScoreCalibrationError(
                f"{name}: unavailable value requires a reason"
            )
        return
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ScoreCalibrationError(f"{name} must be finite Decimal or None")
    if reason is not None:
        raise ScoreCalibrationError(
            f"{name}: available value cannot carry unavailable reason"
        )


def _summary_dict(summary: RobustSummary) -> dict[str, Any]:
    return {
        "count": summary.count,
        "minimum": str(summary.minimum),
        "first_quartile": str(summary.first_quartile),
        "median": str(summary.median),
        "third_quartile": str(summary.third_quartile),
        "maximum": str(summary.maximum),
        "interquartile_range": str(summary.interquartile_range),
        "median_absolute_deviation": str(summary.median_absolute_deviation),
        "quantile_method": summary.quantile_method.value,
    }


def _fraction(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ScoreCalibrationError(f"{name} must be finite Decimal")
    if value < Decimal(0) or value > Decimal(1):
        raise ScoreCalibrationError(f"{name} must be between 0 and 1")


def _digest_hex(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ScoreCalibrationError(f"{name} must be a SHA-256 hex digest")
    return digest


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
