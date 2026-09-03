"""Frozen empirical-tail anomaly detector for ProcureLens.

Implements an ECOD-paper-inspired detector using training-only empirical tail
references. Per-feature left/right empirical tail masses and the training skew
direction are frozen at fit time; scoring new rows never recomputes the reference
distribution with test data.

The final score follows the paper-level aggregation pattern: independently sum
left-tail, right-tail, and skew-selected negative-log tail evidence, then take
the maximum of those three row scores. Tail probability is floored at the finite
empirical resolution 1/n, avoiding arbitrary epsilon constants and literal zero
probabilities outside the observed training range.

This is deliberately named an empirical-tail detector rather than claiming exact
behavioral identity with a third-party ECOD implementation.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from decimal import Decimal, localcontext
from enum import Enum
from hashlib import sha256
import json
import math
from typing import Any

from procurelens.detectors.base import (
    DetectorScoreBatch,
    ScoreOrientation,
    build_detector_scores,
)
from procurelens.model.matrix import PreprocessedMatrix


class EmpiricalTailDetectorError(ValueError):
    """Raised when empirical-tail detector evidence is inconsistent."""


DECIMAL_WORKING_PRECISION = 50
ALGORITHM_NAME = "procurelens_frozen_empirical_tail"
ALGORITHM_FAMILY = "empirical_tail"
ALGORITHM_REVISION = 1
TAIL_FLOOR_RULE = "one_over_training_count"
AGGREGATION_RULE = "max_of_left_right_skew_selected_sums"
ZERO_SKEW_TIE_RULE = "right"


class SkewDirection(str, Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True, slots=True)
class FeatureTailReference:
    """Frozen sorted training values and deterministic skew-tail direction."""

    feature_name: str
    sorted_training_values: tuple[Decimal, ...]
    skew_direction: SkewDirection

    def __post_init__(self) -> None:
        name = self.feature_name.strip()
        if not name:
            raise EmpiricalTailDetectorError("feature_name must not be blank")
        object.__setattr__(self, "feature_name", name)
        values = tuple(self.sorted_training_values)
        if len(values) < 2:
            raise EmpiricalTailDetectorError(
                "feature tail reference requires at least two training values"
            )
        if any(
            not isinstance(value, Decimal) or not value.is_finite()
            for value in values
        ):
            raise EmpiricalTailDetectorError(
                "feature tail reference values must be finite Decimal"
            )
        if tuple(sorted(values)) != values:
            raise EmpiricalTailDetectorError(
                "feature tail reference values must be deterministically sorted"
            )
        object.__setattr__(self, "sorted_training_values", values)
        object.__setattr__(
            self, "skew_direction", SkewDirection(self.skew_direction)
        )

    @property
    def training_count(self) -> int:
        return len(self.sorted_training_values)

    @property
    def minimum(self) -> Decimal:
        return self.sorted_training_values[0]

    @property
    def maximum(self) -> Decimal:
        return self.sorted_training_values[-1]

    @property
    def sha256_hex(self) -> str:
        return _digest(
            {
                "feature_name": self.feature_name,
                "skew_direction": self.skew_direction.value,
                "training_values": [
                    str(value) for value in self.sorted_training_values
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class FittedEmpiricalTailDetector:
    """Frozen empirical-tail reference fitted on one exact training matrix."""

    training_matrix_sha256: str
    preprocessor_sha256: str
    output_feature_names: tuple[str, ...]
    training_row_count: int
    references: tuple[FeatureTailReference, ...]
    fitted_model_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "training_matrix_sha256",
            "preprocessor_sha256",
            "fitted_model_sha256",
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
            raise EmpiricalTailDetectorError(
                "output_feature_names must be non-empty and unique"
            )
        if (
            isinstance(self.training_row_count, bool)
            or not isinstance(self.training_row_count, int)
            or self.training_row_count < 2
        ):
            raise EmpiricalTailDetectorError(
                "training_row_count must be an integer of at least two"
            )
        references = tuple(self.references)
        if tuple(reference.feature_name for reference in references) != names:
            raise EmpiricalTailDetectorError(
                "tail references must match feature order exactly"
            )
        if any(
            reference.training_count != self.training_row_count
            for reference in references
        ):
            raise EmpiricalTailDetectorError(
                "every feature tail reference must cover all training rows"
            )
        object.__setattr__(self, "output_feature_names", names)
        object.__setattr__(self, "references", references)

    @property
    def feature_count(self) -> int:
        return len(self.output_feature_names)

    @property
    def config_sha256(self) -> str:
        return empirical_tail_config_sha256()


@dataclass(frozen=True, slots=True)
class EmpiricalTailRowEvidence:
    """Exact Decimal component scores for one scored row."""

    transaction_id: str
    award_id: str
    left_score: Decimal
    right_score: Decimal
    skew_selected_score: Decimal
    final_score: Decimal

    def __post_init__(self) -> None:
        for field_name in ("transaction_id", "award_id"):
            text = getattr(self, field_name).strip()
            if not text:
                raise EmpiricalTailDetectorError(
                    f"{field_name} must not be blank"
                )
            object.__setattr__(self, field_name, text)
        for field_name in (
            "left_score",
            "right_score",
            "skew_selected_score",
            "final_score",
        ):
            value = getattr(self, field_name)
            if (
                not isinstance(value, Decimal)
                or not value.is_finite()
                or value < 0
            ):
                raise EmpiricalTailDetectorError(
                    f"{field_name} must be non-negative finite Decimal"
                )
        expected = max(
            self.left_score,
            self.right_score,
            self.skew_selected_score,
        )
        if self.final_score != expected:
            raise EmpiricalTailDetectorError(
                "final empirical-tail score differs from component maximum"
            )

    @property
    def identity(self) -> tuple[str, str]:
        return self.transaction_id, self.award_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "award_id": self.award_id,
            "left_score": str(self.left_score),
            "right_score": str(self.right_score),
            "skew_selected_score": str(self.skew_selected_score),
            "final_score": str(self.final_score),
        }


@dataclass(frozen=True, slots=True)
class EmpiricalTailScoreBatch:
    """Detector-specific exact evidence plus generic detector-score contract."""

    fitted_model_sha256: str
    training_matrix_sha256: str
    scoring_matrix_sha256: str
    preprocessor_sha256: str
    row_evidence: tuple[EmpiricalTailRowEvidence, ...]
    generic_scores: DetectorScoreBatch

    def __post_init__(self) -> None:
        for field_name in (
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
        evidence = tuple(self.row_evidence)
        if not evidence:
            raise EmpiricalTailDetectorError(
                "empirical-tail score batch requires row evidence"
            )
        identities = tuple(item.identity for item in evidence)
        if len(identities) != len(set(identities)):
            raise EmpiricalTailDetectorError(
                "empirical-tail score batch has duplicate row identities"
            )
        if not isinstance(self.generic_scores, DetectorScoreBatch):
            raise TypeError("generic_scores must be DetectorScoreBatch")
        if self.generic_scores.row_identities != identities:
            raise EmpiricalTailDetectorError(
                "generic detector scores differ from exact row evidence identities"
            )
        if self.generic_scores.fitted_model_sha256 != self.fitted_model_sha256:
            raise EmpiricalTailDetectorError(
                "generic detector model fingerprint differs from exact evidence"
            )
        if self.generic_scores.training_matrix_sha256 != self.training_matrix_sha256:
            raise EmpiricalTailDetectorError(
                "generic detector training matrix differs from exact evidence"
            )
        if self.generic_scores.scoring_matrix_sha256 != self.scoring_matrix_sha256:
            raise EmpiricalTailDetectorError(
                "generic detector scoring matrix differs from exact evidence"
            )
        if self.generic_scores.preprocessor_sha256 != self.preprocessor_sha256:
            raise EmpiricalTailDetectorError(
                "generic detector preprocessor differs from exact evidence"
            )
        for exact, generic in zip(evidence, self.generic_scores.scores):
            converted = _decimal_to_float(exact.final_score, "final_score")
            if generic.raw_score != converted or generic.anomaly_score != converted:
                raise EmpiricalTailDetectorError(
                    "generic detector score differs from exact Decimal evidence"
                )
        object.__setattr__(self, "row_evidence", evidence)

    @property
    def row_count(self) -> int:
        return len(self.row_evidence)

    @property
    def evidence_sha256(self) -> str:
        return _digest(
            {
                "fitted_model_sha256": self.fitted_model_sha256,
                "training_matrix_sha256": self.training_matrix_sha256,
                "scoring_matrix_sha256": self.scoring_matrix_sha256,
                "preprocessor_sha256": self.preprocessor_sha256,
                "row_evidence": [item.as_dict() for item in self.row_evidence],
                "generic_score_batch_sha256": self.generic_scores.evidence_sha256,
            }
        )


def empirical_tail_config_sha256() -> str:
    """Fingerprint the fixed statistical definition, not a review/risk policy."""

    return _digest(
        {
            "algorithm_name": ALGORITHM_NAME,
            "algorithm_revision": ALGORITHM_REVISION,
            "tail_floor_rule": TAIL_FLOOR_RULE,
            "aggregation_rule": AGGREGATION_RULE,
            "zero_skew_tie_rule": ZERO_SKEW_TIE_RULE,
            "working_decimal_precision": DECIMAL_WORKING_PRECISION,
        }
    )


def fit_empirical_tail_detector(
    matrix: PreprocessedMatrix,
) -> FittedEmpiricalTailDetector:
    """Fit frozen per-feature empirical references from the training matrix."""

    if not isinstance(matrix, PreprocessedMatrix):
        raise TypeError("matrix must be PreprocessedMatrix")
    if matrix.row_count < 2:
        raise EmpiricalTailDetectorError(
            "empirical-tail detector requires at least two training rows"
        )

    references: list[FeatureTailReference] = []
    for column_index, name in enumerate(matrix.output_feature_names):
        values = tuple(
            row[column_index]
            for row in matrix.values
        )
        ordered = tuple(sorted(values))
        direction = _skew_direction(values)
        references.append(
            FeatureTailReference(
                feature_name=name,
                sorted_training_values=ordered,
                skew_direction=direction,
            )
        )

    model_sha = _model_fingerprint(
        matrix_sha256=matrix.evidence_sha256,
        preprocessor_sha256=matrix.preprocessor_sha256,
        feature_names=matrix.output_feature_names,
        references=tuple(references),
    )
    return FittedEmpiricalTailDetector(
        training_matrix_sha256=matrix.evidence_sha256,
        preprocessor_sha256=matrix.preprocessor_sha256,
        output_feature_names=matrix.output_feature_names,
        training_row_count=matrix.row_count,
        references=tuple(references),
        fitted_model_sha256=model_sha,
    )


def score_empirical_tail_detector(
    fitted: FittedEmpiricalTailDetector,
    matrix: PreprocessedMatrix,
) -> EmpiricalTailScoreBatch:
    """Score rows against frozen training ECDF references without refitting."""

    if not isinstance(fitted, FittedEmpiricalTailDetector):
        raise TypeError("fitted must be FittedEmpiricalTailDetector")
    if not isinstance(matrix, PreprocessedMatrix):
        raise TypeError("matrix must be PreprocessedMatrix")
    if matrix.preprocessor_sha256 != fitted.preprocessor_sha256:
        raise EmpiricalTailDetectorError(
            "scoring matrix preprocessor differs from fitted empirical-tail detector"
        )
    if matrix.output_feature_names != fitted.output_feature_names:
        raise EmpiricalTailDetectorError(
            "scoring matrix feature order differs from fitted empirical-tail detector"
        )

    surprises = _surprise_table(fitted.training_row_count)
    exact: list[EmpiricalTailRowEvidence] = []
    for identity, values in zip(matrix.row_identities, matrix.values):
        left_total = Decimal(0)
        right_total = Decimal(0)
        skew_total = Decimal(0)
        for value, reference in zip(values, fitted.references):
            left_count = bisect_right(reference.sorted_training_values, value)
            right_count = (
                reference.training_count
                - bisect_left(reference.sorted_training_values, value)
            )
            left_surprise = surprises[max(1, left_count)]
            right_surprise = surprises[max(1, right_count)]
            left_total += left_surprise
            right_total += right_surprise
            skew_total += (
                left_surprise
                if reference.skew_direction is SkewDirection.LEFT
                else right_surprise
            )

        exact.append(
            EmpiricalTailRowEvidence(
                transaction_id=identity[0],
                award_id=identity[1],
                left_score=left_total,
                right_score=right_total,
                skew_selected_score=skew_total,
                final_score=max(left_total, right_total, skew_total),
            )
        )

    raw_scores = tuple(
        _decimal_to_float(item.final_score, "empirical_tail_score")
        for item in exact
    )
    generic = build_detector_scores(
        matrix.row_identities,
        raw_scores,
        orientation=ScoreOrientation.HIGHER_IS_MORE_ANOMALOUS,
    )
    generic_batch = DetectorScoreBatch(
        detector_name=ALGORITHM_NAME,
        detector_family=ALGORITHM_FAMILY,
        implementation_name="procurelens",
        implementation_version=str(ALGORITHM_REVISION),
        score_orientation=ScoreOrientation.HIGHER_IS_MORE_ANOMALOUS,
        config_sha256=fitted.config_sha256,
        fitted_model_sha256=fitted.fitted_model_sha256,
        training_matrix_sha256=fitted.training_matrix_sha256,
        scoring_matrix_sha256=matrix.evidence_sha256,
        preprocessor_sha256=fitted.preprocessor_sha256,
        output_feature_names=fitted.output_feature_names,
        scores=generic,
    )
    return EmpiricalTailScoreBatch(
        fitted_model_sha256=fitted.fitted_model_sha256,
        training_matrix_sha256=fitted.training_matrix_sha256,
        scoring_matrix_sha256=matrix.evidence_sha256,
        preprocessor_sha256=fitted.preprocessor_sha256,
        row_evidence=tuple(exact),
        generic_scores=generic_batch,
    )


def _skew_direction(values: tuple[Decimal, ...]) -> SkewDirection:
    """Use sign of the third central moment; zero/undefined variance ties right."""

    if not values:
        raise EmpiricalTailDetectorError(
            "cannot compute skew direction from empty values"
        )
    with localcontext() as context:
        context.prec = DECIMAL_WORKING_PRECISION
        mean = sum(values, Decimal(0)) / Decimal(len(values))
        third_moment_numerator = sum(
            ((value - mean) ** 3 for value in values),
            Decimal(0),
        )
    return (
        SkewDirection.LEFT
        if third_moment_numerator < 0
        else SkewDirection.RIGHT
    )


def _surprise_table(training_count: int) -> tuple[Decimal, ...]:
    if (
        isinstance(training_count, bool)
        or not isinstance(training_count, int)
        or training_count < 2
    ):
        raise EmpiricalTailDetectorError(
            "training_count must be integer of at least two"
        )
    result = [Decimal(0)] * (training_count + 1)
    denominator = Decimal(training_count)
    with localcontext() as context:
        context.prec = DECIMAL_WORKING_PRECISION
        for count in range(1, training_count + 1):
            probability = Decimal(count) / denominator
            result[count] = -probability.ln()
    return tuple(result)


def _model_fingerprint(
    *,
    matrix_sha256: str,
    preprocessor_sha256: str,
    feature_names: tuple[str, ...],
    references: tuple[FeatureTailReference, ...],
) -> str:
    return _digest(
        {
            "algorithm_config_sha256": empirical_tail_config_sha256(),
            "training_matrix_sha256": matrix_sha256,
            "preprocessor_sha256": preprocessor_sha256,
            "feature_names": list(feature_names),
            "references": [
                {
                    "feature_name": reference.feature_name,
                    "skew_direction": reference.skew_direction.value,
                    "reference_sha256": reference.sha256_hex,
                }
                for reference in references
            ],
        }
    )


def _decimal_to_float(value: Decimal, name: str) -> float:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise EmpiricalTailDetectorError(
            f"{name} must be finite Decimal before float conversion"
        )
    converted = float(value)
    if not math.isfinite(converted):
        raise EmpiricalTailDetectorError(
            f"{name} cannot be represented as finite float"
        )
    return converted


def _digest_hex(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise EmpiricalTailDetectorError(
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
