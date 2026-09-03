"""Transparent 0-100 anomaly review-priority scores for ProcureLens.

Converts calibrated ensemble empirical-position evidence into the client-facing
0-100 review scale using only a linear display transform. The score is not a
probability of fraud, corruption, collusion, or misconduct and is not a calibrated
probability of any outcome.

Uncertainty, detector disagreement, feature completeness, and run-to-run
stability remain separate evidence fields. They never silently increase or
decrease the 0-100 score in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from typing import Any

from procurelens.review.evidence import (
    ReviewEvidenceBatch,
    ReviewEvidenceRow,
)


class ReviewScoreError(ValueError):
    """Raised when review-priority score evidence is inconsistent."""


SCORE_MINIMUM = Decimal(0)
SCORE_MAXIMUM = Decimal(100)
SCORE_SEMANTICS = (
    "relative_anomaly_review_priority_not_probability_or_fraud_determination"
)


@dataclass(frozen=True, slots=True)
class ReviewPriorityScore:
    transaction_id: str
    award_id: str
    source_review_evidence_sha256: str

    review_score_lower: Decimal
    review_score: Decimal
    review_score_upper: Decimal

    detector_disagreement_points: Decimal
    feature_completeness_fraction: Decimal

    stability_available: bool
    stability_position_span_points: Decimal | None
    stability_median_absolute_deviation_points: Decimal | None

    def __post_init__(self) -> None:
        for field_name in ("transaction_id", "award_id"):
            text = getattr(self, field_name).strip()
            if not text:
                raise ReviewScoreError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)
        object.__setattr__(
            self,
            "source_review_evidence_sha256",
            _digest_hex(
                self.source_review_evidence_sha256,
                "source_review_evidence_sha256",
            ),
        )

        lower = self.review_score_lower
        score = self.review_score
        upper = self.review_score_upper
        for field_name, value in (
            ("review_score_lower", lower),
            ("review_score", score),
            ("review_score_upper", upper),
            ("detector_disagreement_points", self.detector_disagreement_points),
        ):
            _score_value(value, field_name)
        if not lower <= score <= upper:
            raise ReviewScoreError(
                "review-priority score interval is inconsistent"
            )

        _fraction(
            self.feature_completeness_fraction,
            "feature_completeness_fraction",
        )
        if not isinstance(self.stability_available, bool):
            raise ReviewScoreError("stability_available must be bool")

        optional = (
            self.stability_position_span_points,
            self.stability_median_absolute_deviation_points,
        )
        if self.stability_available:
            if any(value is None for value in optional):
                raise ReviewScoreError(
                    "available stability requires complete stability point measures"
                )
            for field_name, value in (
                (
                    "stability_position_span_points",
                    self.stability_position_span_points,
                ),
                (
                    "stability_median_absolute_deviation_points",
                    self.stability_median_absolute_deviation_points,
                ),
            ):
                assert value is not None
                _score_value(value, field_name)
        elif any(value is not None for value in optional):
            raise ReviewScoreError(
                "unavailable stability cannot carry stability point measures"
            )

    @property
    def identity(self) -> tuple[str, str]:
        return self.transaction_id, self.award_id

    @property
    def semantics(self) -> str:
        return SCORE_SEMANTICS

    @property
    def evidence_sha256(self) -> str:
        return _digest(self.as_dict(include_sha=False))

    def as_dict(self, *, include_sha: bool = True) -> dict[str, Any]:
        result = {
            "transaction_id": self.transaction_id,
            "award_id": self.award_id,
            "source_review_evidence_sha256":
                self.source_review_evidence_sha256,
            "review_score_lower": str(self.review_score_lower),
            "review_score": str(self.review_score),
            "review_score_upper": str(self.review_score_upper),
            "detector_disagreement_points":
                str(self.detector_disagreement_points),
            "feature_completeness_fraction":
                str(self.feature_completeness_fraction),
            "stability_available": self.stability_available,
            "stability_position_span_points": _decimal_text(
                self.stability_position_span_points
            ),
            "stability_median_absolute_deviation_points": _decimal_text(
                self.stability_median_absolute_deviation_points
            ),
            "score_semantics": SCORE_SEMANTICS,
        }
        if include_sha:
            result["evidence_sha256"] = self.evidence_sha256
        return result


@dataclass(frozen=True, slots=True)
class ReviewScoreBatch:
    source_review_batch_sha256: str
    score_semantics: str
    rows: tuple[ReviewPriorityScore, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_review_batch_sha256",
            _digest_hex(
                self.source_review_batch_sha256,
                "source_review_batch_sha256",
            ),
        )
        semantics = self.score_semantics.strip()
        if semantics != SCORE_SEMANTICS:
            raise ReviewScoreError(
                "review score batch uses unsupported score semantics"
            )
        object.__setattr__(self, "score_semantics", semantics)

        rows = tuple(self.rows)
        if not rows:
            raise ReviewScoreError(
                "review score batch requires at least one row"
            )
        identities = tuple(row.identity for row in rows)
        if len(identities) != len(set(identities)):
            raise ReviewScoreError(
                "review score batch contains duplicate row identities"
            )
        object.__setattr__(self, "rows", rows)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def row_identities(self) -> tuple[tuple[str, str], ...]:
        return tuple(row.identity for row in self.rows)

    @property
    def evidence_sha256(self) -> str:
        return _digest(
            {
                "source_review_batch_sha256":
                    self.source_review_batch_sha256,
                "score_semantics": self.score_semantics,
                "rows": [row.evidence_sha256 for row in self.rows],
            }
        )


def build_review_scores(
    evidence: ReviewEvidenceBatch,
) -> ReviewScoreBatch:
    """Create an auditable linear 0-100 view of calibrated review evidence."""

    if not isinstance(evidence, ReviewEvidenceBatch):
        raise TypeError("evidence must be ReviewEvidenceBatch")

    rows = tuple(_score_row(row) for row in evidence.rows)
    return ReviewScoreBatch(
        source_review_batch_sha256=evidence.evidence_sha256,
        score_semantics=SCORE_SEMANTICS,
        rows=rows,
    )


def _score_row(row: ReviewEvidenceRow) -> ReviewPriorityScore:
    if not isinstance(row, ReviewEvidenceRow):
        raise TypeError("row must be ReviewEvidenceRow")

    total_features = row.available_feature_count + row.missing_feature_count
    if total_features < 1:
        raise ReviewScoreError(
            "review evidence row has no candidate features"
        )
    completeness = (
        Decimal(row.available_feature_count) / Decimal(total_features)
    )

    return ReviewPriorityScore(
        transaction_id=row.transaction_id,
        award_id=row.award_id,
        source_review_evidence_sha256=row.evidence_sha256,
        review_score_lower=_to_score(row.ensemble_lower_fraction),
        review_score=_to_score(row.ensemble_midpoint_fraction),
        review_score_upper=_to_score(row.ensemble_upper_fraction),
        detector_disagreement_points=_to_score(
            row.detector_disagreement_span
        ),
        feature_completeness_fraction=completeness,
        stability_available=row.stability_available,
        stability_position_span_points=(
            None
            if row.stability_position_span is None
            else _to_score(row.stability_position_span)
        ),
        stability_median_absolute_deviation_points=(
            None
            if row.stability_median_absolute_deviation is None
            else _to_score(row.stability_median_absolute_deviation)
        ),
    )


def _to_score(fraction: Decimal) -> Decimal:
    _fraction(fraction, "score source fraction")
    return fraction * SCORE_MAXIMUM


def _fraction(value: Decimal, name: str) -> None:
    if (
        not isinstance(value, Decimal)
        or not value.is_finite()
        or value < Decimal(0)
        or value > Decimal(1)
    ):
        raise ReviewScoreError(
            f"{name} must be finite Decimal in [0, 1]"
        )


def _score_value(value: Decimal, name: str) -> None:
    if (
        not isinstance(value, Decimal)
        or not value.is_finite()
        or value < SCORE_MINIMUM
        or value > SCORE_MAXIMUM
    ):
        raise ReviewScoreError(
            f"{name} must be finite Decimal in [0, 100]"
        )


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _digest_hex(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ReviewScoreError(
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
