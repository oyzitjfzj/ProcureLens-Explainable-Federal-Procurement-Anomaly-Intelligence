"""Faithful investigator-facing explanations for ProcureLens review decisions.

Builds deterministic explanations from evidence that already exists in the
pipeline. Explanations report review priority, uncertainty, selection policy,
stability, and an explicit caller-selected set of catalog-backed feature facts.

Feature facts are described as observations/model inputs, never as causal
contributions to an anomaly score. This module does not run post-hoc attribution,
invent missing evidence, diagnose fraud, or claim that a red flag proves
misconduct.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from typing import Any, Iterable

from procurelens.features.catalog import (
    FeatureCatalog,
    FeatureCatalogEntry,
    FeatureSource,
    feature_catalog,
)
from procurelens.model.feature_row import (
    CandidateFeatureRow,
    CandidateFeatureValue,
)
from procurelens.review.evidence import (
    ReviewEvidenceBatch,
    ReviewEvidenceRow,
)
from procurelens.review.policy import (
    ReviewSelectionBatch,
    ReviewSelectionDecision,
)
from procurelens.review.score import (
    ReviewPriorityScore,
    ReviewScoreBatch,
)


class ExplanationError(ValueError):
    """Raised when explanation evidence or provenance is inconsistent."""


EXPLANATION_SEMANTICS = (
    "anomaly_review_evidence_not_causal_attribution_or_fraud_determination"
)


@dataclass(frozen=True, slots=True)
class ExplanationSpec:
    """Explicit catalog-pinned feature facts to surface to an investigator."""

    name: str
    description: str
    catalog_sha256: str
    feature_names: tuple[str, ...]
    include_missing_features: bool

    def __post_init__(self) -> None:
        for field_name in ("name", "description"):
            text = getattr(self, field_name).strip()
            if not text:
                raise ExplanationError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)
        object.__setattr__(
            self,
            "catalog_sha256",
            _digest_hex(self.catalog_sha256, "catalog_sha256"),
        )
        names = tuple(_feature_name(name) for name in self.feature_names)
        if not names:
            raise ExplanationError(
                "explanation spec requires at least one feature name"
            )
        if len(names) != len(set(names)):
            raise ExplanationError(
                "explanation spec contains duplicate feature names"
            )
        object.__setattr__(self, "feature_names", names)
        if not isinstance(self.include_missing_features, bool):
            raise ExplanationError("include_missing_features must be bool")

    @property
    def sha256_hex(self) -> str:
        return _digest(self.as_dict(include_sha=False))

    def as_dict(self, *, include_sha: bool = True) -> dict[str, Any]:
        result = {
            "name": self.name,
            "description": self.description,
            "catalog_sha256": self.catalog_sha256,
            "feature_names": list(self.feature_names),
            "include_missing_features": self.include_missing_features,
        }
        if include_sha:
            result["sha256"] = self.sha256_hex
        return result


@dataclass(frozen=True, slots=True)
class ExplanationFeatureFact:
    source: FeatureSource
    family: str
    name: str
    description: str
    value: Decimal | None
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", FeatureSource(self.source))
        for field_name in ("family", "name", "description"):
            text = getattr(self, field_name).strip()
            if not text:
                raise ExplanationError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)

        if self.value is None:
            reason = (
                None
                if self.unavailable_reason is None
                else self.unavailable_reason.strip()
            )
            if not reason:
                raise ExplanationError(
                    f"{self.name}: unavailable feature fact requires a reason"
                )
            object.__setattr__(self, "unavailable_reason", reason)
        else:
            if (
                not isinstance(self.value, Decimal)
                or not self.value.is_finite()
            ):
                raise ExplanationError(
                    f"{self.name}: feature fact value must be finite Decimal"
                )
            if self.unavailable_reason is not None:
                raise ExplanationError(
                    f"{self.name}: available feature fact cannot carry a reason"
                )

    @property
    def available(self) -> bool:
        return self.value is not None

    @property
    def narrative(self) -> str:
        if self.value is None:
            return (
                f"{self.description} Evidence unavailable: "
                f"{self.unavailable_reason}."
            )
        return f"{self.description} Observed value: {self.value}."

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "family": self.family,
            "name": self.name,
            "description": self.description,
            "value": None if self.value is None else str(self.value),
            "unavailable_reason": self.unavailable_reason,
            "narrative": self.narrative,
        }


@dataclass(frozen=True, slots=True)
class ReviewExplanation:
    transaction_id: str
    award_id: str
    explanation_spec_sha256: str
    feature_row_evidence_sha256: str
    review_evidence_sha256: str
    review_score_evidence_sha256: str
    review_selection_evidence_sha256: str

    review_score_lower: Decimal
    review_score: Decimal
    review_score_upper: Decimal
    rank_lower: int
    rank_upper: int
    flagged_for_review: bool
    selection_reason: str

    detector_disagreement_points: Decimal
    feature_completeness_fraction: Decimal
    stability_available: bool
    stability_position_span_points: Decimal | None
    stability_median_absolute_deviation_points: Decimal | None

    feature_facts: tuple[ExplanationFeatureFact, ...]

    def __post_init__(self) -> None:
        for field_name in ("transaction_id", "award_id"):
            text = getattr(self, field_name).strip()
            if not text:
                raise ExplanationError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)
        for field_name in (
            "explanation_spec_sha256",
            "feature_row_evidence_sha256",
            "review_evidence_sha256",
            "review_score_evidence_sha256",
            "review_selection_evidence_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest_hex(getattr(self, field_name), field_name),
            )
        for field_name in (
            "review_score_lower",
            "review_score",
            "review_score_upper",
            "detector_disagreement_points",
        ):
            _score_value(getattr(self, field_name), field_name)
        if not self.review_score_lower <= self.review_score <= self.review_score_upper:
            raise ExplanationError("review score interval is inconsistent")
        _fraction(
            self.feature_completeness_fraction,
            "feature_completeness_fraction",
        )
        for field_name in ("rank_lower", "rank_upper"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ExplanationError(
                    f"{field_name} must be a positive integer"
                )
        if self.rank_lower > self.rank_upper:
            raise ExplanationError("review rank interval is inconsistent")
        if not isinstance(self.flagged_for_review, bool):
            raise ExplanationError("flagged_for_review must be bool")
        reason = self.selection_reason.strip()
        if not reason:
            raise ExplanationError("selection_reason must not be blank")
        object.__setattr__(self, "selection_reason", reason)
        if not isinstance(self.stability_available, bool):
            raise ExplanationError("stability_available must be bool")

        optional = (
            self.stability_position_span_points,
            self.stability_median_absolute_deviation_points,
        )
        if self.stability_available:
            if any(value is None for value in optional):
                raise ExplanationError(
                    "available stability requires complete point measurements"
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
            raise ExplanationError(
                "unavailable stability cannot carry stability measurements"
            )

        facts = tuple(self.feature_facts)
        names = tuple(fact.name for fact in facts)
        if len(names) != len(set(names)):
            raise ExplanationError(
                "explanation feature facts contain duplicate names"
            )
        object.__setattr__(self, "feature_facts", facts)

    @property
    def identity(self) -> tuple[str, str]:
        return self.transaction_id, self.award_id

    @property
    def semantics(self) -> str:
        return EXPLANATION_SEMANTICS

    @property
    def summary_text(self) -> str:
        flag_text = (
            "Flagged for investigator review"
            if self.flagged_for_review
            else "Not flagged for investigator review"
        )
        rank_text = (
            str(self.rank_lower)
            if self.rank_lower == self.rank_upper
            else f"{self.rank_lower}-{self.rank_upper}"
        )
        stability_text = (
            " Run-to-run stability evidence was not attached."
            if not self.stability_available
            else (
                " Run-to-run position span: "
                f"{self.stability_position_span_points} points; "
                "median absolute deviation: "
                f"{self.stability_median_absolute_deviation_points} points."
            )
        )
        return (
            f"Review priority {self.review_score}/100 "
            f"(empirical interval {self.review_score_lower}-"
            f"{self.review_score_upper}); {flag_text}. "
            f"Tie-aware review rank: {rank_text}. "
            f"Selection reason: {self.selection_reason}. "
            f"Detector disagreement: {self.detector_disagreement_points} points. "
            f"Feature completeness: {self.feature_completeness_fraction}."
            f"{stability_text} "
            "This is anomaly-review evidence, not a fraud, corruption, "
            "or collusion determination."
        )

    @property
    def evidence_sha256(self) -> str:
        return _digest(self.as_dict(include_sha=False))

    def as_dict(self, *, include_sha: bool = True) -> dict[str, Any]:
        result = {
            "transaction_id": self.transaction_id,
            "award_id": self.award_id,
            "explanation_spec_sha256": self.explanation_spec_sha256,
            "feature_row_evidence_sha256": self.feature_row_evidence_sha256,
            "review_evidence_sha256": self.review_evidence_sha256,
            "review_score_evidence_sha256": self.review_score_evidence_sha256,
            "review_selection_evidence_sha256":
                self.review_selection_evidence_sha256,
            "review_score_lower": str(self.review_score_lower),
            "review_score": str(self.review_score),
            "review_score_upper": str(self.review_score_upper),
            "rank_lower": self.rank_lower,
            "rank_upper": self.rank_upper,
            "flagged_for_review": self.flagged_for_review,
            "selection_reason": self.selection_reason,
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
            "summary_text": self.summary_text,
            "explanation_semantics": EXPLANATION_SEMANTICS,
            "feature_facts": [fact.as_dict() for fact in self.feature_facts],
        }
        if include_sha:
            result["evidence_sha256"] = self.evidence_sha256
        return result


@dataclass(frozen=True, slots=True)
class ExplanationBatch:
    spec: ExplanationSpec
    feature_catalog_sha256: str
    review_evidence_batch_sha256: str
    review_score_batch_sha256: str
    review_selection_batch_sha256: str
    rows: tuple[ReviewExplanation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ExplanationSpec):
            raise TypeError("spec must be ExplanationSpec")
        for field_name in (
            "feature_catalog_sha256",
            "review_evidence_batch_sha256",
            "review_score_batch_sha256",
            "review_selection_batch_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest_hex(getattr(self, field_name), field_name),
            )
        if self.feature_catalog_sha256 != self.spec.catalog_sha256:
            raise ExplanationError(
                "explanation batch catalog differs from explanation spec"
            )
        rows = tuple(self.rows)
        if not rows:
            raise ExplanationError("explanation batch requires at least one row")
        identities = tuple(row.identity for row in rows)
        if len(identities) != len(set(identities)):
            raise ExplanationError(
                "explanation batch contains duplicate row identities"
            )
        if any(
            row.explanation_spec_sha256 != self.spec.sha256_hex
            for row in rows
        ):
            raise ExplanationError(
                "explanation row spec fingerprint differs from batch spec"
            )
        object.__setattr__(self, "rows", rows)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def evidence_sha256(self) -> str:
        return _digest(
            {
                "explanation_spec_sha256": self.spec.sha256_hex,
                "feature_catalog_sha256": self.feature_catalog_sha256,
                "review_evidence_batch_sha256":
                    self.review_evidence_batch_sha256,
                "review_score_batch_sha256": self.review_score_batch_sha256,
                "review_selection_batch_sha256":
                    self.review_selection_batch_sha256,
                "rows": [row.evidence_sha256 for row in self.rows],
            }
        )


def build_explanations(
    *,
    feature_rows: Iterable[CandidateFeatureRow],
    review_evidence: ReviewEvidenceBatch,
    review_scores: ReviewScoreBatch,
    review_selection: ReviewSelectionBatch,
    spec: ExplanationSpec,
    catalog: FeatureCatalog | None = None,
) -> ExplanationBatch:
    """Build faithful deterministic explanations over one exact row population."""

    if isinstance(feature_rows, (str, bytes)):
        raise ExplanationError(
            "feature_rows must be an iterable of CandidateFeatureRow"
        )
    features = tuple(feature_rows)
    if not features:
        raise ExplanationError("at least one candidate feature row is required")
    if any(not isinstance(row, CandidateFeatureRow) for row in features):
        raise TypeError("all feature_rows must be CandidateFeatureRow")
    if not isinstance(review_evidence, ReviewEvidenceBatch):
        raise TypeError("review_evidence must be ReviewEvidenceBatch")
    if not isinstance(review_scores, ReviewScoreBatch):
        raise TypeError("review_scores must be ReviewScoreBatch")
    if not isinstance(review_selection, ReviewSelectionBatch):
        raise TypeError("review_selection must be ReviewSelectionBatch")
    if not isinstance(spec, ExplanationSpec):
        raise TypeError("spec must be ExplanationSpec")

    catalog = feature_catalog() if catalog is None else catalog
    if not isinstance(catalog, FeatureCatalog):
        raise TypeError("catalog must be FeatureCatalog")
    if catalog.sha256_hex != spec.catalog_sha256:
        raise ExplanationError(
            "explanation spec catalog fingerprint differs from current catalog"
        )

    entries = catalog.select(spec.feature_names)
    identities = tuple((row.transaction_id, row.award_id) for row in features)
    if identities != review_evidence.row_identities:
        raise ExplanationError(
            "feature rows differ from review-evidence row population"
        )
    if identities != review_scores.row_identities:
        raise ExplanationError(
            "feature rows differ from review-score row population"
        )
    if identities != tuple(
        decision.identity for decision in review_selection.decisions
    ):
        raise ExplanationError(
            "feature rows differ from review-selection row population"
        )

    if review_scores.source_review_batch_sha256 != review_evidence.evidence_sha256:
        raise ExplanationError(
            "review scores were not produced from the supplied review evidence"
        )
    if (
        review_selection.source_score_batch_sha256
        != review_scores.evidence_sha256
    ):
        raise ExplanationError(
            "review selection was not produced from the supplied review scores"
        )
    catalog_hashes = {row.catalog_sha256 for row in features}
    if catalog_hashes != {catalog.sha256_hex}:
        raise ExplanationError(
            "candidate feature rows differ from explanation catalog"
        )

    rows: list[ReviewExplanation] = []
    for feature_row, evidence_row, score_row, decision in zip(
        features,
        review_evidence.rows,
        review_scores.rows,
        review_selection.decisions,
    ):
        _validate_row_chain(feature_row, evidence_row, score_row, decision)
        facts = _feature_facts(
            feature_row,
            entries,
            include_missing=spec.include_missing_features,
        )
        rows.append(
            ReviewExplanation(
                transaction_id=feature_row.transaction_id,
                award_id=feature_row.award_id,
                explanation_spec_sha256=spec.sha256_hex,
                feature_row_evidence_sha256=feature_row.evidence_sha256,
                review_evidence_sha256=evidence_row.evidence_sha256,
                review_score_evidence_sha256=score_row.evidence_sha256,
                review_selection_evidence_sha256=decision.evidence_sha256,
                review_score_lower=score_row.review_score_lower,
                review_score=score_row.review_score,
                review_score_upper=score_row.review_score_upper,
                rank_lower=decision.rank_lower,
                rank_upper=decision.rank_upper,
                flagged_for_review=decision.flagged_for_review,
                selection_reason=decision.selection_reason.value,
                detector_disagreement_points=
                    score_row.detector_disagreement_points,
                feature_completeness_fraction=
                    score_row.feature_completeness_fraction,
                stability_available=score_row.stability_available,
                stability_position_span_points=
                    score_row.stability_position_span_points,
                stability_median_absolute_deviation_points=
                    score_row.stability_median_absolute_deviation_points,
                feature_facts=facts,
            )
        )

    return ExplanationBatch(
        spec=spec,
        feature_catalog_sha256=catalog.sha256_hex,
        review_evidence_batch_sha256=review_evidence.evidence_sha256,
        review_score_batch_sha256=review_scores.evidence_sha256,
        review_selection_batch_sha256=review_selection.evidence_sha256,
        rows=tuple(rows),
    )


def _validate_row_chain(
    feature_row: CandidateFeatureRow,
    evidence_row: ReviewEvidenceRow,
    score_row: ReviewPriorityScore,
    decision: ReviewSelectionDecision,
) -> None:
    identity = (feature_row.transaction_id, feature_row.award_id)
    if (
        evidence_row.identity != identity
        or score_row.identity != identity
        or decision.identity != identity
    ):
        raise ExplanationError(
            "feature/review/score/selection row identities differ"
        )
    if evidence_row.feature_row_evidence_sha256 != feature_row.evidence_sha256:
        raise ExplanationError(
            "review evidence does not reference supplied feature row"
        )
    if score_row.source_review_evidence_sha256 != evidence_row.evidence_sha256:
        raise ExplanationError(
            "review score does not reference supplied review evidence row"
        )
    if decision.source_score_evidence_sha256 != score_row.evidence_sha256:
        raise ExplanationError(
            "review selection does not reference supplied score row"
        )
    if decision.review_score != score_row.review_score:
        raise ExplanationError(
            "review selection score differs from supplied score row"
        )


def _feature_facts(
    row: CandidateFeatureRow,
    entries: tuple[FeatureCatalogEntry, ...],
    *,
    include_missing: bool,
) -> tuple[ExplanationFeatureFact, ...]:
    facts: list[ExplanationFeatureFact] = []
    for entry in entries:
        candidate = row.get(entry.name)
        if candidate.source is not entry.source:
            raise ExplanationError(
                f"{entry.name}: candidate source differs from catalog"
            )
        if candidate.value is None and not include_missing:
            continue
        facts.append(_fact(entry, candidate))
    return tuple(facts)


def _fact(
    entry: FeatureCatalogEntry,
    value: CandidateFeatureValue,
) -> ExplanationFeatureFact:
    return ExplanationFeatureFact(
        source=entry.source,
        family=entry.family,
        name=entry.name,
        description=entry.description,
        value=value.value,
        unavailable_reason=value.unavailable_reason,
    )


def _feature_name(value: str) -> str:
    if not isinstance(value, str):
        raise ExplanationError("feature names must be strings")
    text = value.strip()
    if not text:
        raise ExplanationError("feature names must not be blank")
    return text


def _fraction(value: Decimal, name: str) -> None:
    if (
        not isinstance(value, Decimal)
        or not value.is_finite()
        or value < Decimal(0)
        or value > Decimal(1)
    ):
        raise ExplanationError(
            f"{name} must be finite Decimal in [0, 1]"
        )


def _score_value(value: Decimal, name: str) -> None:
    if (
        not isinstance(value, Decimal)
        or not value.is_finite()
        or value < Decimal(0)
        or value > Decimal(100)
    ):
        raise ExplanationError(
            f"{name} must be finite Decimal in [0, 100]"
        )


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _digest_hex(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ExplanationError(
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
