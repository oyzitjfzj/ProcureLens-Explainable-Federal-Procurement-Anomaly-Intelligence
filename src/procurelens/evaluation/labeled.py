"""Label-aware evaluation for ProcureLens anomaly review rankings.

Evaluates transparent review-priority scores only against caller-supplied,
explicit ground-truth labels. Unknown/unreviewed procurement rows are never
silently treated as negatives.

Ranking metrics are threshold-free and tie-aware. AUROC uses the equivalent
positive-vs-negative pair interpretation with half credit for equal scores.
Average precision advances at distinct score thresholds, so a tied score group
is evaluated as a group rather than by an arbitrary row ordering.

Operational review-queue metrics evaluate the exact supplied review-selection
artifact. Unknown labels inside the selected queue remain visible and are
excluded from known-label precision denominators rather than counted as errors.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import json
from typing import Any

from procurelens.review.policy import ReviewSelectionBatch
from procurelens.review.score import ReviewScoreBatch


class LabeledEvaluationError(ValueError):
    """Raised when labeled evaluation evidence is inconsistent."""


class GroundTruthState(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class GroundTruthLabel:
    transaction_id: str
    award_id: str
    state: GroundTruthState
    evidence_reference: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("transaction_id", "award_id"):
            text = getattr(self, field_name).strip()
            if not text:
                raise LabeledEvaluationError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)
        object.__setattr__(self, "state", GroundTruthState(self.state))
        if self.evidence_reference is not None:
            text = self.evidence_reference.strip()
            object.__setattr__(self, "evidence_reference", text or None)

    @property
    def identity(self) -> tuple[str, str]:
        return self.transaction_id, self.award_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "award_id": self.award_id,
            "state": self.state.value,
            "evidence_reference": self.evidence_reference,
        }


@dataclass(frozen=True, slots=True)
class GroundTruthSet:
    """Explicit label semantics and exact population used for evaluation."""

    name: str
    source_name: str
    source_sha256: str
    positive_class_description: str
    negative_class_description: str
    labels: tuple[GroundTruthLabel, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "source_name",
            "positive_class_description",
            "negative_class_description",
        ):
            text = getattr(self, field_name).strip()
            if not text:
                raise LabeledEvaluationError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)
        object.__setattr__(
            self,
            "source_sha256",
            _digest_hex(self.source_sha256, "source_sha256"),
        )
        labels = tuple(self.labels)
        if not labels:
            raise LabeledEvaluationError(
                "ground-truth set requires at least one label"
            )
        identities = tuple(label.identity for label in labels)
        if len(identities) != len(set(identities)):
            raise LabeledEvaluationError(
                "ground-truth labels contain duplicate identities"
            )
        object.__setattr__(self, "labels", labels)

    @property
    def row_count(self) -> int:
        return len(self.labels)

    @property
    def known_count(self) -> int:
        return sum(
            label.state is not GroundTruthState.UNKNOWN for label in self.labels
        )

    @property
    def unknown_count(self) -> int:
        return self.row_count - self.known_count

    @property
    def positive_count(self) -> int:
        return sum(
            label.state is GroundTruthState.POSITIVE for label in self.labels
        )

    @property
    def negative_count(self) -> int:
        return sum(
            label.state is GroundTruthState.NEGATIVE for label in self.labels
        )

    @property
    def sha256_hex(self) -> str:
        return _digest(
            {
                "name": self.name,
                "source_name": self.source_name,
                "source_sha256": self.source_sha256,
                "positive_class_description": self.positive_class_description,
                "negative_class_description": self.negative_class_description,
                "labels": [
                    label.as_dict()
                    for label in sorted(
                        self.labels, key=lambda item: item.identity
                    )
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class OptionalMetric:
    value: Decimal | None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.value is None:
            reason = (
                None
                if self.unavailable_reason is None
                else self.unavailable_reason.strip()
            )
            if not reason:
                raise LabeledEvaluationError(
                    "unavailable metric requires a reason"
                )
            object.__setattr__(self, "unavailable_reason", reason)
        else:
            if (
                not isinstance(self.value, Decimal)
                or not self.value.is_finite()
            ):
                raise LabeledEvaluationError(
                    "metric value must be finite Decimal or None"
                )
            if self.unavailable_reason is not None:
                raise LabeledEvaluationError(
                    "available metric cannot carry unavailable reason"
                )

    def as_dict(self) -> dict[str, str | None]:
        return {
            "value": None if self.value is None else str(self.value),
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True, slots=True)
class ReviewQueueEvaluation:
    source_selection_sha256: str
    selected_count: int
    selected_known_count: int
    selected_unknown_count: int
    selected_positive_count: int
    selected_negative_count: int
    precision_among_known_selected: OptionalMetric
    recall_of_known_positives: OptionalMetric
    lift_over_known_prevalence: OptionalMetric

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_selection_sha256",
            _digest_hex(
                self.source_selection_sha256, "source_selection_sha256"
            ),
        )
        for field_name in (
            "selected_count",
            "selected_known_count",
            "selected_unknown_count",
            "selected_positive_count",
            "selected_negative_count",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise LabeledEvaluationError(
                    f"{field_name} must be a non-negative integer"
                )
        if (
            self.selected_known_count + self.selected_unknown_count
            != self.selected_count
        ):
            raise LabeledEvaluationError(
                "selected known/unknown counts do not sum to selected_count"
            )
        if (
            self.selected_positive_count + self.selected_negative_count
            != self.selected_known_count
        ):
            raise LabeledEvaluationError(
                "selected positive/negative counts do not sum to known selected"
            )
        for field_name in (
            "precision_among_known_selected",
            "recall_of_known_positives",
            "lift_over_known_prevalence",
        ):
            if not isinstance(getattr(self, field_name), OptionalMetric):
                raise TypeError(f"{field_name} must be OptionalMetric")

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_selection_sha256": self.source_selection_sha256,
            "selected_count": self.selected_count,
            "selected_known_count": self.selected_known_count,
            "selected_unknown_count": self.selected_unknown_count,
            "selected_positive_count": self.selected_positive_count,
            "selected_negative_count": self.selected_negative_count,
            "precision_among_known_selected":
                self.precision_among_known_selected.as_dict(),
            "recall_of_known_positives":
                self.recall_of_known_positives.as_dict(),
            "lift_over_known_prevalence":
                self.lift_over_known_prevalence.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class LabeledEvaluationReport:
    source_score_batch_sha256: str
    ground_truth_sha256: str
    row_count: int
    known_label_count: int
    unknown_label_count: int
    positive_count: int
    negative_count: int
    label_coverage: Decimal
    known_positive_prevalence: OptionalMetric
    auroc: OptionalMetric
    average_precision: OptionalMetric
    queue_evaluation: ReviewQueueEvaluation | None

    def __post_init__(self) -> None:
        for field_name in (
            "source_score_batch_sha256",
            "ground_truth_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest_hex(getattr(self, field_name), field_name),
            )
        for field_name in (
            "row_count",
            "known_label_count",
            "unknown_label_count",
            "positive_count",
            "negative_count",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise LabeledEvaluationError(
                    f"{field_name} must be a non-negative integer"
                )
        if self.row_count < 1:
            raise LabeledEvaluationError(
                "evaluation row_count must be positive"
            )
        if self.known_label_count + self.unknown_label_count != self.row_count:
            raise LabeledEvaluationError(
                "known/unknown label counts do not sum to row_count"
            )
        if self.positive_count + self.negative_count != self.known_label_count:
            raise LabeledEvaluationError(
                "positive/negative counts do not sum to known labels"
            )
        _fraction(self.label_coverage, "label_coverage")
        if self.label_coverage != (
            Decimal(self.known_label_count) / Decimal(self.row_count)
        ):
            raise LabeledEvaluationError("label_coverage is inconsistent")
        for field_name in (
            "known_positive_prevalence",
            "auroc",
            "average_precision",
        ):
            if not isinstance(getattr(self, field_name), OptionalMetric):
                raise TypeError(f"{field_name} must be OptionalMetric")
        if self.queue_evaluation is not None and not isinstance(
            self.queue_evaluation, ReviewQueueEvaluation
        ):
            raise TypeError(
                "queue_evaluation must be ReviewQueueEvaluation or None"
            )

    @property
    def evidence_sha256(self) -> str:
        return _digest(
            {
                "source_score_batch_sha256": self.source_score_batch_sha256,
                "ground_truth_sha256": self.ground_truth_sha256,
                "row_count": self.row_count,
                "known_label_count": self.known_label_count,
                "unknown_label_count": self.unknown_label_count,
                "positive_count": self.positive_count,
                "negative_count": self.negative_count,
                "label_coverage": str(self.label_coverage),
                "known_positive_prevalence":
                    self.known_positive_prevalence.as_dict(),
                "auroc": self.auroc.as_dict(),
                "average_precision": self.average_precision.as_dict(),
                "queue_evaluation": (
                    None
                    if self.queue_evaluation is None
                    else self.queue_evaluation.as_dict()
                ),
            }
        )


def evaluate_labeled_scores(
    scores: ReviewScoreBatch,
    ground_truth: GroundTruthSet,
    *,
    selection: ReviewSelectionBatch | None = None,
) -> LabeledEvaluationReport:
    """Evaluate one score ranking against explicit known/unknown labels."""

    if not isinstance(scores, ReviewScoreBatch):
        raise TypeError("scores must be ReviewScoreBatch")
    if not isinstance(ground_truth, GroundTruthSet):
        raise TypeError("ground_truth must be GroundTruthSet")
    if selection is not None and not isinstance(
        selection, ReviewSelectionBatch
    ):
        raise TypeError("selection must be ReviewSelectionBatch or None")

    labels_by_identity = {
        label.identity: label for label in ground_truth.labels
    }
    score_identities = scores.row_identities
    if set(labels_by_identity) != set(score_identities):
        raise LabeledEvaluationError(
            "ground-truth population must exactly match review-score identities"
        )

    ordered_labels = tuple(
        labels_by_identity[identity] for identity in score_identities
    )
    known_pairs = tuple(
        (score.review_score, label.state)
        for score, label in zip(scores.rows, ordered_labels)
        if label.state is not GroundTruthState.UNKNOWN
    )
    positive_count = sum(
        state is GroundTruthState.POSITIVE for _, state in known_pairs
    )
    negative_count = sum(
        state is GroundTruthState.NEGATIVE for _, state in known_pairs
    )
    known_count = len(known_pairs)
    unknown_count = scores.row_count - known_count

    prevalence = (
        OptionalMetric(
            Decimal(positive_count) / Decimal(known_count),
            None,
        )
        if known_count
        else OptionalMetric(None, "no_known_labels")
    )

    if positive_count and negative_count:
        auroc = OptionalMetric(_auroc(known_pairs), None)
        average_precision = OptionalMetric(
            _average_precision(known_pairs, positive_count), None
        )
    else:
        reason = "both_positive_and_negative_known_labels_required"
        auroc = OptionalMetric(None, reason)
        average_precision = OptionalMetric(None, reason)

    queue = None
    if selection is not None:
        if selection.source_score_batch_sha256 != scores.evidence_sha256:
            raise LabeledEvaluationError(
                "review selection was not produced from supplied score batch"
            )
        if (
            tuple(decision.identity for decision in selection.decisions)
            != score_identities
        ):
            raise LabeledEvaluationError(
                "review selection population differs from score population"
            )
        queue = _evaluate_queue(
            selection,
            labels_by_identity,
            positive_count=positive_count,
            known_count=known_count,
        )

    return LabeledEvaluationReport(
        source_score_batch_sha256=scores.evidence_sha256,
        ground_truth_sha256=ground_truth.sha256_hex,
        row_count=scores.row_count,
        known_label_count=known_count,
        unknown_label_count=unknown_count,
        positive_count=positive_count,
        negative_count=negative_count,
        label_coverage=Decimal(known_count) / Decimal(scores.row_count),
        known_positive_prevalence=prevalence,
        auroc=auroc,
        average_precision=average_precision,
        queue_evaluation=queue,
    )


def _auroc(
    known_pairs: tuple[tuple[Decimal, GroundTruthState], ...],
) -> Decimal:
    groups: dict[Decimal, list[GroundTruthState]] = {}
    for score, state in known_pairs:
        groups.setdefault(score, []).append(state)
    positives = sum(
        state is GroundTruthState.POSITIVE for _, state in known_pairs
    )
    negatives = sum(
        state is GroundTruthState.NEGATIVE for _, state in known_pairs
    )
    if not positives or not negatives:
        raise LabeledEvaluationError(
            "AUROC requires both known positive and negative labels"
        )

    negatives_below = 0
    favorable_pairs = Decimal(0)
    for score in sorted(groups):
        states = groups[score]
        group_positive = sum(
            state is GroundTruthState.POSITIVE for state in states
        )
        group_negative = sum(
            state is GroundTruthState.NEGATIVE for state in states
        )
        favorable_pairs += (
            Decimal(group_positive) * Decimal(negatives_below)
            + Decimal(group_positive * group_negative) / Decimal(2)
        )
        negatives_below += group_negative

    return favorable_pairs / Decimal(positives * negatives)


def _average_precision(
    known_pairs: tuple[tuple[Decimal, GroundTruthState], ...],
    total_positives: int,
) -> Decimal:
    if total_positives < 1:
        raise LabeledEvaluationError(
            "average precision requires known positives"
        )
    groups: dict[Decimal, list[GroundTruthState]] = {}
    for score, state in known_pairs:
        groups.setdefault(score, []).append(state)

    cumulative_positive = 0
    cumulative_total = 0
    result = Decimal(0)
    for score in sorted(groups, reverse=True):
        states = groups[score]
        group_positive = sum(
            state is GroundTruthState.POSITIVE for state in states
        )
        cumulative_positive += group_positive
        cumulative_total += len(states)
        if group_positive:
            precision = (
                Decimal(cumulative_positive) / Decimal(cumulative_total)
            )
            recall_increment = (
                Decimal(group_positive) / Decimal(total_positives)
            )
            result += precision * recall_increment
    return result


def _evaluate_queue(
    selection: ReviewSelectionBatch,
    labels_by_identity: dict[tuple[str, str], GroundTruthLabel],
    *,
    positive_count: int,
    known_count: int,
) -> ReviewQueueEvaluation:
    selected = tuple(
        decision
        for decision in selection.decisions
        if decision.flagged_for_review
    )
    states = tuple(
        labels_by_identity[decision.identity].state for decision in selected
    )
    selected_positive = sum(
        state is GroundTruthState.POSITIVE for state in states
    )
    selected_negative = sum(
        state is GroundTruthState.NEGATIVE for state in states
    )
    selected_unknown = sum(
        state is GroundTruthState.UNKNOWN for state in states
    )
    selected_known = selected_positive + selected_negative

    if selected_known:
        precision_value = (
            Decimal(selected_positive) / Decimal(selected_known)
        )
        precision = OptionalMetric(precision_value, None)
    else:
        precision_value = None
        precision = OptionalMetric(
            None, "no_known_labels_in_selected_review_queue"
        )

    recall = (
        OptionalMetric(
            Decimal(selected_positive) / Decimal(positive_count),
            None,
        )
        if positive_count
        else OptionalMetric(None, "no_known_positive_labels")
    )

    if precision_value is None:
        lift = OptionalMetric(
            None, "known_selected_precision_unavailable"
        )
    elif not known_count or not positive_count:
        lift = OptionalMetric(
            None, "known_positive_prevalence_unavailable"
        )
    else:
        prevalence = Decimal(positive_count) / Decimal(known_count)
        lift = OptionalMetric(precision_value / prevalence, None)

    return ReviewQueueEvaluation(
        source_selection_sha256=selection.evidence_sha256,
        selected_count=len(selected),
        selected_known_count=selected_known,
        selected_unknown_count=selected_unknown,
        selected_positive_count=selected_positive,
        selected_negative_count=selected_negative,
        precision_among_known_selected=precision,
        recall_of_known_positives=recall,
        lift_over_known_prevalence=lift,
    )


def _fraction(value: Decimal, name: str) -> None:
    if (
        not isinstance(value, Decimal)
        or not value.is_finite()
        or value < Decimal(0)
        or value > Decimal(1)
    ):
        raise LabeledEvaluationError(
            f"{name} must be finite Decimal in [0, 1]"
        )


def _digest_hex(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise LabeledEvaluationError(
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
