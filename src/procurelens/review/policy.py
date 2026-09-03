"""Explicit investigator-review selection policies for ProcureLens.

Turns transparent 0-100 review-priority scores into a review queue only through
caller-supplied policy. No default anomaly threshold, contamination fraction,
fraud label, or hidden quality/stability penalty lives here.

Two policy families are supported:
- an explicit minimum review-priority score; and
- an explicit top-N review budget.

Top-N boundary ties are never broken silently. The caller must choose whether to
include the full boundary tie group, exclude a tie group that crosses the budget,
or use a deterministic identity tie-break solely to satisfy an exact operational
budget. Tie-aware rank intervals are retained so equal anomaly scores remain
visibly equal evidence even when an operational exact-budget tie-break is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import json
from typing import Any

from procurelens.review.score import (
    ReviewPriorityScore,
    ReviewScoreBatch,
)


class ReviewPolicyError(ValueError):
    """Raised when review-selection policy or evidence is inconsistent."""


class ReviewSelectionMethod(str, Enum):
    MINIMUM_SCORE = "minimum_score"
    TOP_N = "top_n"


class ReviewTiePolicy(str, Enum):
    INCLUDE_BOUNDARY_TIES = "include_boundary_ties"
    EXCLUDE_BOUNDARY_TIES = "exclude_boundary_ties"
    DETERMINISTIC_IDENTITY = "deterministic_identity"


class ReviewSelectionReason(str, Enum):
    SCORE_AT_OR_ABOVE_MINIMUM = "score_at_or_above_minimum"
    SCORE_BELOW_MINIMUM = "score_below_minimum"
    ABOVE_TOP_N_BOUNDARY = "above_top_n_boundary"
    BELOW_TOP_N_BOUNDARY = "below_top_n_boundary"
    BOUNDARY_SELECTED_NO_CROSSING_TIE = "boundary_selected_no_crossing_tie"
    BOUNDARY_TIE_INCLUDED = "boundary_tie_included"
    BOUNDARY_TIE_EXCLUDED = "boundary_tie_excluded"
    BOUNDARY_TIE_SELECTED_BY_IDENTITY = "boundary_tie_selected_by_identity"
    BOUNDARY_TIE_NOT_SELECTED_BY_IDENTITY = "boundary_tie_not_selected_by_identity"


@dataclass(frozen=True, slots=True)
class ReviewPolicySpec:
    """Caller-owned review-queue selection policy; never a fraud threshold."""

    name: str
    description: str
    method: ReviewSelectionMethod
    minimum_score: Decimal | None = None
    top_n: int | None = None
    tie_policy: ReviewTiePolicy | None = None

    def __post_init__(self) -> None:
        for field_name in ("name", "description"):
            text = getattr(self, field_name).strip()
            if not text:
                raise ReviewPolicyError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)

        object.__setattr__(self, "method", ReviewSelectionMethod(self.method))

        if self.method is ReviewSelectionMethod.MINIMUM_SCORE:
            if self.minimum_score is None:
                raise ReviewPolicyError(
                    "minimum_score policy requires an explicit minimum_score"
                )
            _score_value(self.minimum_score, "minimum_score")
            if self.top_n is not None or self.tie_policy is not None:
                raise ReviewPolicyError(
                    "minimum_score policy cannot carry top_n or tie_policy"
                )
        else:
            if self.minimum_score is not None:
                raise ReviewPolicyError(
                    "top_n policy cannot carry minimum_score"
                )
            if (
                isinstance(self.top_n, bool)
                or not isinstance(self.top_n, int)
                or self.top_n < 1
            ):
                raise ReviewPolicyError(
                    "top_n policy requires an explicit positive integer top_n"
                )
            if self.tie_policy is None:
                raise ReviewPolicyError(
                    "top_n policy requires an explicit boundary tie policy"
                )
            object.__setattr__(
                self, "tie_policy", ReviewTiePolicy(self.tie_policy)
            )

    @property
    def sha256_hex(self) -> str:
        return _digest(self.as_dict(include_sha=False))

    def as_dict(self, *, include_sha: bool = True) -> dict[str, Any]:
        result = {
            "name": self.name,
            "description": self.description,
            "method": self.method.value,
            "minimum_score": (
                None if self.minimum_score is None else str(self.minimum_score)
            ),
            "top_n": self.top_n,
            "tie_policy": (
                None if self.tie_policy is None else self.tie_policy.value
            ),
        }
        if include_sha:
            result["sha256"] = self.sha256_hex
        return result


@dataclass(frozen=True, slots=True)
class ReviewSelectionDecision:
    transaction_id: str
    award_id: str
    source_score_evidence_sha256: str
    policy_sha256: str
    review_score: Decimal
    rank_lower: int
    rank_upper: int
    flagged_for_review: bool
    selection_reason: ReviewSelectionReason
    exact_budget_identity_tiebreak_used: bool

    def __post_init__(self) -> None:
        for field_name in ("transaction_id", "award_id"):
            text = getattr(self, field_name).strip()
            if not text:
                raise ReviewPolicyError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)
        for field_name in ("source_score_evidence_sha256", "policy_sha256"):
            object.__setattr__(
                self,
                field_name,
                _digest_hex(getattr(self, field_name), field_name),
            )
        _score_value(self.review_score, "review_score")
        for field_name in ("rank_lower", "rank_upper"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ReviewPolicyError(
                    f"{field_name} must be a positive integer"
                )
        if self.rank_lower > self.rank_upper:
            raise ReviewPolicyError("rank interval is inconsistent")
        if not isinstance(self.flagged_for_review, bool):
            raise ReviewPolicyError("flagged_for_review must be bool")
        object.__setattr__(
            self,
            "selection_reason",
            ReviewSelectionReason(self.selection_reason),
        )
        if not isinstance(self.exact_budget_identity_tiebreak_used, bool):
            raise ReviewPolicyError(
                "exact_budget_identity_tiebreak_used must be bool"
            )

        tiebreak_reasons = {
            ReviewSelectionReason.BOUNDARY_TIE_SELECTED_BY_IDENTITY,
            ReviewSelectionReason.BOUNDARY_TIE_NOT_SELECTED_BY_IDENTITY,
        }
        if (
            self.selection_reason in tiebreak_reasons
        ) != self.exact_budget_identity_tiebreak_used:
            raise ReviewPolicyError(
                "identity tie-break flag differs from selection reason"
            )

    @property
    def identity(self) -> tuple[str, str]:
        return self.transaction_id, self.award_id

    @property
    def evidence_sha256(self) -> str:
        return _digest(self.as_dict(include_sha=False))

    def as_dict(self, *, include_sha: bool = True) -> dict[str, Any]:
        result = {
            "transaction_id": self.transaction_id,
            "award_id": self.award_id,
            "source_score_evidence_sha256": self.source_score_evidence_sha256,
            "policy_sha256": self.policy_sha256,
            "review_score": str(self.review_score),
            "rank_lower": self.rank_lower,
            "rank_upper": self.rank_upper,
            "flagged_for_review": self.flagged_for_review,
            "selection_reason": self.selection_reason.value,
            "exact_budget_identity_tiebreak_used":
                self.exact_budget_identity_tiebreak_used,
        }
        if include_sha:
            result["evidence_sha256"] = self.evidence_sha256
        return result


@dataclass(frozen=True, slots=True)
class ReviewSelectionBatch:
    source_score_batch_sha256: str
    policy: ReviewPolicySpec
    row_population_sha256: str
    requested_top_n: int | None
    boundary_score: Decimal | None
    selected_count: int
    decisions: tuple[ReviewSelectionDecision, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_score_batch_sha256",
            _digest_hex(
                self.source_score_batch_sha256, "source_score_batch_sha256"
            ),
        )
        if not isinstance(self.policy, ReviewPolicySpec):
            raise TypeError("policy must be ReviewPolicySpec")
        object.__setattr__(
            self,
            "row_population_sha256",
            _digest_hex(self.row_population_sha256, "row_population_sha256"),
        )

        decisions = tuple(self.decisions)
        if not decisions:
            raise ReviewPolicyError(
                "review selection batch requires at least one decision"
            )
        identities = tuple(item.identity for item in decisions)
        if len(identities) != len(set(identities)):
            raise ReviewPolicyError(
                "review selection decisions contain duplicate identities"
            )
        if _population_digest(identities) != self.row_population_sha256:
            raise ReviewPolicyError(
                "row_population_sha256 differs from review decisions"
            )
        if any(
            item.policy_sha256 != self.policy.sha256_hex
            for item in decisions
        ):
            raise ReviewPolicyError(
                "review decision policy fingerprint differs from batch policy"
            )
        if (
            isinstance(self.selected_count, bool)
            or not isinstance(self.selected_count, int)
            or self.selected_count < 0
            or self.selected_count > len(decisions)
        ):
            raise ReviewPolicyError(
                "selected_count is inconsistent with batch size"
            )
        if self.selected_count != sum(
            item.flagged_for_review for item in decisions
        ):
            raise ReviewPolicyError(
                "selected_count differs from flagged review decisions"
            )

        if self.policy.method is ReviewSelectionMethod.MINIMUM_SCORE:
            if (
                self.requested_top_n is not None
                or self.boundary_score is not None
            ):
                raise ReviewPolicyError(
                    "minimum-score batch cannot carry top-N metadata"
                )
        else:
            if self.requested_top_n != self.policy.top_n:
                raise ReviewPolicyError(
                    "requested_top_n differs from top-N policy"
                )
            if self.boundary_score is None:
                raise ReviewPolicyError(
                    "top-N batch requires a boundary score"
                )
            _score_value(self.boundary_score, "boundary_score")
            if (
                self.policy.tie_policy
                is ReviewTiePolicy.DETERMINISTIC_IDENTITY
                and self.selected_count != self.requested_top_n
            ):
                raise ReviewPolicyError(
                    "deterministic-identity policy must satisfy exact top-N budget"
                )

        object.__setattr__(self, "decisions", decisions)

    @property
    def row_count(self) -> int:
        return len(self.decisions)

    @property
    def selected_identities(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            item.identity
            for item in self.decisions
            if item.flagged_for_review
        )

    @property
    def evidence_sha256(self) -> str:
        return _digest(
            {
                "source_score_batch_sha256": self.source_score_batch_sha256,
                "policy_sha256": self.policy.sha256_hex,
                "row_population_sha256": self.row_population_sha256,
                "requested_top_n": self.requested_top_n,
                "boundary_score": (
                    None
                    if self.boundary_score is None
                    else str(self.boundary_score)
                ),
                "selected_count": self.selected_count,
                "decisions": [
                    item.evidence_sha256 for item in self.decisions
                ],
            }
        )


def apply_review_policy(
    scores: ReviewScoreBatch,
    policy: ReviewPolicySpec,
) -> ReviewSelectionBatch:
    """Apply one explicit review-queue policy without changing anomaly scores."""

    if not isinstance(scores, ReviewScoreBatch):
        raise TypeError("scores must be ReviewScoreBatch")
    if not isinstance(policy, ReviewPolicySpec):
        raise TypeError("policy must be ReviewPolicySpec")

    rank_intervals = _rank_intervals(scores.rows)

    if policy.method is ReviewSelectionMethod.MINIMUM_SCORE:
        assert policy.minimum_score is not None
        decisions = tuple(
            _minimum_score_decision(
                row,
                policy,
                rank_intervals[row.identity],
            )
            for row in scores.rows
        )
        return ReviewSelectionBatch(
            source_score_batch_sha256=scores.evidence_sha256,
            policy=policy,
            row_population_sha256=_population_digest(scores.row_identities),
            requested_top_n=None,
            boundary_score=None,
            selected_count=sum(
                item.flagged_for_review for item in decisions
            ),
            decisions=decisions,
        )

    assert policy.top_n is not None
    if policy.top_n > scores.row_count:
        raise ReviewPolicyError(
            "top_n exceeds available review-score rows; "
            "policy must state a feasible review budget"
        )
    assert policy.tie_policy is not None

    ordered = sorted(
        scores.rows,
        key=lambda row: (
            -row.review_score,
            row.transaction_id,
            row.award_id,
        ),
    )
    boundary_score = ordered[policy.top_n - 1].review_score
    higher = tuple(
        row for row in ordered if row.review_score > boundary_score
    )
    boundary = tuple(
        row for row in ordered if row.review_score == boundary_score
    )
    crosses_budget = len(higher) < policy.top_n < len(higher) + len(boundary)

    selected: set[tuple[str, str]] = {row.identity for row in higher}
    identity_tiebreak_selected: set[tuple[str, str]] = set()

    if not crosses_budget:
        selected.update(row.identity for row in boundary)
    elif policy.tie_policy is ReviewTiePolicy.INCLUDE_BOUNDARY_TIES:
        selected.update(row.identity for row in boundary)
    elif policy.tie_policy is ReviewTiePolicy.EXCLUDE_BOUNDARY_TIES:
        pass
    else:
        slots = policy.top_n - len(higher)
        chosen = tuple(
            sorted(
                boundary,
                key=lambda row: (row.transaction_id, row.award_id),
            )[:slots]
        )
        identity_tiebreak_selected = {row.identity for row in chosen}
        selected.update(identity_tiebreak_selected)

    decisions = tuple(
        _top_n_decision(
            row=row,
            policy=policy,
            rank_interval=rank_intervals[row.identity],
            boundary_score=boundary_score,
            selected=selected,
            boundary_id_tiebreak_selected=identity_tiebreak_selected,
            crosses_budget=crosses_budget,
        )
        for row in scores.rows
    )

    return ReviewSelectionBatch(
        source_score_batch_sha256=scores.evidence_sha256,
        policy=policy,
        row_population_sha256=_population_digest(scores.row_identities),
        requested_top_n=policy.top_n,
        boundary_score=boundary_score,
        selected_count=sum(item.flagged_for_review for item in decisions),
        decisions=decisions,
    )


def _minimum_score_decision(
    row: ReviewPriorityScore,
    policy: ReviewPolicySpec,
    rank_interval: tuple[int, int],
) -> ReviewSelectionDecision:
    assert policy.minimum_score is not None
    flagged = row.review_score >= policy.minimum_score
    return ReviewSelectionDecision(
        transaction_id=row.transaction_id,
        award_id=row.award_id,
        source_score_evidence_sha256=row.evidence_sha256,
        policy_sha256=policy.sha256_hex,
        review_score=row.review_score,
        rank_lower=rank_interval[0],
        rank_upper=rank_interval[1],
        flagged_for_review=flagged,
        selection_reason=(
            ReviewSelectionReason.SCORE_AT_OR_ABOVE_MINIMUM
            if flagged
            else ReviewSelectionReason.SCORE_BELOW_MINIMUM
        ),
        exact_budget_identity_tiebreak_used=False,
    )


def _top_n_decision(
    *,
    row: ReviewPriorityScore,
    policy: ReviewPolicySpec,
    rank_interval: tuple[int, int],
    boundary_score: Decimal,
    selected: set[tuple[str, str]],
    boundary_id_tiebreak_selected: set[tuple[str, str]],
    crosses_budget: bool,
) -> ReviewSelectionDecision:
    assert policy.tie_policy is not None
    flagged = row.identity in selected
    tiebreak_used = False

    if row.review_score > boundary_score:
        reason = ReviewSelectionReason.ABOVE_TOP_N_BOUNDARY
    elif row.review_score < boundary_score:
        reason = ReviewSelectionReason.BELOW_TOP_N_BOUNDARY
    elif not crosses_budget:
        reason = ReviewSelectionReason.BOUNDARY_SELECTED_NO_CROSSING_TIE
    elif policy.tie_policy is ReviewTiePolicy.INCLUDE_BOUNDARY_TIES:
        reason = ReviewSelectionReason.BOUNDARY_TIE_INCLUDED
    elif policy.tie_policy is ReviewTiePolicy.EXCLUDE_BOUNDARY_TIES:
        reason = ReviewSelectionReason.BOUNDARY_TIE_EXCLUDED
    else:
        tiebreak_used = True
        reason = (
            ReviewSelectionReason.BOUNDARY_TIE_SELECTED_BY_IDENTITY
            if row.identity in boundary_id_tiebreak_selected
            else ReviewSelectionReason.BOUNDARY_TIE_NOT_SELECTED_BY_IDENTITY
        )

    return ReviewSelectionDecision(
        transaction_id=row.transaction_id,
        award_id=row.award_id,
        source_score_evidence_sha256=row.evidence_sha256,
        policy_sha256=policy.sha256_hex,
        review_score=row.review_score,
        rank_lower=rank_interval[0],
        rank_upper=rank_interval[1],
        flagged_for_review=flagged,
        selection_reason=reason,
        exact_budget_identity_tiebreak_used=tiebreak_used,
    )


def _rank_intervals(
    rows: tuple[ReviewPriorityScore, ...],
) -> dict[tuple[str, str], tuple[int, int]]:
    scores = tuple(row.review_score for row in rows)
    result: dict[tuple[str, str], tuple[int, int]] = {}
    for row in rows:
        strictly_higher = sum(value > row.review_score for value in scores)
        equal = sum(value == row.review_score for value in scores)
        result[row.identity] = (
            strictly_higher + 1,
            strictly_higher + equal,
        )
    return result


def _population_digest(
    identities: tuple[tuple[str, str], ...] | list[tuple[str, str]],
) -> str:
    normalized: list[tuple[str, str]] = []
    for identity in identities:
        if not isinstance(identity, tuple) or len(identity) != 2:
            raise ReviewPolicyError(
                "row identities must be (transaction_id, award_id) tuples"
            )
        txid, award_id = identity[0].strip(), identity[1].strip()
        if not txid or not award_id:
            raise ReviewPolicyError(
                "row identity values must not be blank"
            )
        normalized.append((txid, award_id))
    result = tuple(normalized)
    if not result or len(result) != len(set(result)):
        raise ReviewPolicyError(
            "row identities must be non-empty and unique"
        )
    return _digest(result)


def _score_value(value: Decimal, name: str) -> None:
    if (
        not isinstance(value, Decimal)
        or not value.is_finite()
        or value < Decimal(0)
        or value > Decimal(100)
    ):
        raise ReviewPolicyError(
            f"{name} must be finite Decimal in [0, 100]"
        )


def _digest_hex(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ReviewPolicyError(
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
