"""Descriptive amount context for ProcureLens.

Takes a resolved leave-one-out peer amount sample and turns it into reviewable
statistics. This module does not classify a transaction as anomalous, risky,
fraudulent, or suspicious. It deliberately exposes several complementary views
instead of hiding them behind one magic cutoff.

Signed values answer "where does this transaction change sit?"
Magnitude values answer "how large is this change regardless of direction?"
The sign breakdown keeps de-obligations, zero-dollar actions, and positive
obligations distinguishable for later policy and explanation layers.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any

from procurelens.features.amount_reference import (
    AmountBasis,
    AmountReferenceResult,
    AmountReferenceSample,
)
from procurelens.statistics.robust import (
    EmpiricalPosition,
    QuantileMethod,
    RobustSummary,
    empirical_position_sorted,
    iqr_distance,
    modified_z,
    summarize_sorted,
)


class AmountContextError(ValueError):
    """Raised when an amount context is internally inconsistent."""


class AmountDirection(str, Enum):
    NEGATIVE = "negative"
    ZERO = "zero"
    POSITIVE = "positive"


@dataclass(frozen=True, slots=True)
class AmountSignBreakdown:
    """How many peers are negative, zero, or positive."""

    negative: int
    zero: int
    positive: int

    def __post_init__(self) -> None:
        for name in ("negative", "zero", "positive"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AmountContextError(f"{name} count must be a non-negative integer")
        if self.total < 1:
            raise AmountContextError("peer sign breakdown must contain at least one peer")

    @property
    def total(self) -> int:
        return self.negative + self.zero + self.positive

    def count(self, direction: AmountDirection) -> int:
        direction = AmountDirection(direction)
        if direction is AmountDirection.NEGATIVE:
            return self.negative
        if direction is AmountDirection.ZERO:
            return self.zero
        return self.positive


@dataclass(frozen=True, slots=True)
class DistributionContext:
    """One target value described against one non-empty peer distribution."""

    target: Decimal
    summary: RobustSummary
    position: EmpiricalPosition
    delta_from_median: Decimal
    absolute_delta_from_median: Decimal
    modified_z: Decimal | None
    iqr_distance: Decimal | None

    def __post_init__(self) -> None:
        if not isinstance(self.target, Decimal) or not self.target.is_finite():
            raise AmountContextError("distribution target must be a finite Decimal")
        if not isinstance(self.summary, RobustSummary):
            raise TypeError("summary must be RobustSummary")
        if not isinstance(self.position, EmpiricalPosition):
            raise TypeError("position must be EmpiricalPosition")
        if self.position.total != self.summary.count:
            raise AmountContextError("position and summary must describe the same peer sample")
        if self.delta_from_median != self.target - self.summary.median:
            raise AmountContextError("delta_from_median is inconsistent")
        if self.absolute_delta_from_median != abs(self.delta_from_median):
            raise AmountContextError("absolute_delta_from_median is inconsistent")


@dataclass(frozen=True, slots=True)
class AmountContext:
    """Reviewable descriptive evidence for one transaction amount."""

    transaction_id: str
    amount_basis: AmountBasis
    target_amount: Decimal
    target_direction: AmountDirection
    absolute_target_amount: Decimal

    peer_count: int
    peer_signs: AmountSignBreakdown
    selected_peer_level: str
    selected_peer_key_sha256: str
    target_excluded_from_peers: bool
    reference_minimum_peer_count: int
    reference_plan_name: str
    reference_plan_sha256: str
    quantile_method: QuantileMethod

    signed: DistributionContext
    magnitude: DistributionContext
    same_direction_magnitude: DistributionContext | None
    same_direction_peer_count: int

    def __post_init__(self) -> None:
        txid = self.transaction_id.strip()
        if not txid:
            raise AmountContextError("transaction_id must not be blank")
        object.__setattr__(self, "transaction_id", txid)
        object.__setattr__(self, "amount_basis", AmountBasis(self.amount_basis))
        object.__setattr__(self, "target_direction", AmountDirection(self.target_direction))
        object.__setattr__(self, "quantile_method", QuantileMethod(self.quantile_method))

        if not isinstance(self.target_amount, Decimal) or not self.target_amount.is_finite():
            raise AmountContextError("target_amount must be a finite Decimal")
        if self.absolute_target_amount != abs(self.target_amount):
            raise AmountContextError("absolute_target_amount is inconsistent")
        if self.target_direction is not _direction(self.target_amount):
            raise AmountContextError("target_direction is inconsistent")

        if isinstance(self.peer_count, bool) or not isinstance(self.peer_count, int) or self.peer_count < 1:
            raise AmountContextError("peer_count must be a positive integer")
        if self.peer_signs.total != self.peer_count:
            raise AmountContextError("peer sign counts must sum to peer_count")
        if self.signed.summary.count != self.peer_count:
            raise AmountContextError("signed distribution must describe every peer")
        if self.magnitude.summary.count != self.peer_count:
            raise AmountContextError("magnitude distribution must describe every peer")
        if self.signed.target != self.target_amount:
            raise AmountContextError("signed target is inconsistent")
        if self.magnitude.target != self.absolute_target_amount:
            raise AmountContextError("magnitude target is inconsistent")

        expected_directional = self.peer_signs.count(self.target_direction)
        if self.same_direction_peer_count != expected_directional:
            raise AmountContextError("same_direction_peer_count is inconsistent")
        if expected_directional == 0:
            if self.same_direction_magnitude is not None:
                raise AmountContextError(
                    "same-direction context must be absent when there are no same-direction peers"
                )
        else:
            if self.same_direction_magnitude is None:
                raise AmountContextError(
                    "same-direction context is required when same-direction peers exist"
                )
            if self.same_direction_magnitude.summary.count != expected_directional:
                raise AmountContextError("same-direction distribution count is inconsistent")
            if self.same_direction_magnitude.target != self.absolute_target_amount:
                raise AmountContextError("same-direction target is inconsistent")

        if not self.selected_peer_level.strip():
            raise AmountContextError("selected_peer_level must not be blank")
        digest = self.selected_peer_key_sha256.strip().lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise AmountContextError("selected_peer_key_sha256 must be a SHA-256 hex digest")
        object.__setattr__(self, "selected_peer_key_sha256", digest)

        for name in ("reference_plan_name", "reference_plan_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise AmountContextError(f"{name} must not be blank")
        if (
            isinstance(self.reference_minimum_peer_count, bool)
            or not isinstance(self.reference_minimum_peer_count, int)
            or self.reference_minimum_peer_count < 1
        ):
            raise AmountContextError("reference_minimum_peer_count must be positive")

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly evidence without converting Decimal money to binary floats."""

        return {
            "transaction_id": self.transaction_id,
            "amount_basis": self.amount_basis.value,
            "target_amount": str(self.target_amount),
            "target_direction": self.target_direction.value,
            "absolute_target_amount": str(self.absolute_target_amount),
            "peer_count": self.peer_count,
            "peer_signs": {
                "negative": self.peer_signs.negative,
                "zero": self.peer_signs.zero,
                "positive": self.peer_signs.positive,
            },
            "selected_peer_level": self.selected_peer_level,
            "selected_peer_key_sha256": self.selected_peer_key_sha256,
            "target_excluded_from_peers": self.target_excluded_from_peers,
            "reference_minimum_peer_count": self.reference_minimum_peer_count,
            "reference_plan_name": self.reference_plan_name,
            "reference_plan_sha256": self.reference_plan_sha256,
            "quantile_method": self.quantile_method.value,
            "signed": _distribution_dict(self.signed),
            "magnitude": _distribution_dict(self.magnitude),
            "same_direction_peer_count": self.same_direction_peer_count,
            "same_direction_magnitude": (
                None
                if self.same_direction_magnitude is None
                else _distribution_dict(self.same_direction_magnitude)
            ),
        }


@dataclass(frozen=True, slots=True)
class AmountContextResult:
    """Available context or a transparent reason why it could not be built."""

    amount_basis: AmountBasis
    target_amount: Decimal | None
    context: AmountContext | None
    unavailable_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "amount_basis", AmountBasis(self.amount_basis))
        if self.context is not None and self.unavailable_reasons:
            raise AmountContextError(
                "available amount context cannot also carry unavailable reasons"
            )
        if self.context is not None:
            if self.context.amount_basis is not self.amount_basis:
                raise AmountContextError("result and context amount bases differ")
            if self.target_amount != self.context.target_amount:
                raise AmountContextError("result and context target amounts differ")

    @property
    def available(self) -> bool:
        return self.context is not None


def build_amount_context(
    reference: AmountReferenceResult,
    *,
    quantile_method: QuantileMethod = QuantileMethod.LINEAR_TYPE7,
) -> AmountContextResult:
    """Turn a resolved amount-reference sample into descriptive context."""

    if not isinstance(reference, AmountReferenceResult):
        raise TypeError("reference must be AmountReferenceResult")
    method = QuantileMethod(quantile_method)

    if not reference.available or reference.sample is None:
        reasons = reference.unavailable_reasons or ("amount_reference_unavailable",)
        return AmountContextResult(
            reference.amount_basis,
            reference.target_amount,
            None,
            tuple(reasons),
        )

    sample = reference.sample
    _validate_sample(sample)

    selected = sample.resolution.selected
    if selected is None or selected.key is None:
        raise AmountContextError("available reference sample lacks a selected peer group")

    peers = sample.peer_values
    target = sample.target_amount
    signed = _distribution(target, peers, method=method)

    magnitude_peers = tuple(sorted(abs(value) for value in peers))
    magnitude = _distribution(abs(target), magnitude_peers, method=method)

    direction = _direction(target)
    same_direction_values = tuple(
        sorted(abs(value) for value in peers if _direction(value) is direction)
    )
    directional = (
        None
        if not same_direction_values
        else _distribution(abs(target), same_direction_values, method=method)
    )

    signs = _sign_breakdown(peers)
    context = AmountContext(
        transaction_id=sample.transaction_id,
        amount_basis=sample.amount_basis,
        target_amount=target,
        target_direction=direction,
        absolute_target_amount=abs(target),
        peer_count=len(peers),
        peer_signs=signs,
        selected_peer_level=selected.level_name,
        selected_peer_key_sha256=selected.key.sha256_hex,
        target_excluded_from_peers=sample.target_excluded,
        reference_minimum_peer_count=sample.resolution.minimum_peer_count,
        reference_plan_name=sample.resolution.plan_name,
        reference_plan_sha256=sample.resolution.plan_sha256,
        quantile_method=method,
        signed=signed,
        magnitude=magnitude,
        same_direction_magnitude=directional,
        same_direction_peer_count=len(same_direction_values),
    )
    return AmountContextResult(sample.amount_basis, target, context)


def _distribution(
    target: Decimal,
    sorted_values: tuple[Decimal, ...],
    *,
    method: QuantileMethod,
) -> DistributionContext:
    summary = summarize_sorted(sorted_values, method=method)
    position = empirical_position_sorted(sorted_values, target)
    delta = target - summary.median
    return DistributionContext(
        target=target,
        summary=summary,
        position=position,
        delta_from_median=delta,
        absolute_delta_from_median=abs(delta),
        modified_z=modified_z(target, summary),
        iqr_distance=iqr_distance(target, summary),
    )


def _direction(value: Decimal) -> AmountDirection:
    if value < 0:
        return AmountDirection.NEGATIVE
    if value > 0:
        return AmountDirection.POSITIVE
    return AmountDirection.ZERO


def _sign_breakdown(values: tuple[Decimal, ...]) -> AmountSignBreakdown:
    negative = sum(value < 0 for value in values)
    zero = sum(value == 0 for value in values)
    positive = len(values) - negative - zero
    return AmountSignBreakdown(negative, zero, positive)


def _validate_sample(sample: AmountReferenceSample) -> None:
    if not isinstance(sample, AmountReferenceSample):
        raise TypeError("sample must be AmountReferenceSample")
    if not sample.peer_values:
        raise AmountContextError("amount reference sample has no peers")
    if sample.resolution.selected is None:
        raise AmountContextError("amount reference sample is not resolved")


def _distribution_dict(value: DistributionContext) -> dict[str, Any]:
    summary = value.summary
    position = value.position

    def optional_decimal(item: Decimal | None) -> str | None:
        return None if item is None else str(item)

    return {
        "target": str(value.target),
        "peer_count": summary.count,
        "minimum": str(summary.minimum),
        "first_quartile": str(summary.first_quartile),
        "median": str(summary.median),
        "third_quartile": str(summary.third_quartile),
        "maximum": str(summary.maximum),
        "interquartile_range": str(summary.interquartile_range),
        "median_absolute_deviation": str(summary.median_absolute_deviation),
        "delta_from_median": str(value.delta_from_median),
        "absolute_delta_from_median": str(value.absolute_delta_from_median),
        "modified_z": optional_decimal(value.modified_z),
        "iqr_distance": optional_decimal(value.iqr_distance),
        "empirical_position": {
            "less": position.less,
            "equal": position.equal,
            "greater": position.greater,
            "lower_fraction": str(position.lower_fraction),
            "midpoint_fraction": str(position.midpoint_fraction),
            "upper_fraction": str(position.upper_fraction),
        },
    }
