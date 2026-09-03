"""Peer-group monetary reference samples for ProcureLens.

Stores peer amounts for contextual analysis, without computing anomaly scores or
risk. When the target transaction is part of the reference population, its own
amount is removed before the peer sample is returned.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any

from procurelens.domain.transaction import ProcurementTransaction
from procurelens.features.peer_groups import (
    PeerGroupCandidate,
    PeerGroupKey,
    PeerGroupPlan,
    federal_contract_amount_peer_plan,
    peer_group_candidates,
)
from procurelens.statistics.robust import decimal_identity, remove_one_sorted


class AmountReferenceError(ValueError):
    pass


class AmountBasis(str, Enum):
    ACTION_OBLIGATION = "action_obligation"
    AWARD_TOTAL_OBLIGATION = "award_total_obligation"


@dataclass(frozen=True, slots=True)
class AmountReferencePolicy:
    # Resource budgets only; these are not anomaly/risk thresholds.
    max_transaction_ids: int | None = None
    max_distinct_groups: int | None = None
    max_group_observations: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_transaction_ids",
            "max_distinct_groups",
            "max_group_observations",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise AmountReferenceError(
                    f"{name} must be a positive integer or None"
                )


@dataclass(frozen=True, slots=True)
class AmountPeerAttempt:
    level_name: str
    key: PeerGroupKey | None
    population_count: int | None
    peer_count: int | None
    sufficient: bool
    unavailable_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AmountReferenceResolution:
    plan_name: str
    plan_sha256: str
    amount_basis: AmountBasis
    minimum_peer_count: int
    target_was_indexed: bool
    selected: AmountPeerAttempt | None
    attempts: tuple[AmountPeerAttempt, ...]

    @property
    def resolved(self) -> bool:
        return self.selected is not None


@dataclass(frozen=True, slots=True)
class AmountReferenceSample:
    transaction_id: str
    amount_basis: AmountBasis
    target_amount: Decimal
    peer_values: tuple[Decimal, ...]
    resolution: AmountReferenceResolution
    target_excluded: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "amount_basis", AmountBasis(self.amount_basis))
        if not self.transaction_id.strip():
            raise AmountReferenceError("transaction_id must not be blank")
        if not isinstance(self.target_amount, Decimal) or not self.target_amount.is_finite():
            raise AmountReferenceError("target_amount must be a finite Decimal")
        if not self.resolution.resolved or self.resolution.selected is None:
            raise AmountReferenceError("sample requires a resolved peer group")
        if not self.peer_values:
            raise AmountReferenceError("sample must contain at least one peer")
        if any(not isinstance(v, Decimal) or not v.is_finite() for v in self.peer_values):
            raise AmountReferenceError("peer amounts must be finite Decimal values")
        if any(self.peer_values[i] > self.peer_values[i + 1] for i in range(len(self.peer_values) - 1)):
            raise AmountReferenceError("peer amounts must be sorted")
        if self.resolution.selected.peer_count != len(self.peer_values):
            raise AmountReferenceError("peer count does not match returned values")


@dataclass(frozen=True, slots=True)
class AmountReferenceResult:
    amount_basis: AmountBasis
    target_amount: Decimal | None
    resolution: AmountReferenceResolution | None
    sample: AmountReferenceSample | None
    unavailable_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "amount_basis", AmountBasis(self.amount_basis))
        if self.sample is not None and self.unavailable_reasons:
            raise AmountReferenceError(
                "available sample cannot also carry unavailable reasons"
            )

    @property
    def available(self) -> bool:
        return self.sample is not None


@dataclass(frozen=True, slots=True)
class AmountReferenceSnapshot:
    plan: PeerGroupPlan
    amount_basis: AmountBasis
    total_transactions: int
    transactions_with_amount: int
    group_values: Mapping[PeerGroupKey, tuple[Decimal, ...]]
    transaction_fingerprints: Mapping[str, str]
    group_observation_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "amount_basis", AmountBasis(self.amount_basis))
        if not isinstance(self.plan, PeerGroupPlan):
            raise TypeError("plan must be PeerGroupPlan")
        if (
            isinstance(self.total_transactions, bool)
            or not isinstance(self.total_transactions, int)
            or self.total_transactions < 0
        ):
            raise AmountReferenceError("total_transactions must be non-negative")
        if not (0 <= self.transactions_with_amount <= self.total_transactions):
            raise AmountReferenceError("transactions_with_amount is inconsistent")
        if len(self.transaction_fingerprints) != self.total_transactions:
            raise AmountReferenceError(
                "one transaction fingerprint is required per observed transaction"
            )

        frozen: dict[PeerGroupKey, tuple[Decimal, ...]] = {}
        observed = 0
        for key, values in self.group_values.items():
            ordered = tuple(values)
            if not ordered:
                raise AmountReferenceError("stored amount groups must not be empty")
            if any(not isinstance(v, Decimal) or not v.is_finite() for v in ordered):
                raise AmountReferenceError("stored amounts must be finite Decimal values")
            if any(ordered[i] > ordered[i + 1] for i in range(len(ordered) - 1)):
                raise AmountReferenceError("stored amounts must be sorted")
            frozen[key] = ordered
            observed += len(ordered)
        if observed != self.group_observation_count:
            raise AmountReferenceError(
                "group_observation_count does not match stored values"
            )

        object.__setattr__(self, "group_values", MappingProxyType(frozen))
        object.__setattr__(
            self,
            "transaction_fingerprints",
            MappingProxyType(dict(self.transaction_fingerprints)),
        )

    @property
    def transactions_without_amount(self) -> int:
        return self.total_transactions - self.transactions_with_amount

    def resolve(
        self,
        transaction: ProcurementTransaction,
        *,
        minimum_peer_count: int,
    ) -> AmountReferenceResult:
        if not isinstance(transaction, ProcurementTransaction):
            raise TypeError("transaction must be ProcurementTransaction")
        if (
            isinstance(minimum_peer_count, bool)
            or not isinstance(minimum_peer_count, int)
            or minimum_peer_count < 1
        ):
            raise AmountReferenceError(
                "minimum_peer_count must be caller-supplied and positive"
            )

        candidates = peer_group_candidates(transaction, self.plan)
        amount = _amount(transaction, self.amount_basis)
        fingerprint = _fingerprint(candidates, amount, self.amount_basis)
        stored = self.transaction_fingerprints.get(transaction.transaction_id)
        indexed = stored is not None
        if indexed and stored != fingerprint:
            raise AmountReferenceError(
                "target transaction differs from its indexed reference record"
            )

        if amount is None:
            return AmountReferenceResult(
                self.amount_basis,
                None,
                None,
                None,
                ("target_amount_missing",),
            )

        attempts: list[AmountPeerAttempt] = []
        selected: AmountPeerAttempt | None = None
        for candidate in candidates:
            if candidate.key is None:
                attempt = AmountPeerAttempt(
                    candidate.level_name,
                    None,
                    None,
                    None,
                    False,
                    candidate.unavailable_reasons,
                )
            else:
                values = self.group_values.get(candidate.key, ())
                population = len(values)
                if indexed and population == 0:
                    raise AmountReferenceError(
                        "indexed target is missing from an expected amount group"
                    )
                peers = population - (1 if indexed else 0)
                attempt = AmountPeerAttempt(
                    candidate.level_name,
                    candidate.key,
                    population,
                    peers,
                    peers >= minimum_peer_count,
                )
                if attempt.sufficient:
                    selected = attempt
            attempts.append(attempt)
            if selected is not None:
                break

        resolution = AmountReferenceResolution(
            self.plan.name,
            self.plan.sha256_hex,
            self.amount_basis,
            minimum_peer_count,
            indexed,
            selected,
            tuple(attempts),
        )
        if selected is None or selected.key is None:
            return AmountReferenceResult(
                self.amount_basis,
                amount,
                resolution,
                None,
                ("no_peer_group_with_required_support",),
            )

        stored_values = self.group_values[selected.key]
        peers = remove_one_sorted(stored_values, amount) if indexed else tuple(stored_values)
        sample = AmountReferenceSample(
            transaction.transaction_id,
            self.amount_basis,
            amount,
            peers,
            resolution,
            indexed,
        )
        return AmountReferenceResult(
            self.amount_basis,
            amount,
            resolution,
            sample,
        )


class AmountReferenceIndex:
    """One-pass builder. Transaction objects themselves are never retained."""

    def __init__(
        self,
        *,
        plan: PeerGroupPlan | None = None,
        amount_basis: AmountBasis = AmountBasis.ACTION_OBLIGATION,
        policy: AmountReferencePolicy | None = None,
    ) -> None:
        self.plan = plan or federal_contract_amount_peer_plan()
        if not isinstance(self.plan, PeerGroupPlan):
            raise TypeError("plan must be PeerGroupPlan")
        self.amount_basis = AmountBasis(amount_basis)
        self.policy = policy or AmountReferencePolicy()
        self._groups: dict[PeerGroupKey, list[Decimal]] = {}
        self._fingerprints: dict[str, str] = {}
        self._total = 0
        self._with_amount = 0
        self._group_observations = 0

    def observe(self, transaction: ProcurementTransaction) -> None:
        """Pre-check every failure condition before mutating any state."""
        if not isinstance(transaction, ProcurementTransaction):
            raise TypeError("transaction must be ProcurementTransaction")

        txid = transaction.transaction_id
        if txid in self._fingerprints:
            raise AmountReferenceError(
                f"duplicate transaction_id in amount population: {txid!r}"
            )

        candidates = peer_group_candidates(transaction, self.plan)
        amount = _amount(transaction, self.amount_basis)
        fingerprint = _fingerprint(candidates, amount, self.amount_basis)

        limit = self.policy.max_transaction_ids
        if limit is not None and len(self._fingerprints) >= limit:
            raise AmountReferenceError(
                f"transaction index exceeds max_transaction_ids={limit}"
            )

        valid_keys: tuple[PeerGroupKey, ...] = ()
        if amount is not None:
            valid_keys = tuple(c.key for c in candidates if c.key is not None)
            new_keys = sum(key not in self._groups for key in valid_keys)

            limit = self.policy.max_distinct_groups
            if limit is not None and len(self._groups) + new_keys > limit:
                raise AmountReferenceError(
                    f"group index exceeds max_distinct_groups={limit}"
                )

            limit = self.policy.max_group_observations
            if limit is not None and self._group_observations + len(valid_keys) > limit:
                raise AmountReferenceError(
                    f"group observations exceed max_group_observations={limit}"
                )

        self._fingerprints[txid] = fingerprint
        self._total += 1
        if amount is None:
            return

        self._with_amount += 1
        for key in valid_keys:
            self._groups.setdefault(key, []).append(amount)
        self._group_observations += len(valid_keys)

    def observe_many(
        self,
        transactions: Iterable[ProcurementTransaction],
    ) -> "AmountReferenceIndex":
        for transaction in transactions:
            self.observe(transaction)
        return self

    def snapshot(self) -> AmountReferenceSnapshot:
        return AmountReferenceSnapshot(
            self.plan,
            self.amount_basis,
            self._total,
            self._with_amount,
            {key: tuple(sorted(values)) for key, values in self._groups.items()},
            dict(self._fingerprints),
            self._group_observations,
        )


def _amount(
    transaction: ProcurementTransaction,
    basis: AmountBasis,
) -> Decimal | None:
    if basis is AmountBasis.ACTION_OBLIGATION:
        return transaction.action_obligation
    if basis is AmountBasis.AWARD_TOTAL_OBLIGATION:
        return transaction.award_total_obligation
    raise AmountReferenceError(f"unsupported amount basis: {basis!r}")


def _fingerprint(
    candidates: tuple[PeerGroupCandidate, ...],
    amount: Decimal | None,
    basis: AmountBasis,
) -> str:
    return _digest(
        {
            "basis": basis.value,
            "amount": None if amount is None else decimal_identity(amount),
            "candidates": [
                {
                    "level": c.level_name,
                    "key": None if c.key is None else c.key.sha256_hex,
                    "unavailable": list(c.unavailable_reasons),
                }
                for c in candidates
            ],
        }
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
