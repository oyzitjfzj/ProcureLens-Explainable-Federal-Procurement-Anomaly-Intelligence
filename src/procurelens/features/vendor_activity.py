"""Deterministic vendor activity facts for ProcureLens.

Aggregates procurement transactions for one explicit vendor identity scope.
This module separates transaction frequency from distinct-award frequency and
keeps money movement signed. It does not score risk, infer fraud, reconcile
aliases across identifier systems, or compare a vendor with a market.

The supplied transaction population defines the observation window. Later
context layers can compare these facts within agencies, categories, or time
periods without this module inventing a hidden benchmark.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any

from procurelens.domain.transaction import ProcurementTransaction
from procurelens.features.vendor_identity import (
    VendorIdentity,
    VendorIdentityMethod,
    VendorIdentityScope,
    resolve_vendor_identity,
)


class VendorActivityError(ValueError):
    """Raised when vendor activity input or accumulated state is inconsistent."""


@dataclass(frozen=True, slots=True)
class VendorActivityPolicy:
    """Optional memory budgets only; no risk or anomaly thresholds live here."""

    max_transaction_ids: int | None = None
    max_vendor_identities: int | None = None
    max_distinct_awards_per_vendor: int | None = None
    max_context_values_per_vendor: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_transaction_ids",
            "max_vendor_identities",
            "max_distinct_awards_per_vendor",
            "max_context_values_per_vendor",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise VendorActivityError(
                    f"{name} must be a positive integer or None"
                )


@dataclass(frozen=True, slots=True)
class VendorActivity:
    """Descriptive facts for one resolved vendor identity in one population."""

    identity: VendorIdentity
    transaction_count: int
    distinct_award_count: int
    single_transaction_award_count: int
    multi_transaction_award_count: int
    maximum_transactions_on_one_award: int

    net_action_obligation: Decimal
    absolute_action_obligation_activity: Decimal
    positive_action_obligation: Decimal
    deobligation_magnitude: Decimal
    zero_obligation_transaction_count: int

    first_action_date: date
    last_action_date: date
    distinct_awarding_agency_count: int
    distinct_awarding_subtier_count: int
    distinct_psc_count: int
    distinct_naics_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.identity, VendorIdentity):
            raise TypeError("identity must be VendorIdentity")
        integer_fields = (
            "transaction_count",
            "distinct_award_count",
            "single_transaction_award_count",
            "multi_transaction_award_count",
            "maximum_transactions_on_one_award",
            "zero_obligation_transaction_count",
            "distinct_awarding_agency_count",
            "distinct_awarding_subtier_count",
            "distinct_psc_count",
            "distinct_naics_count",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise VendorActivityError(f"{name} must be a non-negative integer")
        if self.transaction_count < 1 or self.distinct_award_count < 1:
            raise VendorActivityError("vendor activity needs at least one transaction and award")
        if self.distinct_award_count > self.transaction_count:
            raise VendorActivityError("distinct awards cannot exceed transactions")
        if self.single_transaction_award_count + self.multi_transaction_award_count != self.distinct_award_count:
            raise VendorActivityError("award transaction-frequency counts are inconsistent")
        if not (1 <= self.maximum_transactions_on_one_award <= self.transaction_count):
            raise VendorActivityError("maximum transactions per award is inconsistent")
        if self.zero_obligation_transaction_count > self.transaction_count:
            raise VendorActivityError("zero-obligation count cannot exceed transactions")

        money = (
            self.net_action_obligation,
            self.absolute_action_obligation_activity,
            self.positive_action_obligation,
            self.deobligation_magnitude,
        )
        if any(not isinstance(value, Decimal) or not value.is_finite() for value in money):
            raise VendorActivityError("money values must be finite Decimal values")
        if self.absolute_action_obligation_activity < 0 or self.positive_action_obligation < 0 or self.deobligation_magnitude < 0:
            raise VendorActivityError("money magnitudes must not be negative")
        if self.absolute_action_obligation_activity != self.positive_action_obligation + self.deobligation_magnitude:
            raise VendorActivityError("absolute activity must equal positive obligations plus de-obligations")
        if self.net_action_obligation != self.positive_action_obligation - self.deobligation_magnitude:
            raise VendorActivityError("net obligation is inconsistent with signed components")
        if self.first_action_date > self.last_action_date:
            raise VendorActivityError("first_action_date must not be after last_action_date")

    @property
    def transactions_per_award(self) -> Decimal:
        return Decimal(self.transaction_count) / Decimal(self.distinct_award_count)

    @property
    def has_stable_vendor_identifier(self) -> bool:
        return self.identity.method is not VendorIdentityMethod.NORMALIZED_NAME

    def as_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.as_dict(),
            "transaction_count": self.transaction_count,
            "distinct_award_count": self.distinct_award_count,
            "single_transaction_award_count": self.single_transaction_award_count,
            "multi_transaction_award_count": self.multi_transaction_award_count,
            "maximum_transactions_on_one_award": self.maximum_transactions_on_one_award,
            "transactions_per_award": str(self.transactions_per_award),
            "net_action_obligation": str(self.net_action_obligation),
            "absolute_action_obligation_activity": str(self.absolute_action_obligation_activity),
            "positive_action_obligation": str(self.positive_action_obligation),
            "deobligation_magnitude": str(self.deobligation_magnitude),
            "zero_obligation_transaction_count": self.zero_obligation_transaction_count,
            "first_action_date": self.first_action_date.isoformat(),
            "last_action_date": self.last_action_date.isoformat(),
            "distinct_awarding_agency_count": self.distinct_awarding_agency_count,
            "distinct_awarding_subtier_count": self.distinct_awarding_subtier_count,
            "distinct_psc_count": self.distinct_psc_count,
            "distinct_naics_count": self.distinct_naics_count,
            "has_stable_vendor_identifier": self.has_stable_vendor_identifier,
        }


@dataclass(frozen=True, slots=True)
class VendorActivitySnapshot:
    """Immutable activity population for one vendor identity scope."""

    scope: VendorIdentityScope
    total_transactions: int
    transactions_with_identity: int
    transactions_without_identity: int
    observed_start_date: date | None
    observed_end_date: date | None
    activities: Mapping[str, VendorActivity]
    transaction_population_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", VendorIdentityScope(self.scope))
        for name in (
            "total_transactions",
            "transactions_with_identity",
            "transactions_without_identity",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise VendorActivityError(f"{name} must be a non-negative integer")
        if self.transactions_with_identity + self.transactions_without_identity != self.total_transactions:
            raise VendorActivityError("identity coverage counts do not sum to total transactions")
        if self.total_transactions == 0:
            if self.observed_start_date is not None or self.observed_end_date is not None:
                raise VendorActivityError("empty population must not carry observation dates")
        else:
            if self.observed_start_date is None or self.observed_end_date is None:
                raise VendorActivityError("non-empty population requires observation dates")
            if self.observed_start_date > self.observed_end_date:
                raise VendorActivityError("observation window is inconsistent")
        frozen = dict(self.activities)
        if any(key != activity.identity.key for key, activity in frozen.items()):
            raise VendorActivityError("activity mapping key does not match vendor identity key")
        if any(activity.identity.scope is not self.scope for activity in frozen.values()):
            raise VendorActivityError("activity identity scope differs from snapshot scope")
        if sum(activity.transaction_count for activity in frozen.values()) != self.transactions_with_identity:
            raise VendorActivityError("vendor transaction counts do not match identity coverage")
        object.__setattr__(self, "activities", MappingProxyType(frozen))
        digest = self.transaction_population_sha256.strip().lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise VendorActivityError("transaction_population_sha256 must be a SHA-256 hex digest")
        object.__setattr__(self, "transaction_population_sha256", digest)

    @property
    def vendor_count(self) -> int:
        return len(self.activities)

    def get(self, identity: VendorIdentity | str) -> VendorActivity | None:
        key = identity.key if isinstance(identity, VendorIdentity) else str(identity)
        return self.activities.get(key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope.value,
            "total_transactions": self.total_transactions,
            "transactions_with_identity": self.transactions_with_identity,
            "transactions_without_identity": self.transactions_without_identity,
            "observed_start_date": None if self.observed_start_date is None else self.observed_start_date.isoformat(),
            "observed_end_date": None if self.observed_end_date is None else self.observed_end_date.isoformat(),
            "vendor_count": self.vendor_count,
            "transaction_population_sha256": self.transaction_population_sha256,
            "activities": [self.activities[key].as_dict() for key in sorted(self.activities)],
        }


@dataclass(slots=True)
class _Accumulator:
    identity: VendorIdentity
    transaction_count: int
    award_transaction_counts: dict[str, int]
    net: Decimal
    absolute: Decimal
    positive: Decimal
    deobligation: Decimal
    zero_count: int
    first_date: date
    last_date: date
    agencies: set[str]
    subtiers: set[str]
    pscs: set[str]
    naics: set[str]


class VendorActivityIndex:
    """One-pass vendor activity builder that never retains transaction objects."""

    def __init__(
        self,
        *,
        scope: VendorIdentityScope = VendorIdentityScope.ENTITY,
        policy: VendorActivityPolicy | None = None,
    ) -> None:
        self.scope = VendorIdentityScope(scope)
        self.policy = policy or VendorActivityPolicy()
        self._vendors: dict[str, _Accumulator] = {}
        self._transaction_fingerprints: dict[str, str] = {}
        self._population_fingerprints: list[str] = []
        self._total = 0
        self._with_identity = 0
        self._without_identity = 0
        self._start: date | None = None
        self._end: date | None = None

    def observe(self, transaction: ProcurementTransaction) -> None:
        """Check all configured failure conditions before mutating accumulated state."""

        if not isinstance(transaction, ProcurementTransaction):
            raise TypeError("transaction must be ProcurementTransaction")
        txid = transaction.transaction_id
        if txid in self._transaction_fingerprints:
            raise VendorActivityError(f"duplicate transaction_id in vendor activity population: {txid!r}")

        identity = resolve_vendor_identity(transaction, self.scope)
        fingerprint = _transaction_fingerprint(transaction, identity, self.scope)
        policy = self.policy
        if policy.max_transaction_ids is not None and len(self._transaction_fingerprints) >= policy.max_transaction_ids:
            raise VendorActivityError(f"transaction index exceeds max_transaction_ids={policy.max_transaction_ids}")

        accumulator = None if identity is None else self._vendors.get(identity.key)
        if identity is not None and accumulator is None:
            if policy.max_vendor_identities is not None and len(self._vendors) >= policy.max_vendor_identities:
                raise VendorActivityError(f"vendor index exceeds max_vendor_identities={policy.max_vendor_identities}")

        if identity is not None:
            current_awards = 0 if accumulator is None else len(accumulator.award_transaction_counts)
            is_new_award = accumulator is None or transaction.award_id not in accumulator.award_transaction_counts
            if is_new_award and policy.max_distinct_awards_per_vendor is not None and current_awards >= policy.max_distinct_awards_per_vendor:
                raise VendorActivityError(
                    f"vendor {identity.key!r} exceeds max_distinct_awards_per_vendor={policy.max_distinct_awards_per_vendor}"
                )
            if policy.max_context_values_per_vendor is not None:
                additions = _context_additions(transaction, accumulator)
                existing = 0 if accumulator is None else (
                    len(accumulator.agencies) + len(accumulator.subtiers) + len(accumulator.pscs) + len(accumulator.naics)
                )
                if existing + additions > policy.max_context_values_per_vendor:
                    raise VendorActivityError(
                        f"vendor {identity.key!r} exceeds max_context_values_per_vendor={policy.max_context_values_per_vendor}"
                    )

        self._transaction_fingerprints[txid] = fingerprint
        self._population_fingerprints.append(fingerprint)
        self._total += 1
        self._start = transaction.action_date if self._start is None else min(self._start, transaction.action_date)
        self._end = transaction.action_date if self._end is None else max(self._end, transaction.action_date)

        if identity is None:
            self._without_identity += 1
            return
        self._with_identity += 1

        amount = transaction.action_obligation
        accumulator = self._vendors.get(identity.key)
        if accumulator is None:
            accumulator = _Accumulator(
                identity=identity,
                transaction_count=0,
                award_transaction_counts={},
                net=Decimal(0),
                absolute=Decimal(0),
                positive=Decimal(0),
                deobligation=Decimal(0),
                zero_count=0,
                first_date=transaction.action_date,
                last_date=transaction.action_date,
                agencies=set(),
                subtiers=set(),
                pscs=set(),
                naics=set(),
            )
            self._vendors[identity.key] = accumulator

        accumulator.transaction_count += 1
        accumulator.award_transaction_counts[transaction.award_id] = (
            accumulator.award_transaction_counts.get(transaction.award_id, 0) + 1
        )
        accumulator.net += amount
        accumulator.absolute += abs(amount)
        if amount > 0:
            accumulator.positive += amount
        elif amount < 0:
            accumulator.deobligation += abs(amount)
        else:
            accumulator.zero_count += 1
        accumulator.first_date = min(accumulator.first_date, transaction.action_date)
        accumulator.last_date = max(accumulator.last_date, transaction.action_date)
        _add_context(transaction, accumulator)

    def observe_many(self, transactions: Iterable[ProcurementTransaction]) -> "VendorActivityIndex":
        for transaction in transactions:
            self.observe(transaction)
        return self

    def snapshot(self) -> VendorActivitySnapshot:
        activities = {
            key: _freeze_activity(accumulator)
            for key, accumulator in self._vendors.items()
        }
        population_digest = _digest(sorted(self._population_fingerprints))
        return VendorActivitySnapshot(
            scope=self.scope,
            total_transactions=self._total,
            transactions_with_identity=self._with_identity,
            transactions_without_identity=self._without_identity,
            observed_start_date=self._start,
            observed_end_date=self._end,
            activities=activities,
            transaction_population_sha256=population_digest,
        )


def _freeze_activity(accumulator: _Accumulator) -> VendorActivity:
    award_counts = tuple(accumulator.award_transaction_counts.values())
    single = sum(count == 1 for count in award_counts)
    multi = len(award_counts) - single
    return VendorActivity(
        identity=accumulator.identity,
        transaction_count=accumulator.transaction_count,
        distinct_award_count=len(award_counts),
        single_transaction_award_count=single,
        multi_transaction_award_count=multi,
        maximum_transactions_on_one_award=max(award_counts),
        net_action_obligation=accumulator.net,
        absolute_action_obligation_activity=accumulator.absolute,
        positive_action_obligation=accumulator.positive,
        deobligation_magnitude=accumulator.deobligation,
        zero_obligation_transaction_count=accumulator.zero_count,
        first_action_date=accumulator.first_date,
        last_action_date=accumulator.last_date,
        distinct_awarding_agency_count=len(accumulator.agencies),
        distinct_awarding_subtier_count=len(accumulator.subtiers),
        distinct_psc_count=len(accumulator.pscs),
        distinct_naics_count=len(accumulator.naics),
    )


def _context_additions(transaction: ProcurementTransaction, accumulator: _Accumulator | None) -> int:
    existing = (
        (set(), set(), set(), set())
        if accumulator is None
        else (accumulator.agencies, accumulator.subtiers, accumulator.pscs, accumulator.naics)
    )
    candidates = (
        _agency_key(transaction.awarding_agency_code, transaction.awarding_agency_name),
        _agency_key(transaction.awarding_subtier_agency_code, transaction.awarding_subtier_agency_name),
        _code(transaction.psc_code),
        _code(transaction.naics_code),
    )
    return sum(value is not None and value not in bucket for value, bucket in zip(candidates, existing, strict=True))


def _add_context(transaction: ProcurementTransaction, accumulator: _Accumulator) -> None:
    agency = _agency_key(transaction.awarding_agency_code, transaction.awarding_agency_name)
    subtier = _agency_key(transaction.awarding_subtier_agency_code, transaction.awarding_subtier_agency_name)
    psc = _code(transaction.psc_code)
    naics = _code(transaction.naics_code)
    if agency is not None:
        accumulator.agencies.add(agency)
    if subtier is not None:
        accumulator.subtiers.add(subtier)
    if psc is not None:
        accumulator.pscs.add(psc)
    if naics is not None:
        accumulator.naics.add(naics)


def _agency_key(code: str | None, name: str | None) -> str | None:
    normalized_code = _code(code)
    if normalized_code is not None:
        return f"code:{normalized_code}"
    if name is None:
        return None
    cleaned = " ".join(name.split()).casefold()
    return None if not cleaned else f"name:{cleaned}"


def _code(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().upper()
    return cleaned or None


def _transaction_fingerprint(
    transaction: ProcurementTransaction,
    identity: VendorIdentity | None,
    scope: VendorIdentityScope,
) -> str:
    return _digest(
        {
            "transaction_id": transaction.transaction_id,
            "award_id": transaction.award_id,
            "action_date": transaction.action_date.isoformat(),
            "action_obligation": _decimal_identity(transaction.action_obligation),
            "scope": scope.value,
            "identity_key": None if identity is None else identity.key,
            "agency": _agency_key(transaction.awarding_agency_code, transaction.awarding_agency_name),
            "subtier": _agency_key(transaction.awarding_subtier_agency_code, transaction.awarding_subtier_agency_name),
            "psc": _code(transaction.psc_code),
            "naics": _code(transaction.naics_code),
        }
    )


def _decimal_identity(value: Decimal) -> str:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise VendorActivityError("action_obligation must be a finite Decimal")
    if value.is_zero():
        return "0"
    sign, digits_tuple, exponent = value.as_tuple()
    digits = list(digits_tuple)
    while digits and digits[-1] == 0:
        digits.pop()
        exponent += 1
    return f"{sign}:{''.join(str(digit) for digit in digits)}:{exponent}"


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
