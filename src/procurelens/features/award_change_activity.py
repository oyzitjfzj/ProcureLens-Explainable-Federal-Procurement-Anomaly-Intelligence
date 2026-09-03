"""Award-level contract change activity for ProcureLens.

Aggregates observed base actions, later modifications, lifecycle-unknown actions,
and modification obligation flows without deciding whether any change is risky
or justified. Contract modifications are common legal procurement actions, so
this module records evidence only.

Award-total obligation is never summed across transactions. Additive change
activity uses transaction-level action_obligation, preserving deobligations.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any

from procurelens.domain.transaction import ProcurementTransaction
from procurelens.features.award_lifecycle import (
    AwardActionKind,
    classify_award_action,
)


class AwardChangeActivityError(ValueError):
    """Raised when award-change activity evidence or state is inconsistent."""


@dataclass(frozen=True, slots=True)
class AwardChangePolicy:
    """Optional resource budgets only; no risk thresholds live here."""

    max_transaction_ids: int | None = None
    max_awards: int | None = None
    max_distinct_modification_numbers_per_award: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_transaction_ids",
            "max_awards",
            "max_distinct_modification_numbers_per_award",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise AwardChangeActivityError(
                    f"{name} must be a positive integer or None"
                )


@dataclass(frozen=True, slots=True)
class AwardChangeActivity:
    """Observed lifecycle and modification-money facts for one award."""

    award_id: str
    transaction_count: int
    base_award_action_count: int
    modification_action_count: int
    lifecycle_unknown_action_count: int

    first_action_date: date
    last_action_date: date
    earliest_base_action_date: date | None
    latest_base_action_date: date | None
    first_modification_date: date | None
    last_modification_date: date | None

    distinct_modification_number_count: int
    maximum_transactions_on_one_modification_number: int

    net_modification_obligation: Decimal
    absolute_modification_obligation_activity: Decimal
    positive_modification_obligation: Decimal
    deobligation_magnitude: Decimal
    zero_modification_obligation_count: int
    maximum_absolute_modification_obligation: Decimal | None

    def __post_init__(self) -> None:
        award_id = self.award_id.strip()
        if not award_id:
            raise AwardChangeActivityError("award_id must not be blank")
        object.__setattr__(self, "award_id", award_id)

        counts = (
            self.transaction_count,
            self.base_award_action_count,
            self.modification_action_count,
            self.lifecycle_unknown_action_count,
            self.distinct_modification_number_count,
            self.maximum_transactions_on_one_modification_number,
            self.zero_modification_obligation_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise AwardChangeActivityError("activity counts must be non-negative integers")
        if self.transaction_count < 1:
            raise AwardChangeActivityError("activity requires at least one transaction")
        if (
            self.base_award_action_count
            + self.modification_action_count
            + self.lifecycle_unknown_action_count
            != self.transaction_count
        ):
            raise AwardChangeActivityError("lifecycle counts do not sum to transaction_count")
        if self.distinct_modification_number_count > self.modification_action_count:
            raise AwardChangeActivityError(
                "distinct modification numbers cannot exceed modification actions"
            )
        if self.modification_action_count == 0:
            if (
                self.distinct_modification_number_count != 0
                or self.maximum_transactions_on_one_modification_number != 0
                or self.first_modification_date is not None
                or self.last_modification_date is not None
                or self.maximum_absolute_modification_obligation is not None
            ):
                raise AwardChangeActivityError(
                    "award without modifications cannot carry modification detail"
                )
        elif (
            self.distinct_modification_number_count < 1
            or self.maximum_transactions_on_one_modification_number < 1
            or self.first_modification_date is None
            or self.last_modification_date is None
            or self.maximum_absolute_modification_obligation is None
        ):
            raise AwardChangeActivityError(
                "observed modifications require modification detail"
            )

        if self.first_action_date > self.last_action_date:
            raise AwardChangeActivityError("action-date range is inconsistent")
        if (self.earliest_base_action_date is None) != (self.latest_base_action_date is None):
            raise AwardChangeActivityError("base-action date range must be complete or absent")
        if (
            self.earliest_base_action_date is not None
            and self.earliest_base_action_date > self.latest_base_action_date
        ):
            raise AwardChangeActivityError("base-action date range is inconsistent")
        if (
            self.first_modification_date is not None
            and self.first_modification_date > self.last_modification_date
        ):
            raise AwardChangeActivityError("modification date range is inconsistent")

        money = (
            self.net_modification_obligation,
            self.absolute_modification_obligation_activity,
            self.positive_modification_obligation,
            self.deobligation_magnitude,
        )
        if any(not isinstance(value, Decimal) or not value.is_finite() for value in money):
            raise AwardChangeActivityError("modification money must be finite Decimal")
        if (
            self.absolute_modification_obligation_activity < 0
            or self.positive_modification_obligation < 0
            or self.deobligation_magnitude < 0
        ):
            raise AwardChangeActivityError("modification activity magnitudes cannot be negative")
        if (
            self.absolute_modification_obligation_activity
            != self.positive_modification_obligation + self.deobligation_magnitude
        ):
            raise AwardChangeActivityError("absolute modification activity is inconsistent")
        if (
            self.net_modification_obligation
            != self.positive_modification_obligation - self.deobligation_magnitude
        ):
            raise AwardChangeActivityError("net modification obligation is inconsistent")

        maximum = self.maximum_absolute_modification_obligation
        if maximum is not None and (
            not isinstance(maximum, Decimal)
            or not maximum.is_finite()
            or maximum < 0
        ):
            raise AwardChangeActivityError(
                "maximum absolute modification obligation must be non-negative Decimal or None"
            )

    @property
    def has_observed_base_action(self) -> bool:
        return self.base_award_action_count > 0

    @property
    def has_multiple_observed_base_actions(self) -> bool:
        return self.base_award_action_count > 1

    @property
    def observed_days_base_to_first_modification(self) -> int | None:
        if self.earliest_base_action_date is None or self.first_modification_date is None:
            return None
        return (self.first_modification_date - self.earliest_base_action_date).days

    @property
    def modification_transactions_per_distinct_number(self) -> Decimal | None:
        if self.distinct_modification_number_count == 0:
            return None
        return Decimal(self.modification_action_count) / Decimal(
            self.distinct_modification_number_count
        )

    @property
    def evidence_sha256(self) -> str:
        return _digest(self.as_dict(include_evidence_sha=False))

    def as_dict(self, *, include_evidence_sha: bool = True) -> dict[str, Any]:
        result = {
            "award_id": self.award_id,
            "transaction_count": self.transaction_count,
            "base_award_action_count": self.base_award_action_count,
            "modification_action_count": self.modification_action_count,
            "lifecycle_unknown_action_count": self.lifecycle_unknown_action_count,
            "first_action_date": self.first_action_date.isoformat(),
            "last_action_date": self.last_action_date.isoformat(),
            "earliest_base_action_date": _date_text(self.earliest_base_action_date),
            "latest_base_action_date": _date_text(self.latest_base_action_date),
            "first_modification_date": _date_text(self.first_modification_date),
            "last_modification_date": _date_text(self.last_modification_date),
            "distinct_modification_number_count": self.distinct_modification_number_count,
            "maximum_transactions_on_one_modification_number":
                self.maximum_transactions_on_one_modification_number,
            "net_modification_obligation": str(self.net_modification_obligation),
            "absolute_modification_obligation_activity":
                str(self.absolute_modification_obligation_activity),
            "positive_modification_obligation":
                str(self.positive_modification_obligation),
            "deobligation_magnitude": str(self.deobligation_magnitude),
            "zero_modification_obligation_count":
                self.zero_modification_obligation_count,
            "maximum_absolute_modification_obligation":
                None
                if self.maximum_absolute_modification_obligation is None
                else str(self.maximum_absolute_modification_obligation),
            "has_observed_base_action": self.has_observed_base_action,
            "has_multiple_observed_base_actions":
                self.has_multiple_observed_base_actions,
            "observed_days_base_to_first_modification":
                self.observed_days_base_to_first_modification,
            "modification_transactions_per_distinct_number":
                None
                if self.modification_transactions_per_distinct_number is None
                else str(self.modification_transactions_per_distinct_number),
        }
        if include_evidence_sha:
            result["evidence_sha256"] = self.evidence_sha256
        return result


@dataclass(slots=True)
class _Accumulator:
    transaction_count: int = 0
    base_award_action_count: int = 0
    modification_action_count: int = 0
    lifecycle_unknown_action_count: int = 0
    first_action_date: date | None = None
    last_action_date: date | None = None
    earliest_base_action_date: date | None = None
    latest_base_action_date: date | None = None
    first_modification_date: date | None = None
    last_modification_date: date | None = None
    modification_number_counts: dict[str, int] = field(default_factory=dict)
    net_modification_obligation: Decimal = Decimal(0)
    absolute_modification_obligation_activity: Decimal = Decimal(0)
    positive_modification_obligation: Decimal = Decimal(0)
    deobligation_magnitude: Decimal = Decimal(0)
    zero_modification_obligation_count: int = 0
    maximum_absolute_modification_obligation: Decimal | None = None

    def add(self, transaction: ProcurementTransaction, kind: AwardActionKind, normalized: str | None) -> None:
        self.transaction_count += 1
        self.first_action_date = _earlier(self.first_action_date, transaction.action_date)
        self.last_action_date = _later(self.last_action_date, transaction.action_date)

        if kind is AwardActionKind.BASE_AWARD:
            self.base_award_action_count += 1
            self.earliest_base_action_date = _earlier(
                self.earliest_base_action_date, transaction.action_date
            )
            self.latest_base_action_date = _later(
                self.latest_base_action_date, transaction.action_date
            )
            return

        if kind is AwardActionKind.UNKNOWN:
            self.lifecycle_unknown_action_count += 1
            return

        assert normalized is not None
        self.modification_action_count += 1
        self.first_modification_date = _earlier(
            self.first_modification_date, transaction.action_date
        )
        self.last_modification_date = _later(
            self.last_modification_date, transaction.action_date
        )
        self.modification_number_counts[normalized] = (
            self.modification_number_counts.get(normalized, 0) + 1
        )

        amount = transaction.action_obligation
        self.net_modification_obligation += amount
        self.absolute_modification_obligation_activity += abs(amount)
        if amount > 0:
            self.positive_modification_obligation += amount
        elif amount < 0:
            self.deobligation_magnitude += -amount
        else:
            self.zero_modification_obligation_count += 1
        magnitude = abs(amount)
        if (
            self.maximum_absolute_modification_obligation is None
            or magnitude > self.maximum_absolute_modification_obligation
        ):
            self.maximum_absolute_modification_obligation = magnitude

    def freeze(self, award_id: str) -> AwardChangeActivity:
        if self.transaction_count < 1 or self.first_action_date is None or self.last_action_date is None:
            raise AwardChangeActivityError("cannot freeze empty award-change accumulator")
        return AwardChangeActivity(
            award_id=award_id,
            transaction_count=self.transaction_count,
            base_award_action_count=self.base_award_action_count,
            modification_action_count=self.modification_action_count,
            lifecycle_unknown_action_count=self.lifecycle_unknown_action_count,
            first_action_date=self.first_action_date,
            last_action_date=self.last_action_date,
            earliest_base_action_date=self.earliest_base_action_date,
            latest_base_action_date=self.latest_base_action_date,
            first_modification_date=self.first_modification_date,
            last_modification_date=self.last_modification_date,
            distinct_modification_number_count=len(self.modification_number_counts),
            maximum_transactions_on_one_modification_number=(
                max(self.modification_number_counts.values())
                if self.modification_number_counts else 0
            ),
            net_modification_obligation=self.net_modification_obligation,
            absolute_modification_obligation_activity=
                self.absolute_modification_obligation_activity,
            positive_modification_obligation=self.positive_modification_obligation,
            deobligation_magnitude=self.deobligation_magnitude,
            zero_modification_obligation_count=self.zero_modification_obligation_count,
            maximum_absolute_modification_obligation=
                self.maximum_absolute_modification_obligation,
        )


@dataclass(frozen=True, slots=True)
class AwardChangeSnapshot:
    total_transactions_seen: int
    award_count: int
    base_award_action_count: int
    modification_action_count: int
    lifecycle_unknown_action_count: int
    observed_start_date: date | None
    observed_end_date: date | None
    transaction_population_sha256: str
    activities: Mapping[str, AwardChangeActivity]

    def __post_init__(self) -> None:
        counts = (
            self.total_transactions_seen,
            self.award_count,
            self.base_award_action_count,
            self.modification_action_count,
            self.lifecycle_unknown_action_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise AwardChangeActivityError("snapshot counts must be non-negative integers")
        if (
            self.base_award_action_count
            + self.modification_action_count
            + self.lifecycle_unknown_action_count
            != self.total_transactions_seen
        ):
            raise AwardChangeActivityError("snapshot lifecycle counts do not sum to total")
        if (self.observed_start_date is None) != (self.observed_end_date is None):
            raise AwardChangeActivityError("snapshot date range must be complete or absent")
        if (
            self.observed_start_date is not None
            and self.observed_start_date > self.observed_end_date
        ):
            raise AwardChangeActivityError("snapshot date range is inconsistent")
        object.__setattr__(
            self,
            "transaction_population_sha256",
            _validate_digest(
                self.transaction_population_sha256,
                "transaction_population_sha256",
            ),
        )
        activities = dict(self.activities)
        if len(activities) != self.award_count:
            raise AwardChangeActivityError("award_count differs from activity mapping")
        if any(key != activity.award_id for key, activity in activities.items()):
            raise AwardChangeActivityError("activity mapping key differs from award_id")
        object.__setattr__(self, "activities", MappingProxyType(activities))

    def get(self, award_id: str) -> AwardChangeActivity | None:
        return self.activities.get(award_id)


class AwardChangeIndex:
    """One-pass award-change aggregator that does not retain full transactions."""

    def __init__(self, *, policy: AwardChangePolicy | None = None) -> None:
        self.policy = policy or AwardChangePolicy()
        self._transaction_fingerprints: dict[str, str] = {}
        self._awards: dict[str, _Accumulator] = {}
        self._base_count = 0
        self._modification_count = 0
        self._unknown_count = 0
        self._start: date | None = None
        self._end: date | None = None

    def observe(self, transaction: ProcurementTransaction) -> None:
        if not isinstance(transaction, ProcurementTransaction):
            raise TypeError("transaction must be ProcurementTransaction")

        txid = transaction.transaction_id
        if txid in self._transaction_fingerprints:
            raise AwardChangeActivityError(
                f"duplicate transaction_id in award-change activity: {txid!r}"
            )
        limit = self.policy.max_transaction_ids
        if limit is not None and len(self._transaction_fingerprints) >= limit:
            raise AwardChangeActivityError(
                f"transaction index exceeds max_transaction_ids={limit}"
            )

        award_id = transaction.award_id
        accumulator = self._awards.get(award_id)
        if accumulator is None:
            award_limit = self.policy.max_awards
            if award_limit is not None and len(self._awards) >= award_limit:
                raise AwardChangeActivityError(
                    f"award index exceeds max_awards={award_limit}"
                )

        lifecycle = classify_award_action(transaction)
        normalized = lifecycle.normalized_modification_number
        if lifecycle.kind is AwardActionKind.MODIFICATION:
            existing_numbers = (
                set() if accumulator is None
                else set(accumulator.modification_number_counts)
            )
            if normalized not in existing_numbers:
                number_limit = self.policy.max_distinct_modification_numbers_per_award
                if (
                    number_limit is not None
                    and len(existing_numbers) >= number_limit
                ):
                    raise AwardChangeActivityError(
                        "award exceeds "
                        "max_distinct_modification_numbers_per_award="
                        f"{number_limit}"
                    )

        fingerprint = _transaction_fingerprint(transaction, lifecycle.sha256_hex)

        # Commit only after every configured guard has passed.
        if accumulator is None:
            accumulator = _Accumulator()
            self._awards[award_id] = accumulator
        accumulator.add(transaction, lifecycle.kind, normalized)
        self._transaction_fingerprints[txid] = fingerprint
        self._start = _earlier(self._start, transaction.action_date)
        self._end = _later(self._end, transaction.action_date)
        if lifecycle.kind is AwardActionKind.BASE_AWARD:
            self._base_count += 1
        elif lifecycle.kind is AwardActionKind.MODIFICATION:
            self._modification_count += 1
        else:
            self._unknown_count += 1

    def observe_many(
        self, transactions: Iterable[ProcurementTransaction]
    ) -> "AwardChangeIndex":
        for transaction in transactions:
            self.observe(transaction)
        return self

    def snapshot(self) -> AwardChangeSnapshot:
        return AwardChangeSnapshot(
            total_transactions_seen=len(self._transaction_fingerprints),
            award_count=len(self._awards),
            base_award_action_count=self._base_count,
            modification_action_count=self._modification_count,
            lifecycle_unknown_action_count=self._unknown_count,
            observed_start_date=self._start,
            observed_end_date=self._end,
            transaction_population_sha256=_population_digest(
                self._transaction_fingerprints
            ),
            activities={
                award_id: accumulator.freeze(award_id)
                for award_id, accumulator in self._awards.items()
            },
        )


def _transaction_fingerprint(
    transaction: ProcurementTransaction,
    lifecycle_sha256: str,
) -> str:
    return _digest({
        "transaction_id": transaction.transaction_id,
        "award_id": transaction.award_id,
        "action_date": transaction.action_date.isoformat(),
        "action_obligation": str(transaction.action_obligation),
        "lifecycle_sha256": lifecycle_sha256,
    })


def _population_digest(fingerprints: Mapping[str, str]) -> str:
    return _digest([
        (txid, fingerprints[txid])
        for txid in sorted(fingerprints)
    ])


def _earlier(current: date | None, candidate: date) -> date:
    return candidate if current is None or candidate < current else current


def _later(current: date | None, candidate: date) -> date:
    return candidate if current is None or candidate > current else current


def _date_text(value: date | None) -> str | None:
    return None if value is None else value.isoformat()


def _validate_digest(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise AwardChangeActivityError(f"{name} must be a SHA-256 hex digest")
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
