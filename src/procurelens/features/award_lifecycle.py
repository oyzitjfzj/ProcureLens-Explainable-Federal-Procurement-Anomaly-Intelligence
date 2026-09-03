"""Award lifecycle classification for ProcureLens.

Separates observed base-award actions from later modifications before vendor
frequency is interpreted as "awards won". USAspending/FPDS contract actions
report new awards as modification 0; later contract changes use other
modification numbers.

This module is deliberately narrow and descriptive. It does not count vendor
frequency, score risk, infer competition, or guess when the reported
modification number is missing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from hashlib import sha256
import json
from typing import Any

from procurelens.domain.transaction import ProcurementTransaction


class AwardLifecycleError(ValueError):
    """Raised when lifecycle evidence is internally inconsistent."""


class AwardActionKind(str, Enum):
    BASE_AWARD = "base_award"
    MODIFICATION = "modification"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class AwardActionEvidence:
    """One transaction's lifecycle position based on reported source facts."""

    transaction_id: str
    award_id: str
    action_date: date
    modification_number: str | None
    normalized_modification_number: str | None
    kind: AwardActionKind
    classification_reason: str

    def __post_init__(self) -> None:
        txid = self.transaction_id.strip()
        award_id = self.award_id.strip()
        reason = self.classification_reason.strip()
        if not txid or not award_id or not reason:
            raise AwardLifecycleError(
                "transaction_id, award_id, and classification_reason must not be blank"
            )
        object.__setattr__(self, "transaction_id", txid)
        object.__setattr__(self, "award_id", award_id)
        object.__setattr__(self, "kind", AwardActionKind(self.kind))
        object.__setattr__(self, "classification_reason", reason)

        raw = self.modification_number
        if raw is not None:
            raw = raw.strip() or None
            object.__setattr__(self, "modification_number", raw)

        normalized = self.normalized_modification_number
        if normalized is not None:
            normalized = normalized.strip().upper() or None
            object.__setattr__(self, "normalized_modification_number", normalized)

        if self.kind is AwardActionKind.UNKNOWN:
            if self.normalized_modification_number is not None:
                raise AwardLifecycleError(
                    "unknown lifecycle action must not carry a normalized modification number"
                )
        elif self.normalized_modification_number is None:
            raise AwardLifecycleError(
                "classified lifecycle action requires a normalized modification number"
            )

        if (
            self.kind is AwardActionKind.BASE_AWARD
            and self.normalized_modification_number != "0"
        ):
            raise AwardLifecycleError(
                "base-award evidence must normalize modification number to '0'"
            )
        if (
            self.kind is AwardActionKind.MODIFICATION
            and _is_zero_modification(self.normalized_modification_number)
        ):
            raise AwardLifecycleError(
                "modification evidence cannot carry a zero modification number"
            )

    @property
    def is_observed_new_award_action(self) -> bool:
        """True only when source evidence identifies an observed base award action."""

        return self.kind is AwardActionKind.BASE_AWARD

    @property
    def sha256_hex(self) -> str:
        return _digest(
            {
                "transaction_id": self.transaction_id,
                "award_id": self.award_id,
                "action_date": self.action_date.isoformat(),
                "modification_number": self.normalized_modification_number,
                "kind": self.kind.value,
                "classification_reason": self.classification_reason,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "award_id": self.award_id,
            "action_date": self.action_date.isoformat(),
            "modification_number": self.modification_number,
            "normalized_modification_number": self.normalized_modification_number,
            "kind": self.kind.value,
            "classification_reason": self.classification_reason,
            "is_observed_new_award_action": self.is_observed_new_award_action,
            "sha256": self.sha256_hex,
        }


def classify_award_action(
    transaction: ProcurementTransaction,
) -> AwardActionEvidence:
    """Classify lifecycle position only from the reported modification number.

    Missing modification numbers stay UNKNOWN. ProcureLens does not infer
    "new award" from action date, dollar amount, PIID shape, or transaction
    ordering because those are weaker signals than the reported lifecycle field.
    """

    if not isinstance(transaction, ProcurementTransaction):
        raise TypeError("transaction must be ProcurementTransaction")

    raw = transaction.modification_number
    normalized = None if raw is None else raw.strip().upper() or None

    if normalized is None:
        return AwardActionEvidence(
            transaction_id=transaction.transaction_id,
            award_id=transaction.award_id,
            action_date=transaction.action_date,
            modification_number=raw,
            normalized_modification_number=None,
            kind=AwardActionKind.UNKNOWN,
            classification_reason="modification_number_missing",
        )

    if _is_zero_modification(normalized):
        return AwardActionEvidence(
            transaction_id=transaction.transaction_id,
            award_id=transaction.award_id,
            action_date=transaction.action_date,
            modification_number=raw,
            normalized_modification_number="0",
            kind=AwardActionKind.BASE_AWARD,
            classification_reason="reported_modification_zero",
        )

    return AwardActionEvidence(
        transaction_id=transaction.transaction_id,
        award_id=transaction.award_id,
        action_date=transaction.action_date,
        modification_number=raw,
        normalized_modification_number=normalized,
        kind=AwardActionKind.MODIFICATION,
        classification_reason="reported_nonzero_modification_number",
    )


def _is_zero_modification(value: str | None) -> bool:
    """Accept only numeric zero spellings; never guess from unrelated text."""

    return (
        value is not None
        and bool(value)
        and value.isdigit()
        and all(character == "0" for character in value)
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
