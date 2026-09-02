"""Canonical procurement transaction contract for ProcureLens.

This module deliberately contains no anomaly-model or fraud/risk policy. It
defines the normalized transaction shape that every source adapter, feature
builder, detector, scorer, explainer, and exporter must share.

USAspending stores transaction-level obligation changes separately from
award-level totals. ProcureLens keeps those concepts separate so downstream
models cannot silently treat a cumulative award value as a transaction amount.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Mapping


class TransactionContractError(ValueError):
    """Raised when a transaction violates a non-negotiable domain invariant."""


def _freeze(value: Any) -> Any:
    """Detach and recursively freeze flexible source attributes."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class SourceRecordRef:
    """Lineage metadata for one normalized procurement transaction."""

    source_name: str
    source_transaction_id: str
    retrieved_at: datetime
    source_schema: str | None = None
    raw_record_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.source_name.strip():
            raise TransactionContractError("source_name must not be blank")
        if not self.source_transaction_id.strip():
            raise TransactionContractError("source_transaction_id must not be blank")

        timestamp = self.retrieved_at
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise TransactionContractError("retrieved_at must be timezone-aware")

        object.__setattr__(self, "source_name", self.source_name.strip())
        object.__setattr__(
            self, "source_transaction_id", self.source_transaction_id.strip()
        )

        if self.source_schema is not None:
            object.__setattr__(self, "source_schema", self.source_schema.strip() or None)

        if self.raw_record_sha256 is not None:
            digest = self.raw_record_sha256.strip().lower()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise TransactionContractError(
                    "raw_record_sha256 must be a 64-character hexadecimal SHA-256"
                )
            object.__setattr__(self, "raw_record_sha256", digest)


@dataclass(frozen=True, slots=True)
class ProcurementTransaction:
    """Normalized prime-award contract transaction used throughout ProcureLens.

    The model is intentionally source-neutral. USAspending-specific column names
    belong in a source adapter, not in this domain contract.
    """

    lineage: SourceRecordRef

    # Stable identity.
    award_id: str
    transaction_id: str
    piid: str | None
    modification_number: str | None
    parent_award_id: str | None
    award_type_code: str | None

    # Recipient/vendor identity.
    recipient_name: str | None
    recipient_uei: str | None
    recipient_legacy_id: str | None
    parent_recipient_name: str | None
    parent_recipient_uei: str | None
    parent_recipient_legacy_id: str | None

    # Transaction economics. Negative action obligations are valid because
    # contract modifications may de-obligate previously obligated funds.
    action_date: date
    action_obligation: Decimal
    award_total_obligation: Decimal | None = None

    # Organizational context.
    awarding_agency_code: str | None = None
    awarding_agency_name: str | None = None
    awarding_subtier_agency_code: str | None = None
    awarding_subtier_agency_name: str | None = None
    awarding_office_code: str | None = None
    awarding_office_name: str | None = None
    funding_agency_code: str | None = None
    funding_agency_name: str | None = None

    # Procurement/category context.
    naics_code: str | None = None
    psc_code: str | None = None
    description: str | None = None

    # Competition context. These stay as reported facts; interpretation belongs
    # to a later, configurable competition-risk layer.
    extent_competed_code: str | None = None
    extent_competed_description: str | None = None
    number_of_offers_received: int | None = None
    other_than_full_and_open_code: str | None = None
    other_than_full_and_open_description: str | None = None
    solicitation_procedure_code: str | None = None
    solicitation_procedure_description: str | None = None

    # Forward-compatible payload for useful source fields that are not yet part
    # of the canonical contract. Adapters therefore do not have to discard
    # useful data just because the core model has not adopted a field yet.
    attributes: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.award_id.strip():
            raise TransactionContractError("award_id must not be blank")
        if not self.transaction_id.strip():
            raise TransactionContractError("transaction_id must not be blank")

        self._validate_decimal("action_obligation", self.action_obligation)
        if self.award_total_obligation is not None:
            self._validate_decimal(
                "award_total_obligation", self.award_total_obligation
            )

        offers = self.number_of_offers_received
        if offers is not None:
            if isinstance(offers, bool) or not isinstance(offers, int):
                raise TransactionContractError(
                    "number_of_offers_received must be an integer or None"
                )
            if offers < 0:
                raise TransactionContractError(
                    "number_of_offers_received must not be negative"
                )

        object.__setattr__(self, "award_id", self.award_id.strip())
        object.__setattr__(self, "transaction_id", self.transaction_id.strip())

        optional_text_fields = (
            "piid",
            "modification_number",
            "parent_award_id",
            "award_type_code",
            "recipient_name",
            "recipient_uei",
            "recipient_legacy_id",
            "parent_recipient_name",
            "parent_recipient_uei",
            "parent_recipient_legacy_id",
            "awarding_agency_code",
            "awarding_agency_name",
            "awarding_subtier_agency_code",
            "awarding_subtier_agency_name",
            "awarding_office_code",
            "awarding_office_name",
            "funding_agency_code",
            "funding_agency_name",
            "naics_code",
            "psc_code",
            "description",
            "extent_competed_code",
            "extent_competed_description",
            "other_than_full_and_open_code",
            "other_than_full_and_open_description",
            "solicitation_procedure_code",
            "solicitation_procedure_description",
        )
        for name in optional_text_fields:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, value.strip() or None)

        object.__setattr__(self, "attributes", _freeze(dict(self.attributes)))

    @staticmethod
    def _validate_decimal(name: str, value: Decimal) -> None:
        if not isinstance(value, Decimal):
            raise TransactionContractError(f"{name} must be Decimal")
        if not value.is_finite():
            raise TransactionContractError(f"{name} must be finite")

    @property
    def vendor_key(self) -> str | None:
        """Best available grouping key, preferring government identifiers."""

        if self.recipient_uei:
            return f"uei:{self.recipient_uei}"
        if self.recipient_legacy_id:
            return f"legacy:{self.recipient_legacy_id}"
        if self.recipient_name:
            return f"name:{self.recipient_name.casefold()}"
        return None

    @property
    def category_key(self) -> str | None:
        """Most specific available product/service grouping key."""

        if self.psc_code:
            return f"psc:{self.psc_code}"
        if self.naics_code:
            return f"naics:{self.naics_code}"
        return None

    def missing_analysis_fields(self) -> tuple[str, ...]:
        """Report missing context that weakens analysis without invalidating a row."""

        analysis_context = {
            "recipient_identity": self.vendor_key,
            "procurement_category": self.category_key,
            "awarding_agency": self.awarding_agency_code
            or self.awarding_agency_name,
            "extent_competed": self.extent_competed_code,
            "number_of_offers_received": self.number_of_offers_received,
        }
        return tuple(
            name for name, value in analysis_context.items() if value is None
        )


def utc_now() -> datetime:
    """Return an aware UTC timestamp for source-lineage creation."""

    return datetime.now(timezone.utc)
