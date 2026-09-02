"""USAspending transaction normalization for ProcureLens.

This module is deliberately pure: no HTTP, persistence, feature engineering,
anomaly scoring, or risk policy. It translates USAspending transaction-shaped
mappings into the canonical domain contract while preserving lineage and
failing closed when field meaning is ambiguous.

The default map targets USAspending transaction-search/bulk field names.
Alternative endpoint/export shapes can inject explicit aliases without changing
downstream analytics.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import re
from types import MappingProxyType
from typing import Any

from procurelens.domain.transaction import (
    ProcurementTransaction,
    SourceRecordRef,
    TransactionContractError,
)


class USAspendingAdapterError(TransactionContractError):
    """Raised when a source record cannot be normalized without guessing."""


_TEXT_FIELDS = frozenset(
    {
        "award_id",
        "transaction_id",
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
    }
)
_DECIMAL_FIELDS = frozenset({"action_obligation", "award_total_obligation"})
_DATE_FIELDS = frozenset({"action_date"})
_INTEGER_FIELDS = frozenset({"number_of_offers_received"})
_SUPPORTED_FIELDS = (
    _TEXT_FIELDS | _DECIMAL_FIELDS | _DATE_FIELDS | _INTEGER_FIELDS
)
_REQUIRED_FIELDS = frozenset(
    {"award_id", "transaction_id", "action_date", "action_obligation"}
)

# Source facts only; no fraud/risk semantics live here. Explicit aliases make
# upstream schema changes observable instead of silently changing meaning.
_DEFAULT_FIELD_ALIASES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "award_id": ("generated_unique_award_id",),
        "transaction_id": (
            "usaspending_unique_transaction_id",
            "transaction_unique_id",
        ),
        "piid": ("piid",),
        "modification_number": ("modification_number",),
        "parent_award_id": ("parent_award_id",),
        "award_type_code": ("type", "type_raw", "contract_award_type"),
        "recipient_name": ("recipient_name", "recipient_name_raw"),
        "recipient_uei": ("recipient_uei",),
        "recipient_legacy_id": ("recipient_unique_id",),
        "parent_recipient_name": (
            "parent_recipient_name",
            "parent_recipient_name_raw",
        ),
        "parent_recipient_uei": ("parent_uei",),
        "parent_recipient_legacy_id": ("parent_recipient_unique_id",),
        "action_date": ("action_date",),
        "action_obligation": ("federal_action_obligation",),
        "award_total_obligation": ("award_amount", "total_obligated_amount"),
        "awarding_agency_code": ("awarding_agency_code",),
        "awarding_agency_name": (
            "awarding_toptier_agency_name",
            "awarding_toptier_agency_name_raw",
        ),
        "awarding_subtier_agency_code": ("awarding_sub_tier_agency_c",),
        "awarding_subtier_agency_name": (
            "awarding_subtier_agency_name",
            "awarding_subtier_agency_name_raw",
        ),
        "awarding_office_code": ("awarding_office_code",),
        "awarding_office_name": ("awarding_office_name",),
        "funding_agency_code": ("funding_agency_code",),
        "funding_agency_name": (
            "funding_toptier_agency_name",
            "funding_toptier_agency_name_raw",
        ),
        "naics_code": ("naics_code",),
        "psc_code": ("product_or_service_code",),
        "description": ("transaction_description",),
        "extent_competed_code": ("extent_competed",),
        "extent_competed_description": ("extent_compete_description",),
        "number_of_offers_received": ("number_of_offers_received",),
        "other_than_full_and_open_code": ("other_than_full_and_open_c",),
        "other_than_full_and_open_description": ("other_than_full_and_o_desc",),
        "solicitation_procedure_code": ("solicitation_procedures",),
        "solicitation_procedure_description": ("solicitation_procedur_desc",),
    }
)

_KEY_SEPARATOR = re.compile(r"[^a-z0-9]+")


def _normalized_key(value: object) -> str:
    text = str(value).strip().casefold()
    return _KEY_SEPARATOR.sub("_", text).strip("_")


def _values_equivalent(left: Any, right: Any) -> bool:
    if left == right:
        return True
    if left is None or right is None:
        return False
    return str(left).strip() == str(right).strip()


@dataclass(frozen=True, slots=True)
class AdaptationDiagnostics:
    """Observable schema details from one normalization operation."""

    consumed_source_fields: tuple[str, ...]
    unmapped_source_fields: tuple[str, ...]
    fallback_aliases_used: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class AdaptationResult:
    transaction: ProcurementTransaction
    diagnostics: AdaptationDiagnostics


class _RecordView:
    """Collision-aware view over one source mapping."""

    __slots__ = ("_record", "_index", "_consumed", "_fallbacks")

    def __init__(self, record: Mapping[str, Any]) -> None:
        self._record = record
        self._index: dict[str, list[tuple[str, Any]]] = {}
        self._consumed: set[str] = set()
        self._fallbacks: list[tuple[str, str]] = []

        for key, value in record.items():
            original = str(key)
            normalized = _normalized_key(original)
            if not normalized:
                raise USAspendingAdapterError("source record contains a blank field name")
            self._index.setdefault(normalized, []).append((original, value))

    @staticmethod
    def _present(value: Any) -> bool:
        return value is not None and (
            not isinstance(value, str) or bool(value.strip())
        )

    def resolve(self, canonical: str, aliases: Sequence[str]) -> Any:
        for alias_index, alias in enumerate(aliases):
            matches = self._index.get(_normalized_key(alias), ())
            if not matches:
                continue

            present = [(key, value) for key, value in matches if self._present(value)]
            self._consumed.update(key for key, _ in matches)
            if not present:
                continue

            first_key, first_value = present[0]
            conflicting = [
                key
                for key, value in present[1:]
                if not _values_equivalent(first_value, value)
            ]
            if conflicting:
                fields = ", ".join((first_key, *conflicting))
                raise USAspendingAdapterError(
                    f"ambiguous source fields for {canonical}: {fields}"
                )

            if alias_index:
                self._fallbacks.append((canonical, first_key))
            return first_value
        return None

    def unmapped(self) -> dict[str, Any]:
        return {
            str(key): value
            for key, value in self._record.items()
            if str(key) not in self._consumed
        }

    def diagnostics(self) -> AdaptationDiagnostics:
        all_fields = {str(key) for key in self._record}
        return AdaptationDiagnostics(
            consumed_source_fields=tuple(sorted(self._consumed)),
            unmapped_source_fields=tuple(sorted(all_fields - self._consumed)),
            fallback_aliases_used=tuple(self._fallbacks),
        )


def _text(value: Any, field: str, required: bool) -> str | None:
    if value is None:
        if required:
            raise USAspendingAdapterError(f"{field} is required")
        return None
    parsed = str(value).strip()
    if not parsed:
        if required:
            raise USAspendingAdapterError(f"{field} is required")
        return None
    return parsed


def _decimal(value: Any, field: str, required: bool) -> Decimal | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise USAspendingAdapterError(f"{field} is required")
        return None
    if isinstance(value, bool):
        raise USAspendingAdapterError(f"{field} must be numeric, not boolean")

    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise USAspendingAdapterError(
            f"{field} cannot be parsed as an exact decimal: {value!r}"
        ) from exc
    if not parsed.is_finite():
        raise USAspendingAdapterError(f"{field} must be finite")
    return parsed


def _integer(value: Any, field: str, required: bool) -> int | None:
    parsed = _decimal(value, field, required)
    if parsed is None:
        return None
    integral = parsed.to_integral_value()
    if parsed != integral:
        raise USAspendingAdapterError(
            f"{field} must be an integer-valued number: {value!r}"
        )
    return int(integral)


def _date(value: Any, field: str, required: bool) -> date | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise USAspendingAdapterError(f"{field} is required")
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    text = str(value).strip()
    try:
        return date.fromisoformat(text)
    except ValueError:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError as exc:
            raise USAspendingAdapterError(
                f"{field} must be an ISO-8601 date or datetime: {value!r}"
            ) from exc


def _parse_field(field: str, value: Any) -> Any:
    required = field in _REQUIRED_FIELDS
    if field in _TEXT_FIELDS:
        return _text(value, field, required)
    if field in _DECIMAL_FIELDS:
        return _decimal(value, field, required)
    if field in _DATE_FIELDS:
        return _date(value, field, required)
    if field in _INTEGER_FIELDS:
        return _integer(value, field, required)
    raise USAspendingAdapterError(f"no parser registered for {field}")


def _aware_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise USAspendingAdapterError("retrieved_at must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise USAspendingAdapterError("retrieved_at must be timezone-aware")
    return value.astimezone(timezone.utc)


def _raw_digest(raw_record: bytes | str | None) -> str | None:
    if raw_record is None:
        return None
    payload = raw_record.encode("utf-8") if isinstance(raw_record, str) else raw_record
    if not isinstance(payload, bytes):
        raise USAspendingAdapterError("raw_record must be bytes, str, or None")
    return sha256(payload).hexdigest()


def _clean_aliases(field: str, aliases: Sequence[str]) -> tuple[str, ...]:
    if isinstance(aliases, (str, bytes)):
        raise USAspendingAdapterError(
            f"{field} aliases must be a sequence of field names, not a string"
        )

    cleaned: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        if not isinstance(alias, str):
            raise USAspendingAdapterError(f"{field} aliases must all be strings")
        value = alias.strip()
        normalized = _normalized_key(value)
        if value and normalized not in seen:
            cleaned.append(value)
            seen.add(normalized)

    if not cleaned:
        raise USAspendingAdapterError(f"{field} must have a source alias")
    return tuple(cleaned)


def _validate_alias_ownership(
    aliases: Mapping[str, Sequence[str]],
) -> None:
    owner_by_alias: dict[str, str] = {}
    for field, source_aliases in aliases.items():
        for alias in source_aliases:
            normalized = _normalized_key(alias)
            previous = owner_by_alias.get(normalized)
            if previous is not None and previous != field:
                raise USAspendingAdapterError(
                    "one source alias cannot map to multiple canonical fields: "
                    f"{alias!r} -> {previous}, {field}"
                )
            owner_by_alias[normalized] = field


class USAspendingTransactionAdapter:
    """Normalize USAspending contract transactions into ProcureLens.

    Mapping overrides are explicit and reviewable rather than fuzzy. Unknown
    source fields may be retained in `attributes`; the default leaves them out
    of the in-memory canonical object so hundreds of raw columns are not copied
    across every row of a 10k-25k+ transaction dataset.
    """

    __slots__ = ("_aliases", "preserve_unmapped_fields", "source_name")

    def __init__(
        self,
        *,
        field_aliases: Mapping[str, Sequence[str]] | None = None,
        preserve_unmapped_fields: bool = False,
        source_name: str = "USAspending.gov",
    ) -> None:
        aliases = {
            field: _clean_aliases(field, values)
            for field, values in _DEFAULT_FIELD_ALIASES.items()
        }

        if field_aliases:
            for field, values in field_aliases.items():
                if field not in _SUPPORTED_FIELDS:
                    raise USAspendingAdapterError(
                        f"unsupported canonical field mapping: {field}"
                    )
                aliases[field] = _clean_aliases(field, values)

        missing = _REQUIRED_FIELDS - aliases.keys()
        if missing:
            raise USAspendingAdapterError(
                "required canonical fields have no source mapping: "
                + ", ".join(sorted(missing))
            )
        _validate_alias_ownership(aliases)

        cleaned_source_name = source_name.strip()
        if not cleaned_source_name:
            raise USAspendingAdapterError("source_name must not be blank")

        self._aliases = MappingProxyType(aliases)
        self.preserve_unmapped_fields = bool(preserve_unmapped_fields)
        self.source_name = cleaned_source_name

    @property
    def field_aliases(self) -> Mapping[str, tuple[str, ...]]:
        return self._aliases

    def adapt(
        self,
        record: Mapping[str, Any],
        *,
        retrieved_at: datetime,
        source_schema: str | None = None,
        raw_record: bytes | str | None = None,
    ) -> ProcurementTransaction:
        return self.adapt_with_diagnostics(
            record,
            retrieved_at=retrieved_at,
            source_schema=source_schema,
            raw_record=raw_record,
        ).transaction

    def adapt_with_diagnostics(
        self,
        record: Mapping[str, Any],
        *,
        retrieved_at: datetime,
        source_schema: str | None = None,
        raw_record: bytes | str | None = None,
    ) -> AdaptationResult:
        if not isinstance(record, Mapping):
            raise USAspendingAdapterError("record must be a mapping")

        view = _RecordView(record)
        values = {
            field: _parse_field(field, view.resolve(field, self._aliases[field]))
            for field in _SUPPORTED_FIELDS
        }

        # The parser enforces these. This guard protects the domain constructor
        # if required-field policy and parser registration ever drift apart.
        for field in _REQUIRED_FIELDS:
            if values[field] is None:
                raise USAspendingAdapterError(
                    f"required field normalization failed: {field}"
                )

        transaction_id = values["transaction_id"]
        if not isinstance(transaction_id, str):
            raise USAspendingAdapterError("transaction_id normalization failed")

        transaction = ProcurementTransaction(
            lineage=SourceRecordRef(
                source_name=self.source_name,
                source_transaction_id=transaction_id,
                retrieved_at=_aware_utc(retrieved_at),
                source_schema=_text(source_schema, "source_schema", False),
                raw_record_sha256=_raw_digest(raw_record),
            ),
            **values,
            attributes=view.unmapped() if self.preserve_unmapped_fields else {},
        )
        return AdaptationResult(transaction, view.diagnostics())

    def adapt_many(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        retrieved_at: datetime,
        source_schema: str | None = None,
    ) -> Iterable[ProcurementTransaction]:
        """Lazily normalize a batch using one explicit retrieval timestamp."""

        for record in records:
            yield self.adapt(
                record,
                retrieved_at=retrieved_at,
                source_schema=source_schema,
            )
