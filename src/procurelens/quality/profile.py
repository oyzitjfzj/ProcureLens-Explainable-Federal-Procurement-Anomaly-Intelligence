"""Streaming, source-neutral data-quality profiling for ProcureLens.

This module measures what the canonical transaction population actually contains
before any anomaly model or risk policy is allowed to depend on it.

It intentionally does not invent a single "data quality score". Completeness,
identifier strength, competition coverage, source/schema mix, and missingness
patterns are separate facts. A later policy layer can decide whether those facts
are sufficient for a specific analysis.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import Any

from procurelens.domain.transaction import ProcurementTransaction


class QualityProfileError(ValueError):
    """Raised when quality profiling receives invalid canonical input."""


@dataclass(frozen=True, slots=True)
class Coverage:
    """Observed presence for one field or derived analysis context."""

    present: int
    total: int

    def __post_init__(self) -> None:
        if isinstance(self.present, bool) or not isinstance(self.present, int):
            raise QualityProfileError("coverage present must be an integer")
        if isinstance(self.total, bool) or not isinstance(self.total, int):
            raise QualityProfileError("coverage total must be an integer")
        if self.total < 0 or self.present < 0 or self.present > self.total:
            raise QualityProfileError("coverage counts are inconsistent")

    @property
    def missing(self) -> int:
        return self.total - self.present

    @property
    def rate(self) -> float | None:
        """Return present/total, or None for an empty population."""
        if self.total == 0:
            return None
        return self.present / self.total

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "present": self.present,
            "missing": self.missing,
            "total": self.total,
            "rate": self.rate,
        }


@dataclass(frozen=True, slots=True)
class NamedCount:
    """One deterministic category count."""

    name: str
    count: int

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise QualityProfileError("NamedCount name must not be blank")
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 0:
            raise QualityProfileError("NamedCount count must be a non-negative integer")
        object.__setattr__(self, "name", name)


@dataclass(frozen=True, slots=True)
class MissingContextPattern:
    """Frequency of one exact missing-analysis-context pattern."""

    missing_fields: tuple[str, ...]
    count: int

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.missing_fields))) != self.missing_fields:
            raise QualityProfileError(
                "missing_fields must be unique and sorted for deterministic reporting"
            )
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise QualityProfileError("missing-context count must be positive")


@dataclass(frozen=True, slots=True)
class TimeRange:
    """Observed inclusive range for dates or timestamps."""

    earliest: date | datetime | None
    latest: date | datetime | None

    def __post_init__(self) -> None:
        if (self.earliest is None) != (self.latest is None):
            raise QualityProfileError("time range must contain both endpoints or neither")
        if self.earliest is not None and self.latest is not None:
            if type(self.earliest) is not type(self.latest):
                raise QualityProfileError("time range endpoints must have the same type")
            if self.latest < self.earliest:
                raise QualityProfileError("time range latest precedes earliest")

    @property
    def empty(self) -> bool:
        return self.earliest is None


# These are canonical-domain fields, not source-column names or risk thresholds.
# Keeping the list here makes the quality surface explicit and reviewable.
_CANONICAL_COVERAGE_FIELDS: tuple[str, ...] = (
    "award_id",
    "transaction_id",
    "action_date",
    "action_obligation",
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
    "award_total_obligation",
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
    "number_of_offers_received",
    "other_than_full_and_open_code",
    "other_than_full_and_open_description",
    "solicitation_procedure_code",
    "solicitation_procedure_description",
)

_DERIVED_CONTEXT_NAMES: tuple[str, ...] = (
    "vendor_identity",
    "procurement_category",
    "awarding_agency",
    "competition_extent",
    "competition_offers",
    "competition_other_than_full_and_open",
    "competition_solicitation_procedure",
    "competition_any",
    "competition_core_pair",
)


@dataclass(frozen=True, slots=True)
class QualityProfile:
    """Immutable evidence about one canonical transaction population."""

    total_transactions: int
    field_coverage: Mapping[str, Coverage]
    context_coverage: Mapping[str, Coverage]

    vendor_identity_modes: tuple[NamedCount, ...]
    category_identity_modes: tuple[NamedCount, ...]
    agency_identity_modes: tuple[NamedCount, ...]
    competition_core_modes: tuple[NamedCount, ...]
    obligation_signs: tuple[NamedCount, ...]

    missing_context_patterns: tuple[MissingContextPattern, ...]
    source_names: tuple[NamedCount, ...]
    source_schemas: tuple[NamedCount, ...]
    action_fiscal_years: tuple[NamedCount, ...]

    action_date_range: TimeRange
    retrieval_time_range: TimeRange
    transactions_with_missing_analysis_context: int

    def __post_init__(self) -> None:
        total = self.total_transactions
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise QualityProfileError("total_transactions must be a non-negative integer")

        expected_fields = set(_CANONICAL_COVERAGE_FIELDS)
        if set(self.field_coverage) != expected_fields:
            raise QualityProfileError("field_coverage keys do not match canonical quality fields")
        if set(self.context_coverage) != set(_DERIVED_CONTEXT_NAMES):
            raise QualityProfileError("context_coverage keys do not match derived contexts")

        for coverage in (*self.field_coverage.values(), *self.context_coverage.values()):
            if coverage.total != total:
                raise QualityProfileError("all coverage metrics must use total_transactions")

        missing = self.transactions_with_missing_analysis_context
        if isinstance(missing, bool) or not isinstance(missing, int) or not (0 <= missing <= total):
            raise QualityProfileError(
                "transactions_with_missing_analysis_context is inconsistent"
            )

        complete_breakdowns = (
            ("vendor_identity_modes", self.vendor_identity_modes),
            ("category_identity_modes", self.category_identity_modes),
            ("agency_identity_modes", self.agency_identity_modes),
            ("competition_core_modes", self.competition_core_modes),
            ("obligation_signs", self.obligation_signs),
            ("source_names", self.source_names),
            ("source_schemas", self.source_schemas),
            ("action_fiscal_years", self.action_fiscal_years),
        )
        for name, items in complete_breakdowns:
            if sum(item.count for item in items) != total:
                raise QualityProfileError(f"{name} counts must sum to total_transactions")

        if sum(item.count for item in self.missing_context_patterns) != total:
            raise QualityProfileError(
                "missing_context_patterns counts must sum to total_transactions"
            )

        ranges_empty = self.action_date_range.empty and self.retrieval_time_range.empty
        if (total == 0) != ranges_empty:
            raise QualityProfileError(
                "time ranges must be empty exactly when the population is empty"
            )

        object.__setattr__(
            self,
            "field_coverage",
            MappingProxyType(dict(self.field_coverage)),
        )
        object.__setattr__(
            self,
            "context_coverage",
            MappingProxyType(dict(self.context_coverage)),
        )

    @property
    def transactions_with_complete_analysis_context(self) -> int:
        return self.total_transactions - self.transactions_with_missing_analysis_context

    @property
    def complete_analysis_context_coverage(self) -> Coverage:
        return Coverage(
            self.transactions_with_complete_analysis_context,
            self.total_transactions,
        )

    @property
    def schema_variant_count(self) -> int:
        return len(self.source_schemas)

    @property
    def source_count(self) -> int:
        return len(self.source_names)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation with deterministic ordering."""

        def counts(items: tuple[NamedCount, ...]) -> list[dict[str, Any]]:
            return [{"name": item.name, "count": item.count} for item in items]

        return {
            "total_transactions": self.total_transactions,
            "field_coverage": {
                name: metric.as_dict()
                for name, metric in self.field_coverage.items()
            },
            "context_coverage": {
                name: metric.as_dict()
                for name, metric in self.context_coverage.items()
            },
            "vendor_identity_modes": counts(self.vendor_identity_modes),
            "category_identity_modes": counts(self.category_identity_modes),
            "agency_identity_modes": counts(self.agency_identity_modes),
            "competition_core_modes": counts(self.competition_core_modes),
            "obligation_signs": counts(self.obligation_signs),
            "missing_context_patterns": [
                {
                    "missing_fields": list(item.missing_fields),
                    "count": item.count,
                }
                for item in self.missing_context_patterns
            ],
            "source_names": counts(self.source_names),
            "source_schemas": counts(self.source_schemas),
            "action_fiscal_years": counts(self.action_fiscal_years),
            "action_date_range": {
                "earliest": (
                    self.action_date_range.earliest.isoformat()
                    if self.action_date_range.earliest is not None
                    else None
                ),
                "latest": (
                    self.action_date_range.latest.isoformat()
                    if self.action_date_range.latest is not None
                    else None
                ),
            },
            "retrieval_time_range": {
                "earliest": (
                    self.retrieval_time_range.earliest.isoformat()
                    if self.retrieval_time_range.earliest is not None
                    else None
                ),
                "latest": (
                    self.retrieval_time_range.latest.isoformat()
                    if self.retrieval_time_range.latest is not None
                    else None
                ),
            },
            "transactions_with_missing_analysis_context": (
                self.transactions_with_missing_analysis_context
            ),
            "complete_analysis_context_coverage": (
                self.complete_analysis_context_coverage.as_dict()
            ),
            "schema_variant_count": self.schema_variant_count,
            "source_count": self.source_count,
        }


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _counter_tuple(counter: Counter[str]) -> tuple[NamedCount, ...]:
    return tuple(
        NamedCount(name, counter[name])
        for name in sorted(counter)
        if counter[name] > 0
    )


def _vendor_mode(tx: ProcurementTransaction) -> str:
    if tx.recipient_uei:
        return "uei"
    if tx.recipient_legacy_id:
        return "legacy_identifier"
    if tx.recipient_name:
        return "name_only"
    return "missing"


def _category_mode(tx: ProcurementTransaction) -> str:
    has_psc = bool(tx.psc_code)
    has_naics = bool(tx.naics_code)
    if has_psc and has_naics:
        return "psc_and_naics"
    if has_psc:
        return "psc_only"
    if has_naics:
        return "naics_only"
    return "missing"


def _agency_mode(tx: ProcurementTransaction) -> str:
    has_code = bool(tx.awarding_agency_code)
    has_name = bool(tx.awarding_agency_name)
    if has_code and has_name:
        return "code_and_name"
    if has_code:
        return "code_only"
    if has_name:
        return "name_only"
    return "missing"


def _competition_core_mode(tx: ProcurementTransaction) -> str:
    has_extent = bool(tx.extent_competed_code or tx.extent_competed_description)
    has_offers = tx.number_of_offers_received is not None
    if has_extent and has_offers:
        return "extent_and_offers"
    if has_extent:
        return "extent_only"
    if has_offers:
        return "offers_only"
    return "missing"


def _obligation_sign(tx: ProcurementTransaction) -> str:
    if tx.action_obligation > 0:
        return "positive"
    if tx.action_obligation < 0:
        return "negative"
    return "zero"


def _federal_fiscal_year(value: date) -> int:
    """Return the U.S. federal fiscal year containing an action date."""
    return value.year + 1 if value.month >= 10 else value.year


def _derived_contexts(tx: ProcurementTransaction) -> Mapping[str, bool]:
    competition_extent = bool(tx.extent_competed_code or tx.extent_competed_description)
    competition_offers = tx.number_of_offers_received is not None
    competition_other = bool(
        tx.other_than_full_and_open_code
        or tx.other_than_full_and_open_description
    )
    competition_solicitation = bool(
        tx.solicitation_procedure_code
        or tx.solicitation_procedure_description
    )
    return {
        "vendor_identity": tx.vendor_key is not None,
        "procurement_category": tx.category_key is not None,
        "awarding_agency": bool(tx.awarding_agency_code or tx.awarding_agency_name),
        "competition_extent": competition_extent,
        "competition_offers": competition_offers,
        "competition_other_than_full_and_open": competition_other,
        "competition_solicitation_procedure": competition_solicitation,
        "competition_any": (
            competition_extent
            or competition_offers
            or competition_other
            or competition_solicitation
        ),
        "competition_core_pair": competition_extent and competition_offers,
    }


class QualityAccumulator:
    """One-pass profiler that does not retain transaction objects."""

    __slots__ = (
        "_total",
        "_field_present",
        "_context_present",
        "_vendor_modes",
        "_category_modes",
        "_agency_modes",
        "_competition_modes",
        "_obligation_signs",
        "_missing_patterns",
        "_sources",
        "_schemas",
        "_action_fiscal_years",
        "_action_earliest",
        "_action_latest",
        "_retrieval_earliest",
        "_retrieval_latest",
        "_rows_with_missing_context",
    )

    def __init__(self) -> None:
        self._total = 0
        self._field_present: Counter[str] = Counter()
        self._context_present: Counter[str] = Counter()
        self._vendor_modes: Counter[str] = Counter()
        self._category_modes: Counter[str] = Counter()
        self._agency_modes: Counter[str] = Counter()
        self._competition_modes: Counter[str] = Counter()
        self._obligation_signs: Counter[str] = Counter()
        self._missing_patterns: Counter[tuple[str, ...]] = Counter()
        self._sources: Counter[str] = Counter()
        self._schemas: Counter[str] = Counter()
        self._action_fiscal_years: Counter[str] = Counter()
        self._action_earliest: date | None = None
        self._action_latest: date | None = None
        self._retrieval_earliest: datetime | None = None
        self._retrieval_latest: datetime | None = None
        self._rows_with_missing_context = 0

    @property
    def total_transactions(self) -> int:
        return self._total

    def observe(self, transaction: ProcurementTransaction) -> None:
        """Measure one already-normalized transaction."""

        if not isinstance(transaction, ProcurementTransaction):
            raise QualityProfileError(
                "quality profiler accepts ProcurementTransaction objects only"
            )

        self._total += 1

        for name in _CANONICAL_COVERAGE_FIELDS:
            if _is_present(getattr(transaction, name)):
                self._field_present[name] += 1

        for name, present in _derived_contexts(transaction).items():
            if present:
                self._context_present[name] += 1

        self._vendor_modes[_vendor_mode(transaction)] += 1
        self._category_modes[_category_mode(transaction)] += 1
        self._agency_modes[_agency_mode(transaction)] += 1
        self._competition_modes[_competition_core_mode(transaction)] += 1
        self._obligation_signs[_obligation_sign(transaction)] += 1

        missing = tuple(sorted(transaction.missing_analysis_fields()))
        self._missing_patterns[missing] += 1
        if missing:
            self._rows_with_missing_context += 1

        self._sources[transaction.lineage.source_name] += 1
        schema = transaction.lineage.source_schema or "<unspecified>"
        self._schemas[schema] += 1
        self._action_fiscal_years[str(_federal_fiscal_year(transaction.action_date))] += 1

        action = transaction.action_date
        if self._action_earliest is None or action < self._action_earliest:
            self._action_earliest = action
        if self._action_latest is None or action > self._action_latest:
            self._action_latest = action

        retrieved = transaction.lineage.retrieved_at
        if self._retrieval_earliest is None or retrieved < self._retrieval_earliest:
            self._retrieval_earliest = retrieved
        if self._retrieval_latest is None or retrieved > self._retrieval_latest:
            self._retrieval_latest = retrieved

    def observe_many(
        self,
        transactions: Iterable[ProcurementTransaction],
    ) -> "QualityAccumulator":
        """Consume a stream without materializing it and return self."""

        for transaction in transactions:
            self.observe(transaction)
        return self

    def snapshot(self) -> QualityProfile:
        """Freeze the current measurements into an immutable report."""

        total = self._total
        field_coverage = {
            name: Coverage(self._field_present[name], total)
            for name in _CANONICAL_COVERAGE_FIELDS
        }
        context_coverage = {
            name: Coverage(self._context_present[name], total)
            for name in _DERIVED_CONTEXT_NAMES
        }

        missing_patterns = tuple(
            MissingContextPattern(fields, count)
            for fields, count in sorted(
                self._missing_patterns.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if count > 0
        )

        return QualityProfile(
            total_transactions=total,
            field_coverage=field_coverage,
            context_coverage=context_coverage,
            vendor_identity_modes=_counter_tuple(self._vendor_modes),
            category_identity_modes=_counter_tuple(self._category_modes),
            agency_identity_modes=_counter_tuple(self._agency_modes),
            competition_core_modes=_counter_tuple(self._competition_modes),
            obligation_signs=_counter_tuple(self._obligation_signs),
            missing_context_patterns=missing_patterns,
            source_names=_counter_tuple(self._sources),
            source_schemas=_counter_tuple(self._schemas),
            action_fiscal_years=_counter_tuple(self._action_fiscal_years),
            action_date_range=TimeRange(self._action_earliest, self._action_latest),
            retrieval_time_range=TimeRange(
                self._retrieval_earliest,
                self._retrieval_latest,
            ),
            transactions_with_missing_analysis_context=(
                self._rows_with_missing_context
            ),
        )


def profile_transactions(
    transactions: Iterable[ProcurementTransaction],
) -> QualityProfile:
    """Profile a canonical transaction stream in one pass."""

    return QualityAccumulator().observe_many(transactions).snapshot()
