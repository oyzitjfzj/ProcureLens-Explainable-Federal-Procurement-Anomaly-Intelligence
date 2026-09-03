"""Typed export records for ProcureLens review results.

Binds canonical procurement transactions to verified investigator explanations
without performing file-format serialization. Export records preserve source
lineage, procurement facts, transparent anomaly review priority, explicit flag
semantics, explanation provenance, and structured evidence facts.

The 0-100 value is a relative anomaly review-priority score, not a probability
of fraud, corruption, collusion, or misconduct. File-format quoting and
spreadsheet-safety policy belong to export.writer.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from hashlib import sha256
import json
from typing import Any, Iterable

from procurelens.domain.transaction import ProcurementTransaction
from procurelens.review.explanation import (
    EXPLANATION_SEMANTICS,
    ExplanationBatch,
    ExplanationFeatureFact,
    ReviewExplanation,
)
from procurelens.review.score import SCORE_SEMANTICS


class ExportRecordError(ValueError):
    """Raised when export-record evidence or provenance is inconsistent."""


@dataclass(frozen=True, slots=True)
class ExportEvidenceFact:
    source: str
    family: str
    name: str
    description: str
    value: Decimal | None
    unavailable_reason: str | None
    narrative: str

    def __post_init__(self) -> None:
        for field_name in (
            "source",
            "family",
            "name",
            "description",
            "narrative",
        ):
            text = getattr(self, field_name).strip()
            if not text:
                raise ExportRecordError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)
        if self.value is None:
            reason = (
                None
                if self.unavailable_reason is None
                else self.unavailable_reason.strip()
            )
            if not reason:
                raise ExportRecordError(
                    f"{self.name}: unavailable export fact requires a reason"
                )
            object.__setattr__(self, "unavailable_reason", reason)
        else:
            if (
                not isinstance(self.value, Decimal)
                or not self.value.is_finite()
            ):
                raise ExportRecordError(
                    f"{self.name}: export fact value must be finite Decimal"
                )
            if self.unavailable_reason is not None:
                raise ExportRecordError(
                    f"{self.name}: available export fact cannot carry a reason"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "family": self.family,
            "name": self.name,
            "description": self.description,
            "value": None if self.value is None else str(self.value),
            "unavailable_reason": self.unavailable_reason,
            "narrative": self.narrative,
        }


@dataclass(frozen=True, slots=True)
class ProcureLensExportRecord:
    """One canonical transaction plus its exact review/explanation evidence."""

    transaction_id: str
    award_id: str
    piid: str | None
    modification_number: str | None
    action_date: date

    vendor_name: str | None
    vendor_uei: str | None
    vendor_legacy_id: str | None

    awarding_agency_name: str | None
    awarding_subtier_agency_name: str | None
    psc_code: str | None
    naics_code: str | None
    award_type_code: str | None

    action_obligation: Decimal
    award_total_obligation: Decimal | None

    extent_competed_description: str | None
    number_of_offers_received: int | None
    solicitation_procedure_description: str | None
    other_than_full_and_open_description: str | None

    review_priority_score_lower: Decimal
    review_priority_score: Decimal
    review_priority_score_upper: Decimal
    anomaly_position: Decimal
    score_semantics: str

    flagged_for_review: bool
    review_rank_lower: int
    review_rank_upper: int
    selection_reason: str

    detector_disagreement_points: Decimal
    feature_completeness_fraction: Decimal
    stability_available: bool
    stability_position_span_points: Decimal | None
    stability_median_absolute_deviation_points: Decimal | None

    explanation_summary: str
    explanation_semantics: str
    evidence_facts: tuple[ExportEvidenceFact, ...]

    source_name: str
    source_transaction_id: str
    source_schema: str | None
    source_retrieved_at: datetime
    raw_record_sha256: str | None

    transaction_evidence_sha256: str
    explanation_evidence_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "transaction_id",
            "award_id",
            "selection_reason",
            "explanation_summary",
            "source_name",
            "source_transaction_id",
        ):
            text = getattr(self, field_name).strip()
            if not text:
                raise ExportRecordError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)

        for field_name in (
            "piid",
            "modification_number",
            "vendor_name",
            "vendor_uei",
            "vendor_legacy_id",
            "awarding_agency_name",
            "awarding_subtier_agency_name",
            "psc_code",
            "naics_code",
            "award_type_code",
            "extent_competed_description",
            "solicitation_procedure_description",
            "other_than_full_and_open_description",
            "source_schema",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, value.strip() or None)

        if not isinstance(self.action_date, date):
            raise ExportRecordError("action_date must be date")
        for field_name in (
            "action_obligation",
            "review_priority_score_lower",
            "review_priority_score",
            "review_priority_score_upper",
            "anomaly_position",
            "detector_disagreement_points",
            "feature_completeness_fraction",
        ):
            value = getattr(self, field_name)
            if (
                not isinstance(value, Decimal)
                or not value.is_finite()
            ):
                raise ExportRecordError(
                    f"{field_name} must be finite Decimal"
                )
        if self.award_total_obligation is not None and (
            not isinstance(self.award_total_obligation, Decimal)
            or not self.award_total_obligation.is_finite()
        ):
            raise ExportRecordError(
                "award_total_obligation must be finite Decimal or None"
            )

        _score(self.review_priority_score_lower, "review_priority_score_lower")
        _score(self.review_priority_score, "review_priority_score")
        _score(self.review_priority_score_upper, "review_priority_score_upper")
        if not (
            self.review_priority_score_lower
            <= self.review_priority_score
            <= self.review_priority_score_upper
        ):
            raise ExportRecordError(
                "review-priority score interval is inconsistent"
            )
        _fraction(self.anomaly_position, "anomaly_position")
        if self.anomaly_position * Decimal(100) != self.review_priority_score:
            raise ExportRecordError(
                "anomaly_position differs from 0-100 review score"
            )
        if self.score_semantics.strip() != SCORE_SEMANTICS:
            raise ExportRecordError("unsupported score semantics")
        object.__setattr__(self, "score_semantics", SCORE_SEMANTICS)

        if not isinstance(self.flagged_for_review, bool):
            raise ExportRecordError("flagged_for_review must be bool")
        for field_name in ("review_rank_lower", "review_rank_upper"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ExportRecordError(
                    f"{field_name} must be a positive integer"
                )
        if self.review_rank_lower > self.review_rank_upper:
            raise ExportRecordError("review rank interval is inconsistent")
        _score(
            self.detector_disagreement_points,
            "detector_disagreement_points",
        )
        _fraction(
            self.feature_completeness_fraction,
            "feature_completeness_fraction",
        )

        if not isinstance(self.stability_available, bool):
            raise ExportRecordError("stability_available must be bool")
        optional_stability = (
            self.stability_position_span_points,
            self.stability_median_absolute_deviation_points,
        )
        if self.stability_available:
            if any(value is None for value in optional_stability):
                raise ExportRecordError(
                    "available stability requires complete point measures"
                )
            for field_name, value in (
                (
                    "stability_position_span_points",
                    self.stability_position_span_points,
                ),
                (
                    "stability_median_absolute_deviation_points",
                    self.stability_median_absolute_deviation_points,
                ),
            ):
                assert value is not None
                _score(value, field_name)
        elif any(value is not None for value in optional_stability):
            raise ExportRecordError(
                "unavailable stability cannot carry stability point measures"
            )

        if self.explanation_semantics.strip() != EXPLANATION_SEMANTICS:
            raise ExportRecordError("unsupported explanation semantics")
        object.__setattr__(
            self, "explanation_semantics", EXPLANATION_SEMANTICS
        )

        facts = tuple(self.evidence_facts)
        names = tuple(fact.name for fact in facts)
        if len(names) != len(set(names)):
            raise ExportRecordError(
                "export evidence facts contain duplicate names"
            )
        object.__setattr__(self, "evidence_facts", facts)

        offers = self.number_of_offers_received
        if offers is not None and (
            isinstance(offers, bool)
            or not isinstance(offers, int)
            or offers < 0
        ):
            raise ExportRecordError(
                "number_of_offers_received must be non-negative int or None"
            )

        timestamp = self.source_retrieved_at
        if (
            not isinstance(timestamp, datetime)
            or timestamp.tzinfo is None
            or timestamp.utcoffset() is None
        ):
            raise ExportRecordError(
                "source_retrieved_at must be timezone-aware datetime"
            )

        if self.raw_record_sha256 is not None:
            object.__setattr__(
                self,
                "raw_record_sha256",
                _digest_hex(self.raw_record_sha256, "raw_record_sha256"),
            )
        for field_name in (
            "transaction_evidence_sha256",
            "explanation_evidence_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest_hex(getattr(self, field_name), field_name),
            )

    @property
    def identity(self) -> tuple[str, str]:
        return self.transaction_id, self.award_id

    @property
    def review_rank_text(self) -> str:
        if self.review_rank_lower == self.review_rank_upper:
            return str(self.review_rank_lower)
        return f"{self.review_rank_lower}-{self.review_rank_upper}"

    @property
    def evidence_narratives(self) -> tuple[str, ...]:
        return tuple(fact.narrative for fact in self.evidence_facts)

    @property
    def evidence_sha256(self) -> str:
        return _digest(self.as_dict(include_sha=False))

    def as_dict(self, *, include_sha: bool = True) -> dict[str, Any]:
        result = {
            "transaction_id": self.transaction_id,
            "award_id": self.award_id,
            "piid": self.piid,
            "modification_number": self.modification_number,
            "action_date": self.action_date.isoformat(),
            "vendor": {
                "name": self.vendor_name,
                "uei": self.vendor_uei,
                "legacy_id": self.vendor_legacy_id,
            },
            "awarding": {
                "agency_name": self.awarding_agency_name,
                "subtier_agency_name": self.awarding_subtier_agency_name,
            },
            "category": {
                "psc_code": self.psc_code,
                "naics_code": self.naics_code,
                "award_type_code": self.award_type_code,
            },
            "amounts": {
                "action_obligation": str(self.action_obligation),
                "award_total_obligation": (
                    None
                    if self.award_total_obligation is None
                    else str(self.award_total_obligation)
                ),
            },
            "competition": {
                "extent_competed_description":
                    self.extent_competed_description,
                "number_of_offers_received":
                    self.number_of_offers_received,
                "solicitation_procedure_description":
                    self.solicitation_procedure_description,
                "other_than_full_and_open_description":
                    self.other_than_full_and_open_description,
            },
            "review": {
                "review_priority_score_lower":
                    str(self.review_priority_score_lower),
                "review_priority_score":
                    str(self.review_priority_score),
                "review_priority_score_upper":
                    str(self.review_priority_score_upper),
                "anomaly_position": str(self.anomaly_position),
                "score_semantics": self.score_semantics,
                "flagged_for_review": self.flagged_for_review,
                "review_rank_lower": self.review_rank_lower,
                "review_rank_upper": self.review_rank_upper,
                "selection_reason": self.selection_reason,
                "detector_disagreement_points":
                    str(self.detector_disagreement_points),
                "feature_completeness_fraction":
                    str(self.feature_completeness_fraction),
                "stability_available": self.stability_available,
                "stability_position_span_points": _decimal_text(
                    self.stability_position_span_points
                ),
                "stability_median_absolute_deviation_points":
                    _decimal_text(
                        self.stability_median_absolute_deviation_points
                    ),
            },
            "explanation": {
                "summary": self.explanation_summary,
                "semantics": self.explanation_semantics,
                "evidence_facts": [
                    fact.as_dict() for fact in self.evidence_facts
                ],
            },
            "source": {
                "source_name": self.source_name,
                "source_transaction_id": self.source_transaction_id,
                "source_schema": self.source_schema,
                "retrieved_at": self.source_retrieved_at.isoformat(),
                "raw_record_sha256": self.raw_record_sha256,
            },
            "provenance": {
                "transaction_evidence_sha256":
                    self.transaction_evidence_sha256,
                "explanation_evidence_sha256":
                    self.explanation_evidence_sha256,
            },
        }
        if include_sha:
            result["record_evidence_sha256"] = self.evidence_sha256
        return result


@dataclass(frozen=True, slots=True)
class ExportRecordBatch:
    explanation_batch_sha256: str
    records: tuple[ProcureLensExportRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "explanation_batch_sha256",
            _digest_hex(
                self.explanation_batch_sha256,
                "explanation_batch_sha256",
            ),
        )
        records = tuple(self.records)
        if not records:
            raise ExportRecordError(
                "export record batch requires at least one record"
            )
        identities = tuple(record.identity for record in records)
        if len(identities) != len(set(identities)):
            raise ExportRecordError(
                "export record batch contains duplicate identities"
            )
        object.__setattr__(self, "records", records)

    @property
    def row_count(self) -> int:
        return len(self.records)

    @property
    def evidence_sha256(self) -> str:
        return _digest(
            {
                "explanation_batch_sha256":
                    self.explanation_batch_sha256,
                "records": [
                    record.evidence_sha256 for record in self.records
                ],
            }
        )


def build_export_records(
    transactions: Iterable[ProcurementTransaction],
    explanations: ExplanationBatch,
) -> ExportRecordBatch:
    """Bind exact canonical transactions to the matching explanation population."""

    if isinstance(transactions, (str, bytes)):
        raise ExportRecordError(
            "transactions must be an iterable of ProcurementTransaction"
        )
    items = tuple(transactions)
    if not items:
        raise ExportRecordError(
            "at least one canonical transaction is required"
        )
    if any(not isinstance(item, ProcurementTransaction) for item in items):
        raise TypeError("all transactions must be ProcurementTransaction")
    if not isinstance(explanations, ExplanationBatch):
        raise TypeError("explanations must be ExplanationBatch")

    by_identity: dict[
        tuple[str, str], ProcurementTransaction
    ] = {}
    for transaction in items:
        identity = (
            transaction.transaction_id,
            transaction.award_id,
        )
        if identity in by_identity:
            raise ExportRecordError(
                f"duplicate canonical transaction identity: {identity!r}"
            )
        by_identity[identity] = transaction

    explanation_identities = tuple(
        row.identity for row in explanations.rows
    )
    if set(by_identity) != set(explanation_identities):
        raise ExportRecordError(
            "canonical transaction population must exactly match explanations"
        )

    records = tuple(
        _build_record(by_identity[row.identity], row)
        for row in explanations.rows
    )
    return ExportRecordBatch(
        explanation_batch_sha256=explanations.evidence_sha256,
        records=records,
    )


def _build_record(
    transaction: ProcurementTransaction,
    explanation: ReviewExplanation,
) -> ProcureLensExportRecord:
    if (
        transaction.transaction_id != explanation.transaction_id
        or transaction.award_id != explanation.award_id
    ):
        raise ExportRecordError(
            "transaction and explanation identities differ"
        )

    facts = tuple(
        _export_fact(fact)
        for fact in explanation.feature_facts
    )
    lineage = transaction.lineage
    return ProcureLensExportRecord(
        transaction_id=transaction.transaction_id,
        award_id=transaction.award_id,
        piid=transaction.piid,
        modification_number=transaction.modification_number,
        action_date=transaction.action_date,
        vendor_name=transaction.recipient_name,
        vendor_uei=transaction.recipient_uei,
        vendor_legacy_id=transaction.recipient_legacy_id,
        awarding_agency_name=transaction.awarding_agency_name,
        awarding_subtier_agency_name=
            transaction.awarding_subtier_agency_name,
        psc_code=transaction.psc_code,
        naics_code=transaction.naics_code,
        award_type_code=transaction.award_type_code,
        action_obligation=transaction.action_obligation,
        award_total_obligation=transaction.award_total_obligation,
        extent_competed_description=
            transaction.extent_competed_description,
        number_of_offers_received=
            transaction.number_of_offers_received,
        solicitation_procedure_description=
            transaction.solicitation_procedure_description,
        other_than_full_and_open_description=
            transaction.other_than_full_and_open_description,
        review_priority_score_lower=
            explanation.review_score_lower,
        review_priority_score=explanation.review_score,
        review_priority_score_upper=
            explanation.review_score_upper,
        anomaly_position=
            explanation.review_score / Decimal(100),
        score_semantics=SCORE_SEMANTICS,
        flagged_for_review=explanation.flagged_for_review,
        review_rank_lower=explanation.rank_lower,
        review_rank_upper=explanation.rank_upper,
        selection_reason=explanation.selection_reason,
        detector_disagreement_points=
            explanation.detector_disagreement_points,
        feature_completeness_fraction=
            explanation.feature_completeness_fraction,
        stability_available=explanation.stability_available,
        stability_position_span_points=
            explanation.stability_position_span_points,
        stability_median_absolute_deviation_points=
            explanation.stability_median_absolute_deviation_points,
        explanation_summary=explanation.summary_text,
        explanation_semantics=EXPLANATION_SEMANTICS,
        evidence_facts=facts,
        source_name=lineage.source_name,
        source_transaction_id=lineage.source_transaction_id,
        source_schema=lineage.source_schema,
        source_retrieved_at=lineage.retrieved_at,
        raw_record_sha256=lineage.raw_record_sha256,
        transaction_evidence_sha256=_transaction_digest(transaction),
        explanation_evidence_sha256=explanation.evidence_sha256,
    )


def _export_fact(fact: ExplanationFeatureFact) -> ExportEvidenceFact:
    return ExportEvidenceFact(
        source=fact.source.value,
        family=fact.family,
        name=fact.name,
        description=fact.description,
        value=fact.value,
        unavailable_reason=fact.unavailable_reason,
        narrative=fact.narrative,
    )


def _transaction_digest(transaction: ProcurementTransaction) -> str:
    return _digest(
        {
            "transaction_id": transaction.transaction_id,
            "award_id": transaction.award_id,
            "source_name": transaction.lineage.source_name,
            "source_transaction_id":
                transaction.lineage.source_transaction_id,
            "raw_record_sha256":
                transaction.lineage.raw_record_sha256,
            "action_date": transaction.action_date.isoformat(),
            "action_obligation": str(transaction.action_obligation),
            "award_total_obligation": (
                None
                if transaction.award_total_obligation is None
                else str(transaction.award_total_obligation)
            ),
        }
    )


def _fraction(value: Decimal, name: str) -> None:
    if (
        not isinstance(value, Decimal)
        or not value.is_finite()
        or value < Decimal(0)
        or value > Decimal(1)
    ):
        raise ExportRecordError(
            f"{name} must be finite Decimal in [0, 1]"
        )


def _score(value: Decimal, name: str) -> None:
    if (
        not isinstance(value, Decimal)
        or not value.is_finite()
        or value < Decimal(0)
        or value > Decimal(100)
    ):
        raise ExportRecordError(
            f"{name} must be finite Decimal in [0, 100]"
        )


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _digest_hex(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ExportRecordError(
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
