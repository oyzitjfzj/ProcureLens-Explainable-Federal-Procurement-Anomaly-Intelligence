"""Reported competition evidence for ProcureLens.

Normalizes competition facts already present on a procurement transaction.
Competition process, solicitation procedure, offer outcome, and any reported
other-than-full-and-open authority remain separate evidence dimensions.

No anomaly threshold, risk score, or fraud/collusion conclusion lives here.
In particular, one received offer is an observed outcome, not proof that the
procurement process itself was noncompetitive.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import re
from typing import Any

from procurelens.domain.transaction import ProcurementTransaction


class CompetitionEvidenceError(ValueError):
    pass


class CompetitionExtentKind(str, Enum):
    COMPETED_UNDER_SAP = "competed_under_sap"
    FULL_AND_OPEN = "full_and_open"
    FULL_AND_OPEN_AFTER_EXCLUSION = "full_and_open_after_exclusion"
    NOT_AVAILABLE = "not_available_for_competition"
    NOT_COMPETED_UNDER_SAP = "not_competed_under_sap"
    NOT_COMPETED = "not_competed"
    UNKNOWN = "unknown"


class SolicitationProcedureKind(str, Enum):
    SIMPLIFIED_ACQUISITION = "simplified_acquisition"
    ONLY_ONE_SOURCE = "only_one_source_solicited"
    NEGOTIATED_PROPOSAL_QUOTE = "negotiated_proposal_quote"
    SEALED_BID = "sealed_bid"
    TWO_STEP = "two_step"
    ARCHITECT_ENGINEER = "architect_engineer"
    BASIC_RESEARCH = "basic_research"
    ALTERNATIVE_SOURCES = "alternative_sources"
    MULTIPLE_AWARD_FAIR_OPPORTUNITY = "multiple_award_fair_opportunity"
    UNKNOWN = "unknown"


class OfferOutcomeKind(str, Enum):
    ZERO_REPORTED = "zero_reported"
    SINGLE_OFFER = "single_offer"
    MULTIPLE_OFFERS = "multiple_offers"
    UNKNOWN = "unknown"


class OtherThanFullOpenAuthorityKind(str, Enum):
    NONE_REPORTED = "none_reported"
    UNIQUE_SOURCE = "unique_source"
    FOLLOW_ON = "follow_on_contract"
    UNSOLICITED_RESEARCH = "unsolicited_research_proposal"
    PATENT_OR_DATA_RIGHTS = "patent_or_data_rights"
    UTILITIES = "utilities"
    STANDARDIZATION = "standardization"
    ONLY_ONE_SOURCE_OTHER = "only_one_source_other"
    URGENCY = "urgency"
    MOBILIZATION_CAPABILITY_OR_EXPERT = "mobilization_capability_or_expert_services"
    INTERNATIONAL_AGREEMENT = "international_agreement"
    AUTHORIZED_BY_STATUTE = "authorized_by_statute"
    AUTHORIZED_FOR_RESALE = "authorized_for_resale"
    NATIONAL_SECURITY = "national_security"
    PUBLIC_INTEREST = "public_interest"
    SAP_NONCOMPETITION = "sap_noncompetition"
    UNKNOWN_REPORTED = "unknown_reported"


_EXTENT = {
    "competed under sap": CompetitionExtentKind.COMPETED_UNDER_SAP,
    "full and open competition": CompetitionExtentKind.FULL_AND_OPEN,
    "full and open competition after exclusion of sources":
        CompetitionExtentKind.FULL_AND_OPEN_AFTER_EXCLUSION,
    "not available for competition": CompetitionExtentKind.NOT_AVAILABLE,
    "not competed under sap": CompetitionExtentKind.NOT_COMPETED_UNDER_SAP,
    "not competed": CompetitionExtentKind.NOT_COMPETED,
}

_PROCEDURE = {
    "simplified acquisition": SolicitationProcedureKind.SIMPLIFIED_ACQUISITION,
    "only one source solicited": SolicitationProcedureKind.ONLY_ONE_SOURCE,
    "only one source": SolicitationProcedureKind.ONLY_ONE_SOURCE,
    "negotiated proposal quote":
        SolicitationProcedureKind.NEGOTIATED_PROPOSAL_QUOTE,
    "sealed bid": SolicitationProcedureKind.SEALED_BID,
    "two step": SolicitationProcedureKind.TWO_STEP,
    "architect engineer far 6 102":
        SolicitationProcedureKind.ARCHITECT_ENGINEER,
    "architect engineer": SolicitationProcedureKind.ARCHITECT_ENGINEER,
    "basic research": SolicitationProcedureKind.BASIC_RESEARCH,
    "alternative sources": SolicitationProcedureKind.ALTERNATIVE_SOURCES,
    "subject to multiple award fair opportunity":
        SolicitationProcedureKind.MULTIPLE_AWARD_FAIR_OPPORTUNITY,
}

_AUTHORITY = {
    "unique source": OtherThanFullOpenAuthorityKind.UNIQUE_SOURCE,
    "follow on contract": OtherThanFullOpenAuthorityKind.FOLLOW_ON,
    "unsolicited research proposal":
        OtherThanFullOpenAuthorityKind.UNSOLICITED_RESEARCH,
    "patent data rights": OtherThanFullOpenAuthorityKind.PATENT_OR_DATA_RIGHTS,
    "utilities": OtherThanFullOpenAuthorityKind.UTILITIES,
    "standardization": OtherThanFullOpenAuthorityKind.STANDARDIZATION,
    "only one source other": OtherThanFullOpenAuthorityKind.ONLY_ONE_SOURCE_OTHER,
    "urgency": OtherThanFullOpenAuthorityKind.URGENCY,
    "unusual and compelling urgency": OtherThanFullOpenAuthorityKind.URGENCY,
    "mobilization essential r d capability or expert services":
        OtherThanFullOpenAuthorityKind.MOBILIZATION_CAPABILITY_OR_EXPERT,
    "international agreement":
        OtherThanFullOpenAuthorityKind.INTERNATIONAL_AGREEMENT,
    "authorized by statute":
        OtherThanFullOpenAuthorityKind.AUTHORIZED_BY_STATUTE,
    "authorized for resale":
        OtherThanFullOpenAuthorityKind.AUTHORIZED_FOR_RESALE,
    "national security": OtherThanFullOpenAuthorityKind.NATIONAL_SECURITY,
    "public interest": OtherThanFullOpenAuthorityKind.PUBLIC_INTEREST,
    "sap non competition": OtherThanFullOpenAuthorityKind.SAP_NONCOMPETITION,
    "sap noncompetition": OtherThanFullOpenAuthorityKind.SAP_NONCOMPETITION,
}


@dataclass(frozen=True, slots=True)
class CompetitionEvidence:
    transaction_id: str
    award_id: str

    extent_code: str | None
    extent_description: str | None
    extent_kind: CompetitionExtentKind

    solicitation_procedure_code: str | None
    solicitation_procedure_description: str | None
    solicitation_procedure_kind: SolicitationProcedureKind

    number_of_offers_received: int | None
    offer_outcome_kind: OfferOutcomeKind

    other_than_full_open_code: str | None
    other_than_full_open_description: str | None
    other_than_full_open_authority_kind: OtherThanFullOpenAuthorityKind

    reported_process_competitive: bool | None
    reported_sources_restricted_or_unavailable: bool | None
    evidence_conflicts: tuple[str, ...]
    evidence_notes: tuple[str, ...]
    missing_core_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        txid, award_id = self.transaction_id.strip(), self.award_id.strip()
        if not txid or not award_id:
            raise CompetitionEvidenceError(
                "transaction_id and award_id must not be blank"
            )
        object.__setattr__(self, "transaction_id", txid)
        object.__setattr__(self, "award_id", award_id)
        object.__setattr__(self, "extent_kind", CompetitionExtentKind(self.extent_kind))
        object.__setattr__(
            self,
            "solicitation_procedure_kind",
            SolicitationProcedureKind(self.solicitation_procedure_kind),
        )
        object.__setattr__(
            self, "offer_outcome_kind", OfferOutcomeKind(self.offer_outcome_kind)
        )
        object.__setattr__(
            self,
            "other_than_full_open_authority_kind",
            OtherThanFullOpenAuthorityKind(self.other_than_full_open_authority_kind),
        )

        for name in (
            "extent_code",
            "extent_description",
            "solicitation_procedure_code",
            "solicitation_procedure_description",
            "other_than_full_open_code",
            "other_than_full_open_description",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, value.strip() or None)

        offers = self.number_of_offers_received
        if offers is not None and (
            isinstance(offers, bool) or not isinstance(offers, int) or offers < 0
        ):
            raise CompetitionEvidenceError(
                "number_of_offers_received must be a non-negative integer or None"
            )

        if _offer_kind(offers) is not self.offer_outcome_kind:
            raise CompetitionEvidenceError("offer outcome does not match offer count")
        if _process_competitive(self.extent_kind) is not self.reported_process_competitive:
            raise CompetitionEvidenceError("process flag does not match extent kind")
        if _sources_restricted(self.extent_kind) is not self.reported_sources_restricted_or_unavailable:
            raise CompetitionEvidenceError("restriction flag does not match extent kind")

        for name in ("evidence_conflicts", "evidence_notes", "missing_core_fields"):
            values = tuple(item.strip() for item in getattr(self, name) if item.strip())
            if len(values) != len(set(values)):
                raise CompetitionEvidenceError(f"{name} contains duplicates")
            object.__setattr__(self, name, values)

    @property
    def single_offer_reported(self) -> bool | None:
        return (
            None
            if self.number_of_offers_received is None
            else self.number_of_offers_received == 1
        )

    @property
    def other_than_full_open_authority_reported(self) -> bool:
        return (
            self.other_than_full_open_authority_kind
            is not OtherThanFullOpenAuthorityKind.NONE_REPORTED
        )

    @property
    def has_conflicting_evidence(self) -> bool:
        return bool(self.evidence_conflicts)

    @property
    def evidence_sha256(self) -> str:
        return _digest(self.as_dict(include_evidence_sha=False))

    def as_dict(self, *, include_evidence_sha: bool = True) -> dict[str, Any]:
        result = {
            "transaction_id": self.transaction_id,
            "award_id": self.award_id,
            "extent": {
                "code": self.extent_code,
                "description": self.extent_description,
                "kind": self.extent_kind.value,
            },
            "solicitation_procedure": {
                "code": self.solicitation_procedure_code,
                "description": self.solicitation_procedure_description,
                "kind": self.solicitation_procedure_kind.value,
            },
            "offers": {
                "number_received": self.number_of_offers_received,
                "outcome_kind": self.offer_outcome_kind.value,
                "single_offer_reported": self.single_offer_reported,
            },
            "other_than_full_and_open": {
                "code": self.other_than_full_open_code,
                "description": self.other_than_full_open_description,
                "authority_kind": self.other_than_full_open_authority_kind.value,
                "authority_reported": self.other_than_full_open_authority_reported,
            },
            "reported_process_competitive": self.reported_process_competitive,
            "reported_sources_restricted_or_unavailable":
                self.reported_sources_restricted_or_unavailable,
            "evidence_conflicts": list(self.evidence_conflicts),
            "evidence_notes": list(self.evidence_notes),
            "missing_core_fields": list(self.missing_core_fields),
        }
        if include_evidence_sha:
            result["evidence_sha256"] = self.evidence_sha256
        return result


def build_competition_evidence(
    transaction: ProcurementTransaction,
) -> CompetitionEvidence:
    if not isinstance(transaction, ProcurementTransaction):
        raise TypeError("transaction must be ProcurementTransaction")

    extent = _classify(transaction.extent_competed_description, _EXTENT,
                       CompetitionExtentKind.UNKNOWN)
    procedure = _classify(
        transaction.solicitation_procedure_description,
        _PROCEDURE,
        SolicitationProcedureKind.UNKNOWN,
    )
    authority = _authority(transaction.other_than_full_and_open_description)
    offers = transaction.number_of_offers_received

    return CompetitionEvidence(
        transaction_id=transaction.transaction_id,
        award_id=transaction.award_id,
        extent_code=transaction.extent_competed_code,
        extent_description=transaction.extent_competed_description,
        extent_kind=extent,
        solicitation_procedure_code=transaction.solicitation_procedure_code,
        solicitation_procedure_description=transaction.solicitation_procedure_description,
        solicitation_procedure_kind=procedure,
        number_of_offers_received=offers,
        offer_outcome_kind=_offer_kind(offers),
        other_than_full_open_code=transaction.other_than_full_and_open_code,
        other_than_full_open_description=transaction.other_than_full_and_open_description,
        other_than_full_open_authority_kind=authority,
        reported_process_competitive=_process_competitive(extent),
        reported_sources_restricted_or_unavailable=_sources_restricted(extent),
        evidence_conflicts=_conflicts(extent, procedure, authority),
        evidence_notes=_notes(transaction, extent, procedure, authority),
        missing_core_fields=_missing(transaction, extent),
    )


def _classify(value: str | None, table: dict[str, Any], unknown: Any) -> Any:
    normalized = _normalize(value)
    return unknown if normalized is None else table.get(normalized, unknown)


def _authority(value: str | None) -> OtherThanFullOpenAuthorityKind:
    normalized = _normalize(value)
    if normalized is None:
        return OtherThanFullOpenAuthorityKind.NONE_REPORTED
    return _AUTHORITY.get(
        normalized, OtherThanFullOpenAuthorityKind.UNKNOWN_REPORTED
    )


def _offer_kind(offers: int | None) -> OfferOutcomeKind:
    if offers is None:
        return OfferOutcomeKind.UNKNOWN
    if offers == 0:
        return OfferOutcomeKind.ZERO_REPORTED
    return (
        OfferOutcomeKind.SINGLE_OFFER
        if offers == 1
        else OfferOutcomeKind.MULTIPLE_OFFERS
    )


def _process_competitive(extent: CompetitionExtentKind) -> bool | None:
    if extent in {
        CompetitionExtentKind.COMPETED_UNDER_SAP,
        CompetitionExtentKind.FULL_AND_OPEN,
        CompetitionExtentKind.FULL_AND_OPEN_AFTER_EXCLUSION,
    }:
        return True
    if extent in {
        CompetitionExtentKind.NOT_AVAILABLE,
        CompetitionExtentKind.NOT_COMPETED_UNDER_SAP,
        CompetitionExtentKind.NOT_COMPETED,
    }:
        return False
    return None


def _sources_restricted(extent: CompetitionExtentKind) -> bool | None:
    if extent in {
        CompetitionExtentKind.COMPETED_UNDER_SAP,
        CompetitionExtentKind.FULL_AND_OPEN,
    }:
        return False
    if extent in {
        CompetitionExtentKind.FULL_AND_OPEN_AFTER_EXCLUSION,
        CompetitionExtentKind.NOT_AVAILABLE,
        CompetitionExtentKind.NOT_COMPETED_UNDER_SAP,
        CompetitionExtentKind.NOT_COMPETED,
    }:
        return True
    return None


def _procedure_hint(procedure: SolicitationProcedureKind) -> bool | None:
    if procedure is SolicitationProcedureKind.ONLY_ONE_SOURCE:
        return False
    if procedure in {
        SolicitationProcedureKind.NEGOTIATED_PROPOSAL_QUOTE,
        SolicitationProcedureKind.SEALED_BID,
        SolicitationProcedureKind.TWO_STEP,
        SolicitationProcedureKind.ARCHITECT_ENGINEER,
        SolicitationProcedureKind.BASIC_RESEARCH,
        SolicitationProcedureKind.ALTERNATIVE_SOURCES,
        SolicitationProcedureKind.MULTIPLE_AWARD_FAIR_OPPORTUNITY,
    }:
        return True
    # Simplified acquisition can be competitive or noncompetitive.
    return None


def _conflicts(
    extent: CompetitionExtentKind,
    procedure: SolicitationProcedureKind,
    authority: OtherThanFullOpenAuthorityKind,
) -> tuple[str, ...]:
    conflicts: list[str] = []
    process, hint = _process_competitive(extent), _procedure_hint(procedure)
    if process is True and hint is False:
        conflicts.append("competitive_extent_with_only_one_source_procedure")
    elif process is False and hint is True:
        conflicts.append("noncompetitive_extent_with_competitive_procedure")

    if (
        extent is CompetitionExtentKind.FULL_AND_OPEN
        and authority
        not in {
            OtherThanFullOpenAuthorityKind.NONE_REPORTED,
            OtherThanFullOpenAuthorityKind.UNKNOWN_REPORTED,
        }
    ):
        conflicts.append("full_and_open_extent_with_noncompetition_authority")
    return tuple(conflicts)


def _notes(
    tx: ProcurementTransaction,
    extent: CompetitionExtentKind,
    procedure: SolicitationProcedureKind,
    authority: OtherThanFullOpenAuthorityKind,
) -> tuple[str, ...]:
    notes: list[str] = []
    if tx.number_of_offers_received == 1:
        notes.append("single_offer_is_outcome_not_process_classification")
    elif tx.number_of_offers_received == 0:
        notes.append("zero_offers_reported")

    if extent is CompetitionExtentKind.FULL_AND_OPEN_AFTER_EXCLUSION:
        notes.append("competition_reported_after_exclusion_of_sources")
    if procedure is SolicitationProcedureKind.SIMPLIFIED_ACQUISITION:
        notes.append("simplified_acquisition_can_be_competitive_or_noncompetitive")
    if extent is CompetitionExtentKind.UNKNOWN and tx.extent_competed_description:
        notes.append("unrecognized_extent_description")
    if (
        procedure is SolicitationProcedureKind.UNKNOWN
        and tx.solicitation_procedure_description
    ):
        notes.append("unrecognized_solicitation_procedure_description")
    if (
        authority is OtherThanFullOpenAuthorityKind.UNKNOWN_REPORTED
        and tx.other_than_full_and_open_description
    ):
        notes.append("unrecognized_other_than_full_open_authority")
    if (
        _process_competitive(extent) is False
        and authority is OtherThanFullOpenAuthorityKind.NONE_REPORTED
    ):
        notes.append("noncompetitive_extent_without_reported_authority")
    return tuple(notes)


def _missing(
    tx: ProcurementTransaction,
    extent: CompetitionExtentKind,
) -> tuple[str, ...]:
    missing: list[str] = []
    if tx.extent_competed_code is None and tx.extent_competed_description is None:
        missing.append("extent_competed")
    if tx.number_of_offers_received is None:
        missing.append("number_of_offers_received")
    if (
        tx.solicitation_procedure_code is None
        and tx.solicitation_procedure_description is None
    ):
        missing.append("solicitation_procedure")
    if (
        _process_competitive(extent) is False
        and tx.other_than_full_and_open_code is None
        and tx.other_than_full_and_open_description is None
    ):
        missing.append("other_than_full_and_open_authority")
    return tuple(missing)


def _normalize(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return " ".join(
        re.sub(r"[^a-z0-9]+", " ", value.strip().casefold()).split()
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
