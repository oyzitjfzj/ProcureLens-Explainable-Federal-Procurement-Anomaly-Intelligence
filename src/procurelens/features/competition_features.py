"""Stable, explainable competition feature candidates for ProcureLens.

Transforms one CompetitionEvidence object into deterministic numeric candidates
for later anomaly detectors. Competition process, solicitation procedure, offer
outcome, and reported noncompetition authority remain separate dimensions.

Categorical procurement states are never encoded as fake ordinal numbers.
Unavailable/unknown evidence remains explicitly missing. No imputation, anomaly
cutoff, risk score, or fraud/collusion conclusion lives here.
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

from procurelens.features.competition_evidence import (
    CompetitionEvidence,
    CompetitionExtentKind,
    OfferOutcomeKind,
    OtherThanFullOpenAuthorityKind,
    SolicitationProcedureKind,
)


class CompetitionFeatureError(ValueError):
    """Raised when competition feature evidence is inconsistent."""


class CompetitionFeatureFamily(str, Enum):
    PROCESS = "process"
    OUTCOME = "outcome"
    PROCEDURE = "procedure"
    AUTHORITY = "authority"
    INTERACTION = "interaction"
    QUALITY = "quality"


class CompetitionFeatureName(str, Enum):
    OFFERS_RECEIVED = "competition_offers_received"
    PROCESS_COMPETITIVE = "competition_reported_process_competitive"
    SOURCES_RESTRICTED_OR_UNAVAILABLE = (
        "competition_reported_sources_restricted_or_unavailable"
    )
    SINGLE_OFFER = "competition_single_offer_reported"
    MULTIPLE_OFFERS = "competition_multiple_offers_reported"
    ZERO_OFFERS = "competition_zero_offers_reported"
    ONLY_ONE_SOURCE_SOLICITED = "competition_only_one_source_solicited"
    OTHER_THAN_FULL_OPEN_AUTHORITY_REPORTED = (
        "competition_other_than_full_open_authority_reported"
    )
    FULL_OPEN_AFTER_EXCLUSION = "competition_full_open_after_exclusion"
    SINGLE_OFFER_COMPETITIVE_PROCESS = (
        "competition_single_offer_under_reported_competitive_process"
    )
    SINGLE_OFFER_NONCOMPETITIVE_PROCESS = (
        "competition_single_offer_under_reported_noncompetitive_process"
    )
    CONFLICT_COUNT = "competition_evidence_conflict_count"
    MISSING_CORE_FIELD_COUNT = "competition_missing_core_field_count"
    UNRECOGNIZED_REPORTED_VALUE_COUNT = (
        "competition_unrecognized_reported_value_count"
    )


@dataclass(frozen=True, slots=True)
class CompetitionFeatureDefinition:
    name: CompetitionFeatureName
    family: CompetitionFeatureFamily
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", CompetitionFeatureName(self.name))
        object.__setattr__(self, "family", CompetitionFeatureFamily(self.family))
        text = self.description.strip()
        if not text:
            raise CompetitionFeatureError("feature description must not be blank")
        object.__setattr__(self, "description", text)


_DEFINITIONS = (
    CompetitionFeatureDefinition(
        CompetitionFeatureName.OFFERS_RECEIVED,
        CompetitionFeatureFamily.OUTCOME,
        "Reported number of offers received; missing stays unavailable.",
    ),
    CompetitionFeatureDefinition(
        CompetitionFeatureName.PROCESS_COMPETITIVE,
        CompetitionFeatureFamily.PROCESS,
        "Whether the reported extent-of-competition category denotes a competitive process.",
    ),
    CompetitionFeatureDefinition(
        CompetitionFeatureName.SOURCES_RESTRICTED_OR_UNAVAILABLE,
        CompetitionFeatureFamily.PROCESS,
        "Whether reported competition evidence indicates excluded, restricted, or unavailable sources.",
    ),
    CompetitionFeatureDefinition(
        CompetitionFeatureName.SINGLE_OFFER,
        CompetitionFeatureFamily.OUTCOME,
        "Whether exactly one offer was reported; this is an outcome, not a process verdict.",
    ),
    CompetitionFeatureDefinition(
        CompetitionFeatureName.MULTIPLE_OFFERS,
        CompetitionFeatureFamily.OUTCOME,
        "Whether two or more offers were reported.",
    ),
    CompetitionFeatureDefinition(
        CompetitionFeatureName.ZERO_OFFERS,
        CompetitionFeatureFamily.OUTCOME,
        "Whether zero offers were reported; preserved as reported evidence.",
    ),
    CompetitionFeatureDefinition(
        CompetitionFeatureName.ONLY_ONE_SOURCE_SOLICITED,
        CompetitionFeatureFamily.PROCEDURE,
        "Whether the recognized solicitation procedure reports only one source solicited.",
    ),
    CompetitionFeatureDefinition(
        CompetitionFeatureName.OTHER_THAN_FULL_OPEN_AUTHORITY_REPORTED,
        CompetitionFeatureFamily.AUTHORITY,
        "Whether any other-than-full-and-open authority/reason was reported, recognized or not.",
    ),
    CompetitionFeatureDefinition(
        CompetitionFeatureName.FULL_OPEN_AFTER_EXCLUSION,
        CompetitionFeatureFamily.PROCESS,
        "Whether extent of competition is full and open after exclusion of sources.",
    ),
    CompetitionFeatureDefinition(
        CompetitionFeatureName.SINGLE_OFFER_COMPETITIVE_PROCESS,
        CompetitionFeatureFamily.INTERACTION,
        "Whether one offer resulted from a process reported as competitive.",
    ),
    CompetitionFeatureDefinition(
        CompetitionFeatureName.SINGLE_OFFER_NONCOMPETITIVE_PROCESS,
        CompetitionFeatureFamily.INTERACTION,
        "Whether one offer accompanied a process reported as noncompetitive.",
    ),
    CompetitionFeatureDefinition(
        CompetitionFeatureName.CONFLICT_COUNT,
        CompetitionFeatureFamily.QUALITY,
        "Count of explicit conflicts among reported competition evidence dimensions.",
    ),
    CompetitionFeatureDefinition(
        CompetitionFeatureName.MISSING_CORE_FIELD_COUNT,
        CompetitionFeatureFamily.QUALITY,
        "Count of core competition evidence fields that are missing for interpretation.",
    ),
    CompetitionFeatureDefinition(
        CompetitionFeatureName.UNRECOGNIZED_REPORTED_VALUE_COUNT,
        CompetitionFeatureFamily.QUALITY,
        "Count of populated competition fields whose reported category is not recognized by the current contract.",
    ),
)
_DEFINITION_BY_NAME: Mapping[
    CompetitionFeatureName, CompetitionFeatureDefinition
] = MappingProxyType({definition.name: definition for definition in _DEFINITIONS})
if len(_DEFINITION_BY_NAME) != len(_DEFINITIONS):
    raise RuntimeError("competition feature definitions contain duplicate names")


def competition_feature_definitions() -> tuple[CompetitionFeatureDefinition, ...]:
    return _DEFINITIONS


def competition_feature_definition_sha256() -> str:
    return _digest(
        [
            {
                "name": definition.name.value,
                "family": definition.family.value,
                "description": definition.description,
            }
            for definition in _DEFINITIONS
        ]
    )


@dataclass(frozen=True, slots=True)
class CompetitionFeature:
    definition: CompetitionFeatureDefinition
    value: Decimal | None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        canonical = _DEFINITION_BY_NAME.get(CompetitionFeatureName(self.definition.name))
        if canonical != self.definition:
            raise CompetitionFeatureError(
                "feature definition is not the canonical contract entry"
            )
        if self.value is None:
            reason = (
                None
                if self.unavailable_reason is None
                else self.unavailable_reason.strip()
            )
            if not reason:
                raise CompetitionFeatureError(
                    f"{self.definition.name.value}: missing feature requires a reason"
                )
            object.__setattr__(self, "unavailable_reason", reason)
            return
        if not isinstance(self.value, Decimal) or not self.value.is_finite():
            raise CompetitionFeatureError(
                f"{self.definition.name.value}: value must be finite Decimal or None"
            )
        if self.unavailable_reason is not None:
            raise CompetitionFeatureError(
                f"{self.definition.name.value}: available feature cannot carry a missing reason"
            )

    @property
    def name(self) -> CompetitionFeatureName:
        return self.definition.name

    @property
    def family(self) -> CompetitionFeatureFamily:
        return self.definition.family

    @property
    def available(self) -> bool:
        return self.value is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "family": self.family.value,
            "value": None if self.value is None else str(self.value),
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True, slots=True)
class CompetitionFeatureSet:
    transaction_id: str
    award_id: str
    evidence_sha256: str
    extent_kind: CompetitionExtentKind
    solicitation_procedure_kind: SolicitationProcedureKind
    offer_outcome_kind: OfferOutcomeKind
    other_than_full_open_authority_kind: OtherThanFullOpenAuthorityKind
    features: tuple[CompetitionFeature, ...]

    def __post_init__(self) -> None:
        for name in ("transaction_id", "award_id"):
            value = getattr(self, name).strip()
            if not value:
                raise CompetitionFeatureError(f"{name} must not be blank")
            object.__setattr__(self, name, value)

        object.__setattr__(
            self, "evidence_sha256", _validate_digest(self.evidence_sha256)
        )
        object.__setattr__(
            self, "extent_kind", CompetitionExtentKind(self.extent_kind)
        )
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
            OtherThanFullOpenAuthorityKind(
                self.other_than_full_open_authority_kind
            ),
        )
        if not isinstance(self.features, tuple):
            object.__setattr__(self, "features", tuple(self.features))
        observed = tuple(feature.name for feature in self.features)
        expected = tuple(definition.name for definition in _DEFINITIONS)
        if observed != expected:
            raise CompetitionFeatureError(
                "feature set must contain every canonical competition feature "
                "exactly once in contract order"
            )

    @property
    def definition_sha256(self) -> str:
        return competition_feature_definition_sha256()

    @property
    def feature_evidence_sha256(self) -> str:
        return _digest(
            {
                "transaction_id": self.transaction_id,
                "award_id": self.award_id,
                "source_evidence_sha256": self.evidence_sha256,
                "definition_sha256": self.definition_sha256,
                "extent_kind": self.extent_kind.value,
                "solicitation_procedure_kind": self.solicitation_procedure_kind.value,
                "offer_outcome_kind": self.offer_outcome_kind.value,
                "other_than_full_open_authority_kind":
                    self.other_than_full_open_authority_kind.value,
                "features": [feature.as_dict() for feature in self.features],
            }
        )

    def get(
        self, name: CompetitionFeatureName | str
    ) -> CompetitionFeature:
        wanted = CompetitionFeatureName(name)
        for feature in self.features:
            if feature.name is wanted:
                return feature
        raise CompetitionFeatureError(f"unknown competition feature: {wanted.value}")

    def select(
        self,
        names: Iterable[CompetitionFeatureName | str],
        *,
        require_complete: bool = True,
    ) -> tuple[CompetitionFeature, ...]:
        if isinstance(names, (str, bytes)):
            raise CompetitionFeatureError(
                "feature names must be an iterable of names"
            )
        selected: list[CompetitionFeature] = []
        seen: set[CompetitionFeatureName] = set()
        for item in names:
            name = CompetitionFeatureName(item)
            if name in seen:
                raise CompetitionFeatureError(
                    f"duplicate selected feature: {name.value}"
                )
            seen.add(name)
            feature = self.get(name)
            if require_complete and not feature.available:
                raise CompetitionFeatureError(
                    f"selected feature {name.value} is unavailable: "
                    f"{feature.unavailable_reason}"
                )
            selected.append(feature)
        if not selected:
            raise CompetitionFeatureError("at least one feature must be selected")
        return tuple(selected)

    def select_families(
        self,
        families: Iterable[CompetitionFeatureFamily | str],
        *,
        require_complete: bool = False,
    ) -> tuple[CompetitionFeature, ...]:
        if isinstance(families, (str, bytes)):
            raise CompetitionFeatureError("families must be an iterable")
        wanted = {CompetitionFeatureFamily(item) for item in families}
        if not wanted:
            raise CompetitionFeatureError("at least one family must be selected")
        selected = tuple(
            feature for feature in self.features if feature.family in wanted
        )
        if require_complete:
            missing = [feature for feature in selected if not feature.available]
            if missing:
                names = ", ".join(feature.name.value for feature in missing)
                raise CompetitionFeatureError(
                    f"selected feature families contain missing values: {names}"
                )
        return selected

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "award_id": self.award_id,
            "source_evidence_sha256": self.evidence_sha256,
            "definition_sha256": self.definition_sha256,
            "feature_evidence_sha256": self.feature_evidence_sha256,
            "extent_kind": self.extent_kind.value,
            "solicitation_procedure_kind": self.solicitation_procedure_kind.value,
            "offer_outcome_kind": self.offer_outcome_kind.value,
            "other_than_full_open_authority_kind":
                self.other_than_full_open_authority_kind.value,
            "features": [feature.as_dict() for feature in self.features],
        }


def build_competition_features(
    evidence: CompetitionEvidence,
) -> CompetitionFeatureSet:
    if not isinstance(evidence, CompetitionEvidence):
        raise TypeError("evidence must be CompetitionEvidence")

    values: dict[
        CompetitionFeatureName, tuple[Decimal | None, str | None]
    ] = {
        CompetitionFeatureName.OFFERS_RECEIVED:
            _offers_value(evidence),
        CompetitionFeatureName.PROCESS_COMPETITIVE:
            _optional_bool(
                evidence.reported_process_competitive,
                "competition_process_unknown",
            ),
        CompetitionFeatureName.SOURCES_RESTRICTED_OR_UNAVAILABLE:
            _optional_bool(
                evidence.reported_sources_restricted_or_unavailable,
                "source_restriction_status_unknown",
            ),
        CompetitionFeatureName.SINGLE_OFFER:
            _offer_indicator(evidence, OfferOutcomeKind.SINGLE_OFFER),
        CompetitionFeatureName.MULTIPLE_OFFERS:
            _offer_indicator(evidence, OfferOutcomeKind.MULTIPLE_OFFERS),
        CompetitionFeatureName.ZERO_OFFERS:
            _offer_indicator(evidence, OfferOutcomeKind.ZERO_REPORTED),
        CompetitionFeatureName.ONLY_ONE_SOURCE_SOLICITED:
            _known_category_indicator(
                evidence.solicitation_procedure_kind,
                SolicitationProcedureKind.UNKNOWN,
                SolicitationProcedureKind.ONLY_ONE_SOURCE,
                "solicitation_procedure_unknown",
            ),
        CompetitionFeatureName.OTHER_THAN_FULL_OPEN_AUTHORITY_REPORTED:
            (_decimal_bool(evidence.other_than_full_open_authority_reported), None),
        CompetitionFeatureName.FULL_OPEN_AFTER_EXCLUSION:
            _known_category_indicator(
                evidence.extent_kind,
                CompetitionExtentKind.UNKNOWN,
                CompetitionExtentKind.FULL_AND_OPEN_AFTER_EXCLUSION,
                "competition_extent_unknown",
            ),
        CompetitionFeatureName.SINGLE_OFFER_COMPETITIVE_PROCESS:
            _single_offer_process_interaction(evidence, competitive=True),
        CompetitionFeatureName.SINGLE_OFFER_NONCOMPETITIVE_PROCESS:
            _single_offer_process_interaction(evidence, competitive=False),
        CompetitionFeatureName.CONFLICT_COUNT:
            (Decimal(len(evidence.evidence_conflicts)), None),
        CompetitionFeatureName.MISSING_CORE_FIELD_COUNT:
            (Decimal(len(evidence.missing_core_fields)), None),
        CompetitionFeatureName.UNRECOGNIZED_REPORTED_VALUE_COUNT:
            (Decimal(_unrecognized_reported_value_count(evidence)), None),
    }

    features = tuple(
        CompetitionFeature(
            definition,
            values[definition.name][0],
            values[definition.name][1],
        )
        for definition in _DEFINITIONS
    )
    return CompetitionFeatureSet(
        transaction_id=evidence.transaction_id,
        award_id=evidence.award_id,
        evidence_sha256=evidence.evidence_sha256,
        extent_kind=evidence.extent_kind,
        solicitation_procedure_kind=evidence.solicitation_procedure_kind,
        offer_outcome_kind=evidence.offer_outcome_kind,
        other_than_full_open_authority_kind=
            evidence.other_than_full_open_authority_kind,
        features=features,
    )


def _offers_value(
    evidence: CompetitionEvidence,
) -> tuple[Decimal | None, str | None]:
    offers = evidence.number_of_offers_received
    if offers is None:
        return None, "number_of_offers_received_missing"
    return Decimal(offers), None


def _optional_bool(
    value: bool | None, missing_reason: str
) -> tuple[Decimal | None, str | None]:
    if value is None:
        return None, missing_reason
    return _decimal_bool(value), None


def _offer_indicator(
    evidence: CompetitionEvidence, wanted: OfferOutcomeKind
) -> tuple[Decimal | None, str | None]:
    if evidence.offer_outcome_kind is OfferOutcomeKind.UNKNOWN:
        return None, "number_of_offers_received_missing"
    return _decimal_bool(evidence.offer_outcome_kind is wanted), None


def _known_category_indicator(
    value: Enum,
    unknown: Enum,
    wanted: Enum,
    missing_reason: str,
) -> tuple[Decimal | None, str | None]:
    if value is unknown:
        return None, missing_reason
    return _decimal_bool(value is wanted), None


def _single_offer_process_interaction(
    evidence: CompetitionEvidence,
    *,
    competitive: bool,
) -> tuple[Decimal | None, str | None]:
    if evidence.offer_outcome_kind is OfferOutcomeKind.UNKNOWN:
        return None, "number_of_offers_received_missing"
    process = evidence.reported_process_competitive
    if process is None:
        return None, "competition_process_unknown"
    return _decimal_bool(
        evidence.offer_outcome_kind is OfferOutcomeKind.SINGLE_OFFER
        and process is competitive
    ), None


def _unrecognized_reported_value_count(
    evidence: CompetitionEvidence,
) -> int:
    count = 0
    if (
        evidence.extent_kind is CompetitionExtentKind.UNKNOWN
        and (evidence.extent_code is not None or evidence.extent_description is not None)
    ):
        count += 1
    if (
        evidence.solicitation_procedure_kind
        is SolicitationProcedureKind.UNKNOWN
        and (
            evidence.solicitation_procedure_code is not None
            or evidence.solicitation_procedure_description is not None
        )
    ):
        count += 1
    if (
        evidence.other_than_full_open_authority_kind
        is OtherThanFullOpenAuthorityKind.UNKNOWN_REPORTED
    ):
        count += 1
    return count


def _decimal_bool(value: bool) -> Decimal:
    return Decimal(1) if value else Decimal(0)


def _validate_digest(value: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise CompetitionFeatureError(
            "evidence_sha256 must be a SHA-256 hex digest"
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
