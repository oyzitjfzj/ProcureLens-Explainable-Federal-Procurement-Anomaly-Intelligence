"""Context-aware competition feature candidates for ProcureLens.

Adds only peer-market support, prevalence, and signed target-minus-peer
deviations to the transaction-level competition features already defined
elsewhere. No smoothing prior, imputation, anomaly threshold, risk score, or
misconduct conclusion lives here.
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

from procurelens.features.competition_context import (
    CompetitionContext,
    CompetitionContextMode,
    CompetitionContextResult,
)
from procurelens.features.competition_evidence import (
    CompetitionExtentKind,
    OfferOutcomeKind,
)


class CompetitionContextFeatureError(ValueError):
    """Raised when contextual competition feature evidence is inconsistent."""


class CompetitionContextFeatureFamily(str, Enum):
    SUPPORT = "support"
    PREVALENCE = "prevalence"
    DEVIATION = "deviation"
    QUALITY = "quality"


class CompetitionContextFeatureName(str, Enum):
    REFERENCE_BASE_AWARD_COUNT = "competition_context_reference_base_award_count"
    PROCESS_COVERAGE = "competition_context_process_coverage"
    OFFER_COVERAGE = "competition_context_offer_coverage"
    PROCEDURE_COVERAGE = "competition_context_procedure_coverage"
    PEER_NONCOMPETITIVE_PROCESS_RATE = "competition_context_peer_noncompetitive_process_rate"
    PEER_SINGLE_OFFER_RATE = "competition_context_peer_single_offer_rate"
    PEER_ZERO_OFFER_RATE = "competition_context_peer_zero_offer_rate"
    PEER_ONLY_ONE_SOURCE_SOLICITED_RATE = "competition_context_peer_only_one_source_solicited_rate"
    PEER_NONCOMPETITION_AUTHORITY_REPORTED_RATE = "competition_context_peer_noncompetition_authority_reported_rate"
    PEER_FULL_OPEN_AFTER_EXCLUSION_RATE = "competition_context_peer_full_open_after_exclusion_rate"
    PEER_CONFLICT_RATE = "competition_context_peer_conflict_rate"
    PEER_MISSING_CORE_FIELD_RATE = "competition_context_peer_missing_core_field_rate"
    NONCOMPETITIVE_PROCESS_DEVIATION = "competition_context_noncompetitive_process_deviation"
    SINGLE_OFFER_DEVIATION = "competition_context_single_offer_deviation"
    ZERO_OFFER_DEVIATION = "competition_context_zero_offer_deviation"
    ONLY_ONE_SOURCE_SOLICITED_DEVIATION = "competition_context_only_one_source_solicited_deviation"
    NONCOMPETITION_AUTHORITY_REPORTED_DEVIATION = "competition_context_noncompetition_authority_reported_deviation"
    FULL_OPEN_AFTER_EXCLUSION_DEVIATION = "competition_context_full_open_after_exclusion_deviation"
    CONFLICT_DEVIATION = "competition_context_conflict_deviation"
    MISSING_CORE_FIELD_DEVIATION = "competition_context_missing_core_field_deviation"


@dataclass(frozen=True, slots=True)
class CompetitionContextFeatureDefinition:
    name: CompetitionContextFeatureName
    family: CompetitionContextFeatureFamily
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", CompetitionContextFeatureName(self.name))
        object.__setattr__(self, "family", CompetitionContextFeatureFamily(self.family))
        text = self.description.strip()
        if not text:
            raise CompetitionContextFeatureError("feature description must not be blank")
        object.__setattr__(self, "description", text)


_SPECS = (
    (CompetitionContextFeatureName.REFERENCE_BASE_AWARD_COUNT, CompetitionContextFeatureFamily.SUPPORT,
     "Peer base-award count in the selected leave-one-out or external reference market."),
    (CompetitionContextFeatureName.PROCESS_COVERAGE, CompetitionContextFeatureFamily.SUPPORT,
     "Reference fraction with recognized competition-process evidence."),
    (CompetitionContextFeatureName.OFFER_COVERAGE, CompetitionContextFeatureFamily.SUPPORT,
     "Reference fraction with reported offer counts."),
    (CompetitionContextFeatureName.PROCEDURE_COVERAGE, CompetitionContextFeatureFamily.SUPPORT,
     "Reference fraction with recognized solicitation procedure."),
    (CompetitionContextFeatureName.PEER_NONCOMPETITIVE_PROCESS_RATE, CompetitionContextFeatureFamily.PREVALENCE,
     "Peer noncompetitive-process prevalence among process-known awards."),
    (CompetitionContextFeatureName.PEER_SINGLE_OFFER_RATE, CompetitionContextFeatureFamily.PREVALENCE,
     "Peer single-offer prevalence among offer-known awards."),
    (CompetitionContextFeatureName.PEER_ZERO_OFFER_RATE, CompetitionContextFeatureFamily.PREVALENCE,
     "Peer zero-offer prevalence among offer-known awards."),
    (CompetitionContextFeatureName.PEER_ONLY_ONE_SOURCE_SOLICITED_RATE, CompetitionContextFeatureFamily.PREVALENCE,
     "Peer only-one-source prevalence among procedure-known awards."),
    (CompetitionContextFeatureName.PEER_NONCOMPETITION_AUTHORITY_REPORTED_RATE, CompetitionContextFeatureFamily.PREVALENCE,
     "Peer prevalence of reported other-than-full-and-open authority or reason."),
    (CompetitionContextFeatureName.PEER_FULL_OPEN_AFTER_EXCLUSION_RATE, CompetitionContextFeatureFamily.PREVALENCE,
     "Peer prevalence of full-open competition after exclusion of sources."),
    (CompetitionContextFeatureName.PEER_CONFLICT_RATE, CompetitionContextFeatureFamily.QUALITY,
     "Peer prevalence of explicit competition-evidence conflicts."),
    (CompetitionContextFeatureName.PEER_MISSING_CORE_FIELD_RATE, CompetitionContextFeatureFamily.QUALITY,
     "Peer prevalence of missing core competition evidence."),
    (CompetitionContextFeatureName.NONCOMPETITIVE_PROCESS_DEVIATION, CompetitionContextFeatureFamily.DEVIATION,
     "Target noncompetitive indicator minus peer prevalence."),
    (CompetitionContextFeatureName.SINGLE_OFFER_DEVIATION, CompetitionContextFeatureFamily.DEVIATION,
     "Target single-offer indicator minus peer prevalence."),
    (CompetitionContextFeatureName.ZERO_OFFER_DEVIATION, CompetitionContextFeatureFamily.DEVIATION,
     "Target zero-offer indicator minus peer prevalence."),
    (CompetitionContextFeatureName.ONLY_ONE_SOURCE_SOLICITED_DEVIATION, CompetitionContextFeatureFamily.DEVIATION,
     "Target only-one-source indicator minus peer prevalence."),
    (CompetitionContextFeatureName.NONCOMPETITION_AUTHORITY_REPORTED_DEVIATION, CompetitionContextFeatureFamily.DEVIATION,
     "Target reported-authority indicator minus peer prevalence."),
    (CompetitionContextFeatureName.FULL_OPEN_AFTER_EXCLUSION_DEVIATION, CompetitionContextFeatureFamily.DEVIATION,
     "Target full-open-after-exclusion indicator minus peer prevalence."),
    (CompetitionContextFeatureName.CONFLICT_DEVIATION, CompetitionContextFeatureFamily.QUALITY,
     "Target evidence-conflict indicator minus peer prevalence."),
    (CompetitionContextFeatureName.MISSING_CORE_FIELD_DEVIATION, CompetitionContextFeatureFamily.QUALITY,
     "Target missing-core-evidence indicator minus peer prevalence."),
)
_DEFINITIONS = tuple(CompetitionContextFeatureDefinition(*spec) for spec in _SPECS)
_DEFINITION_BY_NAME: Mapping[CompetitionContextFeatureName, CompetitionContextFeatureDefinition] = (
    MappingProxyType({definition.name: definition for definition in _DEFINITIONS})
)
if len(_DEFINITION_BY_NAME) != len(_DEFINITIONS):
    raise RuntimeError("competition-context feature definitions contain duplicate names")


def competition_context_feature_definitions() -> tuple[CompetitionContextFeatureDefinition, ...]:
    return _DEFINITIONS


def competition_context_feature_definition_sha256() -> str:
    return _digest([
        {"name": item.name.value, "family": item.family.value, "description": item.description}
        for item in _DEFINITIONS
    ])


@dataclass(frozen=True, slots=True)
class CompetitionContextFeature:
    definition: CompetitionContextFeatureDefinition
    value: Decimal | None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        canonical = _DEFINITION_BY_NAME.get(CompetitionContextFeatureName(self.definition.name))
        if canonical != self.definition:
            raise CompetitionContextFeatureError("feature definition is not canonical")
        if self.value is None:
            reason = None if self.unavailable_reason is None else self.unavailable_reason.strip()
            if not reason:
                raise CompetitionContextFeatureError(
                    f"{self.definition.name.value}: missing feature requires a reason"
                )
            object.__setattr__(self, "unavailable_reason", reason)
            return
        if not isinstance(self.value, Decimal) or not self.value.is_finite():
            raise CompetitionContextFeatureError("feature value must be finite Decimal or None")
        if self.unavailable_reason is not None:
            raise CompetitionContextFeatureError("available feature cannot carry a missing reason")

    @property
    def name(self) -> CompetitionContextFeatureName:
        return self.definition.name

    @property
    def family(self) -> CompetitionContextFeatureFamily:
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
class CompetitionContextFeatureSet:
    transaction_id: str
    award_id: str
    source_result_sha256: str
    source_context_sha256: str | None
    context_available: bool
    reference_mode: CompetitionContextMode | None
    market_level: str | None
    market_key_sha256: str | None
    unavailable_reasons: tuple[str, ...]
    features: tuple[CompetitionContextFeature, ...]

    def __post_init__(self) -> None:
        for name in ("transaction_id", "award_id"):
            value = getattr(self, name).strip()
            if not value:
                raise CompetitionContextFeatureError(f"{name} must not be blank")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, "source_result_sha256",
            _validate_digest(self.source_result_sha256, "source_result_sha256"),
        )
        if not isinstance(self.context_available, bool):
            raise CompetitionContextFeatureError("context_available must be bool")
        if self.reference_mode is not None:
            object.__setattr__(self, "reference_mode", CompetitionContextMode(self.reference_mode))
        object.__setattr__(
            self, "market_level",
            None if self.market_level is None else self.market_level.strip() or None,
        )
        for name in ("source_context_sha256", "market_key_sha256"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _validate_digest(value, name))
        reasons = tuple(item.strip() for item in self.unavailable_reasons if item.strip())
        if len(reasons) != len(set(reasons)):
            raise CompetitionContextFeatureError("unavailable reasons contain duplicates")
        object.__setattr__(self, "unavailable_reasons", reasons)

        provenance = (
            self.source_context_sha256, self.reference_mode,
            self.market_level, self.market_key_sha256,
        )
        if self.context_available:
            if any(item is None for item in provenance) or reasons:
                raise CompetitionContextFeatureError(
                    "available context requires complete selected-market provenance"
                )
        elif any(item is not None for item in provenance) or not reasons:
            raise CompetitionContextFeatureError(
                "unavailable context must carry reasons and no selected-market provenance"
            )

        object.__setattr__(self, "features", tuple(self.features))
        expected = tuple(item.name for item in _DEFINITIONS)
        if tuple(item.name for item in self.features) != expected:
            raise CompetitionContextFeatureError(
                "feature set must contain every canonical feature exactly once in order"
            )
        if not self.context_available and any(item.available for item in self.features):
            raise CompetitionContextFeatureError(
                "unavailable context cannot produce available contextual features"
            )

    @property
    def definition_sha256(self) -> str:
        return competition_context_feature_definition_sha256()

    @property
    def feature_evidence_sha256(self) -> str:
        return _digest({
            "transaction_id": self.transaction_id,
            "award_id": self.award_id,
            "source_result_sha256": self.source_result_sha256,
            "source_context_sha256": self.source_context_sha256,
            "definition_sha256": self.definition_sha256,
            "context_available": self.context_available,
            "reference_mode": None if self.reference_mode is None else self.reference_mode.value,
            "market_level": self.market_level,
            "market_key_sha256": self.market_key_sha256,
            "unavailable_reasons": list(self.unavailable_reasons),
            "features": [item.as_dict() for item in self.features],
        })

    def get(self, name: CompetitionContextFeatureName | str) -> CompetitionContextFeature:
        wanted = CompetitionContextFeatureName(name)
        for item in self.features:
            if item.name is wanted:
                return item
        raise CompetitionContextFeatureError(f"unknown feature: {wanted.value}")

    def select(
        self,
        names: Iterable[CompetitionContextFeatureName | str],
        *,
        require_complete: bool = True,
    ) -> tuple[CompetitionContextFeature, ...]:
        if isinstance(names, (str, bytes)):
            raise CompetitionContextFeatureError("feature names must be an iterable")
        selected, seen = [], set()
        for raw in names:
            name = CompetitionContextFeatureName(raw)
            if name in seen:
                raise CompetitionContextFeatureError(f"duplicate selected feature: {name.value}")
            seen.add(name)
            item = self.get(name)
            if require_complete and not item.available:
                raise CompetitionContextFeatureError(
                    f"selected feature {name.value} is unavailable: {item.unavailable_reason}"
                )
            selected.append(item)
        if not selected:
            raise CompetitionContextFeatureError("at least one feature must be selected")
        return tuple(selected)

    def select_families(
        self,
        families: Iterable[CompetitionContextFeatureFamily | str],
        *,
        require_complete: bool = False,
    ) -> tuple[CompetitionContextFeature, ...]:
        if isinstance(families, (str, bytes)):
            raise CompetitionContextFeatureError("families must be an iterable")
        wanted = {CompetitionContextFeatureFamily(item) for item in families}
        if not wanted:
            raise CompetitionContextFeatureError("at least one family must be selected")
        selected = tuple(item for item in self.features if item.family in wanted)
        if require_complete and any(not item.available for item in selected):
            missing = ", ".join(item.name.value for item in selected if not item.available)
            raise CompetitionContextFeatureError(
                f"selected feature families contain missing values: {missing}"
            )
        return selected

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "award_id": self.award_id,
            "source_result_sha256": self.source_result_sha256,
            "source_context_sha256": self.source_context_sha256,
            "definition_sha256": self.definition_sha256,
            "feature_evidence_sha256": self.feature_evidence_sha256,
            "context_available": self.context_available,
            "reference_mode": None if self.reference_mode is None else self.reference_mode.value,
            "market_level": self.market_level,
            "market_key_sha256": self.market_key_sha256,
            "unavailable_reasons": list(self.unavailable_reasons),
            "features": [item.as_dict() for item in self.features],
        }


def build_competition_context_features(
    result: CompetitionContextResult,
) -> CompetitionContextFeatureSet:
    if not isinstance(result, CompetitionContextResult):
        raise TypeError("result must be CompetitionContextResult")

    result_sha = _digest(result.as_dict())
    context = result.context
    if context is None:
        reasons = tuple(result.unavailable_reasons) or ("competition_context_unavailable",)
        reason = "competition_context_unavailable:" + ",".join(reasons)
        return CompetitionContextFeatureSet(
            transaction_id=result.transaction_id,
            award_id=result.award_id,
            source_result_sha256=result_sha,
            source_context_sha256=None,
            context_available=False,
            reference_mode=None,
            market_level=None,
            market_key_sha256=None,
            unavailable_reasons=reasons,
            features=tuple(
                CompetitionContextFeature(definition, None, reason)
                for definition in _DEFINITIONS
            ),
        )

    values = _context_values(context)
    return CompetitionContextFeatureSet(
        transaction_id=result.transaction_id,
        award_id=result.award_id,
        source_result_sha256=result_sha,
        source_context_sha256=context.evidence_sha256,
        context_available=True,
        reference_mode=context.reference_mode,
        market_level=context.market_level,
        market_key_sha256=context.market_key.sha256_hex,
        unavailable_reasons=(),
        features=tuple(
            CompetitionContextFeature(
                definition,
                values[definition.name][0],
                values[definition.name][1],
            )
            for definition in _DEFINITIONS
        ),
    )


def _context_values(
    context: CompetitionContext,
) -> dict[CompetitionContextFeatureName, tuple[Decimal | None, str | None]]:
    if not isinstance(context, CompetitionContext):
        raise TypeError("context must be CompetitionContext")

    process_noncompetitive = (
        None if context.target_reported_process_competitive is None
        else not context.target_reported_process_competitive
    )
    target = {
        "noncompetitive": process_noncompetitive,
        "single": context.target_single_offer_reported,
        "zero": _zero_offer(context),
        "one_source": context.target_only_one_source_solicited,
        "authority": context.target_evidence.other_than_full_open_authority_reported,
        "after_exclusion": _after_exclusion(context),
        "conflict": bool(context.target_evidence.evidence_conflicts),
        "missing": bool(context.target_evidence.missing_core_fields),
    }
    rates = {
        "noncompetitive": context.noncompetitive_process_rate,
        "single": context.single_offer_rate,
        "zero": context.zero_offer_rate,
        "one_source": context.only_one_source_solicited_rate,
        "authority": context.noncompetition_authority_reported_rate,
        "after_exclusion": context.full_open_after_exclusion_rate,
        "conflict": context.conflict_rate,
        "missing": context.missing_core_field_rate,
    }

    return {
        CompetitionContextFeatureName.REFERENCE_BASE_AWARD_COUNT: (Decimal(context.base_award_count), None),
        CompetitionContextFeatureName.PROCESS_COVERAGE: (context.process_coverage, None),
        CompetitionContextFeatureName.OFFER_COVERAGE: (context.offer_coverage, None),
        CompetitionContextFeatureName.PROCEDURE_COVERAGE: (context.procedure_coverage, None),
        CompetitionContextFeatureName.PEER_NONCOMPETITIVE_PROCESS_RATE:
            _rate(rates["noncompetitive"], "peer_noncompetitive_process_rate_unavailable"),
        CompetitionContextFeatureName.PEER_SINGLE_OFFER_RATE:
            _rate(rates["single"], "peer_single_offer_rate_unavailable"),
        CompetitionContextFeatureName.PEER_ZERO_OFFER_RATE:
            _rate(rates["zero"], "peer_zero_offer_rate_unavailable"),
        CompetitionContextFeatureName.PEER_ONLY_ONE_SOURCE_SOLICITED_RATE:
            _rate(rates["one_source"], "peer_only_one_source_solicited_rate_unavailable"),
        CompetitionContextFeatureName.PEER_NONCOMPETITION_AUTHORITY_REPORTED_RATE:
            (rates["authority"], None),
        CompetitionContextFeatureName.PEER_FULL_OPEN_AFTER_EXCLUSION_RATE:
            (rates["after_exclusion"], None),
        CompetitionContextFeatureName.PEER_CONFLICT_RATE: (rates["conflict"], None),
        CompetitionContextFeatureName.PEER_MISSING_CORE_FIELD_RATE: (rates["missing"], None),
        CompetitionContextFeatureName.NONCOMPETITIVE_PROCESS_DEVIATION:
            _deviation(target["noncompetitive"], rates["noncompetitive"],
                       "target_competition_process_unknown",
                       "peer_noncompetitive_process_rate_unavailable"),
        CompetitionContextFeatureName.SINGLE_OFFER_DEVIATION:
            _deviation(target["single"], rates["single"],
                       "target_number_of_offers_missing",
                       "peer_single_offer_rate_unavailable"),
        CompetitionContextFeatureName.ZERO_OFFER_DEVIATION:
            _deviation(target["zero"], rates["zero"],
                       "target_number_of_offers_missing",
                       "peer_zero_offer_rate_unavailable"),
        CompetitionContextFeatureName.ONLY_ONE_SOURCE_SOLICITED_DEVIATION:
            _deviation(target["one_source"], rates["one_source"],
                       "target_solicitation_procedure_unknown",
                       "peer_only_one_source_solicited_rate_unavailable"),
        CompetitionContextFeatureName.NONCOMPETITION_AUTHORITY_REPORTED_DEVIATION:
            _deviation(target["authority"], rates["authority"],
                       "target_noncompetition_authority_status_unknown",
                       "peer_noncompetition_authority_rate_unavailable"),
        CompetitionContextFeatureName.FULL_OPEN_AFTER_EXCLUSION_DEVIATION:
            _deviation(target["after_exclusion"], rates["after_exclusion"],
                       "target_competition_extent_unknown",
                       "peer_full_open_after_exclusion_rate_unavailable"),
        CompetitionContextFeatureName.CONFLICT_DEVIATION:
            _deviation(target["conflict"], rates["conflict"],
                       "target_conflict_status_unknown",
                       "peer_conflict_rate_unavailable"),
        CompetitionContextFeatureName.MISSING_CORE_FIELD_DEVIATION:
            _deviation(target["missing"], rates["missing"],
                       "target_missing_core_field_status_unknown",
                       "peer_missing_core_field_rate_unavailable"),
    }


def _zero_offer(context: CompetitionContext) -> bool | None:
    outcome = context.target_evidence.offer_outcome_kind
    if outcome is OfferOutcomeKind.UNKNOWN:
        return None
    return outcome is OfferOutcomeKind.ZERO_REPORTED


def _after_exclusion(context: CompetitionContext) -> bool | None:
    extent = context.target_evidence.extent_kind
    if extent is CompetitionExtentKind.UNKNOWN:
        return None
    return extent is CompetitionExtentKind.FULL_AND_OPEN_AFTER_EXCLUSION


def _rate(value: Decimal | None, reason: str) -> tuple[Decimal | None, str | None]:
    if value is None:
        return None, reason
    _validate_fraction(value, "peer prevalence")
    return value, None


def _deviation(
    target: bool | None,
    peer_rate: Decimal | None,
    target_missing_reason: str,
    peer_missing_reason: str,
) -> tuple[Decimal | None, str | None]:
    if target is None:
        return None, target_missing_reason
    if peer_rate is None:
        return None, peer_missing_reason
    _validate_fraction(peer_rate, "peer rate")
    return (Decimal(1) if target else Decimal(0)) - peer_rate, None


def _validate_fraction(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise CompetitionContextFeatureError(f"{name} must be finite Decimal")
    if value < Decimal(0) or value > Decimal(1):
        raise CompetitionContextFeatureError(f"{name} must be between 0 and 1")


def _validate_digest(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise CompetitionContextFeatureError(f"{name} must be a SHA-256 hex digest")
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
