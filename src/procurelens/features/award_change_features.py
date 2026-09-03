"""Explainable award-change feature candidates for ProcureLens.

Transforms one resolved award-change context into deterministic candidate
features for later anomaly detectors. Raw activity, peer prevalence, exposure-
adjusted rates, base-value-normalized ratios, empirical position, and robust
distance remain separate measurements.

No smoothing prior, imputation, anomaly cutoff, risk score, or misconduct
conclusion lives here. Later model specifications explicitly select feature
names/families rather than consuming this contract blindly.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Callable

from procurelens.features.award_change_context import (
    AwardChangeContext,
    AwardChangeContextResult,
)
from procurelens.features.award_change_reference import AwardChangeObservation
from procurelens.statistics.robust import (
    RobustSummary,
    empirical_position_sorted,
    iqr_distance,
    modified_z,
    sorted_decimals,
    summarize_sorted,
)


class AwardChangeFeatureError(ValueError):
    """Raised when award-change feature evidence is inconsistent."""


class AwardChangeFeatureFamily(str, Enum):
    SUPPORT = "support"
    ACTIVITY = "activity"
    RATE = "rate"
    VALUE_RATIO = "value_ratio"
    PREVALENCE = "prevalence"
    RELATIVE = "relative"
    RANK = "rank"
    ROBUST_DISTANCE = "robust_distance"
    QUALITY = "quality"


class AwardChangeFeatureName(str, Enum):
    PEER_AWARD_COUNT = "award_change_peer_award_count"
    BASE_OBLIGATION_COVERAGE = "award_change_peer_base_obligation_coverage"
    TARGET_OBSERVABLE_FOLLOWUP_DAYS = "award_change_target_observable_followup_days"
    LIFECYCLE_UNKNOWN_ACTION_COUNT = "award_change_lifecycle_unknown_action_count"
    MAX_TRANSACTIONS_ONE_MODIFICATION_NUMBER = (
        "award_change_max_transactions_on_one_modification_number"
    )

    MODIFICATION_ACTION_COUNT = "award_change_modification_action_count"
    DISTINCT_MODIFICATION_NUMBER_COUNT = (
        "award_change_distinct_modification_number_count"
    )
    MODIFICATION_ACTION_RATE = (
        "award_change_modification_actions_per_observed_day"
    )
    GROSS_MODIFICATION_ACTIVITY_PER_DAY = (
        "award_change_gross_modification_activity_per_observed_day"
    )
    GROSS_MODIFICATION_TO_BASE = (
        "award_change_gross_modification_to_base_obligation"
    )
    NET_MODIFICATION_TO_BASE = (
        "award_change_net_modification_to_base_obligation"
    )
    DEOBLIGATION_TO_BASE = (
        "award_change_deobligation_to_base_obligation"
    )

    PEER_MODIFICATION_PREVALENCE = (
        "award_change_peer_modification_prevalence"
    )
    MODIFICATION_PRESENCE_DEVIATION = (
        "award_change_modification_presence_deviation"
    )
    PEER_DEOBLIGATION_PREVALENCE = (
        "award_change_peer_deobligation_prevalence"
    )
    DEOBLIGATION_PRESENCE_DEVIATION = (
        "award_change_deobligation_presence_deviation"
    )
    PEER_POSITIVE_MODIFICATION_PREVALENCE = (
        "award_change_peer_positive_modification_prevalence"
    )
    POSITIVE_MODIFICATION_PRESENCE_DEVIATION = (
        "award_change_positive_modification_presence_deviation"
    )

    MODIFICATION_COUNT_POSITION = "award_change_modification_count_position"
    MODIFICATION_COUNT_MODIFIED_Z = (
        "award_change_modification_count_modified_z"
    )
    MODIFICATION_COUNT_IQR_DISTANCE = (
        "award_change_modification_count_iqr_distance"
    )

    MODIFICATION_RATE_POSITION = "award_change_modification_rate_position"
    MODIFICATION_RATE_MODIFIED_Z = (
        "award_change_modification_rate_modified_z"
    )
    MODIFICATION_RATE_IQR_DISTANCE = (
        "award_change_modification_rate_iqr_distance"
    )

    GROSS_RATIO_POSITION = "award_change_gross_ratio_position"
    GROSS_RATIO_MODIFIED_Z = "award_change_gross_ratio_modified_z"
    GROSS_RATIO_IQR_DISTANCE = "award_change_gross_ratio_iqr_distance"

    NET_RATIO_POSITION = "award_change_net_ratio_position"
    NET_RATIO_MODIFIED_Z = "award_change_net_ratio_modified_z"
    NET_RATIO_IQR_DISTANCE = "award_change_net_ratio_iqr_distance"

    DEOBLIGATION_RATIO_POSITION = "award_change_deobligation_ratio_position"
    DEOBLIGATION_RATIO_MODIFIED_Z = (
        "award_change_deobligation_ratio_modified_z"
    )
    DEOBLIGATION_RATIO_IQR_DISTANCE = (
        "award_change_deobligation_ratio_iqr_distance"
    )


@dataclass(frozen=True, slots=True)
class AwardChangeFeatureDefinition:
    name: AwardChangeFeatureName
    family: AwardChangeFeatureFamily
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", AwardChangeFeatureName(self.name))
        object.__setattr__(self, "family", AwardChangeFeatureFamily(self.family))
        text = self.description.strip()
        if not text:
            raise AwardChangeFeatureError("feature description must not be blank")
        object.__setattr__(self, "description", text)


def _d(
    name: AwardChangeFeatureName,
    family: AwardChangeFeatureFamily,
    description: str,
) -> AwardChangeFeatureDefinition:
    return AwardChangeFeatureDefinition(name, family, description)


_DEFINITIONS = (
    _d(AwardChangeFeatureName.PEER_AWARD_COUNT, AwardChangeFeatureFamily.SUPPORT,
       "Number of other eligible awards in the selected formation-context market."),
    _d(AwardChangeFeatureName.BASE_OBLIGATION_COVERAGE, AwardChangeFeatureFamily.SUPPORT,
       "Fraction of peer awards with a usable non-zero base-award obligation magnitude."),
    _d(AwardChangeFeatureName.TARGET_OBSERVABLE_FOLLOWUP_DAYS, AwardChangeFeatureFamily.SUPPORT,
       "Observable calendar days from target base action through the reference population end."),
    _d(AwardChangeFeatureName.LIFECYCLE_UNKNOWN_ACTION_COUNT, AwardChangeFeatureFamily.QUALITY,
       "Target award actions whose lifecycle position cannot be classified from modification number."),
    _d(AwardChangeFeatureName.MAX_TRANSACTIONS_ONE_MODIFICATION_NUMBER, AwardChangeFeatureFamily.QUALITY,
       "Maximum target transactions sharing one reported non-zero modification number."),
    _d(AwardChangeFeatureName.MODIFICATION_ACTION_COUNT, AwardChangeFeatureFamily.ACTIVITY,
       "Observed target modification action count."),
    _d(AwardChangeFeatureName.DISTINCT_MODIFICATION_NUMBER_COUNT, AwardChangeFeatureFamily.ACTIVITY,
       "Observed distinct non-zero modification-number count."),
    _d(AwardChangeFeatureName.MODIFICATION_ACTION_RATE, AwardChangeFeatureFamily.RATE,
       "Observed modification actions divided by observable follow-up days."),
    _d(AwardChangeFeatureName.GROSS_MODIFICATION_ACTIVITY_PER_DAY, AwardChangeFeatureFamily.RATE,
       "Absolute modification obligation activity divided by observable follow-up days."),
    _d(AwardChangeFeatureName.GROSS_MODIFICATION_TO_BASE, AwardChangeFeatureFamily.VALUE_RATIO,
       "Absolute modification obligation activity divided by base-award obligation magnitude."),
    _d(AwardChangeFeatureName.NET_MODIFICATION_TO_BASE, AwardChangeFeatureFamily.VALUE_RATIO,
       "Signed net modification obligation divided by base-award obligation magnitude."),
    _d(AwardChangeFeatureName.DEOBLIGATION_TO_BASE, AwardChangeFeatureFamily.VALUE_RATIO,
       "Modification deobligation magnitude divided by base-award obligation magnitude."),
    _d(AwardChangeFeatureName.PEER_MODIFICATION_PREVALENCE, AwardChangeFeatureFamily.PREVALENCE,
       "Fraction of peer awards with at least one observed modification."),
    _d(AwardChangeFeatureName.MODIFICATION_PRESENCE_DEVIATION, AwardChangeFeatureFamily.RELATIVE,
       "Target modification-presence indicator minus peer modification prevalence."),
    _d(AwardChangeFeatureName.PEER_DEOBLIGATION_PREVALENCE, AwardChangeFeatureFamily.PREVALENCE,
       "Fraction of peer awards with at least one modification deobligation."),
    _d(AwardChangeFeatureName.DEOBLIGATION_PRESENCE_DEVIATION, AwardChangeFeatureFamily.RELATIVE,
       "Target deobligation-presence indicator minus peer deobligation prevalence."),
    _d(AwardChangeFeatureName.PEER_POSITIVE_MODIFICATION_PREVALENCE, AwardChangeFeatureFamily.PREVALENCE,
       "Fraction of peer awards with positive modification obligation activity."),
    _d(AwardChangeFeatureName.POSITIVE_MODIFICATION_PRESENCE_DEVIATION, AwardChangeFeatureFamily.RELATIVE,
       "Target positive-modification indicator minus peer positive-modification prevalence."),
    _d(AwardChangeFeatureName.MODIFICATION_COUNT_POSITION, AwardChangeFeatureFamily.RANK,
       "Tie-aware target modification-count position among peer awards."),
    _d(AwardChangeFeatureName.MODIFICATION_COUNT_MODIFIED_Z, AwardChangeFeatureFamily.ROBUST_DISTANCE,
       "Median/MAD distance of target modification count from peer awards."),
    _d(AwardChangeFeatureName.MODIFICATION_COUNT_IQR_DISTANCE, AwardChangeFeatureFamily.ROBUST_DISTANCE,
       "Median-centered target modification-count distance in peer IQR units."),
    _d(AwardChangeFeatureName.MODIFICATION_RATE_POSITION, AwardChangeFeatureFamily.RANK,
       "Tie-aware target exposure-adjusted modification-rate position among peers."),
    _d(AwardChangeFeatureName.MODIFICATION_RATE_MODIFIED_Z, AwardChangeFeatureFamily.ROBUST_DISTANCE,
       "Median/MAD distance of target exposure-adjusted modification rate."),
    _d(AwardChangeFeatureName.MODIFICATION_RATE_IQR_DISTANCE, AwardChangeFeatureFamily.ROBUST_DISTANCE,
       "Median-centered target modification-rate distance in peer IQR units."),
    _d(AwardChangeFeatureName.GROSS_RATIO_POSITION, AwardChangeFeatureFamily.RANK,
       "Tie-aware target gross-modification/base-obligation ratio position."),
    _d(AwardChangeFeatureName.GROSS_RATIO_MODIFIED_Z, AwardChangeFeatureFamily.ROBUST_DISTANCE,
       "Median/MAD target gross-modification/base-obligation ratio distance."),
    _d(AwardChangeFeatureName.GROSS_RATIO_IQR_DISTANCE, AwardChangeFeatureFamily.ROBUST_DISTANCE,
       "Median-centered target gross-modification/base ratio distance in IQR units."),
    _d(AwardChangeFeatureName.NET_RATIO_POSITION, AwardChangeFeatureFamily.RANK,
       "Tie-aware target signed net-modification/base-obligation ratio position."),
    _d(AwardChangeFeatureName.NET_RATIO_MODIFIED_Z, AwardChangeFeatureFamily.ROBUST_DISTANCE,
       "Median/MAD target signed net-modification/base-obligation ratio distance."),
    _d(AwardChangeFeatureName.NET_RATIO_IQR_DISTANCE, AwardChangeFeatureFamily.ROBUST_DISTANCE,
       "Median-centered target net-modification/base ratio distance in IQR units."),
    _d(AwardChangeFeatureName.DEOBLIGATION_RATIO_POSITION, AwardChangeFeatureFamily.RANK,
       "Tie-aware target deobligation/base-obligation ratio position."),
    _d(AwardChangeFeatureName.DEOBLIGATION_RATIO_MODIFIED_Z, AwardChangeFeatureFamily.ROBUST_DISTANCE,
       "Median/MAD target deobligation/base-obligation ratio distance."),
    _d(AwardChangeFeatureName.DEOBLIGATION_RATIO_IQR_DISTANCE, AwardChangeFeatureFamily.ROBUST_DISTANCE,
       "Median-centered target deobligation/base ratio distance in IQR units."),
)
_BY_NAME: Mapping[AwardChangeFeatureName, AwardChangeFeatureDefinition] = MappingProxyType(
    {item.name: item for item in _DEFINITIONS}
)
if len(_BY_NAME) != len(_DEFINITIONS):
    raise RuntimeError("duplicate award-change feature definitions")


def award_change_feature_definitions() -> tuple[AwardChangeFeatureDefinition, ...]:
    return _DEFINITIONS


def award_change_feature_definition_sha256() -> str:
    return _digest([(item.name.value, item.family.value, item.description) for item in _DEFINITIONS])


@dataclass(frozen=True, slots=True)
class AwardChangeFeature:
    definition: AwardChangeFeatureDefinition
    value: Decimal | None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if _BY_NAME.get(self.definition.name) != self.definition:
            raise AwardChangeFeatureError("non-canonical feature definition")
        if self.value is None:
            reason = None if self.unavailable_reason is None else self.unavailable_reason.strip()
            if not reason:
                raise AwardChangeFeatureError(
                    f"{self.definition.name.value}: missing value requires a reason"
                )
            object.__setattr__(self, "unavailable_reason", reason)
            return
        if not isinstance(self.value, Decimal) or not self.value.is_finite():
            raise AwardChangeFeatureError("feature value must be finite Decimal")
        if self.unavailable_reason is not None:
            raise AwardChangeFeatureError("available feature cannot carry an unavailable reason")

    @property
    def name(self) -> AwardChangeFeatureName:
        return self.definition.name

    @property
    def family(self) -> AwardChangeFeatureFamily:
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
class AwardChangeFeatureSet:
    award_id: str
    context_available: bool
    context_evidence_sha256: str | None
    market_level: str | None
    market_key_sha256: str | None
    peer_distribution_sha256: str | None
    feature_summaries: Mapping[str, RobustSummary | None]
    features: tuple[AwardChangeFeature, ...]

    def __post_init__(self) -> None:
        award_id = self.award_id.strip()
        if not award_id:
            raise AwardChangeFeatureError("award_id must not be blank")
        object.__setattr__(self, "award_id", award_id)
        if not isinstance(self.context_available, bool):
            raise AwardChangeFeatureError("context_available must be bool")
        if self.context_available:
            if not self.context_evidence_sha256:
                raise AwardChangeFeatureError("available feature set requires context evidence sha")
            object.__setattr__(self, "context_evidence_sha256",
                               _validate_digest(self.context_evidence_sha256, "context_evidence_sha256"))
            if not self.market_level or not self.market_level.strip():
                raise AwardChangeFeatureError("available feature set requires market_level")
            object.__setattr__(self, "market_level", self.market_level.strip())
            if not self.market_key_sha256 or not self.peer_distribution_sha256:
                raise AwardChangeFeatureError("available feature set requires market and peer digests")
            object.__setattr__(self, "market_key_sha256",
                               _validate_digest(self.market_key_sha256, "market_key_sha256"))
            object.__setattr__(self, "peer_distribution_sha256",
                               _validate_digest(self.peer_distribution_sha256, "peer_distribution_sha256"))
        else:
            if any(value is not None for value in (
                self.context_evidence_sha256, self.market_level,
                self.market_key_sha256, self.peer_distribution_sha256,
            )):
                raise AwardChangeFeatureError("unavailable feature set cannot carry context metadata")

        summaries = dict(self.feature_summaries)
        for key, summary in summaries.items():
            if not key.strip():
                raise AwardChangeFeatureError("summary key must not be blank")
            if summary is not None and not isinstance(summary, RobustSummary):
                raise TypeError("feature summary values must be RobustSummary or None")
        object.__setattr__(self, "feature_summaries", MappingProxyType(summaries))

        features = tuple(self.features)
        observed = tuple(feature.name for feature in features)
        expected = tuple(definition.name for definition in _DEFINITIONS)
        if observed != expected:
            raise AwardChangeFeatureError(
                "feature set must contain every canonical feature exactly once in contract order"
            )
        object.__setattr__(self, "features", features)

    @property
    def definition_sha256(self) -> str:
        return award_change_feature_definition_sha256()

    @property
    def feature_evidence_sha256(self) -> str:
        return _digest({
            "award_id": self.award_id,
            "context_available": self.context_available,
            "context_evidence_sha256": self.context_evidence_sha256,
            "market_level": self.market_level,
            "market_key_sha256": self.market_key_sha256,
            "peer_distribution_sha256": self.peer_distribution_sha256,
            "definition_sha256": self.definition_sha256,
            "summaries": {key: _summary_dict(value) for key, value in sorted(self.feature_summaries.items())},
            "features": [feature.as_dict() for feature in self.features],
        })

    def get(self, name: AwardChangeFeatureName | str) -> AwardChangeFeature:
        wanted = AwardChangeFeatureName(name)
        for feature in self.features:
            if feature.name is wanted:
                return feature
        raise AwardChangeFeatureError(f"unknown feature: {wanted.value}")

    def select(
        self,
        names: Iterable[AwardChangeFeatureName | str],
        *,
        require_complete: bool = True,
    ) -> tuple[AwardChangeFeature, ...]:
        if isinstance(names, (str, bytes)):
            raise AwardChangeFeatureError("feature names must be an iterable")
        selected: list[AwardChangeFeature] = []
        seen: set[AwardChangeFeatureName] = set()
        for item in names:
            name = AwardChangeFeatureName(item)
            if name in seen:
                raise AwardChangeFeatureError(f"duplicate selected feature: {name.value}")
            seen.add(name)
            feature = self.get(name)
            if require_complete and not feature.available:
                raise AwardChangeFeatureError(
                    f"selected feature {name.value} unavailable: {feature.unavailable_reason}"
                )
            selected.append(feature)
        if not selected:
            raise AwardChangeFeatureError("at least one feature must be selected")
        return tuple(selected)

    def as_dict(self) -> dict[str, Any]:
        return {
            "award_id": self.award_id,
            "context_available": self.context_available,
            "context_evidence_sha256": self.context_evidence_sha256,
            "market_level": self.market_level,
            "market_key_sha256": self.market_key_sha256,
            "peer_distribution_sha256": self.peer_distribution_sha256,
            "definition_sha256": self.definition_sha256,
            "feature_evidence_sha256": self.feature_evidence_sha256,
            "feature_summaries": {key: _summary_dict(value) for key, value in sorted(self.feature_summaries.items())},
            "features": [feature.as_dict() for feature in self.features],
        }


@dataclass(frozen=True, slots=True)
class _MetricResult:
    target: Decimal | None
    peers: tuple[Decimal, ...]
    summary: RobustSummary | None
    position: Decimal | None
    modified_z_value: Decimal | None
    iqr_distance_value: Decimal | None
    unavailable_reason: str | None


def build_award_change_features(result: AwardChangeContextResult) -> AwardChangeFeatureSet:
    if not isinstance(result, AwardChangeContextResult):
        raise TypeError("result must be AwardChangeContextResult")
    if not result.available:
        reason = result.unavailable_reasons[0] if result.unavailable_reasons else "award_change_context_unavailable"
        features = tuple(AwardChangeFeature(definition, None, reason) for definition in _DEFINITIONS)
        return AwardChangeFeatureSet(
            award_id=result.award_id,
            context_available=False,
            context_evidence_sha256=None,
            market_level=None,
            market_key_sha256=None,
            peer_distribution_sha256=None,
            feature_summaries={},
            features=features,
        )

    context = result.context
    assert context is not None
    target = context.target

    mod_count = _metric(target, context, lambda item: Decimal(item.modification_action_count),
                        "modification_action_count_unavailable")
    mod_rate = _metric(target, context, lambda item: item.modification_actions_per_observed_day,
                       "modification_action_rate_unavailable")
    gross_ratio = _metric(target, context, lambda item: item.gross_modification_to_base_obligation,
                          "base_award_obligation_unavailable")
    net_ratio = _metric(target, context, lambda item: item.net_modification_to_base_obligation,
                        "base_award_obligation_unavailable")
    deob_ratio = _metric(target, context, lambda item: item.deobligation_to_base_obligation,
                         "base_award_obligation_unavailable")

    values: dict[AwardChangeFeatureName, tuple[Decimal | None, str | None]] = {
        AwardChangeFeatureName.PEER_AWARD_COUNT: (Decimal(context.peer_award_count), None),
        AwardChangeFeatureName.BASE_OBLIGATION_COVERAGE: (context.base_obligation_coverage, None),
        AwardChangeFeatureName.TARGET_OBSERVABLE_FOLLOWUP_DAYS: (Decimal(target.observable_followup_days), None),
        AwardChangeFeatureName.LIFECYCLE_UNKNOWN_ACTION_COUNT: (Decimal(target.lifecycle_unknown_action_count), None),
        AwardChangeFeatureName.MAX_TRANSACTIONS_ONE_MODIFICATION_NUMBER: (
            Decimal(target.maximum_transactions_on_one_modification_number), None),
        AwardChangeFeatureName.MODIFICATION_ACTION_COUNT: (Decimal(target.modification_action_count), None),
        AwardChangeFeatureName.DISTINCT_MODIFICATION_NUMBER_COUNT: (
            Decimal(target.distinct_modification_number_count), None),
        AwardChangeFeatureName.MODIFICATION_ACTION_RATE: (target.modification_actions_per_observed_day, None),
        AwardChangeFeatureName.GROSS_MODIFICATION_ACTIVITY_PER_DAY: (
            target.gross_modification_activity_per_observed_day, None),
        AwardChangeFeatureName.GROSS_MODIFICATION_TO_BASE: _optional_value(
            target.gross_modification_to_base_obligation, "base_award_obligation_unavailable"),
        AwardChangeFeatureName.NET_MODIFICATION_TO_BASE: _optional_value(
            target.net_modification_to_base_obligation, "base_award_obligation_unavailable"),
        AwardChangeFeatureName.DEOBLIGATION_TO_BASE: _optional_value(
            target.deobligation_to_base_obligation, "base_award_obligation_unavailable"),
        AwardChangeFeatureName.PEER_MODIFICATION_PREVALENCE: (context.peer_modification_prevalence, None),
        AwardChangeFeatureName.MODIFICATION_PRESENCE_DEVIATION: (
            _indicator(target.has_modifications) - context.peer_modification_prevalence, None),
        AwardChangeFeatureName.PEER_DEOBLIGATION_PREVALENCE: (context.peer_deobligation_prevalence, None),
        AwardChangeFeatureName.DEOBLIGATION_PRESENCE_DEVIATION: (
            _indicator(target.has_deobligation) - context.peer_deobligation_prevalence, None),
        AwardChangeFeatureName.PEER_POSITIVE_MODIFICATION_PREVALENCE: (
            context.peer_positive_modification_prevalence, None),
        AwardChangeFeatureName.POSITIVE_MODIFICATION_PRESENCE_DEVIATION: (
            _indicator(target.has_positive_modification) - context.peer_positive_modification_prevalence, None),
        AwardChangeFeatureName.MODIFICATION_COUNT_POSITION: _metric_value(mod_count.position, mod_count.unavailable_reason),
        AwardChangeFeatureName.MODIFICATION_COUNT_MODIFIED_Z: _distance_value(
            mod_count.modified_z_value, mod_count, "peer_modification_count_mad_zero"),
        AwardChangeFeatureName.MODIFICATION_COUNT_IQR_DISTANCE: _distance_value(
            mod_count.iqr_distance_value, mod_count, "peer_modification_count_iqr_zero"),
        AwardChangeFeatureName.MODIFICATION_RATE_POSITION: _metric_value(mod_rate.position, mod_rate.unavailable_reason),
        AwardChangeFeatureName.MODIFICATION_RATE_MODIFIED_Z: _distance_value(
            mod_rate.modified_z_value, mod_rate, "peer_modification_rate_mad_zero"),
        AwardChangeFeatureName.MODIFICATION_RATE_IQR_DISTANCE: _distance_value(
            mod_rate.iqr_distance_value, mod_rate, "peer_modification_rate_iqr_zero"),
        AwardChangeFeatureName.GROSS_RATIO_POSITION: _metric_value(gross_ratio.position, gross_ratio.unavailable_reason),
        AwardChangeFeatureName.GROSS_RATIO_MODIFIED_Z: _distance_value(
            gross_ratio.modified_z_value, gross_ratio, "peer_gross_ratio_mad_zero"),
        AwardChangeFeatureName.GROSS_RATIO_IQR_DISTANCE: _distance_value(
            gross_ratio.iqr_distance_value, gross_ratio, "peer_gross_ratio_iqr_zero"),
        AwardChangeFeatureName.NET_RATIO_POSITION: _metric_value(net_ratio.position, net_ratio.unavailable_reason),
        AwardChangeFeatureName.NET_RATIO_MODIFIED_Z: _distance_value(
            net_ratio.modified_z_value, net_ratio, "peer_net_ratio_mad_zero"),
        AwardChangeFeatureName.NET_RATIO_IQR_DISTANCE: _distance_value(
            net_ratio.iqr_distance_value, net_ratio, "peer_net_ratio_iqr_zero"),
        AwardChangeFeatureName.DEOBLIGATION_RATIO_POSITION: _metric_value(deob_ratio.position, deob_ratio.unavailable_reason),
        AwardChangeFeatureName.DEOBLIGATION_RATIO_MODIFIED_Z: _distance_value(
            deob_ratio.modified_z_value, deob_ratio, "peer_deobligation_ratio_mad_zero"),
        AwardChangeFeatureName.DEOBLIGATION_RATIO_IQR_DISTANCE: _distance_value(
            deob_ratio.iqr_distance_value, deob_ratio, "peer_deobligation_ratio_iqr_zero"),
    }

    features = tuple(AwardChangeFeature(
        definition, values[definition.name][0], values[definition.name][1]
    ) for definition in _DEFINITIONS)
    summaries = {
        "modification_action_count": mod_count.summary,
        "modification_action_rate": mod_rate.summary,
        "gross_modification_to_base_obligation": gross_ratio.summary,
        "net_modification_to_base_obligation": net_ratio.summary,
        "deobligation_to_base_obligation": deob_ratio.summary,
    }
    return AwardChangeFeatureSet(
        award_id=context.award_id,
        context_available=True,
        context_evidence_sha256=context.evidence_sha256,
        market_level=context.market_level,
        market_key_sha256=context.market_key.sha256_hex,
        peer_distribution_sha256=_peer_distribution_digest(context),
        feature_summaries=summaries,
        features=features,
    )


def _metric(
    target: AwardChangeObservation,
    context: AwardChangeContext,
    extractor: Callable[[AwardChangeObservation], Decimal | None],
    missing_reason: str,
) -> _MetricResult:
    target_value = extractor(target)
    if target_value is not None:
        _finite(target_value, "target metric")
    peer_values = []
    for peer in context.peers:
        value = extractor(peer)
        if value is None:
            continue
        _finite(value, "peer metric")
        peer_values.append(value)
    if target_value is None:
        return _MetricResult(None, tuple(sorted(peer_values)), None, None, None, None, missing_reason)
    if not peer_values:
        return _MetricResult(target_value, (), None, None, None, None, "peer_metric_unavailable")
    ordered = sorted_decimals(peer_values)
    summary = summarize_sorted(ordered)
    position = empirical_position_sorted(ordered, target_value).midpoint_fraction
    return _MetricResult(
        target_value, ordered, summary, position,
        modified_z(target_value, summary),
        iqr_distance(target_value, summary), None,
    )


def _metric_value(value: Decimal | None, reason: str | None) -> tuple[Decimal | None, str | None]:
    return (value, None) if value is not None else (None, reason or "peer_metric_unavailable")


def _distance_value(
    value: Decimal | None,
    metric: _MetricResult,
    zero_spread_reason: str,
) -> tuple[Decimal | None, str | None]:
    if metric.unavailable_reason is not None:
        return None, metric.unavailable_reason
    if value is None:
        return None, zero_spread_reason
    return value, None


def _optional_value(value: Decimal | None, reason: str) -> tuple[Decimal | None, str | None]:
    if value is None:
        return None, reason
    _finite(value, "feature value")
    return value, None


def _indicator(value: bool) -> Decimal:
    return Decimal(1) if value else Decimal(0)


def _finite(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise AwardChangeFeatureError(f"{name} must be finite Decimal")


def _peer_distribution_digest(context: AwardChangeContext) -> str:
    return _digest([(peer.award_id, peer.evidence_sha256) for peer in context.peers])


def _summary_dict(summary: RobustSummary | None) -> dict[str, str | int] | None:
    if summary is None:
        return None
    return {
        "count": summary.count,
        "minimum": str(summary.minimum),
        "first_quartile": str(summary.first_quartile),
        "median": str(summary.median),
        "third_quartile": str(summary.third_quartile),
        "maximum": str(summary.maximum),
        "interquartile_range": str(summary.interquartile_range),
        "median_absolute_deviation": str(summary.median_absolute_deviation),
        "quantile_method": summary.quantile_method.value,
    }


def _validate_digest(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise AwardChangeFeatureError(f"{name} must be a SHA-256 hex digest")
    return digest


def _digest(value: Any) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()
