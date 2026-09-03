"""Explainable vendor new-award frequency features for ProcureLens.

Transforms one resolved vendor-market context into deterministic candidate
features for later anomaly detectors. Peer-rank and robust-distance features
compare the target vendor with *other* observed winning vendors; market-share
and HHI features still describe the full selected reference market.

The exact snapshot is cross-checked against the context before any feature is
computed. No imputation, anomaly cutoff, risk score, or fraud conclusion lives
here.
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

from procurelens.features.vendor_market import (
    ObservedVendorMarket,
    VendorMarketSnapshot,
)
from procurelens.features.vendor_market_context import (
    TargetReferenceMode,
    VendorMarketContext,
    VendorMarketContextResult,
)
from procurelens.statistics.robust import (
    QuantileMethod,
    RobustSummary,
    empirical_position_sorted,
    iqr_distance,
    modified_z,
    sorted_decimals,
    summarize_sorted,
)


class VendorFrequencyFeatureError(ValueError):
    """Raised when vendor-frequency evidence is inconsistent."""


class VendorFrequencyFeatureFamily(str, Enum):
    COUNT = "count"
    SHARE = "share"
    RELATIVE = "relative"
    RANK = "rank"
    ROBUST_DISTANCE = "robust_distance"
    CONCENTRATION = "concentration"


class VendorFrequencyFeatureName(str, Enum):
    NEW_AWARD_COUNT = "vendor_new_award_count"
    SHARE_IDENTIFIED = "vendor_share_identified_new_awards"
    SHARE_ALL_OBSERVED = "vendor_share_all_observed_new_awards"
    EQUAL_SHARE_LIFT = "vendor_equal_observed_winner_share_lift"
    WIN_COUNT_POSITION = "vendor_win_count_position"
    WIN_COUNT_MODIFIED_Z = "vendor_win_count_modified_z"
    WIN_COUNT_IQR_DISTANCE = "vendor_win_count_iqr_distance"
    SHARE_OF_LARGEST_WINNER = "vendor_share_of_largest_observed_winner"
    HHI_CONTRIBUTION = "vendor_award_count_hhi_contribution"
    HHI_CONTRIBUTION_FRACTION = "vendor_fraction_of_award_count_hhi"


@dataclass(frozen=True, slots=True)
class VendorFrequencyFeatureDefinition:
    name: VendorFrequencyFeatureName
    family: VendorFrequencyFeatureFamily
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", VendorFrequencyFeatureName(self.name))
        object.__setattr__(self, "family", VendorFrequencyFeatureFamily(self.family))
        text = self.description.strip()
        if not text:
            raise VendorFrequencyFeatureError("feature description must not be blank")
        object.__setattr__(self, "description", text)


_DEFINITIONS = (
    VendorFrequencyFeatureDefinition(
        VendorFrequencyFeatureName.NEW_AWARD_COUNT,
        VendorFrequencyFeatureFamily.COUNT,
        "Target vendor observed new-award count in the selected reference market.",
    ),
    VendorFrequencyFeatureDefinition(
        VendorFrequencyFeatureName.SHARE_IDENTIFIED,
        VendorFrequencyFeatureFamily.SHARE,
        "Target share of new awards whose winning-vendor identity is known.",
    ),
    VendorFrequencyFeatureDefinition(
        VendorFrequencyFeatureName.SHARE_ALL_OBSERVED,
        VendorFrequencyFeatureFamily.SHARE,
        "Target share of all observed new awards, retaining unidentified winners.",
    ),
    VendorFrequencyFeatureDefinition(
        VendorFrequencyFeatureName.EQUAL_SHARE_LIFT,
        VendorFrequencyFeatureFamily.RELATIVE,
        "Target identified-award share divided by equal share across observed winners.",
    ),
    VendorFrequencyFeatureDefinition(
        VendorFrequencyFeatureName.WIN_COUNT_POSITION,
        VendorFrequencyFeatureFamily.RANK,
        "Tie-aware target count position among other observed winning vendors.",
    ),
    VendorFrequencyFeatureDefinition(
        VendorFrequencyFeatureName.WIN_COUNT_MODIFIED_Z,
        VendorFrequencyFeatureFamily.ROBUST_DISTANCE,
        "Median/MAD target count distance from other observed winning vendors.",
    ),
    VendorFrequencyFeatureDefinition(
        VendorFrequencyFeatureName.WIN_COUNT_IQR_DISTANCE,
        VendorFrequencyFeatureFamily.ROBUST_DISTANCE,
        "Median-centered target count distance in peer winner-count IQR units.",
    ),
    VendorFrequencyFeatureDefinition(
        VendorFrequencyFeatureName.SHARE_OF_LARGEST_WINNER,
        VendorFrequencyFeatureFamily.RELATIVE,
        "Target identified-award share divided by the largest observed winner share.",
    ),
    VendorFrequencyFeatureDefinition(
        VendorFrequencyFeatureName.HHI_CONTRIBUTION,
        VendorFrequencyFeatureFamily.CONCENTRATION,
        "Squared target identified-award share: target additive contribution to count HHI.",
    ),
    VendorFrequencyFeatureDefinition(
        VendorFrequencyFeatureName.HHI_CONTRIBUTION_FRACTION,
        VendorFrequencyFeatureFamily.CONCENTRATION,
        "Fraction of count-based market HHI attributable to the target vendor.",
    ),
)
_BY_NAME: Mapping[
    VendorFrequencyFeatureName, VendorFrequencyFeatureDefinition
] = MappingProxyType({item.name: item for item in _DEFINITIONS})
if len(_BY_NAME) != len(_DEFINITIONS):
    raise RuntimeError("duplicate vendor-frequency feature definitions")


def vendor_frequency_feature_definitions() -> tuple[VendorFrequencyFeatureDefinition, ...]:
    return _DEFINITIONS


def vendor_frequency_feature_definition_sha256() -> str:
    return _digest(
        [
            (item.name.value, item.family.value, item.description)
            for item in _DEFINITIONS
        ]
    )


@dataclass(frozen=True, slots=True)
class VendorFrequencyFeature:
    definition: VendorFrequencyFeatureDefinition
    value: Decimal | None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if _BY_NAME.get(self.definition.name) != self.definition:
            raise VendorFrequencyFeatureError("non-canonical feature definition")
        if self.value is None:
            reason = None if self.unavailable_reason is None else self.unavailable_reason.strip()
            if not reason:
                raise VendorFrequencyFeatureError(
                    f"{self.definition.name.value}: missing value requires a reason"
                )
            object.__setattr__(self, "unavailable_reason", reason)
        else:
            if not isinstance(self.value, Decimal) or not self.value.is_finite():
                raise VendorFrequencyFeatureError("feature value must be finite Decimal")
            if self.unavailable_reason is not None:
                raise VendorFrequencyFeatureError(
                    "available feature cannot carry an unavailable reason"
                )

    @property
    def name(self) -> VendorFrequencyFeatureName:
        return self.definition.name

    @property
    def family(self) -> VendorFrequencyFeatureFamily:
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
class VendorFrequencyFeatureSet:
    transaction_id: str
    award_id: str
    target_identity_sha256: str
    reference_mode: TargetReferenceMode
    market_level: str
    market_key_sha256: str
    plan_sha256: str
    support_spec_sha256: str
    context_evidence_sha256: str
    reference_distribution_sha256: str
    peer_distribution_sha256: str
    identified_new_award_count: int
    observed_winning_vendor_count: int
    peer_winning_vendor_count: int
    peer_winner_count_summary: RobustSummary | None
    features: tuple[VendorFrequencyFeature, ...]

    def __post_init__(self) -> None:
        for name in ("transaction_id", "award_id", "market_level"):
            text = getattr(self, name).strip()
            if not text:
                raise VendorFrequencyFeatureError(f"{name} must not be blank")
            object.__setattr__(self, name, text)
        object.__setattr__(self, "reference_mode", TargetReferenceMode(self.reference_mode))
        for name in (
            "target_identity_sha256",
            "market_key_sha256",
            "plan_sha256",
            "support_spec_sha256",
            "context_evidence_sha256",
            "reference_distribution_sha256",
            "peer_distribution_sha256",
        ):
            object.__setattr__(self, name, _validate_digest(getattr(self, name), name))
        if self.identified_new_award_count < 1 or self.observed_winning_vendor_count < 1:
            raise VendorFrequencyFeatureError(
                "selected market needs identified awards and observed winners"
            )
        if not 0 <= self.peer_winning_vendor_count <= self.observed_winning_vendor_count:
            raise VendorFrequencyFeatureError("peer winning-vendor count is inconsistent")
        if self.peer_winner_count_summary is None:
            if self.peer_winning_vendor_count:
                raise VendorFrequencyFeatureError(
                    "non-empty peer population requires a robust summary"
                )
        elif (
            not isinstance(self.peer_winner_count_summary, RobustSummary)
            or self.peer_winner_count_summary.count != self.peer_winning_vendor_count
        ):
            raise VendorFrequencyFeatureError("peer winner-count summary is inconsistent")
        if not isinstance(self.features, tuple):
            object.__setattr__(self, "features", tuple(self.features))
        if tuple(feature.name for feature in self.features) != tuple(
            item.name for item in _DEFINITIONS
        ):
            raise VendorFrequencyFeatureError(
                "feature set must contain every canonical feature once in contract order"
            )

    @property
    def definition_sha256(self) -> str:
        return vendor_frequency_feature_definition_sha256()

    @property
    def evidence_sha256(self) -> str:
        return _digest(self.as_dict(include_evidence_sha=False))

    def get(self, name: VendorFrequencyFeatureName | str) -> VendorFrequencyFeature:
        wanted = VendorFrequencyFeatureName(name)
        for feature in self.features:
            if feature.name is wanted:
                return feature
        raise VendorFrequencyFeatureError(f"unknown feature: {wanted.value}")

    def select(
        self,
        names: Iterable[VendorFrequencyFeatureName | str],
        *,
        require_complete: bool = True,
    ) -> tuple[VendorFrequencyFeature, ...]:
        if isinstance(names, (str, bytes)):
            raise VendorFrequencyFeatureError("feature names must be an iterable")
        result: list[VendorFrequencyFeature] = []
        seen: set[VendorFrequencyFeatureName] = set()
        for raw in names:
            name = VendorFrequencyFeatureName(raw)
            if name in seen:
                raise VendorFrequencyFeatureError(f"duplicate selected feature: {name.value}")
            seen.add(name)
            feature = self.get(name)
            if require_complete and not feature.available:
                raise VendorFrequencyFeatureError(
                    f"{name.value} unavailable: {feature.unavailable_reason}"
                )
            result.append(feature)
        if not result:
            raise VendorFrequencyFeatureError("at least one feature must be selected")
        return tuple(result)

    def as_dict(self, *, include_evidence_sha: bool = True) -> dict[str, Any]:
        summary = self.peer_winner_count_summary
        result = {
            "transaction_id": self.transaction_id,
            "award_id": self.award_id,
            "target_identity_sha256": self.target_identity_sha256,
            "reference_mode": self.reference_mode.value,
            "market_level": self.market_level,
            "market_key_sha256": self.market_key_sha256,
            "plan_sha256": self.plan_sha256,
            "support_spec_sha256": self.support_spec_sha256,
            "context_evidence_sha256": self.context_evidence_sha256,
            "reference_distribution_sha256": self.reference_distribution_sha256,
            "peer_distribution_sha256": self.peer_distribution_sha256,
            "identified_new_award_count": self.identified_new_award_count,
            "observed_winning_vendor_count": self.observed_winning_vendor_count,
            "peer_winning_vendor_count": self.peer_winning_vendor_count,
            "peer_winner_count_summary": None if summary is None else {
                "count": summary.count,
                "minimum": str(summary.minimum),
                "first_quartile": str(summary.first_quartile),
                "median": str(summary.median),
                "third_quartile": str(summary.third_quartile),
                "maximum": str(summary.maximum),
                "interquartile_range": str(summary.interquartile_range),
                "median_absolute_deviation": str(summary.median_absolute_deviation),
                "quantile_method": summary.quantile_method.value,
            },
            "definition_sha256": self.definition_sha256,
            "features": [feature.as_dict() for feature in self.features],
        }
        if include_evidence_sha:
            result["evidence_sha256"] = self.evidence_sha256
        return result


@dataclass(frozen=True, slots=True)
class VendorFrequencyFeatureResult:
    transaction_id: str
    feature_set: VendorFrequencyFeatureSet | None
    unavailable_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        txid = self.transaction_id.strip()
        if not txid:
            raise VendorFrequencyFeatureError("transaction_id must not be blank")
        object.__setattr__(self, "transaction_id", txid)
        if self.feature_set is not None:
            if self.feature_set.transaction_id != txid or self.unavailable_reasons:
                raise VendorFrequencyFeatureError("feature result is internally inconsistent")
        object.__setattr__(
            self,
            "unavailable_reasons",
            tuple(reason.strip() for reason in self.unavailable_reasons if reason.strip()),
        )

    @property
    def available(self) -> bool:
        return self.feature_set is not None


def build_vendor_frequency_features(
    context_result: VendorMarketContextResult,
    snapshot: VendorMarketSnapshot,
    *,
    quantile_method: QuantileMethod = QuantileMethod.LINEAR_TYPE7,
) -> VendorFrequencyFeatureResult:
    """Build features from the exact selected market and a target-excluded peer set."""

    if not isinstance(context_result, VendorMarketContextResult):
        raise TypeError("context_result must be VendorMarketContextResult")
    if not isinstance(snapshot, VendorMarketSnapshot):
        raise TypeError("snapshot must be VendorMarketSnapshot")
    quantile_method = QuantileMethod(quantile_method)

    if not context_result.available or context_result.context is None:
        return VendorFrequencyFeatureResult(
            context_result.transaction_id,
            None,
            context_result.unavailable_reasons
            or ("vendor_market_context_unavailable",),
        )

    context = context_result.context
    _validate_snapshot(context_result, context, snapshot)
    market = snapshot.get(context.market_key)
    if market is None:
        raise VendorFrequencyFeatureError("selected market is absent from snapshot")
    reference = (
        market.without_one_award(context.target_identity.key)
        if context.reference_mode is TargetReferenceMode.LEAVE_ONE_OUT
        else market
    )
    _validate_reference(context, reference)

    peer_counts = dict(reference.awards_by_winning_vendor)
    peer_counts.pop(context.target_identity.key, None)
    target_count = Decimal(context.target_vendor_new_award_count)

    summary: RobustSummary | None = None
    position: Decimal | None = None
    mad_z: Decimal | None = None
    iqr_units: Decimal | None = None
    if peer_counts:
        ordered = sorted_decimals(Decimal(count) for count in peer_counts.values())
        summary = summarize_sorted(ordered, method=quantile_method)
        position = empirical_position_sorted(ordered, target_count).midpoint_fraction
        mad_z = modified_z(target_count, summary)
        iqr_units = iqr_distance(target_count, summary)

    share_identified = context.target_share_of_identified_new_awards
    share_all = context.target_share_of_all_observed_new_awards
    equal_share = Decimal(1) / Decimal(context.observed_winning_vendor_count)
    largest_share = context.largest_winner_award_share
    hhi_contribution = share_identified ** 2
    if context.award_count_hhi <= 0 or largest_share <= 0:
        raise VendorFrequencyFeatureError("selected market concentration is invalid")

    values: dict[
        VendorFrequencyFeatureName, tuple[Decimal | None, str | None]
    ] = {
        VendorFrequencyFeatureName.NEW_AWARD_COUNT: (target_count, None),
        VendorFrequencyFeatureName.SHARE_IDENTIFIED: (share_identified, None),
        VendorFrequencyFeatureName.SHARE_ALL_OBSERVED: (share_all, None),
        VendorFrequencyFeatureName.EQUAL_SHARE_LIFT: (
            share_identified / equal_share, None
        ),
        VendorFrequencyFeatureName.WIN_COUNT_POSITION: _optional(
            position, "no_other_observed_winning_vendors"
        ),
        VendorFrequencyFeatureName.WIN_COUNT_MODIFIED_Z: _distance_optional(
            mad_z, bool(peer_counts), "peer_winner_count_mad_zero"
        ),
        VendorFrequencyFeatureName.WIN_COUNT_IQR_DISTANCE: _distance_optional(
            iqr_units, bool(peer_counts), "peer_winner_count_iqr_zero"
        ),
        VendorFrequencyFeatureName.SHARE_OF_LARGEST_WINNER: (
            share_identified / largest_share, None
        ),
        VendorFrequencyFeatureName.HHI_CONTRIBUTION: (hhi_contribution, None),
        VendorFrequencyFeatureName.HHI_CONTRIBUTION_FRACTION: (
            hhi_contribution / context.award_count_hhi, None
        ),
    }
    features = tuple(
        VendorFrequencyFeature(
            definition,
            values[definition.name][0],
            values[definition.name][1],
        )
        for definition in _DEFINITIONS
    )

    return VendorFrequencyFeatureResult(
        context.transaction_id,
        VendorFrequencyFeatureSet(
            transaction_id=context.transaction_id,
            award_id=context.award_id,
            target_identity_sha256=context.target_identity.sha256_hex,
            reference_mode=context.reference_mode,
            market_level=context.market_level,
            market_key_sha256=context.market_key.sha256_hex,
            plan_sha256=context.plan_sha256,
            support_spec_sha256=context.support_spec_sha256,
            context_evidence_sha256=context.evidence_sha256,
            reference_distribution_sha256=_distribution_sha256(
                context.market_key.sha256_hex,
                reference.awards_by_winning_vendor,
            ),
            peer_distribution_sha256=_distribution_sha256(
                context.market_key.sha256_hex,
                peer_counts,
            ),
            identified_new_award_count=context.identified_new_award_count,
            observed_winning_vendor_count=context.observed_winning_vendor_count,
            peer_winning_vendor_count=len(peer_counts),
            peer_winner_count_summary=summary,
            features=features,
        ),
    )


def _validate_snapshot(
    result: VendorMarketContextResult,
    context: VendorMarketContext,
    snapshot: VendorMarketSnapshot,
) -> None:
    if snapshot.scope is not result.scope or context.target_identity.scope is not snapshot.scope:
        raise VendorFrequencyFeatureError("vendor identity scope mismatch")
    if snapshot.plan.name != context.plan_name or snapshot.plan.sha256_hex != context.plan_sha256:
        raise VendorFrequencyFeatureError("market plan mismatch")
    if result.support_spec.sha256_hex != context.support_spec_sha256:
        raise VendorFrequencyFeatureError("support-spec fingerprint mismatch")


def _validate_reference(
    context: VendorMarketContext,
    reference: ObservedVendorMarket,
) -> None:
    expected = (
        context.observed_new_award_count,
        context.identified_new_award_count,
        context.unidentified_new_award_count,
        context.observed_winning_vendor_count,
        context.vendor_identity_coverage,
        context.award_count_hhi,
        context.largest_winner_award_share,
        context.target_vendor_new_award_count,
    )
    observed = (
        reference.observed_new_award_count,
        reference.new_awards_with_vendor_identity,
        reference.new_awards_without_vendor_identity,
        reference.observed_winning_vendor_count,
        reference.vendor_identity_coverage,
        reference.award_count_hhi,
        reference.largest_winner_award_share,
        reference.awards_by_winning_vendor.get(context.target_identity.key, 0),
    )
    if observed != expected:
        raise VendorFrequencyFeatureError(
            "selected reference market no longer matches context evidence"
        )


def _optional(
    value: Decimal | None, reason: str
) -> tuple[Decimal | None, str | None]:
    return (None, reason) if value is None else (value, None)


def _distance_optional(
    value: Decimal | None,
    has_peers: bool,
    zero_spread_reason: str,
) -> tuple[Decimal | None, str | None]:
    if not has_peers:
        return None, "no_other_observed_winning_vendors"
    return _optional(value, zero_spread_reason)


def _distribution_sha256(
    market_key_sha256: str,
    counts: Mapping[str, int],
) -> str:
    return _digest(
        {
            "market_key_sha256": market_key_sha256,
            "awards_by_winning_vendor": [
                (key, counts[key]) for key in sorted(counts)
            ],
        }
    )


def _validate_digest(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise VendorFrequencyFeatureError(f"{name} must be a SHA-256 hex digest")
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
