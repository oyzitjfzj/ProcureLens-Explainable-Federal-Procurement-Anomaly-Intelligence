"""Stable, explicit amount-feature candidates for ProcureLens.

This module converts descriptive AmountContext evidence into a fixed feature
contract for later anomaly detectors. It does not choose a detector, impute
missing values, scale features, assign anomaly thresholds, or compute risk.

Every optional measurement remains explicitly missing with a reason instead of
being silently replaced by zero. Later model specifications must choose which
feature names/families they use, so adding multiple descriptive measurements
here does not force a detector to consume redundant signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import json
from types import MappingProxyType
from collections.abc import Iterable, Mapping
from typing import Any

from procurelens.features.amount_context import (
    AmountContext,
    AmountContextResult,
    AmountDirection,
    DistributionContext,
)
from procurelens.features.amount_reference import AmountBasis


class AmountFeatureError(ValueError):
    """Raised when amount-feature evidence or selection is inconsistent."""


class AmountFeatureFamily(str, Enum):
    RANK = "rank"
    MAD_DISTANCE = "mad_distance"
    IQR_DISTANCE = "iqr_distance"
    DIRECTION = "direction"
    SUPPORT = "support"


class AmountFeatureName(str, Enum):
    SIGNED_POSITION = "amount_signed_position"
    MAGNITUDE_POSITION = "amount_magnitude_position"
    SAME_DIRECTION_MAGNITUDE_POSITION = "amount_same_direction_magnitude_position"
    SIGNED_MODIFIED_Z = "amount_signed_modified_z"
    MAGNITUDE_MODIFIED_Z = "amount_magnitude_modified_z"
    SAME_DIRECTION_MAGNITUDE_MODIFIED_Z = "amount_same_direction_magnitude_modified_z"
    SIGNED_IQR_DISTANCE = "amount_signed_iqr_distance"
    MAGNITUDE_IQR_DISTANCE = "amount_magnitude_iqr_distance"
    SAME_DIRECTION_MAGNITUDE_IQR_DISTANCE = "amount_same_direction_magnitude_iqr_distance"
    DIRECTION_CODE = "amount_direction_code"
    SAME_DIRECTION_PEER_FRACTION = "amount_same_direction_peer_fraction"


@dataclass(frozen=True, slots=True)
class AmountFeatureDefinition:
    name: AmountFeatureName
    family: AmountFeatureFamily
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", AmountFeatureName(self.name))
        object.__setattr__(self, "family", AmountFeatureFamily(self.family))
        description = self.description.strip()
        if not description:
            raise AmountFeatureError("feature description must not be blank")
        object.__setattr__(self, "description", description)


_DEFINITIONS = (
    AmountFeatureDefinition(AmountFeatureName.SIGNED_POSITION, AmountFeatureFamily.RANK,
        "Tie-aware midpoint empirical position of the signed target amount among peers."),
    AmountFeatureDefinition(AmountFeatureName.MAGNITUDE_POSITION, AmountFeatureFamily.RANK,
        "Tie-aware midpoint empirical position of absolute target magnitude among peers."),
    AmountFeatureDefinition(AmountFeatureName.SAME_DIRECTION_MAGNITUDE_POSITION, AmountFeatureFamily.RANK,
        "Magnitude position among peers with the same amount direction."),
    AmountFeatureDefinition(AmountFeatureName.SIGNED_MODIFIED_Z, AmountFeatureFamily.MAD_DISTANCE,
        "Signed median/MAD distance for the signed amount distribution."),
    AmountFeatureDefinition(AmountFeatureName.MAGNITUDE_MODIFIED_Z, AmountFeatureFamily.MAD_DISTANCE,
        "Median/MAD distance for absolute amount magnitude."),
    AmountFeatureDefinition(AmountFeatureName.SAME_DIRECTION_MAGNITUDE_MODIFIED_Z, AmountFeatureFamily.MAD_DISTANCE,
        "Median/MAD magnitude distance among same-direction peers."),
    AmountFeatureDefinition(AmountFeatureName.SIGNED_IQR_DISTANCE, AmountFeatureFamily.IQR_DISTANCE,
        "Signed median-centered distance measured in peer IQR units."),
    AmountFeatureDefinition(AmountFeatureName.MAGNITUDE_IQR_DISTANCE, AmountFeatureFamily.IQR_DISTANCE,
        "Median-centered absolute-magnitude distance measured in peer IQR units."),
    AmountFeatureDefinition(AmountFeatureName.SAME_DIRECTION_MAGNITUDE_IQR_DISTANCE, AmountFeatureFamily.IQR_DISTANCE,
        "Magnitude distance in IQR units among same-direction peers."),
    AmountFeatureDefinition(AmountFeatureName.DIRECTION_CODE, AmountFeatureFamily.DIRECTION,
        "Direction only: negative=-1, zero=0, positive=1."),
    AmountFeatureDefinition(AmountFeatureName.SAME_DIRECTION_PEER_FRACTION, AmountFeatureFamily.SUPPORT,
        "Fraction of peers sharing the target transaction's amount direction."),
)
_DEFINITION_BY_NAME: Mapping[AmountFeatureName, AmountFeatureDefinition] = MappingProxyType(
    {definition.name: definition for definition in _DEFINITIONS}
)
if len(_DEFINITION_BY_NAME) != len(_DEFINITIONS):
    raise RuntimeError("amount feature definitions contain duplicate names")


def amount_feature_definitions() -> tuple[AmountFeatureDefinition, ...]:
    return _DEFINITIONS


def amount_feature_definition_sha256() -> str:
    return _digest([
        {"name": d.name.value, "family": d.family.value, "description": d.description}
        for d in _DEFINITIONS
    ])


@dataclass(frozen=True, slots=True)
class AmountFeature:
    definition: AmountFeatureDefinition
    value: Decimal | None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        canonical = _DEFINITION_BY_NAME.get(AmountFeatureName(self.definition.name))
        if canonical != self.definition:
            raise AmountFeatureError("feature definition is not the canonical contract entry")
        if self.value is None:
            reason = None if self.unavailable_reason is None else self.unavailable_reason.strip()
            if not reason:
                raise AmountFeatureError(f"{self.definition.name.value}: missing feature requires a reason")
            object.__setattr__(self, "unavailable_reason", reason)
            return
        if not isinstance(self.value, Decimal) or not self.value.is_finite():
            raise AmountFeatureError(f"{self.definition.name.value}: value must be a finite Decimal or None")
        if self.unavailable_reason is not None:
            raise AmountFeatureError(f"{self.definition.name.value}: available feature cannot carry a missing reason")

    @property
    def name(self) -> AmountFeatureName:
        return self.definition.name

    @property
    def family(self) -> AmountFeatureFamily:
        return self.definition.family

    @property
    def available(self) -> bool:
        return self.value is not None

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name.value, "family": self.family.value,
                "value": None if self.value is None else str(self.value),
                "unavailable_reason": self.unavailable_reason}


@dataclass(frozen=True, slots=True)
class AmountFeatureSet:
    transaction_id: str
    amount_basis: AmountBasis
    selected_peer_level: str
    selected_peer_key_sha256: str
    reference_plan_sha256: str
    quantile_method: str
    peer_count: int
    features: tuple[AmountFeature, ...]

    def __post_init__(self) -> None:
        txid = self.transaction_id.strip()
        if not txid:
            raise AmountFeatureError("transaction_id must not be blank")
        object.__setattr__(self, "transaction_id", txid)
        object.__setattr__(self, "amount_basis", AmountBasis(self.amount_basis))
        for name in ("selected_peer_level", "quantile_method"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise AmountFeatureError(f"{name} must not be blank")
            object.__setattr__(self, name, value.strip())
        for name in ("selected_peer_key_sha256", "reference_plan_sha256"):
            digest = getattr(self, name).strip().lower()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise AmountFeatureError(f"{name} must be a SHA-256 hex digest")
            object.__setattr__(self, name, digest)
        if isinstance(self.peer_count, bool) or not isinstance(self.peer_count, int) or self.peer_count < 1:
            raise AmountFeatureError("peer_count must be a positive integer")
        if not isinstance(self.features, tuple):
            object.__setattr__(self, "features", tuple(self.features))
        observed_names = tuple(feature.name for feature in self.features)
        expected_names = tuple(definition.name for definition in _DEFINITIONS)
        if observed_names != expected_names:
            raise AmountFeatureError("feature set must contain every canonical amount feature exactly once in contract order")

    @property
    def definition_sha256(self) -> str:
        return amount_feature_definition_sha256()

    @property
    def evidence_sha256(self) -> str:
        return _digest({
            "transaction_id": self.transaction_id,
            "amount_basis": self.amount_basis.value,
            "selected_peer_level": self.selected_peer_level,
            "selected_peer_key_sha256": self.selected_peer_key_sha256,
            "reference_plan_sha256": self.reference_plan_sha256,
            "quantile_method": self.quantile_method,
            "peer_count": self.peer_count,
            "definition_sha256": self.definition_sha256,
            "features": [feature.as_dict() for feature in self.features],
        })

    def get(self, name: AmountFeatureName | str) -> AmountFeature:
        wanted = AmountFeatureName(name)
        for feature in self.features:
            if feature.name is wanted:
                return feature
        raise AmountFeatureError(f"unknown amount feature: {wanted.value}")

    def select(self, names: Iterable[AmountFeatureName | str], *, require_complete: bool = True) -> tuple[AmountFeature, ...]:
        if isinstance(names, (str, bytes)):
            raise AmountFeatureError("feature names must be an iterable of names")
        selected: list[AmountFeature] = []
        seen: set[AmountFeatureName] = set()
        for item in names:
            name = AmountFeatureName(item)
            if name in seen:
                raise AmountFeatureError(f"duplicate selected feature: {name.value}")
            seen.add(name)
            feature = self.get(name)
            if require_complete and not feature.available:
                raise AmountFeatureError(f"selected feature {name.value} is unavailable: {feature.unavailable_reason}")
            selected.append(feature)
        if not selected:
            raise AmountFeatureError("at least one feature must be selected")
        return tuple(selected)

    def select_families(self, families: Iterable[AmountFeatureFamily | str], *, require_complete: bool = False) -> tuple[AmountFeature, ...]:
        if isinstance(families, (str, bytes)):
            raise AmountFeatureError("families must be an iterable")
        wanted = {AmountFeatureFamily(item) for item in families}
        if not wanted:
            raise AmountFeatureError("at least one family must be selected")
        selected = tuple(feature for feature in self.features if feature.family in wanted)
        if require_complete:
            missing = [feature for feature in selected if not feature.available]
            if missing:
                names = ", ".join(feature.name.value for feature in missing)
                raise AmountFeatureError(f"selected feature families contain missing values: {names}")
        return selected

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "amount_basis": self.amount_basis.value,
            "selected_peer_level": self.selected_peer_level,
            "selected_peer_key_sha256": self.selected_peer_key_sha256,
            "reference_plan_sha256": self.reference_plan_sha256,
            "quantile_method": self.quantile_method,
            "peer_count": self.peer_count,
            "definition_sha256": self.definition_sha256,
            "evidence_sha256": self.evidence_sha256,
            "features": [feature.as_dict() for feature in self.features],
        }


@dataclass(frozen=True, slots=True)
class AmountFeatureResult:
    amount_basis: AmountBasis
    transaction_id: str | None
    feature_set: AmountFeatureSet | None
    unavailable_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "amount_basis", AmountBasis(self.amount_basis))
        if self.feature_set is not None and self.unavailable_reasons:
            raise AmountFeatureError("available amount feature set cannot also carry unavailable reasons")
        if self.feature_set is not None:
            if self.feature_set.amount_basis is not self.amount_basis:
                raise AmountFeatureError("result and feature-set amount bases differ")
            if self.transaction_id != self.feature_set.transaction_id:
                raise AmountFeatureError("result and feature-set transaction ids differ")
        elif self.transaction_id is not None:
            txid = self.transaction_id.strip()
            object.__setattr__(self, "transaction_id", txid or None)

    @property
    def available(self) -> bool:
        return self.feature_set is not None


def build_amount_features(context_result: AmountContextResult) -> AmountFeatureResult:
    if not isinstance(context_result, AmountContextResult):
        raise TypeError("context_result must be AmountContextResult")
    if not context_result.available or context_result.context is None:
        reasons = context_result.unavailable_reasons or ("amount_context_unavailable",)
        return AmountFeatureResult(context_result.amount_basis, None, None, tuple(reasons))
    context = context_result.context
    feature_set = AmountFeatureSet(
        transaction_id=context.transaction_id,
        amount_basis=context.amount_basis,
        selected_peer_level=context.selected_peer_level,
        selected_peer_key_sha256=context.selected_peer_key_sha256,
        reference_plan_sha256=context.reference_plan_sha256,
        quantile_method=context.quantile_method.value,
        peer_count=context.peer_count,
        features=_features_from_context(context),
    )
    return AmountFeatureResult(context.amount_basis, context.transaction_id, feature_set)


def _features_from_context(context: AmountContext) -> tuple[AmountFeature, ...]:
    if not isinstance(context, AmountContext):
        raise TypeError("context must be AmountContext")
    same = context.same_direction_magnitude
    same_fraction = Decimal(context.same_direction_peer_count) / Decimal(context.peer_count)
    values: dict[AmountFeatureName, tuple[Decimal | None, str | None]] = {
        AmountFeatureName.SIGNED_POSITION: (context.signed.position.midpoint_fraction, None),
        AmountFeatureName.MAGNITUDE_POSITION: (context.magnitude.position.midpoint_fraction, None),
        AmountFeatureName.SAME_DIRECTION_MAGNITUDE_POSITION: _same_direction_value(
            same, lambda item: item.position.midpoint_fraction, "no_same_direction_peers"),
        AmountFeatureName.SIGNED_MODIFIED_Z: _optional_distance(context.signed.modified_z, "peer_mad_zero"),
        AmountFeatureName.MAGNITUDE_MODIFIED_Z: _optional_distance(context.magnitude.modified_z, "peer_magnitude_mad_zero"),
        AmountFeatureName.SAME_DIRECTION_MAGNITUDE_MODIFIED_Z: _same_direction_distance(same, kind="mad"),
        AmountFeatureName.SIGNED_IQR_DISTANCE: _optional_distance(context.signed.iqr_distance, "peer_iqr_zero"),
        AmountFeatureName.MAGNITUDE_IQR_DISTANCE: _optional_distance(context.magnitude.iqr_distance, "peer_magnitude_iqr_zero"),
        AmountFeatureName.SAME_DIRECTION_MAGNITUDE_IQR_DISTANCE: _same_direction_distance(same, kind="iqr"),
        AmountFeatureName.DIRECTION_CODE: (_direction_code(context.target_direction), None),
        AmountFeatureName.SAME_DIRECTION_PEER_FRACTION: (same_fraction, None),
    }
    return tuple(AmountFeature(definition, values[definition.name][0], values[definition.name][1]) for definition in _DEFINITIONS)


def _optional_distance(value: Decimal | None, zero_spread_reason: str) -> tuple[Decimal | None, str | None]:
    return (None, zero_spread_reason) if value is None else (value, None)


def _same_direction_value(context: DistributionContext | None, extractor: Any, missing_reason: str) -> tuple[Decimal | None, str | None]:
    if context is None:
        return None, missing_reason
    value = extractor(context)
    if not isinstance(value, Decimal) or not value.is_finite():
        raise AmountFeatureError("same-direction extractor returned invalid value")
    return value, None


def _same_direction_distance(context: DistributionContext | None, *, kind: str) -> tuple[Decimal | None, str | None]:
    if context is None:
        return None, "no_same_direction_peers"
    if kind == "mad":
        return _optional_distance(context.modified_z, "same_direction_peer_mad_zero")
    if kind == "iqr":
        return _optional_distance(context.iqr_distance, "same_direction_peer_iqr_zero")
    raise AmountFeatureError(f"unsupported same-direction distance kind: {kind!r}")


def _direction_code(direction: AmountDirection) -> Decimal:
    direction = AmountDirection(direction)
    if direction is AmountDirection.NEGATIVE:
        return Decimal(-1)
    if direction is AmountDirection.ZERO:
        return Decimal(0)
    return Decimal(1)


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
