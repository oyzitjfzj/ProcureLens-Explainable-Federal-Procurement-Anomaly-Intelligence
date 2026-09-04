"""Explicit feature-construction plans for ProcureLens.

Pins the reference-population policy used to construct every candidate evidence
family before model preparation. No peer-support threshold, identity scope,
amount basis, peer hierarchy, quantile method, or memory budget is inferred by
the feature orchestrator.

Reference snapshots are built from a caller-supplied reference population and
then frozen. Target/scoring transactions are resolved against those snapshots
without mutating them.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Mapping

from procurelens.features.amount_reference import (
    AmountBasis,
    AmountReferencePolicy,
)
from procurelens.features.award_change_context import (
    AwardChangeContextSupportSpec,
)
from procurelens.features.award_change_reference import (
    AwardChangeReferencePolicy,
)
from procurelens.features.competition_context import (
    CompetitionContextSupportSpec,
)
from procurelens.features.competition_reference import (
    CompetitionReferencePolicy,
)
from procurelens.features.peer_groups import PeerGroupPlan
from procurelens.features.vendor_identity import VendorIdentityScope
from procurelens.features.vendor_market import VendorMarketPolicy
from procurelens.features.vendor_market_context import VendorMarketSupportSpec
from procurelens.statistics.robust import QuantileMethod


class FeaturePipelineConfigError(ValueError):
    """Raised when feature-construction policy is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class FeatureBuildPlan:
    """Complete explicit policy for frozen-reference candidate feature creation."""

    name: str
    description: str
    feature_catalog_sha256: str

    amount_peer_plan: PeerGroupPlan
    amount_basis: AmountBasis
    amount_minimum_peer_count: int
    amount_reference_policy: AmountReferencePolicy

    vendor_peer_plan: PeerGroupPlan
    vendor_scope: VendorIdentityScope
    vendor_support: VendorMarketSupportSpec
    vendor_market_policy: VendorMarketPolicy

    competition_peer_plan: PeerGroupPlan
    competition_support: CompetitionContextSupportSpec
    competition_reference_policy: CompetitionReferencePolicy

    award_change_peer_plan: PeerGroupPlan
    award_change_support: AwardChangeContextSupportSpec
    award_change_reference_policy: AwardChangeReferencePolicy

    quantile_method: QuantileMethod

    def __post_init__(self) -> None:
        for field_name in ("name", "description"):
            text = getattr(self, field_name).strip()
            if not text:
                raise FeaturePipelineConfigError(
                    f"{field_name} must not be blank"
                )
            object.__setattr__(self, field_name, text)

        object.__setattr__(
            self,
            "feature_catalog_sha256",
            _digest_hex(
                self.feature_catalog_sha256,
                "feature_catalog_sha256",
            ),
        )

        for field_name in (
            "amount_peer_plan",
            "vendor_peer_plan",
            "competition_peer_plan",
            "award_change_peer_plan",
        ):
            if not isinstance(getattr(self, field_name), PeerGroupPlan):
                raise TypeError(f"{field_name} must be PeerGroupPlan")

        object.__setattr__(self, "amount_basis", AmountBasis(self.amount_basis))
        object.__setattr__(self, "vendor_scope", VendorIdentityScope(self.vendor_scope))
        object.__setattr__(
            self, "quantile_method", QuantileMethod(self.quantile_method)
        )

        if (
            isinstance(self.amount_minimum_peer_count, bool)
            or not isinstance(self.amount_minimum_peer_count, int)
            or self.amount_minimum_peer_count < 1
        ):
            raise FeaturePipelineConfigError(
                "amount_minimum_peer_count must be a positive integer"
            )

        expected_types = (
            ("amount_reference_policy", AmountReferencePolicy),
            ("vendor_support", VendorMarketSupportSpec),
            ("vendor_market_policy", VendorMarketPolicy),
            ("competition_support", CompetitionContextSupportSpec),
            ("competition_reference_policy", CompetitionReferencePolicy),
            ("award_change_support", AwardChangeContextSupportSpec),
            ("award_change_reference_policy", AwardChangeReferencePolicy),
        )
        for field_name, expected in expected_types:
            if not isinstance(getattr(self, field_name), expected):
                raise TypeError(
                    f"{field_name} must be {expected.__name__}"
                )

    @property
    def sha256_hex(self) -> str:
        return _digest(self.as_dict(include_sha=False))

    def as_dict(self, *, include_sha: bool = True) -> dict[str, Any]:
        result = {
            "name": self.name,
            "description": self.description,
            "feature_catalog_sha256": self.feature_catalog_sha256,
            "amount": {
                "peer_plan_sha256": self.amount_peer_plan.sha256_hex,
                "amount_basis": self.amount_basis.value,
                "minimum_peer_count": self.amount_minimum_peer_count,
                "reference_policy": _canonical(self.amount_reference_policy),
            },
            "vendor": {
                "peer_plan_sha256": self.vendor_peer_plan.sha256_hex,
                "scope": self.vendor_scope.value,
                "support_sha256": self.vendor_support.sha256_hex,
                "market_policy": _canonical(self.vendor_market_policy),
            },
            "competition": {
                "peer_plan_sha256": self.competition_peer_plan.sha256_hex,
                "support_sha256": self.competition_support.sha256_hex,
                "reference_policy": _canonical(
                    self.competition_reference_policy
                ),
            },
            "award_change": {
                "peer_plan_sha256": self.award_change_peer_plan.sha256_hex,
                "support_sha256": self.award_change_support.sha256_hex,
                "reference_policy": _canonical(
                    self.award_change_reference_policy
                ),
            },
            "quantile_method": self.quantile_method.value,
        }
        if include_sha:
            result["sha256"] = self.sha256_hex
        return result


def _canonical(value: Any) -> Any:
    """Convert policy dataclasses into stable JSON-compatible evidence."""

    if isinstance(value, Decimal):
        if not value.is_finite():
            raise FeaturePipelineConfigError(
                "policy Decimal values must be finite"
            )
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            item.name: _canonical(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(
                value.items(), key=lambda pair: str(pair[0])
            )
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise FeaturePipelineConfigError(
        f"unsupported policy value in fingerprint: {type(value).__name__}"
    )


def _digest_hex(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise FeaturePipelineConfigError(f"{name} must be a string")
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise FeaturePipelineConfigError(
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
