"""Target-vendor context inside observed new-award markets.

Resolves a procurement transaction into the first caller-approved market with
enough evidence, then reports the target vendor's observed new-award frequency
and shares. Market support rules are supplied by the caller; this module hides
no minimum sample size, identity-coverage cutoff, or anomaly threshold.

When the target is an indexed BASE_AWARD action, its exact stored market/vendor
fingerprint must match before leave-one-out subtraction is allowed. This avoids
letting an award influence the reference market used to evaluate itself.

Award data exposes observed winners, not every firm that could have competed,
so all outputs use observed-winner language and do not claim bidder-market
coverage.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import json
from typing import Any

from procurelens.domain.transaction import ProcurementTransaction
from procurelens.features.award_lifecycle import AwardActionKind, classify_award_action
from procurelens.features.peer_groups import (
    PeerGroupCandidate,
    PeerGroupKey,
    peer_group_candidates,
)
from procurelens.features.vendor_identity import (
    VendorIdentity,
    VendorIdentityScope,
    resolve_vendor_identity,
)
from procurelens.features.vendor_market import (
    ObservedVendorMarket,
    VendorMarketSnapshot,
)


class VendorMarketContextError(ValueError):
    """Raised when target-market evidence or configuration is inconsistent."""


class TargetReferenceMode(str, Enum):
    """How the target relates to the market evidence used for comparison."""

    LEAVE_ONE_OUT = "leave_one_out"
    NOT_INDEXED = "not_indexed"
    NOT_BASE_AWARD = "not_base_award"


@dataclass(frozen=True, slots=True)
class VendorMarketSupportSpec:
    """Caller-owned evidence requirements for selecting a comparison market."""

    minimum_observed_new_awards: int
    minimum_identified_new_awards: int
    minimum_observed_winning_vendors: int
    minimum_vendor_identity_coverage: Decimal

    def __post_init__(self) -> None:
        for name in (
            "minimum_observed_new_awards",
            "minimum_identified_new_awards",
            "minimum_observed_winning_vendors",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise VendorMarketContextError(f"{name} must be a positive integer")

        coverage = self.minimum_vendor_identity_coverage
        if not isinstance(coverage, Decimal) or not coverage.is_finite():
            raise VendorMarketContextError(
                "minimum_vendor_identity_coverage must be a finite Decimal"
            )
        if coverage < Decimal(0) or coverage > Decimal(1):
            raise VendorMarketContextError(
                "minimum_vendor_identity_coverage must be between 0 and 1"
            )

    @property
    def sha256_hex(self) -> str:
        return _digest(
            {
                "minimum_observed_new_awards": self.minimum_observed_new_awards,
                "minimum_identified_new_awards": self.minimum_identified_new_awards,
                "minimum_observed_winning_vendors": self.minimum_observed_winning_vendors,
                "minimum_vendor_identity_coverage": str(
                    self.minimum_vendor_identity_coverage
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class VendorMarketAttempt:
    """One market level considered for the target transaction."""

    level_name: str
    market_key_sha256: str | None
    reference_mode: TargetReferenceMode
    observed_new_awards: int | None
    identified_new_awards: int | None
    observed_winning_vendors: int | None
    vendor_identity_coverage: Decimal | None
    sufficient: bool
    unavailable_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        level = self.level_name.strip()
        if not level:
            raise VendorMarketContextError("market attempt level_name must not be blank")
        object.__setattr__(self, "level_name", level)
        object.__setattr__(self, "reference_mode", TargetReferenceMode(self.reference_mode))

        if self.market_key_sha256 is not None:
            object.__setattr__(
                self,
                "market_key_sha256",
                _validate_digest(self.market_key_sha256, "market_key_sha256"),
            )

        counts = (
            self.observed_new_awards,
            self.identified_new_awards,
            self.observed_winning_vendors,
        )
        if any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            )
            for value in counts
        ):
            raise VendorMarketContextError(
                "market attempt counts must be non-negative integers or None"
            )
        if (
            self.observed_new_awards is not None
            and self.identified_new_awards is not None
            and self.identified_new_awards > self.observed_new_awards
        ):
            raise VendorMarketContextError(
                "identified new awards cannot exceed observed new awards"
            )

        coverage = self.vendor_identity_coverage
        if coverage is not None:
            if not isinstance(coverage, Decimal) or not coverage.is_finite():
                raise VendorMarketContextError(
                    "vendor_identity_coverage must be finite Decimal or None"
                )
            if coverage < Decimal(0) or coverage > Decimal(1):
                raise VendorMarketContextError(
                    "vendor_identity_coverage must be between 0 and 1"
                )

        object.__setattr__(
            self,
            "unavailable_reasons",
            tuple(reason.strip() for reason in self.unavailable_reasons if reason.strip()),
        )


@dataclass(frozen=True, slots=True)
class VendorMarketContext:
    """Selected leave-one-out or external reference market for one target vendor."""

    transaction_id: str
    award_id: str
    target_identity: VendorIdentity
    target_lifecycle_kind: AwardActionKind
    reference_mode: TargetReferenceMode
    market_key: PeerGroupKey
    plan_name: str
    plan_sha256: str
    support_spec_sha256: str

    observed_new_award_count: int
    identified_new_award_count: int
    unidentified_new_award_count: int
    observed_winning_vendor_count: int
    vendor_identity_coverage: Decimal

    target_vendor_new_award_count: int
    target_share_of_identified_new_awards: Decimal
    target_share_of_all_observed_new_awards: Decimal

    award_count_hhi: Decimal
    largest_winner_award_share: Decimal

    def __post_init__(self) -> None:
        for name in ("transaction_id", "award_id", "plan_name"):
            value = getattr(self, name).strip()
            if not value:
                raise VendorMarketContextError(f"{name} must not be blank")
            object.__setattr__(self, name, value)

        if not isinstance(self.target_identity, VendorIdentity):
            raise TypeError("target_identity must be VendorIdentity")
        object.__setattr__(
            self, "target_lifecycle_kind", AwardActionKind(self.target_lifecycle_kind)
        )
        object.__setattr__(self, "reference_mode", TargetReferenceMode(self.reference_mode))
        if not isinstance(self.market_key, PeerGroupKey):
            raise TypeError("market_key must be PeerGroupKey")

        for name in ("plan_sha256", "support_spec_sha256"):
            object.__setattr__(
                self, name, _validate_digest(getattr(self, name), name)
            )

        counts = (
            self.observed_new_award_count,
            self.identified_new_award_count,
            self.unidentified_new_award_count,
            self.observed_winning_vendor_count,
            self.target_vendor_new_award_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise VendorMarketContextError(
                "vendor-market context counts must be non-negative integers"
            )
        if (
            self.identified_new_award_count + self.unidentified_new_award_count
            != self.observed_new_award_count
        ):
            raise VendorMarketContextError(
                "identified and unidentified awards must sum to observed awards"
            )
        if self.target_vendor_new_award_count > self.identified_new_award_count:
            raise VendorMarketContextError(
                "target vendor awards cannot exceed identified awards"
            )

        for name in (
            "vendor_identity_coverage",
            "target_share_of_identified_new_awards",
            "target_share_of_all_observed_new_awards",
            "award_count_hhi",
            "largest_winner_award_share",
        ):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite():
                raise VendorMarketContextError(f"{name} must be a finite Decimal")
            if value < Decimal(0) or value > Decimal(1):
                raise VendorMarketContextError(f"{name} must be between 0 and 1")

    @property
    def market_level(self) -> str:
        return self.market_key.level_name

    @property
    def evidence_sha256(self) -> str:
        return _digest(self.as_dict(include_evidence_sha=False))

    def as_dict(self, *, include_evidence_sha: bool = True) -> dict[str, Any]:
        result = {
            "transaction_id": self.transaction_id,
            "award_id": self.award_id,
            "target_identity": self.target_identity.as_dict(),
            "target_lifecycle_kind": self.target_lifecycle_kind.value,
            "reference_mode": self.reference_mode.value,
            "market_level": self.market_level,
            "market_key_sha256": self.market_key.sha256_hex,
            "plan_name": self.plan_name,
            "plan_sha256": self.plan_sha256,
            "support_spec_sha256": self.support_spec_sha256,
            "observed_new_award_count": self.observed_new_award_count,
            "identified_new_award_count": self.identified_new_award_count,
            "unidentified_new_award_count": self.unidentified_new_award_count,
            "observed_winning_vendor_count": self.observed_winning_vendor_count,
            "vendor_identity_coverage": str(self.vendor_identity_coverage),
            "target_vendor_new_award_count": self.target_vendor_new_award_count,
            "target_share_of_identified_new_awards": str(
                self.target_share_of_identified_new_awards
            ),
            "target_share_of_all_observed_new_awards": str(
                self.target_share_of_all_observed_new_awards
            ),
            "award_count_hhi": str(self.award_count_hhi),
            "largest_winner_award_share": str(self.largest_winner_award_share),
        }
        if include_evidence_sha:
            result["evidence_sha256"] = self.evidence_sha256
        return result


@dataclass(frozen=True, slots=True)
class VendorMarketContextResult:
    """Resolution report that preserves failed/broader market attempts."""

    transaction_id: str
    scope: VendorIdentityScope
    support_spec: VendorMarketSupportSpec
    target_identity: VendorIdentity | None
    context: VendorMarketContext | None
    attempts: tuple[VendorMarketAttempt, ...]
    unavailable_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        txid = self.transaction_id.strip()
        if not txid:
            raise VendorMarketContextError("transaction_id must not be blank")
        object.__setattr__(self, "transaction_id", txid)
        object.__setattr__(self, "scope", VendorIdentityScope(self.scope))
        if not isinstance(self.support_spec, VendorMarketSupportSpec):
            raise TypeError("support_spec must be VendorMarketSupportSpec")
        if self.target_identity is not None:
            if self.target_identity.scope is not self.scope:
                raise VendorMarketContextError(
                    "target identity scope differs from result scope"
                )
        if self.context is not None:
            if self.target_identity is None:
                raise VendorMarketContextError(
                    "available context requires target vendor identity"
                )
            if self.context.transaction_id != self.transaction_id:
                raise VendorMarketContextError(
                    "result and context transaction ids differ"
                )
            if self.context.target_identity != self.target_identity:
                raise VendorMarketContextError(
                    "result and context target identities differ"
                )
            if self.unavailable_reasons:
                raise VendorMarketContextError(
                    "available context cannot also carry unavailable reasons"
                )
        object.__setattr__(
            self,
            "attempts",
            tuple(self.attempts),
        )
        object.__setattr__(
            self,
            "unavailable_reasons",
            tuple(reason.strip() for reason in self.unavailable_reasons if reason.strip()),
        )

    @property
    def available(self) -> bool:
        return self.context is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "scope": self.scope.value,
            "support_spec": {
                "minimum_observed_new_awards": self.support_spec.minimum_observed_new_awards,
                "minimum_identified_new_awards": self.support_spec.minimum_identified_new_awards,
                "minimum_observed_winning_vendors": self.support_spec.minimum_observed_winning_vendors,
                "minimum_vendor_identity_coverage": str(
                    self.support_spec.minimum_vendor_identity_coverage
                ),
                "sha256": self.support_spec.sha256_hex,
            },
            "target_identity": (
                None if self.target_identity is None else self.target_identity.as_dict()
            ),
            "available": self.available,
            "unavailable_reasons": list(self.unavailable_reasons),
            "attempts": [
                {
                    "level_name": attempt.level_name,
                    "market_key_sha256": attempt.market_key_sha256,
                    "reference_mode": attempt.reference_mode.value,
                    "observed_new_awards": attempt.observed_new_awards,
                    "identified_new_awards": attempt.identified_new_awards,
                    "observed_winning_vendors": attempt.observed_winning_vendors,
                    "vendor_identity_coverage": (
                        None
                        if attempt.vendor_identity_coverage is None
                        else str(attempt.vendor_identity_coverage)
                    ),
                    "sufficient": attempt.sufficient,
                    "unavailable_reasons": list(attempt.unavailable_reasons),
                }
                for attempt in self.attempts
            ],
            "context": None if self.context is None else self.context.as_dict(),
        }


def resolve_vendor_market_context(
    transaction: ProcurementTransaction,
    snapshot: VendorMarketSnapshot,
    *,
    support_spec: VendorMarketSupportSpec,
) -> VendorMarketContextResult:
    """Resolve target vendor against the first market satisfying caller rules."""

    if not isinstance(transaction, ProcurementTransaction):
        raise TypeError("transaction must be ProcurementTransaction")
    if not isinstance(snapshot, VendorMarketSnapshot):
        raise TypeError("snapshot must be VendorMarketSnapshot")
    if not isinstance(support_spec, VendorMarketSupportSpec):
        raise TypeError("support_spec must be VendorMarketSupportSpec")

    identity = resolve_vendor_identity(transaction, snapshot.scope)
    candidates = peer_group_candidates(transaction, snapshot.plan)
    lifecycle = classify_award_action(transaction)

    if identity is None:
        return VendorMarketContextResult(
            transaction_id=transaction.transaction_id,
            scope=snapshot.scope,
            support_spec=support_spec,
            target_identity=None,
            context=None,
            attempts=tuple(
                VendorMarketAttempt(
                    level_name=candidate.level_name,
                    market_key_sha256=(
                        None if candidate.key is None else candidate.key.sha256_hex
                    ),
                    reference_mode=_reference_mode(
                        transaction, snapshot, lifecycle.kind
                    ),
                    observed_new_awards=None,
                    identified_new_awards=None,
                    observed_winning_vendors=None,
                    vendor_identity_coverage=None,
                    sufficient=False,
                    unavailable_reasons=(
                        candidate.unavailable_reasons
                        + ("target_vendor_identity_missing",)
                    ),
                )
                for candidate in candidates
            ),
            unavailable_reasons=("target_vendor_identity_missing",),
        )

    mode = _reference_mode(transaction, snapshot, lifecycle.kind)
    if mode is TargetReferenceMode.LEAVE_ONE_OUT:
        expected = snapshot.base_award_fingerprints[transaction.award_id]
        actual = _target_base_award_fingerprint(
            transaction,
            lifecycle.normalized_modification_number,
            candidates,
            identity,
            snapshot.scope,
        )
        if actual != expected:
            raise VendorMarketContextError(
                "indexed target base award differs from stored vendor-market evidence"
            )

    attempts: list[VendorMarketAttempt] = []
    for candidate in candidates:
        if candidate.key is None:
            attempts.append(
                VendorMarketAttempt(
                    level_name=candidate.level_name,
                    market_key_sha256=None,
                    reference_mode=mode,
                    observed_new_awards=None,
                    identified_new_awards=None,
                    observed_winning_vendors=None,
                    vendor_identity_coverage=None,
                    sufficient=False,
                    unavailable_reasons=candidate.unavailable_reasons,
                )
            )
            continue

        market = snapshot.get(candidate.key)
        if market is None:
            attempts.append(
                VendorMarketAttempt(
                    level_name=candidate.level_name,
                    market_key_sha256=candidate.key.sha256_hex,
                    reference_mode=mode,
                    observed_new_awards=0,
                    identified_new_awards=0,
                    observed_winning_vendors=0,
                    vendor_identity_coverage=None,
                    sufficient=False,
                    unavailable_reasons=("market_not_observed",),
                )
            )
            continue

        reference = (
            market.without_one_award(identity.key)
            if mode is TargetReferenceMode.LEAVE_ONE_OUT
            else market
        )
        reasons = _support_failures(reference, support_spec)
        sufficient = not reasons
        attempts.append(
            VendorMarketAttempt(
                level_name=candidate.level_name,
                market_key_sha256=candidate.key.sha256_hex,
                reference_mode=mode,
                observed_new_awards=reference.observed_new_award_count,
                identified_new_awards=reference.new_awards_with_vendor_identity,
                observed_winning_vendors=reference.observed_winning_vendor_count,
                vendor_identity_coverage=reference.vendor_identity_coverage,
                sufficient=sufficient,
                unavailable_reasons=tuple(reasons),
            )
        )
        if not sufficient:
            continue

        context = _build_context(
            transaction,
            snapshot,
            support_spec,
            identity,
            lifecycle.kind,
            mode,
            reference,
        )
        return VendorMarketContextResult(
            transaction_id=transaction.transaction_id,
            scope=snapshot.scope,
            support_spec=support_spec,
            target_identity=identity,
            context=context,
            attempts=tuple(attempts),
        )

    return VendorMarketContextResult(
        transaction_id=transaction.transaction_id,
        scope=snapshot.scope,
        support_spec=support_spec,
        target_identity=identity,
        context=None,
        attempts=tuple(attempts),
        unavailable_reasons=("no_market_satisfies_support_requirements",),
    )


def _build_context(
    transaction: ProcurementTransaction,
    snapshot: VendorMarketSnapshot,
    support_spec: VendorMarketSupportSpec,
    identity: VendorIdentity,
    lifecycle_kind: AwardActionKind,
    mode: TargetReferenceMode,
    reference: ObservedVendorMarket,
) -> VendorMarketContext:
    identified = reference.new_awards_with_vendor_identity
    total = reference.observed_new_award_count
    if identified < 1 or total < 1:
        raise VendorMarketContextError(
            "selected reference market must contain observed and identified awards"
        )
    coverage = reference.vendor_identity_coverage
    hhi = reference.award_count_hhi
    largest = reference.largest_winner_award_share
    if coverage is None or hhi is None or largest is None:
        raise VendorMarketContextError(
            "selected reference market is missing required market structure"
        )

    vendor_count = reference.awards_by_winning_vendor.get(identity.key, 0)
    return VendorMarketContext(
        transaction_id=transaction.transaction_id,
        award_id=transaction.award_id,
        target_identity=identity,
        target_lifecycle_kind=lifecycle_kind,
        reference_mode=mode,
        market_key=reference.key,
        plan_name=snapshot.plan.name,
        plan_sha256=snapshot.plan.sha256_hex,
        support_spec_sha256=support_spec.sha256_hex,
        observed_new_award_count=total,
        identified_new_award_count=identified,
        unidentified_new_award_count=reference.new_awards_without_vendor_identity,
        observed_winning_vendor_count=reference.observed_winning_vendor_count,
        vendor_identity_coverage=coverage,
        target_vendor_new_award_count=vendor_count,
        target_share_of_identified_new_awards=Decimal(vendor_count)
        / Decimal(identified),
        target_share_of_all_observed_new_awards=Decimal(vendor_count)
        / Decimal(total),
        award_count_hhi=hhi,
        largest_winner_award_share=largest,
    )


def _support_failures(
    market: ObservedVendorMarket,
    spec: VendorMarketSupportSpec,
) -> list[str]:
    reasons: list[str] = []
    if market.observed_new_award_count < spec.minimum_observed_new_awards:
        reasons.append("insufficient_observed_new_awards")
    if market.new_awards_with_vendor_identity < spec.minimum_identified_new_awards:
        reasons.append("insufficient_identified_new_awards")
    if market.observed_winning_vendor_count < spec.minimum_observed_winning_vendors:
        reasons.append("insufficient_observed_winning_vendors")

    coverage = market.vendor_identity_coverage
    if coverage is None or coverage < spec.minimum_vendor_identity_coverage:
        reasons.append("insufficient_vendor_identity_coverage")
    return reasons


def _reference_mode(
    transaction: ProcurementTransaction,
    snapshot: VendorMarketSnapshot,
    lifecycle_kind: AwardActionKind,
) -> TargetReferenceMode:
    if lifecycle_kind is not AwardActionKind.BASE_AWARD:
        return TargetReferenceMode.NOT_BASE_AWARD
    if transaction.award_id in snapshot.base_award_fingerprints:
        return TargetReferenceMode.LEAVE_ONE_OUT
    return TargetReferenceMode.NOT_INDEXED


def _target_base_award_fingerprint(
    transaction: ProcurementTransaction,
    normalized_modification_number: str | None,
    candidates: tuple[PeerGroupCandidate, ...],
    identity: VendorIdentity | None,
    scope: VendorIdentityScope,
) -> str:
    """Mirror the vendor-market snapshot's stored base-award evidence contract."""

    return _digest(
        {
            "award_id": transaction.award_id,
            "transaction_id": transaction.transaction_id,
            "action_date": transaction.action_date.isoformat(),
            "modification_number": normalized_modification_number,
            "scope": scope.value,
            "vendor_key": None if identity is None else identity.key,
            "candidates": _candidate_payload(candidates),
        }
    )


def _candidate_payload(
    candidates: tuple[PeerGroupCandidate, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "level": candidate.level_name,
            "key": None if candidate.key is None else candidate.key.sha256_hex,
            "unavailable_reasons": list(candidate.unavailable_reasons),
        }
        for candidate in candidates
    ]


def _validate_digest(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise VendorMarketContextError(f"{name} must be a SHA-256 hex digest")
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
