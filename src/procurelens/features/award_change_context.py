"""Award-change peer context resolution for ProcureLens.

Selects the first caller-supported formation-context market for one indexed award
and removes that award from its own peer population. The module preserves peer
membership, observable follow-up evidence, and metric coverage without computing
an anomaly score or statistical extremeness.

Only awards eligible for award_change_reference participate. No missing base
context, ambiguous multiple-base context, or unavailable metric is guessed.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from typing import Any

from procurelens.features.award_change_reference import (
    AwardChangeBaseReference,
    AwardChangeObservation,
    AwardChangeReferenceSnapshot,
)
from procurelens.features.peer_groups import PeerGroupKey


class AwardChangeContextError(ValueError):
    """Raised when award-change context evidence is inconsistent."""


@dataclass(frozen=True, slots=True)
class AwardChangeContextSupportSpec:
    """Caller-owned peer support requirement; not a risk threshold."""

    minimum_peer_awards: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_peer_awards, bool)
            or not isinstance(self.minimum_peer_awards, int)
            or self.minimum_peer_awards < 1
        ):
            raise AwardChangeContextError(
                "minimum_peer_awards must be a positive caller-supplied integer"
            )

    @property
    def sha256_hex(self) -> str:
        return _digest({"minimum_peer_awards": self.minimum_peer_awards})


@dataclass(frozen=True, slots=True)
class AwardChangeContextAttempt:
    level_name: str
    market_key_sha256: str | None
    market_award_count: int | None
    peer_award_count: int | None
    sufficient: bool
    unavailable_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        level = self.level_name.strip()
        if not level:
            raise AwardChangeContextError("attempt level_name must not be blank")
        object.__setattr__(self, "level_name", level)
        if self.market_key_sha256 is not None:
            object.__setattr__(
                self,
                "market_key_sha256",
                _validate_digest(self.market_key_sha256, "market_key_sha256"),
            )
        for name in ("market_award_count", "peer_award_count"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise AwardChangeContextError(
                    f"{name} must be a non-negative integer or None"
                )
        if (
            self.market_award_count is not None
            and self.peer_award_count is not None
            and self.peer_award_count > self.market_award_count
        ):
            raise AwardChangeContextError(
                "peer_award_count cannot exceed market_award_count"
            )
        reasons = tuple(
            reason.strip() for reason in self.unavailable_reasons if reason.strip()
        )
        if len(reasons) != len(set(reasons)):
            raise AwardChangeContextError("attempt reasons contain duplicates")
        object.__setattr__(self, "unavailable_reasons", reasons)


@dataclass(frozen=True, slots=True)
class AwardChangeContext:
    """Selected leave-one-award-out peer context."""

    award_id: str
    target: AwardChangeObservation
    base_reference: AwardChangeBaseReference
    market_key: PeerGroupKey
    plan_name: str
    plan_sha256: str
    reference_snapshot_sha256: str
    support_spec_sha256: str
    peers: tuple[AwardChangeObservation, ...]

    def __post_init__(self) -> None:
        award_id = self.award_id.strip()
        plan_name = self.plan_name.strip()
        if not award_id or not plan_name:
            raise AwardChangeContextError(
                "award_id and plan_name must not be blank"
            )
        object.__setattr__(self, "award_id", award_id)
        object.__setattr__(self, "plan_name", plan_name)
        if not isinstance(self.target, AwardChangeObservation):
            raise TypeError("target must be AwardChangeObservation")
        if not isinstance(self.base_reference, AwardChangeBaseReference):
            raise TypeError("base_reference must be AwardChangeBaseReference")
        if self.target.award_id != award_id:
            raise AwardChangeContextError(
                "target observation award_id differs from context"
            )
        if self.base_reference.award_id != award_id:
            raise AwardChangeContextError(
                "base reference award_id differs from context"
            )
        if self.target.base_transaction_id != self.base_reference.transaction_id:
            raise AwardChangeContextError(
                "target observation and base reference formation ids differ"
            )
        if not isinstance(self.market_key, PeerGroupKey):
            raise TypeError("market_key must be PeerGroupKey")
        for name in (
            "plan_sha256",
            "reference_snapshot_sha256",
            "support_spec_sha256",
        ):
            object.__setattr__(
                self, name, _validate_digest(getattr(self, name), name)
            )

        peers = tuple(self.peers)
        if not peers:
            raise AwardChangeContextError("selected context requires at least one peer")
        ids = tuple(peer.award_id for peer in peers)
        if award_id in ids:
            raise AwardChangeContextError(
                "target award must not appear in its own peer population"
            )
        if len(ids) != len(set(ids)):
            raise AwardChangeContextError("peer population contains duplicate awards")
        if tuple(sorted(ids)) != ids:
            raise AwardChangeContextError(
                "peer observations must be deterministically sorted by award_id"
            )
        object.__setattr__(self, "peers", peers)

    @property
    def market_level(self) -> str:
        return self.market_key.level_name

    @property
    def peer_award_count(self) -> int:
        return len(self.peers)

    @property
    def peers_with_base_obligation(self) -> int:
        return sum(
            peer.base_award_obligation_magnitude is not None
            for peer in self.peers
        )

    @property
    def base_obligation_coverage(self) -> Decimal:
        return Decimal(self.peers_with_base_obligation) / Decimal(
            self.peer_award_count
        )

    @property
    def peer_modification_prevalence(self) -> Decimal:
        return Decimal(sum(peer.has_modifications for peer in self.peers)) / Decimal(
            self.peer_award_count
        )

    @property
    def peer_deobligation_prevalence(self) -> Decimal:
        return Decimal(sum(peer.has_deobligation for peer in self.peers)) / Decimal(
            self.peer_award_count
        )

    @property
    def peer_positive_modification_prevalence(self) -> Decimal:
        return Decimal(
            sum(peer.has_positive_modification for peer in self.peers)
        ) / Decimal(self.peer_award_count)

    @property
    def minimum_peer_followup_days(self) -> int:
        return min(peer.observable_followup_days for peer in self.peers)

    @property
    def maximum_peer_followup_days(self) -> int:
        return max(peer.observable_followup_days for peer in self.peers)

    @property
    def evidence_sha256(self) -> str:
        return _digest(self.as_dict(include_evidence_sha=False))

    def as_dict(self, *, include_evidence_sha: bool = True) -> dict[str, Any]:
        result = {
            "award_id": self.award_id,
            "market_level": self.market_level,
            "market_key_sha256": self.market_key.sha256_hex,
            "plan_name": self.plan_name,
            "plan_sha256": self.plan_sha256,
            "reference_snapshot_sha256": self.reference_snapshot_sha256,
            "support_spec_sha256": self.support_spec_sha256,
            "target": self.target.as_dict(),
            "peer_award_count": self.peer_award_count,
            "peer_award_ids": [peer.award_id for peer in self.peers],
            "peers_with_base_obligation": self.peers_with_base_obligation,
            "base_obligation_coverage": str(self.base_obligation_coverage),
            "peer_modification_prevalence": str(
                self.peer_modification_prevalence
            ),
            "peer_deobligation_prevalence": str(
                self.peer_deobligation_prevalence
            ),
            "peer_positive_modification_prevalence": str(
                self.peer_positive_modification_prevalence
            ),
            "minimum_peer_followup_days": self.minimum_peer_followup_days,
            "maximum_peer_followup_days": self.maximum_peer_followup_days,
        }
        if include_evidence_sha:
            result["evidence_sha256"] = self.evidence_sha256
        return result


@dataclass(frozen=True, slots=True)
class AwardChangeContextResult:
    award_id: str
    support_spec: AwardChangeContextSupportSpec
    context: AwardChangeContext | None
    attempts: tuple[AwardChangeContextAttempt, ...]
    unavailable_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        award_id = self.award_id.strip()
        if not award_id:
            raise AwardChangeContextError("award_id must not be blank")
        object.__setattr__(self, "award_id", award_id)
        if not isinstance(self.support_spec, AwardChangeContextSupportSpec):
            raise TypeError("support_spec must be AwardChangeContextSupportSpec")
        if self.context is not None:
            if self.context.award_id != award_id:
                raise AwardChangeContextError(
                    "context award_id differs from result"
                )
            if self.unavailable_reasons:
                raise AwardChangeContextError(
                    "available context cannot carry unavailable reasons"
                )
        object.__setattr__(self, "attempts", tuple(self.attempts))
        reasons = tuple(
            reason.strip() for reason in self.unavailable_reasons if reason.strip()
        )
        if len(reasons) != len(set(reasons)):
            raise AwardChangeContextError("result reasons contain duplicates")
        object.__setattr__(self, "unavailable_reasons", reasons)

    @property
    def available(self) -> bool:
        return self.context is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "award_id": self.award_id,
            "available": self.available,
            "support_spec": {
                "minimum_peer_awards": self.support_spec.minimum_peer_awards,
                "sha256": self.support_spec.sha256_hex,
            },
            "unavailable_reasons": list(self.unavailable_reasons),
            "attempts": [
                {
                    "level_name": attempt.level_name,
                    "market_key_sha256": attempt.market_key_sha256,
                    "market_award_count": attempt.market_award_count,
                    "peer_award_count": attempt.peer_award_count,
                    "sufficient": attempt.sufficient,
                    "unavailable_reasons": list(attempt.unavailable_reasons),
                }
                for attempt in self.attempts
            ],
            "context": None if self.context is None else self.context.as_dict(),
        }


def resolve_award_change_context(
    award_id: str,
    snapshot: AwardChangeReferenceSnapshot,
    *,
    support_spec: AwardChangeContextSupportSpec,
) -> AwardChangeContextResult:
    """Resolve an indexed award into its first supported leave-one-out market."""

    if not isinstance(snapshot, AwardChangeReferenceSnapshot):
        raise TypeError("snapshot must be AwardChangeReferenceSnapshot")
    if not isinstance(support_spec, AwardChangeContextSupportSpec):
        raise TypeError("support_spec must be AwardChangeContextSupportSpec")
    award_id = award_id.strip()
    if not award_id:
        raise AwardChangeContextError("award_id must not be blank")

    target = snapshot.get_observation(award_id)
    base = snapshot.get_base_reference(award_id)
    if target is None:
        if base is not None:
            reason = "award_change_reference_ambiguous_multiple_base_actions"
        else:
            reason = "award_change_reference_missing_or_no_observed_base_action"
        return AwardChangeContextResult(
            award_id=award_id,
            support_spec=support_spec,
            context=None,
            attempts=(),
            unavailable_reasons=(reason,),
        )
    if base is None:
        raise AwardChangeContextError(
            "eligible target observation lacks base reference"
        )
    if target.base_transaction_id != base.transaction_id:
        raise AwardChangeContextError(
            "target observation differs from stored base reference"
        )

    attempts: list[AwardChangeContextAttempt] = []
    for candidate in base.candidates:
        if candidate.key is None:
            attempts.append(
                AwardChangeContextAttempt(
                    level_name=candidate.level_name,
                    market_key_sha256=None,
                    market_award_count=None,
                    peer_award_count=None,
                    sufficient=False,
                    unavailable_reasons=candidate.unavailable_reasons,
                )
            )
            continue

        market = snapshot.get_market(candidate.key)
        if market is None:
            attempts.append(
                AwardChangeContextAttempt(
                    level_name=candidate.level_name,
                    market_key_sha256=candidate.key.sha256_hex,
                    market_award_count=0,
                    peer_award_count=0,
                    sufficient=False,
                    unavailable_reasons=("market_not_observed",),
                )
            )
            continue
        if award_id not in market.award_ids:
            raise AwardChangeContextError(
                "eligible target award is absent from its formation market"
            )

        peer_ids = tuple(
            item for item in market.award_ids if item != award_id
        )
        sufficient = len(peer_ids) >= support_spec.minimum_peer_awards
        reasons = () if sufficient else ("insufficient_peer_awards",)
        attempts.append(
            AwardChangeContextAttempt(
                level_name=candidate.level_name,
                market_key_sha256=candidate.key.sha256_hex,
                market_award_count=market.award_count,
                peer_award_count=len(peer_ids),
                sufficient=sufficient,
                unavailable_reasons=reasons,
            )
        )
        if not sufficient:
            continue

        peers = tuple(
            snapshot.observations[peer_id]
            for peer_id in peer_ids
        )
        return AwardChangeContextResult(
            award_id=award_id,
            support_spec=support_spec,
            context=AwardChangeContext(
                award_id=award_id,
                target=target,
                base_reference=base,
                market_key=candidate.key,
                plan_name=snapshot.plan.name,
                plan_sha256=snapshot.plan.sha256_hex,
                reference_snapshot_sha256=snapshot.snapshot_sha256,
                support_spec_sha256=support_spec.sha256_hex,
                peers=peers,
            ),
            attempts=tuple(attempts),
        )

    return AwardChangeContextResult(
        award_id=award_id,
        support_spec=support_spec,
        context=None,
        attempts=tuple(attempts),
        unavailable_reasons=("no_supported_award_change_market",),
    )


def _validate_digest(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise AwardChangeContextError(f"{name} must be a SHA-256 hex digest")
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
