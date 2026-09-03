"""Contextual competition evidence for ProcureLens.

Resolves one target award against award-level competition reference markets.
The reference population is built from observed BASE_AWARD actions only, so
later modifications cannot inflate prevalence.

For an indexed award, ProcureLens uses the stored base-award formation context
and removes that award from the selected reference market before comparison.
A modification therefore inherits the base award's original peer market rather
than being reclassified by the modification action date. Lifecycle-unknown
transactions are only resolvable when an indexed base award is already known.

All support requirements are caller supplied. This module reports prevalence,
coverage, provenance, and target facts; it does not assign anomaly thresholds,
risk scores, or misconduct conclusions.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import json
from typing import Any

from procurelens.domain.transaction import ProcurementTransaction
from procurelens.features.award_lifecycle import (
    AwardActionKind,
    classify_award_action,
)
from procurelens.features.competition_evidence import (
    CompetitionEvidence,
    OfferOutcomeKind,
    SolicitationProcedureKind,
    build_competition_evidence,
)
from procurelens.features.competition_reference import (
    CompetitionBaseAwardReference,
    CompetitionMarketReference,
    CompetitionReferenceSnapshot,
    base_award_fingerprint,
)
from procurelens.features.peer_groups import (
    PeerGroupCandidate,
    PeerGroupKey,
    peer_group_candidates,
)


class CompetitionContextError(ValueError):
    """Raised when competition context evidence is inconsistent."""


class CompetitionContextMode(str, Enum):
    """How target award-formation evidence is related to the reference population."""

    INDEXED_BASE_AWARD = "indexed_base_award_leave_one_out"
    INDEXED_AWARD = "indexed_award_uses_base_award_leave_one_out"
    EXTERNAL_BASE_AWARD = "external_base_award_not_indexed"


@dataclass(frozen=True, slots=True)
class CompetitionContextSupportSpec:
    """Caller-owned evidence requirements for selecting a competition market."""

    minimum_base_awards: int
    minimum_process_known: int
    minimum_offers_known: int
    minimum_procedure_known: int
    minimum_process_coverage: Decimal
    minimum_offer_coverage: Decimal
    minimum_procedure_coverage: Decimal

    def __post_init__(self) -> None:
        for name in (
            "minimum_base_awards",
            "minimum_process_known",
            "minimum_offers_known",
            "minimum_procedure_known",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise CompetitionContextError(
                    f"{name} must be a non-negative integer"
                )
        if self.minimum_base_awards < 1:
            raise CompetitionContextError(
                "minimum_base_awards must be at least 1"
            )
        for name in (
            "minimum_process_coverage",
            "minimum_offer_coverage",
            "minimum_procedure_coverage",
        ):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite():
                raise CompetitionContextError(
                    f"{name} must be a finite Decimal"
                )
            if value < Decimal(0) or value > Decimal(1):
                raise CompetitionContextError(
                    f"{name} must be between 0 and 1"
                )

    @property
    def sha256_hex(self) -> str:
        return _digest(
            {
                "minimum_base_awards": self.minimum_base_awards,
                "minimum_process_known": self.minimum_process_known,
                "minimum_offers_known": self.minimum_offers_known,
                "minimum_procedure_known": self.minimum_procedure_known,
                "minimum_process_coverage": str(self.minimum_process_coverage),
                "minimum_offer_coverage": str(self.minimum_offer_coverage),
                "minimum_procedure_coverage": str(self.minimum_procedure_coverage),
            }
        )


@dataclass(frozen=True, slots=True)
class CompetitionContextAttempt:
    """One peer-market level considered for the target award."""

    level_name: str
    market_key_sha256: str | None
    reference_mode: CompetitionContextMode | None
    base_award_count: int | None
    process_known_count: int | None
    offers_known_count: int | None
    procedure_known_count: int | None
    process_coverage: Decimal | None
    offer_coverage: Decimal | None
    procedure_coverage: Decimal | None
    sufficient: bool
    unavailable_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        level = self.level_name.strip()
        if not level:
            raise CompetitionContextError(
                "competition context attempt level_name must not be blank"
            )
        object.__setattr__(self, "level_name", level)
        if self.market_key_sha256 is not None:
            object.__setattr__(
                self,
                "market_key_sha256",
                _validate_digest(
                    self.market_key_sha256, "market_key_sha256"
                ),
            )
        if self.reference_mode is not None:
            object.__setattr__(
                self,
                "reference_mode",
                CompetitionContextMode(self.reference_mode),
            )
        for name in (
            "base_award_count",
            "process_known_count",
            "offers_known_count",
            "procedure_known_count",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise CompetitionContextError(
                    f"{name} must be a non-negative integer or None"
                )
        for name in (
            "process_coverage",
            "offer_coverage",
            "procedure_coverage",
        ):
            value = getattr(self, name)
            if value is not None:
                _validate_fraction(value, name)
        reasons = tuple(
            reason.strip()
            for reason in self.unavailable_reasons
            if reason.strip()
        )
        if len(reasons) != len(set(reasons)):
            raise CompetitionContextError(
                "competition context attempt reasons contain duplicates"
            )
        object.__setattr__(self, "unavailable_reasons", reasons)


@dataclass(frozen=True, slots=True)
class CompetitionContext:
    """Selected award-level competition context for one target transaction."""

    transaction_id: str
    award_id: str
    target_lifecycle_kind: AwardActionKind
    reference_mode: CompetitionContextMode
    formation_transaction_id: str
    target_evidence: CompetitionEvidence

    market_key: PeerGroupKey
    plan_name: str
    plan_sha256: str
    support_spec_sha256: str

    base_award_count: int

    process_known_count: int
    competitive_process_count: int
    noncompetitive_process_count: int
    process_coverage: Decimal
    competitive_process_rate: Decimal | None
    noncompetitive_process_rate: Decimal | None

    offers_known_count: int
    single_offer_count: int
    multiple_offer_count: int
    zero_offer_count: int
    offer_coverage: Decimal
    single_offer_rate: Decimal | None
    multiple_offer_rate: Decimal | None
    zero_offer_rate: Decimal | None

    procedure_known_count: int
    only_one_source_solicited_count: int
    procedure_coverage: Decimal
    only_one_source_solicited_rate: Decimal | None

    noncompetition_authority_reported_count: int
    noncompetition_authority_reported_rate: Decimal
    full_open_after_exclusion_count: int
    full_open_after_exclusion_rate: Decimal
    conflict_award_count: int
    conflict_rate: Decimal
    missing_core_field_award_count: int
    missing_core_field_rate: Decimal

    def __post_init__(self) -> None:
        for name in (
            "transaction_id",
            "award_id",
            "formation_transaction_id",
            "plan_name",
        ):
            value = getattr(self, name).strip()
            if not value:
                raise CompetitionContextError(f"{name} must not be blank")
            object.__setattr__(self, name, value)

        object.__setattr__(
            self,
            "target_lifecycle_kind",
            AwardActionKind(self.target_lifecycle_kind),
        )
        object.__setattr__(
            self, "reference_mode", CompetitionContextMode(self.reference_mode)
        )
        if not isinstance(self.target_evidence, CompetitionEvidence):
            raise TypeError("target_evidence must be CompetitionEvidence")
        if self.target_evidence.award_id != self.award_id:
            raise CompetitionContextError(
                "target evidence award_id differs from context award_id"
            )
        if self.target_evidence.transaction_id != self.formation_transaction_id:
            raise CompetitionContextError(
                "target evidence must describe the formation/base transaction"
            )
        if not isinstance(self.market_key, PeerGroupKey):
            raise TypeError("market_key must be PeerGroupKey")
        for name in ("plan_sha256", "support_spec_sha256"):
            object.__setattr__(
                self, name, _validate_digest(getattr(self, name), name)
            )

        count_names = (
            "base_award_count",
            "process_known_count",
            "competitive_process_count",
            "noncompetitive_process_count",
            "offers_known_count",
            "single_offer_count",
            "multiple_offer_count",
            "zero_offer_count",
            "procedure_known_count",
            "only_one_source_solicited_count",
            "noncompetition_authority_reported_count",
            "full_open_after_exclusion_count",
            "conflict_award_count",
            "missing_core_field_award_count",
        )
        for name in count_names:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > self.base_award_count
            ):
                raise CompetitionContextError(
                    f"{name} is inconsistent with base_award_count"
                )
        if self.base_award_count < 1:
            raise CompetitionContextError(
                "selected competition context requires at least one reference award"
            )
        if (
            self.competitive_process_count + self.noncompetitive_process_count
            != self.process_known_count
        ):
            raise CompetitionContextError(
                "process counts do not match process-known count"
            )
        if (
            self.single_offer_count
            + self.multiple_offer_count
            + self.zero_offer_count
            != self.offers_known_count
        ):
            raise CompetitionContextError(
                "offer counts do not match offers-known count"
            )
        if self.only_one_source_solicited_count > self.procedure_known_count:
            raise CompetitionContextError(
                "one-source count cannot exceed procedure-known count"
            )

        for name in (
            "process_coverage",
            "offer_coverage",
            "procedure_coverage",
            "noncompetition_authority_reported_rate",
            "full_open_after_exclusion_rate",
            "conflict_rate",
            "missing_core_field_rate",
        ):
            _validate_fraction(getattr(self, name), name)
        for name in (
            "competitive_process_rate",
            "noncompetitive_process_rate",
            "single_offer_rate",
            "multiple_offer_rate",
            "zero_offer_rate",
            "only_one_source_solicited_rate",
        ):
            value = getattr(self, name)
            if value is not None:
                _validate_fraction(value, name)

        if self.process_coverage != Decimal(self.process_known_count) / Decimal(
            self.base_award_count
        ):
            raise CompetitionContextError("process coverage is inconsistent")
        if self.offer_coverage != Decimal(self.offers_known_count) / Decimal(
            self.base_award_count
        ):
            raise CompetitionContextError("offer coverage is inconsistent")
        if self.procedure_coverage != Decimal(
            self.procedure_known_count
        ) / Decimal(self.base_award_count):
            raise CompetitionContextError("procedure coverage is inconsistent")

    @property
    def market_level(self) -> str:
        return self.market_key.level_name

    @property
    def target_reported_process_competitive(self) -> bool | None:
        return self.target_evidence.reported_process_competitive

    @property
    def target_offer_outcome_kind(self) -> OfferOutcomeKind:
        return self.target_evidence.offer_outcome_kind

    @property
    def target_single_offer_reported(self) -> bool | None:
        return self.target_evidence.single_offer_reported

    @property
    def target_only_one_source_solicited(self) -> bool | None:
        if (
            self.target_evidence.solicitation_procedure_kind
            is SolicitationProcedureKind.UNKNOWN
        ):
            return None
        return (
            self.target_evidence.solicitation_procedure_kind
            is SolicitationProcedureKind.ONLY_ONE_SOURCE
        )

    @property
    def evidence_sha256(self) -> str:
        return _digest(self.as_dict(include_evidence_sha=False))

    def as_dict(self, *, include_evidence_sha: bool = True) -> dict[str, Any]:
        result = {
            "transaction_id": self.transaction_id,
            "award_id": self.award_id,
            "target_lifecycle_kind": self.target_lifecycle_kind.value,
            "reference_mode": self.reference_mode.value,
            "formation_transaction_id": self.formation_transaction_id,
            "target_competition_evidence": self.target_evidence.as_dict(),
            "market_level": self.market_level,
            "market_key_sha256": self.market_key.sha256_hex,
            "plan_name": self.plan_name,
            "plan_sha256": self.plan_sha256,
            "support_spec_sha256": self.support_spec_sha256,
            "reference": {
                "base_award_count": self.base_award_count,
                "process_known_count": self.process_known_count,
                "process_coverage": str(self.process_coverage),
                "competitive_process_count": self.competitive_process_count,
                "noncompetitive_process_count": self.noncompetitive_process_count,
                "competitive_process_rate": _decimal_text(self.competitive_process_rate),
                "noncompetitive_process_rate": _decimal_text(
                    self.noncompetitive_process_rate
                ),
                "offers_known_count": self.offers_known_count,
                "offer_coverage": str(self.offer_coverage),
                "single_offer_count": self.single_offer_count,
                "multiple_offer_count": self.multiple_offer_count,
                "zero_offer_count": self.zero_offer_count,
                "single_offer_rate": _decimal_text(self.single_offer_rate),
                "multiple_offer_rate": _decimal_text(self.multiple_offer_rate),
                "zero_offer_rate": _decimal_text(self.zero_offer_rate),
                "procedure_known_count": self.procedure_known_count,
                "procedure_coverage": str(self.procedure_coverage),
                "only_one_source_solicited_count":
                    self.only_one_source_solicited_count,
                "only_one_source_solicited_rate": _decimal_text(
                    self.only_one_source_solicited_rate
                ),
                "noncompetition_authority_reported_count":
                    self.noncompetition_authority_reported_count,
                "noncompetition_authority_reported_rate": str(
                    self.noncompetition_authority_reported_rate
                ),
                "full_open_after_exclusion_count":
                    self.full_open_after_exclusion_count,
                "full_open_after_exclusion_rate": str(
                    self.full_open_after_exclusion_rate
                ),
                "conflict_award_count": self.conflict_award_count,
                "conflict_rate": str(self.conflict_rate),
                "missing_core_field_award_count":
                    self.missing_core_field_award_count,
                "missing_core_field_rate": str(
                    self.missing_core_field_rate
                ),
            },
        }
        if include_evidence_sha:
            result["evidence_sha256"] = self.evidence_sha256
        return result


@dataclass(frozen=True, slots=True)
class CompetitionContextResult:
    """Resolution report preserving all attempted peer levels."""

    transaction_id: str
    award_id: str
    support_spec: CompetitionContextSupportSpec
    context: CompetitionContext | None
    attempts: tuple[CompetitionContextAttempt, ...]
    unavailable_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        txid, award_id = self.transaction_id.strip(), self.award_id.strip()
        if not txid or not award_id:
            raise CompetitionContextError(
                "transaction_id and award_id must not be blank"
            )
        object.__setattr__(self, "transaction_id", txid)
        object.__setattr__(self, "award_id", award_id)
        if not isinstance(self.support_spec, CompetitionContextSupportSpec):
            raise TypeError(
                "support_spec must be CompetitionContextSupportSpec"
            )
        if self.context is not None:
            if self.context.transaction_id != txid:
                raise CompetitionContextError(
                    "context transaction_id differs from result"
                )
            if self.context.award_id != award_id:
                raise CompetitionContextError(
                    "context award_id differs from result"
                )
            if self.unavailable_reasons:
                raise CompetitionContextError(
                    "available context cannot carry unavailable reasons"
                )
        object.__setattr__(self, "attempts", tuple(self.attempts))
        reasons = tuple(
            reason.strip()
            for reason in self.unavailable_reasons
            if reason.strip()
        )
        if len(reasons) != len(set(reasons)):
            raise CompetitionContextError(
                "competition context result reasons contain duplicates"
            )
        object.__setattr__(self, "unavailable_reasons", reasons)

    @property
    def available(self) -> bool:
        return self.context is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "award_id": self.award_id,
            "support_spec": {
                "minimum_base_awards": self.support_spec.minimum_base_awards,
                "minimum_process_known": self.support_spec.minimum_process_known,
                "minimum_offers_known": self.support_spec.minimum_offers_known,
                "minimum_procedure_known":
                    self.support_spec.minimum_procedure_known,
                "minimum_process_coverage": str(
                    self.support_spec.minimum_process_coverage
                ),
                "minimum_offer_coverage": str(
                    self.support_spec.minimum_offer_coverage
                ),
                "minimum_procedure_coverage": str(
                    self.support_spec.minimum_procedure_coverage
                ),
                "sha256": self.support_spec.sha256_hex,
            },
            "available": self.available,
            "unavailable_reasons": list(self.unavailable_reasons),
            "attempts": [
                {
                    "level_name": attempt.level_name,
                    "market_key_sha256": attempt.market_key_sha256,
                    "reference_mode": (
                        None
                        if attempt.reference_mode is None
                        else attempt.reference_mode.value
                    ),
                    "base_award_count": attempt.base_award_count,
                    "process_known_count": attempt.process_known_count,
                    "offers_known_count": attempt.offers_known_count,
                    "procedure_known_count": attempt.procedure_known_count,
                    "process_coverage": _decimal_text(
                        attempt.process_coverage
                    ),
                    "offer_coverage": _decimal_text(
                        attempt.offer_coverage
                    ),
                    "procedure_coverage": _decimal_text(
                        attempt.procedure_coverage
                    ),
                    "sufficient": attempt.sufficient,
                    "unavailable_reasons": list(
                        attempt.unavailable_reasons
                    ),
                }
                for attempt in self.attempts
            ],
            "context": (
                None if self.context is None else self.context.as_dict()
            ),
        }


def resolve_competition_context(
    transaction: ProcurementTransaction,
    snapshot: CompetitionReferenceSnapshot,
    *,
    support_spec: CompetitionContextSupportSpec,
) -> CompetitionContextResult:
    """Resolve target into the first caller-approved award-formation market."""

    if not isinstance(transaction, ProcurementTransaction):
        raise TypeError("transaction must be ProcurementTransaction")
    if not isinstance(snapshot, CompetitionReferenceSnapshot):
        raise TypeError("snapshot must be CompetitionReferenceSnapshot")
    if not isinstance(support_spec, CompetitionContextSupportSpec):
        raise TypeError(
            "support_spec must be CompetitionContextSupportSpec"
        )

    lifecycle = classify_award_action(transaction)
    indexed = snapshot.get_base_award(transaction.award_id)

    resolved = _target_formation_evidence(
        transaction, snapshot, lifecycle.kind, indexed
    )
    if resolved is None:
        return CompetitionContextResult(
            transaction_id=transaction.transaction_id,
            award_id=transaction.award_id,
            support_spec=support_spec,
            context=None,
            attempts=(),
            unavailable_reasons=(
                _formation_unavailable_reason(lifecycle.kind),
            ),
        )

    mode, formation_evidence, candidates, contribution = resolved
    attempts: list[CompetitionContextAttempt] = []

    for candidate in candidates:
        if candidate.key is None:
            attempts.append(
                CompetitionContextAttempt(
                    level_name=candidate.level_name,
                    market_key_sha256=None,
                    reference_mode=mode,
                    base_award_count=None,
                    process_known_count=None,
                    offers_known_count=None,
                    procedure_known_count=None,
                    process_coverage=None,
                    offer_coverage=None,
                    procedure_coverage=None,
                    sufficient=False,
                    unavailable_reasons=candidate.unavailable_reasons,
                )
            )
            continue

        market = snapshot.get_market(candidate.key)
        if market is None:
            attempts.append(
                CompetitionContextAttempt(
                    level_name=candidate.level_name,
                    market_key_sha256=candidate.key.sha256_hex,
                    reference_mode=mode,
                    base_award_count=0,
                    process_known_count=0,
                    offers_known_count=0,
                    procedure_known_count=0,
                    process_coverage=None,
                    offer_coverage=None,
                    procedure_coverage=None,
                    sufficient=False,
                    unavailable_reasons=("market_not_observed",),
                )
            )
            continue

        reference = (
            market.without(contribution)
            if mode is not CompetitionContextMode.EXTERNAL_BASE_AWARD
            else market
        )
        reasons = _support_failures(reference, support_spec)
        sufficient = not reasons
        attempts.append(
            CompetitionContextAttempt(
                level_name=candidate.level_name,
                market_key_sha256=candidate.key.sha256_hex,
                reference_mode=mode,
                base_award_count=reference.base_award_count,
                process_known_count=reference.process_known_count,
                offers_known_count=reference.offers_known_count,
                procedure_known_count=reference.procedure_known_count,
                process_coverage=reference.process_coverage,
                offer_coverage=reference.offer_coverage,
                procedure_coverage=reference.procedure_coverage,
                sufficient=sufficient,
                unavailable_reasons=tuple(reasons),
            )
        )
        if not sufficient:
            continue

        return CompetitionContextResult(
            transaction_id=transaction.transaction_id,
            award_id=transaction.award_id,
            support_spec=support_spec,
            context=_build_context(
                transaction,
                snapshot,
                support_spec,
                mode,
                formation_evidence,
                reference,
            ),
            attempts=tuple(attempts),
        )

    return CompetitionContextResult(
        transaction_id=transaction.transaction_id,
        award_id=transaction.award_id,
        support_spec=support_spec,
        context=None,
        attempts=tuple(attempts),
        unavailable_reasons=("no_supported_competition_market",),
    )


def _target_formation_evidence(
    transaction: ProcurementTransaction,
    snapshot: CompetitionReferenceSnapshot,
    lifecycle_kind: AwardActionKind,
    indexed: CompetitionBaseAwardReference | None,
) -> tuple[
    CompetitionContextMode,
    CompetitionEvidence,
    tuple[PeerGroupCandidate, ...],
    Any,
] | None:
    if indexed is not None:
        if lifecycle_kind is AwardActionKind.BASE_AWARD:
            lifecycle = classify_award_action(transaction)
            evidence = build_competition_evidence(transaction)
            candidates = peer_group_candidates(transaction, snapshot.plan)
            actual = base_award_fingerprint(
                transaction, lifecycle, evidence, candidates
            )
            if actual != indexed.fingerprint_sha256:
                raise CompetitionContextError(
                    "indexed base award differs from stored competition reference"
                )
            return (
                CompetitionContextMode.INDEXED_BASE_AWARD,
                indexed.evidence,
                indexed.candidates,
                indexed.contribution,
            )

        return (
            CompetitionContextMode.INDEXED_AWARD,
            indexed.evidence,
            indexed.candidates,
            indexed.contribution,
        )

    if lifecycle_kind is not AwardActionKind.BASE_AWARD:
        return None

    evidence = build_competition_evidence(transaction)
    candidates = peer_group_candidates(transaction, snapshot.plan)
    return (
        CompetitionContextMode.EXTERNAL_BASE_AWARD,
        evidence,
        candidates,
        None,
    )


def _formation_unavailable_reason(kind: AwardActionKind) -> str:
    if kind is AwardActionKind.MODIFICATION:
        return "base_award_reference_missing_for_modification"
    if kind is AwardActionKind.UNKNOWN:
        return "base_award_reference_missing_and_lifecycle_unknown"
    return "award_formation_context_unavailable"


def _support_failures(
    reference: CompetitionMarketReference,
    spec: CompetitionContextSupportSpec,
) -> list[str]:
    reasons: list[str] = []
    if reference.base_award_count < spec.minimum_base_awards:
        reasons.append("insufficient_base_awards")
    if reference.process_known_count < spec.minimum_process_known:
        reasons.append("insufficient_process_known")
    if reference.offers_known_count < spec.minimum_offers_known:
        reasons.append("insufficient_offers_known")
    if reference.procedure_known_count < spec.minimum_procedure_known:
        reasons.append("insufficient_procedure_known")

    if (
        reference.process_coverage is None
        or reference.process_coverage < spec.minimum_process_coverage
    ):
        reasons.append("insufficient_process_coverage")
    if (
        reference.offer_coverage is None
        or reference.offer_coverage < spec.minimum_offer_coverage
    ):
        reasons.append("insufficient_offer_coverage")
    if (
        reference.procedure_coverage is None
        or reference.procedure_coverage < spec.minimum_procedure_coverage
    ):
        reasons.append("insufficient_procedure_coverage")
    return reasons


def _build_context(
    transaction: ProcurementTransaction,
    snapshot: CompetitionReferenceSnapshot,
    support_spec: CompetitionContextSupportSpec,
    mode: CompetitionContextMode,
    evidence: CompetitionEvidence,
    reference: CompetitionMarketReference,
) -> CompetitionContext:
    lifecycle = classify_award_action(transaction)
    return CompetitionContext(
        transaction_id=transaction.transaction_id,
        award_id=transaction.award_id,
        target_lifecycle_kind=lifecycle.kind,
        reference_mode=mode,
        formation_transaction_id=evidence.transaction_id,
        target_evidence=evidence,
        market_key=reference.key,
        plan_name=snapshot.plan.name,
        plan_sha256=snapshot.plan.sha256_hex,
        support_spec_sha256=support_spec.sha256_hex,
        base_award_count=reference.base_award_count,
        process_known_count=reference.process_known_count,
        competitive_process_count=reference.competitive_process_count,
        noncompetitive_process_count=reference.noncompetitive_process_count,
        process_coverage=reference.process_coverage,
        competitive_process_rate=reference.competitive_process_rate,
        noncompetitive_process_rate=reference.noncompetitive_process_rate,
        offers_known_count=reference.offers_known_count,
        single_offer_count=reference.single_offer_count,
        multiple_offer_count=reference.multiple_offer_count,
        zero_offer_count=reference.zero_offer_count,
        offer_coverage=reference.offer_coverage,
        single_offer_rate=reference.single_offer_rate,
        multiple_offer_rate=reference.multiple_offer_rate,
        zero_offer_rate=reference.zero_offer_rate,
        procedure_known_count=reference.procedure_known_count,
        only_one_source_solicited_count=
            reference.only_one_source_solicited_count,
        procedure_coverage=reference.procedure_coverage,
        only_one_source_solicited_rate=
            reference.only_one_source_solicited_rate,
        noncompetition_authority_reported_count=
            reference.noncompetition_authority_reported_count,
        noncompetition_authority_reported_rate=
            reference.noncompetition_authority_reported_rate,
        full_open_after_exclusion_count=
            reference.full_open_after_exclusion_count,
        full_open_after_exclusion_rate=
            reference.full_open_after_exclusion_rate,
        conflict_award_count=reference.conflict_award_count,
        conflict_rate=reference.conflict_rate,
        missing_core_field_award_count=
            reference.missing_core_field_award_count,
        missing_core_field_rate=reference.missing_core_field_rate,
    )


def _validate_fraction(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise CompetitionContextError(f"{name} must be a finite Decimal")
    if value < Decimal(0) or value > Decimal(1):
        raise CompetitionContextError(f"{name} must be between 0 and 1")


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _validate_digest(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise CompetitionContextError(
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
