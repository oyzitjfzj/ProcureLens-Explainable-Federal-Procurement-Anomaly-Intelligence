"""Award-level competition reference markets for ProcureLens.

Builds contextual prevalence from observed BASE_AWARD actions only. Later award
modifications never become extra market observations, preventing one contract
with many modifications from inflating single-offer or noncompetitive rates.

The index stores fixed-size competition contributions and the minimum base-award
evidence needed for safe leave-one-award-out resolution later. It does not
evaluate a target, assign anomaly thresholds, score risk, or infer misconduct.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any

from procurelens.domain.transaction import ProcurementTransaction
from procurelens.features.award_lifecycle import (
    AwardActionEvidence,
    AwardActionKind,
    classify_award_action,
)
from procurelens.features.competition_evidence import (
    CompetitionEvidence,
    CompetitionExtentKind,
    OfferOutcomeKind,
    OtherThanFullOpenAuthorityKind,
    SolicitationProcedureKind,
    build_competition_evidence,
)
from procurelens.features.peer_groups import (
    AgencyScope,
    CategoryScope,
    PeerGroupCandidate,
    PeerGroupKey,
    PeerGroupLevel,
    PeerGroupPlan,
    TimeScope,
    peer_group_candidates,
)


class CompetitionReferenceError(ValueError):
    """Raised when competition reference evidence or state is inconsistent."""


@dataclass(frozen=True, slots=True)
class CompetitionContribution:
    """Fixed-size contribution of one base award to one comparison market."""

    process_competitive: bool | None
    offer_outcome_kind: OfferOutcomeKind
    solicitation_procedure_known: bool
    only_one_source_solicited: bool
    noncompetition_authority_reported: bool
    full_open_after_exclusion: bool
    has_conflict: bool
    has_missing_core_fields: bool

    def __post_init__(self) -> None:
        if self.process_competitive not in (True, False, None):
            raise CompetitionReferenceError(
                "process_competitive must be bool or None"
            )
        object.__setattr__(
            self, "offer_outcome_kind", OfferOutcomeKind(self.offer_outcome_kind)
        )
        for name in (
            "solicitation_procedure_known",
            "only_one_source_solicited",
            "noncompetition_authority_reported",
            "full_open_after_exclusion",
            "has_conflict",
            "has_missing_core_fields",
        ):
            if not isinstance(getattr(self, name), bool):
                raise CompetitionReferenceError(f"{name} must be bool")
        if self.only_one_source_solicited and not self.solicitation_procedure_known:
            raise CompetitionReferenceError(
                "one-source solicitation requires a recognized procedure"
            )


@dataclass(frozen=True, slots=True)
class CompetitionMarketReference:
    """Aggregate award-level competition evidence for one peer-group key."""

    key: PeerGroupKey
    base_award_count: int
    process_known_count: int
    competitive_process_count: int
    noncompetitive_process_count: int
    offers_known_count: int
    single_offer_count: int
    multiple_offer_count: int
    zero_offer_count: int
    procedure_known_count: int
    only_one_source_solicited_count: int
    noncompetition_authority_reported_count: int
    full_open_after_exclusion_count: int
    conflict_award_count: int
    missing_core_field_award_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.key, PeerGroupKey):
            raise TypeError("key must be PeerGroupKey")
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
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CompetitionReferenceError(
                    f"{name} must be a non-negative integer"
                )
            if value > self.base_award_count:
                raise CompetitionReferenceError(
                    f"{name} cannot exceed base_award_count"
                )
        if (
            self.competitive_process_count + self.noncompetitive_process_count
            != self.process_known_count
        ):
            raise CompetitionReferenceError(
                "process outcome counts do not match process-known count"
            )
        if (
            self.single_offer_count
            + self.multiple_offer_count
            + self.zero_offer_count
            != self.offers_known_count
        ):
            raise CompetitionReferenceError(
                "offer outcome counts do not match offers-known count"
            )
        if self.only_one_source_solicited_count > self.procedure_known_count:
            raise CompetitionReferenceError(
                "one-source count cannot exceed procedure-known count"
            )

    @property
    def process_coverage(self) -> Decimal | None:
        return _rate(self.process_known_count, self.base_award_count)

    @property
    def offer_coverage(self) -> Decimal | None:
        return _rate(self.offers_known_count, self.base_award_count)

    @property
    def procedure_coverage(self) -> Decimal | None:
        return _rate(self.procedure_known_count, self.base_award_count)

    @property
    def competitive_process_rate(self) -> Decimal | None:
        return _rate(self.competitive_process_count, self.process_known_count)

    @property
    def noncompetitive_process_rate(self) -> Decimal | None:
        return _rate(self.noncompetitive_process_count, self.process_known_count)

    @property
    def single_offer_rate(self) -> Decimal | None:
        return _rate(self.single_offer_count, self.offers_known_count)

    @property
    def multiple_offer_rate(self) -> Decimal | None:
        return _rate(self.multiple_offer_count, self.offers_known_count)

    @property
    def zero_offer_rate(self) -> Decimal | None:
        return _rate(self.zero_offer_count, self.offers_known_count)

    @property
    def only_one_source_solicited_rate(self) -> Decimal | None:
        return _rate(
            self.only_one_source_solicited_count, self.procedure_known_count
        )

    @property
    def noncompetition_authority_reported_rate(self) -> Decimal | None:
        return _rate(
            self.noncompetition_authority_reported_count, self.base_award_count
        )

    @property
    def full_open_after_exclusion_rate(self) -> Decimal | None:
        return _rate(self.full_open_after_exclusion_count, self.base_award_count)

    @property
    def conflict_rate(self) -> Decimal | None:
        return _rate(self.conflict_award_count, self.base_award_count)

    @property
    def missing_core_field_rate(self) -> Decimal | None:
        return _rate(self.missing_core_field_award_count, self.base_award_count)

    def without(
        self, contribution: CompetitionContribution
    ) -> "CompetitionMarketReference":
        """Return exact leave-one-award-out market evidence."""

        if self.base_award_count < 1:
            raise CompetitionReferenceError(
                "cannot remove a base award from an empty market"
            )
        delta = contribution_counts(contribution)
        values = {
            "base_award_count": self.base_award_count - 1,
            "process_known_count": self.process_known_count
            - delta["process_known_count"],
            "competitive_process_count": self.competitive_process_count
            - delta["competitive_process_count"],
            "noncompetitive_process_count": self.noncompetitive_process_count
            - delta["noncompetitive_process_count"],
            "offers_known_count": self.offers_known_count
            - delta["offers_known_count"],
            "single_offer_count": self.single_offer_count
            - delta["single_offer_count"],
            "multiple_offer_count": self.multiple_offer_count
            - delta["multiple_offer_count"],
            "zero_offer_count": self.zero_offer_count
            - delta["zero_offer_count"],
            "procedure_known_count": self.procedure_known_count
            - delta["procedure_known_count"],
            "only_one_source_solicited_count":
                self.only_one_source_solicited_count
                - delta["only_one_source_solicited_count"],
            "noncompetition_authority_reported_count":
                self.noncompetition_authority_reported_count
                - delta["noncompetition_authority_reported_count"],
            "full_open_after_exclusion_count": self.full_open_after_exclusion_count
            - delta["full_open_after_exclusion_count"],
            "conflict_award_count": self.conflict_award_count
            - delta["conflict_award_count"],
            "missing_core_field_award_count": self.missing_core_field_award_count
            - delta["missing_core_field_award_count"],
        }
        if any(value < 0 for value in values.values()):
            raise CompetitionReferenceError(
                "base-award contribution is absent from market aggregate"
            )
        return CompetitionMarketReference(key=self.key, **values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "market_level": self.key.level_name,
            "market_key_sha256": self.key.sha256_hex,
            "base_award_count": self.base_award_count,
            "process_known_count": self.process_known_count,
            "process_coverage": _decimal_text(self.process_coverage),
            "competitive_process_count": self.competitive_process_count,
            "noncompetitive_process_count": self.noncompetitive_process_count,
            "competitive_process_rate": _decimal_text(self.competitive_process_rate),
            "noncompetitive_process_rate": _decimal_text(
                self.noncompetitive_process_rate
            ),
            "offers_known_count": self.offers_known_count,
            "offer_coverage": _decimal_text(self.offer_coverage),
            "single_offer_count": self.single_offer_count,
            "multiple_offer_count": self.multiple_offer_count,
            "zero_offer_count": self.zero_offer_count,
            "single_offer_rate": _decimal_text(self.single_offer_rate),
            "multiple_offer_rate": _decimal_text(self.multiple_offer_rate),
            "zero_offer_rate": _decimal_text(self.zero_offer_rate),
            "procedure_known_count": self.procedure_known_count,
            "procedure_coverage": _decimal_text(self.procedure_coverage),
            "only_one_source_solicited_count": self.only_one_source_solicited_count,
            "only_one_source_solicited_rate": _decimal_text(
                self.only_one_source_solicited_rate
            ),
            "noncompetition_authority_reported_count":
                self.noncompetition_authority_reported_count,
            "noncompetition_authority_reported_rate": _decimal_text(
                self.noncompetition_authority_reported_rate
            ),
            "full_open_after_exclusion_count": self.full_open_after_exclusion_count,
            "full_open_after_exclusion_rate": _decimal_text(
                self.full_open_after_exclusion_rate
            ),
            "conflict_award_count": self.conflict_award_count,
            "conflict_rate": _decimal_text(self.conflict_rate),
            "missing_core_field_award_count": self.missing_core_field_award_count,
            "missing_core_field_rate": _decimal_text(self.missing_core_field_rate),
        }


@dataclass(frozen=True, slots=True)
class CompetitionBaseAwardReference:
    """Stored base-award evidence required for later safe context resolution."""

    award_id: str
    transaction_id: str
    lifecycle_sha256: str
    evidence: CompetitionEvidence
    candidates: tuple[PeerGroupCandidate, ...]
    contribution: CompetitionContribution
    fingerprint_sha256: str

    def __post_init__(self) -> None:
        award_id, txid = self.award_id.strip(), self.transaction_id.strip()
        if not award_id or not txid:
            raise CompetitionReferenceError(
                "base-award reference ids must not be blank"
            )
        object.__setattr__(self, "award_id", award_id)
        object.__setattr__(self, "transaction_id", txid)
        object.__setattr__(
            self,
            "lifecycle_sha256",
            _validate_digest(self.lifecycle_sha256, "lifecycle_sha256"),
        )
        if not isinstance(self.evidence, CompetitionEvidence):
            raise TypeError("evidence must be CompetitionEvidence")
        if self.evidence.transaction_id != txid or self.evidence.award_id != award_id:
            raise CompetitionReferenceError(
                "base-award ids differ from competition evidence"
            )
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if not isinstance(self.contribution, CompetitionContribution):
            raise TypeError("contribution must be CompetitionContribution")
        object.__setattr__(
            self,
            "fingerprint_sha256",
            _validate_digest(self.fingerprint_sha256, "fingerprint_sha256"),
        )


@dataclass(frozen=True, slots=True)
class CompetitionReferenceSnapshot:
    plan: PeerGroupPlan
    total_transactions_seen: int
    observed_base_awards: int
    observed_modifications: int
    lifecycle_unknown_transactions: int
    markets: Mapping[PeerGroupKey, CompetitionMarketReference]
    base_awards: Mapping[str, CompetitionBaseAwardReference]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, PeerGroupPlan):
            raise TypeError("plan must be PeerGroupPlan")
        for name in (
            "total_transactions_seen",
            "observed_base_awards",
            "observed_modifications",
            "lifecycle_unknown_transactions",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CompetitionReferenceError(
                    f"{name} must be a non-negative integer"
                )
        if (
            self.observed_base_awards
            + self.observed_modifications
            + self.lifecycle_unknown_transactions
            != self.total_transactions_seen
        ):
            raise CompetitionReferenceError(
                "lifecycle counts do not sum to total transactions"
            )
        markets = dict(self.markets)
        if any(key != market.key for key, market in markets.items()):
            raise CompetitionReferenceError(
                "market mapping key differs from market evidence key"
            )
        object.__setattr__(self, "markets", MappingProxyType(markets))
        awards = dict(self.base_awards)
        if len(awards) != self.observed_base_awards:
            raise CompetitionReferenceError(
                "base-award references differ from observed base-award count"
            )
        if any(key != reference.award_id for key, reference in awards.items()):
            raise CompetitionReferenceError(
                "base-award mapping key differs from reference award id"
            )
        object.__setattr__(self, "base_awards", MappingProxyType(awards))

    @property
    def market_count(self) -> int:
        return len(self.markets)

    def get_market(
        self, key: PeerGroupKey
    ) -> CompetitionMarketReference | None:
        return self.markets.get(key)

    def get_base_award(
        self, award_id: str
    ) -> CompetitionBaseAwardReference | None:
        return self.base_awards.get(award_id)


@dataclass(frozen=True, slots=True)
class CompetitionReferencePolicy:
    """Optional memory budgets only; no anomaly/risk thresholds live here."""

    max_transaction_ids: int | None = None
    max_base_award_ids: int | None = None
    max_distinct_markets: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_transaction_ids",
            "max_base_award_ids",
            "max_distinct_markets",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise CompetitionReferenceError(
                    f"{name} must be a positive integer or None"
                )


@dataclass(slots=True)
class _Accumulator:
    base_award_count: int = 0
    process_known_count: int = 0
    competitive_process_count: int = 0
    noncompetitive_process_count: int = 0
    offers_known_count: int = 0
    single_offer_count: int = 0
    multiple_offer_count: int = 0
    zero_offer_count: int = 0
    procedure_known_count: int = 0
    only_one_source_solicited_count: int = 0
    noncompetition_authority_reported_count: int = 0
    full_open_after_exclusion_count: int = 0
    conflict_award_count: int = 0
    missing_core_field_award_count: int = 0

    def add(self, contribution: CompetitionContribution) -> None:
        counts = contribution_counts(contribution)
        self.base_award_count += 1
        for name, value in counts.items():
            setattr(self, name, getattr(self, name) + value)

    def freeze(self, key: PeerGroupKey) -> CompetitionMarketReference:
        return CompetitionMarketReference(
            key=key,
            base_award_count=self.base_award_count,
            process_known_count=self.process_known_count,
            competitive_process_count=self.competitive_process_count,
            noncompetitive_process_count=self.noncompetitive_process_count,
            offers_known_count=self.offers_known_count,
            single_offer_count=self.single_offer_count,
            multiple_offer_count=self.multiple_offer_count,
            zero_offer_count=self.zero_offer_count,
            procedure_known_count=self.procedure_known_count,
            only_one_source_solicited_count=self.only_one_source_solicited_count,
            noncompetition_authority_reported_count=
                self.noncompetition_authority_reported_count,
            full_open_after_exclusion_count=self.full_open_after_exclusion_count,
            conflict_award_count=self.conflict_award_count,
            missing_core_field_award_count=self.missing_core_field_award_count,
        )


class CompetitionReferenceIndex:
    """One-pass award-level reference builder; full transactions are not retained."""

    def __init__(
        self,
        *,
        plan: PeerGroupPlan | None = None,
        policy: CompetitionReferencePolicy | None = None,
    ) -> None:
        self.plan = plan or federal_competition_context_plan()
        if not isinstance(self.plan, PeerGroupPlan):
            raise TypeError("plan must be PeerGroupPlan")
        self.policy = policy or CompetitionReferencePolicy()
        self._transaction_ids: set[str] = set()
        self._base_awards: dict[str, CompetitionBaseAwardReference] = {}
        self._markets: dict[PeerGroupKey, _Accumulator] = {}
        self._total = 0
        self._base = 0
        self._modifications = 0
        self._unknown = 0

    def observe(self, transaction: ProcurementTransaction) -> None:
        """Atomically add one transaction after every configured check passes."""

        if not isinstance(transaction, ProcurementTransaction):
            raise TypeError("transaction must be ProcurementTransaction")

        txid = transaction.transaction_id
        if txid in self._transaction_ids:
            raise CompetitionReferenceError(
                f"duplicate transaction_id in competition reference: {txid!r}"
            )
        transaction_limit = self.policy.max_transaction_ids
        if (
            transaction_limit is not None
            and len(self._transaction_ids) >= transaction_limit
        ):
            raise CompetitionReferenceError(
                f"transaction index exceeds max_transaction_ids={transaction_limit}"
            )

        lifecycle = classify_award_action(transaction)
        reference: CompetitionBaseAwardReference | None = None
        keys: tuple[PeerGroupKey, ...] = ()

        if lifecycle.kind is AwardActionKind.BASE_AWARD:
            if transaction.award_id in self._base_awards:
                raise CompetitionReferenceError(
                    f"multiple base-award actions for award_id={transaction.award_id!r}"
                )
            base_limit = self.policy.max_base_award_ids
            if base_limit is not None and len(self._base_awards) >= base_limit:
                raise CompetitionReferenceError(
                    f"base-award index exceeds max_base_award_ids={base_limit}"
                )

            evidence = build_competition_evidence(transaction)
            candidates = peer_group_candidates(transaction, self.plan)
            keys = tuple(
                candidate.key
                for candidate in candidates
                if candidate.key is not None
            )
            new_keys = {key for key in keys if key not in self._markets}
            market_limit = self.policy.max_distinct_markets
            if (
                market_limit is not None
                and len(self._markets) + len(new_keys) > market_limit
            ):
                raise CompetitionReferenceError(
                    f"market index exceeds max_distinct_markets={market_limit}"
                )

            contribution = competition_contribution(evidence)
            reference = CompetitionBaseAwardReference(
                award_id=transaction.award_id,
                transaction_id=transaction.transaction_id,
                lifecycle_sha256=lifecycle.sha256_hex,
                evidence=evidence,
                candidates=candidates,
                contribution=contribution,
                fingerprint_sha256=base_award_fingerprint(
                    transaction, lifecycle, evidence, candidates
                ),
            )

        # Commit only after all operations that can fail have passed.
        self._transaction_ids.add(txid)
        self._total += 1

        if lifecycle.kind is AwardActionKind.MODIFICATION:
            self._modifications += 1
            return
        if lifecycle.kind is AwardActionKind.UNKNOWN:
            self._unknown += 1
            return

        assert reference is not None
        self._base_awards[transaction.award_id] = reference
        self._base += 1
        for key in keys:
            accumulator = self._markets.get(key)
            if accumulator is None:
                accumulator = _Accumulator()
                self._markets[key] = accumulator
            accumulator.add(reference.contribution)

    def observe_many(
        self, transactions: Iterable[ProcurementTransaction]
    ) -> "CompetitionReferenceIndex":
        for transaction in transactions:
            self.observe(transaction)
        return self

    def snapshot(self) -> CompetitionReferenceSnapshot:
        return CompetitionReferenceSnapshot(
            plan=self.plan,
            total_transactions_seen=self._total,
            observed_base_awards=self._base,
            observed_modifications=self._modifications,
            lifecycle_unknown_transactions=self._unknown,
            markets={
                key: accumulator.freeze(key)
                for key, accumulator in self._markets.items()
            },
            base_awards=dict(self._base_awards),
        )


def federal_competition_context_plan(
    *,
    time_scope: TimeScope = TimeScope.FEDERAL_FISCAL_YEAR,
    include_broader_naics_fallbacks: bool = True,
) -> PeerGroupPlan:
    """Competition peer hierarchy with controlled broadening and no global bucket."""

    time_scope = TimeScope(time_scope)
    levels = [
        PeerGroupLevel(
            "competition_subtier_psc_exact",
            agency_scope=AgencyScope.SUBTIER,
            category_scope=CategoryScope.PSC_EXACT,
            time_scope=time_scope,
            include_award_type=True,
        ),
        PeerGroupLevel(
            "competition_agency_psc_exact",
            agency_scope=AgencyScope.TOP_LEVEL,
            category_scope=CategoryScope.PSC_EXACT,
            time_scope=time_scope,
        ),
        PeerGroupLevel(
            "competition_agency_naics_6",
            agency_scope=AgencyScope.TOP_LEVEL,
            category_scope=CategoryScope.NAICS_6,
            time_scope=time_scope,
        ),
    ]
    if include_broader_naics_fallbacks:
        levels.extend(
            (
                PeerGroupLevel(
                    "competition_agency_naics_4",
                    agency_scope=AgencyScope.TOP_LEVEL,
                    category_scope=CategoryScope.NAICS_4,
                    time_scope=time_scope,
                ),
                PeerGroupLevel(
                    "competition_agency_naics_2",
                    agency_scope=AgencyScope.TOP_LEVEL,
                    category_scope=CategoryScope.NAICS_2,
                    time_scope=time_scope,
                ),
            )
        )
    levels.extend(
        (
            PeerGroupLevel(
                "competition_psc_exact",
                category_scope=CategoryScope.PSC_EXACT,
                time_scope=time_scope,
            ),
            PeerGroupLevel(
                "competition_naics_6",
                category_scope=CategoryScope.NAICS_6,
                time_scope=time_scope,
            ),
        )
    )
    if include_broader_naics_fallbacks:
        levels.extend(
            (
                PeerGroupLevel(
                    "competition_naics_4",
                    category_scope=CategoryScope.NAICS_4,
                    time_scope=time_scope,
                ),
                PeerGroupLevel(
                    "competition_naics_2",
                    category_scope=CategoryScope.NAICS_2,
                    time_scope=time_scope,
                ),
            )
        )
    return PeerGroupPlan("federal-contract-competition-context", tuple(levels))


def competition_contribution(
    evidence: CompetitionEvidence,
) -> CompetitionContribution:
    if not isinstance(evidence, CompetitionEvidence):
        raise TypeError("evidence must be CompetitionEvidence")
    procedure_known = (
        evidence.solicitation_procedure_kind
        is not SolicitationProcedureKind.UNKNOWN
    )
    return CompetitionContribution(
        process_competitive=evidence.reported_process_competitive,
        offer_outcome_kind=evidence.offer_outcome_kind,
        solicitation_procedure_known=procedure_known,
        only_one_source_solicited=(
            evidence.solicitation_procedure_kind
            is SolicitationProcedureKind.ONLY_ONE_SOURCE
        ),
        noncompetition_authority_reported=(
            evidence.other_than_full_open_authority_kind
            is not OtherThanFullOpenAuthorityKind.NONE_REPORTED
        ),
        full_open_after_exclusion=(
            evidence.extent_kind
            is CompetitionExtentKind.FULL_AND_OPEN_AFTER_EXCLUSION
        ),
        has_conflict=bool(evidence.evidence_conflicts),
        has_missing_core_fields=bool(evidence.missing_core_fields),
    )


def contribution_counts(
    contribution: CompetitionContribution,
) -> dict[str, int]:
    if not isinstance(contribution, CompetitionContribution):
        raise TypeError("contribution must be CompetitionContribution")
    process_known = contribution.process_competitive is not None
    offer = contribution.offer_outcome_kind
    return {
        "process_known_count": int(process_known),
        "competitive_process_count": int(
            contribution.process_competitive is True
        ),
        "noncompetitive_process_count": int(
            contribution.process_competitive is False
        ),
        "offers_known_count": int(offer is not OfferOutcomeKind.UNKNOWN),
        "single_offer_count": int(offer is OfferOutcomeKind.SINGLE_OFFER),
        "multiple_offer_count": int(offer is OfferOutcomeKind.MULTIPLE_OFFERS),
        "zero_offer_count": int(offer is OfferOutcomeKind.ZERO_REPORTED),
        "procedure_known_count": int(contribution.solicitation_procedure_known),
        "only_one_source_solicited_count": int(
            contribution.only_one_source_solicited
        ),
        "noncompetition_authority_reported_count": int(
            contribution.noncompetition_authority_reported
        ),
        "full_open_after_exclusion_count": int(
            contribution.full_open_after_exclusion
        ),
        "conflict_award_count": int(contribution.has_conflict),
        "missing_core_field_award_count": int(
            contribution.has_missing_core_fields
        ),
    }


def base_award_fingerprint(
    transaction: ProcurementTransaction,
    lifecycle: AwardActionEvidence,
    evidence: CompetitionEvidence,
    candidates: tuple[PeerGroupCandidate, ...],
) -> str:
    if not isinstance(transaction, ProcurementTransaction):
        raise TypeError("transaction must be ProcurementTransaction")
    if not isinstance(lifecycle, AwardActionEvidence):
        raise TypeError("lifecycle must be AwardActionEvidence")
    if lifecycle.kind is not AwardActionKind.BASE_AWARD:
        raise CompetitionReferenceError(
            "base-award fingerprint requires BASE_AWARD lifecycle evidence"
        )
    if not isinstance(evidence, CompetitionEvidence):
        raise TypeError("evidence must be CompetitionEvidence")
    return _digest(
        {
            "transaction_id": transaction.transaction_id,
            "award_id": transaction.award_id,
            "action_date": transaction.action_date.isoformat(),
            "lifecycle_sha256": lifecycle.sha256_hex,
            "competition_evidence_sha256": evidence.evidence_sha256,
            "candidates": [
                {
                    "level": candidate.level_name,
                    "key": None
                    if candidate.key is None
                    else candidate.key.sha256_hex,
                    "unavailable_reasons": list(candidate.unavailable_reasons),
                }
                for candidate in candidates
            ],
        }
    )


def _rate(numerator: int, denominator: int) -> Decimal | None:
    if denominator == 0:
        return None
    return Decimal(numerator) / Decimal(denominator)


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _validate_digest(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise CompetitionReferenceError(f"{name} must be a SHA-256 hex digest")
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
