"""Award-change reference populations for ProcureLens.

Builds fair award-level comparison markets for observed contract-change activity.
Peer markets are anchored to the BASE_AWARD formation transaction, not later
modification dates. Awards with no observed base action or multiple observed
base actions remain explicit quality exclusions rather than being guessed into a
market.

The reference records observable follow-up from base action to the dataset's
observed end date because recurrent modification counts depend on exposure time.
It stores descriptive evidence only: no anomaly cutoff, risk score, smoothing
prior, or misconduct conclusion lives here.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any

from procurelens.domain.transaction import ProcurementTransaction
from procurelens.features.award_change_activity import (
    AwardChangeActivity,
    AwardChangeIndex,
    AwardChangePolicy,
)
from procurelens.features.award_lifecycle import AwardActionKind, classify_award_action
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


class AwardChangeReferenceError(ValueError):
    """Raised when award-change reference evidence or state is inconsistent."""


@dataclass(frozen=True, slots=True)
class AwardChangeReferencePolicy:
    """Resource budgets only; no anomaly or risk thresholds live here."""

    activity_policy: AwardChangePolicy = field(default_factory=AwardChangePolicy)
    max_base_context_awards: int | None = None
    max_distinct_markets: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.activity_policy, AwardChangePolicy):
            raise TypeError("activity_policy must be AwardChangePolicy")
        for name in ("max_base_context_awards", "max_distinct_markets"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise AwardChangeReferenceError(
                    f"{name} must be a positive integer or None"
                )


@dataclass(frozen=True, slots=True)
class AwardChangeBaseReference:
    """Minimum formation evidence needed to place one award into peer markets."""

    award_id: str
    transaction_id: str
    action_date: date
    award_total_obligation: Decimal | None
    candidates: tuple[PeerGroupCandidate, ...]
    fingerprint_sha256: str

    def __post_init__(self) -> None:
        award_id, txid = self.award_id.strip(), self.transaction_id.strip()
        if not award_id or not txid:
            raise AwardChangeReferenceError(
                "base-reference award_id and transaction_id must not be blank"
            )
        object.__setattr__(self, "award_id", award_id)
        object.__setattr__(self, "transaction_id", txid)
        if self.award_total_obligation is not None and (
            not isinstance(self.award_total_obligation, Decimal)
            or not self.award_total_obligation.is_finite()
        ):
            raise AwardChangeReferenceError(
                "award_total_obligation must be finite Decimal or None"
            )
        candidates = tuple(self.candidates)
        if not candidates:
            raise AwardChangeReferenceError("base reference requires peer candidates")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self,
            "fingerprint_sha256",
            _validate_digest(self.fingerprint_sha256, "fingerprint_sha256"),
        )


@dataclass(frozen=True, slots=True)
class AwardChangeObservation:
    """One award's change activity expressed against observable follow-up."""

    award_id: str
    base_transaction_id: str
    base_action_date: date
    observed_through_date: date
    observable_followup_days: int
    activity_evidence_sha256: str

    modification_action_count: int
    distinct_modification_number_count: int
    maximum_transactions_on_one_modification_number: int
    lifecycle_unknown_action_count: int
    zero_modification_obligation_count: int

    net_modification_obligation: Decimal
    absolute_modification_obligation_activity: Decimal
    positive_modification_obligation: Decimal
    deobligation_magnitude: Decimal

    base_award_obligation_magnitude: Decimal | None

    def __post_init__(self) -> None:
        for name in ("award_id", "base_transaction_id"):
            text = getattr(self, name).strip()
            if not text:
                raise AwardChangeReferenceError(f"{name} must not be blank")
            object.__setattr__(self, name, text)
        if self.base_action_date > self.observed_through_date:
            raise AwardChangeReferenceError(
                "base_action_date cannot be after observed_through_date"
            )
        expected_days = (self.observed_through_date - self.base_action_date).days + 1
        if (
            isinstance(self.observable_followup_days, bool)
            or not isinstance(self.observable_followup_days, int)
            or self.observable_followup_days != expected_days
            or self.observable_followup_days < 1
        ):
            raise AwardChangeReferenceError("observable_followup_days is inconsistent")
        object.__setattr__(
            self,
            "activity_evidence_sha256",
            _validate_digest(
                self.activity_evidence_sha256, "activity_evidence_sha256"
            ),
        )

        for name in (
            "modification_action_count",
            "distinct_modification_number_count",
            "maximum_transactions_on_one_modification_number",
            "lifecycle_unknown_action_count",
            "zero_modification_obligation_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AwardChangeReferenceError(
                    f"{name} must be a non-negative integer"
                )
        if self.distinct_modification_number_count > self.modification_action_count:
            raise AwardChangeReferenceError(
                "distinct modification numbers cannot exceed modification actions"
            )
        if (
            self.modification_action_count == 0
            and self.maximum_transactions_on_one_modification_number != 0
        ):
            raise AwardChangeReferenceError(
                "award without modifications cannot have repeated modification activity"
            )

        money = (
            self.net_modification_obligation,
            self.absolute_modification_obligation_activity,
            self.positive_modification_obligation,
            self.deobligation_magnitude,
        )
        if any(
            not isinstance(value, Decimal) or not value.is_finite()
            for value in money
        ):
            raise AwardChangeReferenceError(
                "modification obligation evidence must be finite Decimal"
            )
        if (
            self.absolute_modification_obligation_activity < 0
            or self.positive_modification_obligation < 0
            or self.deobligation_magnitude < 0
        ):
            raise AwardChangeReferenceError(
                "modification obligation magnitudes cannot be negative"
            )
        if (
            self.absolute_modification_obligation_activity
            != self.positive_modification_obligation + self.deobligation_magnitude
        ):
            raise AwardChangeReferenceError(
                "absolute modification activity is inconsistent"
            )
        if (
            self.net_modification_obligation
            != self.positive_modification_obligation - self.deobligation_magnitude
        ):
            raise AwardChangeReferenceError(
                "net modification obligation is inconsistent"
            )

        base = self.base_award_obligation_magnitude
        if base is not None and (
            not isinstance(base, Decimal) or not base.is_finite() or base <= 0
        ):
            raise AwardChangeReferenceError(
                "base_award_obligation_magnitude must be positive Decimal or None"
            )

    @property
    def has_modifications(self) -> bool:
        return self.modification_action_count > 0

    @property
    def has_deobligation(self) -> bool:
        return self.deobligation_magnitude > 0

    @property
    def has_positive_modification(self) -> bool:
        return self.positive_modification_obligation > 0

    @property
    def modification_actions_per_observed_day(self) -> Decimal:
        return Decimal(self.modification_action_count) / Decimal(
            self.observable_followup_days
        )

    @property
    def gross_modification_activity_per_observed_day(self) -> Decimal:
        return self.absolute_modification_obligation_activity / Decimal(
            self.observable_followup_days
        )

    @property
    def gross_modification_to_base_obligation(self) -> Decimal | None:
        base = self.base_award_obligation_magnitude
        if base is None:
            return None
        return self.absolute_modification_obligation_activity / base

    @property
    def net_modification_to_base_obligation(self) -> Decimal | None:
        base = self.base_award_obligation_magnitude
        if base is None:
            return None
        return self.net_modification_obligation / base

    @property
    def deobligation_to_base_obligation(self) -> Decimal | None:
        base = self.base_award_obligation_magnitude
        if base is None:
            return None
        return self.deobligation_magnitude / base

    @property
    def evidence_sha256(self) -> str:
        return _digest(self.as_dict(include_evidence_sha=False))

    def as_dict(self, *, include_evidence_sha: bool = True) -> dict[str, Any]:
        result = {
            "award_id": self.award_id,
            "base_transaction_id": self.base_transaction_id,
            "base_action_date": self.base_action_date.isoformat(),
            "observed_through_date": self.observed_through_date.isoformat(),
            "observable_followup_days": self.observable_followup_days,
            "activity_evidence_sha256": self.activity_evidence_sha256,
            "modification_action_count": self.modification_action_count,
            "distinct_modification_number_count":
                self.distinct_modification_number_count,
            "maximum_transactions_on_one_modification_number":
                self.maximum_transactions_on_one_modification_number,
            "lifecycle_unknown_action_count": self.lifecycle_unknown_action_count,
            "zero_modification_obligation_count":
                self.zero_modification_obligation_count,
            "net_modification_obligation": str(self.net_modification_obligation),
            "absolute_modification_obligation_activity":
                str(self.absolute_modification_obligation_activity),
            "positive_modification_obligation":
                str(self.positive_modification_obligation),
            "deobligation_magnitude": str(self.deobligation_magnitude),
            "base_award_obligation_magnitude":
                None
                if self.base_award_obligation_magnitude is None
                else str(self.base_award_obligation_magnitude),
            "has_modifications": self.has_modifications,
            "has_deobligation": self.has_deobligation,
            "has_positive_modification": self.has_positive_modification,
            "modification_actions_per_observed_day":
                str(self.modification_actions_per_observed_day),
            "gross_modification_activity_per_observed_day":
                str(self.gross_modification_activity_per_observed_day),
            "gross_modification_to_base_obligation":
                _decimal_text(self.gross_modification_to_base_obligation),
            "net_modification_to_base_obligation":
                _decimal_text(self.net_modification_to_base_obligation),
            "deobligation_to_base_obligation":
                _decimal_text(self.deobligation_to_base_obligation),
        }
        if include_evidence_sha:
            result["evidence_sha256"] = self.evidence_sha256
        return result


@dataclass(frozen=True, slots=True)
class AwardChangeMarketReference:
    """Award IDs contributing to one immutable formation-context market."""

    key: PeerGroupKey
    award_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.key, PeerGroupKey):
            raise TypeError("key must be PeerGroupKey")
        award_ids = tuple(item.strip() for item in self.award_ids)
        if not award_ids or any(not item for item in award_ids):
            raise AwardChangeReferenceError(
                "market reference requires non-blank award ids"
            )
        if tuple(sorted(award_ids)) != award_ids:
            raise AwardChangeReferenceError(
                "market award_ids must be deterministically sorted"
            )
        if len(award_ids) != len(set(award_ids)):
            raise AwardChangeReferenceError(
                "market award_ids must not contain duplicates"
            )
        object.__setattr__(self, "award_ids", award_ids)

    @property
    def award_count(self) -> int:
        return len(self.award_ids)

    @property
    def sha256_hex(self) -> str:
        return _digest(
            {
                "market_key_sha256": self.key.sha256_hex,
                "award_ids": self.award_ids,
            }
        )


@dataclass(frozen=True, slots=True)
class AwardChangeReferenceSnapshot:
    """Frozen award-change comparison universe."""

    plan: PeerGroupPlan
    activity_population_sha256: str
    observed_through_date: date | None
    total_awards: int
    eligible_awards: int
    awards_without_observed_base_action: int
    awards_with_multiple_observed_base_actions: int
    observations: Mapping[str, AwardChangeObservation]
    base_references: Mapping[str, AwardChangeBaseReference]
    markets: Mapping[PeerGroupKey, AwardChangeMarketReference]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, PeerGroupPlan):
            raise TypeError("plan must be PeerGroupPlan")
        object.__setattr__(
            self,
            "activity_population_sha256",
            _validate_digest(
                self.activity_population_sha256, "activity_population_sha256"
            ),
        )
        for name in (
            "total_awards",
            "eligible_awards",
            "awards_without_observed_base_action",
            "awards_with_multiple_observed_base_actions",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AwardChangeReferenceError(
                    f"{name} must be a non-negative integer"
                )
        if (
            self.eligible_awards
            + self.awards_without_observed_base_action
            + self.awards_with_multiple_observed_base_actions
            != self.total_awards
        ):
            raise AwardChangeReferenceError(
                "award eligibility counts do not sum to total_awards"
            )

        observations = dict(self.observations)
        if len(observations) != self.eligible_awards:
            raise AwardChangeReferenceError(
                "eligible_awards differs from observation mapping"
            )
        if any(key != obs.award_id for key, obs in observations.items()):
            raise AwardChangeReferenceError(
                "observation mapping key differs from award_id"
            )
        object.__setattr__(
            self, "observations", MappingProxyType(observations)
        )

        bases = dict(self.base_references)
        if any(key != ref.award_id for key, ref in bases.items()):
            raise AwardChangeReferenceError(
                "base-reference mapping key differs from award_id"
            )
        if any(award_id not in bases for award_id in observations):
            raise AwardChangeReferenceError(
                "eligible observation lacks base reference"
            )
        object.__setattr__(
            self, "base_references", MappingProxyType(bases)
        )

        markets = dict(self.markets)
        for key, market in markets.items():
            if key != market.key:
                raise AwardChangeReferenceError(
                    "market mapping key differs from market key"
                )
            if any(award_id not in observations for award_id in market.award_ids):
                raise AwardChangeReferenceError(
                    "market references an ineligible award"
                )
        object.__setattr__(self, "markets", MappingProxyType(markets))

    @property
    def market_count(self) -> int:
        return len(self.markets)

    @property
    def snapshot_sha256(self) -> str:
        return _digest(
            {
                "plan_sha256": self.plan.sha256_hex,
                "activity_population_sha256": self.activity_population_sha256,
                "observed_through_date":
                    None
                    if self.observed_through_date is None
                    else self.observed_through_date.isoformat(),
                "total_awards": self.total_awards,
                "eligible_awards": self.eligible_awards,
                "without_base": self.awards_without_observed_base_action,
                "multiple_base": self.awards_with_multiple_observed_base_actions,
                "observations": [
                    (award_id, self.observations[award_id].evidence_sha256)
                    for award_id in sorted(self.observations)
                ],
                "markets": [
                    (key.sha256_hex, self.markets[key].sha256_hex)
                    for key in sorted(
                        self.markets, key=lambda item: item.sha256_hex
                    )
                ],
            }
        )

    def get_observation(
        self, award_id: str
    ) -> AwardChangeObservation | None:
        return self.observations.get(award_id)

    def get_base_reference(
        self, award_id: str
    ) -> AwardChangeBaseReference | None:
        return self.base_references.get(award_id)

    def get_market(
        self, key: PeerGroupKey
    ) -> AwardChangeMarketReference | None:
        return self.markets.get(key)


class AwardChangeReferenceIndex:
    """One-pass activity + formation-context builder."""

    def __init__(
        self,
        *,
        plan: PeerGroupPlan | None = None,
        policy: AwardChangeReferencePolicy | None = None,
    ) -> None:
        self.plan = plan or federal_award_change_reference_plan()
        if not isinstance(self.plan, PeerGroupPlan):
            raise TypeError("plan must be PeerGroupPlan")
        self.policy = policy or AwardChangeReferencePolicy()
        if not isinstance(self.policy, AwardChangeReferencePolicy):
            raise TypeError("policy must be AwardChangeReferencePolicy")
        self._activity = AwardChangeIndex(
            policy=self.policy.activity_policy
        )
        self._base_references: dict[str, AwardChangeBaseReference] = {}

    def observe(self, transaction: ProcurementTransaction) -> None:
        if not isinstance(transaction, ProcurementTransaction):
            raise TypeError("transaction must be ProcurementTransaction")

        lifecycle = classify_award_action(transaction)
        base_reference: AwardChangeBaseReference | None = None
        if (
            lifecycle.kind is AwardActionKind.BASE_AWARD
            and transaction.award_id not in self._base_references
        ):
            limit = self.policy.max_base_context_awards
            if (
                limit is not None
                and len(self._base_references) >= limit
            ):
                raise AwardChangeReferenceError(
                    f"base context index exceeds max_base_context_awards={limit}"
                )
            candidates = peer_group_candidates(transaction, self.plan)
            base_reference = AwardChangeBaseReference(
                award_id=transaction.award_id,
                transaction_id=transaction.transaction_id,
                action_date=transaction.action_date,
                award_total_obligation=transaction.award_total_obligation,
                candidates=candidates,
                fingerprint_sha256=_base_reference_fingerprint(
                    transaction, lifecycle.sha256_hex, candidates
                ),
            )

        # Delegate only after our own preflight; store base context only if
        # activity aggregation also succeeds.
        self._activity.observe(transaction)
        if base_reference is not None:
            self._base_references[transaction.award_id] = base_reference

    def observe_many(
        self, transactions: Iterable[ProcurementTransaction]
    ) -> "AwardChangeReferenceIndex":
        for transaction in transactions:
            self.observe(transaction)
        return self

    def snapshot(self) -> AwardChangeReferenceSnapshot:
        activity_snapshot = self._activity.snapshot()
        observed_end = activity_snapshot.observed_end_date

        observations: dict[str, AwardChangeObservation] = {}
        markets: dict[PeerGroupKey, list[str]] = {}
        without_base = 0
        multiple_base = 0

        for award_id in sorted(activity_snapshot.activities):
            activity = activity_snapshot.activities[award_id]
            if activity.base_award_action_count == 0:
                without_base += 1
                continue
            if activity.base_award_action_count != 1:
                multiple_base += 1
                continue

            base = self._base_references.get(award_id)
            if base is None:
                raise AwardChangeReferenceError(
                    "single-base award lacks stored formation context"
                )
            if observed_end is None:
                raise AwardChangeReferenceError(
                    "non-empty activity population lacks observed end date"
                )
            observation = _award_observation(activity, base, observed_end)
            observations[award_id] = observation
            for candidate in base.candidates:
                if candidate.key is not None:
                    markets.setdefault(candidate.key, []).append(award_id)

        limit = self.policy.max_distinct_markets
        if limit is not None and len(markets) > limit:
            raise AwardChangeReferenceError(
                f"market index exceeds max_distinct_markets={limit}"
            )

        frozen_markets = {
            key: AwardChangeMarketReference(
                key=key, award_ids=tuple(sorted(award_ids))
            )
            for key, award_ids in markets.items()
        }
        return AwardChangeReferenceSnapshot(
            plan=self.plan,
            activity_population_sha256=
                activity_snapshot.transaction_population_sha256,
            observed_through_date=observed_end,
            total_awards=activity_snapshot.award_count,
            eligible_awards=len(observations),
            awards_without_observed_base_action=without_base,
            awards_with_multiple_observed_base_actions=multiple_base,
            observations=observations,
            base_references=dict(self._base_references),
            markets=frozen_markets,
        )


def federal_award_change_reference_plan(
    *,
    time_scope: TimeScope = TimeScope.FEDERAL_FISCAL_YEAR,
    include_broader_naics_fallbacks: bool = True,
) -> PeerGroupPlan:
    """Formation-context hierarchy for comparing award change activity."""

    time_scope = TimeScope(time_scope)
    levels = [
        PeerGroupLevel(
            "award_change_subtier_psc_exact",
            agency_scope=AgencyScope.SUBTIER,
            category_scope=CategoryScope.PSC_EXACT,
            time_scope=time_scope,
            include_award_type=True,
        ),
        PeerGroupLevel(
            "award_change_agency_psc_exact",
            agency_scope=AgencyScope.TOP_LEVEL,
            category_scope=CategoryScope.PSC_EXACT,
            time_scope=time_scope,
        ),
        PeerGroupLevel(
            "award_change_agency_naics_6",
            agency_scope=AgencyScope.TOP_LEVEL,
            category_scope=CategoryScope.NAICS_6,
            time_scope=time_scope,
        ),
    ]
    if include_broader_naics_fallbacks:
        levels.extend(
            (
                PeerGroupLevel(
                    "award_change_agency_naics_4",
                    agency_scope=AgencyScope.TOP_LEVEL,
                    category_scope=CategoryScope.NAICS_4,
                    time_scope=time_scope,
                ),
                PeerGroupLevel(
                    "award_change_agency_naics_2",
                    agency_scope=AgencyScope.TOP_LEVEL,
                    category_scope=CategoryScope.NAICS_2,
                    time_scope=time_scope,
                ),
            )
        )
    levels.extend(
        (
            PeerGroupLevel(
                "award_change_psc_exact",
                category_scope=CategoryScope.PSC_EXACT,
                time_scope=time_scope,
            ),
            PeerGroupLevel(
                "award_change_naics_6",
                category_scope=CategoryScope.NAICS_6,
                time_scope=time_scope,
            ),
        )
    )
    if include_broader_naics_fallbacks:
        levels.extend(
            (
                PeerGroupLevel(
                    "award_change_naics_4",
                    category_scope=CategoryScope.NAICS_4,
                    time_scope=time_scope,
                ),
                PeerGroupLevel(
                    "award_change_naics_2",
                    category_scope=CategoryScope.NAICS_2,
                    time_scope=time_scope,
                ),
            )
        )
    return PeerGroupPlan(
        "federal-contract-award-change-reference", tuple(levels)
    )


def _award_observation(
    activity: AwardChangeActivity,
    base: AwardChangeBaseReference,
    observed_end: date,
) -> AwardChangeObservation:
    if activity.award_id != base.award_id:
        raise AwardChangeReferenceError(
            "activity and base reference award ids differ"
        )
    base_total = base.award_total_obligation
    magnitude = None
    if base_total is not None and base_total != 0:
        magnitude = abs(base_total)

    followup_days = (observed_end - base.action_date).days + 1
    if followup_days < 1:
        raise AwardChangeReferenceError(
            "observed end date precedes base action date"
        )
    return AwardChangeObservation(
        award_id=activity.award_id,
        base_transaction_id=base.transaction_id,
        base_action_date=base.action_date,
        observed_through_date=observed_end,
        observable_followup_days=followup_days,
        activity_evidence_sha256=activity.evidence_sha256,
        modification_action_count=activity.modification_action_count,
        distinct_modification_number_count=
            activity.distinct_modification_number_count,
        maximum_transactions_on_one_modification_number=
            activity.maximum_transactions_on_one_modification_number,
        lifecycle_unknown_action_count=
            activity.lifecycle_unknown_action_count,
        zero_modification_obligation_count=
            activity.zero_modification_obligation_count,
        net_modification_obligation=activity.net_modification_obligation,
        absolute_modification_obligation_activity=
            activity.absolute_modification_obligation_activity,
        positive_modification_obligation=
            activity.positive_modification_obligation,
        deobligation_magnitude=activity.deobligation_magnitude,
        base_award_obligation_magnitude=magnitude,
    )


def _base_reference_fingerprint(
    transaction: ProcurementTransaction,
    lifecycle_sha256: str,
    candidates: tuple[PeerGroupCandidate, ...],
) -> str:
    return _digest(
        {
            "transaction_id": transaction.transaction_id,
            "award_id": transaction.award_id,
            "action_date": transaction.action_date.isoformat(),
            "award_total_obligation":
                None
                if transaction.award_total_obligation is None
                else str(transaction.award_total_obligation),
            "lifecycle_sha256": lifecycle_sha256,
            "candidates": [
                {
                    "level": candidate.level_name,
                    "key":
                        None
                        if candidate.key is None
                        else candidate.key.sha256_hex,
                    "unavailable_reasons":
                        list(candidate.unavailable_reasons),
                }
                for candidate in candidates
            ],
        }
    )


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _validate_digest(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise AwardChangeReferenceError(
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
