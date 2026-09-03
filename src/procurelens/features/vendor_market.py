"""Observed new-award vendor markets for ProcureLens.

Builds count-based market structure from observed BASE_AWARD actions inside
explicit procurement comparison groups. Award data reveals winners, not the
full population of bidders or firms that could have competed, so this module
uses the precise term "observed winning vendors".

Modifications and lifecycle-unknown actions never contribute to new-award
frequency. The module does not score risk, infer fraud/collusion, or hide
minimum-support thresholds inside market construction.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any

from procurelens.domain.transaction import ProcurementTransaction
from procurelens.features.award_lifecycle import (
    AwardActionKind,
    classify_award_action,
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
from procurelens.features.vendor_identity import (
    VendorIdentity,
    VendorIdentityScope,
    resolve_vendor_identity,
)


class VendorMarketError(ValueError):
    """Raised when observed vendor-market evidence is inconsistent."""


@dataclass(frozen=True, slots=True)
class VendorMarketPolicy:
    """Optional memory budgets only; never anomaly/risk thresholds."""

    max_transaction_ids: int | None = None
    max_base_award_ids: int | None = None
    max_distinct_markets: int | None = None
    max_vendor_market_pairs: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_transaction_ids",
            "max_base_award_ids",
            "max_distinct_markets",
            "max_vendor_market_pairs",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise VendorMarketError(
                    f"{name} must be a positive integer or None"
                )


@dataclass(frozen=True, slots=True)
class ObservedVendorMarket:
    """Count-based structure for one observed new-award comparison group."""

    key: PeerGroupKey
    observed_new_award_count: int
    new_awards_with_vendor_identity: int
    new_awards_without_vendor_identity: int
    awards_by_winning_vendor: Mapping[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.key, PeerGroupKey):
            raise TypeError("key must be PeerGroupKey")
        for name in (
            "observed_new_award_count",
            "new_awards_with_vendor_identity",
            "new_awards_without_vendor_identity",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise VendorMarketError(
                    f"{name} must be a non-negative integer"
                )
        if (
            self.new_awards_with_vendor_identity
            + self.new_awards_without_vendor_identity
            != self.observed_new_award_count
        ):
            raise VendorMarketError(
                "vendor identity coverage does not sum to observed new awards"
            )

        frozen = dict(self.awards_by_winning_vendor)
        for vendor_key, count in frozen.items():
            if not isinstance(vendor_key, str) or not vendor_key:
                raise VendorMarketError(
                    "winning-vendor keys must be non-empty strings"
                )
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 1
            ):
                raise VendorMarketError(
                    "winning-vendor award counts must be positive integers"
                )
        if sum(frozen.values()) != self.new_awards_with_vendor_identity:
            raise VendorMarketError(
                "winning-vendor counts do not match identified new awards"
            )
        object.__setattr__(
            self, "awards_by_winning_vendor", MappingProxyType(frozen)
        )

    @property
    def observed_winning_vendor_count(self) -> int:
        return len(self.awards_by_winning_vendor)

    @property
    def vendor_identity_coverage(self) -> Decimal | None:
        if self.observed_new_award_count == 0:
            return None
        return (
            Decimal(self.new_awards_with_vendor_identity)
            / Decimal(self.observed_new_award_count)
        )

    @property
    def award_count_hhi(self) -> Decimal | None:
        """HHI across identified winners using new-award count shares."""

        total = self.new_awards_with_vendor_identity
        if total == 0:
            return None
        denominator = Decimal(total)
        return sum(
            (Decimal(count) / denominator) ** 2
            for count in self.awards_by_winning_vendor.values()
        )

    @property
    def largest_winner_award_share(self) -> Decimal | None:
        total = self.new_awards_with_vendor_identity
        if total == 0:
            return None
        return (
            Decimal(max(self.awards_by_winning_vendor.values()))
            / Decimal(total)
        )

    def without_one_award(
        self, vendor_key: str | None
    ) -> "ObservedVendorMarket":
        """Return leave-one-out evidence for one indexed base award."""

        if self.observed_new_award_count < 1:
            raise VendorMarketError(
                "cannot remove an award from an empty observed market"
            )
        counts = dict(self.awards_by_winning_vendor)
        identified = self.new_awards_with_vendor_identity
        unidentified = self.new_awards_without_vendor_identity

        if vendor_key is None:
            if unidentified < 1:
                raise VendorMarketError(
                    "indexed unidentified base award is absent from market"
                )
            unidentified -= 1
        else:
            current = counts.get(vendor_key)
            if current is None:
                raise VendorMarketError(
                    "indexed base-award vendor is absent from market"
                )
            identified -= 1
            if current == 1:
                del counts[vendor_key]
            else:
                counts[vendor_key] = current - 1

        return ObservedVendorMarket(
            key=self.key,
            observed_new_award_count=self.observed_new_award_count - 1,
            new_awards_with_vendor_identity=identified,
            new_awards_without_vendor_identity=unidentified,
            awards_by_winning_vendor=counts,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "market_level": self.key.level_name,
            "market_key_sha256": self.key.sha256_hex,
            "observed_new_award_count": self.observed_new_award_count,
            "new_awards_with_vendor_identity": self.new_awards_with_vendor_identity,
            "new_awards_without_vendor_identity": self.new_awards_without_vendor_identity,
            "vendor_identity_coverage": (
                None
                if self.vendor_identity_coverage is None
                else str(self.vendor_identity_coverage)
            ),
            "observed_winning_vendor_count": self.observed_winning_vendor_count,
            "award_count_hhi": (
                None if self.award_count_hhi is None else str(self.award_count_hhi)
            ),
            "largest_winner_award_share": (
                None
                if self.largest_winner_award_share is None
                else str(self.largest_winner_award_share)
            ),
        }


@dataclass(frozen=True, slots=True)
class VendorMarketSnapshot:
    """Immutable observed new-award markets for one vendor identity scope."""

    plan: PeerGroupPlan
    scope: VendorIdentityScope
    total_transactions_seen: int
    observed_base_awards: int
    observed_modifications: int
    lifecycle_unknown_transactions: int
    markets: Mapping[PeerGroupKey, ObservedVendorMarket]
    base_award_fingerprints: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, PeerGroupPlan):
            raise TypeError("plan must be PeerGroupPlan")
        object.__setattr__(self, "scope", VendorIdentityScope(self.scope))

        for name in (
            "total_transactions_seen",
            "observed_base_awards",
            "observed_modifications",
            "lifecycle_unknown_transactions",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise VendorMarketError(
                    f"{name} must be a non-negative integer"
                )
        if (
            self.observed_base_awards
            + self.observed_modifications
            + self.lifecycle_unknown_transactions
            != self.total_transactions_seen
        ):
            raise VendorMarketError(
                "lifecycle counts do not sum to total transactions"
            )

        frozen_markets = dict(self.markets)
        if any(key != market.key for key, market in frozen_markets.items()):
            raise VendorMarketError(
                "market mapping key differs from market evidence key"
            )
        object.__setattr__(
            self, "markets", MappingProxyType(frozen_markets)
        )

        frozen_awards = dict(self.base_award_fingerprints)
        if len(frozen_awards) != self.observed_base_awards:
            raise VendorMarketError(
                "base-award fingerprint count differs from observed base awards"
            )
        object.__setattr__(
            self,
            "base_award_fingerprints",
            MappingProxyType(frozen_awards),
        )

    @property
    def market_count(self) -> int:
        return len(self.markets)

    def get(self, key: PeerGroupKey) -> ObservedVendorMarket | None:
        return self.markets.get(key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_name": self.plan.name,
            "plan_sha256": self.plan.sha256_hex,
            "scope": self.scope.value,
            "total_transactions_seen": self.total_transactions_seen,
            "observed_base_awards": self.observed_base_awards,
            "observed_modifications": self.observed_modifications,
            "lifecycle_unknown_transactions": self.lifecycle_unknown_transactions,
            "market_count": self.market_count,
            "markets": [
                self.markets[key].as_dict()
                for key in sorted(
                    self.markets,
                    key=lambda item: (item.level_name, item.sha256_hex),
                )
            ],
        }


@dataclass(slots=True)
class _MarketAccumulator:
    observed_new_award_count: int
    identified_new_awards: int
    unidentified_new_awards: int
    vendor_counts: Counter[str]


class VendorMarketIndex:
    """One-pass builder; transaction objects themselves are never retained."""

    def __init__(
        self,
        *,
        plan: PeerGroupPlan | None = None,
        scope: VendorIdentityScope = VendorIdentityScope.ENTITY,
        policy: VendorMarketPolicy | None = None,
    ) -> None:
        self.plan = plan or federal_vendor_market_plan()
        if not isinstance(self.plan, PeerGroupPlan):
            raise TypeError("plan must be PeerGroupPlan")
        self.scope = VendorIdentityScope(scope)
        self.policy = policy or VendorMarketPolicy()

        self._markets: dict[PeerGroupKey, _MarketAccumulator] = {}
        self._transaction_ids: set[str] = set()
        self._base_awards: dict[str, str] = {}
        self._vendor_market_pairs: set[tuple[PeerGroupKey, str]] = set()
        self._total = 0
        self._base = 0
        self._modifications = 0
        self._unknown = 0

    def observe(self, transaction: ProcurementTransaction) -> None:
        """Atomically add one transaction after every configured check passes."""

        if not isinstance(transaction, ProcurementTransaction):
            raise TypeError("transaction must be ProcurementTransaction")

        lifecycle = classify_award_action(transaction)
        candidates = peer_group_candidates(transaction, self.plan)
        identity = resolve_vendor_identity(transaction, self.scope)
        txid = transaction.transaction_id
        if txid in self._transaction_ids:
            raise VendorMarketError(
                f"duplicate transaction_id in vendor-market population: {txid!r}"
            )
        transaction_limit = self.policy.max_transaction_ids
        if (
            transaction_limit is not None
            and len(self._transaction_ids) >= transaction_limit
        ):
            raise VendorMarketError(
                f"transaction index exceeds max_transaction_ids={transaction_limit}"
            )

        keys = tuple(
            candidate.key
            for candidate in candidates
            if candidate.key is not None
        )

        base_digest: str | None = None
        new_pairs: set[tuple[PeerGroupKey, str]] = set()
        if lifecycle.kind is AwardActionKind.BASE_AWARD:
            if transaction.award_id in self._base_awards:
                raise VendorMarketError(
                    "multiple base-award actions observed for "
                    f"award_id={transaction.award_id!r}"
                )

            base_limit = self.policy.max_base_award_ids
            if (
                base_limit is not None
                and len(self._base_awards) >= base_limit
            ):
                raise VendorMarketError(
                    f"base-award index exceeds max_base_award_ids={base_limit}"
                )

            base_digest = _base_award_fingerprint(
                transaction,
                lifecycle.normalized_modification_number,
                candidates,
                identity,
                self.scope,
            )

            new_market_keys = {
                key for key in keys if key not in self._markets
            }
            market_limit = self.policy.max_distinct_markets
            if (
                market_limit is not None
                and len(self._markets) + len(new_market_keys) > market_limit
            ):
                raise VendorMarketError(
                    f"market index exceeds max_distinct_markets={market_limit}"
                )

            if identity is not None:
                new_pairs = {
                    (key, identity.key)
                    for key in keys
                    if (key, identity.key) not in self._vendor_market_pairs
                }
                pair_limit = self.policy.max_vendor_market_pairs
                if (
                    pair_limit is not None
                    and len(self._vendor_market_pairs) + len(new_pairs)
                    > pair_limit
                ):
                    raise VendorMarketError(
                        "vendor-market index exceeds "
                        f"max_vendor_market_pairs={pair_limit}"
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

        assert base_digest is not None
        self._base_awards[transaction.award_id] = base_digest
        self._base += 1
        self._vendor_market_pairs.update(new_pairs)

        for key in keys:
            accumulator = self._markets.get(key)
            if accumulator is None:
                accumulator = _MarketAccumulator(0, 0, 0, Counter())
                self._markets[key] = accumulator

            accumulator.observed_new_award_count += 1
            if identity is None:
                accumulator.unidentified_new_awards += 1
            else:
                accumulator.identified_new_awards += 1
                accumulator.vendor_counts[identity.key] += 1

    def observe_many(
        self, transactions: Iterable[ProcurementTransaction]
    ) -> "VendorMarketIndex":
        for transaction in transactions:
            self.observe(transaction)
        return self

    def snapshot(self) -> VendorMarketSnapshot:
        return VendorMarketSnapshot(
            plan=self.plan,
            scope=self.scope,
            total_transactions_seen=self._total,
            observed_base_awards=self._base,
            observed_modifications=self._modifications,
            lifecycle_unknown_transactions=self._unknown,
            markets={
                key: ObservedVendorMarket(
                    key=key,
                    observed_new_award_count=value.observed_new_award_count,
                    new_awards_with_vendor_identity=value.identified_new_awards,
                    new_awards_without_vendor_identity=value.unidentified_new_awards,
                    awards_by_winning_vendor=dict(value.vendor_counts),
                )
                for key, value in self._markets.items()
            },
            base_award_fingerprints=dict(self._base_awards),
        )


def federal_vendor_market_plan(
    *,
    time_scope: TimeScope = TimeScope.FEDERAL_FISCAL_YEAR,
    include_broader_naics_fallbacks: bool = True,
) -> PeerGroupPlan:
    """Observed-winner plan with controlled broadening and no global fallback."""

    time_scope = TimeScope(time_scope)
    levels = [
        PeerGroupLevel(
            "vendor_subtier_psc_exact",
            agency_scope=AgencyScope.SUBTIER,
            category_scope=CategoryScope.PSC_EXACT,
            time_scope=time_scope,
        ),
        PeerGroupLevel(
            "vendor_agency_psc_exact",
            agency_scope=AgencyScope.TOP_LEVEL,
            category_scope=CategoryScope.PSC_EXACT,
            time_scope=time_scope,
        ),
        PeerGroupLevel(
            "vendor_agency_naics_6",
            agency_scope=AgencyScope.TOP_LEVEL,
            category_scope=CategoryScope.NAICS_6,
            time_scope=time_scope,
        ),
    ]

    if include_broader_naics_fallbacks:
        levels.extend(
            (
                PeerGroupLevel(
                    "vendor_agency_naics_4",
                    agency_scope=AgencyScope.TOP_LEVEL,
                    category_scope=CategoryScope.NAICS_4,
                    time_scope=time_scope,
                ),
                PeerGroupLevel(
                    "vendor_agency_naics_2",
                    agency_scope=AgencyScope.TOP_LEVEL,
                    category_scope=CategoryScope.NAICS_2,
                    time_scope=time_scope,
                ),
            )
        )

    levels.extend(
        (
            PeerGroupLevel(
                "vendor_psc_exact",
                category_scope=CategoryScope.PSC_EXACT,
                time_scope=time_scope,
            ),
            PeerGroupLevel(
                "vendor_naics_6",
                category_scope=CategoryScope.NAICS_6,
                time_scope=time_scope,
            ),
        )
    )

    if include_broader_naics_fallbacks:
        levels.extend(
            (
                PeerGroupLevel(
                    "vendor_naics_4",
                    category_scope=CategoryScope.NAICS_4,
                    time_scope=time_scope,
                ),
                PeerGroupLevel(
                    "vendor_naics_2",
                    category_scope=CategoryScope.NAICS_2,
                    time_scope=time_scope,
                ),
            )
        )

    return PeerGroupPlan(
        "federal-contract-observed-new-award-vendor-market",
        tuple(levels),
    )


def _base_award_fingerprint(
    transaction: ProcurementTransaction,
    normalized_modification_number: str | None,
    candidates: tuple[PeerGroupCandidate, ...],
    identity: VendorIdentity | None,
    scope: VendorIdentityScope,
) -> str:
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
