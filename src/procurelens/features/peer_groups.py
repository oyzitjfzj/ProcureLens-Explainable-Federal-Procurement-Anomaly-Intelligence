"""Context-aware peer groups for ProcureLens.

Amount anomaly is contextual: a transaction should be compared with meaningfully
similar procurement, not with the whole dataset. This module defines reviewable
comparison levels, counts their support, and selects the first level with enough
other transactions. It does not score anomalies and it contains no hidden
minimum group size.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any

from procurelens.domain.transaction import ProcurementTransaction


class PeerGroupError(ValueError):
    pass


class PeerGroupResolutionError(PeerGroupError):
    pass


class AgencyScope(str, Enum):
    SUBTIER = "subtier"
    TOP_LEVEL = "top_level"
    NONE = "none"


class CategoryScope(str, Enum):
    PSC_EXACT = "psc_exact"
    NAICS_6 = "naics_6"
    NAICS_5 = "naics_5"
    NAICS_4 = "naics_4"
    NAICS_3 = "naics_3"
    NAICS_2 = "naics_2"
    NONE = "none"


class TimeScope(str, Enum):
    MONTH = "month"
    QUARTER = "quarter"
    FEDERAL_FISCAL_YEAR = "federal_fiscal_year"
    CALENDAR_YEAR = "calendar_year"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class PeerGroupLevel:
    name: str
    agency_scope: AgencyScope = AgencyScope.NONE
    category_scope: CategoryScope = CategoryScope.NONE
    time_scope: TimeScope = TimeScope.NONE
    include_award_type: bool = False
    allow_unscoped: bool = False

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise PeerGroupError("peer-group level name must not be blank")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "agency_scope", AgencyScope(self.agency_scope))
        object.__setattr__(self, "category_scope", CategoryScope(self.category_scope))
        object.__setattr__(self, "time_scope", TimeScope(self.time_scope))
        scoped = (
            self.agency_scope is not AgencyScope.NONE
            or self.category_scope is not CategoryScope.NONE
            or self.time_scope is not TimeScope.NONE
            or self.include_award_type
        )
        if not scoped and not self.allow_unscoped:
            raise PeerGroupError(
                f"{name}: contextless peer group requires allow_unscoped=True"
            )


@dataclass(frozen=True, slots=True)
class PeerGroupPlan:
    name: str
    levels: tuple[PeerGroupLevel, ...]

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name or not self.levels:
            raise PeerGroupError("peer-group plan needs a name and at least one level")
        names = [level.name.casefold() for level in self.levels]
        if len(names) != len(set(names)):
            raise PeerGroupError("peer-group level names must be unique")
        object.__setattr__(self, "name", name)

    @property
    def sha256_hex(self) -> str:
        return _digest({
            "name": self.name,
            "levels": [
                {
                    "name": x.name,
                    "agency": x.agency_scope.value,
                    "category": x.category_scope.value,
                    "time": x.time_scope.value,
                    "award_type": x.include_award_type,
                    "unscoped": x.allow_unscoped,
                }
                for x in self.levels
            ],
        })


@dataclass(frozen=True, slots=True)
class PeerGroupKey:
    level_name: str
    components: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.level_name.strip() or not self.components:
            raise PeerGroupError("peer-group key needs level and components")
        labels = [name for name, _ in self.components]
        if len(labels) != len(set(labels)):
            raise PeerGroupError("peer-group component labels must be unique")

    @property
    def sha256_hex(self) -> str:
        return _digest({"level": self.level_name, "components": self.components})


@dataclass(frozen=True, slots=True)
class PeerGroupCandidate:
    level_name: str
    key: PeerGroupKey | None
    unavailable_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PeerGroupAttempt:
    level_name: str
    key: PeerGroupKey | None
    population_count: int | None
    peer_count: int | None
    sufficient: bool
    unavailable_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PeerGroupResolution:
    plan_name: str
    plan_sha256: str
    minimum_peer_count: int
    selected: PeerGroupAttempt | None
    attempts: tuple[PeerGroupAttempt, ...]

    @property
    def resolved(self) -> bool:
        return self.selected is not None


@dataclass(frozen=True, slots=True)
class PeerGroupIndexPolicy:
    track_transaction_context: bool = True
    reject_duplicate_transaction_ids: bool = True
    max_transaction_ids: int | None = None
    max_distinct_groups: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_transaction_ids", "max_distinct_groups"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise PeerGroupError(f"{name} must be a positive integer or None")
        if self.reject_duplicate_transaction_ids and not self.track_transaction_context:
            raise PeerGroupError("duplicate rejection requires transaction tracking")


@dataclass(frozen=True, slots=True)
class PeerGroupSnapshot:
    plan: PeerGroupPlan
    total_transactions: int
    group_counts: Mapping[PeerGroupKey, int]
    transaction_context_sha256: Mapping[str, str] | None

    def __post_init__(self) -> None:
        if self.total_transactions < 0:
            raise PeerGroupError("total_transactions must not be negative")
        if any(count < 1 or count > self.total_transactions for count in self.group_counts.values()):
            raise PeerGroupError("peer-group counts are inconsistent")
        object.__setattr__(self, "group_counts", MappingProxyType(dict(self.group_counts)))
        if self.transaction_context_sha256 is not None:
            object.__setattr__(
                self,
                "transaction_context_sha256",
                MappingProxyType(dict(self.transaction_context_sha256)),
            )

    def support(
        self,
        transaction: ProcurementTransaction,
        *,
        minimum_peer_count: int,
        exclude_target_if_indexed: bool = True,
    ) -> PeerGroupResolution:
        if (
            isinstance(minimum_peer_count, bool)
            or not isinstance(minimum_peer_count, int)
            or minimum_peer_count < 1
        ):
            raise PeerGroupResolutionError("minimum_peer_count must be caller-supplied and positive")

        candidates = peer_group_candidates(transaction, self.plan)
        indexed = False
        if exclude_target_if_indexed:
            if self.transaction_context_sha256 is None:
                raise PeerGroupResolutionError("target exclusion requires transaction tracking")
            stored = self.transaction_context_sha256.get(transaction.transaction_id)
            if stored is not None:
                if stored != _candidate_digest(candidates):
                    raise PeerGroupResolutionError(
                        "target context differs from its indexed reference record"
                    )
                indexed = True

        attempts: list[PeerGroupAttempt] = []
        selected: PeerGroupAttempt | None = None
        for candidate in candidates:
            if candidate.key is None:
                attempt = PeerGroupAttempt(
                    candidate.level_name, None, None, None, False,
                    candidate.unavailable_reasons,
                )
            else:
                population = self.group_counts.get(candidate.key, 0)
                peers = population - (1 if indexed and population else 0)
                attempt = PeerGroupAttempt(
                    candidate.level_name, candidate.key, population, peers,
                    peers >= minimum_peer_count,
                )
                if attempt.sufficient:
                    selected = attempt
            attempts.append(attempt)
            if selected is not None:
                break
        return PeerGroupResolution(
            self.plan.name, self.plan.sha256_hex, minimum_peer_count,
            selected, tuple(attempts),
        )


class PeerGroupIndex:
    """One-pass support counter; transaction objects are not retained."""

    def __init__(
        self,
        plan: PeerGroupPlan,
        *,
        policy: PeerGroupIndexPolicy | None = None,
    ) -> None:
        self.plan = plan
        self.policy = policy or PeerGroupIndexPolicy()
        self._counts: Counter[PeerGroupKey] = Counter()
        self._contexts: dict[str, str] | None = (
            {} if self.policy.track_transaction_context else None
        )
        self._total = 0

    def observe(self, transaction: ProcurementTransaction) -> None:
        """Atomically add one transaction after all configured limits pass."""

        candidates = peer_group_candidates(transaction, self.plan)
        txid = transaction.transaction_id
        context_digest = _candidate_digest(candidates)

        # Preflight every operation that can fail before mutating counts/context.
        add_context = False
        if self._contexts is not None:
            if txid in self._contexts:
                if self.policy.reject_duplicate_transaction_ids:
                    raise PeerGroupError(
                        f"duplicate transaction_id in peer population: {txid!r}"
                    )
            else:
                limit = self.policy.max_transaction_ids
                if limit is not None and len(self._contexts) >= limit:
                    raise PeerGroupError(
                        f"transaction index exceeds max_transaction_ids={limit}"
                    )
                add_context = True

        keys = tuple(
            candidate.key for candidate in candidates if candidate.key is not None
        )
        new_keys = {key for key in keys if key not in self._counts}
        group_limit = self.policy.max_distinct_groups
        if group_limit is not None and len(self._counts) + len(new_keys) > group_limit:
            raise PeerGroupError(
                f"group index exceeds max_distinct_groups={group_limit}"
            )

        # Commit only after the full preflight has succeeded.
        if add_context and self._contexts is not None:
            self._contexts[txid] = context_digest
        for key in keys:
            self._counts[key] += 1
        self._total += 1

    def observe_many(self, transactions: Iterable[ProcurementTransaction]) -> "PeerGroupIndex":
        for transaction in transactions:
            self.observe(transaction)
        return self

    def snapshot(self) -> PeerGroupSnapshot:
        return PeerGroupSnapshot(
            self.plan, self._total, dict(self._counts),
            None if self._contexts is None else dict(self._contexts),
        )


def peer_group_candidates(
    transaction: ProcurementTransaction,
    plan: PeerGroupPlan,
) -> tuple[PeerGroupCandidate, ...]:
    if not isinstance(transaction, ProcurementTransaction):
        raise TypeError("transaction must be ProcurementTransaction")
    result: list[PeerGroupCandidate] = []
    for level in plan.levels:
        components: list[tuple[str, str]] = []
        missing: list[str] = []
        _add(components, missing, "agency", _agency(transaction, level.agency_scope), level.agency_scope)
        _add(components, missing, "category", _category(transaction, level.category_scope), level.category_scope)
        _add(components, missing, "time", _time(transaction, level.time_scope), level.time_scope)
        if level.include_award_type:
            award_type = _code(transaction.award_type_code)
            if award_type is None:
                missing.append("award_type")
            else:
                components.append(("award_type", award_type))
        if missing:
            result.append(PeerGroupCandidate(level.name, None, tuple(missing)))
        else:
            if not components:
                components.append(("scope", "all"))
            result.append(PeerGroupCandidate(level.name, PeerGroupKey(level.name, tuple(components))))
    return tuple(result)


def federal_contract_amount_peer_plan(
    *,
    time_scope: TimeScope = TimeScope.FEDERAL_FISCAL_YEAR,
    include_award_type_at_most_specific_level: bool = True,
    include_broader_naics_fallbacks: bool = True,
) -> PeerGroupPlan:
    """Procurement-focused amount-comparison plan with no global fallback."""
    time_scope = TimeScope(time_scope)
    levels = [
        PeerGroupLevel("subtier_psc_exact", AgencyScope.SUBTIER, CategoryScope.PSC_EXACT, time_scope, include_award_type_at_most_specific_level),
        PeerGroupLevel("agency_psc_exact", AgencyScope.TOP_LEVEL, CategoryScope.PSC_EXACT, time_scope),
        PeerGroupLevel("agency_naics_6", AgencyScope.TOP_LEVEL, CategoryScope.NAICS_6, time_scope),
    ]
    if include_broader_naics_fallbacks:
        levels += [
            PeerGroupLevel("agency_naics_4", AgencyScope.TOP_LEVEL, CategoryScope.NAICS_4, time_scope),
            PeerGroupLevel("agency_naics_2", AgencyScope.TOP_LEVEL, CategoryScope.NAICS_2, time_scope),
        ]
    levels += [
        PeerGroupLevel("psc_exact", category_scope=CategoryScope.PSC_EXACT, time_scope=time_scope),
        PeerGroupLevel("naics_6", category_scope=CategoryScope.NAICS_6, time_scope=time_scope),
    ]
    if include_broader_naics_fallbacks:
        levels += [
            PeerGroupLevel("naics_4", category_scope=CategoryScope.NAICS_4, time_scope=time_scope),
            PeerGroupLevel("naics_2", category_scope=CategoryScope.NAICS_2, time_scope=time_scope),
        ]
    return PeerGroupPlan("federal-contract-amount-context", tuple(levels))


def _add(parts: list[tuple[str, str]], missing: list[str], label: str, value: str | None, scope: Enum) -> None:
    if scope.value == "none":
        return
    if value is None:
        missing.append(label)
    else:
        parts.append((label, value))


def _code(value: str | None) -> str | None:
    return None if value is None or not value.strip() else value.strip().upper()


def _name(value: str | None) -> str | None:
    return None if value is None or not value.strip() else " ".join(value.split()).casefold()


def _agency(tx: ProcurementTransaction, scope: AgencyScope) -> str | None:
    if scope is AgencyScope.NONE:
        return None
    if scope is AgencyScope.SUBTIER:
        return (
            f"subtier-code:{code}" if (code := _code(tx.awarding_subtier_agency_code))
            else f"subtier-name:{name}" if (name := _name(tx.awarding_subtier_agency_name))
            else None
        )
    if scope is AgencyScope.TOP_LEVEL:
        return (
            f"agency-code:{code}" if (code := _code(tx.awarding_agency_code))
            else f"agency-name:{name}" if (name := _name(tx.awarding_agency_name))
            else None
        )
    raise PeerGroupError(f"unsupported agency scope: {scope!r}")


def _category(tx: ProcurementTransaction, scope: CategoryScope) -> str | None:
    if scope is CategoryScope.NONE:
        return None
    if scope is CategoryScope.PSC_EXACT:
        return None if (code := _code(tx.psc_code)) is None else f"psc:{code}"
    width = {
        CategoryScope.NAICS_6: 6, CategoryScope.NAICS_5: 5,
        CategoryScope.NAICS_4: 4, CategoryScope.NAICS_3: 3,
        CategoryScope.NAICS_2: 2,
    }.get(scope)
    code = _naics(tx.naics_code)
    if width is None or code is None or len(code) < width:
        return None
    return f"naics-{width}:{code[:width]}"


def _naics(value: str | None) -> str | None:
    if value is None:
        return None
    code = value.strip()
    return code if code.isdigit() and 2 <= len(code) <= 6 else None


def _time(tx: ProcurementTransaction, scope: TimeScope) -> str | None:
    d = tx.action_date
    if scope is TimeScope.NONE:
        return None
    if scope is TimeScope.MONTH:
        return f"month:{d.year:04d}-{d.month:02d}"
    if scope is TimeScope.QUARTER:
        return f"calendar-quarter:{d.year:04d}-Q{((d.month - 1) // 3) + 1}"
    if scope is TimeScope.CALENDAR_YEAR:
        return f"calendar-year:{d.year:04d}"
    if scope is TimeScope.FEDERAL_FISCAL_YEAR:
        fy = d.year + 1 if d.month >= 10 else d.year
        return f"federal-fiscal-year:{fy:04d}"
    raise PeerGroupError(f"unsupported time scope: {scope!r}")


def _candidate_digest(candidates: tuple[PeerGroupCandidate, ...]) -> str:
    return _digest([
        (x.level_name, None if x.key is None else x.key.sha256_hex, x.unavailable_reasons)
        for x in candidates
    ])


def _digest(value: Any) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()