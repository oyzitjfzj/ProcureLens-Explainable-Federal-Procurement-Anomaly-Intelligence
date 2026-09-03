"""Preflighted streaming loader for verified USAspending transaction artifacts.

The loader composes artifact integrity, tabular structure, source-schema
semantics, and canonical transaction normalization without collapsing those
boundaries. It emits no risk or fraud judgement.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import date
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import json
from typing import Any

from procurelens.domain.transaction import ProcurementTransaction, TransactionContractError
from procurelens.sources.usaspending.adapter import (
    USAspendingAdapterError,
    USAspendingTransactionAdapter,
)
from procurelens.sources.usaspending.artifact import ArtifactReceipt
from procurelens.sources.usaspending.download_schema import (
    DEFAULT_DOWNLOAD_SCHEMA_REGISTRY,
    DownloadSchemaProfile,
    DownloadSchemaRegistry,
    SchemaCompatibility,
)
from procurelens.sources.usaspending.reader import (
    MemberSchema,
    RowProvenance,
    TabularRecord,
    USAspendingArchiveReader,
    normalize_header,
)


class USAspendingLoaderError(RuntimeError):
    """Base error for composed ingestion failures."""


class LoadPlanError(USAspendingLoaderError):
    """Raised when an artifact cannot be safely preflighted."""


class RowLoadError(USAspendingLoaderError):
    """Fail-closed row error retaining exact source location."""

    def __init__(self, message: str, provenance: RowProvenance) -> None:
        super().__init__(message)
        self.provenance = provenance


class DuplicateTransactionError(USAspendingLoaderError):
    """Raised for an unsafe repeated transaction identity."""


class LoaderLimitError(USAspendingLoaderError):
    """Raised when an explicit resource/error budget is exceeded."""


class RowErrorMode(str, Enum):
    RAISE = "raise"
    QUARANTINE = "quarantine"


class ExactDuplicateMode(str, Enum):
    DROP = "drop"
    RAISE = "raise"


@dataclass(frozen=True, slots=True)
class LoaderPolicy:
    """Non-risk coordination policy; every tolerance is caller-configurable."""

    row_error_mode: RowErrorMode = RowErrorMode.RAISE
    conflicting_duplicate_mode: RowErrorMode = RowErrorMode.RAISE
    exact_duplicate_mode: ExactDuplicateMode = ExactDuplicateMode.DROP
    reject_additive_schema_drift: bool = False
    allow_mixed_schema_profiles: bool = False
    preserve_unmapped_fields: bool = False
    max_quarantined_rows: int | None = None
    max_identity_index_entries: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_error_mode", RowErrorMode(self.row_error_mode))
        object.__setattr__(
            self,
            "conflicting_duplicate_mode",
            RowErrorMode(self.conflicting_duplicate_mode),
        )
        object.__setattr__(
            self,
            "exact_duplicate_mode",
            ExactDuplicateMode(self.exact_duplicate_mode),
        )
        for name in ("max_quarantined_rows", "max_identity_index_entries"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")


@dataclass(frozen=True, slots=True)
class MemberLoadPlan:
    member: MemberSchema
    profile: DownloadSchemaProfile
    compatibility: SchemaCompatibility

    def __post_init__(self) -> None:
        report = self.compatibility
        if not report.compatible:
            raise LoadPlanError(f"incompatible member: {self.member.member_name}")
        if report.profile_name != self.profile.name:
            raise LoadPlanError("profile/report mismatch")
        if report.profile_sha256 != self.profile.sha256_hex:
            raise LoadPlanError("profile fingerprint mismatch")
        if report.observed_schema_sha256 != self.member.schema_sha256:
            raise LoadPlanError("observed schema fingerprint mismatch")


@dataclass(frozen=True, slots=True)
class LoadPlan:
    artifact_sha256: str
    artifact_size_bytes: int
    request_fingerprint_sha256: str | None
    members: tuple[MemberLoadPlan, ...]
    ignored_members: tuple[str, ...]
    plan_sha256: str

    @property
    def member_names(self) -> tuple[str, ...]:
        return tuple(item.member.member_name for item in self.members)


@dataclass(frozen=True, slots=True)
class LoadedTransaction:
    transaction: ProcurementTransaction
    provenance: RowProvenance
    profile_name: str
    profile_sha256: str
    row_value_sha256: str
    semantic_sha256: str
    additive_schema_headers: tuple[str, ...]
    request_fingerprint_sha256: str | None


@dataclass(frozen=True, slots=True)
class QuarantinedRow:
    provenance: RowProvenance
    profile_name: str
    row_value_sha256: str
    error_type: str
    error_message: str
    transaction_id_hint: str | None


@dataclass(frozen=True, slots=True)
class DroppedExactDuplicate:
    transaction_id: str
    semantic_sha256: str
    first_provenance: RowProvenance
    duplicate_provenance: RowProvenance


LoadOutcome = LoadedTransaction | QuarantinedRow | DroppedExactDuplicate


@dataclass(frozen=True, slots=True)
class LoadReport:
    artifact_sha256: str
    plan_sha256: str
    rows_seen: int
    transactions_emitted: int
    rows_quarantined: int
    exact_duplicates_dropped: int
    conflicting_duplicates: int
    indexed_transaction_ids: int
    complete: bool


class USAspendingLoadSession:
    """Single-use stream; `complete` becomes true only after full exhaustion."""

    def __init__(self, loader: "USAspendingDatasetLoader", receipt: ArtifactReceipt, plan: LoadPlan) -> None:
        if receipt.sha256_hex != plan.artifact_sha256 or receipt.size_bytes != plan.artifact_size_bytes:
            raise LoadPlanError("load plan does not match artifact receipt")
        if receipt.request_fingerprint_sha256 != plan.request_fingerprint_sha256:
            raise LoadPlanError("request fingerprint changed since preflight")
        self.loader, self.receipt, self.plan = loader, receipt, plan
        self._started = self._complete = False
        self._rows_seen = self._emitted = self._quarantined = 0
        self._dropped = self._conflicts = 0
        self._seen: dict[str, tuple[str, RowProvenance]] = {}

    @property
    def report(self) -> LoadReport:
        return LoadReport(
            self.plan.artifact_sha256, self.plan.plan_sha256, self._rows_seen,
            self._emitted, self._quarantined, self._dropped, self._conflicts,
            len(self._seen), self._complete,
        )

    def iter_transactions(self) -> Iterator[LoadedTransaction]:
        for outcome in self.iter_outcomes():
            if isinstance(outcome, LoadedTransaction):
                yield outcome

    def iter_outcomes(self) -> Iterator[LoadOutcome]:
        if self._started:
            raise RuntimeError("load session is single-use")
        self._started = True
        by_member = {item.member.member_name: item for item in self.plan.members}
        adapters: dict[str, USAspendingTransactionAdapter] = {}
        for record in self.loader.reader.iter_records(self.receipt, member_names=self.plan.member_names):
            self._rows_seen += 1
            planned = by_member.get(record.provenance.member_name)
            if planned is None or record.provenance.schema_sha256 != planned.member.schema_sha256:
                raise LoadPlanError("member/schema changed after preflight")
            adapter = adapters.get(planned.profile.name)
            if adapter is None:
                adapter = self.loader._adapter(planned.profile)
                adapters[planned.profile.name] = adapter
            row_hash = _row_digest(record.values)
            try:
                tx = adapter.adapt(
                    record.values,
                    retrieved_at=self.receipt.downloaded_at,
                    source_schema=_schema_identity(planned),
                )
            except (USAspendingAdapterError, TransactionContractError) as exc:
                outcome = self._row_failure(record, planned.profile, row_hash, exc)
                if outcome is not None:
                    yield outcome
                continue
            semantic_hash = _semantic_digest(tx)
            duplicate = self._duplicate(tx, semantic_hash, record, planned.profile, row_hash)
            if duplicate is not None:
                yield duplicate
                continue
            self._emitted += 1
            yield LoadedTransaction(
                tx, record.provenance, planned.profile.name, planned.profile.sha256_hex,
                row_hash, semantic_hash, planned.compatibility.unrecognized_headers,
                self.receipt.request_fingerprint_sha256,
            )
        self._complete = True

    def _row_failure(self, record: TabularRecord, profile: DownloadSchemaProfile, row_hash: str, error: BaseException) -> QuarantinedRow | None:
        if self.loader.policy.row_error_mode is RowErrorMode.RAISE:
            raise RowLoadError(f"{type(error).__name__}: {error}", record.provenance) from error
        self._quarantined += 1
        self._check_quarantine_budget()
        return QuarantinedRow(
            record.provenance, profile.name, row_hash, type(error).__name__, str(error),
            _transaction_id_hint(record.values, profile),
        )

    def _duplicate(self, tx: ProcurementTransaction, semantic_hash: str, record: TabularRecord, profile: DownloadSchemaProfile, row_hash: str) -> LoadOutcome | None:
        previous = self._seen.get(tx.transaction_id)
        if previous is None:
            limit = self.loader.policy.max_identity_index_entries
            if limit is not None and len(self._seen) >= limit:
                raise LoaderLimitError(f"identity index exceeds max_identity_index_entries={limit}")
            self._seen[tx.transaction_id] = (semantic_hash, record.provenance)
            return None
        previous_hash, first_provenance = previous
        if previous_hash == semantic_hash:
            if self.loader.policy.exact_duplicate_mode is ExactDuplicateMode.RAISE:
                raise DuplicateTransactionError(f"exact duplicate transaction_id {tx.transaction_id!r}")
            self._dropped += 1
            return DroppedExactDuplicate(tx.transaction_id, semantic_hash, first_provenance, record.provenance)
        self._conflicts += 1
        if self.loader.policy.conflicting_duplicate_mode is RowErrorMode.RAISE:
            raise DuplicateTransactionError(f"conflicting duplicate transaction_id {tx.transaction_id!r}")
        self._quarantined += 1
        self._check_quarantine_budget()
        return QuarantinedRow(
            record.provenance, profile.name, row_hash, "ConflictingDuplicateTransaction",
            f"transaction_id {tx.transaction_id!r} repeats with different canonical semantics",
            tx.transaction_id,
        )

    def _check_quarantine_budget(self) -> None:
        limit = self.loader.policy.max_quarantined_rows
        if limit is not None and self._quarantined > limit:
            raise LoaderLimitError(f"quarantine exceeds max_quarantined_rows={limit}")


class USAspendingDatasetLoader:
    """Whole-artifact preflight followed by provenance-preserving row streaming."""

    def __init__(
        self,
        *,
        reader: USAspendingArchiveReader | None = None,
        registry: DownloadSchemaRegistry | None = None,
        policy: LoaderPolicy | None = None,
    ) -> None:
        self.reader = reader or USAspendingArchiveReader()
        self.registry = registry or DEFAULT_DOWNLOAD_SCHEMA_REGISTRY
        self.policy = policy or LoaderPolicy()

    def plan(self, receipt: ArtifactReceipt, *, member_names: Sequence[str] | None = None, profile_name: str | None = None) -> LoadPlan:
        scan = self.reader.scan(receipt)
        selected = _select_members(scan.members, member_names)
        if not selected:
            raise LoadPlanError("no tabular members selected")
        resolved: list[MemberLoadPlan] = []
        for member in selected:
            profile, _ = self.registry.resolve(
                member.headers,
                profile_name=profile_name,
                reject_additive_drift=self.policy.reject_additive_schema_drift,
            )
            report = profile.inspect_member(member)
            if self.policy.reject_additive_schema_drift and report.unrecognized_headers:
                raise LoadPlanError(f"{member.member_name}: additive schema drift rejected")
            resolved.append(MemberLoadPlan(member, profile, report))
        profiles = {item.profile.name for item in resolved}
        if len(profiles) > 1 and not self.policy.allow_mixed_schema_profiles:
            raise LoadPlanError("multiple schema profiles require explicit opt-in")
        members = tuple(resolved)
        return LoadPlan(
            receipt.sha256_hex,
            receipt.size_bytes,
            receipt.request_fingerprint_sha256,
            members,
            scan.ignored_members,
            _plan_digest(receipt, members, scan.ignored_members),
        )

    def open_session(self, receipt: ArtifactReceipt, *, plan: LoadPlan | None = None, member_names: Sequence[str] | None = None, profile_name: str | None = None) -> USAspendingLoadSession:
        if plan is not None and (member_names is not None or profile_name is not None):
            raise ValueError("existing plan cannot be combined with member/profile overrides")
        return USAspendingLoadSession(
            self,
            receipt,
            plan or self.plan(receipt, member_names=member_names, profile_name=profile_name),
        )

    def _adapter(self, profile: DownloadSchemaProfile) -> USAspendingTransactionAdapter:
        return USAspendingTransactionAdapter(
            field_aliases=profile.adapter_aliases(),
            preserve_unmapped_fields=self.policy.preserve_unmapped_fields,
            source_name="USAspending.gov",
        )


def _select_members(members: Sequence[MemberSchema], names: Sequence[str] | None) -> tuple[MemberSchema, ...]:
    by_name = {member.member_name: member for member in members}
    if len(by_name) != len(members):
        raise LoadPlanError("duplicate member names in archive scan")
    if names is None:
        return tuple(members)
    if isinstance(names, (str, bytes)):
        raise TypeError("member_names must be a sequence")
    selected, seen = [], set()
    for name in names:
        if not isinstance(name, str) or not name or name in seen:
            raise LoadPlanError(f"invalid/duplicate member name: {name!r}")
        seen.add(name)
        if name not in by_name:
            raise LoadPlanError(f"requested member not found: {name}")
        selected.append(by_name[name])
    return tuple(selected)


def _json_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return sha256(raw).hexdigest()


def _row_digest(values: Mapping[str, str]) -> str:
    ordered = sorted(((str(k), str(v)) for k, v in values.items()), key=lambda item: (normalize_header(item[0]), item[0]))
    return _json_digest(ordered)


def _decimal_identity(value: Decimal) -> str:
    if value.is_zero():
        return "0"
    sign, digits_tuple, exponent = value.as_tuple()
    digits = list(digits_tuple)
    while digits and digits[-1] == 0:
        digits.pop(); exponent += 1
    return f"{sign}:{''.join(str(d) for d in digits)}:{exponent}"


def _semantic_digest(tx: ProcurementTransaction) -> str:
    payload = []
    for field in fields(tx):
        if field.name in {"lineage", "attributes"}:
            continue
        value = getattr(tx, field.name)
        if isinstance(value, Decimal): value = _decimal_identity(value)
        elif isinstance(value, date): value = value.isoformat()
        elif not (value is None or isinstance(value, (str, int, bool))):
            raise TypeError(f"unsupported canonical field type: {type(value).__name__}")
        payload.append((field.name, value))
    return _json_digest(payload)


def _schema_identity(plan: MemberLoadPlan) -> str:
    return f"{plan.profile.name}@{plan.profile.sha256_hex};observed={plan.member.schema_sha256}"


def _transaction_id_hint(values: Mapping[str, str], profile: DownloadSchemaProfile) -> str | None:
    indexed = {normalize_header(k): v for k, v in values.items()}
    for alias in profile.adapter_aliases().get("transaction_id", ()):
        value = indexed.get(normalize_header(alias))
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _plan_digest(receipt: ArtifactReceipt, members: tuple[MemberLoadPlan, ...], ignored: tuple[str, ...]) -> str:
    return _json_digest({
        "artifact_sha256": receipt.sha256_hex,
        "artifact_size_bytes": receipt.size_bytes,
        "request_fingerprint_sha256": receipt.request_fingerprint_sha256,
        "members": [
            {
                "name": item.member.member_name,
                "schema_sha256": item.member.schema_sha256,
                "profile": item.profile.name,
                "profile_sha256": item.profile.sha256_hex,
                "unrecognized": list(item.compatibility.unrecognized_headers),
            }
            for item in members
        ],
        "ignored_members": list(ignored),
    })
