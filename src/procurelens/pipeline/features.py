"""Frozen-reference candidate feature orchestration for ProcureLens.

Builds immutable comparison snapshots from one explicit reference population, then
constructs the complete canonical candidate-feature row for each target
transaction without mutating those snapshots. This preserves the same train/score
separation used later by preprocessing, detector fitting, and calibration.

The orchestrator owns no hidden peer thresholds, market hierarchies, identity
scope, amount basis, quantile choice, or memory budget. Every such choice comes
from FeatureBuildPlan.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import json
from typing import Any

from procurelens.domain.transaction import ProcurementTransaction
from procurelens.features.amount_context import build_amount_context
from procurelens.features.amount_features import build_amount_features
from procurelens.features.amount_reference import AmountReferenceIndex, AmountReferenceSnapshot
from procurelens.features.award_change_context import resolve_award_change_context
from procurelens.features.award_change_features import build_award_change_features
from procurelens.features.award_change_reference import AwardChangeReferenceIndex, AwardChangeReferenceSnapshot
from procurelens.features.catalog import FeatureCatalog, feature_catalog
from procurelens.features.competition_context import resolve_competition_context
from procurelens.features.competition_context_features import build_competition_context_features
from procurelens.features.competition_evidence import build_competition_evidence
from procurelens.features.competition_features import build_competition_features
from procurelens.features.competition_reference import CompetitionReferenceIndex, CompetitionReferenceSnapshot
from procurelens.features.vendor_frequency_features import build_vendor_frequency_features
from procurelens.features.vendor_market import VendorMarketIndex, VendorMarketSnapshot
from procurelens.features.vendor_market_context import resolve_vendor_market_context
from procurelens.model.feature_row import CandidateFeatureRow, build_candidate_feature_row
from procurelens.pipeline.feature_config import FeatureBuildPlan


class FeaturePipelineError(ValueError):
    """Raised when reference/target feature orchestration is inconsistent."""


@dataclass(frozen=True, slots=True)
class FrozenFeatureReferences:
    """Exact immutable comparison universe used for candidate feature creation."""

    plan: FeatureBuildPlan
    reference_population_sha256: str
    reference_transaction_count: int
    amount: AmountReferenceSnapshot
    vendor: VendorMarketSnapshot
    competition: CompetitionReferenceSnapshot
    award_change: AwardChangeReferenceSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.plan, FeatureBuildPlan):
            raise TypeError("plan must be FeatureBuildPlan")
        object.__setattr__(self, "reference_population_sha256", _digest_hex(self.reference_population_sha256, "reference_population_sha256"))
        if isinstance(self.reference_transaction_count, bool) or not isinstance(self.reference_transaction_count, int) or self.reference_transaction_count < 1:
            raise FeaturePipelineError("reference_transaction_count must be a positive integer")
        if not isinstance(self.amount, AmountReferenceSnapshot):
            raise TypeError("amount must be AmountReferenceSnapshot")
        if not isinstance(self.vendor, VendorMarketSnapshot):
            raise TypeError("vendor must be VendorMarketSnapshot")
        if not isinstance(self.competition, CompetitionReferenceSnapshot):
            raise TypeError("competition must be CompetitionReferenceSnapshot")
        if not isinstance(self.award_change, AwardChangeReferenceSnapshot):
            raise TypeError("award_change must be AwardChangeReferenceSnapshot")
        if self.amount.plan.sha256_hex != self.plan.amount_peer_plan.sha256_hex:
            raise FeaturePipelineError("amount reference peer plan differs from feature plan")
        if self.amount.amount_basis is not self.plan.amount_basis:
            raise FeaturePipelineError("amount reference basis differs from feature plan")
        if self.vendor.plan.sha256_hex != self.plan.vendor_peer_plan.sha256_hex:
            raise FeaturePipelineError("vendor reference peer plan differs from feature plan")
        if self.vendor.scope is not self.plan.vendor_scope:
            raise FeaturePipelineError("vendor reference identity scope differs from feature plan")
        if self.competition.plan.sha256_hex != self.plan.competition_peer_plan.sha256_hex:
            raise FeaturePipelineError("competition reference peer plan differs from feature plan")
        if self.award_change.plan.sha256_hex != self.plan.award_change_peer_plan.sha256_hex:
            raise FeaturePipelineError("award-change reference peer plan differs from feature plan")
        if self.amount.total_transactions != self.reference_transaction_count:
            raise FeaturePipelineError("amount reference transaction count differs from population")
        if self.vendor.total_transactions_seen != self.reference_transaction_count:
            raise FeaturePipelineError("vendor reference transaction count differs from population")
        if self.competition.total_transactions_seen != self.reference_transaction_count:
            raise FeaturePipelineError("competition reference transaction count differs from population")

    @property
    def amount_snapshot_sha256(self) -> str:
        return _evidence_digest(self.amount)

    @property
    def vendor_snapshot_sha256(self) -> str:
        return _evidence_digest(self.vendor)

    @property
    def competition_snapshot_sha256(self) -> str:
        return _evidence_digest(self.competition)

    @property
    def award_change_snapshot_sha256(self) -> str:
        return _evidence_digest(self.award_change)

    @property
    def evidence_sha256(self) -> str:
        return _digest({
            "plan_sha256": self.plan.sha256_hex,
            "reference_population_sha256": self.reference_population_sha256,
            "reference_transaction_count": self.reference_transaction_count,
            "amount_snapshot_sha256": self.amount_snapshot_sha256,
            "vendor_snapshot_sha256": self.vendor_snapshot_sha256,
            "competition_snapshot_sha256": self.competition_snapshot_sha256,
            "award_change_snapshot_sha256": self.award_change_snapshot_sha256,
        })

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_sha256": self.plan.sha256_hex,
            "reference_population_sha256": self.reference_population_sha256,
            "reference_transaction_count": self.reference_transaction_count,
            "amount_snapshot_sha256": self.amount_snapshot_sha256,
            "vendor_snapshot_sha256": self.vendor_snapshot_sha256,
            "competition_snapshot_sha256": self.competition_snapshot_sha256,
            "award_change_snapshot_sha256": self.award_change_snapshot_sha256,
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True, slots=True)
class CandidateFeatureBatch:
    """Ordered model-candidate rows resolved against one frozen reference bundle."""

    plan_sha256: str
    reference_bundle_sha256: str
    target_population_sha256: str
    rows: tuple[CandidateFeatureRow, ...]

    def __post_init__(self) -> None:
        for field_name in ("plan_sha256", "reference_bundle_sha256", "target_population_sha256"):
            object.__setattr__(self, field_name, _digest_hex(getattr(self, field_name), field_name))
        rows = tuple(self.rows)
        if not rows:
            raise FeaturePipelineError("candidate feature batch requires at least one row")
        if any(not isinstance(row, CandidateFeatureRow) for row in rows):
            raise TypeError("rows must be CandidateFeatureRow")
        identities = tuple((row.transaction_id, row.award_id) for row in rows)
        if len(identities) != len(set(identities)):
            raise FeaturePipelineError("candidate feature batch contains duplicate row identities")
        catalog_hashes = {row.catalog_sha256 for row in rows}
        if len(catalog_hashes) != 1:
            raise FeaturePipelineError("candidate feature rows use different feature catalogs")
        object.__setattr__(self, "rows", rows)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def row_identities(self) -> tuple[tuple[str, str], ...]:
        return tuple((row.transaction_id, row.award_id) for row in self.rows)

    @property
    def feature_catalog_sha256(self) -> str:
        return self.rows[0].catalog_sha256

    @property
    def evidence_sha256(self) -> str:
        return _digest({
            "plan_sha256": self.plan_sha256,
            "reference_bundle_sha256": self.reference_bundle_sha256,
            "target_population_sha256": self.target_population_sha256,
            "rows": [row.evidence_sha256 for row in self.rows],
        })

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_sha256": self.plan_sha256,
            "reference_bundle_sha256": self.reference_bundle_sha256,
            "target_population_sha256": self.target_population_sha256,
            "row_count": self.row_count,
            "feature_catalog_sha256": self.feature_catalog_sha256,
            "row_identities": [list(item) for item in self.row_identities],
            "row_evidence_sha256": [row.evidence_sha256 for row in self.rows],
            "evidence_sha256": self.evidence_sha256,
        }


def build_feature_references(reference_transactions: Iterable[ProcurementTransaction], plan: FeatureBuildPlan, *, catalog: FeatureCatalog | None = None) -> FrozenFeatureReferences:
    """Build all frozen contextual reference snapshots from one exact population."""
    if not isinstance(plan, FeatureBuildPlan):
        raise TypeError("plan must be FeatureBuildPlan")
    catalog = feature_catalog() if catalog is None else catalog
    _validate_catalog(plan, catalog)
    transactions = _transactions(reference_transactions, name="reference_transactions", require_unique_identity=True)
    amount_index = AmountReferenceIndex(plan=plan.amount_peer_plan, amount_basis=plan.amount_basis, policy=plan.amount_reference_policy)
    vendor_index = VendorMarketIndex(plan=plan.vendor_peer_plan, scope=plan.vendor_scope, policy=plan.vendor_market_policy)
    competition_index = CompetitionReferenceIndex(plan=plan.competition_peer_plan, policy=plan.competition_reference_policy)
    award_change_index = AwardChangeReferenceIndex(plan=plan.award_change_peer_plan, policy=plan.award_change_reference_policy)
    for transaction in transactions:
        amount_index.observe(transaction)
        vendor_index.observe(transaction)
        competition_index.observe(transaction)
        award_change_index.observe(transaction)
    return FrozenFeatureReferences(
        plan=plan,
        reference_population_sha256=_population_digest(transactions, order_sensitive=False),
        reference_transaction_count=len(transactions),
        amount=amount_index.snapshot(),
        vendor=vendor_index.snapshot(),
        competition=competition_index.snapshot(),
        award_change=award_change_index.snapshot(),
    )


def build_candidate_features(target_transactions: Iterable[ProcurementTransaction], references: FrozenFeatureReferences, *, catalog: FeatureCatalog | None = None) -> CandidateFeatureBatch:
    """Resolve target rows against frozen references without mutating fit state."""
    if not isinstance(references, FrozenFeatureReferences):
        raise TypeError("references must be FrozenFeatureReferences")
    catalog = feature_catalog() if catalog is None else catalog
    _validate_catalog(references.plan, catalog)
    transactions = _transactions(target_transactions, name="target_transactions", require_unique_identity=True)
    rows = tuple(_build_target_row(transaction, references, catalog) for transaction in transactions)
    batch = CandidateFeatureBatch(
        plan_sha256=references.plan.sha256_hex,
        reference_bundle_sha256=references.evidence_sha256,
        target_population_sha256=_population_digest(transactions, order_sensitive=True),
        rows=rows,
    )
    if batch.feature_catalog_sha256 != references.plan.feature_catalog_sha256:
        raise FeaturePipelineError("candidate feature batch catalog differs from feature plan")
    return batch


def _build_target_row(transaction: ProcurementTransaction, references: FrozenFeatureReferences, catalog: FeatureCatalog) -> CandidateFeatureRow:
    plan = references.plan
    amount_reference = references.amount.resolve(transaction, minimum_peer_count=plan.amount_minimum_peer_count)
    amount_context = build_amount_context(amount_reference, quantile_method=plan.quantile_method)
    amount_features = build_amount_features(amount_context).feature_set
    vendor_context = resolve_vendor_market_context(transaction, references.vendor, support_spec=plan.vendor_support)
    vendor_features = build_vendor_frequency_features(vendor_context, references.vendor, quantile_method=plan.quantile_method).feature_set
    competition_evidence = build_competition_evidence(transaction)
    competition_transaction_features = build_competition_features(competition_evidence)
    competition_context = resolve_competition_context(transaction, references.competition, support_spec=plan.competition_support)
    competition_context_features = build_competition_context_features(competition_context)
    award_change_context = resolve_award_change_context(transaction.award_id, references.award_change, support_spec=plan.award_change_support)
    award_change_features = build_award_change_features(award_change_context)
    return build_candidate_feature_row(
        transaction_id=transaction.transaction_id,
        award_id=transaction.award_id,
        amount=amount_features,
        vendor_frequency=vendor_features,
        competition_transaction=competition_transaction_features,
        competition_context=competition_context_features,
        award_change=award_change_features,
        catalog=catalog,
    )


def _validate_catalog(plan: FeatureBuildPlan, catalog: FeatureCatalog) -> None:
    if not isinstance(catalog, FeatureCatalog):
        raise TypeError("catalog must be FeatureCatalog")
    if catalog.sha256_hex != plan.feature_catalog_sha256:
        raise FeaturePipelineError("feature plan catalog differs from supplied/current catalog")


def _transactions(values: Iterable[ProcurementTransaction], *, name: str, require_unique_identity: bool) -> tuple[ProcurementTransaction, ...]:
    if isinstance(values, (str, bytes)):
        raise FeaturePipelineError(f"{name} must be an iterable of ProcurementTransaction")
    items = tuple(values)
    if not items:
        raise FeaturePipelineError(f"{name} must not be empty")
    if any(not isinstance(item, ProcurementTransaction) for item in items):
        raise TypeError(f"all {name} must be ProcurementTransaction")
    if require_unique_identity:
        transaction_ids = tuple(item.transaction_id for item in items)
        if len(transaction_ids) != len(set(transaction_ids)):
            raise FeaturePipelineError(
                f"{name} contains duplicate transaction_id values"
            )
        identities = tuple((item.transaction_id, item.award_id) for item in items)
        if len(identities) != len(set(identities)):
            raise FeaturePipelineError(
                f"{name} contains duplicate transaction/award identities"
            )
    return items


def _population_digest(transactions: tuple[ProcurementTransaction, ...], *, order_sensitive: bool) -> str:
    evidence = [{"identity": [transaction.transaction_id, transaction.award_id], "transaction_sha256": _evidence_digest(transaction)} for transaction in transactions]
    if not order_sensitive:
        evidence.sort(key=lambda item: (item["identity"][0], item["identity"][1], item["transaction_sha256"]))
    return _digest(evidence)


def _evidence_digest(value: Any) -> str:
    return _digest(_canonical_evidence(value))


def _canonical_evidence(value: Any) -> Any:
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise FeaturePipelineError("evidence Decimal values must be finite")
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise FeaturePipelineError("evidence datetime values must be timezone-aware")
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _canonical_evidence(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            rendered = _mapping_key(key)
            if rendered in normalized:
                raise FeaturePipelineError("canonical evidence mapping key collision")
            normalized[rendered] = _canonical_evidence(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (tuple, list)):
        return [_canonical_evidence(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical_evidence(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _canonical_evidence(value.as_dict())
    raise FeaturePipelineError("unsupported evidence value for deterministic fingerprint: " + type(value).__name__)


def _mapping_key(value: Any) -> str:
    if isinstance(value, str):
        return "str:" + value
    if isinstance(value, Enum):
        return "enum:" + str(value.value)
    digest = getattr(value, "sha256_hex", None)
    if isinstance(digest, str):
        return type(value).__name__ + ":" + _digest_hex(digest, "mapping key sha256")
    if is_dataclass(value):
        return type(value).__name__ + ":" + _digest(_canonical_evidence(value))
    raise FeaturePipelineError("unsupported mapping key for deterministic fingerprint: " + type(value).__name__)


def _digest_hex(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise FeaturePipelineError(f"{name} must be a string")
    digest = value.strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise FeaturePipelineError(f"{name} must be a SHA-256 hex digest")
    return digest


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
