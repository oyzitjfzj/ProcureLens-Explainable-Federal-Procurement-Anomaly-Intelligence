"""Unified candidate-feature rows for ProcureLens model preparation.

Binds canonical feature-set outputs from independent evidence families into one
immutable row keyed by transaction and award. Missing source families and
missing measurements remain explicit; this module never imputes, scales, clips,
or converts missing evidence to zero.

Rows preserve source definition and evidence fingerprints so later matrix and
preprocessing layers can validate provenance before fitting a detector.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any

from procurelens.features.amount_features import AmountFeatureSet
from procurelens.features.award_change_features import AwardChangeFeatureSet
from procurelens.features.catalog import (
    FeatureCatalog,
    FeatureCatalogEntry,
    FeatureSource,
    feature_catalog,
)
from procurelens.features.competition_context_features import (
    CompetitionContextFeatureSet,
)
from procurelens.features.competition_features import CompetitionFeatureSet
from procurelens.features.vendor_frequency_features import (
    VendorFrequencyFeatureSet,
)
from procurelens.model.feature_spec import ResolvedFeatureSelection


class FeatureRowError(ValueError):
    """Raised when unified candidate-feature row evidence is inconsistent."""


@dataclass(frozen=True, slots=True)
class CandidateFeatureValue:
    source: FeatureSource
    name: str
    value: Decimal | None
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", FeatureSource(self.source))
        name = self.name.strip()
        if not name:
            raise FeatureRowError("feature name must not be blank")
        object.__setattr__(self, "name", name)

        if self.value is None:
            reason = (
                None
                if self.unavailable_reason is None
                else self.unavailable_reason.strip()
            )
            if not reason:
                raise FeatureRowError(
                    f"{name}: missing feature requires an unavailable reason"
                )
            object.__setattr__(self, "unavailable_reason", reason)
            return

        if not isinstance(self.value, Decimal) or not self.value.is_finite():
            raise FeatureRowError(
                f"{name}: feature value must be finite Decimal or None"
            )
        if self.unavailable_reason is not None:
            raise FeatureRowError(
                f"{name}: available feature cannot carry unavailable reason"
            )

    @property
    def available(self) -> bool:
        return self.value is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "name": self.name,
            "value": None if self.value is None else str(self.value),
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True, slots=True)
class CandidateFeatureRow:
    transaction_id: str
    award_id: str
    catalog_sha256: str
    source_definition_sha256: Mapping[FeatureSource, str]
    source_evidence_sha256: Mapping[FeatureSource, str | None]
    values: tuple[CandidateFeatureValue, ...]

    def __post_init__(self) -> None:
        for field_name in ("transaction_id", "award_id"):
            text = getattr(self, field_name).strip()
            if not text:
                raise FeatureRowError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)
        object.__setattr__(
            self,
            "catalog_sha256",
            _validate_digest(self.catalog_sha256, "catalog_sha256"),
        )

        definitions = {
            FeatureSource(source): _validate_digest(
                digest, f"{FeatureSource(source).value}.definition_sha256"
            )
            for source, digest in dict(self.source_definition_sha256).items()
        }
        evidence: dict[FeatureSource, str | None] = {}
        for source, digest in dict(self.source_evidence_sha256).items():
            normalized = FeatureSource(source)
            evidence[normalized] = (
                None
                if digest is None
                else _validate_digest(
                    digest, f"{normalized.value}.evidence_sha256"
                )
            )
        if set(definitions) != set(evidence):
            raise FeatureRowError(
                "source definition/evidence mappings must have identical sources"
            )
        object.__setattr__(
            self, "source_definition_sha256", MappingProxyType(definitions)
        )
        object.__setattr__(
            self, "source_evidence_sha256", MappingProxyType(evidence)
        )

        values = tuple(self.values)
        names = tuple(item.name for item in values)
        if not values or len(names) != len(set(names)):
            raise FeatureRowError(
                "row values must be non-empty and globally unique"
            )
        if {item.source for item in values} != set(definitions):
            raise FeatureRowError(
                "row value sources must match source provenance mappings"
            )
        object.__setattr__(self, "values", values)

    @property
    def available_count(self) -> int:
        return sum(item.available for item in self.values)

    @property
    def missing_count(self) -> int:
        return len(self.values) - self.available_count

    @property
    def evidence_sha256(self) -> str:
        return _digest(self.as_dict(include_evidence_sha=False))

    def get(self, name: str) -> CandidateFeatureValue:
        wanted = name.strip()
        if not wanted:
            raise FeatureRowError("feature name must not be blank")
        for item in self.values:
            if item.name == wanted:
                return item
        raise FeatureRowError(f"unknown row feature: {wanted}")

    def select(
        self,
        selection: ResolvedFeatureSelection,
        *,
        require_complete: bool = False,
    ) -> tuple[CandidateFeatureValue, ...]:
        if not isinstance(selection, ResolvedFeatureSelection):
            raise TypeError("selection must be ResolvedFeatureSelection")
        if selection.spec.catalog_sha256 != self.catalog_sha256:
            raise FeatureRowError(
                "row catalog fingerprint differs from feature selection"
            )
        selected = tuple(self.get(name) for name in selection.feature_names)
        if require_complete:
            missing = tuple(item for item in selected if not item.available)
            if missing:
                details = ", ".join(
                    f"{item.name}={item.unavailable_reason}" for item in missing
                )
                raise FeatureRowError(
                    "selected row features are unavailable: " + details
                )
        return selected

    def as_dict(
        self, *, include_evidence_sha: bool = True
    ) -> dict[str, Any]:
        result = {
            "transaction_id": self.transaction_id,
            "award_id": self.award_id,
            "catalog_sha256": self.catalog_sha256,
            "source_definition_sha256": {
                source.value: self.source_definition_sha256[source]
                for source in sorted(
                    self.source_definition_sha256, key=lambda item: item.value
                )
            },
            "source_evidence_sha256": {
                source.value: self.source_evidence_sha256[source]
                for source in sorted(
                    self.source_evidence_sha256, key=lambda item: item.value
                )
            },
            "available_count": self.available_count,
            "missing_count": self.missing_count,
            "values": [item.as_dict() for item in self.values],
        }
        if include_evidence_sha:
            result["evidence_sha256"] = self.evidence_sha256
        return result


@dataclass(frozen=True, slots=True)
class _SourcePayload:
    source: FeatureSource
    definition_sha256: str
    evidence_sha256: str
    features: tuple[Any, ...]


def build_candidate_feature_row(
    *,
    transaction_id: str,
    award_id: str,
    amount: AmountFeatureSet | None = None,
    vendor_frequency: VendorFrequencyFeatureSet | None = None,
    competition_transaction: CompetitionFeatureSet | None = None,
    competition_context: CompetitionContextFeatureSet | None = None,
    award_change: AwardChangeFeatureSet | None = None,
    catalog: FeatureCatalog | None = None,
) -> CandidateFeatureRow:
    """Build a complete catalog-shaped row without imputing missing evidence."""

    txid, aid = transaction_id.strip(), award_id.strip()
    if not txid or not aid:
        raise FeatureRowError("transaction_id and award_id must not be blank")
    catalog = feature_catalog() if catalog is None else catalog
    if not isinstance(catalog, FeatureCatalog):
        raise TypeError("catalog must be FeatureCatalog")

    payloads = _payloads(
        transaction_id=txid,
        award_id=aid,
        amount=amount,
        vendor_frequency=vendor_frequency,
        competition_transaction=competition_transaction,
        competition_context=competition_context,
        award_change=award_change,
    )
    by_source = {payload.source: payload for payload in payloads}
    if len(by_source) != len(payloads):
        raise FeatureRowError("duplicate feature source payload")

    source_definitions: dict[FeatureSource, str] = {}
    source_evidence: dict[FeatureSource, str | None] = {}
    source_values: dict[FeatureSource, dict[str, CandidateFeatureValue]] = {}

    for source in catalog.sources:
        expected_def = catalog.source_definition_sha256[source]
        source_definitions[source] = expected_def
        payload = by_source.get(source)
        entries = tuple(
            entry for entry in catalog.entries if entry.source is source
        )
        if payload is None:
            source_evidence[source] = None
            source_values[source] = {
                entry.name: CandidateFeatureValue(
                    source=source,
                    name=entry.name,
                    value=None,
                    unavailable_reason="source_feature_set_unavailable",
                )
                for entry in entries
            }
            continue

        if payload.definition_sha256 != expected_def:
            raise FeatureRowError(
                f"{source.value}: feature definition fingerprint differs from catalog"
            )
        source_evidence[source] = payload.evidence_sha256
        normalized = _normalize_payload_features(payload, entries)
        source_values[source] = normalized

    unexpected_sources = set(by_source) - set(catalog.sources)
    if unexpected_sources:
        names = ", ".join(sorted(source.value for source in unexpected_sources))
        raise FeatureRowError(f"payload sources absent from catalog: {names}")

    values = tuple(
        source_values[entry.source][entry.name]
        for entry in catalog.entries
    )
    return CandidateFeatureRow(
        transaction_id=txid,
        award_id=aid,
        catalog_sha256=catalog.sha256_hex,
        source_definition_sha256=source_definitions,
        source_evidence_sha256=source_evidence,
        values=values,
    )


def _payloads(
    *,
    transaction_id: str,
    award_id: str,
    amount: AmountFeatureSet | None,
    vendor_frequency: VendorFrequencyFeatureSet | None,
    competition_transaction: CompetitionFeatureSet | None,
    competition_context: CompetitionContextFeatureSet | None,
    award_change: AwardChangeFeatureSet | None,
) -> tuple[_SourcePayload, ...]:
    result: list[_SourcePayload] = []

    if amount is not None:
        if not isinstance(amount, AmountFeatureSet):
            raise TypeError("amount must be AmountFeatureSet or None")
        if amount.transaction_id != transaction_id:
            raise FeatureRowError("amount feature transaction_id differs from row")
        result.append(
            _SourcePayload(
                FeatureSource.AMOUNT,
                amount.definition_sha256,
                amount.evidence_sha256,
                amount.features,
            )
        )

    if vendor_frequency is not None:
        if not isinstance(vendor_frequency, VendorFrequencyFeatureSet):
            raise TypeError(
                "vendor_frequency must be VendorFrequencyFeatureSet or None"
            )
        _validate_ids(
            transaction_id, award_id,
            vendor_frequency.transaction_id, vendor_frequency.award_id,
            "vendor_frequency",
        )
        result.append(
            _SourcePayload(
                FeatureSource.VENDOR_FREQUENCY,
                vendor_frequency.definition_sha256,
                vendor_frequency.evidence_sha256,
                vendor_frequency.features,
            )
        )

    if competition_transaction is not None:
        if not isinstance(competition_transaction, CompetitionFeatureSet):
            raise TypeError(
                "competition_transaction must be CompetitionFeatureSet or None"
            )
        _validate_ids(
            transaction_id, award_id,
            competition_transaction.transaction_id,
            competition_transaction.award_id,
            "competition_transaction",
        )
        result.append(
            _SourcePayload(
                FeatureSource.COMPETITION_TRANSACTION,
                competition_transaction.definition_sha256,
                competition_transaction.feature_evidence_sha256,
                competition_transaction.features,
            )
        )

    if competition_context is not None:
        if not isinstance(competition_context, CompetitionContextFeatureSet):
            raise TypeError(
                "competition_context must be CompetitionContextFeatureSet or None"
            )
        _validate_ids(
            transaction_id, award_id,
            competition_context.transaction_id, competition_context.award_id,
            "competition_context",
        )
        result.append(
            _SourcePayload(
                FeatureSource.COMPETITION_CONTEXT,
                competition_context.definition_sha256,
                competition_context.feature_evidence_sha256,
                competition_context.features,
            )
        )

    if award_change is not None:
        if not isinstance(award_change, AwardChangeFeatureSet):
            raise TypeError(
                "award_change must be AwardChangeFeatureSet or None"
            )
        if award_change.award_id != award_id:
            raise FeatureRowError(
                "award_change feature award_id differs from row"
            )
        result.append(
            _SourcePayload(
                FeatureSource.AWARD_CHANGE,
                award_change.definition_sha256,
                award_change.feature_evidence_sha256,
                award_change.features,
            )
        )

    return tuple(result)


def _normalize_payload_features(
    payload: _SourcePayload,
    entries: tuple[FeatureCatalogEntry, ...],
) -> dict[str, CandidateFeatureValue]:
    expected = tuple(entry.name for entry in entries)
    observed = tuple(feature.name.value for feature in payload.features)
    if len(observed) != len(set(observed)):
        raise FeatureRowError(
            f"{payload.source.value}: feature set contains duplicate names"
        )
    expected_set = set(expected)
    observed_set = set(observed)
    if observed_set != expected_set:
        missing = sorted(expected_set - observed_set)
        extra = sorted(observed_set - expected_set)
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise FeatureRowError(
            f"{payload.source.value}: feature-set names differ from catalog"
            + (" (" + "; ".join(details) + ")" if details else "")
        )

    result: dict[str, CandidateFeatureValue] = {}
    for feature in payload.features:
        name = feature.name.value
        result[name] = CandidateFeatureValue(
            source=payload.source,
            name=name,
            value=feature.value,
            unavailable_reason=feature.unavailable_reason,
        )
    return result


def _validate_ids(
    transaction_id: str,
    award_id: str,
    observed_transaction_id: str,
    observed_award_id: str,
    source: str,
) -> None:
    if observed_transaction_id != transaction_id:
        raise FeatureRowError(
            f"{source}: transaction_id differs from row"
        )
    if observed_award_id != award_id:
        raise FeatureRowError(f"{source}: award_id differs from row")


def _validate_digest(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise FeatureRowError(f"{name} must be a SHA-256 hex digest")
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
