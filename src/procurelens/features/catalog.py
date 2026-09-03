"""Unified candidate-feature catalog for ProcureLens.

Collects the canonical feature-definition contracts produced by each evidence
family into one immutable namespace. The catalog provides identity, provenance,
source-family metadata, and stable fingerprints only.

It does not decide which features a detector must consume, how missing values
are imputed, how values are scaled, or how anomaly/risk thresholds are chosen.
Those are explicit later model-specification responsibilities.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Callable

from procurelens.features.amount_features import (
    amount_feature_definition_sha256,
    amount_feature_definitions,
)
from procurelens.features.award_change_features import (
    award_change_feature_definition_sha256,
    award_change_feature_definitions,
)
from procurelens.features.competition_context_features import (
    competition_context_feature_definition_sha256,
    competition_context_feature_definitions,
)
from procurelens.features.competition_features import (
    competition_feature_definition_sha256,
    competition_feature_definitions,
)
from procurelens.features.vendor_frequency_features import (
    vendor_frequency_feature_definition_sha256,
    vendor_frequency_feature_definitions,
)


class FeatureCatalogError(ValueError):
    """Raised when unified feature-catalog evidence is inconsistent."""


class FeatureSource(str, Enum):
    AMOUNT = "amount"
    VENDOR_FREQUENCY = "vendor_frequency"
    COMPETITION_TRANSACTION = "competition_transaction"
    COMPETITION_CONTEXT = "competition_context"
    AWARD_CHANGE = "award_change"


@dataclass(frozen=True, slots=True)
class FeatureCatalogEntry:
    source: FeatureSource
    name: str
    family: str
    description: str
    source_definition_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", FeatureSource(self.source))
        for field_name in ("name", "family", "description"):
            text = getattr(self, field_name).strip()
            if not text:
                raise FeatureCatalogError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, text)
        object.__setattr__(
            self,
            "source_definition_sha256",
            _validate_digest(
                self.source_definition_sha256, "source_definition_sha256"
            ),
        )

    @property
    def identity(self) -> str:
        """Globally unique canonical feature name."""
        return self.name

    @property
    def sha256_hex(self) -> str:
        return _digest(
            {
                "source": self.source.value,
                "name": self.name,
                "family": self.family,
                "description": self.description,
                "source_definition_sha256": self.source_definition_sha256,
            }
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "source": self.source.value,
            "name": self.name,
            "family": self.family,
            "description": self.description,
            "source_definition_sha256": self.source_definition_sha256,
            "entry_sha256": self.sha256_hex,
        }


@dataclass(frozen=True, slots=True)
class FeatureCatalog:
    entries: tuple[FeatureCatalogEntry, ...]
    source_definition_sha256: Mapping[FeatureSource, str]

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if not entries:
            raise FeatureCatalogError("feature catalog must not be empty")
        names = tuple(entry.name for entry in entries)
        if len(names) != len(set(names)):
            duplicates = sorted(
                {name for name in names if names.count(name) > 1}
            )
            raise FeatureCatalogError(
                "feature names must be globally unique: "
                + ", ".join(duplicates)
            )
        canonical_order = tuple(
            sorted(entries, key=lambda item: (item.source.value, item.name))
        )
        if entries != canonical_order:
            raise FeatureCatalogError(
                "catalog entries must be deterministically sorted"
            )
        object.__setattr__(self, "entries", entries)

        source_hashes = {
            FeatureSource(source): _validate_digest(digest, f"{source}.sha256")
            for source, digest in dict(self.source_definition_sha256).items()
        }
        observed_sources = {entry.source for entry in entries}
        if set(source_hashes) != observed_sources:
            raise FeatureCatalogError(
                "source hash mapping must exactly match catalog sources"
            )
        for entry in entries:
            if source_hashes[entry.source] != entry.source_definition_sha256:
                raise FeatureCatalogError(
                    f"{entry.name}: source definition hash mismatch"
                )
        object.__setattr__(
            self,
            "source_definition_sha256",
            MappingProxyType(source_hashes),
        )

    @property
    def feature_count(self) -> int:
        return len(self.entries)

    @property
    def sources(self) -> tuple[FeatureSource, ...]:
        return tuple(sorted(self.source_definition_sha256, key=lambda x: x.value))

    @property
    def sha256_hex(self) -> str:
        return _digest(
            {
                "sources": [
                    {
                        "source": source.value,
                        "definition_sha256":
                            self.source_definition_sha256[source],
                    }
                    for source in self.sources
                ],
                "entries": [entry.as_dict() for entry in self.entries],
            }
        )

    def get(self, name: str) -> FeatureCatalogEntry:
        wanted = name.strip()
        if not wanted:
            raise FeatureCatalogError("feature name must not be blank")
        for entry in self.entries:
            if entry.name == wanted:
                return entry
        raise FeatureCatalogError(f"unknown feature: {wanted}")

    def select(
        self, names: Iterable[str]
    ) -> tuple[FeatureCatalogEntry, ...]:
        if isinstance(names, (str, bytes)):
            raise FeatureCatalogError("feature names must be an iterable")
        selected: list[FeatureCatalogEntry] = []
        seen: set[str] = set()
        for item in names:
            if not isinstance(item, str):
                raise FeatureCatalogError("feature names must be strings")
            name = item.strip()
            if not name:
                raise FeatureCatalogError("feature names must not be blank")
            if name in seen:
                raise FeatureCatalogError(
                    f"duplicate selected feature: {name}"
                )
            seen.add(name)
            selected.append(self.get(name))
        if not selected:
            raise FeatureCatalogError("at least one feature must be selected")
        return tuple(selected)

    def select_sources(
        self, sources: Iterable[FeatureSource | str]
    ) -> tuple[FeatureCatalogEntry, ...]:
        if isinstance(sources, (str, bytes)):
            raise FeatureCatalogError("sources must be an iterable")
        wanted = {FeatureSource(source) for source in sources}
        if not wanted:
            raise FeatureCatalogError("at least one source must be selected")
        return tuple(entry for entry in self.entries if entry.source in wanted)

    def select_source_families(
        self,
        pairs: Iterable[tuple[FeatureSource | str, str]],
    ) -> tuple[FeatureCatalogEntry, ...]:
        if isinstance(pairs, (str, bytes)):
            raise FeatureCatalogError(
                "source-family pairs must be an iterable"
            )
        wanted: set[tuple[FeatureSource, str]] = set()
        for source, family in pairs:
            family_text = family.strip()
            if not family_text:
                raise FeatureCatalogError("family must not be blank")
            wanted.add((FeatureSource(source), family_text))
        if not wanted:
            raise FeatureCatalogError(
                "at least one source-family pair must be selected"
            )
        return tuple(
            entry
            for entry in self.entries
            if (entry.source, entry.family) in wanted
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_count": self.feature_count,
            "catalog_sha256": self.sha256_hex,
            "sources": [
                {
                    "source": source.value,
                    "definition_sha256":
                        self.source_definition_sha256[source],
                }
                for source in self.sources
            ],
            "entries": [entry.as_dict() for entry in self.entries],
        }


@dataclass(frozen=True, slots=True)
class _SourceContract:
    source: FeatureSource
    definitions: Callable[[], tuple[Any, ...]]
    definition_sha256: Callable[[], str]


_SOURCE_CONTRACTS = (
    _SourceContract(
        FeatureSource.AMOUNT,
        amount_feature_definitions,
        amount_feature_definition_sha256,
    ),
    _SourceContract(
        FeatureSource.VENDOR_FREQUENCY,
        vendor_frequency_feature_definitions,
        vendor_frequency_feature_definition_sha256,
    ),
    _SourceContract(
        FeatureSource.COMPETITION_TRANSACTION,
        competition_feature_definitions,
        competition_feature_definition_sha256,
    ),
    _SourceContract(
        FeatureSource.COMPETITION_CONTEXT,
        competition_context_feature_definitions,
        competition_context_feature_definition_sha256,
    ),
    _SourceContract(
        FeatureSource.AWARD_CHANGE,
        award_change_feature_definitions,
        award_change_feature_definition_sha256,
    ),
)


def build_feature_catalog() -> FeatureCatalog:
    """Build the authoritative catalog from canonical source contracts."""

    entries: list[FeatureCatalogEntry] = []
    source_hashes: dict[FeatureSource, str] = {}
    for contract in _SOURCE_CONTRACTS:
        source_hash = _validate_digest(
            contract.definition_sha256(),
            f"{contract.source.value}.definition_sha256",
        )
        definitions = tuple(contract.definitions())
        if not definitions:
            raise FeatureCatalogError(
                f"{contract.source.value}: feature definitions must not be empty"
            )
        source_hashes[contract.source] = source_hash
        for definition in definitions:
            try:
                name = definition.name.value
                family = definition.family.value
                description = definition.description
            except AttributeError as exc:
                raise FeatureCatalogError(
                    f"{contract.source.value}: malformed feature definition"
                ) from exc
            entries.append(
                FeatureCatalogEntry(
                    source=contract.source,
                    name=name,
                    family=family,
                    description=description,
                    source_definition_sha256=source_hash,
                )
            )

    return FeatureCatalog(
        entries=tuple(
            sorted(entries, key=lambda item: (item.source.value, item.name))
        ),
        source_definition_sha256=source_hashes,
    )


def _validate_digest(value: str, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise FeatureCatalogError(f"{name} must be a SHA-256 hex digest")
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


_DEFAULT_CATALOG = build_feature_catalog()


def feature_catalog() -> FeatureCatalog:
    return _DEFAULT_CATALOG


def feature_catalog_sha256() -> str:
    return _DEFAULT_CATALOG.sha256_hex
